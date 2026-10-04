"""Native stages for the paired four/eight convolution fault sweep."""
from dataclasses import asdict
from pathlib import Path
import json, os, time, math
import torch
from torch.nn import functional as F
from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.runtime import source_model, evaluate, save, load
from experiments.cifar_crossbar.model import CrossbarSuffix, state_hash, tensor_hash
from experiments.cifar_crossbar.devices import clone_cpu
from experiments.cifar_crossbar.fault_runtime import cache_context
from experiments.cifar_crossbar.full_epoch_runtime import build_cache, trajectory, array_identity
from experiments.cifar_crossbar.hwa_fault_runtime import augment_images, sampled_weights, case_name
from experiments.cifar_crossbar.sweep_config import SweepSpec, SOURCES, cases
from experiments.cifar_crossbar.sweep_devices import make_array, MixedOmSampler, build_kernels, CompactCheckpoints

ROOT=Path(__file__).resolve().parents[2]


def configure_runtime(spec):
    """Deterministic FP32; enable compatible cuDNN on the Blackwell workers.

    Ampere hosts keep the verified ATen path because their installed cuDNN
    stack is incompatible. This policy changes execution, not device physics.
    """
    torch.set_num_threads(spec.cpu_threads); torch.manual_seed(spec.seed)
    device=torch.device(spec.device)
    compatible=device.type!='cuda' or torch.cuda.get_device_capability(device)[0]>=12
    torch.backends.cudnn.enabled=not spec.disable_cudnn and compatible
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    return {'cudnn_enabled':torch.backends.cudnn.enabled,'tf32':False,'deterministic':True,
            'torch_version':str(torch.__version__),'device':str(device)}


def context_for(spec,digest,student):
    return {**cache_context(spec,digest,student),'protocol':'cifar_crossbar_fault_sweep.v1','max_examples':spec.max_examples}


def development_score(student,cache,spec,populations,selected):
    rows=[]
    for kind,ppm in selected:
        values=[]
        for seed in (211001,211002,211003):
            array=make_array(student,student.q.detach(),spec,kind,ppm,populations,seed=seed)
            values.append(evaluate(student,cache['development'],student.q.device,q=array.read()))
        rows.append({'case':case_name(kind,ppm),'teacher_kl':sum(x['teacher_kl'] for x in values)/3,
                     'accuracy_percent':sum(x['accuracy_percent'] for x in values)/3})
    return rows


def fit_candidate(teacher,student,cache,spec,populations,incoming,store,context):
    generic=spec.candidate_family=='generic'
    strength=spec.candidate_noise
    if not generic:
        if incoming.get('generic',{}).get('context')!=context: raise ValueError('CDT requires matching generic selection.')
        strength=incoming['generic']['sources']['noise_hwa']['fit']['noise_strength']
    selected=cases(spec,None if generic else 'cdt_'+spec.candidate_family,selection=True)
    if spec.max_examples: selected=[(k,p) for k,p in selected if p in (0,50000)]
    criteria=['noise_hwa']+(['standard_hwa'] if strength==1 else []) if generic else ['cdt_'+spec.candidate_family]
    student.enable_calibration(True); student.q.requires_grad_(True)
    optimizer=torch.optim.Adam([student.q]+student.calibration_parameters(),lr=spec.candidate_learning_rate)
    generator=torch.Generator(device=student.q.device).manual_seed(spec.seed+10000)
    rate_generator=torch.Generator().manual_seed(spec.seed+20000)
    sampler=None
    if spec.backend=='om':
        kernel=incoming['kernel']
        if not kernel['adequate']: raise ValueError('Unvalidated OM kernel.')
        sampler=MixedOmSampler(kernel['tables']['fit'],spec.seed+10000,student.q.device)
    raw=cache['hwa_raw'].to(student.q.device)
    prefix=student.prefix_hash(); best={}; history=[]; histogram={str(p):0 for p in (20000,30000,50000)}
    limit=spec.hwa_epochs; epoch=0; rejected=None
    while epoch<=limit:
        if epoch:
            data_rng=torch.Generator(device=student.q.device).manual_seed(spec.data_seed+epoch)
            order=torch.randperm(len(raw),generator=data_rng,device=student.q.device)
            student.train()
            for batch,begin in enumerate(range(0,len(raw),spec.hwa_batch_size)):
                x=augment_images(raw[order[begin:begin+spec.hwa_batch_size]],spec.dataset,data_rng)
                with torch.no_grad():
                    t=teacher(x).softmax(1); features=student.features(x)
                rate=0.; kind='open'
                if not generic:
                    ppm=(20000,30000,50000)[int(torch.randint(3,(),generator=rate_generator))]
                    histogram[str(ppm)]+=1; rate=ppm/1e6; kind=spec.candidate_family
                q=(sampled_weights(student.q,generator,strength,kind,rate)[0] if sampler is None
                   else sampler.perturb(student.q,kind,rate,strength))
                optimizer.zero_grad(set_to_none=True)
                loss=F.kl_div(student.forward_features(features,q).log_softmax(1),t,reduction='batchmean')
                if not bool(torch.isfinite(loss)):
                    rejected={'epoch':epoch,'batch':batch,'reason':'nonfinite HWA objective'}; break
                loss.backward(); optimizer.step()
                with torch.no_grad(): student.q.clamp_(-1,1)
                if batch%100==0:
                    store.append_metric({'stage':'fit','epoch':epoch,'batch':batch,'loss':float(loss.detach())})
            if rejected: break
            if any(not bool(torch.isfinite(p).all()) for p in student.parameters()):
                rejected={'epoch':epoch,'reason':'nonfinite parameters'}; break
        if epoch==0 or epoch%spec.validation_every==0 or epoch==limit:
            rows=development_score(student,cache,spec,populations,selected)
            scores={'noise_hwa':sum(x['teacher_kl'] for x in rows)/len(rows),
                    'standard_hwa':rows[0]['teacher_kl']}
            if not generic: scores[criteria[0]]=scores['noise_hwa']
            if not all(math.isfinite(scores[k]) for k in criteria):
                rejected={'epoch':epoch,'reason':'nonfinite development score'}; break
            history.append({'epoch':epoch,'cases':rows,'scores':{k:scores[k] for k in criteria}})
            for k in criteria:
                if k not in best or scores[k]<best[k]['score']:
                    best[k]={'epoch':epoch,'score':scores[k],'model':clone_cpu(student.state_dict())}
            store.append_metric({'stage':'fit','development':history[-1]})
            save(store.run_dir/'checkpoints/progress.pt',{'epoch':epoch,'model':student.state_dict(),
                 'optimizer':optimizer.state_dict(),'rate_rng':rate_generator.get_state(),
                 'endpoint_rng':generator.get_state() if sampler is None else sampler.state_dict()})
            print(json.dumps({'epoch':epoch,'scores':history[-1]['scores']}),flush=True)
        if epoch==spec.hwa_epochs and limit==spec.hwa_epochs and spec.extension_epochs and any(v['epoch']>=.8*limit for v in best.values()):
            limit+=spec.extension_epochs
        epoch+=1
    if student.prefix_hash()!=prefix: raise RuntimeError('HWA changed frozen prefix.')
    metadata={'family':spec.candidate_family,'learning_rate':spec.candidate_learning_rate,'noise_strength':strength,
              'status':'rejected_nonfinite' if rejected else 'complete','rejection':rejected,
              'epochs_run':min(epoch-1,limit),'maximum_epochs':limit,'rate_histogram':histogram,
              'history':history,'seed':spec.seed}
    for role,item in best.items():
        item['fit']={**metadata,'selected_epoch':item['epoch'],'development_kl':item['score'],
                     'convergence_review_required':bool(rejected) or item['epoch']>=.8*limit}
    bundle={'context':context,'backend':spec.backend,'best':best,'fit':metadata}
    save(store.run_dir/'checkpoints/fit.pt',bundle)
    return {'fit':metadata,'selected':{k:{'epoch':v['epoch'],'score':v['score']} for k,v in best.items()}}


def select_sources(student,spec,incoming,context):
    fits=incoming['fits']
    expected=9
    if len(fits)!=expected: raise ValueError('Expected nine candidate results for selection.')
    if any(f['context']!=context or f['backend']!=spec.backend for f in fits): raise ValueError('Unmatched fits.')
    def choose(role):
        candidates=[f['best'][role] for f in fits if f['fit']['status']=='complete' and role in f['best']]
        if not candidates: raise RuntimeError('No completed candidate for '+role)
        best=min(candidates,key=lambda x:(x['score'],x['fit']['learning_rate'],x['fit']['noise_strength'],x['epoch']))
        return {'model':best['model'],'fit':best['fit']}
    if spec.stage=='select_generic':
        keys={(f['fit']['learning_rate'],f['fit']['noise_strength']) for f in fits}
        if len(keys)!=9 or any(f['fit']['family']!='generic' for f in fits): raise ValueError('Incomplete generic grid.')
        sources={'digital':{'model':clone_cpu(student.state_dict()),'fit':{'kind':'original_teacher'}},
                 'standard_hwa':choose('standard_hwa'),'noise_hwa':choose('noise_hwa')}
    else:
        generic=incoming['generic']
        if generic['context']!=context or generic['backend']!=spec.backend: raise ValueError('Generic sources mismatch.')
        keys={(f['fit']['family'],f['fit']['learning_rate']) for f in fits}
        if len(keys)!=9 or {x[0] for x in keys}!={'open','gmax','random'}: raise ValueError('Incomplete CDT grid.')
        sources=dict(generic['sources'])
        for kind in ('open','gmax','random'): sources['cdt_'+kind]=choose('cdt_'+kind)
    return {'context':context,'backend':spec.backend,'sources':sources,'candidate_fits':[f['fit'] for f in fits]}


def screen(student,cache,spec,populations,incoming,store,context):
    if incoming['context']!=context or incoming['backend']!=spec.backend or set(incoming['sources'])!=set(SOURCES):
        raise ValueError('Unmatched final sources.')
    rows=[]; writer=CompactCheckpoints(store.run_dir)
    for name in SOURCES:
        source=incoming['sources'][name]
        student.load_state_dict(source['model']); target=student.q.detach().clone()
        clean=evaluate(student,cache['test'],target.device)
        for kind,ppm in cases(spec,name):
            student.load_state_dict(source['model'])
            array=make_array(student,target,spec,kind,ppm,populations)
            directory=store.run_dir/'checkpoints'/name/case_name(kind,ppm)
            p0={'context':context,'model':source['model'],'array':array.state_dict(),
                'target':target,'identity':array_identity(array,spec)}
            writer(directory/'p0.pt',p0)
            initial=evaluate(student,cache['test'],target.device,q=array.read())
            for method in spec.methods:
                student.load_state_dict(source['model']); array.load_state_dict(p0['array'])
                base={'source':name,'case':case_name(kind,ppm),'array_seed':spec.array_seed,'backend':spec.backend,
                      'suffix':spec.suffix,'p0_sha256':sha256_file(directory/'p0.pt'),
                      'identity':p0['identity'],'clean':clean,'initial_test':initial}
                result=({'method':method,'curve':[],'initial_apparent_sha256':tensor_hash(array.read())} if method=='none'
                        else trajectory(student,array,target,cache,spec,method,store,directory,checkpoint_writer=writer))
                rows.append({**base,**result})
                atomic_write_json(store.run_dir/'measurements.json',rows)
    expected=sum(len(cases(spec,name)) for name in SOURCES)*5
    if len(rows)!=expected: raise RuntimeError('Incomplete screen.')
    return {'measurements':rows,'completed_controls':len(rows)}


def run_train(request):
    spec=request.spec
    if not isinstance(spec,SweepSpec) or request.teacher_weights is None: raise ValueError('Expected sweep and explicit teacher.')
    if any(getattr(request,k,None) is not None for k in ('resume','base_weights','device_state')): raise ValueError('Unexpected input.')
    inputs=[{'role':k,'path':str(getattr(request,k).resolve()),'sha256':sha256_file(getattr(request,k))}
            for k in ('teacher_weights','weights','device_data','device_model','selection_receipt') if getattr(request,k,None)]
    receipt_path=os.environ.get('EBL_SOURCE_RECEIPT')
    if receipt_path:
        receipt=json.loads(Path(receipt_path).read_text())
        if any(sha256_file(ROOT/n)!=h for n,h in receipt['files'].items()): raise ValueError('Frozen source changed.')
        inputs.append({'role':'source_snapshot','path':receipt_path,'sha256':sha256_file(Path(receipt_path))})
    store=RunStore.create(output_root=request.output_dir,experiment_id=spec.experiment_id,resolved_config=asdict(spec),
        command=request.command,repo_root=ROOT,input_artifacts=inputs,resume_capability='unsupported')
    started=time.monotonic()
    try:
        runtime_math=configure_runtime(spec)
        device=torch.device(spec.device)
        teacher,digest=source_model(request.teacher_weights,spec,device)
        student=CrossbarSuffix(teacher,spec.suffix,spec.tile_size)
        context=context_for(spec,digest,student)
        atomic_write_json(store.run_dir/'mapping.json',student.mapping_receipt())
        populations=load(request.device_model) if request.device_model else None
        incoming=load(request.weights) if request.weights else None
        if populations is not None and populations['dataset']!=spec.dataset: raise ValueError('Population dataset mismatch.')
        if spec.stage=='cache':
            cache=build_cache(teacher,student,spec,device)
            save(store.run_dir/'checkpoints/cache.pt',{'context':context,**cache})
            result={'digital_test':evaluate(student,cache['test'],device),'cohorts':{k:cache[k]['cohort'] for k in ('training','development','test')}}
        elif spec.stage=='kernel':
            if spec.backend!='om' or populations is None: raise ValueError('OM kernels require literal populations.')
            result={'gates':build_kernels(populations,spec,device,store)['gates']}
        elif spec.stage in ('select_generic','sources'):
            bundle=select_sources(student,spec,incoming,context)
            save(store.run_dir/'checkpoints/sources.pt',bundle)
            result={'sources':{k:{'sha256':state_hash(v['model']),'fit':v['fit']} for k,v in bundle['sources'].items()}}
        else:
            if request.device_data is None: raise ValueError('Missing cache.')
            cache=load(request.device_data)
            if cache['context']!=context: raise ValueError('Cache mismatch.')
            for key,n in (('training',45000),('development',1000),('test',10000)):
                if len(cache[key]['labels'])!=(spec.max_examples or n): raise ValueError('Incomplete cohort.')
                for field in ('features','teacher_logits'): cache[key][field]=cache[key][field].to(device)
            if spec.backend=='om' and populations is None: raise ValueError('Missing OM population.')
            result=(fit_candidate(teacher,student,cache,spec,populations,incoming or {},store,context) if spec.stage=='fit'
                    else screen(student,cache,spec,populations,incoming,store,context))
        result.update(stage=spec.stage,dataset=spec.dataset,backend=spec.backend,suffix=spec.suffix,runtime_math=runtime_math,
                      smoke_only=bool(spec.max_examples),evidence_class='exploratory_model_based',
                      epochs=spec.epochs,recovery_images=spec.max_examples or 45000,
                      elapsed_seconds=time.monotonic()-started)
        store.append_metric({'stage':spec.stage,'terminal':True,'elapsed_seconds':result['elapsed_seconds']})
        artifacts=[store.artifact_record(p,kind='checkpoint' if p.suffix=='.pt' else 'analysis') for p in store.run_dir.rglob('*')
                   if p.is_file() and (p.suffix=='.pt' or p.name in ('mapping.json','measurements.json'))]
        store.complete(metrics=result,artifacts=artifacts)
        print(json.dumps({'run_dir':str(store.run_dir),'status':'complete'}),flush=True)
        return 0
    except BaseException as exc:
        store.fail(exc); raise

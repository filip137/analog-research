from methods.hwa.opt import hardware_view
"""Public ebl preparation, whole-model HWA and exact-state continuation."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ebl.runtime.store import RunStore, atomic_write_json, content_hash, sha256_file
from ebl.schema.opt_cmo_drn import EXPERIMENT_ID, compatible_config_hashes
from networks.opt.data import pinned_load, cohorts, load_corpus, blocks, TRANSFER_TESTS
from networks.opt.decoder import OptConfig, from_hugging_face
from networks.opt.drn import SourceDecoder
from networks.opt.drn import reconstruct_source
from networks.opt.drn import OptDrn
from methods.calibration.drn_mapping import fit_mapping
from methods.programming.cmo import program_rewritten
from hardware.faults import state_hash
from networks.resistive.differentiable import DrnSettings
from methods.core.engine import ExperimentComponents, train_epoch, GradientsReadyEvent
from methods.core.guards import FiniteGradientGuard

MODEL=OptConfig(50272,768,12,12,3072,2048)
ROOT=Path(__file__).resolve().parents[2]


def configure_runtime(config):
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.training.seed)
    random.seed(config.training.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    limit=os.environ.get('OPT_DRN_MEMORY_LIMIT_MIB')
    if limit is not None and str(config.device).startswith('cuda'):
        fraction=float(limit)*1024**2/torch.cuda.get_device_properties(config.device).total_memory
        if not 0<fraction<=1:raise ValueError('Invalid declared CUDA allocator limit.')
        torch.cuda.set_per_process_memory_fraction(fraction,config.device)


def save(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def teacher_model(path,config):
    return from_hugging_face(pinned_load(path,config.teacher_sha256),MODEL).eval().requires_grad_(False)


def from_mapped(teacher,config,payload):
    if payload.get('schema')!='opt_cmo_drn.weights.v1':
        raise ValueError('Expected a mapped OPT DRN weights artifact.')
    contract={'teacher_sha256':config.teacher_sha256,'source_sha256':config.source_sha256,
              'initialization':config.initialization,'solver':asdict(config.solver)}
    if payload['contract']!=contract:
        raise ValueError('Mapped source/teacher/initialization/solver contract differs from this run.')
    skeleton=SourceDecoder(teacher.config)
    skeleton.load_state_dict(teacher.state_dict())
    skeleton.start=len(teacher.layers)-config.analog_layers
    skeleton.register_buffer('gains',torch.ones(2*config.analog_layers))
    model=OptDrn(skeleton,DrnSettings(**asdict(config.solver)))
    model.load_state_dict(payload['model'],strict=True)
    return model


def weights_payload(model,config,**metadata):
    return {'schema':'opt_cmo_drn.weights.v1','model':cpu_state(model),
            'contract':{'teacher_sha256':config.teacher_sha256,'source_sha256':config.source_sha256,
                        'initialization':config.initialization,'solver':asdict(config.solver)},
            **metadata}


@torch.no_grad()
def evaluate(model,teacher,tokens,device,view=None,microbatch=1):
    model.eval()
    kl=ce=0.0
    count=0
    for token_batch in tokens.split(microbatch):
        token_batch=token_batch.to(device)
        out=model(token_batch,view)
        reference=teacher(token_batch)[:,:-1]
        if not bool(torch.isfinite(out).all()):
            raise FloatingPointError('Nonfinite DRN language-model logits.')
        flat=out.flatten(0,1)
        kl+=float(F.kl_div(flat.log_softmax(-1),reference.flatten(0,1).softmax(-1),reduction='sum'))
        ce+=float(F.cross_entropy(flat,token_batch[:,1:].flatten(),reduction='sum'))
        count+=flat.shape[0]
    return {'teacher_kl':kl/count,'cross_entropy':ce/count,'perplexity':math.exp(ce/count),'scored_tokens':count,
            'numerical_diagnostics':model.diagnostics(tokens[:microbatch].to(device),view)}


def selection(model,teacher,development,config,progress=None):
    rows=[]
    for assignment in config.selection.assignment_seeds:
        for kind in config.selection.fault_kinds:
            view,receipt=program_rewritten(model,assignment,kind,0 if kind=='none' else config.selection.random_rate)
            row={'assignment':assignment,'case':kind,'metrics':evaluate(model,teacher,development,config.device,view,config.training.microbatch),
                 'programming':receipt}
            rows.append(row)
            if progress: progress(row)
            del view
    return {'teacher_kl':sum(r['metrics']['teacher_kl'] for r in rows)/len(rows),'rows':rows,
            'policy':'held_apparent_after_verify_partner_rewrite_v1'}


from methods.hwa.opt_training import make_optimizer


from methods.hwa.opt_training import lr_factor


from methods.hwa.opt_training import CoreStep


def rng_state(generator):
    return {'noise':generator.get_state(),'torch':torch.get_rng_state(),'python':random.getstate(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state,generator):
    generator.set_state(state['noise'])
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_payload(model,optimizer,config,generator,step,epoch,cursor,best,identity,best_weights=None,initial_selection=None):
    return {'schema':'opt_cmo_drn.training.v1','weights':weights_payload(model,config),'optimizer':optimizer.state_dict(),
            'rng':rng_state(generator),'step':step,'epoch':epoch,'cursor':cursor,'best':best,
            'config_sha256':content_hash(asdict(config)),'identity':identity,'torch_version':str(torch.__version__),
            'best_weights':best_weights,'initial_selection':initial_selection}


def restore_checkpoint(payload,model,optimizer,config,generator,identity):
    if payload.get('schema')!='opt_cmo_drn.training.v1' or payload['config_sha256'] not in compatible_config_hashes(config):
        raise ValueError('Exact resume requires the identical resolved configuration.')
    if payload['identity']!=identity or payload['torch_version']!=str(torch.__version__):
        raise ValueError('Exact resume requires identical input identities and PyTorch environment.')
    model.load_state_dict(payload['weights']['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'],generator)
    return payload['step'],payload['epoch'],payload['cursor'],payload['best']


def _store(request):
    identities=[]
    snapshot=os.environ.get('OPT_DRN_SOURCE_SNAPSHOT')
    if snapshot:
        import json
        snapshot=Path(snapshot)
        receipt=json.loads(snapshot.read_text())
        for name,digest in receipt['files'].items():
            path=(snapshot.parent/name).resolve()
            if not path.is_relative_to(snapshot.parent.resolve()) or sha256_file(path)!=digest:
                raise ValueError(f'Source snapshot mismatch: {name}.')
        identities.append({'role':'source_snapshot','path':str(snapshot),'sha256':sha256_file(snapshot)})
    for name in ('teacher_weights','weights','resume','device_state'):
        path=getattr(request,name,None)
        if path: identities.append({'role':name,'path':str(path.resolve()),'sha256':sha256_file(path)})
    store=RunStore.create(output_root=request.output_dir,experiment_id=EXPERIMENT_ID,resolved_config=asdict(request.spec),
                          command=request.command,repo_root=ROOT,input_artifacts=identities,
                          run_id=os.environ.get('OPT_DRN_RUN_ID'))
    print(f'RUN_DIR={store.run_dir}',flush=True)
    return store


def run_train(request):
    c=request.spec
    if c.stage=='drift_recovery':
        from ebl.runtime.opt_cmo_drn.drift_recovery import run
        return run(request)
    if c.stage=='retention':
        from ebl.runtime.opt_cmo_drn.retention import run
        return run(request)
    if c.stage=='recovery_benchmark':
        from ebl.runtime.opt_cmo_drn.microbatch_benchmark import run_benchmark
        return run_benchmark(request)
    if c.stage in ('deploy','recover'):
        from ebl.runtime.opt_cmo_drn.recovery import run_physical
        return run_physical(request)
    if getattr(request,'device_state',None) is not None:
        raise ValueError('Device-state inputs belong to physical recovery.')
    if request.teacher_weights is None or (request.weights is None and request.resume is None):
        raise ValueError('Expected explicit --teacher-weights and --weights or --resume.')
    if any(getattr(request,k,None) is not None for k in ('base_weights','device_data','device_model','deployment')):
        raise ValueError('Unexpected legacy device/checkpoint input for OPT DRN HWA.')
    configure_runtime(c)
    teacher=teacher_model(request.teacher_weights,c)
    train,dev,cohort_identity=cohorts(c)
    store=_store(request)
    started=time.monotonic()
    try:
        if c.stage=='prepare':
            if request.resume: raise ValueError('Mapping preparation does not support optimizer resume.')
            payload=pinned_load(request.weights,c.source_sha256)
            if payload.get('context',{}).get('teacher_sha256')!=c.teacher_sha256:
                raise ValueError('CMO initialization source names a different teacher.')
            source=reconstruct_source(teacher,payload,c.analog_layers)
            model=OptDrn(source,DrnSettings(**asdict(c.solver))).to(c.device)
            source=source.to(c.device)
            def mapping_progress(row):
                store.append_metric({'event':'mapping_fit',**row,'elapsed_seconds':time.monotonic()-started})
                if time.monotonic()-started>c.training.max_wall_seconds:
                    raise TimeoutError('Mapping readiness exceeded its wall-time allowance.')
            report=fit_mapping(model,source,train[:c.data.mapping_sequences].to(c.device),progress=mapping_progress)
            del source,payload
            teacher=teacher.to(c.device)
            metrics={'mapping':report,'clean_development':evaluate(model,teacher,dev,c.device,microbatch=c.training.microbatch),
                     'cohorts':cohort_identity,'physical_devices':sum(p.numel() for p in model.parameter_groups()[0]),
                     'calibration_parameters':sum(p.numel() for p in model.parameter_groups()[1])}
            path=store.run_dir/'checkpoints/mapped.pt'
            save(path,weights_payload(model,c,mapping=report,cohorts=cohort_identity))
            store.complete(metrics=metrics,artifacts=[store.artifact_record(path,kind='mapped_weights')])
            return 0
        resumed=None if request.resume is None else torch.load(request.resume,map_location='cpu',weights_only=True)
        mapped=resumed['weights'] if resumed is not None else torch.load(request.weights,map_location='cpu',weights_only=True)
        model=from_mapped(teacher,c,mapped).to(c.device)
        del mapped
        teacher=teacher.to(c.device)
        optimizer=make_optimizer(model,c)
        generator=torch.Generator(device=c.device).manual_seed(c.training.seed)
        identity={'mapped_sha256':resumed['identity']['mapped_sha256'] if resumed is not None else sha256_file(request.weights),
                  'teacher_sha256':c.teacher_sha256,'cohorts':cohort_identity}
        total=c.training.epochs*math.ceil(len(train)/c.training.effective_batch)
        if c.training.max_updates: total=min(total,c.training.max_updates)
        step=epoch=cursor=0
        best={'teacher_kl':None,'step':0}
        initial_selection=None
        if resumed is not None:
            step,epoch,cursor,best=restore_checkpoint(resumed,model,optimizer,c,generator,identity)
            initial_selection=resumed['initial_selection']
            if initial_selection is None:
                raise ValueError('Resumable run is missing its initialization selection evidence.')
            if resumed['best_weights'] is None:
                raise ValueError('Resumable run is missing its portable selected checkpoint.')
            save(store.run_dir/'checkpoints/best.pt',resumed['best_weights'])
            del resumed
            if step>=total:
                raise ValueError('Checkpoint has already reached the training budget; use validate instead.')
        core=CoreStep(model,teacher,optimizer,c,generator)
        guard=FiniteGradientGuard([name for name,_ in model.named_parameters()])
        def score():
            before=time.monotonic()
            scored=selection(model,teacher,dev,c,lambda row:store.append_metric({'event':'selection_case','step':step,**row}))
            store.append_metric({'event':'selection','step':step,'selection_seconds':time.monotonic()-before,**scored})
            return scored
        def save_latest():
            before=time.monotonic()
            selected=torch.load(store.run_dir/'checkpoints/best.pt',map_location='cpu',weights_only=True)
            save(store.run_dir/'checkpoints/latest.pt',checkpoint_payload(model,optimizer,c,generator,step,epoch,cursor,best,identity,selected,initial_selection))
            store.append_metric({'event':'checkpoint','step':step,'checkpoint_seconds':time.monotonic()-before,
                                 'size_bytes':(store.run_dir/'checkpoints/latest.pt').stat().st_size})
        if step==0:
            scored=score()
            initial_selection=scored
            best={'teacher_kl':scored['teacher_kl'],'step':0}
            save(store.run_dir/'checkpoints/best.pt',weights_payload(model,c,selection=best))
        while step<total:
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(c.data.seed+epoch))
            epoch_batches=math.ceil(len(train)/c.training.effective_batch)
            while cursor<epoch_batches and step<total:
                if time.monotonic()-started>=c.training.max_wall_seconds:
                    save_latest()
                    raise TimeoutError('Declared wall-time allowance exhausted; partial checkpoint preserved.')
                if str(c.device).startswith('cuda'):torch.cuda.synchronize()
                before=time.monotonic()
                indices=order[cursor*c.training.effective_batch:(cursor+1)*c.training.effective_batch]
                batch=train[indices]
                factor=lr_factor(step,total,c)
                for group in optimizer.param_groups: group['lr']=group['initial_lr']*factor
                train_epoch(core.components(),[(batch,batch)],epoch=epoch,start_global_step=step,event_handlers=(guard,core.observe_gradients))
                if str(c.device).startswith('cuda'):torch.cuda.synchronize()
                step+=1
                cursor+=1
                store.append_metric({'event':'update','step':step,'epoch':epoch,'cursor':cursor,'lr_factor':factor,
                                     'update_seconds':time.monotonic()-before,'elapsed_seconds':time.monotonic()-started,
                                     'peak_memory_bytes':torch.cuda.max_memory_allocated() if str(c.device).startswith('cuda') else 0,**core.last})
                if step%c.selection.every_updates==0 or cursor==epoch_batches or step==total:
                    scored=score()
                    if scored['teacher_kl']<best['teacher_kl']:
                        best={'teacher_kl':scored['teacher_kl'],'step':step}
                        save(store.run_dir/'checkpoints/best.pt',weights_payload(model,c,selection=best))
                if step%c.training.checkpoint_every==0 or cursor==epoch_batches or step==total:
                    save_latest()
            epoch+=1
            cursor=0
        if step>=total and 'scored' not in locals(): scored=score()
        final=store.run_dir/'checkpoints/final.pt'
        save(final,weights_payload(model,c,selection={'step':step,'teacher_kl':scored['teacher_kl']},initialization_sha256=identity['mapped_sha256']))
        artifacts=[store.artifact_record(final,kind='final_weights')]
        best_path=store.run_dir/'checkpoints/best.pt'
        if c.training.retention=='final':
            removed=[]
            for path in (best_path,store.run_dir/'checkpoints/latest.pt'):
                if path.exists():
                    removed.append({'path':str(path.relative_to(store.run_dir)),'sha256':sha256_file(path),'size_bytes':path.stat().st_size})
                    path.unlink()
            receipt=store.run_dir/'artifacts/retention.json'
            atomic_write_json(receipt,{'policy':'final','removed':removed,'reason':'Completed tuning retains fixed-final weights and metrics.'})
            artifacts.append(store.artifact_record(receipt,kind='retention_receipt'))
        else:
            artifacts.append(store.artifact_record(store.run_dir/'checkpoints/latest.pt',kind='training_state'))
            if best_path.exists(): artifacts.append(store.artifact_record(best_path,kind='best_weights'))
        store.complete(metrics={'updates':step,'best':best,'final_selection':scored,'initial_selection':initial_selection,
                                'initialization_sha256':identity['mapped_sha256'],'cohorts':cohort_identity,
                                'input_token_presentations':sum(min(c.training.effective_batch,len(train)-i*c.training.effective_batch)
                                    for i in range(math.ceil(len(train)/c.training.effective_batch)))*c.data.sequence_length*c.training.epochs
                                    if not c.training.max_updates else None},artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


def run_validate(request):
    c=request.spec
    if c.stage=='recover':
        from ebl.runtime.opt_cmo_drn.recovery import validate_physical
        return validate_physical(request)
    if request.teacher_weights is None: raise ValueError('Expected explicit --teacher-weights.')
    configure_runtime(c)
    teacher=teacher_model(request.teacher_weights,c)
    payload=torch.load(request.weights,map_location='cpu',weights_only=True)
    model=from_mapped(teacher,c,payload).to(c.device)
    teacher=teacher.to(c.device)
    store=_store(request)
    try:
        results={}
        for label,name,digest,count in TRANSFER_TESTS:
            tokens=blocks(load_corpus(name,digest)['test'],c.data.sequence_length,count)
            results[label]={'clean':evaluate(model,teacher,tokens,c.device,microbatch=c.training.microbatch),
                            'programmed':selection(model,teacher,tokens,c)}
        path=store.run_dir/'artifacts/held_out.json'
        atomic_write_json(path,results)
        store.append_metric({'event':'held_out_evaluation','results':results})
        store.complete(metrics=results,artifacts=[store.artifact_record(path,kind='held_out_evaluation')])
        return 0
    except BaseException as error:
        store.fail(error)
        raise

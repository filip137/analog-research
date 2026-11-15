"""Bounded CIFAR training with explicit full-state, mid-epoch resume."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json, math, os, shutil, signal, time
import numpy as np
import torch
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.runtime.pseudo_drn_scale import load_data, tokens
from networks.energy_transformer.pseudo_large import Model, gradient_probe

ROOT = Path(__file__).resolve().parents[2]


def atomic_save(path, payload):
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run_train(request):
    c = request.spec
    for name in ['weights','base_weights','device_data','device_model','teacher_weights','deployment','device_state']:
        if getattr(request, name, None) is not None:
            raise ValueError('Unsupported external initialization')
    started = time.monotonic()
    budget = json.loads(Path(c.budget_file).read_text())
    deadline = datetime.fromisoformat(budget['deadline']).timestamp()
    if time.time() >= deadline:
        raise TimeoutError('Original shared allowance expired')
    torch.set_num_threads(4);torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    data, paths = load_data(c.data_root)
    inputs = [{'path':str(p),'sha256':sha256_file(p)} for p in paths]
    if request.resume is not None:
        inputs.append({'path':str(request.resume),'sha256':sha256_file(request.resume)})
    store = RunStore.create(output_root=request.output_dir,experiment_id=c.experiment_id,
                            resolved_config=asdict(c),command=request.command,repo_root=ROOT,
                            input_artifacts=inputs,resume_capability='stateful_nondeterministic')
    art = store.run_dir/'artifacts'
    (store.run_dir/'worker.pid').write_text(str(os.getpid()))
    atomic_write_json(Path(request.output_dir)/'active-run.json',{'run_dir':str(store.run_dir),'pid':os.getpid()})
    print('RUN_DIR='+str(store.run_dir),flush=True)
    stopped = [False]
    def stop(signum, frame): stopped[0] = True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    model = None; elapsed_before = 0
    try:
        model=Model(c.seed,c.width,c.depth,c.heads,c.hidden,c.mlp).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay)
        order=torch.Generator().manual_seed(4101)
        aug=torch.Generator(device='cuda').manual_seed(4102)
        mix=np.random.default_rng(4103)
        mixing=torch.Generator(device='cuda').manual_seed(4104)
        epoch=0;offset=0;permutation=None;step=0;best_accuracy=-1.;best_epoch=0
        all_probes_pass=True;epoch_loss=0.;epoch_examples=0
        if request.resume is not None:
            ck=torch.load(request.resume,map_location='cpu',weights_only=False)
            if ck['config'] != asdict(c):raise ValueError('Resume requires identical scientific config and allowance')
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            for state in opt.state.values():
                for key,value in state.items():
                    if isinstance(value,torch.Tensor):state[key]=value.cuda()
            epoch,offset,permutation,step=ck['epoch'],ck['offset'],ck['permutation'],ck['step']
            order.set_state(ck['order_rng']);aug.set_state(ck['aug_rng']);mix.bit_generator.state=ck['mix_rng'];mixing.set_state(ck['mixing_rng'])
            torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
            elapsed_before=ck['elapsed'];best_accuracy,best_epoch=ck['best_accuracy'],ck['best_epoch']
            all_probes_pass=ck['all_probes_pass'];epoch_loss,epoch_examples=ck['epoch_loss'],ck['epoch_examples']
            old_best=Path(request.resume).parent/'best.pt'
            if best_accuracy >= 0:
                if not old_best.is_file():raise ValueError('Missing validation-best checkpoint from resumed run')
                old_selection=torch.load(old_best,map_location='cpu',weights_only=False)
                if old_selection['step'] > step:
                    raise ValueError('Best checkpoint is newer than resume state; resume that complete best checkpoint instead')
                if old_selection['best_accuracy'] != best_accuracy:
                    raise ValueError('Validation-best provenance does not match resume state')
                shutil.copy2(old_best,art/'best.pt')
        def elapsed(): return elapsed_before+time.monotonic()-started
        def exhausted(reserve=300):
            return elapsed()>=c.wall_seconds-reserve or time.time()>=deadline-reserve or stopped[0]
        def save(name):
            atomic_save(art/f'{name}.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),
                'config':asdict(c),'epoch':epoch,'offset':offset,'permutation':permutation,'step':step,
                'order_rng':order.get_state(),'aug_rng':aug.get_state(),'mix_rng':mix.bit_generator.state,
                'mixing_rng':mixing.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'elapsed':elapsed(),'best_accuracy':best_accuracy,'best_epoch':best_epoch,
                'all_probes_pass':all_probes_pass,'epoch_loss':epoch_loss,'epoch_examples':epoch_examples})
        def metric(row):
            row={**row,'step':step,'epoch':epoch,'elapsed':elapsed()}
            store.append_metric(row);print(row,flush=True)
            atomic_write_json(store.run_dir/'progress.json',{'state':'running','heartbeat_at':datetime.now(timezone.utc).isoformat(),**row})
        @torch.no_grad()
        def evaluate(split, count=None):
            model.eval();images,targets=data[split];count=len(images) if count is None else count
            correct=0;ce=0.
            for at in range(0,count,c.batch_size):
                if time.time()>=deadline-15:raise TimeoutError('Shared deadline during final evaluation')
                end=min(at+c.batch_size,count);y=model(tokens(images[at:end]),c.method,c.steps);target=targets[at:end].cuda()
                correct+=int((y.argmax(-1)==target).sum());ce+=float(torch.nn.functional.cross_entropy(y,target,reduction='sum'))
            model.train();return {'count':count,'accuracy':correct/count,'cross_entropy':ce/count}
        def probe(tag):
            nonlocal all_probes_pass
            im,t=data['validation'];p=gradient_probe(model,tokens(im[:2]),t[:2].cuda(),c.steps)
            atomic_write_json(art/f'gradients-{tag}.json',p);all_probes_pass=all_probes_pass and p['pass']
            metric({'stage':'gradient_probe','tag':tag,**{k:v for k,v in p.items() if k!='groups'}})
            return p['pass']
        if request.resume is None:
            save('initial')
            if not probe('initial'):raise RuntimeError('Initial EP/BPTT agreement gate failed')
            metric({'stage':'initial_validation',**evaluate('validation',256)})
        else:metric({'stage':'resumed','checkpoint':str(request.resume)})
        train_images,train_targets=data['train'];size=len(train_images)
        steps_per_epoch=math.ceil(size/c.batch_size);total_steps=c.epochs*steps_per_epoch
        warmup=c.warmup_epochs*steps_per_epoch;reason='epochs_complete'
        while epoch<c.epochs:
            if permutation is None:permutation=torch.randperm(size,generator=order)
            while offset<size:
                if exhausted():reason='budget_limit' if not stopped[0] else 'interrupted';break
                if c.max_updates and step>=c.max_updates:reason='readiness_update_limit';break
                if step<warmup:factor=(step+1)/max(warmup,1)
                else:factor=.02+.98*.5*(1+math.cos(math.pi*(step-warmup)/max(total_steps-warmup,1)))
                for group in opt.param_groups:group['lr']=c.learning_rate*factor
                update_started=time.monotonic()
                ids=permutation[offset:offset+c.batch_size];x=tokens(train_images[ids],aug);target=train_targets[ids].cuda()
                lam=float(mix.beta(c.mixup,c.mixup)) if c.mixup else 1.
                partner=torch.randperm(len(ids),device='cuda',generator=mixing)
                x=lam*x+(1-lam)*x[partner]
                opt.zero_grad(set_to_none=True);logits=model(x,c.method,c.steps)
                loss=lam*torch.nn.functional.cross_entropy(logits,target,label_smoothing=c.label_smoothing)+(1-lam)*torch.nn.functional.cross_entropy(logits,target[partner],label_smoothing=c.label_smoothing)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                step+=1;offset+=len(ids);epoch_examples+=len(ids);epoch_loss+=float(loss.detach())*len(ids)
                if step==1 or step%25==0 or step==c.max_updates:metric({'stage':'train','loss':float(loss.detach()),'lr':c.learning_rate*factor,'grad_norm':float(norm),'step_seconds':time.monotonic()-update_started,'peak_mib':torch.cuda.max_memory_allocated()/2**20})
                if step%250==0:save('latest')
            if offset<size:break
            epoch+=1;offset=0;permutation=None
            val=evaluate('validation');metric({'stage':'validation','train_mixup_loss':epoch_loss/epoch_examples,**val})
            epoch_loss=0.;epoch_examples=0
            if val['accuracy']>best_accuracy:
                best_accuracy=val['accuracy'];best_epoch=epoch;save('best')
            if epoch%c.probe_every==0 and not exhausted(600):
                if not probe(f'epoch{epoch:03d}'):reason='gradient_gate';save('latest');break
            save('latest')
        save('final');save('latest')
        final_validation=evaluate('validation');probe('final')
        if not (art/'best.pt').exists():
            best_accuracy=final_validation['accuracy'];best_epoch=epoch;save('best')
        chosen=torch.load(art/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(chosen['model'])
        selected_validation=evaluate('validation');test=evaluate('test');probe('best')
        summary={'parameters':sum(p.numel() for p in model.parameters()),'mlp':c.mlp,'method':c.method,
                 'epochs_completed':epoch,'updates':step,'planned_epochs':c.epochs,'best_epoch':best_epoch,
                 'stop_reason':reason,'final_validation':final_validation,'selected_validation':selected_validation,
                 'test':test,'all_gradient_probes_pass':all_probes_pass,'seconds':elapsed(),
                 'peak_mib':torch.cuda.max_memory_allocated()/2**20,'boundary':'Ideal buffered EP blocks; score, projections, normalizers and optimizer digital; no physical hardware claim.'}
        atomic_write_json(art/'summary.json',summary)
        store.complete(metrics=summary,artifacts=[store.artifact_record(p,kind='pseudo_drn_large') for p in sorted(art.iterdir()) if p.is_file()])
        atomic_write_json(store.run_dir/'progress.json',{'state':'complete','heartbeat_at':datetime.now(timezone.utc).isoformat(),**summary})
        print(summary,flush=True);return 0
    except BaseException as error:
        store.fail(error)
        atomic_write_json(store.run_dir/'progress.json',{'state':'failed','error':str(error),'heartbeat_at':datetime.now(timezone.utc).isoformat()})
        raise

"""Paired finite-iteration MNIST training, fixed cohorts and complete native evidence."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from networks.energy_transformer.pseudo_learning import Model,objective,gradient_probe
from networks.energy_transformer.pseudo_mnist_data import load_raw,hidden_masks,batch,patchify
from ebl.runtime.store import RunStore,atomic_write_json,sha256_file

ROOT=Path(__file__).resolve().parents[2]


def run_train(request):
    for k in ('weights','base_weights','resume','device_data','device_model','teacher_weights','deployment','device_state'):
        if getattr(request,k,None) is not None: raise ValueError('External inputs/resume unsupported')
    c=request.spec; start=time.monotonic();deadline=start+c.wall_seconds
    torch.set_num_threads(4);torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    images,ids,paths=load_raw(c.data_root)
    labels={s:torch.from_numpy(np.frombuffer((Path(c.data_root)/'MNIST/raw'/f'{p}-labels-idx1-ubyte').read_bytes()[8:],dtype=np.uint8).copy()).long()
            for s,p in [('train','train'),('test','t10k')]}
    store=RunStore.create(output_root=request.output_dir,experiment_id=c.experiment_id,resolved_config=asdict(c),
        command=request.command,repo_root=ROOT,input_artifacts=[{'path':str(p),'sha256':sha256_file(p)} for p in paths],resume_capability='unsupported')
    print('RUN_DIR='+str(store.run_dir),flush=True)
    art=store.run_dir/'artifacts'
    try:
        model=Model(seed=c.seed,task=c.task).to(c.device)
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        perm=torch.randperm(784,generator=torch.Generator().manual_seed(3105))
        fixed={s:hidden_masks(len(ids[s]),torch.Generator().manual_seed(seed)) for s,seed in [('validation',3103),('test',3104)]}
        mean=images['train'][ids['train']].float().mean(0)/255*.8-.4
        order=torch.Generator().manual_seed(3101); masks=torch.Generator().manual_seed(3102)
        def make(indices,split,hidden=None):
            src='test' if split=='test' else 'train';im=images[src][indices]
            if c.task=='completion': return batch(im,hidden,7,c.device)
            im=im.flatten(1)[:,perm].reshape(-1,28,28)
            tokens=patchify(im.to(device=c.device,dtype=torch.float32)/255*.8-.4,7)
            return torch.cat((tokens,torch.ones_like(tokens)),-1),labels[src][indices].to(c.device),None
        def checktime():
            if time.monotonic()>deadline: raise TimeoutError('Declared run wall allowance exceeded')
        @torch.no_grad()
        def evaluate(split,count,ablation=None,baseline=False):
            total=0.;denom=0
            for offset in range(0,count,64):
                checktime(); ix=ids[split][offset:min(offset+64,count)]
                x,t,m=make(ix,split,fixed[split][offset:offset+len(ix)])
                if baseline:
                    y=patchify(mean.expand(len(ix),-1,-1).to(c.device),7) if m is not None else None
                else: y=model(x,c.method,c.steps,ablation=ablation)
                if m is None:
                    total+=int((y.argmax(-1)==t).sum()) if y is not None else int((t==0).sum());denom+=len(ix)
                else: total+=float(((y-t).square()*m).sum())/.64;denom+=float(m.sum())
            return {'count':count,'accuracy' if c.task!='completion' else 'mse':total/denom}
        def save(name,step):
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':asdict(c),
                        'order_rng':order.get_state(),'mask_rng':masks.get_state(),'permutation':perm},art/f'{name}.pt')
        def probes(stage):
            rows=[]
            for i in range(3):
                checktime();ix=ids['validation'][i*8:(i+1)*8]
                x,t,m=make(ix,'validation',fixed['validation'][i*8:(i+1)*8])
                rows.append({'batch':i,'caps':gradient_probe(model,x,t,m)})
            atomic_write_json(art/f'gradients-{stage}.json',rows)
            return rows
        atomic_write_json(art/'environment.json',{'torch':torch.__version__,'device':str(c.device),
            'gpu':torch.cuda.get_device_name() if c.device=='cuda' else None,'dtype':'float32','tf32':False})
        save('initial',0);probes('initial')
        initial=evaluate('validation',256);store.append_metric({'stage':'initial',**initial})
        for step in range(1,c.updates+1):
            checktime();ix=ids['train'][torch.randint(len(ids['train']),(c.batch_size,),generator=order)]
            x,t,m=make(ix,'train',hidden_masks(c.batch_size,masks))
            opt.zero_grad(set_to_none=True);y=model(x,c.method,c.steps);loss=objective(y,t,m);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite loss/gradient')
            opt.step()
            if step%50==0 or step==1:
                row={'stage':'train','step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start}
                store.append_metric(row);print(row,flush=True)
            if step%250==0:
                store.append_metric({'stage':'validation_probe','step':step,**evaluate('validation',256)})
        save('final',c.updates);final_probes=probes('final')
        validation=evaluate('validation',5000);test=evaluate('test',10000)
        baseline=evaluate('test',10000,baseline=True)
        ablations={k:evaluate('validation',256,ablation=k) for k in ['uniform','zero']}
        selected=[row for b in final_probes for row in b['caps'] if row['cap']==c.steps]
        comparisons=[r['groups'][n][kind] for r in selected for n in r['groups'] for kind in ['ep_bptt','ep_exact','bptt_exact']]
        agreement=all(v['cosine']>=.99 and v['relative_l2']<=.05 for v in comparisons if v['reference_norm']>1e-9)
        summary={'task':c.task,'method':c.method,'steps_per_phase':c.steps,'updates':c.updates,'parameters':sum(p.numel() for p in model.parameters()),
            'initial_validation_probe':initial,'validation':validation,'test':test,'baseline':baseline,'ablations':ablations,
            'gradient_agreement':agreement,'minimum_cosine':min(v['cosine'] for v in comparisons),
            'maximum_relative_l2':max(v['relative_l2'] for v in comparisons),'seconds':time.monotonic()-start,
            'boundary':'Ideal encoded-state block Gauss-Seidel; independent per-query value states. EP only through aggregation, BP elsewhere. No physical RC latency/energy claim. Fixed-error centered nudges with reported amplitude normalization; not end-to-end loss-nudged EP.'}
        atomic_write_json(art/'summary.json',summary)
        artifacts=[store.artifact_record(p,kind='pseudo_mnist') for p in sorted(art.rglob('*')) if p.is_file()]
        store.complete(metrics=summary,artifacts=artifacts)
        print(summary,flush=True);return 0
    except BaseException as error:
        store.fail(error);raise

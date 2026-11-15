from dataclasses import asdict
from pathlib import Path
import pickle,time
import numpy as np
import torch
from networks.energy_transformer.pseudo_drn import Model,probe
from ebl.runtime.store import RunStore,atomic_write_json,sha256_file

ROOT=Path(__file__).resolve().parents[2]

def load_data(root):
    files=[Path(root)/f'data_batch_{i}' for i in range(1,6)]+[Path(root)/'test_batch']
    data=[];labels=[]
    for p in files:
        with p.open('rb') as f:d=pickle.load(f,encoding='bytes')
        data.append(torch.from_numpy(d[b'data'].copy()).reshape(-1,3,32,32))
        labels.append(torch.tensor(d[b'labels']))
    im=torch.cat(data[:5]);lab=torch.cat(labels[:5]);gen=torch.Generator().manual_seed(4100)
    train=[];val=[]
    for digit in range(10):
        ids=(lab==digit).nonzero().flatten();ids=ids[torch.randperm(len(ids),generator=gen)]
        val.append(ids[:500]);train.append(ids[500:])
    train=torch.cat(train);val=torch.cat(val);val=val[torch.randperm(len(val),generator=gen)]
    return {'train':(im[train],lab[train]),'validation':(im[val],lab[val]),'test':(data[5],labels[5])},files

def tokens(images,gen=None):
    x=images.cuda().float()/127.5-1
    if gen is not None:
        x=torch.nn.functional.pad(x,(4,4,4,4),mode='reflect')
        offsets=torch.randint(9,(len(x),2),device='cuda',generator=gen)
        grid=torch.arange(32,device='cuda')
        x=x.permute(0,2,3,1)[torch.arange(len(x),device='cuda')[:,None,None],offsets[:,0,None,None]+grid[None,:,None],offsets[:,1,None,None]+grid[None,None,:]].permute(0,3,1,2)
        flip=torch.rand(len(x),device='cuda',generator=gen)<.5
        x=torch.where(flip[:,None,None,None],x.flip(-1),x)
    return x.reshape(-1,3,8,4,8,4).permute(0,2,4,1,3,5).reshape(-1,64,48)

def run_train(request):
    for name in ['weights','base_weights','resume','device_data','device_model','teacher_weights','deployment','device_state']:
        if getattr(request,name,None) is not None:raise ValueError('External initialization/resume unsupported')
    c=request.spec;start=time.monotonic();deadline=start+c.wall_seconds
    torch.set_num_threads(4);torch.manual_seed(c.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data,paths=load_data(c.data_root)
    store=RunStore.create(output_root=request.output_dir,experiment_id=c.experiment_id,resolved_config=asdict(c),command=request.command,
        repo_root=ROOT,input_artifacts=[{'path':str(p),'sha256':sha256_file(p)} for p in paths],resume_capability='unsupported')
    print('RUN_DIR='+str(store.run_dir),flush=True);art=store.run_dir/'artifacts'
    try:
        model=Model(c.seed,c.score,c.mlp).cuda();opt=torch.optim.Adam(model.parameters(),lr=.003)
        order=torch.Generator().manual_seed(4101);aug=torch.Generator(device='cuda').manual_seed(4102)
        def checktime():
            if time.monotonic()>deadline:raise TimeoutError('Declared wall allowance exceeded')
        def save(name,step):torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':asdict(c),'order_rng':order.get_state(),'aug_rng':aug.get_state()},art/f'{name}.pt')
        @torch.no_grad()
        def evaluate(split,count,ablation=None):
            correct=0;ce=0
            for offset in range(0,count,32):
                checktime();im,t=data[split];y=model(tokens(im[offset:offset+32]),c.method,32,ablation);t=t[offset:offset+32].cuda()
                correct+=int((y.argmax(-1)==t).sum());ce+=float(torch.nn.functional.cross_entropy(y,t,reduction='sum'))
            return {'count':count,'accuracy':correct/count,'cross_entropy':ce/count}
        def probes(stage):
            results=[]
            for i in range(3):
                checktime();im,t=data['validation'];results.append(probe(model,tokens(im[8*i:8*i+8]),t[8*i:8*i+8].cuda()))
            atomic_write_json(art/f'gradients-{stage}.json',results)
            return results
        save('initial',0);probes('initial');initial=evaluate('validation',256)
        store.append_metric({'stage':'initial',**initial});atomic_write_json(art/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'float32','tf32':False})
        for step in range(1,c.updates+1):
            checktime();im,t=data['train'];ix=torch.randint(len(im),(32,),generator=order);x=tokens(im[ix],aug);target=t[ix].cuda()
            opt.zero_grad(set_to_none=True);y=model(x,c.method,32);loss=torch.nn.functional.cross_entropy(y,target);loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Invalid gradients')
            opt.step()
            if step%50==0 or step==1:
                row={'stage':'train','step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-start};store.append_metric(row);print(row,flush=True)
            if step%250==0:store.append_metric({'stage':'validation_probe','step':step,**evaluate('validation',256)})
        save('final',c.updates);final=probes('final');validation=evaluate('validation',5000);test=evaluate('test',10000);uniform=evaluate('validation',256,'uniform')
        items=[v for p in final for row in p['caps'] if row['cap']==32 for g in row['groups'].values() for v in g.values() if v['reference_norm']>1e-9]
        refs=[v for p in final for v in p['reference_128_256'].values() if v['reference_norm']>1e-9]
        summary={'method':c.method,'score':c.score,'mlp':c.mlp,'updates':c.updates,'parameters':sum(p.numel() for p in model.parameters()),'steps_per_phase':32,
            'initial':initial,'validation':validation,'test':test,'uniform_probe':uniform,'seconds':time.monotonic()-start,
            'gradient_agreement':all(v['cosine']>=.99 and v['relative_l2']<=.05 for v in items),
            'reference_qualified':all(v['cosine']>=.999 and v['relative_l2']<=.01 for v in refs),
            'min_cosine':min(v['cosine'] for v in items),'max_relative_l2':max(v['relative_l2'] for v in items),
            'diagnostics':[p['diagnostics'] for p in final],
            'boundary':'Ideal passive DRN with reciprocal loads, diode constraint and differential rails. Ideal attention aggregation. Buffered inter-block interfaces and scoring/normalization/Adam digital. No hardware/PDK/latency claim.'}
        atomic_write_json(art/'summary.json',summary)
        store.complete(metrics=summary,artifacts=[store.artifact_record(p,kind='pseudo_drn_scale') for p in sorted(art.rglob('*')) if p.is_file()])
        print(summary,flush=True);return 0
    except BaseException as error:store.fail(error);raise

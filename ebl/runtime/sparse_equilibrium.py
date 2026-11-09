"""First exploratory graph and OPT runs, published through native RunStore."""
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from ebl.runtime.store import RunStore,atomic_write_json,sha256_file
from methods.sparse_equilibrium import BudgetEnd,check_time,rows,evaluate,choose_masks,fit,example_loss
from networks.sparse_equilibrium import PhysicalOpt,grid_edges,graph_solve

ROOT=Path(__file__).resolve().parents[2]

def snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def graph_pilot(config,store,deadline,smoke=False):
    torch.manual_seed(config.seed); device=config.device
    b=grid_edges().to(device); base=torch.full((len(b),),0.4,device=device)
    changed=torch.tensor([1,5,13,21],device=device); target=base.clone(); target[changed]=0.8
    gen=torch.Generator().manual_seed(config.seed)
    cohorts=[torch.randn(n,16,generator=gen).to(device) for n in ((16,8,8) if smoke else (512,128,256))]
    records=[]; steps=2 if smoke else 256
    for constrained in (False,True):
        xs,xv,xt=cohorts
        with torch.no_grad():
            ys=graph_solve(xs,target,b,constrained); yt=graph_solve(xt,target,b,constrained)
            source=graph_solve(xt,base,b,constrained)
        def report(kind,pred,write,extra=None):
            d=pred-source; sv=torch.linalg.svdvals(d); energy=sv.square(); total=energy.sum()
            row={'family':'graph','constrained':constrained,'arm':kind,'test_mse':float((pred-yt).square().mean()),
                 'source_function_change_r90':int(torch.searchsorted(energy.cumsum(0),0.9*total)+1) if total>0 else 0,
                 'ideal_write_commands':write,'active_zero_fraction':float((pred.abs()<1e-7).float().mean()),**(extra or {})}
            records.append(row);store.append_metric(row)
        report('frozen',source,0)
        p=base.clone().requires_grad_(); score=torch.zeros_like(p)
        for x,y in zip(xs[:8 if smoke else 32],ys):
            grad=torch.autograd.grad((graph_solve(x[None],p,b,constrained)-y).square().mean(),p)[0]
            score+=grad.square().detach()
        fisher=score.topk(4).indices; random=torch.randperm(len(b),generator=gen)[:4].to(device)
        for name,idx in [('random',random),('fisher',fisher),('known_edges_oracle',changed),('full',torch.arange(len(b),device=device))]:
            value=nn.Parameter(base[idx].clone()); opt=torch.optim.Adam([value],lr=0.03)
            for step in range(steps):
                check_time(deadline); opt.zero_grad(set_to_none=True)
                ids=torch.randint(len(xs),(min(64,len(xs)),),generator=gen).to(device)
                conductance=base.index_copy(0,idx,value)
                loss=(graph_solve(xs[ids],conductance,b,constrained)-ys[ids]).square().mean()
                loss.backward();opt.step()
                with torch.no_grad():value.clamp_(0.05,1.5)
                if step%32==0:store.append_metric({'family':'graph','constrained':constrained,'arm':name,'step':step,'loss':float(loss)})
            with torch.no_grad():pred=graph_solve(xt,base.index_copy(0,idx,value),b,constrained)
            report(name,pred,steps*len(idx),{'unique_devices':len(idx),'mask':idx.tolist()})
        # Explicit MLP starts from a source-function fit; its mismatch stays visible.
        mlp=nn.Sequential(nn.Linear(16,64,bias=False),nn.ReLU() if constrained else nn.Identity(),nn.Linear(64,16,bias=False)).to(device)
        opt=torch.optim.Adam(mlp.parameters(),lr=0.003)
        with torch.no_grad():source_train=graph_solve(xs,base,b,constrained)
        for step in range(2 if smoke else 1000):
            check_time(deadline);opt.zero_grad(set_to_none=True)
            ids=torch.randint(len(xs),(min(64,len(xs)),),generator=gen).to(device)
            loss=(mlp(xs[ids])-source_train[ids]).square().mean();loss.backward();opt.step()
        state=snapshot(mlp)
        with torch.no_grad():source_mismatch=float((mlp(xt)-source).square().mean())
        report('mlp_frozen',mlp(xt).detach(),0,{'source_fit_mse':source_mismatch})
        for kind in ('random','fisher','full'):
            mlp.load_state_dict(state); ps=list(mlp.parameters()); scores=[torch.zeros_like(p) for p in ps]
            if kind=='fisher':
                for x,y in zip(xs[:8 if smoke else 32],ys):
                    gs=torch.autograd.grad((mlp(x[None])-y).square().mean(),ps)
                    for s,g in zip(scores,gs):s.add_(g.square().detach())
            masks=[]
            for p,s in zip(ps,scores):
                mask=torch.zeros_like(p).flatten()
                idx=(torch.arange(p.numel(),device=device) if kind=='full' else
                     (s.flatten().topk(2).indices if kind=='fisher' else torch.randperm(p.numel(),generator=gen)[:2].to(device)))
                mask[idx]=1;masks.append(mask.view_as(p))
            # Frozen per-matrix scale makes one changed scalar one active cell; references fixed.
            scales=[p.detach().abs().max().clamp_min(1e-4) for p in ps]
            opt=torch.optim.Adam(ps,lr=0.003)
            originals=[p.detach().clone() for p in ps]
            for step in range(steps):
                check_time(deadline);opt.zero_grad(set_to_none=True)
                ids=torch.randint(len(xs),(min(64,len(xs)),),generator=gen).to(device)
                loss=(mlp(xs[ids])-ys[ids]).square().mean();loss.backward()
                for p,m in zip(ps,masks):p.grad.mul_(m)
                opt.step()
                with torch.no_grad():
                    for p,m,v,s in zip(ps,masks,originals,scales):
                        p.clamp_(-s,s);p.copy_(torch.where(m.bool(),p,v))
            count=int(sum(m.sum() for m in masks))
            report('mlp_'+kind,mlp(xt).detach(),steps*count,{'unique_devices':count,'source_fit_mse':source_mismatch})
    return records

def gradient_audit(model,row,labels):
    if model.architecture!='drn':return {'applicable':False}
    tok=torch.tensor(row['tokens'],device=model.z[0].device)[None]
    with torch.no_grad():
        a=model(tok,labels);model.total=48; b=model(tok,labels);model.total=24
    forward_error=float((a-b).norm()/b.norm().clamp_min(1))
    model.configure('full'); vectors=[]
    for tail in (8,16):
        model.tail=tail;model.zero_grad(set_to_none=True);example_loss(model,row,labels).backward()
        vectors.append(torch.cat([z.grad.flatten()[::max(1,z.numel()//1024)].detach().clone() for z in model.z]))
    model.tail=8;model.zero_grad(set_to_none=True);model.configure('frozen')
    cosine=float(F.cosine_similarity(*[v[None] for v in vectors]).item())
    relative=float((vectors[0]-vectors[1]).norm()/vectors[1].norm().clamp_min(1e-12))
    receipt={'forward_24_vs_48_relative_error':forward_error,'gradient_8_vs_16_cosine':cosine,'gradient_relative_error':relative}
    if forward_error>0.001 or cosine<0.99 or relative>0.1:
        raise ValueError(f'DRN convergence/gradient qualification failed: {receipt}')
    return receipt

def sentiment(config,architecture,store,deadline,smoke=False):
    root=Path(config.input_root); data=json.loads((root/'sentiment.json').read_text()); labels=data['label_ids']
    train=data['splits']['train']; dev=data['splits']['dev']
    if smoke:train=train[:4];dev=dev[:2]
    model=PhysicalOpt(root,architecture,config.device)
    store.append_metric({'family':'sentiment','architecture':architecture,'stage':'loaded','eligible_devices':sum(z.numel() for z in model.z)})
    audit=gradient_audit(model,train[0]['original'],labels)
    initial=evaluate(model,dev,labels,deadline)
    source_rows=rows(train,True); all_rows=rows(train)
    params=model.configure('full');source_steps=2 if smoke else config.source_updates
    fit(model,source_rows,labels,params,source_steps,1 if smoke else config.batch,0.0003,config.seed,deadline,
        lambda r:store.append_metric({'family':'sentiment','architecture':architecture,'stage':'source_fit',**r}),
        analog_count=sum(z.numel() for z in model.z))
    model.configure('frozen'); source_audit=gradient_audit(model,train[0]['original'],labels); base=snapshot(model)
    source_metrics=evaluate(model,dev,labels,deadline)
    torch.save(base,store.run_dir/'checkpoints'/f'{architecture}-source.pt')
    atomic_write_json(store.run_dir/'artifacts'/f'{architecture}-baseline.json',{'initial':initial,'source':source_metrics,'audit':audit,'source_audit':source_audit})
    masks={kind:choose_masks(model,all_rows,labels,config.mask_size,kind,config.seed,2 if smoke else config.fisher_examples,deadline)
           for kind in ('random','fisher')}
    torch.save({k:[i.cpu() for i in idx] for k,idx in masks.items()},store.run_dir/'artifacts'/f'{architecture}-masks.pt')
    result=[]; steps=2 if smoke else config.updates
    for rate in ((config.learning_rates[0],) if smoke else config.learning_rates):
        for arm in ('random','fisher','boundary','bias','lora','full'):
            check_time(deadline);model.configure('frozen');model.load_state_dict(base)
            torch.manual_seed(config.seed);params=model.configure(arm,masks.get(arm))
            n=(config.mask_size if arm in masks else sum(z.numel() for z in model.z) if arm=='full' else 0)
            case=f'{architecture}-{arm}-{rate:g}'
            metric=lambda r:store.append_metric({'family':'sentiment','case':case,'stage':'adapt',**r})
            stats=fit(model,all_rows,labels,params,steps,1 if smoke else config.batch,rate,config.seed+1,deadline,metric,analog_count=n)
            evaluation=evaluate(model,dev,labels,deadline)
            if model.masks:
                # Unselected physical base remains unchanged; each dynamic view replaces only mask entries.
                for j,z in enumerate(model.z):
                    if not torch.equal(z.detach().cpu(),base[f'z.{j}']):raise AssertionError('Unselected physical base changed')
            checkpoint={name:p.detach().cpu().clone() for name,p in model.named_parameters() if p.requires_grad}
            torch.save(checkpoint,store.run_dir/'checkpoints'/f'{case}.pt')
            row={'case':case,'architecture':architecture,'arm':arm,'learning_rate':rate,'evaluation_split':'development',
                 'metrics':evaluation,'source_metrics':source_metrics,'unique_devices':n,**stats}
            result.append(row);store.append_metric(row)
            atomic_write_json(store.run_dir/'artifacts'/f'{case}.json',row)
            del params,checkpoint;gc.collect()
            if str(config.device).startswith('cuda'):torch.cuda.empty_cache()
    del model;gc.collect()
    if str(config.device).startswith('cuda'):torch.cuda.empty_cache()
    return result

def run_train(request):
    c=request.spec;torch.set_num_threads(c.cpu_threads);torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(c.input_root); inputs=[{'path':str(p),'sha256':sha256_file(p)} for p in (root/'sentiment.json',root/'inputs.sha256') if p.exists()]
    store=RunStore.create(output_root=request.output_dir,experiment_id=c.experiment_id,resolved_config=asdict(c),
                          command=request.command,repo_root=ROOT,run_id=os.environ.get('SPARSE_RUN_ID'),
                          input_artifacts=inputs,resume_capability='unsupported')
    if (ROOT/'source-snapshot.json').exists():
        atomic_write_json(store.run_dir/'artifacts/source-snapshot.json',json.loads((ROOT/'source-snapshot.json').read_text()))
    deadline=time.monotonic()+c.wall_seconds; output=[]
    print(f'RUN_DIR={store.run_dir}',flush=True)
    try:
        smoke=c.task=='canary'
        if c.task in ('canary','mlp','graph'):output.extend(graph_pilot(c,store,deadline,smoke))
        architectures=('mlp','drn') if smoke else (c.task,) if c.task in ('mlp','drn') else ()
        for architecture in architectures:output.extend(sentiment(c,architecture,store,deadline,smoke))
        result={'schema':'sparse-equilibrium.result.v1','status':'complete','task':c.task,'smoke':smoke,'records':output}
        atomic_write_json(store.run_dir/'artifacts'/'summary.json',result)
        artifacts=[store.artifact_record(p,kind='pilot_evidence') for d in ('artifacts','checkpoints') for p in (store.run_dir/d).iterdir() if p.is_file()]
        store.complete(metrics={'task':c.task,'records':len(output),'smoke':smoke},artifacts=artifacts)
        atomic_write_json(Path(request.output_dir)/'completion.json',{'schema':'sparse-equilibrium.completion.v1','status':'complete',
                         'task':c.task,'run_dir':str(store.run_dir),'summary_sha256':sha256_file(store.run_dir/'artifacts'/'summary.json')})
        return 0
    except BaseException as exc:
        store.fail(exc)
        atomic_write_json(Path(request.output_dir)/'completion.json',{'schema':'sparse-equilibrium.completion.v1',
            'status':'partial' if isinstance(exc,BudgetEnd) else 'failed','task':c.task,'run_dir':str(store.run_dir),'error':str(exc)})
        raise

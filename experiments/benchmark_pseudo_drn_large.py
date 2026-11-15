"""Readiness timing; a successful CUDA context starts the remaining budget clock."""
import argparse,json,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import torch
from ebl.runtime.store import atomic_write_json
from networks.energy_transformer.pseudo_large import Model,gradient_probe


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;budget_path=root/'budget.json';budget=json.loads(budget_path.read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    guard=torch.empty(1,device='cuda')
    if budget.get('deadline') is None:
        now=datetime.now(timezone.utc)
        budget.update(state='active',start=now.isoformat(),deadline=(now+timedelta(seconds=budget['remaining_seconds'])).isoformat())
        atomic_write_json(budget_path,budget)
    if time.time()>=datetime.fromisoformat(budget['deadline']).timestamp()-900:raise TimeoutError('Insufficient original allowance for readiness')
    x=torch.randn(64,64,48,device='cuda');target=torch.randint(10,(64,),device='cuda');results={}
    for mlp in ['drn','digital']:
        model=Model(mlp=mlp).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.05)
        torch.cuda.reset_peak_memory_stats();times=[]
        for step in range(12):
            torch.cuda.synchronize();started=time.monotonic();optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x,'ep' if mlp=='drn' else 'bptt',32),target)
            loss.backward();optimizer.step();torch.cuda.synchronize();times.append(time.monotonic()-started)
            print({'mlp':mlp,'step':step,'loss':float(loss),'seconds':times[-1]},flush=True)
        probe=gradient_probe(model,x[:2],target[:2]);atomic_write_json(root/f'benchmark-gradients-{mlp}.json',probe)
        result={'parameters':sum(p.numel() for p in model.parameters()),'step_seconds':sum(times[2:])/len(times[2:]),
                'peak_mib':torch.cuda.max_memory_allocated()/2**20,'gradients':{k:v for k,v in probe.items() if k!='groups'}}
        results[mlp]=result;atomic_write_json(root/'benchmark.json',results);print(result,flush=True)
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()

"""Bounded serial public-CLI CIFAR runs with a retained local CUDA context."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
    root=a.output_root;root.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    guard=torch.empty(1,device='cuda');state={'pid':os.getpid(),'gpu':torch.cuda.get_device_name(),'readiness_charge_seconds':300,'cases':[]}
    def save():(root/'supervisor.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for name in ['drn-dot-ep','drn-dot-bptt','drn-l1-ep','digital-dot-bptt']:
        if 3600-300-(time.monotonic()-start)<900:raise TimeoutError('Insufficient shared budget for next case')
        for attempt in range(2):
            path=root/f'{name}-attempt{attempt}.log'
            with path.open('w') as log:
                cmd=[sys.executable,'-m','ebl','train','--config',f'examples/pseudo_drn_scale/{name}.json','--output-dir',str(root/name)]
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);row={'case':name,'pid':child.pid,'attempt':attempt,'log':str(path),'state':'running'}
                state['cases'].append(row);save();print(row,flush=True)
                try:code=child.wait(timeout=900)
                except subprocess.TimeoutExpired:
                    child.terminate();child.wait(timeout=30);row['state']='timeout';save();raise
            row.update(state='complete' if code==0 else 'failed',returncode=code);save();print(row,flush=True)
            if code==0:break
            if attempt==0 and 'CUDA driver initialization failed' in path.read_text() and "'stage': 'train'" not in path.read_text():continue
            raise RuntimeError(f'Failed {name}, retained {path}')
    state.update(complete=True,seconds=time.monotonic()-start);save()

if __name__=='__main__':main()

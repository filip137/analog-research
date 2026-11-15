"""Serial bounded native-CLI composition with retained CUDA context."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--readiness-seconds',type=float,default=300)
    args=p.parse_args();root=args.output_root;root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();guard=torch.empty(1,device='cuda')
    state={'supervisor_pid':os.getpid(),'gpu':torch.cuda.get_device_name(),'context_bytes':guard.numel()*guard.element_size(),'cases':[]}
    def save(): (root/'supervisor.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for task in ['completion','permuted_classification']:
        for method in ['ep','bptt']:
            name=f'{task}-{method}'
            if 3600-args.readiness_seconds-(time.monotonic()-start)<900: raise TimeoutError('Shared allowance exhausted')
            cmd=[sys.executable,'-m','ebl','train','--config',f'examples/pseudo_mnist/{name}.json','--output-dir',str(root/name)]
            for attempt in range(2):
                log=root/f'{name}-attempt{attempt}.log'
                with log.open('w') as f:
                    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
                    record={'case':name,'attempt':attempt,'pid':child.pid,'log':str(log),'state':'running'}
                    state['cases'].append(record);save();print(record,flush=True)
                    try: code=child.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        child.terminate();child.wait(timeout=30);record['state']='timeout';save();raise
                record.update(state='complete' if code==0 else 'failed',returncode=code);save()
                print(record,flush=True)
                if code==0: break
                text=log.read_text()
                if attempt==0 and 'CUDA driver initialization failed' in text and "'stage': 'train'" not in text: continue
                raise RuntimeError(f'{name} failed; preserved {log}')
    state['complete']=True;state['seconds']=time.monotonic()-start;save()

if __name__=='__main__':main()

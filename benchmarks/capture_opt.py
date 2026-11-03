"""Engineering replay capture/comparison; explicit baseline or refactor source layout.

Run as a module in the selected checkout, or feed this file on stdin there.
No source paths are added to sys.path. See the architecture migration pilot.
"""
import sys, importlib
from pathlib import Path
import torch
legacy=sys.argv[1]=='baseline'
model=importlib.import_module('workflow.networks.opt_mlp' if legacy else 'networks.opt')
adapt=importlib.import_module('workflow.drift_recovery' if legacy else 'networks.opt_drift')
torch.set_num_threads(1)
result={}
for scope in ('analog','model'):
 torch.manual_seed(1729)
 config=model.OptConfig(64,16,3,2,32,32)
 teacher=model.OptDecoder(config).eval().requires_grad_(False)
 network=model.OptMlpSuffix(teacher,2,8,scope=scope)
 adapter=adapt.CrossbarAdapter(network,'cpu')
 tokens=torch.randint(0,64,(2,9),generator=torch.Generator().manual_seed(2718))
 q=network.q.detach().clone().requires_grad_(True)
 out=network(tokens,q);out.square().mean().backward()
 result[scope+'/native']={'logits':out.detach(),'gradient':q.grad}
 for capture in (False,True):
  for corrected in (False,True):
   physical,*_=adapter.physical_layout()
   physical=[v.detach().clone().requires_grad_(True) for v in physical]
   correction=[(torch.full((16,),1.02),torch.full((16,),.01)) for _ in range(2)] if corrected else None
   output=adapter.forward(tokens,physical,correction,capture)
   logits,pairs=output if capture else (output,[])
   logits.square().mean().backward()
   result[f'{scope}/{capture}/{corrected}']={'logits':logits.detach(),'gradients':[v.grad for v in physical],
       'captures':[(a.detach(),b.detach()) for a,b in pairs]}
torch.save(result,sys.argv[2]);print(len(result),'OPT replay scenarios captured')

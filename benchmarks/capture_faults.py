"""Engineering replay capture/comparison; explicit baseline or refactor source layout.

Run as a module in the selected checkout, or feed this file on stdin there.
No source paths are added to sys.path. See the architecture migration pilot.
"""
import sys,importlib
import torch
legacy=sys.argv[1]=='baseline'
arrays=importlib.import_module('workflow.devices' if legacy else 'hardware.arrays')
config=importlib.import_module('workflow.lifecycle' if legacy else 'hardware.config')
MatrixSpec=importlib.import_module('experiments.cifar_crossbar.model' if legacy else 'hardware.mapping').MatrixSpec
layout=[];offset=0
for name,shape in [('decoder.layers.1.fc1',(32,16)),('decoder.layers.1.fc2',(16,32)),('decoder.layers.1.fc1.lora_a',(2,16)),('decoder.layers.1.fc1.lora_b',(32,2))]:
 size=shape[0]*shape[1];layout.append(MatrixSpec(name,shape,offset,size,1.));offset+=size
kwargs={} if legacy else {'domains':importlib.import_module('networks.mapping').fault_domains(layout)}
result={}
for kind in ('open','gmin','gmax','random'):
 for target in ('all','base','adapter'):
  case=config.DefectCase(kind,50000,hierarchical=True,die_concentration=.5,array_concentration=4.,target=target)
  result[kind+'/'+target]=arrays.hierarchical_faults(layout,8,12,case,271001,'facebook/opt-125m',**kwargs)
torch.save(result,sys.argv[2]);print(len(result),'fault scenarios captured')

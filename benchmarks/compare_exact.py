"""Engineering replay capture/comparison; explicit baseline or refactor source layout.

Run as a module in the selected checkout, or feed this file on stdin there.
No source paths are added to sys.path. See the architecture migration pilot.
"""
import sys,torch
count=0
def compare(a,b,path='root'):
 global count
 if isinstance(a,torch.Tensor):
  assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),path
  count+=1
 elif isinstance(a,dict):
  assert list(a)==list(b),(path,'keys')
  for k in a:compare(a[k],b[k],f'{path}/{k}')
 elif isinstance(a,(list,tuple)):
  assert type(a)==type(b) and len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}/{i}')
 else:assert a==b,(path,a,b)
a,b=[torch.load(p,weights_only=True) for p in sys.argv[1:]]
compare(a,b);print(f'Exact equality: {len(a)} scenarios, {count} tensors, all scalar metadata and receipts')

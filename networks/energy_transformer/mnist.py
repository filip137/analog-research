"""Identical image-space masks across architectures and patch granularities."""
import struct
from pathlib import Path
import numpy as np
import torch


def load_raw(root):
    root = Path(root).expanduser()/'MNIST'/'raw'
    images,labels,paths = {},{},[]
    for split,prefix,count in (('train','train',60000),('test','t10k',10000)):
        ip,lp = root/f'{prefix}-images-idx3-ubyte',root/f'{prefix}-labels-idx1-ubyte'
        raw=ip.read_bytes(); header=struct.unpack('>IIII',raw[:16])
        lab=lp.read_bytes()
        if header!=(2051,count,28,28) or struct.unpack('>II',lab[:8])!=(2049,count):
            raise ValueError('Unexpected MNIST IDX header')
        images[split]=torch.from_numpy(np.frombuffer(raw[16:],dtype=np.uint8).copy().reshape(count,28,28))
        labels[split]=torch.from_numpy(np.frombuffer(lab[8:],dtype=np.uint8).copy())
        paths.extend((ip,lp))
    gen=torch.Generator().manual_seed(3100)
    train,val=[],[]
    for label in range(10):
        ids=torch.where(labels['train']==label)[0]
        ids=ids[torch.randperm(len(ids),generator=gen)]
        val.append(ids[:500]);train.append(ids[500:])
    train=torch.cat(train);val=torch.cat(val)
    train=train[torch.randperm(len(train),generator=gen)]
    val=val[torch.randperm(len(val),generator=gen)]
    return images,{'train':train,'validation':val,'test':torch.arange(10000)},paths


def hidden_masks(count,generator):
    scores=torch.rand(count,16,generator=generator)
    hidden=torch.zeros(count,16,dtype=torch.bool)
    hidden.scatter_(1,scores.argsort(-1)[:,:8],True)
    return hidden.reshape(count,4,4).repeat_interleave(7,1).repeat_interleave(7,2)


def patchify(images,patch):
    n=28//patch
    return images.reshape(-1,n,patch,n,patch).permute(0,1,3,2,4).reshape(-1,n*n,patch*patch)


def unpatchify(tokens,patch):
    n=28//patch
    return tokens.reshape(-1,n,n,patch,patch).permute(0,1,3,2,4).reshape(-1,28,28)


def batch(images,hidden,patch,device='cuda'):
    target=patchify(images.to(device=device,dtype=torch.float32)/255*.8-.4,patch)
    mask=patchify(hidden.to(device=device,dtype=torch.float32),patch)
    x=torch.cat((target*(1-mask),1-mask),-1)
    return x,target,mask

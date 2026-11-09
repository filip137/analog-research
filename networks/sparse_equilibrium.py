"""Physical-coordinate OPT adapters and a small grounded graph control."""
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from networks.opt.decoder import OptConfig, from_hugging_face
from networks.opt.drn import reconstruct_source, SourceDecoder, OptDrn
from networks.resistive.differentiable import DrnSettings
from networks.opt.data import pinned_load

TEACHER_SHA256='2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9'
SOURCE_SHA256='86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028'

DRN_SHA='dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9'
OPT=OptConfig(50272,768,12,12,3072,2048)

class MaskValues(nn.Module):
    def __init__(self, z, indices):
        super().__init__()
        self.register_buffer('indices',indices)
        self.value=nn.Parameter(z.detach().flatten()[indices].clone())
    def forward(self,z):
        return z.flatten().index_copy(0,self.indices,self.value).view_as(z)

class Port(nn.Module):
    def __init__(self,d,kind,rank=5):
        super().__init__(); self.kind=kind
        if kind=='boundary':
            self.a=nn.Parameter(torch.zeros(d)); self.b=nn.Parameter(torch.zeros(d))
            self.c=nn.Parameter(torch.zeros(d)); self.d=nn.Parameter(torch.zeros(d))
        elif kind=='lora':
            self.down=nn.Parameter(torch.randn(rank,d)/d**0.5)
            self.up=nn.Parameter(torch.zeros(d,rank))
    def input(self,x):
        return x*(1+self.a)+self.b if self.kind=='boundary' else x
    def output(self,x,y):
        return y*(1+self.c)+self.d if self.kind=='boundary' else y+x@self.down.T@self.up.T

class PhysicalOpt(nn.Module):
    def __init__(self, root, architecture, device):
        super().__init__(); root=Path(root); self.architecture=architecture
        teacher=from_hugging_face(pinned_load(root/'teacher.pt',TEACHER_SHA256),OPT)
        if architecture=='mlp':
            payload=pinned_load(root/'crossbar-std3.pt',SOURCE_SHA256)
            self.base=reconstruct_source(teacher,payload)
            q=payload['model']['q']; offset=0; values=[]
            for layer in self.base.layers[6:]:
                for linear in (layer.fc1,layer.fc2):
                    n=linear.weight.numel(); values.append((q[offset:offset+n].reshape_as(linear.weight)+1)/2); offset+=n
            scales=payload['model']['channel_scales']; offset=0
            for j,z in enumerate(values):
                rows=z.shape[0]
                self.register_buffer(f'scale_{j}',scales[offset:offset+rows,None].clone()*payload['model']['output_gains'][j])
                offset+=rows
        else:
            payload=pinned_load(root/'drn-std3.pt',DRN_SHA)
            skeleton=SourceDecoder(OPT); skeleton.load_state_dict(teacher.state_dict()); skeleton.start=6
            skeleton.register_buffer('gains',torch.ones(12))
            self.base=OptDrn(skeleton,DrnSettings(**payload['contract']['solver']))
            self.base.load_state_dict(payload['model'],strict=True)
            values=[(u.detach()+1)/2 for b in self.base.blocks for u in (b.u1,b.u2)]
        self.base.eval().requires_grad_(False)
        self.z=nn.ParameterList(nn.Parameter(v.detach().clone(),requires_grad=False) for v in values)
        self.masks=nn.ModuleList(); self.ports=nn.ModuleList(); self.tail=8; self.total=24
        self.to(device)

    def configure(self,arm,indices=None):
        self.requires_grad_(False); self.masks=nn.ModuleList(); self.ports=nn.ModuleList()
        if arm in ('random','fisher'):
            self.masks=nn.ModuleList(MaskValues(z,idx) for z,idx in zip(self.z,indices))
        elif arm=='full':
            for z in self.z:z.requires_grad_(True)
        elif arm in ('boundary','lora'):
            decoder=self.base if self.architecture=='mlp' else self.base.digital
            self.ports=nn.ModuleList(Port(decoder.config.hidden_size,arm).to(self.z[0].device) for _ in range(6))
        elif arm=='bias':
            for k in range(6):
                ps=((self.base.layers[k+6].fc1.bias,self.base.layers[k+6].fc2.bias)
                    if self.architecture=='mlp' else (self.base.blocks[k].bias1,self.base.blocks[k].bias2))
                for p in ps:p.requires_grad_(True)
        elif arm!='frozen':raise ValueError(arm)
        return [p for p in self.parameters() if p.requires_grad]

    def physical(self):
        return [m(z) for m,z in zip(self.masks,self.z)] if self.masks else list(self.z)

    def forward(self,tokens,label_ids):
        decoder=self.base if self.architecture=='mlp' else self.base.digital
        # The prefix is frozen; its activations are not retained for backward.
        with torch.no_grad():
            h=decoder.embed(tokens)
            for layer in decoder.layers[:6]:h=layer(h)
        zs=self.physical()
        for k,layer in enumerate(decoder.layers[6:]):
            h=layer.attention(h); x=layer.final_layer_norm(h)
            xp=self.ports[k].input(x) if self.ports else x
            if self.architecture=='mlp':
                w1=(2*zs[2*k]-1)*getattr(self,f'scale_{2*k}')
                w2=(2*zs[2*k+1]-1)*getattr(self,f'scale_{2*k+1}')
                y=F.linear(F.relu(F.linear(xp,w1,layer.fc1.bias)),w2,layer.fc2.bias)
            else:
                b=self.base.blocks[k]; grad=torch.is_grad_enabled()
                b.solver.settings=replace(b.solver.settings,
                    settling_iterations=self.total-self.tail if grad else self.total,gradient_iterations=self.tail)
                b.solver.train(grad)
                view=tuple((9+79.2*z)/88.2 for z in zs[2*k:2*k+2])
                y=b(xp,view)
            if self.ports:y=self.ports[k].output(x,y)
            h=h+y
        final=decoder.final_layer_norm(h[:,-1])
        return final@decoder.embed_tokens.weight[label_ids].T

    @torch.no_grad()
    def clamp(self):
        if self.masks:
            for m in self.masks:m.value.clamp_(0,1)
        else:
            for z in self.z:z.clamp_(0,1)

def grid_edges(side=4):
    pairs=[]
    for i in range(side):
        for j in range(side):
            v=i*side+j
            if i+1<side:pairs.append((v,v+side))
            if j+1<side:pairs.append((v,v+1))
    b=torch.zeros(len(pairs),side*side)
    for i,(u,v) in enumerate(pairs):b[i,u]=1;b[i,v]=-1
    return b

def graph_solve(x,g,b,nonnegative,iterations=100):
    h=torch.eye(b.shape[1],device=x.device,dtype=x.dtype)+b.T@(g[:,None]*b)
    if not nonnegative:return torch.linalg.solve(h,x.T).T
    # Deterministic projected gradient; bounded g makes this a contraction.
    step=1/(1+2*(g[:,None]*b.abs()).sum(0).max())
    v=torch.zeros_like(x)
    for _ in range(iterations):v=(v-step*(v@h-x)).clamp_min(0)
    return v

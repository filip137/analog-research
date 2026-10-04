"""Layer-bound populations and corruption-aware endpoint samplers."""
from hashlib import sha256
import torch
from experiments.cifar_crossbar.devices import clone_cpu, prepare_population, population_fingerprint, make_plant, program, EndpointKernel
from experiments.cifar_crossbar.full_epoch_runtime import om_population
from experiments.cifar_crossbar.pcm_faults import GaussianEndpointArray, PermanentFaults
from experiments.cifar_crossbar.model import state_hash
from experiments.cifar_crossbar.runtime import save, load
from experiments.artifacts import sha256_file


def binding_seed(seed, dataset, layer):
    return int.from_bytes(sha256(f'{seed}:{dataset}:{layer}'.encode()).digest()[:4], 'little') % 2000000000 + 1


class LayeredPcmArray:
    def __init__(self, target, layout, dataset, kind, rate, seed, endpoint):
        self.children = []
        self.layout = layout
        for m in layout:
            f = PermanentFaults.sample(m.size,kind,rate,binding_seed(seed,dataset,m.name),target.device)
            self.children.append(GaussianEndpointArray(target[m.offset:m.offset+m.size],f,binding_seed(endpoint,dataset,m.name)))
        self.faults = PermanentFaults(torch.cat([a.faults.mask for a in self.children],1),
                                     torch.cat([a.faults.conductance for a in self.children],1),kind,rate,seed)
    def read(self): return torch.cat([a.read() for a in self.children])
    def program(self,target):
        for a,m in zip(self.children,self.layout): a.program(target[m.offset:m.offset+m.size])
    def state_dict(self): return {'kind':'layered_pcm','children':[a.state_dict() for a in self.children]}
    def load_state_dict(self,s):
        if s.get('kind')!='layered_pcm' or len(s['children'])!=len(self.children): raise ValueError('PCM layout changed.')
        for a,v in zip(self.children,s['children']): a.load_state_dict(v)
    def verify_permanence(self):
        for a in self.children: a.verify_permanence()
    def cost(self):
        rows=[a.cost() for a in self.children]
        if len({r['array_reprogram_calls'] for r in rows})!=1: raise RuntimeError('Partial layer write.')
        return {k:(rows[0][k] if k=='array_reprogram_calls' else sum(r[k] for r in rows)) for k in rows[0]}


def combined_om(populations, layout, dataset, seed, endpoint, kind, rate):
    pieces=[]
    for m in layout:
        raw=populations['layers'][str(seed)][m.name]
        if raw['size']!=m.size or raw['seed']!=binding_seed(seed,dataset,m.name): raise ValueError('OM layer binding mismatch.')
        pieces.append(om_population(raw,kind,rate,binding_seed(seed,dataset,m.name)))
    p={k:clone_cpu(v) for k,v in pieces[0].items() if k not in ('hidden','hidden_sha256','fingerprint')}
    if any(x['constants']!=p['constants'] for x in pieces): raise ValueError('Different OM presets.')
    p.update(size=sum(m.size for m in layout),seed=seed,
             hidden={k:torch.cat([x['hidden'][k] for x in pieces],1) for k in pieces[0]['hidden']},
             layer_names=[m.name for m in layout],
             trajectory_seed_blocks=[(m.size,binding_seed(endpoint,dataset,m.name)) for m in layout])
    p['fingerprint']=population_fingerprint(p)
    return p


def make_array(student,target,spec,kind,ppm,populations=None, *, seed=None):
    seed=spec.array_seed if seed is None else seed
    if spec.backend=='pcm':
        return LayeredPcmArray(target,student.layout,spec.dataset,kind,ppm/1e6,seed,seed+10000)
    p=combined_om(populations,student.layout,spec.dataset,seed,seed+10000,kind,ppm/1e6)
    a=make_plant(p,seed+10000,target.device)
    a.deployment=program(a.port(),target,tolerance=a.nominal_step*.5,maximum_pulses=128)
    return a


class MixedOmSampler:
    def __init__(self,tables,seed,device):
        self.kernels={k:EndpointKernel(v if k=='healthy' else v+torch.linspace(-1,1,v.shape[0])[:,None],seed+17*i,device) for i,(k,v) in enumerate(sorted(tables.items()))}
        self.generator=torch.Generator(device=device).manual_seed(seed+999)
    def perturb(self,q,kind,rate,strength):
        healthy=self.kernels['healthy'].perturb(q,strength)
        if not rate: return healthy
        endpoint=self.kernels[kind].perturb(q)-q
        failed=q+(endpoint-q).detach()
        mask=torch.rand(q.shape,device=q.device,generator=self.generator)<rate
        return torch.where(mask,failed,healthy)
    def state_dict(self):
        return {'mask_rng':self.generator.get_state(), 'kernel_rngs':{k:v.generator.get_state() for k,v in self.kernels.items()}}


def build_kernels(populations,spec,device,store):
    tables={}; diagnostics=[]
    grid=torch.linspace(-1,1,spec.kernel_bins,device=device).repeat_interleave(spec.kernel_samples)
    for role in ('fit','heldout'):
        tables[role]={}
        raw=populations['kernel_'+role]
        for kind in ('healthy','open','gmax','random'):
            p=prepare_population(raw) if kind=='healthy' else om_population(raw,kind,1.,raw['seed'])
            a=make_plant(p,raw['seed']+10000,device)
            stats=program(a.port(),grid,tolerance=a.nominal_step*.5,maximum_pulses=128)
            tables[role][kind]=(a.read()-grid).reshape(spec.kernel_bins,spec.kernel_samples).detach().cpu()
            diagnostics.append({'role':role,'kind':kind,'identity':p['fingerprint'],'program':stats,'cost':a.cost()})
    gates={}
    for k,a in tables['fit'].items():
        b=tables['heldout'][k]
        error=(a.mean(1)-b.mean(1)).abs()
        bound=torch.maximum(torch.full_like(error,.02),4*(a.var(1)/a.shape[1]+b.var(1)/b.shape[1]).sqrt())
        gates[k]={'adequate':bool((error<=bound).all()),'max_mean_error':float(error.max()),
                  'mean_error':error.tolist(),'bound':bound.tolist(), 'fit_sd':a.std(1).tolist(),'heldout_sd':b.std(1).tolist()}
    out={'tables':tables,'diagnostics':diagnostics,'gates':gates,'adequate':all(x['adequate'] for x in gates.values())}
    save(store.run_dir/'checkpoints/kernel.pt',out)
    if not out['adequate']: raise RuntimeError('Independent OM endpoint table mean-error gate failed.')
    return out


class CompactCheckpoints:
    """Immutable physical identities and source models are saved only once per run."""
    def __init__(self,root,save_fn=save): self.root=root; self.refs={}; self.save=save_fn
    def reference(self,label,identity,value):
        relative=f'immutable/{label}_{identity}.pt'
        if relative not in self.refs:
            p=self.root/relative
            if not p.exists(): self.save(p,value)
            self.refs[relative]={'__immutable__':relative,'sha256':sha256_file(p)}
        return self.refs[relative]
    def __call__(self,path,value):
        out=dict(value)
        if 'model' in out: out['model']=self.reference('model',state_hash(out['model']),out['model'])
        a=dict(out['array'])
        if a.get('kind')=='layered_pcm':
            children=[]
            for child in a['children']:
                child=dict(child); f=child['faults']
                identity=state_hash({'mask':f['mask'],'conductance':f['conductance']})
                child['faults']=self.reference('faults',identity,f)
                children.append(child)
            a['children']=children
        else:
            p=a['population']; a['population']=self.reference('population',p['fingerprint'],p)
        out['array']=a
        self.save(path,out)


def expand_checkpoint(value,root):
    if isinstance(value,dict):
        if set(value)=={'__immutable__','sha256'}:
            path=(root/value['__immutable__']).resolve()
            if root.resolve() not in path.parents or sha256_file(path)!=value['sha256']: raise ValueError('Invalid immutable checkpoint reference.')
            return load(path)
        return {k:expand_checkpoint(v,root) for k,v in value.items()}
    if isinstance(value,list): return [expand_checkpoint(v,root) for v in value]
    return value

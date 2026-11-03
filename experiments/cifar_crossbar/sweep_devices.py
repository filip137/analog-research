"""experiments.cifar_crossbar.sweep_devices — extracted native implementation; formats and numerical order preserved."""
from __future__ import annotations
import torch
from hardware.plants import prepare_population, make_plant
from methods.programming.pulse import program
from hardware.faults import om_population
from core.tensor_identity import state_hash
from core.serialization import save, load
from core.artifacts import sha256_file
from hardware.layered_pcm import LayeredPcmArray
from hardware.population_binding import combined_om


def make_array(student,target,spec,kind,ppm,populations=None, *, seed=None):
    seed=spec.array_seed if seed is None else seed
    if spec.backend=='pcm':
        return LayeredPcmArray(target,student.layout,spec.dataset,kind,ppm/1e6,seed,seed+10000)
    p=combined_om(populations,student.layout,spec.dataset,seed,seed+10000,kind,ppm/1e6)
    a=make_plant(p,seed+10000,target.device)
    a.deployment=program(a.port(),target,tolerance=a.nominal_step*.5,maximum_pulses=128)
    return a


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

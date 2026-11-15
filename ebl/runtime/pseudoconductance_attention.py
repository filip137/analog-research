"""Run the declared selective-routing and physical-current derivative checks."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import pseudoconductance as circuit

ROOT=Path(__file__).resolve().parents[2]

def gradient_error(actual,expected):
    absolute=float(np.linalg.norm(actual-expected))
    norm=float(np.linalg.norm(expected))
    return {'absolute_l2':absolute,'reference_l2':norm,
            'relative_l2':absolute/norm if norm>1e-6 else None,
            'passed':bool(absolute/norm<.05 if norm>1e-6 else absolute<1e-4)}

def characterize(c,store,exe):
    started=time.monotonic()
    artifacts=store.run_dir/'artifacts'
    stats=[]
    def run(name,values,controls,**kw):
        if time.monotonic()-started>c.wall_seconds:
            raise TimeoutError('Declared characterization allowance exceeded')
        result=circuit.simulate(values,controls,artifacts/name,exe,model=c.model,**kw)
        stats.append(result['metrics']|{'seconds':result['seconds']})
        store.append_metric({'case':name,'spice_calls':len(stats),'output':result['y'].tolist(),
                             'spice_seconds':sum(s['seconds'] for s in stats)})
        if len(stats)%12==0:
            print(f'{c.model}: cases={len(stats)}, last={name}',flush=True)
        return result
    rng=np.random.default_rng(c.seed)
    values=rng.uniform(-.75,.75,(36,3,2))
    selected=np.arange(36)%3
    select_out=[];uniform_out=[];ideal_out=[]
    for i,v in enumerate(values):
        controls=np.full(3,.001);controls[selected[i]]=5.
        result=run(f'select-{i:02}',v,controls)
        select_out.append(result['y']);ideal_out.append(circuit.reference(v,controls))
        uniform_out.append(run(f'uniform-{i:02}',v,np.full(3,controls.sum()/3))['y'])
    select_out=np.array(select_out);uniform_out=np.array(uniform_out)
    targets=values[np.arange(36),selected]
    base=np.array([[-.6,.4],[.2,-.5],[.7,.1]])
    mixture=[]
    for i,t in enumerate(np.linspace(0,1,11)):
        controls=np.array([5*t,5*(1-t),.001])
        result=run(f'mix-{i:02}',base,controls)
        mixture.append({'t':float(t),'output':result['y'].tolist(),
                        'reference':circuit.reference(base,controls).tolist()})
    controls=np.array([.2,.6,1.5])
    free=run('gradient-free',base,controls)
    permutation=np.array([2,0,1])
    permuted=run('permutation',base[permutation],controls[permutation])
    off=run('off',base,np.zeros(3))
    beta=.001
    e=np.array([.7,-.4])
    plus=run('nudge-plus',base,controls,nudge=beta,error=e)
    minus=run('nudge-minus',base,controls,nudge=-beta,error=e)
    lam=-(plus['x_read']-minus['x_read'])/(2*beta)
    input_estimate=lam[:3]
    drop=free['x_read'][:3]-free['x_read'][3]
    dlam=lam[:3]-lam[3]
    control_estimate=-np.sum(drop*dlam,axis=1)
    input_fd=np.zeros_like(base);control_fd=np.zeros(3)
    epsilon=1e-4
    for j,d in np.ndindex(base.shape):
        change=np.zeros_like(base);change[j,d]=epsilon
        p=run(f'value-fd-{j}-{d}-plus',base+change,controls)
        m=run(f'value-fd-{j}-{d}-minus',base-change,controls)
        input_fd[j,d]=e@(p['y']-m['y'])/(2*epsilon)
    for j in range(3):
        change=np.zeros(3);change[j]=epsilon
        p=run(f'control-fd-{j}-plus',base,controls+change)
        m=run(f'control-fd-{j}-minus',base,controls-change)
        control_fd[j]=e@(p['y']-m['y'])/(2*epsilon)
    transient=[]
    for i,start in enumerate([.2,.4]):
        transient.append(run(f'transient-{i}',base,controls,transient=True,initial=start))
    selection_mse=float(np.mean((select_out-targets)**2))
    uniform_mse=float(np.mean((uniform_out-targets)**2))
    mixerr=np.array([m['output'] for m in mixture])-np.array([m['reference'] for m in mixture])
    vg=gradient_error(input_estimate,input_fd)
    ag=gradient_error(control_estimate,control_fd)
    settling=max(float(np.max(abs(t['y']-free['y']))) for t in transient)
    metrics={'selective_rmse':float(np.sqrt(selection_mse)),
             'uniform_rmse':float(np.sqrt(uniform_mse)),
             'uniform_over_selective_mse':uniform_mse/selection_mse,
             'selective_vs_operator_rmse':float(np.sqrt(np.mean((select_out-ideal_out)**2))),
             'mixture_rmse':float(np.sqrt(np.mean(mixerr**2))),
             'permutation_max_abs_error':float(np.max(abs(permuted['y']-free['y']))),
             'off_output':off['y'].tolist(),'value_gradient':vg,'control_gradient':ag,
             'transient_vs_dc_max_abs_error':settling,
             'min_node_voltage':min(s['min_node_voltage'] for s in stats),
             'max_node_voltage':max(s['max_node_voltage'] for s in stats),
             'min_anchor_vds':min(s['min_anchor_vds'] for s in stats),
             'max_read_mirror_ratio_error':max(s['mirror_ratio_error'] for s in stats),
             'spice_seconds':sum(s['seconds'] for s in stats),
             'spice_calls':len(stats)}
    checks={'selection':metrics['selective_rmse']<.05 and uniform_mse>5*selection_mse,
            'mixture':metrics['mixture_rmse']<.05,
            'permutation':metrics['permutation_max_abs_error']<1e-6,
            'value_gradient':vg['passed'],'control_gradient':ag['passed'],
            'settling':settling<1e-3,
            'voltage_range':metrics['min_node_voltage']>0 and metrics['max_node_voltage']<circuit.VDD,
            'anchor_compliance':metrics['min_anchor_vds']>4*circuit.UT}
    summary={'model':c.model,'passed':all(checks.values()),'checks':checks,'metrics':metrics,
             'counts':free['counts'],'total_mos':sum(free['counts'].values()),
             'mixtures':mixture,
             'example':{'values':base.tolist(),'controls':controls.tolist(),'output':free['y'].tolist(),
                        'reference':circuit.reference(base,controls).tolist(),'voltages':free['voltage'].tolist(),
                        'gate_voltages':{k:float(v[-1]) for k,v in free['port'].items() if k.startswith('gate_')}},
             'gradients':{'value_fd':input_fd.tolist(),'value_nudge':input_estimate.tolist(),
                          'control_fd':control_fd.tolist(),'control_nudge':control_estimate.tolist()},
             'claim_boundary':'One-query 3-key 2-feature aggregation with supplied controls. Explicit MOS bias/input/read mirrors; ideal external reference currents, supply/reference/read-load voltages, and nudge source. Software phase storage/statistic evaluation. No analog QK scorer, gradient multiplier, PDK, mismatch/noise validation, silicon, latency/energy advantage or task-training claim.'}
    np.savez_compressed(artifacts/'routing.npz',values=values,selected=selected,targets=targets,
                        selective=select_out,uniform=uniform_out,operator=np.array(ideal_out))
    atomic_write_json(artifacts/'summary.json',summary)
    return summary

def run_characterize(request):
    if any(getattr(request,k,None) is not None for k in ['weights','base_weights','resume','device_data','device_model']):
        raise ConfigError('This standalone circuit pilot does not accept checkpoints or device datasets')
    c=request.spec
    exe,version=executable_identity(c.spice_executable)
    store=RunStore.create(output_root=request.output_dir,experiment_id=c.experiment_id,
        resolved_config=asdict(c),command=request.command,repo_root=ROOT,
        input_artifacts=[{'path':str(exe),'sha256':sha256_file(exe)}],resume_capability='unsupported')
    print(f'RUN_DIR={store.run_dir}',flush=True)
    try:
        atomic_write_json(store.run_dir/'artifacts'/'environment.json',{'simulator':str(exe),'version':version,'numpy':np.__version__})
        result=characterize(c,store,exe)
        artifacts=[store.artifact_record(p,kind='circuit_characterization') for p in sorted((store.run_dir/'artifacts').rglob('*')) if p.is_file()]
        if sum(a.size_bytes for a in artifacts)>250_000_000:
            raise RuntimeError('Per-model storage allowance exceeded')
        store.complete(metrics={'passed':result['passed'],**result['metrics']},artifacts=artifacts)
        print(result['checks'],flush=True)
        return 0 if result['passed'] else 1
    except BaseException as err:
        store.fail(err)
        raise

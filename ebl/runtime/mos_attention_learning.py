"""Train only from measured SPICE phases; no numerical solver or BP gradients."""
from dataclasses import asdict,replace
from pathlib import Path
import gzip
import shutil
import subprocess
import time
import numpy as np
from scipy.special import softmax

from ebl.runtime.store import RunStore,atomic_write_json,sha256_file
from ebl.schema.base import ConfigError
from networks.energy_transformer.convergent import PARAMETERS,loss
from networks.energy_transformer.convergent_spice import Builder,export_netlist,run_spice
from networks.energy_transformer.learning import initial_parameters,boundary,dataset,measured_residual
from networks.energy_transformer import transistor
from networks.energy_transformer.spice import executable_identity
from methods.convergent_energy_transformer import phase_gradient

ROOT=Path(__file__).resolve().parents[2]


def compress(directory):
    """Lossless declared retention of every netlist, log and measured terminal port."""
    total=0
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix in (".cir",".log",".tsv"):
            dest=path.with_suffix(path.suffix+".gz")
            with path.open("rb") as src,gzip.open(dest,"wb",compresslevel=1) as dst:
                shutil.copyfileobj(src,dst)
            path.unlink()
            total+=dest.stat().st_size
    return total


class Circuit:
    def __init__(self,root,executable,config,deadline):
        self.root=Path(root);self.executable=executable;self.config=config;self.deadline=deadline
        self.calls=0;self.bytes=0;self.extensions=0
        self.worst={"residual":0.,"product_error":0.,"tail_error":0.,"vds_error":0.,
                    "softmax_approximation_error":0.,"maximum_operand":0.,
                    "minimum_vds":float('inf'),"minimum_triode_margin":float('inf')}

    def remaining(self):
        remaining=self.deadline-time.monotonic()
        if remaining<=0: raise TimeoutError("Attention training wall budget exhausted")
        if self.bytes>2_000_000_000: raise RuntimeError("Attention artifact budget exceeded")
        return min(remaining,self.config.spice_timeout)

    def phase(self,p,b,label,nudge=0.,full=False):
        for attempt,duration in enumerate((300.,1200.,4800.)):
            folder=self.root/label/f"duration-{duration:g}"
            self.calls+=1
            trace=run_spice(p,b,folder,executable=self.executable,nudge=nudge,
                            softmax_device=transistor.SOFTMAX_DEVICE,multiplier="mos_triode",
                            terminal_only=not full,duration=duration,timeout=self.remaining())
            actual={name:a[-1] for name,a in trace.ports.items()}
            residual=measured_residual(actual,b,nudge)
            product=float(np.max(np.abs(trace.ports['product_output']-
                        trace.ports['product_left']*trace.ports['product_right'])))
            domain=transistor.domain_metrics(trace.ports['product_left'],trace.ports['product_right'])
            tail=float(np.max(np.abs(actual['p'].sum(-1)-1)))
            vds=float(np.max(np.abs(actual['softmax_drains']-
                       actual['softmax_sources'][...,None]-transistor.SOFTMAX_VDS)))
            logits=np.einsum('hia,hja->hij',actual['q'],actual['k'])
            approximation=float(np.max(np.abs(actual['p']-softmax(p.kappa*logits,axis=-1))))
            if product>1e-7 or domain['minimum_vds']<=0 or domain['minimum_triode_margin']<=0:
                raise RuntimeError("Native multiplier left its bilinear domain")
            if domain['maximum_operand']>transistor.INPUT_LIMIT or tail>1e-7 or vds>1e-8 or actual['p'].min()<-1e-8:
                raise RuntimeError("Native attention operating-domain failure")
            self.bytes+=compress(folder)
            if residual>1e-7:
                self.extensions+=1
                continue
            metrics={**domain,"residual":residual,"product_error":product,"tail_error":tail,
                     "vds_error":vds,"softmax_approximation_error":approximation}
            for name,value in metrics.items():
                self.worst[name]=(min if name.startswith('minimum') else max)(self.worst[name],value)
            return trace.states[-1],actual
        raise RuntimeError(f"Unsettled SPICE phase {label}: residual={residual:g}")


def primitive_checks(circuit):
    """Actual DC circuit finite differences, not an analytic substitute for BSIM."""
    root=circuit.root/"bank-checks";root.mkdir(parents=True)
    serial=0
    def bank(point):
        nonlocal serial
        folder=root/str(serial);folder.mkdir();serial+=1
        net=Builder(transistor.SOFTMAX_DEVICE)
        for i,v in enumerate(point): net.lines.append(f"Vinput{i} input{i} 0 {v:.17g}")
        outputs=net.softmax([f"input{i}" for i in range(len(point))],1.)
        net.lines += ['.control','set noaskquit','set numdgt=15','set wr_singlescale','set wr_vecnames','op',
                      'wrdata bank.tsv '+' '.join(f'v({n})' for n in outputs),'quit','.endc','.end']
        (folder/'bank.cir').write_text('\n'.join(net.lines)+'\n')
        with (folder/'stdout.log').open('w') as log:
            subprocess.run([str(circuit.executable),'-n','-b','-o','spice.log','bank.cir'],cwd=folder,
                           stdout=log,stderr=subprocess.STDOUT,timeout=circuit.remaining(),check=True)
        result=np.loadtxt(folder/'bank.tsv',skiprows=1,ndmin=2)[-1,1:]
        circuit.bytes+=compress(folder)
        return result
    point=np.array([-2.,-.5,1.]);step=1e-4;base=bank(point)
    common=float(np.max(np.abs(bank(point+5)-base)))
    jac=np.column_stack([(bank(point+step*e)-bank(point-step*e))/(2*step) for e in np.eye(3)])
    symmetry=float(np.max(np.abs(jac-jac.T)))
    eig=float(np.linalg.eigvalsh((jac+jac.T)/2).min())
    tail=float(abs(base.sum()-1.))
    return {"symmetry_error":symmetry,"minimum_eigenvalue":eig,"common_mode_error":common,
            "tail_error":tail,"jacobian":jac.tolist(),
            "passed":symmetry<1e-6 and eig>-1e-7 and common<1e-7 and tail<1e-7}


def gradient_check(circuit,p,label):
    """Centered EP versus finite differences of actual circuit equilibrium loss."""
    b=boundary([.3,-.1]);beta=circuit.config.beta
    free,fp=circuit.phase(p,b,label+'/free',full=True)
    minus,mp=circuit.phase(p,b,label+'/minus',-beta)
    plus,pp=circuit.phase(p,b,label+'/plus',beta)
    estimate=phase_gradient(p,b,minus,plus,beta,mp,pp)
    rng=np.random.default_rng(20261114);rows={};step=1e-4
    for name in PARAMETERS:
        value=getattr(p,name);direction=rng.normal(size=value.shape);direction/=np.linalg.norm(direction)
        costs=[]
        for sign in (-1,1):
            changed=replace(p,**{name:value+sign*step*direction})
            state,_=circuit.phase(changed,b,f'{label}/fd-{name}-{sign}')
            costs.append(loss(changed,state,b))
        finite=(costs[1]-costs[0])/(2*step)
        ep=float(np.sum(estimate[name]*direction));error=abs(ep-finite)
        rows[name]={"finite_difference":finite,"ep":ep,"absolute_error":error,
                    "relative_error":error/max(abs(finite),1e-8),
                    "passed":error<=1e-7+.01*abs(finite)}
    response=float(np.linalg.norm(plus-minus))
    return {"directions":rows,"nudge_response_l2":response,
            "passed":all(row['passed'] for row in rows.values()) and response<.1}


def evaluate(circuit,p,pairs,label):
    predictions=[]
    for i,pair in enumerate(pairs):
        state,ports=circuit.phase(p,boundary(pair),f'{label}/sample-{i:02d}')
        predictions.append(float(ports['y'][2,0]))
    predictions=np.array(predictions);target=pairs.mean(1)
    np.savez_compressed(circuit.root/(label+'.npz'),pairs=pairs,target=target,predictions=predictions)
    return float(np.mean((predictions-target)**2))


def checkpoint(root,epoch,p,m,v):
    np.savez_compressed(root/f'checkpoint-{epoch:04d}.npz',epoch=epoch,epsilon=p.epsilon,kappa=p.kappa,
        **{name:getattr(p,name) for name in PARAMETERS},
        **{f'adam_m_{name}':m[name] for name in PARAMETERS},
        **{f'adam_v_{name}':v[name] for name in PARAMETERS})


def train(config,store,executable,deadline):
    root=store.run_dir/'artifacts';circuit=Circuit(root,executable,config,deadline)
    p=initial_parameters();train_pairs=dataset(config.train_seed,config.train_count)
    test_pairs=dataset(config.test_seed,config.test_count)
    np.savez_compressed(root/'data.npz',train_pairs=train_pairs,test_pairs=test_pairs)
    netlist,_=export_netlist(p,boundary(train_pairs[0]),softmax_device=transistor.SOFTMAX_DEVICE,multiplier='mos_triode')
    lines=netlist.splitlines()
    counts={prefix:sum(line.startswith(prefix) for line in lines) for prefix in ('Mproduct','Msoft','Rweight','Cstate','Bmultiply','Bmoschannel')}
    m={name:np.zeros_like(getattr(p,name)) for name in PARAMETERS};v={name:a.copy() for name,a in m.items()}
    checkpoint(root,0,p,m,v)
    bank=primitive_checks(circuit);initial_gradient=gradient_check(circuit,p,'initial-gradient')
    atomic_write_json(root/'initial-checks.json',{'bank':bank,'gradient':initial_gradient})
    if not bank['passed'] or not initial_gradient['passed']: raise RuntimeError('Native circuit readiness failed')
    initial_test=evaluate(circuit,p,test_pairs,'initial-test')
    initial_train=None;history=[]
    for epoch in range(config.epochs):
        gradients={name:np.zeros_like(getattr(p,name)) for name in PARAMETERS};costs=[]
        for i,pair in enumerate(train_pairs):
            b=boundary(pair);label=f'train-{epoch:04d}/sample-{i:02d}'
            free,fp=circuit.phase(p,b,label+'/free')
            minus,mp=circuit.phase(p,b,label+'/minus',-config.beta)
            plus,pp=circuit.phase(p,b,label+'/plus',config.beta)
            gradient=phase_gradient(p,b,minus,plus,config.beta,mp,pp)
            costs.append(2*loss(p,free,b))
            for name in PARAMETERS: gradients[name]+=gradient[name]/len(train_pairs)
        train_mse=float(np.mean(costs))
        if initial_train is None: initial_train=train_mse
        new={}
        for name in PARAMETERS:
            grad=gradients[name]
            if not np.isfinite(grad).all(): raise RuntimeError('Nonfinite measured EP gradient')
            m[name]=.9*m[name]+.1*grad;v[name]=.999*v[name]+.001*grad**2
            new[name]=getattr(p,name)-config.learning_rate*(m[name]/(1-.9**(epoch+1)))/(np.sqrt(v[name]/(1-.999**(epoch+1)))+1e-8)
        p=replace(p,**new)
        row={'epoch':epoch,'train_mse':train_mse,'spice_phases':circuit.calls,
             'settling_extensions':circuit.extensions,'seconds_remaining':deadline-time.monotonic()}
        history.append(row);store.append_metric(row)
        print(f"epoch={epoch} train_mse={train_mse:.8g} phases={circuit.calls}",flush=True)
        if (epoch+1)%20==0 or epoch+1==config.epochs: checkpoint(root,epoch+1,p,m,v)
    final_train=evaluate(circuit,p,train_pairs,'final-train')
    final_test=evaluate(circuit,p,test_pairs,'final-test')
    no_attention=replace(p,query=np.zeros_like(p.query),key=np.zeros_like(p.key))
    ablation=evaluate(circuit,no_attention,test_pairs,'no-attention-test')
    final_gradient=gradient_check(circuit,p,'final-gradient')
    baseline=float(np.mean(test_pairs.mean(1)**2))
    checks={'bank_energy_map':bank['passed'],'initial_circuit_ep':initial_gradient['passed'],
            'final_circuit_ep':final_gradient['passed'],'held_out_reduction':final_test<=.25*initial_test,
            'beats_zero_baseline':final_test<baseline,'attention_ablation':ablation>final_test,
            'native_attention':counts['Bmultiply']==counts['Bmoschannel']==0 and counts['Mproduct']==108 and counts['Msoft']==9}
    summary={'schema':'mos-attention-learning.result.v1','passed':all(checks.values()),'checks':checks,
             'initial_train_mse':initial_train,'final_train_mse':final_train,
             'initial_test_mse':initial_test,'final_test_mse':final_test,'zero_baseline_mse':baseline,
             'no_attention_test_mse':ablation,'epochs':config.epochs,'spice_phases':circuit.calls,
             'settling_extensions':circuit.extensions,'worst_circuit_metrics':circuit.worst,
             'counts':counts,'logical_parameters':sum(getattr(p,n).size for n in PARAMETERS),
             'bank_check':bank,'initial_gradient':initial_gradient,'final_gradient':final_gradient,
             'claim_boundary':'Three-token custom convergent energy attention. Native nominal LEVEL=1 products and generic BSIM3 normalized-current bank with ideal drain regulation. Ideal normalization/ReLU/buffers and exact resistor programming; digital Adam/phase-statistic controller. All evaluated phases use SPICE. No independent V projection, foundry PDK, silicon or efficiency claim.'}
    atomic_write_json(root/'summary.json',summary)
    atomic_write_json(root/'history.json',history+[{'epoch':config.epochs,'train_mse':final_train}])
    return summary


def run_train(request):
    unsupported=("weights","base_weights","resume","device_data","device_model",
                 "teacher_weights","deployment","device_state")
    if any(getattr(request,name,None) is not None for name in unsupported):
        raise ConfigError("The MOS attention pilot uses its declared initialization; external inputs/resume are unsupported")
    config=request.spec;executable,version=executable_identity(config.spice_executable)
    deadline=time.monotonic()+config.wall_seconds
    store=RunStore.create(output_root=request.output_dir,experiment_id=config.experiment_id,
        resolved_config=asdict(config),command=request.command,repo_root=ROOT,
        input_artifacts=[{'path':str(executable),'sha256':sha256_file(executable)}],resume_capability='unsupported')
    print(f'RUN_DIR={store.run_dir}',flush=True)
    try:
        atomic_write_json(store.run_dir/'artifacts'/'environment.json',{'spice_executable':str(executable),'spice_version':version,
                            'numpy':np.__version__,'softmax_device':transistor.SOFTMAX_DEVICE})
        result=train(config,store,executable,deadline)
        artifacts=[store.artifact_record(path,kind='mos_attention_learning') for path in sorted((store.run_dir/'artifacts').rglob('*')) if path.is_file()]
        if sum(a.size_bytes for a in artifacts)>2_000_000_000: raise RuntimeError('Artifact budget exceeded')
        if time.monotonic()>deadline: raise TimeoutError('Attention training budget exceeded')
        store.complete(metrics={'passed':result['passed'],'final_test_mse':result['final_test_mse']},artifacts=artifacts)
        return 0 if result['passed'] else 1
    except BaseException as error:
        store.fail(error)
        raise

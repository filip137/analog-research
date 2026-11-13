"""Native circuit-readiness bundle; no task-training claim follows from this pilot."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from ebl.runtime.store import RunStore, atomic_write_json, sha256_file
from networks.energy_transformer.convergent import (PARAMETERS,pilot_case,force,branch_force,
    simulate,energy,torch_energy,dissipation,ports,loss)
from networks.energy_transformer.convergent_spice import run_spice
from networks.energy_transformer.spice import executable_identity
from networks.energy_transformer import mosfet
from methods.convergent_energy_transformer import (eqprop,phase_gradient,
    implicit_gradient,finite_difference_directions,update)

ROOT=Path(__file__).resolve().parents[2]


def errors(estimate,reference):
    result={}
    for name in PARAMETERS:
        absolute=float(np.linalg.norm(estimate[name]-reference[name]))
        norm=float(np.linalg.norm(reference[name]))
        result[name]={"absolute_l2":absolute,"reference_l2":norm,
                      "relative_l2":absolute/norm if norm>1e-8 else None}
    return result


def gradients_pass(values):
    return all(v["relative_l2"]<=2e-3 if v["relative_l2"] is not None else v["absolute_l2"]<=1e-8 for v in values.values())


def characterize(config,store,executable,deadline):
    def remaining():
        left=deadline-time.monotonic()
        if left<=0: raise TimeoutError("Declared CPU circuit-check budget exhausted")
        return min(config.spice_timeout,left)
    p,b=pilot_case()
    root=store.run_dir/"artifacts"
    np.savez_compressed(root/"initial.npz",**{name:getattr(p,name) for name in PARAMETERS},inputs=b.inputs,target=b.target,mask=b.mask,epsilon=p.epsilon,kappa=p.kappa)
    starts=[np.zeros(p.size),np.linspace(-.4,.5,p.size),np.linspace(.5,-.4,p.size)]
    metrics={"numerical_residual":0.,"start_disagreement":0.,"energy_increase":0.,
             "kcl_error":0.,"dissipation_error":0.,"spice_state_error":0.,
             "spice_transient_error":0.,"spice_port_error":0.,"spice_kcl_residual":0.}
    device_metrics=[]
    free=None
    tau=np.r_[np.ones(p.position.size),np.full(np.prod(p.output_shape),1.3)]
    for index,start in enumerate(starts):
        remaining()
        trace=simulate(p,b,initial=start)
        free=trace.states[-1] if free is None else free
        values=np.array([energy(p,s,b) for s in trace.states])
        metrics["numerical_residual"]=max(metrics["numerical_residual"],trace.residual)
        metrics["start_disagreement"]=max(metrics["start_disagreement"],float(np.max(np.abs(free-trace.states[-1]))))
        metrics["energy_increase"]=max(metrics["energy_increase"],float(np.max(np.diff(values))))
        for state in trace.states[::40]:
            metrics["kcl_error"]=max(metrics["kcl_error"],float(np.max(np.abs(force(p,state,b)-branch_force(p,state,b)))))
            raw=torch.tensor(state,dtype=torch.float64,requires_grad=True)
            grad,=torch.autograd.grad(torch_energy(p,raw,b),raw)
            actual=float(grad.detach().numpy()@(force(p,state,b)/tau))
            metrics["dissipation_error"]=max(metrics["dissipation_error"],abs(actual-dissipation(p,state,b)))
        np.savez_compressed(root/f"numerical-{index}.npz",time=trace.time,states=trace.states,energy=values)
    reference,min_eig=implicit_gradient(p,free,b)
    metrics["minimum_activation_hessian_eigenvalue"]=min_eig
    metrics["finite_difference"]=finite_difference_directions(p,b,reference)
    metrics["relu_margin"]=float(np.min(np.abs(ports(p,free)["g"]@p.memory.T)))
    beta_curve=[];gradients={f"implicit_{name}":v for name,v in reference.items()}
    for beta in config.betas:
        remaining()
        estimate,minus,plus=eqprop(p,b,beta,free)
        beta_curve.append({"beta":beta,"errors":errors(estimate,reference)})
        gradients.update({f"ep_{beta:g}_{name}":v for name,v in estimate.items()})
        gradients[f"minus_{beta:g}"]=minus;gradients[f"plus_{beta:g}"]=plus
    metrics["beta_curve"]=beta_curve
    best=min(config.betas)
    phases=[(f"free-{i}",0.,start) for i,start in enumerate(starts)]
    phases += [("minus",-best,free),("plus",best,free)]
    measured={}
    for name,nudge,start in phases:
        trace=run_spice(p,b,root/name,executable=executable,nudge=nudge,initial=start,timeout=remaining(),softmax_device=config.softmax_device)
        if config.softmax_device==mosfet.DEVICE:
            device_metrics.append(mosfet.operating_metrics(trace.ports["softmax_sources"],trace.ports["softmax_gates"],trace.ports["p"]))
        numerical=simulate(p,b,nudge,initial=start,times=trace.time)
        metrics["spice_state_error"]=max(metrics["spice_state_error"],float(np.max(np.abs(trace.states[-1]-numerical.states[-1]))))
        metrics["spice_transient_error"]=max(metrics["spice_transient_error"],float(np.max(np.abs(trace.states-numerical.states))))
        actual={key:value[-1] for key,value in trace.ports.items()}
        ideal=ports(p,trace.states[-1])
        for key in ("g","q","k","p","r"):
            metrics["spice_port_error"]=max(metrics["spice_port_error"],float(np.max(np.abs(actual[key]-ideal[key]))))
        residual=max(np.max(np.abs(actual["token_drive"]-actual["z"])),np.max(np.abs(actual["output_drive"]-actual["y"]-nudge*b.mask*(actual["y"]-b.target))))
        metrics["spice_kcl_residual"]=max(metrics["spice_kcl_residual"],float(residual))
        measured[name]=(trace.states[-1],actual)
        store.append_metric({"stage":"spice","phase":name,"residual":float(residual)})
    estimate=phase_gradient(p,b,measured["minus"][0],measured["plus"][0],best,measured["minus"][1],measured["plus"][1])
    metrics["spice_gradient"]=errors(estimate,reference)
    gradients.update({f"spice_{name}":v for name,v in estimate.items()})
    changed=update(p,estimate,config.learning_rate)
    after=run_spice(changed,b,root/"after-update",executable=executable,timeout=remaining(),softmax_device=config.softmax_device)
    if config.softmax_device==mosfet.DEVICE:
        device_metrics.append(mosfet.operating_metrics(after.ports["softmax_sources"],after.ports["softmax_gates"],after.ports["p"]))
    after_reference=simulate(changed,b).states[-1]
    _,after_eig=implicit_gradient(changed,after_reference,b)
    metrics["updated_hessian_eigenvalue"]=after_eig
    metrics["updated_state_error"]=float(np.max(np.abs(after.states[-1]-after_reference)))
    metrics["loss_before"]=loss(p,measured["free-0"][0],b)
    metrics["loss_after"]=loss(changed,after.states[-1],b)
    fixed=np.linspace(-.4,.7,p.size)
    raw=torch.tensor(fixed,dtype=torch.float64,requires_grad=True)
    gradient,=torch.autograd.grad(torch_energy(p,raw,b),raw)
    wrong=float(gradient.detach().numpy()@(force(p,fixed,b,omit_key_return=True)/tau))
    negative={"missing_key_force_error":float(np.max(np.abs(force(p,fixed,b)-force(p,fixed,b,omit_key_return=True)))),
              "missing_key_dissipation_error":abs(wrong-dissipation(p,fixed,b))}
    checks={
        "physical_kcl":metrics["kcl_error"]<=1e-10,
        "energy":metrics["dissipation_error"]<=1e-10 and metrics["energy_increase"]<=1e-9,
        "settling":metrics["numerical_residual"]<=1e-9 and metrics["start_disagreement"]<=1e-8,
        "local_minimum":min_eig>0 and after_eig>0,
        "finite_difference":max(v["relative_error"] for v in metrics["finite_difference"].values())<=2e-4,
        "eqprop":gradients_pass(next(row["errors"] for row in beta_curve if row["beta"]==best)),
        "spice_equilibrium":metrics["spice_state_error"]<=1e-5 and metrics["spice_kcl_residual"]<=1e-6,
        "spice_transient":metrics["spice_transient_error"]<=1e-4,
        "spice_ports":metrics["spice_port_error"]<=5e-6,
        "spice_ep":gradients_pass(metrics["spice_gradient"]),
        "learning":metrics["loss_after"]<metrics["loss_before"] and metrics["updated_state_error"]<=1e-5,
        "negative_control":negative["missing_key_force_error"]>1e-5 and negative["missing_key_dissipation_error"]>1e-6,
    }
    if device_metrics:
        metrics["mosfet"]={
            "weak_inversion_margin":min(v["weak_inversion_margin"] for v in device_metrics),
            "maximum_branch_current_over_i0":max(v["maximum_branch_current_over_i0"] for v in device_metrics),
            "current_law_error_over_tail":max(v["current_law_error_over_tail"] for v in device_metrics),
            "tail_fraction_error":max(v["tail_fraction_error"] for v in device_metrics),
        }
        m=metrics["mosfet"]
        checks["mosfet_operating_domain"]=m["weak_inversion_margin"]>0 and m["maximum_branch_current_over_i0"]<1
        checks["mosfet_current_law"]=m["current_law_error_over_tail"]<1e-6 and m["tail_fraction_error"]<1e-6
    np.savez_compressed(root/"gradients.npz",**gradients)
    np.savez_compressed(root/"updated.npz",**{name:getattr(changed,name) for name in PARAMETERS})
    result={"schema":"convergent-energy-transformer.result.v1","passed":all(checks.values()),"checks":checks,
            "metrics":metrics,"negative_control":negative,"coverage":{"tokens":4,"heads":2,"logical_parameters":sum(getattr(p,k).size for k in PARAMETERS),
            "free_starts":3,"numerical_nudges":2*len(config.betas),"spice_transients":6,"learning_updates":1,"parameter_groups":list(PARAMETERS)},
            "softmax_device":config.softmax_device,
            "claim_boundary":"Ideal analog-block circuit; MOSFET option uses a three-terminal analytic subthreshold device law, not a foundry compact model. Software phase-statistic/programming controller; single-example check, no task training or silicon validation."}
    atomic_write_json(root/"summary.json",result)
    return result


def run_characterize(request):
    c=request.spec
    torch.set_num_threads(1)
    executable,version=executable_identity(c.spice_executable)
    deadline=time.monotonic()+c.wall_seconds
    store=RunStore.create(output_root=request.output_dir,experiment_id=c.experiment_id,resolved_config=asdict(c),command=request.command,repo_root=ROOT,
        input_artifacts=[{"path":str(executable),"sha256":sha256_file(executable)}],resume_capability="unsupported")
    print(f"RUN_DIR={store.run_dir}",flush=True)
    try:
        atomic_write_json(store.run_dir/"artifacts"/"environment.json",{"spice_executable":str(executable),"spice_version":version,"numpy":np.__version__,"softmax_device":c.softmax_device,"stage":"circuit_characterization"})
        result=characterize(c,store,executable,deadline)
        artifacts=[store.artifact_record(path,kind="circuit_readiness") for path in sorted((store.run_dir/"artifacts").rglob("*")) if path.is_file()]
        if sum(a.size_bytes for a in artifacts)>100_000_000: raise RuntimeError("Artifact budget exceeded")
        if time.monotonic()>deadline: raise TimeoutError("Circuit check wall budget exceeded")
        store.complete(metrics={"passed":result["passed"]},artifacts=artifacts)
        return 0 if result["passed"] else 1
    except BaseException as error:
        store.fail(error)
        raise

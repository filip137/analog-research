"""Read-only iteration selection on trained checkpoints, with honest reference gates."""
from dataclasses import asdict
from pathlib import Path
import time
import torch
from ebl.runtime.store import RunStore,atomic_write_json,sha256_file
from ebl.schema.base import ConfigError
from ebl.schema.energy_transformer_mnist import CASES
from ebl.runtime.energy_transformer_mnist import strict_fp32
from networks.energy_transformer.gpu import EnergyNetwork,EquilibriumError
from networks.energy_transformer.mnist import load_raw,batch
from networks.energy_transformer.finite_time import bptt_gradient,finite_ep,agreement,select_cap

ROOT=Path(__file__).resolve().parents[2]


def phase_record(phase):
    return {'iterations':phase.iterations.max().item(),'residual':phase.residual,
            'maximum_operand':phase.maximum_operand,'minimum_triode_margin':phase.minimum_triode_margin,
            'energy_increase':phase.energy_increase}


def measured(call):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    result=call()
    torch.cuda.synchronize()
    return result,{'seconds':time.monotonic()-start,'peak_tensor_bytes':torch.cuda.max_memory_allocated()}


def evaluate(config,store,payload,cohort,images,deadline):
    n,d,patch=CASES[config.case]
    model=EnergyNetwork(n,d,patch**2,device='cuda',seed=payload['config']['seed'],epsilon=payload['config']['epsilon'])
    model.load_state_dict(payload['model'])
    original={name:p.detach().clone() for name,p in model.named_parameters()}
    count=config.batches*config.batch_size
    indices=cohort['ids']['validation'][:count]
    masks=cohort['validation_masks'][:count]
    torch.save({'validation_ids':indices,'hidden_masks':masks},store.run_dir/'artifacts/cohort.pt')
    rows=[];references=[];failures=[]
    for batch_index in range(config.batches):
        start=batch_index*config.batch_size;stop=start+config.batch_size
        x,target,mask=batch(images['train'][indices[start:stop]],masks[start:stop],patch)
        try:
            saved={}
            reference_gradients=[];reference_phases=[];reference_costs=[]
            for steps in config.reference_steps:
                store.append_metric({'stage':'reference_start','batch':batch_index,'iterations':steps})
                (grad,phase),cost=measured(lambda:bptt_gradient(model,x,target,mask,steps,deadline=deadline))
                reference_gradients.append(grad);reference_phases.append(phase);reference_costs.append(cost)
                saved[f'bptt_reference_{steps}']={name:value.cpu() for name,value in grad.items()}
            stability=agreement(reference_gradients[0],reference_gradients[1])
            ref_global=stability['global']
            valid=(ref_global['cosine'] is not None and ref_global['cosine']>=.999
                   and ref_global['relative_l2']<=.01 and reference_phases[-1].residual<=1e-5)
            ref={'batch':batch_index,'valid':valid,'agreement_512_vs_1024':stability,
                 'phases':[phase_record(p) for p in reference_phases],'costs':reference_costs}
            references.append(ref)
            atomic_write_json(store.run_dir/f'artifacts/reference-{batch_index}.json',ref)
            store.append_metric({'stage':'reference_complete',**ref})
            for steps in config.iterations:
                (bp,bp_phase),bp_cost=measured(lambda:bptt_gradient(model,x,target,mask,steps,deadline=deadline))
                (ep,ep_phases),ep_cost=measured(lambda:finite_ep(model,x,target,mask,steps,beta=config.beta,deadline=deadline))
                row={'batch':batch_index,'iterations':steps,'reference_valid':valid,
                     'ep_vs_bptt':agreement(ep,bp),'ep_vs_reference':agreement(ep,reference_gradients[-1]),
                     'bptt_vs_reference':agreement(bp,reference_gradients[-1]),
                     'bptt_phase':phase_record(bp_phase),'ep_phases':[phase_record(p) for p in ep_phases],
                     'ep_cost':ep_cost,'bptt_cost':bp_cost,'serial_ep_steps':3*steps,
                     'free_terminal_difference':(bp_phase.y-ep_phases[0].y).abs().max().item()}
                rows.append(row);store.append_metric({'stage':'candidate',**row})
                for name,gradient in (('ep',ep),('bptt',bp)):
                    saved[f'{name}_{steps}']={key:value.cpu() for key,value in gradient.items()}
                atomic_write_json(store.run_dir/'artifacts/curve.json',rows)
                print(f"batch={batch_index} T={steps} EP/BPTT={row['ep_vs_bptt']['global']['cosine']:.6f} EP/reference={row['ep_vs_reference']['global']['cosine']:.6f}",flush=True)
            torch.save(saved,store.run_dir/f'artifacts/gradients-{batch_index}.pt')
        except (EquilibriumError,TimeoutError,torch.cuda.OutOfMemoryError) as error:
            failures.append({'batch':batch_index,'error':str(error),'type':type(error).__name__})
            atomic_write_json(store.run_dir/'artifacts/failures.json',failures)
            if isinstance(error,(TimeoutError,torch.cuda.OutOfMemoryError)):break
    if any(not torch.equal(p,original[name]) for name,p in model.named_parameters()):
        raise RuntimeError('Diagnostic mutated checkpoint weights')
    passing=[cap for cap in config.iterations if select_cap(rows,(cap,),config.batches,config.threshold) is not None]
    result={'case':config.case,'checkpoint_step':payload['step'],'candidate_cap':128,
            'selected_cap':min(passing) if passing else None,'passing_caps':passing,
            'coverage_complete':len(rows)==len(config.iterations)*config.batches and not failures,
            'reference_valid':len(references)==config.batches and all(r['valid'] for r in references),
            'selection_rule':'Smallest tested cap whose global EP/BPTT, EP/reference and BPTT/reference cosines are each >=0.9 on every batch; reference must pass 512/1024 stability and residual checks.',
            'references':references,'failures':failures,'rows':rows,
            'claim_boundary':'Read-only FP32 diagnostic. Each candidate is T free plus T positive and T negative iterations (3T serial). Reference uses longer unrolls only for validation. Truncated EP is not exact finite-time backpropagation or an asserted equilibrium gradient.'}
    atomic_write_json(store.run_dir/'artifacts/summary.json',result)
    if result['selected_cap'] is not None:
        training={**payload['config'],'phase_mode':'fixed_steps','max_steps':result['selected_cap']}
        atomic_write_json(store.run_dir/'artifacts/selected-training-config.json',training)
    return result


def run_validate(request):
    if request.teacher_weights is not None or request.device_model is not None:
        raise ConfigError('Iteration diagnostic does not use a teacher or device-model input')
    strict_fp32()
    if not torch.cuda.is_available():raise RuntimeError('Iteration diagnostic requires CUDA')
    # Keep a context active through data loading and runtime provenance collection.
    guard=torch.empty(1,device='cuda')
    config=request.spec;weights=Path(request.weights).resolve()
    payload=torch.load(weights,map_location='cpu',weights_only=False)
    if payload.get('schema')!='energy-transformer-mnist.checkpoint.v1' or payload.get('step')!=250:
        raise ConfigError('Use the explicit checkpoint after 250 trained updates, not initialization')
    if payload['config']['case']!=config.case or payload['config']['arm']!='energy_ep':
        raise ConfigError('Checkpoint case/architecture mismatch')
    cohort_path=weights.parent.parent/'artifacts/cohorts.pt'
    cohort=torch.load(cohort_path,map_location='cpu',weights_only=False)
    images,_,paths=load_raw(config.dataset_root)
    inputs=[{'path':str(p),'sha256':sha256_file(p)} for p in [weights,cohort_path,*paths]]
    original_inputs={r['path']:r['sha256'] for r in __import__('json').loads((weights.parent.parent/'manifest.json').read_text())['inputs']}
    for item in inputs[2:]:
        if original_inputs.get(item['path'])!=item['sha256']:
            raise ConfigError('MNIST identity differs from the training checkpoint')
    store=RunStore.create(output_root=request.output_dir,experiment_id=config.experiment_id,
        resolved_config=asdict(config),command=request.command,repo_root=ROOT,input_artifacts=inputs,resume_capability='unsupported')
    print(f'RUN_DIR={store.run_dir}',flush=True)
    try:
        atomic_write_json(store.run_dir/'artifacts/environment.json',{'gpu':torch.cuda.get_device_name(),
                          'torch':torch.__version__,'precision':'fp32','amp':False,'tf32':False})
        result=evaluate(config,store,payload,cohort,images,time.monotonic()+config.wall_seconds)
        artifacts=[store.artifact_record(p,kind='iteration_gradient') for p in (store.run_dir/'artifacts').rglob('*') if p.is_file()]
        store.complete(metrics={k:result[k] for k in ('selected_cap','coverage_complete','reference_valid')},artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error);raise

"""Public deployment and matched recovery on saved, held physical DRN states."""

from dataclasses import asdict
import math
import time
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from ebl.runtime.store import atomic_write_json, content_hash, sha256_file
from networks.opt.data import cohorts, blocks, load_corpus, TRANSFER_TESTS
from methods.programming.cmo import program_rewritten
from networks.opt.physical import CmoDrnBank
from ebl.schema.opt_cmo_drn import compatible_config_hashes, resume_config_hashes
from methods.cmo_thresholded_adam import ThresholdedWriteAdam
from methods.cmo_direct_adam import CappedNominalWriteAdam, DirectWriteAdam, ThresholdedBoundedWriteAdam
from hardware.devices.cmo import G_MIN_US, G_MAX_US, HALF_RANGE_US, MID_US, programming_sigma
from methods.core.engine import ExperimentComponents, GradientsReadyEvent, train_epoch
from methods.core.guards import FiniteGradientGuard


from methods.opt_recovery import RecoveryWritePort


from methods.opt_recovery import RecoveryStep


@torch.no_grad()
def restore_initial_programming(bank):
    """Reinstate each array's held state from before partner compensation (naive P&V).

    The deployment keeps that state as its raw programming evidence, including the
    permanent faults; no device write is counted and the fault check still applies.
    """
    for array in bank.arrays:
        raw=array.evidence['raw'].to(array.held.device)
        if raw.shape!=array.held.shape or raw.dtype!=array.held.dtype:
            raise ValueError('Raw programming evidence differs from the held array.')
        array.held=raw.clone()
        array.check()


def _calibration_state(model):
    selected={id(p) for p in model.parameter_groups()[1]}
    return {name:p.detach().cpu().clone() for name,p in model.named_parameters() if id(p) in selected}


@torch.no_grad()
def _restore_calibration(model,state):
    for name,p in model.named_parameters():
        if name in state:p.copy_(state[name].to(p))


def run_physical(request):
    # Runtime import here keeps dispatch and the experiment registration acyclic.
    import ebl.runtime.opt_cmo_drn.runtime as runtime
    c=request.spec
    if request.teacher_weights is None:
        raise ValueError('Physical lifecycle requires the explicit frozen teacher.')
    if any(getattr(request,k,None) is not None for k in ('base_weights','device_data','device_model','deployment')):
        raise ValueError('Unexpected legacy input for the CMO physical lifecycle.')
    resumed=None
    if request.resume is not None:
        if c.stage!='recover' or request.weights is not None or getattr(request,'device_state',None) is not None:
            raise ValueError('Physical resume takes only the full recovery checkpoint and teacher.')
        resumed=torch.load(request.resume,map_location='cpu',weights_only=True)
        if resumed.get('schema')!='opt_cmo_drn.recovery_resume.v1':
            raise ValueError('Expected an unfinished full physical recovery checkpoint.')
        if resumed['config_sha256'] not in resume_config_hashes(c) or resumed['torch_version']!=str(torch.__version__):
            raise ValueError('Recovery resume requires identical configuration and PyTorch environment.')
        source=resumed['weights']
        source_sha=resumed['source_weights_sha256']
    else:
        if c.recovery.extends_updates:
            raise ValueError('A horizon extension only continues an unfinished checkpoint.')
        if request.weights is None:raise ValueError('Expected explicit HWA source weights.')
        source=torch.load(request.weights,map_location='cpu',weights_only=True)
        source_sha=sha256_file(request.weights)
    runtime.configure_runtime(c)
    teacher=runtime.teacher_model(request.teacher_weights,c)
    model=runtime.from_mapped(teacher,c,source).to(c.device)
    teacher=teacher.to(c.device)
    train,dev,cohort=cohorts(c)
    store=runtime._store(request)
    started=time.monotonic()
    try:
        if c.stage=='deploy':
            if getattr(request,'device_state',None) is not None:
                raise ValueError('Deployment creates a physical state; it does not accept one.')
            d=c.deployment
            view,receipt,state=program_rewritten(model,d.assignment_seed,d.fault_kind,d.fault_rate,capture_state=True)
            del view
            bank=CmoDrnBank(state,c.device)
            del state
            raw=runtime.evaluate(model,teacher,dev,c.device,bank.view(model,raw=True),c.training.microbatch)
            canonical=runtime.evaluate(model,teacher,dev,c.device,bank.view(model),c.training.microbatch)
            saved={'schema':'opt_cmo_drn.deployment.v1','source_weights_sha256':source_sha,
                   'source_contract':source['contract'],'deployment':asdict(d),'cohorts':cohort,
                   'bank':bank.state_dict(),'raw_development':raw,'development':canonical,
                   'start_held_hashes':bank.held_hashes(),'programming':receipt}
            path=store.run_dir/'checkpoints/deployed.pt'
            runtime.save(path,saved)
            store.append_metric({'event':'deployment','raw_development':raw,'development':canonical,
                                 'costs':bank.costs(),'start_held_hashes':saved['start_held_hashes']})
            store.complete(metrics={'raw_development':raw,'development':canonical,'programming':receipt,
                                    'costs':bank.costs(),'start_held_hashes':saved['start_held_hashes']},
                           artifacts=[store.artifact_record(path,kind='device_state')])
            return 0

        r=c.recovery
        if resumed is not None:
            physical=resumed
            initial_sha=physical['initial_device_state_sha256']
        else:
            if getattr(request,'device_state',None) is None:
                raise ValueError('Matched recovery requires an explicit saved --device-state.')
            physical=torch.load(request.device_state,map_location='cpu',weights_only=True)
            if r.arm=='learning_calibration':
                if physical.get('schema')!='opt_cmo_drn.recovery.v1' or physical.get('arm')!='learning':
                    raise ValueError('Learning-plus-calibration must consume the completed learning trajectory.')
                initial_sha=physical['initial_device_state_sha256']
                model.load_state_dict(physical['weights']['model'],strict=True)
            else:
                if physical.get('schema')!='opt_cmo_drn.deployment.v1':
                    raise ValueError('Recovery controls must restore the canonical deployment.')
                initial_sha=sha256_file(request.device_state)
        if physical['source_weights_sha256']!=source_sha or physical['cohorts']!=cohort:
            raise ValueError('Recovery source weights or token cohorts differ from the physical input.')
        if physical['deployment']!=asdict(c.deployment):
            raise ValueError('Recovery deployment assignment/fault case differs.')
        bank=CmoDrnBank(physical['bank'],c.device)
        initial_hashes=physical['start_held_hashes']
        if physical.get('schema')=='opt_cmo_drn.deployment.v1' and bank.held_hashes()!=initial_hashes:
            raise ValueError('Canonical deployment failed held-state identity check.')
        if r.arm=='naive_hold':
            if resumed is not None or physical.get('schema')!='opt_cmo_drn.deployment.v1':
                raise ValueError('The naive hold reads only a canonical deployment.')
            restore_initial_programming(bank)
        step=resumed['step'] if resumed else 0
        core=None if r.arm in ('hold','naive_hold','rewrite') else RecoveryStep(model,teacher,bank,c)
        noise=torch.Generator(device=c.device).manual_seed(c.training.seed)
        best=resumed['calibration_best'] if resumed else None
        if resumed:
            if core is not None:core.load_state_dict(resumed['engine'])
            runtime.restore_rng(resumed['rng'],noise)
        del physical,resumed
        total=0 if r.arm in ('hold','naive_hold') else r.calibration_updates if r.arm in ('calibration','learning_calibration') else r.updates
        if step and step>=total:raise ValueError('Recovery checkpoint already reached its declared budget.')

        def evaluate():
            before=time.monotonic()
            metrics=runtime.evaluate(model,teacher,dev,c.device,bank.view(model),c.training.microbatch)
            if r.target_bounds=='gmin_gmax':
                store.append_metric({'event':'recovery_evaluation_timing','step':step,
                                     'seconds':time.monotonic()-before})
            return metrics

        def physical_diagnostics():
            return {'conductances':[a.statistics() for a in bank.arrays]} if r.target_bounds=='gmin_gmax' else {}

        if best is None and core is not None and core.calibration:
            baseline=evaluate()
            best={'teacher_kl':baseline['teacher_kl'],'step':0,'parameters':_calibration_state(model)}
        if core is not None and not core.calibration and r.development_every:
            store.append_metric({'event':'recovery_development','step':step,
                                 'reason':'start' if step==0 else 'resume','development':evaluate(),
                                 'costs':bank.costs(),**physical_diagnostics()})

        def payload(complete=False,development=None):
            return {'schema':'opt_cmo_drn.recovery.v1' if complete else 'opt_cmo_drn.recovery_resume.v1',
                    'arm':r.arm,'step':step,'config_sha256':content_hash(asdict(c)),
                    'torch_version':str(torch.__version__),'weights':runtime.weights_payload(model,c),
                    'source_weights_sha256':source_sha,'initial_device_state_sha256':initial_sha,
                    'start_held_hashes':initial_hashes,'deployment':asdict(c.deployment),'cohorts':cohort,
                    'bank':bank.state_dict(),
                    'engine':None if core is None or (complete and not r.retain_final_optimizer) else core.state_dict(),
                    'write_noise_factor':r.write_noise_factor,
                    'conductance_floor':r.conductance_floor,
                    'update_rule':r.update_rule,'target_bounds':r.target_bounds,
                    'writer_diagnostics':[] if core is None else [w.diagnostics() for w in core.writers],
                    'rng':runtime.rng_state(noise),'calibration_best':best,'development':development,
                    'costs':bank.costs(),'primary_state':'held_apparent'}

        latest=store.run_dir/'checkpoints/latest.pt'
        def save_checkpoint(path,*,complete=False,development=None):
            before=time.monotonic()
            runtime.save(path,payload(complete,development))
            store.append_metric({'event':'recovery_checkpoint','arm':r.arm,'step':step,
                                 'checkpoint_seconds':time.monotonic()-before,'size_bytes':path.stat().st_size})

        guard=None if core is None else FiniteGradientGuard(core.names)
        store.append_metric({'event':'recovery_start','arm':r.arm,'step':step,
                             'initial_device_state_sha256':initial_sha,'costs':bank.costs()})
        batches=math.ceil(len(train)/c.training.effective_batch)
        order=None
        previous_epoch=None
        diagnostic_artifacts=[]
        while step<total:
            if time.monotonic()-started>=c.training.max_wall_seconds:
                save_checkpoint(latest)
                raise TimeoutError('Recovery wall-time budget reached; complete continuation state preserved.')
            before=time.monotonic()
            if r.arm=='rewrite':
                for array in bank.arrays:array.rewrite(conductance_floor=r.conductance_floor)
                bank.view(model)
                extra={'policy':'uncapped_frozen_source_targets'}
            else:
                if core.calibration:
                    batch=dev
                else:
                    epoch,cursor=divmod(step,batches)
                    if previous_epoch!=epoch:
                        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(c.data.seed+epoch))
                        previous_epoch=epoch
                    batch=train[order[cursor*c.training.effective_batch:(cursor+1)*c.training.effective_batch]]
                train_epoch(core.components(),[(batch,batch)],start_global_step=step,event_handlers=(guard,core.observe))
                extra=core.last
            step+=1
            training_step_seconds=time.monotonic()-before
            if core is not None and core.calibration:
                scored=evaluate()
                if scored['teacher_kl']<best['teacher_kl']:
                    best={'teacher_kl':scored['teacher_kl'],'step':step,'parameters':_calibration_state(model)}
                extra={**extra,'development_kl':scored['teacher_kl']}
            if core is not None and not core.calibration and r.development_every:
                first_write=core.first_write_record is not None
                if first_write:
                    path=store.run_dir/'artifacts/first-write.pt'
                    runtime.save(path,{'schema':'opt_cmo_drn.first_write.v1','step':step,
                        'initial_device_state_sha256':initial_sha,'source_weights_sha256':source_sha,
                        'write_noise_factor':r.write_noise_factor,'cells':core.first_write_record,
                        'conductance_floor':r.conductance_floor,
                        'update_rule':r.update_rule,'target_bounds':r.target_bounds,
                        'threshold_q':[getattr(w,'threshold_q',None) for w in core.writers]})
                    diagnostic_artifacts.append(store.artifact_record(path,kind='first_write_commands'))
                    core.first_write_record=None
                if first_write or step==1 or step%r.development_every==0:
                    scored=evaluate()
                    store.append_metric({'event':'recovery_development','step':step,
                        'reason':'first_write' if first_write else 'first_update' if step==1 else 'interval',
                        'development':scored,'costs':bank.costs(),
                        'writers':[w.diagnostics() for w in core.writers],**physical_diagnostics()})
            if str(c.device).startswith('cuda'):torch.cuda.synchronize()
            store.append_metric({'event':'recovery_update','arm':r.arm,'step':step,'update_seconds':time.monotonic()-before,
                                 'training_step_seconds':training_step_seconds,
                                 'peak_memory_bytes':torch.cuda.max_memory_allocated() if str(c.device).startswith('cuda') else 0,
                                 'elapsed_seconds':time.monotonic()-started,'costs':bank.costs(),**extra})
            if step%r.checkpoint_every==0 and step<total:save_checkpoint(latest)
        if best is not None:_restore_calibration(model,best['parameters'])
        final_metrics=evaluate()
        if core is not None and core.writers:
            attempts=sum(int(w.attempt_counts.sum()) for w in core.writers)
            if attempts!=bank.costs()['write_attempts']:
                raise RuntimeError('Physical write-attempt accounting differs from controller history.')
        final=store.run_dir/'checkpoints/final.pt'
        save_checkpoint(final,complete=True,development=final_metrics)
        artifacts=[store.artifact_record(final,kind='recovery_state'),*diagnostic_artifacts]
        if latest.exists():
            removed={'path':str(latest.relative_to(store.run_dir)),'sha256':sha256_file(latest),'size_bytes':latest.stat().st_size}
            latest.unlink()
            audit=store.run_dir/'artifacts/retention.json'
            atomic_write_json(audit,{'policy':'completed_recovery_final','removed':[removed]})
            artifacts.append(store.artifact_record(audit,kind='retention_receipt'))
        store.complete(metrics={'arm':r.arm,'updates':step,'development':final_metrics,'costs':bank.costs(),
                                'initial_device_state_sha256':initial_sha,'start_held_hashes':initial_hashes,
                                'final_held_hashes':bank.held_hashes(),
                                'calibration_selected_step':None if best is None else best['step'],
                                **physical_diagnostics()},artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


def validate_physical(request):
    import ebl.runtime.opt_cmo_drn.runtime as runtime
    c=request.spec
    if request.teacher_weights is None:raise ValueError('Expected the explicit frozen teacher.')
    saved=torch.load(request.weights,map_location='cpu',weights_only=True)
    schema=saved.get('schema')
    if schema not in ('opt_cmo_drn.recovery.v1','opt_cmo_drn.recovery_resume.v1'):
        raise ValueError('Expected a completed or resumable physical recovery state.')
    complete=schema=='opt_cmo_drn.recovery.v1'
    if not complete:
        if saved['config_sha256'] not in compatible_config_hashes(c):
            raise ValueError('Recovery endpoint validation requires the unchanged training config.')
        if saved['torch_version']!=str(torch.__version__):
            raise ValueError('Recovery endpoint validation requires the saved PyTorch version.')
        total=c.recovery.calibration_updates if c.recovery.arm in ('calibration','learning_calibration') else c.recovery.updates
        if type(saved['step']) is not int or not 0<saved['step']<=total:
            raise ValueError('Invalid recovery endpoint update count.')
    if saved['arm']!=c.recovery.arm:raise ValueError('Recovery validation arm differs from the saved state.')
    if saved.get('write_noise_factor',1.)!=c.recovery.write_noise_factor:
        raise ValueError('Recovery validation write-noise intervention differs from the saved state.')
    if saved.get('conductance_floor','none')!=c.recovery.conductance_floor:
        raise ValueError('Recovery validation conductance floor differs from the saved state.')
    for key,default in (('update_rule','thresholded_adam'),('target_bounds','none')):
        if saved.get(key,default)!=getattr(c.recovery,key):
            raise ValueError(f'Recovery validation {key} differs from the saved state.')
    runtime.configure_runtime(c)
    teacher=runtime.teacher_model(request.teacher_weights,c)
    model=runtime.from_mapped(teacher,c,saved['weights']).to(c.device)
    teacher=teacher.to(c.device)
    bank=CmoDrnBank(saved['bank'],c.device)
    _,dev,cohort=cohorts(c)
    if cohort!=saved['cohorts'] or saved['deployment']!=asdict(c.deployment):
        raise ValueError('Recovery validation requires identical cohorts and physical case.')
    store=runtime._store(request)
    try:
        held_hashes=bank.held_hashes()
        costs=bank.costs()
        noise_states=[a.generator.get_state().clone() for a in bank.arrays]
        view=bank.view(model)
        replay=runtime.evaluate(model,teacher,dev,c.device,view,c.training.microbatch)
        if complete and replay!=saved['development']:
            raise RuntimeError('Saved physical recovery failed exact development-metric replay.')
        results={'development_replay' if complete else 'development':replay,
                 'checkpoint':{'schema':schema,'updates':saved['step'],'training_complete':complete,
                               'development_replay_verified':complete,'primary_state':'held_apparent',
                               'initial_device_state_sha256':saved['initial_device_state_sha256'],
                               'start_held_hashes':saved['start_held_hashes'],'final_held_hashes':held_hashes,
                               'costs':costs}}
        for label,name,digest,count in TRANSFER_TESTS:
            tokens=blocks(load_corpus(name,digest)['test'],c.data.sequence_length,count)
            results[label]=runtime.evaluate(model,teacher,tokens,c.device,view,c.training.microbatch)
        if bank.held_hashes()!=held_hashes or bank.costs()!=costs or any(
                not torch.equal(before,a.generator.get_state()) for before,a in zip(noise_states,bank.arrays)):
            raise RuntimeError('Physical endpoint validation mutated the held state, write counters or device RNG.')
        path=store.run_dir/'artifacts/held_out.json'
        atomic_write_json(path,results)
        store.append_metric({'event':'held_out_evaluation','arm':saved['arm'],'results':results})
        store.complete(metrics=results,artifacts=[store.artifact_record(path,kind='held_out_evaluation')])
        return 0
    except BaseException as error:
        store.fail(error)
        raise

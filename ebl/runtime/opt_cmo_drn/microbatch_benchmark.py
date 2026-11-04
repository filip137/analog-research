"""Bounded microbatch comparison through the public ebl run surface.

Read-only gradients use two identical effective batches at the canonical held
deployment. Timing trajectories restart that deployment and its write RNG for
each arm/microbatch; they are diagnostics, never LR-selection candidates.
"""

import gc
import math
import statistics
import time
from dataclasses import asdict, replace

import torch

from ebl.runtime.store import atomic_write_json, sha256_file
from networks.opt.data import cohorts
from networks.opt.physical import CmoDrnBank
from ebl.runtime.opt_cmo_drn.recovery import RecoveryStep
from methods.core.engine import train_epoch
from methods.core.guards import FiniteGradientGuard

MICROBATCHES=(1,4,8,16)


def gradient_agreement(reference, candidate):
    """Accumulate in float64, one tensor at a time, without large GPU copies."""
    if len(reference)!=len(candidate):raise ValueError('Gradient catalogs differ.')
    totals=[0.,0.,0.,0.]
    rows=[]
    for a,b in zip(reference,candidate):
        a=a.detach().cpu().double();b=b.detach().cpu().double()
        if a.shape!=b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError('Invalid gradient comparison.')
        aa=float(a.square().sum());bb=float(b.square().sum())
        dd=float((a-b).square().sum());ab=float((a*b).sum())
        rel=math.sqrt(dd/max(aa,1e-30))
        cosine=ab/math.sqrt(aa*bb) if aa and bb else float(aa==bb)
        rows.append({'relative_l2':rel,'cosine':cosine,'max_absolute':float((a-b).abs().max())})
        for i,value in enumerate((aa,bb,dd,ab)):totals[i]+=value
    aa,bb,dd,ab=totals
    return {'relative_l2':math.sqrt(dd/max(aa,1e-30)),
            'cosine':ab/math.sqrt(aa*bb) if aa and bb else float(aa==bb),
            'per_tensor':rows,
            'accepted':all(r['relative_l2']<=1e-3 and r['cosine']>=.999999 for r in rows)}


def select_microbatch(parity,timings):
    """Require both batches, both arms, memory headroom and >=10% speedup."""
    by={(r['arm'],r['microbatch']):r for r in timings}
    eligible=[]
    for mb in MICROBATCHES:
        checks=[r for r in parity if r['microbatch']==mb]
        if len(checks)!=2 or not all(r['accepted'] for r in checks):continue
        if not all((arm,mb) in by for arm in ('noiseless','noisy')):continue
        if any(by[arm,mb]['peak_memory_bytes']>28*1024**3 for arm in ('noiseless','noisy')):continue
        ratios=[by[arm,mb]['median_step_seconds']/by[arm,1]['median_step_seconds'] for arm in ('noiseless','noisy')]
        if mb==1 or max(ratios)<=.9:eligible.append((sum(ratios),mb))
    return min(eligible)[1] if eligible else 1


def run_benchmark(request):
    import ebl.runtime.opt_cmo_drn.runtime as runtime
    c=request.spec
    if any(getattr(request,k,None) is not None for k in ('resume','base_weights','device_data','device_model','deployment')):
        raise ValueError('Benchmark accepts explicit teacher, HWA weights and canonical deployment only.')
    if any(getattr(request,k,None) is None for k in ('teacher_weights','weights','device_state')):
        raise ValueError('Benchmark requires teacher, HWA weights and canonical deployment.')
    runtime.configure_runtime(c)
    source=torch.load(request.weights,map_location='cpu',weights_only=True)
    physical=torch.load(request.device_state,map_location='cpu',weights_only=True)
    teacher=runtime.teacher_model(request.teacher_weights,c)
    model=runtime.from_mapped(teacher,c,source).to(c.device);teacher=teacher.to(c.device)
    train,_,cohort=cohorts(c)
    if (physical.get('schema')!='opt_cmo_drn.deployment.v1'
            or physical['source_weights_sha256']!=sha256_file(request.weights)
            or physical['cohorts']!=cohort or physical['deployment']!=asdict(c.deployment)):
        raise ValueError('Benchmark inputs differ from the canonical deployment.')
    order=torch.randperm(len(train),generator=torch.Generator().manual_seed(c.data.seed))
    batches=[train[order[i*32:(i+1)*32]] for i in range(8)]
    store=runtime._store(request);started=time.monotonic()
    parity=[];timings=[]

    def sync():
        if str(c.device).startswith('cuda'):torch.cuda.synchronize()

    def deadline():
        if time.monotonic()-started>c.training.max_wall_seconds:
            raise TimeoutError('Microbatch benchmark budget exhausted.')

    def setup(mb,arm):
        cfg=replace(c,training=replace(c.training,microbatch=mb),recovery=replace(c.recovery,
            write_noise_factor=float(arm=='noisy'),development_every=0,
            update_rule='bounded_thresholded_adam' if arm=='noisy' else 'direct_adam'))
        bank=CmoDrnBank(physical['bank'],c.device)
        if bank.held_hashes()!=physical['start_held_hashes']:raise ValueError('Held deployment differs.')
        return RecoveryStep(model,teacher,bank,cfg)

    try:
        # No calls to step(): every gradient uses precisely the same held state.
        core=setup(1,'noiseless')
        for batch_index,batch in enumerate(batches[:2]):
            reference=None;reference_loss=None
            for mb in MICROBATCHES:
                deadline()
                core.config=replace(core.config,training=replace(core.config.training,microbatch=mb))
                core.set_input(batch)
                gradients=core.compute_gradient();sync()
                if mb==1:
                    reference=tuple(g.detach().cpu().clone() for g in gradients)
                    reference_loss=core.last['teacher_kl']
                agreement=gradient_agreement(reference,gradients)
                difference=abs(core.last['teacher_kl']-reference_loss)
                row={'event':'microbatch_parity','batch':batch_index,'microbatch':mb,
                     'loss_absolute_difference':difference,**agreement}
                row['accepted']=agreement['accepted'] and difference<=1e-6+1e-4*abs(reference_loss)
                parity.append(row);store.append_metric(row)
            del reference,gradients
        del core
        model.zero_grad(set_to_none=True);gc.collect()
        for arm in ('noiseless','noisy'):
            for mb in MICROBATCHES:
                deadline();core=setup(mb,arm)
                if str(c.device).startswith('cuda'):
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                guard=FiniteGradientGuard(core.names);seconds=[]
                for step,batch in enumerate(batches):
                    deadline();sync();before=time.monotonic()
                    train_epoch(core.components(),[(batch,batch)],start_global_step=step,
                                event_handlers=(guard,core.observe))
                    sync();elapsed=time.monotonic()-before;seconds.append(elapsed)
                    store.append_metric({'event':'microbatch_timing_step','arm':arm,'microbatch':mb,
                        'step':step+1,'seconds':elapsed,'teacher_kl':core.last['teacher_kl']})
                row={'event':'microbatch_timing','arm':arm,'microbatch':mb,
                     'warmup_steps':2,'measured_steps':6,'step_seconds':seconds,
                     'median_step_seconds':statistics.median(seconds[2:]),
                     'peak_memory_bytes':torch.cuda.max_memory_allocated() if str(c.device).startswith('cuda') else 0,
                     'costs':core.bank.costs()}
                timings.append(row);store.append_metric(row)
                del core
                model.zero_grad(set_to_none=True);gc.collect()
        result={'parity':parity,'timings':timings,'selected_microbatch':select_microbatch(parity,timings),
                'effective_batch':32,'sequence_length':c.data.sequence_length,
                'primary_state':'held_apparent','initial_device_state_sha256':sha256_file(request.device_state),
                'decision_rule':'Two-batch per-tensor relative L2 <=1e-3, cosine >=.999999; loss within 1e-6+1e-4*reference; peak <=28 GiB; >=10% median speedup in both arms.'}
        path=store.run_dir/'artifacts/microbatch-benchmark.json';atomic_write_json(path,result)
        store.complete(metrics=result,artifacts=[store.artifact_record(path,kind='microbatch_benchmark')])
        return 0
    except BaseException as error:
        store.fail(error);raise

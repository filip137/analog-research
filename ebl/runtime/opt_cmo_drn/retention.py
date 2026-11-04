"""Native read-only scalar drift on frozen, four-device OPT DRN MLPs."""
from dataclasses import asdict
import json
import time
import torch

from ebl.runtime.store import atomic_write_json
import ebl.runtime.opt_cmo_drn.runtime as runtime
from networks.opt.data import pinned_load, load_corpus, blocks
from methods.programming.cmo import NAMESPACE
from hardware.faults import binding_seed, state_hash, tensor_hash
import hardware.devices.cmo as cmo
from hardware.drift.power_law import snapshot, aggregate_statistics
from methods.evaluation import reference_cache, evaluate


@torch.no_grad()
def program(model, assignment, drift_seed):
    """One ordinary CMO draw, with no verify scan or partner rewrite."""
    states, raw_view, clipped = [], [], 0
    for k, block in enumerate(model.blocks):
        for name, u in zip(('fc1','fc2'),(block.u1,block.u2)):
            name=f'decoder.layers.{model.start+k}.{name}'
            target=(cmo.MID_US+cmo.HALF_RANGE_US*u.T).contiguous()
            pseed=binding_seed(assignment+10000,NAMESPACE,name)
            dseed=binding_seed(drift_seed,NAMESPACE,f'{assignment}/{name}')
            rng=torch.Generator(device=u.device).manual_seed(pseed)
            raw=target+cmo.programming_sigma(target,2.)*torch.randn(target.shape,device=u.device,generator=rng)
            if not bool(torch.isfinite(raw).all() and (raw>=0).all()):
                raise FloatingPointError('Invalid raw DRN programming endpoint.')
            gp=raw.clamp(cmo.G_MIN_US,cmo.G_MAX_US)
            eps=torch.randn(gp.shape,device=u.device,generator=torch.Generator(device=u.device).manual_seed(dseed))
            clipped+=int((gp!=raw).sum())
            states.append({'name':name,'programmed':gp.T.contiguous().cpu(),'epsilon':eps.T.contiguous().cpu(),
                           'programming_seed':pseed,'drift_seed':dseed,'raw_sha256':tensor_hash(raw)})
            raw_view.append(raw.T.contiguous()/cmo.G_MAX_US)
    return states, pairs(raw_view), clipped


def pairs(values):
    return tuple(tuple(values[i:i+2]) for i in range(0,len(values),2))


def held_view(states, seconds, sigma, spec):
    values, stats = [], []
    for state in states:
        g, info=snapshot(state['programmed'],state['epsilon'],seconds,sigma,spec,cmo.G_MIN_US,cmo.G_MAX_US)
        values.append(g/cmo.G_MAX_US)
        stats.append(info)
    return pairs(values), aggregate_statistics(stats)


@torch.no_grad()
def run(request):
    c=request.spec
    if c.retention.calibration is not None:
        from ebl.runtime.opt_cmo_drn.calibration import run as calibrate
        return calibrate(request)
    if request.teacher_weights is None or request.weights is None:
        raise ValueError('Retention requires explicit teacher and frozen HWA weights.')
    if any(getattr(request,k,None) is not None for k in ('resume','device_state','base_weights','device_data','device_model','deployment')):
        raise ValueError('Unexpected input for a frozen-source retention run.')
    runtime.configure_runtime(c)
    settings, spec = c.retention, c.retention.power_law
    teacher=runtime.teacher_model(request.teacher_weights,c)
    source=pinned_load(request.weights,settings.hwa_sha256)
    model=runtime.from_mapped(teacher,c,source).to(c.device).eval().requires_grad_(False)
    del source
    teacher=teacher.to(c.device)
    frozen_hash=state_hash(model.state_dict())
    tokens=blocks(load_corpus(c.data.corpus,c.data.sha256)['validation'],
                  c.data.sequence_length,spec.development_sequences)
    cohort={'split':'validation','first_sequence':0,'sequences':len(tokens),'tokens_sha256':tensor_hash(tokens),
            'source_sha256':settings.hwa_sha256,'corpus':{'name':c.data.corpus,'sha256':c.data.sha256}}
    store=runtime._store(request)
    started=time.monotonic()
    try:
        def forward(batch,view=None): return model(batch.to(c.device),view)
        ref_dir=store.run_dir/'reference_logits'
        refs=reference_cache(tokens,forward,lambda batch:teacher(batch.to(c.device))[:,:-1],ref_dir,cohort)
        clean=evaluate(tokens,forward,refs,c.device)
        states, raw_view, clipped=program(model,c.deployment.assignment_seed,settings.seed)
        raw_metrics=evaluate(tokens,lambda batch:forward(batch,raw_view),refs,c.device)
        del raw_view
        path=store.run_dir/'checkpoints/power-law.pt'
        runtime.save(path,{'schema':'opt_cmo_drn.power_law.v1','source_sha256':settings.hwa_sha256,
                           'cohort':cohort,'settings':asdict(settings),'first_read_clipped_devices':clipped,
                           'series':states})
        del states
        saved=torch.load(path,map_location='cpu',weights_only=True)
        states=[{**s,'programmed':s['programmed'].to(c.device),'epsilon':s['epsilon'].to(c.device)}
                for s in saved['series']]
        del saved
        physical=sum(s['programmed'].numel() for s in states)
        rows=[]
        baseline=None
        grid=[(0.,0.)]+[(t,s) for t in settings.seconds for s in spec.sigmas_us]
        for seconds,sigma in grid:
            before=time.monotonic()
            view,statistics=held_view(states,seconds,sigma,spec)
            replay,_=held_view(states,seconds,sigma,spec)
            if any(not torch.equal(x,y) for a,b in zip(view,replay) for x,y in zip(a,b)):
                raise RuntimeError('Held DRN snapshot did not replay exactly.')
            if seconds==0 and any(not torch.equal(g,states[i]['programmed']/cmo.G_MAX_US)
                                 for i,g in enumerate(g for p in view for g in p)):
                raise RuntimeError('Zero-time drift changed the first-read state.')
            del replay
            metrics=evaluate(tokens,lambda batch:forward(batch,view),refs,c.device)
            if baseline is None: baseline=metrics
            row={'architecture':'drn','snapshot':'programmed' if seconds==0 else f't{seconds:g}s-s{sigma:g}uS',
                 'seconds':seconds,'spread_us':sigma,'mean_us':spec.mean_us,'mean_exponent':spec.mean_exponent,
                 'spread_exponent':spec.spread_exponent,'array_seed':c.deployment.assignment_seed,
                 'source_sha256':settings.hwa_sha256,'cohort':cohort,'metrics':metrics,'baseline':baseline,
                 'clean':clean,'raw_programmed':raw_metrics,'first_read_clipped_devices':clipped,
                 'perplexity_ratio':metrics['perplexity']/baseline['perplexity'],
                 'physical_devices':physical,'logical_weights':physical//4,'replay_exact':True,
                 'apparent_sha256':[tensor_hash(g) for pair in view for g in pair],
                 'drift_statistics':statistics,'numerical_diagnostics':model.diagnostics(tokens[:1].to(c.device),view),
                 'cost':{'retention_device_writes':0,'retention_verify_reads':0,'initial_device_programs':physical},
                 'elapsed_seconds':time.monotonic()-before}
            rows.append(row)
            store.append_metric({'event':'retention','snapshots_complete':len(rows),'snapshots_expected':len(grid),**row})
            atomic_write_json(store.run_dir/'retention_measurements.json',rows)
            print(json.dumps({'snapshot':row['snapshot'],'perplexity_ratio':row['perplexity_ratio']}),flush=True)
        if state_hash(model.state_dict())!=frozen_hash:
            raise RuntimeError('Drift evaluation changed the frozen DRN model.')
        artifacts=[store.artifact_record(path,kind='retention_state.power_law'),
                   store.artifact_record(store.run_dir/'retention_measurements.json',kind='retention_measurements')]
        artifacts += [store.artifact_record(p,kind=f'reference.{p.stem}') for p in sorted(ref_dir.glob('*.pt'))]
        store.complete(metrics={'source_sha256':settings.hwa_sha256,'source_unchanged':True,'cohort':cohort,
                       'snapshots':len(rows),'expected_snapshots':len(grid),'exact_replays':len(rows),
                       'additional_writes':0,'retention':asdict(settings),'clean':clean,'raw_programmed':raw_metrics,
                       'first_read_clipped_devices':clipped,'elapsed_seconds':time.monotonic()-started,
                       'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if str(c.device).startswith('cuda') else 0},
                       artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise

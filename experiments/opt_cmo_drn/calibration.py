"""Closed-form MLP-output calibration of held scalar-drift DRN snapshots.

Reuses a completed retention run: its saved first read, epsilon and frozen
reference logits are hash-checked and replayed, never redrawn. Every aged
snapshot and its uncompensated metrics must equal the parent measurement.
Conductances, solver settings and model parameters do not change.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch

from experiments.artifacts import atomic_write_json, content_hash, sha256_file
from experiments.opt_cmo_drn import runtime
from experiments.opt_cmo_drn.data import cohorts, pinned_load
from experiments.opt_cmo_drn.faults import state_hash, tensor_hash
from experiments.opt_cmo_drn.retention import held_view
from training import output_calibration as oc
from training.drift_evaluation import evaluate


def plain(value):
    return json.loads(json.dumps(value))


def parent_run(state_path, config):
    """Verify the completed retention parent; return its result and a hash-checked resolver."""
    root = Path(state_path).resolve().parent.parent
    result = json.loads((root/'result.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    document = json.loads((root/'config.resolved.json').read_text())
    if result.get('schema')!='ebl.run' or result.get('status')!='complete':
        raise ValueError('Expected a completed native retention parent.')
    if manifest['config']['sha256']!=content_hash(document):
        raise ValueError('Changed parent retention configuration.')
    current = plain(asdict(config))
    current['retention'].pop('calibration')
    document = {**document,'retention':dict(document['retention'])}
    if document['retention'].pop('calibration',None) is not None:
        raise ValueError('Expected an uncalibrated retention parent.')
    if (any(document.get(k)!=v for k,v in current.items() if k in document)
            or any(v is not None for k,v in current.items() if k not in document)):
        raise ValueError('Calibration must keep every parent retention setting.')
    artifacts = {a['kind']:a for a in result['artifacts']}

    def checked(kind):
        entry = artifacts[kind]
        target = (root/entry['path']).resolve()
        if not target.is_relative_to(root) or sha256_file(target)!=entry['sha256']:
            raise ValueError(f'Changed or external retention artifact: {kind}')
        return target

    if checked('retention_state.power_law')!=Path(state_path).resolve():
        raise ValueError('Expected the parent power-law state as --device-state.')
    return root, result, checked


def hashes(view):
    return [tensor_hash(g) for pair in view for g in pair]


@torch.no_grad()
def run(request):
    c = request.spec
    if request.teacher_weights is None or request.weights is None or getattr(request,'device_state',None) is None:
        raise ValueError('Calibration requires the teacher, frozen HWA weights and the parent --device-state.')
    if any(getattr(request,k,None) is not None for k in ('resume','base_weights','device_data','device_model','deployment')):
        raise ValueError('Unexpected input for held-snapshot calibration.')
    runtime.configure_runtime(c)
    settings, spec = c.retention, c.retention.power_law
    calibration = settings.calibration
    root, parent, checked = parent_run(request.device_state, c)
    original = json.loads(checked('retention_measurements').read_text())
    saved = torch.load(checked('retention_state.power_law'), map_location='cpu', weights_only=True)
    parent_settings = plain(saved['settings'])
    if (saved['schema']!='opt_cmo_drn.power_law.v1' or saved['source_sha256']!=settings.hwa_sha256
            or parent_settings.pop('calibration',None) is not None
            or parent_settings!=plain({k:v for k,v in asdict(settings).items() if k!='calibration'})):
        raise ValueError('Changed power-law replay identity.')
    teacher = runtime.teacher_model(request.teacher_weights,c)
    source = pinned_load(request.weights,settings.hwa_sha256)
    model = runtime.from_mapped(teacher,c,source).to(c.device).eval().requires_grad_(False)
    del source, teacher
    frozen_hash = state_hash(model.state_dict())
    cohort = saved['cohort']
    identity = torch.load(checked('reference.identity'), map_location='cpu', weights_only=True)
    tokens = identity['tokens']
    if identity['identity']!=cohort or tensor_hash(tokens)!=cohort['tokens_sha256']:
        raise ValueError("Expected the parent's evaluation cohort.")
    refs = [torch.load(checked(f'reference.{i:04d}'), map_location='cpu', weights_only=True) for i in range(len(tokens))]
    train, _, _ = cohorts(c)
    calibration_tokens = train[:calibration.calibration_sequences].clone()
    del train
    data = {'corpus':{'name':c.data.corpus,'sha256':c.data.sha256},'split':'train',
            'sequences':len(calibration_tokens),'tokens_sha256':tensor_hash(calibration_tokens),'data_seed':c.data.seed}
    states = [{**s,'programmed':s['programmed'].to(c.device),'epsilon':s['epsilon'].to(c.device)} for s in saved['series']]
    del saved
    index = {(r['seconds'],r['spread_us']):r for r in original}
    reference = index[0.,0.]
    baseline, _ = held_view(states,0.,0.,spec)
    if hashes(baseline)!=reference['apparent_sha256']:
        raise ValueError('First-read replay differs from the original measurement.')
    store = runtime._store(request)
    started = time.monotonic()
    try:
        def native(view):
            return lambda batch: model(batch.to(c.device),view)

        layers, channels = len(model.blocks), model.blocks[0].bias2.numel()
        first = tokens[:1].to(c.device)
        decomposed = model.head(oc.forward(model.analog_input(first),model.mlp_branch,layers,baseline))
        if not torch.equal(decomposed,native(baseline)(first)):
            raise RuntimeError('Decomposed MLP-output forward differs from the native forward.')
        prefix = [model.analog_input(b.to(c.device)) for b in calibration_tokens.split(1)]
        statistics = oc.propagate(prefix,model.mlp_branch,layers,baseline)
        reference_path = store.run_dir/'checkpoints/output-reference-statistics.pt'
        runtime.save(reference_path,{'schema':'drift.output_calibration.reference.v1','apparent_sha256':reference['apparent_sha256'],
                                     'calibration':data,'statistics':[s.to_dict() for s in statistics]})
        artifacts = [store.artifact_record(reference_path,kind='output_calibration_reference')]
        grid = [(t,s) for t in calibration.seconds for s in calibration.sigmas_us]
        expected = len(grid)*len(calibration.methods)
        rows = []
        for seconds,sigma in grid:
            before = time.monotonic()
            prior = index[seconds,sigma]
            view, _ = held_view(states,seconds,sigma,spec)
            apparent = hashes(view)
            if apparent!=prior['apparent_sha256']:
                raise ValueError('Aged replay differs from the original measurement.')
            if evaluate(tokens,native(view),refs,c.device)!=prior['metrics']:
                raise ValueError('Uncompensated control no longer exactly reproduces the original evaluation.')
            tag = f't{seconds:g}s-s{sigma:g}uS'
            for method in calibration.methods:
                fitted = oc.propagate(prefix,model.mlp_branch,layers,view,method=method,reference=statistics,
                                      bounds=calibration.gain_bounds)
                path = store.run_dir/f'checkpoints/{tag}-{method}.pt'
                runtime.save(path,{'schema':'drift.output_calibration.v1','source_sha256':settings.hwa_sha256,
                                   'apparent_sha256':apparent,'method':method,'calibration':data,
                                   'reference_apparent_sha256':reference['apparent_sha256'],
                                   'corrections':[f.state_dict() for f in fitted]})
                restored = [oc.OutputCorrection.from_state_dict(s,c.device)
                            for s in torch.load(path,map_location='cpu',weights_only=True)['corrections']]
                if any(not (torch.equal(a.gain,b.gain) and torch.equal(a.shift,b.shift)) for a,b in zip(fitted,restored)):
                    raise RuntimeError('Calibration parameters did not replay exactly.')
                metrics = evaluate(tokens,lambda batch:model.head(oc.forward(
                    model.analog_input(batch.to(c.device)),model.mlp_branch,layers,view,restored)),refs,c.device)
                row = {'architecture':'drn','method':method,'snapshot':prior['snapshot'],'seconds':seconds,
                       'spread_us':sigma,'mean_us':spec.mean_us,'mean_exponent':spec.mean_exponent,
                       'spread_exponent':spec.spread_exponent,'array_seed':c.deployment.assignment_seed,
                       'source_sha256':settings.hwa_sha256,'cohort':cohort,'calibration':data,
                       'clean':prior['clean'],'baseline':reference['metrics'],'uncompensated':prior['metrics'],
                       'compensated':metrics,'perplexity_ratio':metrics['perplexity']/reference['metrics']['perplexity'],
                       'uncompensated_perplexity_ratio':prior['perplexity_ratio'],
                       'physical_devices':prior['physical_devices'],'logical_weights':prior['logical_weights'],
                       'apparent_sha256':apparent,'replay_exact':True,'control_exact':True,'state_unchanged':True,
                       'drift_statistics':prior['drift_statistics'],
                       'cost':oc.costs(method,len(calibration_tokens),calibration_tokens.shape[1],layers,channels),
                       'elapsed_seconds':time.monotonic()-before}
                rows.append(row)
                artifacts.append(store.artifact_record(path,kind=f'calibration_state.{tag}.{method}'))
                store.append_metric({'event':'calibration','complete':len(rows),'expected':expected,**row})
            atomic_write_json(store.run_dir/'calibration_measurements.json',rows)
            print(json.dumps({'snapshot':tag,'uncompensated':prior['perplexity_ratio'],
                              **{r['method']:r['perplexity_ratio'] for r in rows[-len(calibration.methods):]}}),flush=True)
            del view
        if state_hash(model.state_dict())!=frozen_hash:
            raise RuntimeError('Calibration changed the frozen DRN model.')
        artifacts.append(store.artifact_record(store.run_dir/'calibration_measurements.json',kind='calibration_measurements'))
        store.complete(metrics={'source_sha256':settings.hwa_sha256,'source_unchanged':True,'cohort':cohort,
                                'calibration':data,'settings':plain(asdict(calibration)),'snapshots':len(rows),
                                'expected_snapshots':expected,'control_exact':True,'additional_writes':0,
                                'parent':{'run_dir':str(root),'result_sha256':sha256_file(root/'result.json'),
                                          'run_id':parent.get('run_id')},
                                'selection':'all predeclared methods; no selection',
                                'elapsed_seconds':time.monotonic()-started,
                                'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated() if str(c.device).startswith('cuda') else 0},
                       artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise

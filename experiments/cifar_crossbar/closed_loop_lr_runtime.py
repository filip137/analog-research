"""Native development-only controller tuning and frozen matched confirmation."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import time
import torch
from torch.nn import functional as F
from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.cifar_crossbar.closed_loop_lr_config import ClosedLoopLRSpec, LRS, DEV_SEEDS
from experiments.cifar_crossbar.closed_loop_lr_updates import UncappedClosedLoopAdam
from experiments.cifar_crossbar.runtime import load, source_model, evaluate
from experiments.cifar_crossbar.model import CrossbarSuffix, tensor_hash
from experiments.cifar_crossbar.devices import make_plant
from experiments.cifar_crossbar.fault_runtime import cache_context, limited_cache
from experiments.cifar_crossbar.full_epoch_runtime import array_identity, verify_fixed, restore_endpoint
from experiments.cifar_crossbar.sweep_runtime import configure_runtime
from experiments.cifar_crossbar.sweep_devices import make_array, combined_om, CompactCheckpoints, expand_checkpoint
from experiments.cifar_crossbar.hwa_fault_runtime import case_name
from experiments.cifar_crossbar.om_open_loop_runtime import save_compressed, verify_inherited_metrics

ROOT = Path(__file__).resolve().parents[2]


def selected_rates(spec, receipt, source_sha256):
    if spec.phase == 'development':
        if receipt is not None: raise ValueError('Development must not consume a selection receipt.')
        return LRS
    if receipt is None or receipt.get('schema') != 'cifar_om_closed_loop_lr.selection.v1':
        raise ValueError('Confirmation requires the frozen development selection.')
    if receipt['development_arrays'] != list(DEV_SEEDS) or receipt['criterion'] != 'mean_epoch5_development_teacher_kl':
        raise ValueError('Selection partition or criterion changed.')
    key = spec.dataset + '_' + str(4 if spec.suffix == 'last_two_blocks' else 8)
    item = receipt['groups'][key]
    if item['source_checkpoint_sha256'] != source_sha256 or item['selected_learning_rate'] not in LRS:
        raise ValueError('Selection/source mismatch.')
    return tuple(sorted({LRS[0], item['selected_learning_rate']}))


def trajectory(student, array, target, cache, spec, method, lr, store, directory, writer):
    learn = method == 'onchip_calibration'
    student.enable_calibration(True); student.q.requires_grad_(False)
    optimizer = torch.optim.Adam(student.calibration_parameters(), lr=spec.calibration_lr)
    controller = UncappedClosedLoopAdam(array.port(), target, learning_rate=lr, seed=spec.seed) if learn else None
    baseline, initial_state, prefix = array.cost(), array.state_dict(), student.prefix_hash()
    initial_hash = tensor_hash(array.read()); initial_target = controller.target.clone() if controller else target.clone()
    curve = []
    for epoch in range(1, spec.epochs + 1):
        started = time.monotonic()
        order = torch.randperm(len(cache['training']['labels']), generator=torch.Generator().manual_seed(spec.data_seed + epoch))
        student.train(); loss_sum = 0.
        for batch, begin in enumerate(range(0, len(order), spec.batch_size)):
            idx = order[begin:begin + spec.batch_size]
            data = cache['training']; x, teacher = data['features'][idx], data['teacher_logits'][idx]
            student.zero_grad(set_to_none=True); optimizer.zero_grad(set_to_none=True)
            q = array.read().detach().clone().requires_grad_(learn)
            loss = F.kl_div(student.forward_features(x, q).log_softmax(1), teacher.softmax(1), reduction='batchmean')
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite closed-loop teacher KL.')
            loss.backward(); optimizer.step()
            if controller: controller.step(q.grad)
            loss_sum += float(loss.detach()) * len(idx)
            if batch % 100 == 0:
                store.append_metric({'method': method, 'learning_rate': lr, 'epoch': epoch, 'batch': batch,
                                     'case': str(directory.relative_to(store.run_dir)), 'teacher_kl_train': float(loss.detach())})
        verify_fixed(array, spec, initial_state)
        if student.prefix_hash() != prefix: raise RuntimeError('Frozen digital prefix changed.')
        count = controller.count if controller else torch.zeros_like(target, dtype=torch.int64)
        updates = epoch * ((len(order) + spec.batch_size - 1) // spec.batch_size)
        if int(count.max()) > updates: raise RuntimeError('More than one pulse per update.')
        cost = {key: value - baseline[key] for key, value in array.cost().items()}
        cost.update(verify_reads=controller.verify_reads if controller else 0,
                    max_recovery_cell_pulses=int(count.max()), capped_cells=0,
                    pulse_fraction=float((count > 0).float().mean()), mean_pulses_per_cell=float(count.float().mean()),
                    cells_over_old640_cap=int((count > 640).sum()),
                    target_displacement_mean=float((controller.target - initial_target).abs().mean()) if controller else 0.)
        row = {'epoch': epoch, 'teacher_kl_train_mean': loss_sum / len(order), 'image_presentations': epoch * len(order),
               'cost': cost, 'development': evaluate(student, cache['development'], target.device, q=array.read()),
               'epoch_seconds': time.monotonic() - started}
        if spec.phase == 'confirmation':
            row['test'] = evaluate(student, cache['test'], target.device, q=array.read())
            row['persistent_test'] = evaluate(student, cache['test'], target.device, q=array.engine.persistent)
        writer(directory / f'epoch{epoch}.pt', {'p0_sha256': sha256_file(directory.parent / 'p0.pt'), 'epoch': epoch,
               'calibration': {name: p for name, p in student.named_parameters() if name != 'q' and p.requires_grad},
               'array': array.state_dict(), 'optimizer': optimizer.state_dict(),
               'writer': controller.state_dict() if controller else None})
        curve.append(row)
        store.append_metric({'terminal_epoch': True, 'method': method, 'learning_rate': lr,
                             'case': str(directory.relative_to(store.run_dir)), **row})
        print(json.dumps({'method': method, 'learning_rate': lr, 'epoch': epoch,
                          'development': row['development'], 'seconds': row['epoch_seconds']}), flush=True)
    return {'method': method, 'learning_rate': lr, 'curve': curve, 'initial_apparent_sha256': initial_hash}


def screen(student, cache, spec, populations, incoming, rates, store, context):
    rows, replays = [], []
    writer = CompactCheckpoints(store.run_dir, save_fn=save_compressed)
    sources = incoming['sources']
    for name in spec.sources:
        source = sources[name]; student.load_state_dict(source['model']); target = student.q.detach().clone()
        for kind, ppm in spec.fault_cases:
            label = case_name(kind, ppm); student.load_state_dict(source['model'])
            inherited = None
            if spec.phase == 'confirmation':
                inherited = incoming['p0'][name][label]
                population = combined_om(populations, student.layout, spec.dataset, spec.array_seed,
                                         spec.endpoint_seed, kind, ppm / 1e6)
                if population['fingerprint'] != inherited['population_fingerprint']: raise ValueError('Final physical identity changed.')
                array = make_plant(population, spec.endpoint_seed, target.device)
                array.engine.load_state_dict(inherited['engine'])
                if tensor_hash(array.read()) != inherited['apparent_sha256'] or tensor_hash(array.engine.persistent) != inherited['persistent_sha256']:
                    raise ValueError('Exact final P0 replay failed.')
            else:
                array = make_array(student, target, spec, kind, ppm, populations)
            if abs(array.nominal_step * .5 - .04745) > 1e-12: raise ValueError('OM nominal tolerance changed.')
            directory = store.run_dir / 'checkpoints' / name / label
            p0 = {'context': context, 'model': source['model'], 'array': array.state_dict(), 'target': target,
                  'identity': array_identity(array, spec), 'initial_programming_cap': 128,
                  'inherited_p0_sha256': inherited['p0_sha256'] if inherited else None}
            writer(directory / 'p0.pt', p0)
            initial = {'development': evaluate(student, cache['development'], target.device, q=array.read())}
            if spec.phase == 'confirmation':
                initial['test'] = evaluate(student, cache['test'], target.device, q=array.read())
                verify_inherited_metrics(initial['test'], inherited['initial_test'])
            methods = [('calibration', None)] if spec.phase == 'development' else []
            methods += [('onchip_calibration', rate) for rate in rates]
            for method, rate in methods:
                student.load_state_dict(source['model']); array.load_state_dict(p0['array'])
                destination = directory / (method if rate is None else 'lr_' + str(rate))
                result = trajectory(student, array, target, cache, spec, method, rate, store, destination, writer)
                checkpoint = expand_checkpoint(load(destination / 'epoch5.pt'), store.run_dir)
                restore_endpoint(student, array, p0, checkpoint)
                for split in ('development', 'test') if spec.phase == 'confirmation' else ('development',):
                    if evaluate(student, cache[split], target.device, q=array.read()) != result['curve'][-1][split]:
                        raise RuntimeError('Exact final-state replay failed: ' + split)
                replays.append([name, label, method, rate])
                rows.append({'source': name, 'case': label, 'array_seed': spec.array_seed, 'phase': spec.phase,
                             'p0_sha256': sha256_file(directory / 'p0.pt'), 'inherited_p0_sha256': p0['inherited_p0_sha256'],
                             'identity': p0['identity'], 'initial': initial, **result})
                atomic_write_json(store.run_dir / 'measurements.json', rows)
    expected = len(spec.sources) * len(spec.fault_cases) * (len(rates) + (spec.phase == 'development'))
    if len(rows) != expected: raise RuntimeError('Missing LR sweep cells.')
    atomic_write_json(store.run_dir / 'replay.json', {'exact_replays': len(replays), 'checks': replays})
    return {'measurements': rows, 'completed_controls': len(rows), 'exact_replays': len(replays),
            'source_fits': {name: sources[name]['fit'] for name in spec.sources}}


def run_train(request):
    spec = request.spec
    if not isinstance(spec, ClosedLoopLRSpec) or any(getattr(request, key, None) is None for key in
            ('teacher_weights', 'weights', 'device_model', 'device_data')):
        raise ValueError('Expected explicit teacher, frozen source/P0, populations, and feature cache.')
    if any(getattr(request, key, None) is not None for key in ('resume', 'base_weights', 'device_state')):
        raise ValueError('Unexpected initialization input.')
    paths = ('teacher_weights', 'weights', 'device_model', 'device_data', 'selection_receipt')
    inputs = [{'role': key, 'path': str(getattr(request, key).resolve()), 'sha256': sha256_file(getattr(request, key))}
              for key in paths if getattr(request, key, None) is not None]
    receipt_path = os.environ.get('EBL_SOURCE_RECEIPT')
    if receipt_path:
        receipt = json.loads(Path(receipt_path).read_text())
        if any(sha256_file(ROOT / name) != digest for name, digest in receipt['files'].items()):
            raise ValueError('Frozen numerical source changed.')
        inputs.append({'role': 'source_snapshot', 'path': receipt_path, 'sha256': sha256_file(Path(receipt_path))})
    store = RunStore.create(output_root=request.output_dir, experiment_id=spec.experiment_id, resolved_config=asdict(spec),
                            command=request.command, repo_root=ROOT, input_artifacts=inputs, resume_capability='unsupported')
    started = time.monotonic()
    try:
        runtime_math = configure_runtime(spec); device = torch.device(spec.device)
        teacher, digest = source_model(request.teacher_weights, spec, device)
        student = CrossbarSuffix(teacher, spec.suffix, spec.tile_size)
        expected = {**cache_context(spec, digest, student), 'protocol': 'cifar_crossbar_fault_sweep.v1', 'max_examples': 0}
        incoming, populations, cache = load(request.weights), load(request.device_model), load(request.device_data)
        source_context = incoming['source_context'] if spec.phase == 'confirmation' else incoming['context']
        if source_context != expected or cache['context'] != expected or populations['dataset'] != spec.dataset:
            raise ValueError('Frozen source/cache/population context changed.')
        source_sha = incoming['source_checkpoint_sha256'] if spec.phase == 'confirmation' else sha256_file(request.weights)
        selection = json.loads(request.selection_receipt.read_text()) if request.selection_receipt else None
        rates = selected_rates(spec, selection, source_sha)
        if spec.phase == 'confirmation' and (incoming.get('schema') != 'cifar_om_open_loop.inputs.v1' or incoming['array_seed'] != spec.array_seed):
            raise ValueError('Final P0 bundle mismatch.')
        if spec.phase == 'development': del cache['test']
        del cache['hwa_raw']
        for split, size in (('training', 45000), ('development', 1000), ('test', 10000)):
            if split not in cache: continue
            if len(cache[split]['labels']) != size: raise ValueError('Frozen cohort size changed.')
            cache[split] = limited_cache(cache[split], spec.max_examples)
            for field in ('features', 'teacher_logits'): cache[split][field] = cache[split][field].to(device)
        context = {**expected, 'protocol': spec.experiment_id, 'phase': spec.phase,
                   'max_examples': spec.max_examples, 'source_checkpoint_sha256': source_sha}
        atomic_write_json(store.run_dir / 'mapping.json', student.mapping_receipt())
        result = screen(student, cache, spec, populations, incoming, rates, store, context)
        result.update(stage=spec.stage, dataset=spec.dataset, backend='om', suffix=spec.suffix, phase=spec.phase,
                      array_seed=spec.array_seed, learning_rates=rates, source_checkpoint_sha256=source_sha,
                      epochs=spec.epochs, recovery_images=spec.max_examples or 45000,
                      smoke_only=bool(spec.max_examples), test_evaluated=spec.phase == 'confirmation',
                      recovery_pulse_cap=None, per_update_pulse_cap=1, verify_tolerance=.04745,
                      evidence_class='exploratory_controller_tuning', runtime_math=runtime_math,
                      elapsed_seconds=time.monotonic() - started)
        store.append_metric({'terminal': True, 'elapsed_seconds': result['elapsed_seconds']})
        artifacts = [store.artifact_record(path, kind='checkpoint' if path.suffix == '.pt' else 'analysis')
                     for path in store.run_dir.rglob('*') if path.is_file() and
                     (path.suffix == '.pt' or path.name in ('mapping.json', 'measurements.json', 'replay.json'))]
        store.complete(metrics=result, artifacts=artifacts)
        print(json.dumps({'run_dir': str(store.run_dir), 'status': 'complete'}), flush=True)
        return 0
    except BaseException as error:
        store.fail(error); raise

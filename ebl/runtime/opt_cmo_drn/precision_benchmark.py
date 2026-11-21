"""Matched four-update DRN HWA timing with FP32 or FP16 conductance matmuls."""
from dataclasses import asdict, replace
import gc
import statistics
import time

import torch

from ebl.runtime.opt_cmo_drn import runtime
from ebl.runtime.store import atomic_write_json
from methods.core.engine import GradientsReadyEvent, train_epoch
from methods.core.guards import FiniteGradientGuard
from methods.hwa.opt_training import update_tf32, CoreStep, lr_factor, make_optimizer
from networks.opt.data import cohorts


def timed_arm(config, teacher, weights, train, development, mode, store):
    fp16 = mode == "fp16_products"
    payload = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    model = runtime.from_mapped(teacher, config, payload).to(config.device)
    del payload
    for block in model.blocks:
        block.solver.fp16_update_products = fp16
    optimizer = make_optimizer(model, config)
    generator = torch.Generator(device=config.device).manual_seed(config.training.seed)
    core = CoreStep(model, teacher, optimizer, config, generator, loss_scale=128 if fp16 else 1)
    guard = FiniteGradientGuard([name for name, _ in model.named_parameters()])
    batches = torch.randperm(len(train), generator=torch.Generator().manual_seed(config.data.seed + 1_000_000))
    samples = {}
    def sample_first(event):
        if isinstance(event, GradientsReadyEvent):
            for label, params in zip(("conductance", "calibration", "digital"), model.parameter_groups()):
                pieces = []
                for param in params:
                    flat = param.grad.detach().flatten()
                    pieces.append(flat[::max(1, flat.numel() // 256)][:256].cpu())
                samples[label] = torch.cat(pieces)
    seconds = []
    torch.cuda.reset_peak_memory_stats()
    for step, idx in enumerate(batches[:128].split(32)):
        batch = train[idx]
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_factor(step, 4, config)
        torch.cuda.synchronize()
        start = time.monotonic()
        handlers = [guard, core.observe_gradients]
        if step == 0:
            handlers.append(sample_first)
        with update_tf32(mode == "tf32" or getattr(config, "hwa_execution", None) is not None):
            train_epoch(core.components(), [(batch, batch)], start_global_step=step, event_handlers=handlers)
        torch.cuda.synchronize()
        seconds.append(time.monotonic() - start)
        store.append_metric(dict(event="precision_update", arm=mode, step=step+1,
                                 seconds=seconds[-1], **core.last))
        optimizer.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated()
    for block in model.blocks:
        block.solver.fp16_update_products = False
    selection = runtime.selection(model, teacher, development, config)
    result = dict(arm=mode, seconds=seconds, median_warm_seconds=statistics.median(seconds[1:]),
                  peak_update_bytes=peak, selection_kl=selection["teacher_kl"])
    if mode == "tf32-baseline":
        # Profile a separate update after selection so instrumentation cannot affect the comparison.
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        batch = train[batches[:32]]
        with torch.profiler.profile(activities=activities) as profile:
            train_epoch(core.components(), [(batch, batch)], start_global_step=4,
                        event_handlers=[guard, core.observe_gradients])
        events = sorted(profile.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)
        result["profile"] = [dict(operator=e.key, calls=e.count, device_us=e.self_device_time_total)
                             for e in events[:20]]
    store.append_metric(dict(event="precision_arm", **result))
    return result, samples


def run(request):
    c = request.spec
    if not request.weights or not request.teacher_weights or request.resume:
        raise ValueError("Benchmark needs a mapped initialization and the pinned teacher.")
    runtime.configure_runtime(c)
    store = runtime._store(request)
    try:
        teacher = runtime.teacher_model(request.teacher_weights, c).to(c.device)
        train, dev, identity = cohorts(c)
        if c.stage == "hwa_speed_benchmark":
            result = speed_trials(c, teacher, request.weights, train, dev, store)
            result["matched_cohorts"] = identity
            report = store.run_dir / "artifacts/speed-benchmark.json"
            atomic_write_json(report, result)
            store.complete(metrics=result, artifacts=[store.artifact_record(report, kind="speed_benchmark")])
            return 0
        fp32, reference = timed_arm(c, teacher, request.weights, train, dev, "fp32", store)
        mode = "tf32" if c.stage == "hwa_tf32_benchmark" else "fp16_products"
        candidate_arm, trial = timed_arm(c, teacher, request.weights, train, dev, mode, store)
        agreement = {}
        for name, ref in reference.items():
            candidate = trial[name].double()
            ref = ref.double()
            agreement[name] = dict(relative_l2=float((ref-candidate).norm()/ref.norm().clamp_min(1e-30)),
                                   cosine=float(torch.nn.functional.cosine_similarity(ref, candidate, dim=0)))
        result = dict(fp32=fp32, matched_cohorts=identity,
                      first_step_sampled_gradient_agreement=agreement,
                      speedup=fp32["median_warm_seconds"]/candidate_arm["median_warm_seconds"])
        result["tf32" if mode == "tf32" else "fp16"] = candidate_arm
        report = store.run_dir / "artifacts/precision-benchmark.json"
        atomic_write_json(report, result)
        store.complete(metrics=result, artifacts=[store.artifact_record(report, kind="precision_benchmark")])
        return 0
    except BaseException as error:
        store.fail(error)
        raise


def speed_trials(config, teacher, weights, train, development, store):
    from ebl.schema.opt_cmo_drn import ExecutionConfig, UpdateExecution
    baseline = UpdateExecution()
    trials = []
    def measure(label, execution):
        cfg = ExecutionConfig(**{**vars(config), "hwa_execution": execution})
        row, gradients = timed_arm(cfg, teacher, weights, train, development, label, store)
        row["execution"] = asdict(execution)
        return row, gradients

    base, reference = measure("tf32-baseline", baseline)
    def compare(label, execution):
        try:
            row, samples = measure(label, execution)
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(arm=label, status="oom", execution=asdict(execution)))
            return False
        except FloatingPointError as error:
            trials.append(dict(arm=label, status="nonfinite", detail=str(error), execution=asdict(execution)))
            return False
        agreement = {}
        for key, expected in reference.items():
            a, b = expected.double(), samples[key].double()
            agreement[key] = dict(cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)),
                                  relative_l2=float((b-a).norm()/a.norm().clamp_min(1e-30)))
        row["gradient_agreement"] = agreement
        row["relative_selection_kl"] = abs(row["selection_kl"]/base["selection_kl"] - 1)
        row["accepted"] = (all(g["cosine"] >= 0.999 and g["relative_l2"] <= 0.01 for g in agreement.values())
                           and row["relative_selection_kl"] <= 0.01
                           and row["median_warm_seconds"] <= 0.9*base["median_warm_seconds"])
        trials.append(row)
        return row["accepted"]

    best = baseline
    for label, execution in [
        ("checkpoint", replace(baseline, checkpoint_gradient_only=True)),
        ("zero", replace(baseline, skip_zero_feedback=True)),
        ("combined", replace(baseline, checkpoint_gradient_only=True, skip_zero_feedback=True)),
    ]:
        gc.collect()
        torch.cuda.empty_cache()
        compare(label, execution)
    accepted = [r for r in trials if r.get("accepted")]
    if accepted:
        best = UpdateExecution(**min(accepted,key=lambda r:r["median_warm_seconds"])["execution"])
    for size in (2, 4):
        gc.collect()
        torch.cuda.empty_cache()
        compare(f"microbatch-{size}", replace(best, microbatch=size, token_chunk_size=size*256))
    eligible = [base] + [r for r in trials if r.get("accepted")]
    return dict(baseline=base, trials=trials, selected=min(eligible,key=lambda r:r["median_warm_seconds"]))

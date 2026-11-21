"""Matched four-update DRN HWA timing with FP32 or FP16 conductance matmuls."""
import statistics
import time

import torch

from ebl.runtime.opt_cmo_drn import runtime
from ebl.runtime.store import atomic_write_json
from methods.core.engine import GradientsReadyEvent, train_epoch
from methods.core.guards import FiniteGradientGuard
from methods.hwa.opt_training import CoreStep, lr_factor, make_optimizer
from networks.opt.data import cohorts


def timed_arm(config, teacher, weights, train, development, fp16, store):
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
    mode = "fp16_products" if fp16 else "fp32"
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
        fp32, reference = timed_arm(c, teacher, request.weights, train, dev, False, store)
        fp16, trial = timed_arm(c, teacher, request.weights, train, dev, True, store)
        agreement = {}
        for name, ref in reference.items():
            candidate = trial[name].double()
            ref = ref.double()
            agreement[name] = dict(relative_l2=float((ref-candidate).norm()/ref.norm().clamp_min(1e-30)),
                                   cosine=float(torch.nn.functional.cosine_similarity(ref, candidate, dim=0)))
        result = dict(fp32=fp32, fp16=fp16, matched_cohorts=identity,
                      first_step_sampled_gradient_agreement=agreement,
                      speedup=fp32["median_warm_seconds"]/fp16["median_warm_seconds"])
        report = store.run_dir / "artifacts/precision-benchmark.json"
        atomic_write_json(report, result)
        store.complete(metrics=result, artifacts=[store.artifact_record(report, kind="precision_benchmark")])
        return 0
    except BaseException as error:
        store.fail(error)
        raise

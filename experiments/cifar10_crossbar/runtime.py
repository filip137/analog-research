"""Native ebl execution boundary for CIFAR-10 crossbar study stages."""
from pathlib import Path

from experiments.artifacts import RunStore, sha256_file, atomic_write_json
from experiments.schema import to_plain_data
from .common import ROOT, Data, setup_cuda, heartbeat


def run_train(request):
    config = to_plain_data(request.spec)
    stage = config["stage"]
    required = {"pretrain": set(), "hwa": {"teacher_weights"}, "deploy": {"teacher_weights"}, "recover": {"teacher_weights", "device_state"}}[stage]
    if stage == "deploy" and config["settings"]["source"] == "hwa":
        required.add("weights")
    for name in ("teacher_weights", "device_state", "weights", "resume", "base_weights", "device_model", "device_data", "selection_receipt"):
        value = getattr(request, name, None)
        if (name in required) != (value is not None):
            raise ValueError(f"Expected {stage} inputs {sorted(required)}; provided {name}={value}.")
    inputs = [dict(role=k, path=str(Path(getattr(request, k)).resolve()), sha256=sha256_file(Path(getattr(request, k)))) for k in sorted(required)]
    device = setup_cuda(config["seed"])
    store = RunStore.create(output_root=request.output_dir, experiment_id=config["experiment_id"], resolved_config=config,
                            command=request.command, repo_root=ROOT, input_artifacts=inputs, resume_capability="unsupported")
    try:
        heartbeat(store, stage=stage, state="loading_data")
        data = Data(config, device)
        atomic_write_json(store.run_dir / "artifacts/data_receipt.json", data.receipt)
        import torch
        store.append_metric(dict(stage="runtime", device=str(device), gpu=torch.cuda.get_device_name(device), data_receipt=data.receipt))
        if stage == "pretrain":
            from .pretrain import run
            metrics, files = run(config, data, device, store)
        else:
            from .hardware import run
            metrics, files = run(request, config, data, device, store)
        store.complete(metrics=metrics, artifacts=[store.artifact_record(p, kind="checkpoint") for p in files])
        atomic_write_json(request.output_dir / "completed_run.json", dict(run_dir=str(store.run_dir), result=str(store.run_dir / "result.json")))
        print(store.run_dir / "result.json", flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise

"""Labelled multiple-choice tasks, fine-tuned teachers and their lifecycle (exp-018).

A character-level stand-in tokenizer and a tiny random OPT replace OPT-125M.
Templates, splits, scoring, the LoRA merge and a full CMO lifecycle with a
declared teacher and task still run through the production code paths.
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from ebl.cli import TrainRequest
from core.artifacts import sha256_file
from core.serialization import load, save
from ebl.registry import get_definition
from core.contracts import ConfigError
import networks.deployment as deployment
import networks.tasks as tasks
import methods.teacher_lora as teacher_lora
from ebl.lifecycle.cli import main as workflow_main
from ebl.schema.lifecycle import parse_lifecycle, provenance
import networks.opt as opt_mlp
from test_workflow_cmo import case_doc, cmo_inputs, small, template  # noqa: F401  (fixture)
from test_workflow_opt import TINY, tiny_teacher
from workflow_helpers import run_lifecycle
from methods.teacher_config import TARGETS as teacher_TARGETS
from methods.teacher_config import SCORES as teacher_SCORES
from methods.teacher_config import parse_config as teacher_parse_config
from ebl.runtime.teacher_lora import run_train as teacher_run_train

VERIFY_PAIR = {"detection": "verify", "roles": "pair", "seed": 241016, "threshold_sigma": 5.0}


def encode(text):
    """Character-level ids inside the tiny vocabulary (no special tokens)."""

    return [3 + ord(c) % 60 for c in text]


def piqa_records(n, seed=0, goals=None):
    generator = torch.Generator().manual_seed(seed)
    letters = "abcdefgh"
    records = []
    for i in range(n):
        a, b = (letters[int(v)] for v in torch.randint(0, 8, (2,), generator=generator))
        goal = goals[i] if goals else f"g{i % 10}"
        records.append({"goal": goal, "sol1": a + "x", "sol2": b + "yy", "label": int(torch.randint(0, 2, (1,),
                                                                                           generator=generator))})
    return records


def tiny_task(root, name="tinytask", train=24, test=6, development=4, seed=5):
    document = tasks.build_task(task="piqa", model="facebook/opt-125m", train=piqa_records(train, seed),
                                validation=piqa_records(test, seed + 1), encode=encode,
                                development_examples=development, split_seed=17, sources={}, tokenizer={})
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.pt"
    save(path, document)
    return document, sha256_file(path)


# --- templates, encoding and splits -------------------------------------------------


def test_templates_follow_lm_eval():
    query, choices, label, group = tasks.TASKS["hellaswag"].format({
        "activity_label": "Removing ice from car", "ctx_a": "Then, the man writes over the snow.",
        "ctx_b": "then", "endings": ["[header] How to clean [title] Wash it.", " b ", "c", "d"],
        "label": "3", "source_id": "activitynet~v_1"})
    assert query == "Removing ice from car: Then, the man writes over the snow. Then"
    assert choices == [" How to clean. Wash it.", "b", "c", "d"] and label == 3 and group == "activitynet~v_1"
    query, choices, label, _ = tasks.TASKS["sst2"].format({"sentence": "a fine film ", "label": 1})
    assert query == "a fine film \nQuestion: Is this sentence positive or negative?\nAnswer:"
    assert choices == ["negative", "positive"] and label == 1
    query, choices, _, group = tasks.TASKS["piqa"].format({"goal": "Open a jar", "sol1": "twist", "sol2": "pull",
                                                          "label": 0})
    assert query == "Question: Open a jar\nAnswer:" and choices == ["twist", "pull"] and group == "Open a jar"


def test_encode_pair_moves_trailing_context_spaces_into_the_continuation():
    head, tail = tasks.encode_pair(encode, "Answer: ", "yes")
    assert head == encode("Answer:") and tail == encode(" yes")


def test_development_split_is_group_disjoint_exact_and_seeded():
    goals = [f"goal{i // 3}" for i in range(30)]  # ten groups of three examples
    records = piqa_records(30, goals=goals)
    parts = tasks.split_train(tasks.TASKS["piqa"], records, 7, seed=3)
    assert len(parts["development"]) == 7 and len(parts["excluded"]) == 2
    groups = {name: {records[i]["goal"] for i in parts[name]} for name in ("train", "development")}
    assert not groups["train"] & groups["development"]
    assert sorted(parts["train"] + parts["development"] + parts["excluded"]) == list(range(30))
    assert parts == tasks.split_train(tasks.TASKS["piqa"], records, 7, seed=3)
    assert parts != tasks.split_train(tasks.TASKS["piqa"], records, 7, seed=4)


def test_task_file_and_gold_stream_hold_every_example(tmp_path):
    document, _ = tiny_task(tmp_path)
    tasks.check_task(document, "facebook/opt-125m")
    assert document["overlap"]["development_train_groups"] == 0
    assert {name: len(split["labels"]) for name, split in document["splits"].items()} == {
        "train": len(document["splits"]["train"]["labels"]), "development": 4, "test": 6}
    test = document["splits"]["test"]
    records = piqa_records(6, 6)
    stream = tasks.gold_stream(test)
    expected = []
    for record in records:
        query, choices, label, _ = tasks.TASKS["piqa"].format(record)
        head, tail = tasks.encode_pair(encode, query, " " + choices[label])
        expected += [tasks.BOS] + head + tail
    assert stream.tolist() == expected
    assert test["continuation_chars"].tolist() == [len(c) for r in records for c in (r["sol1"], r["sol2"])]


# --- scoring ----------------------------------------------------------------------------


def task_split(document, split="test", count=6):
    value = tasks.cohort(document, split, count, TINY.max_positions)
    value["choices"]["decision"] = "acc_norm"
    return value


def test_scores_do_not_depend_on_padding_or_batch_composition(tmp_path, monkeypatch):
    document, _ = tiny_task(tmp_path)
    teacher = tiny_teacher()
    choices = task_split(document)["choices"]
    batched = teacher_lora.score(teacher, choices, "cpu")
    single = torch.stack([
        teacher_lora.row_sums(teacher, tasks.padded(choices, torch.tensor([r])), "cpu")[0]
        for r in range(len(batched))])
    assert torch.allclose(batched, single, atol=1e-5)
    monkeypatch.setattr(tasks, "TOKEN_BUDGET", 30)
    assert len(tasks.batches(choices, 30)) > len(tasks.batches(choices, 4096)) >= 1
    # The per-batch padding of the evaluation path gives the same task metrics.
    network = opt_mlp.OptMlpSuffix(teacher, 2, tile_size=8)
    small_batches = opt_mlp.evaluate(network, task_split(document), "cpu")
    monkeypatch.setattr(tasks, "TOKEN_BUDGET", 4096)
    large_batches = opt_mlp.evaluate(network, task_split(document), "cpu")
    assert small_batches["predictions"] == large_batches["predictions"]
    assert small_batches["accuracy_norm_percent"] == large_batches["accuracy_norm_percent"]


def test_accuracy_and_normalized_accuracy_can_disagree():
    choices = {"k": 2, "labels": torch.tensor([1]), "chars": torch.tensor([1, 10])}
    sums = torch.tensor([-2.0, -5.0], dtype=torch.float64)
    value = tasks.finish(sums, sums, choices, token_kl=0.0, scored=2, decision="acc_norm")
    assert value["accuracy_percent"] == 0 and value["accuracy_norm_percent"] == 100
    assert value["choice_kl"] == 0 and value["teacher_agreement_percent"] == 100 and value["predictions"] == "1"
    delta, se = tasks.paired_standard_error("1101", "0101", torch.tensor([1, 1, 0, 1]))
    assert delta == 25.0 and se > 0


def test_clean_suffix_reproduces_the_teacher_and_a_perturbed_state_does_not(tmp_path):
    document, _ = tiny_task(tmp_path)
    network = opt_mlp.OptMlpSuffix(tiny_teacher(), 2, tile_size=8)
    split = task_split(document)
    clean = opt_mlp.evaluate(network, split, "cpu")
    assert clean["choice_kl"] <= opt_mlp.REPRODUCTION_KL and clean["teacher_kl"] <= opt_mlp.REPRODUCTION_KL
    assert clean["teacher_agreement_percent"] == 100 and clean == opt_mlp.evaluate(network, split, "cpu")
    q = network.q.detach() + 0.5 * torch.randn(network.q.shape, generator=torch.Generator().manual_seed(1))
    noisy = opt_mlp.evaluate(network, split, "cpu", q=q)
    assert noisy["choice_kl"] > 1e-4 and noisy["teacher_kl"] > 1e-4
    teacher = opt_mlp.teacher_task_metrics(network, split, "cpu")
    assert teacher["predictions"] == clean["predictions"]


# --- the fine-tuned teacher ------------------------------------------------------------


def teacher_config(sha, **training):
    return {
        "experiment_id": "opt_task_teacher.v1", "schema_version": 1, "teacher_id": "tiny-piqa",
        "model": "facebook/opt-125m",
        "task": {"name": "piqa", "choices": {"name": "tinytask", "sha256": sha}, "decision": "acc_norm",
                 "development_examples": 4},
        "lora": {"rank": 2, "alpha": 4.0, "targets": list(teacher_TARGETS)},
        "training": {"learning_rate": 0.01, "weight_decay": 0.0, "warmup_fraction": 0.1, "epochs": 2,
                     "batch_examples": 4, "micro_examples": 2, "max_grad_norm": 1.0, "score": "sum", "seed": 61,
                     "development_every_steps": 3, "max_examples": 0, **training},
        "execution": {"device": "cpu", "cpu_threads": 1, "disable_cudnn": False},
    }


def test_lora_merge_round_trips_through_the_hugging_face_layout(tmp_path):
    base = tiny_teacher()
    config = teacher_parse_config(teacher_config("0" * 64))
    model = copy.deepcopy(base)
    parameters = teacher_lora.add_lora(model, config)
    with torch.no_grad():
        for p in parameters:
            p.add_(0.1 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())))
    merged = teacher_lora.merge_lora(model)
    ids = torch.randint(0, TINY.vocab_size, (3, 9), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.allclose(model(ids), merged(ids), atol=1e-5)
    path = tmp_path / "pytorch_model.bin"
    torch.save(teacher_lora.hugging_face_state(merged), path)
    reloaded = opt_mlp.from_hugging_face(torch.load(path, weights_only=True), TINY)
    assert all(torch.equal(reloaded.state_dict()[k], v) for k, v in merged.state_dict().items())
    with torch.no_grad():
        assert torch.equal(reloaded(ids), merged(ids))
    adapted = [k for k in base.state_dict() if k.endswith(tuple(f"{n}.weight" for n in teacher_TARGETS))]
    assert len(adapted) == TINY.decoder_layers * len(teacher_TARGETS)
    assert all(torch.equal(base.state_dict()[k], merged.state_dict()[k]) for k in base.state_dict() if k not in adapted)
    assert sum(p.numel() for p in parameters) == sum(2 * (m.shape[0] + m.shape[1])
                                                     for k, m in base.state_dict().items() if k in adapted)


def test_teacher_config_is_strict():
    with pytest.raises(ConfigError, match="micro_examples"):
        teacher_parse_config(teacher_config("0" * 64, micro_examples=3))
    doc = teacher_config("0" * 64)
    doc["lora"]["targets"] = ["q_proj", "q_proj"]
    with pytest.raises(ConfigError, match="lora.targets"):
        teacher_parse_config(doc)
    with pytest.raises(ConfigError, match="sha256"):
        teacher_parse_config(teacher_config("XYZ"))


@pytest.mark.parametrize("kind", teacher_SCORES)
def test_teacher_objectives_are_finite_and_differentiable(tmp_path, kind):
    document, _ = tiny_task(tmp_path)
    model = copy.deepcopy(tiny_teacher())
    parameters = teacher_lora.add_lora(model, teacher_parse_config(teacher_config("0" * 64, score=kind)))
    choices = task_split(document, "train", 8)["choices"]
    loss = teacher_lora.objective(model, choices, torch.arange(4), choices["labels"], 2, kind, "cpu")
    loss.backward()
    assert bool(torch.isfinite(loss)) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)
    if kind == "gold_lm":
        rows = torch.arange(4) * 2 + choices["labels"][:4]
        batch = tasks.padded(choices, rows)
        with torch.no_grad():
            expected = -teacher_lora.row_sums(model, batch, "cpu").sum() / (batch["lengths"] - batch["context_lengths"]).sum()
        assert torch.allclose(loss.detach(), expected)


def test_teacher_fine_tuning_selects_on_development_and_exports_a_pinned_teacher(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    base = tiny_teacher()
    monkeypatch.setattr(opt_mlp, "load_teacher", lambda path, network, device: (copy.deepcopy(base), "e" * 64))
    _, sha = tiny_task(tmp_path / "tasks")
    weights = tmp_path / "base.bin"
    weights.write_bytes(b"synthetic base")
    definition = get_definition("opt_task_teacher.v1")
    spec = definition.resolve(definition.parse(teacher_config(sha)), definition.supported_modes[0])
    request = TrainRequest(definition=definition, spec=spec, config_path=Path("teacher.json"),
                           output_dir=tmp_path / "runs", weights=None, resume=None, command=("pytest",),
                           device_data=tmp_path / "tasks/tinytask.pt", teacher_weights=weights)
    assert teacher_run_train(request) == 0
    [run] = sorted((tmp_path / "runs").iterdir())
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "complete"
    provenance_doc = json.loads((run / "teacher_provenance.json").read_text())
    assert provenance_doc["task"]["test_split_read"] is False
    assert all(v is True for k, v in provenance_doc["gates"].items() if isinstance(v, bool))
    assert provenance_doc["gates"]["adapted_weights_changed"] > 0
    history = provenance_doc["selection"]["history"]
    best = max(history, key=lambda h: h["decision_accuracy_percent"])
    assert provenance_doc["development"]["adapter"]["decision_accuracy_percent"] == best["decision_accuracy_percent"]
    merged = run / "checkpoints/pytorch_model.bin"
    assert sha256_file(merged) == provenance_doc["merged"]["sha256"] == result["metrics"]["merged_sha256"]
    assert {a["kind"] for a in result["artifacts"]} == {"teacher_weights", "lora_adapters", "teacher_provenance"}


# --- schema and the lifecycle ----------------------------------------------------------


def teacher_lifecycle(doc, task_sha, teacher_sha, primary="task_accuracy_norm_percent"):
    doc = copy.deepcopy(doc)
    doc["network"]["teacher"] = {"name": "tiny-ft", "sha256": teacher_sha}
    doc["task"] = {"name": "piqa", "choices": {"name": "tinytask", "sha256": task_sha}, "decision": "acc_norm",
                   "development_examples": 4, "evaluation_examples": 6}
    doc["question"]["primary_metric"] = primary
    return doc


def test_teacher_and_task_are_optional_and_enter_identity_only_when_declared():
    base = parse_lifecycle(template())
    assert "teacher" not in base.to_dict()["network"] and "task" not in base.to_dict()
    declared = parse_lifecycle(teacher_lifecycle(template(), "1" * 64, "2" * 64))
    assert declared.network.teacher.sha256 == "2" * 64 and declared.task.name == "piqa"
    assert declared.section_digest("network") != base.section_digest("network")
    source = provenance(declared, "hwa_source", hwa_arm=declared.hwa.arms[0].arm_id)["sections"]
    assert "task" not in source and "task" in provenance(declared, "deployment",
                                                         hwa_arm=declared.hwa.arms[0].arm_id)["sections"]
    # A task changes caches and deployments, never the HWA source.
    other = parse_lifecycle(teacher_lifecycle(template(), "3" * 64, "2" * 64))
    assert provenance(other, "hwa_source", hwa_arm=other.hwa.arms[0].arm_id)["sections"] == source


@pytest.mark.parametrize("change,message", [
    (lambda d: d["question"].update(primary_metric="task_accuracy_norm_percent") or d.pop("task"), "task metric"),
    (lambda d: d["task"].update(name="mmlu"), "task.name"),
    (lambda d: d["task"].update(decision="best"), "task.decision"),
    (lambda d: d["network"]["teacher"].update(sha256="nothex"), "network.teacher.sha256"),
    (lambda d: d["task"].update(extra=1), "task fields"),
])
def test_task_schema_rejects_invalid_settings(change, message):
    doc = teacher_lifecycle(template(), "1" * 64, "2" * 64)
    change(doc)
    with pytest.raises(ConfigError, match=message):
        parse_lifecycle(doc)


def test_declared_teacher_digest_is_the_one_loaded(tmp_path, monkeypatch):
    monkeypatch.setitem(opt_mlp.PINNED, "facebook/opt-125m", {**opt_mlp.PINNED["facebook/opt-125m"], "config": TINY})
    path = tmp_path / "pytorch_model.bin"
    torch.save(teacher_lora.hugging_face_state(tiny_teacher()), path)
    network = parse_lifecycle(teacher_lifecycle(template(), "1" * 64, sha256_file(path))).network
    teacher, digest = opt_mlp.load_teacher(path, network, "cpu")
    assert digest == sha256_file(path) == opt_mlp.pinned_digest(network)
    assert torch.equal(teacher.embed_tokens.weight, tiny_teacher().embed_tokens.weight)
    with pytest.raises(ValueError, match="fine-tuned teacher 'tiny-ft'"):
        opt_mlp.load_teacher(path, parse_lifecycle(teacher_lifecycle(template(), "1" * 64, "2" * 64)).network, "cpu")
    with pytest.raises(ValueError, match="revision"):
        opt_mlp.load_teacher(path, parse_lifecycle(template()).network, "cpu")


def test_task_lifecycle_runs_six_arms_with_task_metrics_and_collects(cmo_inputs, tmp_path):
    weights, digest = cmo_inputs
    _, task_sha = tiny_task(tmp_path / "corpora")
    arms = [
        {"id": "none", "weights": "hold", "calibration": False},
        {"id": "calibration", "weights": "hold", "calibration": True},
        {"id": "pair_rewrite_verify", "weights": "hold", "calibration": False, "fault_rewrite": VERIFY_PAIR},
        {"id": "pair_rewrite_verify_calibration", "weights": "hold", "calibration": True,
         "fault_rewrite": VERIFY_PAIR},
        {"id": "pair_rewrite_verify_recovery", "weights": "learn", "calibration": False,
         "fault_rewrite": VERIFY_PAIR},
        {"id": "pair_rewrite_verify_recovery_calibration", "weights": "learn", "calibration": True,
         "fault_rewrite": VERIFY_PAIR},
    ]
    doc = small(**{
        "data.corpus": {"name": "synthetic", "sha256": digest},
        "data.sequence_length": 9, "data.max_examples": 8,
        "defects.cases": [case_doc(), case_doc("gmax", 50000)], "hwa.arms": [{"id": "digital", "method": "none"}],
        "onchip.batch_size": 4, "onchip.epochs": 1, "onchip.learning_rate": .01,
        "onchip.calibration_learning_rate": .001,
        "onchip.update_law": "cmo_thresholded_write_adam", "onchip.threshold_sigma": 1.,
        "onchip.accumulate_updates": True, "onchip.max_write_attempts_per_device": 4,
        "onchip.development_every_steps": 1, "onchip.arms": arms,
    }).to_dict()
    lc = parse_lifecycle(teacher_lifecycle(doc, task_sha, "f" * 64))
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    prepare = results["prepare"][1]["metrics"]
    assert prepare["task"]["task_test"]["digital"]["choice_kl"] <= opt_mlp.REPRODUCTION_KL
    assert prepare["cohorts"]["task_test"]["count"] == 6 and prepare["cohorts"]["task_development"]["count"] == 4
    assert prepare["teacher"]["scored_tokens"] > 0
    assert results["deploy__digital__271001"][1]["metrics"]["fine_tuned_teacher"]["name"] == "tiny-ft"
    run, result = results["onchip__digital__271001"]
    assert result["metrics"]["exact_replays"] == 12
    assert set(result["metrics"]["rows"][0]["task_test"]["final"]) == set(opt_mlp.TASK_SUMMARY_METRICS)
    rows = {(r["case"], r["onchip_arm"]): r for r in json.loads((run / "measurements.json").read_text())}
    for (case, arm), row in rows.items():
        assert row["fine_tuned_teacher"] == {"name": "tiny-ft", "sha256": "f" * 64}
        for state in ("clean", "initial", "final"):
            assert row[state]["task_test"]["examples"] == 6 and len(row[state]["task_test"]["predictions"]) == 6
            assert row[state]["task_development"]["examples"] == 4
        if arm in ("none", "pair_rewrite_verify"):
            assert row["final"]["task_test"] == (row["initial"]["task_test"] if arm == "none"
                                                 else row["final"]["task_test"])
    nominal = rows["nominal", "none"]
    assert nominal["final"]["task_test"]["choice_kl"] == nominal["initial"]["task_test"]["choice_kl"]
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    flat = json.loads((collected / "rows.json").read_text())
    assert len(flat) == 12 and all("final_task_test_accuracy_norm_percent" in r for r in flat)
    assert all("clean_task_development_choice_kl" in r for r in flat)
    deployments = json.loads((collected / "deployments.json").read_text())
    assert all("deployed_task_test_accuracy_norm_percent" in r for r in deployments)
    # The task cache is rejected for a lifecycle that declares another task file.
    with pytest.raises(ValueError, match="task"):
        cache = load(results["prepare"][0] / "checkpoints/cache.pt")
        opt_mlp.check_cache(cache, parse_lifecycle(teacher_lifecycle(doc, "4" * 64, "f" * 64)))


def test_task_data_command_writes_task_and_stream(tmp_path, monkeypatch):
    pandas = pytest.importorskip("pandas")
    transformers = pytest.importorskip("transformers")

    class Tokenizer:
        model_max_length = 0

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return {"input_ids": encode(text)}

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda path: Tokenizer()))
    for name, records in (("train", piqa_records(24, 5)), ("validation", piqa_records(6, 6))):
        pandas.DataFrame(records).to_parquet(tmp_path / f"{name}.parquet")
    argv = ["task-data", "--task", "piqa", "--name", "piqa-tiny", "--model-dir", str(tmp_path),
            "--train", str(tmp_path / "train.parquet"), "--validation", str(tmp_path / "validation.parquet"),
            "--development-examples", "4", "--split-seed", "17", "--root", str(tmp_path / "root")]
    assert workflow_main(argv) == 0
    document = load(tmp_path / "root/piqa-tiny.pt")
    stream = load(tmp_path / "root/piqa-tiny-stream.pt")
    assert stream["schema"] == opt_mlp.CORPUS_SCHEMA
    assert torch.equal(stream["splits"]["validation"], tasks.gold_stream(document["splits"]["development"]))
    assert stream["sources"]["task_file"]["sha256"] == sha256_file(tmp_path / "root/piqa-tiny.pt")
    assert workflow_main(argv) == 2

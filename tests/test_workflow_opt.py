"""OPT network family: decoder semantics, crossbar suffix, schema and a full lifecycle.

A tiny random OPT stands in for the pinned OPT-125M checkpoint; corpus,
teacher loading and AIHWKit sampling are synthetic, but every stage still runs
as a native bundle with provenance checks and exact on-chip replay.
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from experiments.artifacts import sha256_file
from experiments.cifar_crossbar.runtime import save
from experiments.schema import ConfigError
from workflow import data as data_stage
from workflow import deployment
from workflow import devices as device_stage
from workflow.__main__ import main as workflow_main
from workflow.lifecycle import parse_lifecycle
from workflow.networks import opt_mlp
from workflow_helpers import lifecycle_document, network_for, om_raw, run_lifecycle

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "examples/lifecycles/opt125m-om-mlp4-template.json"
TINY = opt_mlp.OptConfig(vocab_size=64, hidden_size=16, decoder_layers=3, attention_heads=2, ffn_dim=32, max_positions=32)
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--facebook--opt-125m/snapshots" / opt_mlp.PINNED[
    "facebook/opt-125m"
]["revision"]


def template():
    return json.loads(TEMPLATE.read_text())


def changed(doc, dotted, value):
    doc = copy.deepcopy(doc)
    *parents, key = dotted.split(".")
    target = doc
    for part in parents:
        target = target[int(part)] if isinstance(target, list) else target[part]
    target[key] = value
    return doc


def tiny_teacher(seed=7):
    torch.manual_seed(seed)
    return opt_mlp.OptDecoder(TINY).eval().requires_grad_(False)


def tokens(n=3, length=9, seed=3):
    return torch.randint(0, TINY.vocab_size, (n, length), generator=torch.Generator().manual_seed(seed))


def kl(student, teacher):
    return float(
        torch.nn.functional.kl_div(
            student.flatten(0, 1).log_softmax(1), teacher.flatten(0, 1).softmax(1), reduction="batchmean"
        )
    )


# --- decoder and crossbar suffix ------------------------------------------------


def test_decoder_matches_hugging_face_opt():
    transformers = pytest.importorskip("transformers")
    config = transformers.OPTConfig(
        vocab_size=TINY.vocab_size, hidden_size=TINY.hidden_size, num_hidden_layers=TINY.decoder_layers,
        ffn_dim=TINY.ffn_dim, num_attention_heads=TINY.attention_heads, max_position_embeddings=TINY.max_positions,
        word_embed_proj_dim=TINY.hidden_size, do_layer_norm_before=True, dropout=0.0, attention_dropout=0.0,
        activation_dropout=0.0, pad_token_id=1, bos_token_id=2, eos_token_id=2,
    )
    torch.manual_seed(0)
    reference = transformers.OPTForCausalLM(config).eval()
    mine = opt_mlp.from_hugging_face(reference.state_dict(), TINY).eval()
    ids = tokens()
    with torch.no_grad():
        assert torch.allclose(mine(ids), reference(ids).logits, atol=1e-5)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_clean_suffix_reproduces_teacher_and_keeps_state_lean(layers):
    teacher = tiny_teacher()
    suffix = opt_mlp.OptMlpSuffix(teacher, layers, tile_size=8)
    ids = tokens()
    with torch.no_grad():
        hidden = suffix.features(ids)
        student, reference = suffix.forward_features(hidden), suffix.teacher_logits(hidden)
        # Only rescaling and tiled summation separate the clean suffix from the teacher.
        assert torch.allclose(student, reference, atol=1e-5)
        assert kl(student, reference) <= opt_mlp.REPRODUCTION_KL
        assert torch.allclose(reference, teacher(ids)[:, :-1], atol=1e-6)
    start = TINY.decoder_layers - layers
    assert [s.name for s in suffix.layout] == [
        f"decoder.layers.{i}.{m}" for i in range(start, TINY.decoder_layers) for m in ("fc1", "fc2")
    ]
    assert suffix.q.numel() == layers * 2 * TINY.hidden_size * TINY.ffn_dim
    assert {k.split(".")[0] for k in suffix.state_dict()} == {"q", "scales", "output_gains", "mlp_biases", "mlp_norms"}
    suffix.enable_calibration(True)
    assert len(suffix.calibration_parameters()) == 1 + 2 * layers + 2 * layers


def test_training_never_changes_the_frozen_digital_teacher():
    teacher = tiny_teacher()
    suffix = opt_mlp.OptMlpSuffix(teacher, 2, tile_size=8)
    frozen = suffix.prefix_hash()
    suffix.enable_calibration(True)
    suffix.q.requires_grad_(True)
    optimizer = torch.optim.Adam([suffix.q] + suffix.calibration_parameters(), lr=0.1)
    hidden, target, labels = opt_mlp.batch(suffix, {"tokens": tokens()}, torch.arange(3), "cpu")
    loss = opt_mlp.objective(suffix.forward_features(hidden), target, labels, "teacher_kl")
    loss = loss + opt_mlp.objective(suffix.forward_features(hidden) * 1.5, target, labels, "teacher_kl")
    loss.backward()
    optimizer.step()
    assert suffix.prefix_hash() == frozen
    assert all(p.grad is None for p in teacher.parameters())


# --- schema ----------------------------------------------------------------------


def test_template_lifecycle_is_valid():
    lc = parse_lifecycle(template())
    assert lc.network.family == "opt_mlp_suffix" and lc.network.analog_decoder_layers == 4
    assert lc.network.summary()["analog_layer_indices"] == [8, 9, 10, 11]
    assert lc.data.cohort_size("test") == 8 and lc.data.smoke


def test_describe_reports_analog_weights_and_state_estimate(capsys):
    assert workflow_main(["describe", str(TEMPLATE), "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["analog_weights"] == 4 * 2 * 768 * 3072 == 18_874_368
    assert value["trajectories"] == 30
    assert value["onchip_state_gb_estimate"] == round(30 * 18_874_368 * 90 / 1e9, 2)


@pytest.mark.parametrize("convolutions", [0, 2, 4])
def test_cifar_analog_weights_match_the_crossbar_suffix(convolutions):
    lc = parse_lifecycle(changed(lifecycle_document("cifar10-om-conv4-smoke"), "network.analog_convolutions", convolutions))
    _, network = network_for(convolutions)
    assert lc.network.analog_weights == network.q.numel()


@pytest.mark.parametrize(
    "dotted,value,message",
    [
        ("network.analog_decoder_layers", 0, "analog_decoder_layers"),
        ("network.analog_decoder_layers", 13, "analog_decoder_layers"),
        ("network.analog_convolutions", 4, "network fields"),
        ("network.model", "facebook/opt-350m", "network.model"),
        ("question.primary_metric", "accuracy_percent", "for opt_mlp_suffix"),
        ("deployment.calibration", "output_gain_bn_affine_classifier_bias", "deployment.calibration"),
        ("hwa.augment", True, "hwa.augment"),
        ("data.test_sequences", 0, "test_sequences"),
        ("data.sequence_length", 4096, "sequence_length"),
        ("data.corpus.sha256", "ABC", "sha256"),
        ("data.hwa_images", 10, "data fields"),
    ],
)
def test_opt_lifecycle_rejects_invalid_settings(dotted, value, message):
    with pytest.raises(ConfigError, match=message):
        parse_lifecycle(changed(template(), dotted, value))


def test_cifar_lifecycle_rejects_opt_only_settings():
    doc = lifecycle_document("cifar10-om-conv4-smoke")
    with pytest.raises(ConfigError, match="for cifar_resnet32_suffix"):
        parse_lifecycle(changed(doc, "question.primary_metric", "perplexity"))
    with pytest.raises(ConfigError, match="deployment.calibration"):
        parse_lifecycle(changed(doc, "deployment.calibration", "output_gain_mlp_layer_norm_mlp_bias"))


# --- corpus and cohorts ------------------------------------------------------------


def write_corpus(root, name="synthetic", lengths=(400, 120, 120), seed=11):
    generator = torch.Generator().manual_seed(seed)
    corpus = {
        "schema": opt_mlp.CORPUS_SCHEMA,
        "model": "facebook/opt-125m",
        "tokenizer": {"class": "synthetic"},
        "sources": {},
        "splits": {
            split: torch.randint(0, TINY.vocab_size, (n,), generator=generator)
            for split, n in zip(opt_mlp.CORPUS_SPLITS, lengths)
        },
    }
    path = root / f"{name}.pt"
    save(path, corpus)
    return sha256_file(path)


def smoke(corpus_sha256, **changes):
    doc = template()
    doc["network"]["analog_decoder_layers"] = 2
    doc["data"].update(corpus={"name": "synthetic", "sha256": corpus_sha256}, sequence_length=9, max_examples=8)
    doc["deployment"]["tile_size"] = 8
    doc["devices"]["characterization"].update(bins=5, samples=16)
    doc["hwa"]["batch_size"] = 4
    doc["onchip"]["batch_size"] = 4
    for dotted, value in changes.items():
        doc = changed(doc, dotted, value)
    return parse_lifecycle(doc)


@pytest.fixture
def opt_inputs(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    teacher = tiny_teacher()
    monkeypatch.setattr(deployment, "load_teacher", lambda path, network, device: (teacher, "f" * 64))
    monkeypatch.setattr(
        device_stage, "native_sample", lambda *, kind, size, seed, policy, variation, output: om_raw(size, seed)
    )
    corpora = tmp_path / "corpora"
    monkeypatch.setenv("EBL_CORPUS_ROOT", str(corpora))
    weights = tmp_path / "teacher.pt"
    weights.write_bytes(b"synthetic teacher")
    return teacher, weights, write_corpus(corpora)


def test_cache_builds_nested_seeded_cohorts_and_rejects_other_corpora(opt_inputs):
    teacher, _, digest = opt_inputs
    lc = smoke(digest)
    network = deployment.build_network(teacher, lc)
    cache = data_stage.build_cache(teacher, network, lc, "cpu")
    data_stage.check_cache(cache, lc)
    assert {k: tuple(cache[k]["tokens"].shape) for k in ("hwa", "training", "development", "test")} == {
        k: (8, 9) for k in ("hwa", "training", "development", "test")
    }
    assert torch.equal(cache["hwa"]["tokens"], cache["training"]["tokens"])
    assert cache["development"]["cohort"]["split"] == "validation"
    with pytest.raises(ValueError, match="SHA-256"):
        data_stage.build_cache(teacher, network, smoke("1" * 64), "cpu")
    with pytest.raises(ValueError, match="at least 8 validation sequences"):
        data_stage.build_cache(teacher, network, smoke(digest, **{"data.sequence_length": 16}), "cpu")


# --- the full lifecycle ------------------------------------------------------------


def test_opt_om_lifecycle_runs_end_to_end_with_matched_controls(opt_inputs, tmp_path):
    _, weights, digest = opt_inputs
    lc = smoke(digest)
    results = run_lifecycle(lc, weights, tmp_path / "runs")
    prepare_dir, prepare = results["prepare"]
    assert prepare["metrics"]["digital"]["teacher_kl"] <= opt_mlp.REPRODUCTION_KL
    assert prepare["metrics"]["network_family"] == "opt_mlp_suffix"
    mapping = json.loads((prepare_dir / "mapping.json").read_text())
    assert mapping["analog_decoder_layers"] == 2 and mapping["logical_weights"] == 2 * 2 * 16 * 32
    arms = [arm.arm_id for arm in lc.onchip.arms]
    for hwa_arm in lc.hwa.arms:
        run_dir, _ = results[f"onchip__{hwa_arm.arm_id}__271001"]
        rows = json.loads((run_dir / "measurements.json").read_text())
        assert [(r["case"], r["onchip_arm"]) for r in rows] == [
            (case.label, arm) for case in lc.defects.cases for arm in arms
        ]
        for row in rows:
            assert row["replay_exact"] and row["final"]["test"]["scored_tokens"] == 8 * 8
            assert {"teacher_kl", "perplexity", "persistent_test"} <= set(row["final"]["test"]) | set(row["final"])
            if row["onchip_arm"] == "none":
                assert row["final"] == row["initial"] and row["curve"] == []
            if row["weights"] == "hold":
                assert row["cost"].get("physical_pulses", 0) == 0
            else:
                assert row["curve"] and row["sequence_presentations"] == lc.data.max_examples
    _, deploy = results["deploy__cdt_gmax__271001"]
    assert {"deployment_teacher_kl", "deployment_perplexity_ratio"} <= set(deploy["metrics"]["cases"]["gmax_20000ppm"])
    collected = tmp_path / "collected"
    assert workflow_main(["collect", str(tmp_path / "runs"), "--output", str(collected)]) == 0
    rows = json.loads((collected / "rows.json").read_text())
    assert len(rows) == len(lc.hwa.arms) * len(lc.defects.cases) * len(arms)
    assert all("final_perplexity" in row and "sequence_presentations" in row for row in rows)


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="pinned OPT-125M snapshot not cached")
def test_corpus_command_tokenizes_splits_with_the_pinned_tokenizer(tmp_path):
    pytest.importorskip("transformers")
    texts = {}
    for split in ("train", "validation", "test"):
        texts[split] = tmp_path / f"{split}.txt"
        texts[split].write_text(f"The {split} split of a crossbar corpus.\n" * 5)
    argv = ["corpus", "--name", "tiny", "--model-dir", str(SNAPSHOT), "--root", str(tmp_path / "root")]
    argv += [item for split, path in texts.items() for item in (f"--{split}", str(path))]
    assert workflow_main(argv) == 0
    corpus = torch.load(tmp_path / "root/tiny.pt", weights_only=True)
    assert corpus["schema"] == opt_mlp.CORPUS_SCHEMA
    assert all(stream.dtype == torch.int64 and int(stream[0]) == 2 for stream in corpus["splits"].values())
    assert workflow_main(argv) == 2

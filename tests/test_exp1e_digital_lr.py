"""The digital-LR intervention preserves the fixed analog development protocol."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp1e_digital_lr", ROOT / "campaigns/opt-cmo-corruption/launch/exp1e_digital_lr.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def test_candidates_change_only_additional_digital_rate():
    documents = []
    for label, rate in study.RATES.items():
        lc = study.lifecycle(label)
        assert lc.hwa.learning_rate == 0.005
        assert lc.hwa.arms[0].digital_learning_rate == rate
        assert lc.hwa.lr_schedule.factor(25600) == 0.2
        assert lc.hwa.lr_schedule.warmup_steps == 205
        assert lc.hwa.sequences_per_epoch == 8192 and lc.hwa.epochs == 100
        assert lc.hwa.selection_every == 2 and lc.hwa.extension_epochs == 0
        assert lc.hwa.seed == 41 and lc.data.data_seed == 20261008
        assert lc.hwa.noise_std == 0.03 and lc.hwa.arms[0].corruption is None
        assert lc.data.evaluation == "development" and lc.data.test_sequences == 0
        assert lc.transfer_tests is None and lc.onchip is None
        doc = lc.to_dict()
        doc.pop("lifecycle_id")
        doc["hwa"]["arms"][0]["digital_learning_rate"] = 0
        documents.append(doc)
    assert all(doc == documents[0] for doc in documents)


def test_smokes_are_matched_except_scope_and_do_not_change_main_budget():
    analog = study.lifecycle("analog", smoke=True).to_dict()
    zero = study.lifecycle("0", smoke=True).to_dict()
    assert analog["hwa"]["lr_schedule"]["total_steps"] == 512
    assert zero["hwa"]["arms"][0]["digital_learning_rate"] == 0
    analog.pop("lifecycle_id"); zero.pop("lifecycle_id")
    zero["hwa"]["arms"] = analog["hwa"]["arms"]
    assert zero == analog
    with pytest.raises(ValueError):
        study.lifecycle("analog")


def test_partial_collection_never_recommends_a_rate(tmp_path):
    (tmp_path / "index.json").write_text("[]")
    report = study.review(tmp_path)
    assert not report["coverage_complete"]
    assert report["recommended_digital_lr_within_new_grid"] is None
    assert not report["test_data_used"]

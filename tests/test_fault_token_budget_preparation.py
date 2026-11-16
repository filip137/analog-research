import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "campaigns/pilots/20261116-fault-token-budget"
spec = importlib.util.spec_from_file_location("token_config_builder", PILOT / "make_configs.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
CASES = json.loads((PILOT / "cases.json").read_text())


def sample(case):
    return {"sequences":8192,"shape":[8192,255,8],"cohort_sha256":case["cohort_sha256"],
            "file":"test-only-targets.pt","tensor_sha256":"a"*64}


@pytest.mark.parametrize("case",CASES)
@pytest.mark.parametrize("canary",[False,True])
def test_fixed_science_and_exact_case_binding(case,canary):
    doc = builder.build(case,sample(case),canary)
    f=doc["fault_tuning"]
    assert len(f["arms"])==1
    assert f["arms"][0]["learning_rate"]==case["learning_rate"]
    if builder.BACKEND=="drn":
        assert doc["deployment"]=={"assignment_seed":271001,"fault_kind":"random","fault_rate":0.03}
        assert doc["cpu_threads"]==10
        assert doc["recovery"]["threshold_sigma"]==2 and doc["recovery"]["write_noise_factor"]==1
        assert f["steps"]==(32 if canary else 2400)
        assert f["batch"]==8 and f["recovery_sequences"]==8192 and f["samples_count"]==8
        assert f["development_every"]==100 and f["schedule"]=="cosine_to_zero"
        assert f["arms"][0]["method"]=="noisy"
        assert f["development_sha256"]=="7411cf50b625e4c62f65d8e12bac538fd83c6dad3e4c74b29a9f2a6f5ad83191"
    else:
        assert f["assignment_seed"]==271001 and f["write_law"]["threshold_sigma"]==2
        assert f["write_law"]["accumulate_updates"] is True
        assert f["schedule"]["steps"]==(32 if canary else 2400)
        assert f["schedule"]["batch"]==8 and f["recovery"]["sequences"]==8192 and f["targets"]["count"]==8
        assert f["schedule"]["development_every_steps"]==100
        assert f["arms"][0]["method"]=="onchip_noisy"
        assert f["development"]["cohort_sha256"]=="7411cf50b625e4c62f65d8e12bac538fd83c6dad3e4c74b29a9f2a6f5ad83191"
        assert doc["defects"]["cases"][0]["rate_ppm"]==30000


@pytest.mark.parametrize("change",[{"cohort_sha256":"b"*64},{"shape":[8192,255,7]},{"file":"../escape.pt"}])
def test_reject_mismatched_or_unsafe_target_binding(change):
    case=CASES[0];target=sample(case);target.update(change)
    with pytest.raises(ValueError):builder.build(case,target)


def test_reject_wrong_teacher_before_reading_targets(tmp_path,monkeypatch):
    import sys
    receipt={"schema":"fault_token_targets.v1","status":"complete",
             "baseline_prefix_sha256":builder.BASELINE_SAMPLES,"teacher_sha256":"b"*64,
             "samples_seed":20365833,"draw_batch":16,"count":8,"samples":[]}
    path=tmp_path/"receipt.json";path.write_text(json.dumps(receipt))
    monkeypatch.setattr(sys,"argv",["make_configs.py",str(path),str(tmp_path),str(tmp_path/"out")])
    with pytest.raises(ValueError,match="preparation contract"):builder.main()
    assert not (tmp_path/"out").exists()

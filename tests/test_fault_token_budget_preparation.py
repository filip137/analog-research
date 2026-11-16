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


def test_streaming_preserves_fixed_batch_rng_draw():
    import torch
    spec=importlib.util.spec_from_file_location("token_target_prep",PILOT/"prepare_targets.py")
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    tokens=torch.arange(64*5).reshape(64,5)
    def logits(t):return torch.sin(t[:,:-1,None].float()+torch.arange(11).float())
    g=torch.Generator().manual_seed(77)
    streamed=torch.cat(list(prep.draw_chunks(tokens,logits,g)))
    g=torch.Generator().manual_seed(77);old=[]
    hidden=torch.cat([logits(t) for t in tokens.split(16)])
    for part in hidden.split(16):
        probs=part.float().softmax(-1)
        old.append(torch.multinomial(probs.flatten(0,1),8,replacement=True,generator=g).view(16,4,8))
    assert torch.equal(streamed,torch.cat(old))
    with pytest.raises(ValueError,match="prefix changed"):
        prep.validate_prefix(streamed)


def test_target_preparation_refuses_existing_output(tmp_path,monkeypatch):
    import sys
    spec=importlib.util.spec_from_file_location("token_target_prep",PILOT/"prepare_targets.py")
    prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
    monkeypatch.setattr(sys,"argv",["prepare_targets.py","not-read-corpus","not-read-teacher",str(tmp_path)])
    with pytest.raises(FileExistsError):prep.main()

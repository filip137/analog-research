"""Recovery context changes preserve physical/source identity and matched token budgets."""
import copy
import json

import pytest
import torch

import networks.context_views as context_views
import networks.deployment as deployment
from ebl.schema.lifecycle import parse_lifecycle, provenance, verify_provenance, StageSelection
import networks.opt as opt_mlp
from ebl.lifecycle.cli import main as workflow_main
from test_workflow_cmo import cmo_inputs, small, case_doc
from test_workflow_opt import tiny_teacher
from workflow_helpers import run_lifecycle, run_stage


def document(digest="a" * 64, length=4):
    doc = small(**{"data.corpus": {"name": "synthetic", "sha256": digest},
                   "data.sequence_length": 8, "data.max_examples": 0,
                   "data.hwa_sequences": 8, "data.recovery_sequences": 8,
                   "data.development_sequences": 4, "data.test_sequences": 4,
                   "hwa.batch_size": 4, "hwa.arms": [{"id": "digital", "method": "none"}],
                   "defects.cases": [case_doc(), case_doc("random", 50000)]}).to_dict()
    doc["onchip"].update(update_law="cmo_thresholded_write_adam", threshold_sigma=2,
                         accumulate_updates=True, max_write_attempts_per_device=2,
                         epochs=1, batch_size=32 // length, microbatch_size=16 // length,
                         case_labels=["random_50000ppm"], development_every_steps=1)
    doc["onchip"]["arms"] = [{"id":"none","weights":"hold","calibration":False},
                               {"id":"onchip_weights","weights":"learn","calibration":False}]
    doc["recovery_views"] = {"context_length":length,"parent_length":16,"parent_sequences":4,
                              "score_stride":4,"evaluation_lengths":[8,16]}
    return doc


def test_view_provenance_preserves_frozen_source_but_rejects_wrong_cache():
    doc=document(); lc=parse_lifecycle(doc)
    old=copy.deepcopy(doc); old.pop("recovery_views"); old=parse_lifecycle(old)
    for kind in ["hwa_source","deployment"]:
        verify_provenance(provenance(old,kind,hwa_arm="digital"),lc,kind,hwa_arm="digital")
    for source,target in [(old,lc),(lc,old)]:
        with pytest.raises(ValueError,match="recovery_views"):
            verify_provenance(provenance(source,"feature_cache"),target,"feature_cache")
    changed=copy.deepcopy(doc); changed["data"]["sequence_length"]=16
    with pytest.raises(ValueError,match="data"):
        verify_provenance(provenance(old,"deployment",hwa_arm="digital"),parse_lifecycle(changed),"deployment",hwa_arm="digital")
    assert parse_lifecycle(lc.to_dict()).to_dict()==lc.to_dict()
    assert "recovery_views" not in old.to_dict()


@pytest.mark.parametrize("field,value",[("context_length",5),("score_stride",3),
                                      ("evaluation_lengths",[8,8]),("evaluation_lengths",[7])])
def test_invalid_views_rejected(field,value):
    doc=document();doc["recovery_views"][field]=value
    with pytest.raises(ValueError): parse_lifecycle(doc)


def test_tokens_targets_and_update_membership_match_across_contexts(cmo_inputs):
    _,digest=cmo_inputs
    baseline=None
    for length in [4,8,16]:
        lc=parse_lifecycle(document(digest,length)); teacher=tiny_teacher()
        network=deployment.build_network(teacher,lc,"digital")
        cache=opt_mlp.build_cache(teacher,network,lc,"cpu");opt_mlp.check_cache(cache,lc)
        train=cache["recovery_context"]["tokens"]; order=context_views.order(lc,1)
        mask=context_views.loss_mask(lc,"cpu")
        values=(train[order].flatten(),train[order][:,:-1][:,mask].flatten(),train[order][:,1:][:,mask].flatten())
        if baseline is None: baseline=values
        else: assert all(torch.equal(a,b) for a,b in zip(baseline,values))
        assert train[order[:lc.onchip.batch_size]].numel()==32
        assert cache["context16_development"]["tokens"].shape==(2,16)
        assert torch.equal(cache["context16_development"]["tokens"].flatten(),cache["development"]["tokens"].flatten())
        cache["context16_development"]["tokens"][0,0] += 1
        with pytest.raises(ValueError,match="evaluation"): opt_mlp.check_cache(cache,lc)


def test_native_context_recovery_reuses_original_p0_and_replays(cmo_inputs,tmp_path):
    weights,digest=cmo_inputs;doc=document(digest,8)
    old=copy.deepcopy(doc);old.pop("recovery_views");old=parse_lifecycle(old)
    results=run_lifecycle(old,weights,tmp_path/"original")
    lc=parse_lifecycle(doc)
    results["prepare"]=run_stage(lc,StageSelection("prepare"),results,weights,tmp_path/"view")
    run,result=run_stage(lc,StageSelection("onchip","digital",271001),results,weights,tmp_path/"view")
    rows=json.loads((run/"measurements.json").read_text());assert len(rows)==2
    held,learned=rows
    assert held["initial"]==held["final"] and held["initial_apparent_sha256"]==held["final_apparent_sha256"]
    assert all(r["case"]=="random_50000ppm" and r["replay_exact"] for r in rows)
    assert learned["optimizer_updates"]==2 and learned["input_token_presentations"]==64
    assert learned["scored_token_presentations"]==48
    assert learned["cost"]["max_onchip_device_write_attempts"]<=2
    state=torch.load(run/"checkpoints/onchip/random_50000ppm/onchip_weights.pt",weights_only=False)
    assert state["writer"]["step_index"]==2  # four microbatches, only two physical update opportunities
    assert workflow_main(["collect",str(run),"--output",str(tmp_path/"collected")])==0
    flat=json.loads((tmp_path/"collected/rows.json").read_text())
    assert "final_context16_development_teacher_kl" in flat[1]

"""CPU coverage of the native two-sided downstream adaptation stage and its task scorers."""
from types import SimpleNamespace as NS

import torch

from core.artifacts import sha256_file
from core.serialization import save
from ebl.runtime.fault_tuning import Context
from ebl.runtime.supervised_adaptation import run_supervised
from ebl.schema.lifecycle import parse_lifecycle
from methods.physical_child import nested_masks
from methods.task_adaptation import choice_outputs
from methods.task_tuning_config import ROUTES
from networks.opt import OptMlpSuffix
from tests.test_fault_tuning_mapping import TWO_SIDED, hold_only
from tests.test_workflow_opt import tiny_teacher


def test_nested_physical_masks_ties_across_arrays():
    layout = [NS(offset=0, size=4, shape=(2,2)), NS(offset=4, size=2, shape=(1,2))]
    scores = torch.tensor([.7, .8, .2, .1, .8, .3])
    eligible = torch.ones(2, 6, dtype=torch.bool)
    eligible[0, 1] = False
    masks = nested_masks(scores, eligible, layout, (.1, .5), 79.2)
    assert masks[.1].sum() == 1 and masks[.1][1, 1]
    assert masks[.5].sum() == 6
    assert not (masks[.1] & ~masks[.5]).any()


def test_choice_ce_and_character_normalization():
    split = {"tokens": torch.tensor([0, 1, 1, 0, 2, 0, 1, 1, 0, 2]), "offsets": torch.tensor([0,3,5,8,10]),
             "context_lengths": torch.ones(4, dtype=torch.long), "chars": torch.tensor([5,1,5,1]),
             "k": 2, "pad": 0, "labels": torch.tensor([0,1])}
    def forward(tokens):
        return torch.zeros(*tokens[:, :-1].shape, 3, requires_grad=True)
    loss, pred, _ = choice_outputs(forward, split, torch.arange(2), "cpu")
    torch.testing.assert_close(loss, torch.tensor(2.).log().double())
    assert pred.tolist() == [0,0]


def test_native_supervised_routes_and_costs(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setenv("EBL_CORPUS_ROOT", str(tmp_path))
    gens = torch.Generator().manual_seed(31)
    def pin(name):
        doc = {"schema": "s7-task.v1", "task": name, "kind": "lm",
               "order": torch.stack([torch.randperm(4, generator=gens) for _ in range(3)]),
               "splits": {k: {"tokens": torch.randint(0, 64, (4, 8), generator=gens)}
                          for k in ("train", "development", "test")}}
        path = tmp_path / (name + ".pt")
        save(path, doc)
        return {"name": name, "sha256": sha256_file(path)}
    spec = {"phase": "canary", "rates": {"child": .2, "full": .001, "lora": .01}, "proxy_seed": 8511,
            "seed": 41, "task": pin("pubmed"), "retention": pin("wikitext"), "selection": None,
            "fisher_items": 4, "steps": 3, "eval_limit": 2, "microbatch": 2, "routes": list(ROUTES)}
    doc = hold_only(TWO_SIDED)
    cfg = doc["fault_tuning"]
    doc["fault_tuning"] = {k: cfg[k] for k in ("source_sha256", "assignment_seed", "mapping", "fault_rewrite")}
    # The source retains its complete historical devices section with disjoint new recipient identities.
    new_seed = 99910
    assert new_seed not in doc["devices"]["assignment_seeds"]
    doc["fault_tuning"]["assignment_seed"] = new_seed
    doc["fault_tuning"]["task_adaptation"] = spec
    lc = parse_lifecycle(doc)
    assert parse_lifecycle(lc.to_dict()).digest() == lc.digest()
    from ebl.schema.lifecycle import _selection
    assert _selection({"kind": "fault_tuning", "hwa_arm": lc.hwa.arms[0].arm_id, "array_seed": new_seed}, lc).array_seed == new_seed
    net = OptMlpSuffix(tiny_teacher(), 2, 8, "per_output_channel_absmax", "model").requires_grad_(False)
    stage = NS(lifecycle=lc, network=net, device=torch.device("cpu"), selection=NS(array_seed=new_seed),
               store=NS(append_metric=lambda obj: None), run_dir=tmp_path)
    result, _ = run_supervised(Context(stage), net.q.detach().clone())
    assert set(result["scores"]) == set(ROUTES) | {"ideal","mapped","hold","calibration"}
    import json
    rows = json.loads((tmp_path / "measurements.json").read_text())["scores"]
    for key in ("offline_child01", "offline_child10", "onchip_child01", "onchip_child10"):
        assert rows[key]["cost"]["out_of_mask_attempts"] == 0
        assert rows[key]["cost"]["successful_device_writes"] > 0
    assert rows["offline_lora"]["cost"]["successful_device_writes"] == 0
    assert {p.name for p in (tmp_path / "checkpoints").glob("physical-masks-*.pt")} == {
        "physical-masks-chip.pt", "physical-masks-proxy.pt", "physical-masks-clean.pt"}
    from ebl.runtime.supervised_adaptation import archived_v100_readout
    held = [{"g": torch.tensor([[52., 87.3], [24.8, 85.2]], dtype=torch.float32)}]
    expected = (held[0]["g"][0] - held[0]["g"][1]) * torch.tensor(79.2).reciprocal()
    assert torch.equal(archived_v100_readout(held, 79.2), expected)

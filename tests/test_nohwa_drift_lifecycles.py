"""Series003 exp-002 part (b) crossbar lifecycles: CMO exp-001 x4 conditions, A0 / A0-twin / CAL only."""
import json
from pathlib import Path

from ebl.schema.lifecycle import load_lifecycle, parse_lifecycle, stage_selections

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "campaigns/fault-recovery-paper/lifecycles/nohwa-drift-20261119"
EXP001 = ROOT / "campaigns/pilots/20261118-cmo-drift-recovery/lifecycles/full"
SOURCES = ROOT / "campaigns/fault-recovery-paper/lifecycles/mapping-faults-20261119"
TOKEN = ROOT / "campaigns/pilots/20261116-drift-token-budget/lifecycles/full"
PRETRAINED = {"xb6": "3442d7d2b4137c4126b2501a54d27a594b1ccbddc171b39d7329e2e4fd73a1b4",
              "xb12": "480e1323aaed8a12036121d887f1aa38c3316b7cb39354ca14353cfc76c3abc8"}


def test_cells_change_only_arms_mapping_and_source_of_cmo_exp001():
    paths = sorted(CELLS.glob("*.json"))
    assert len(paths) == 7
    for path in paths:
        doc = json.loads(path.read_text())
        _, _, _, model, source, mapping = doc["lifecycle_id"].removesuffix("-replay").split("-")
        ref = json.loads((EXP001 / f"cmo-drift-recovery-20261118-{model}-m4.json").read_text())
        lc = parse_lifecycle(doc)
        assert [s.kind for s in stage_selections(lc)] == ["prepare", "devices", "hwa", "fault_tuning"]
        ft = lc.fault_tuning
        assert [(a.label, a.method, a.learning_rate, a.no_drift) for a in ft.arms] == [
            ("A0", "hold_fault_aware", None, False), ("A0-twin", "hold_fault_aware", None, True),
            ("CAL", "calibration", 0.001, False)]
        assert ft.drift.multiplier == 4.0 and (ft.drift.drift_min_us, ft.drift.drift_max_us) == (5.0, 150.0)
        assert ft.assignment_seed == 271001 and lc.defects.cases[0].rate_ppm == 0
        assert ft.mapping is None
        assert ft.drift.mapping == (None if mapping == "base" else ref["fault_tuning"]["drift"]["mapping"])
        f, g = doc.pop("fault_tuning"), ref.pop("fault_tuning")
        f["drift"].pop("mapping", None), g["drift"].pop("mapping")
        for key in ("arms", "source_sha256"):
            f.pop(key), g.pop(key)
        assert f == g
        if source == "nohwa":
            assert doc["hwa"]["arms"] == [{"id": "pretrained", "method": "none"}]
            assert lc.fault_tuning.source_sha256 == PRETRAINED[model]
        else:
            assert doc["hwa"] == ref["hwa"] and lc.fault_tuning.source_sha256 == ref_sha(model)
        for key in ("hwa", "lifecycle_id", "question"):
            doc.pop(key), ref.pop(key)
        assert doc == ref


def ref_sha(model):
    return json.loads((EXP001 / f"cmo-drift-recovery-20261118-{model}-m4.json").read_text())["fault_tuning"]["source_sha256"]


def test_pretrained_sources_fit_the_cell_provenance_sections():
    # The pinned no-HWA sources come from these lifecycles; verify_provenance checks these sections
    # and the HWA arm digest (shared HWA settings plus the arm).
    for model in ("xb6", "xb12"):
        source = load_lifecycle(SOURCES / f"mapping-faults-20261119-{model}-pretrained-source.json")
        cell = load_lifecycle(CELLS / f"nohwa-drift-20261119-{model}-nohwa-two.json")
        for name in ("network", "data", "devices", "deployment", "program_verify"):
            assert source.section_digest(name) == cell.section_digest(name)
        assert source.hwa_arm_digest("pretrained") == cell.hwa_arm_digest("pretrained")


def test_base_cell_deploys_and_drifts_as_the_drift_token_budget_base():
    # Same assignment, defects, devices, P&V, rewrite, law, draw seed, grid and development cohort;
    # the standard pair in both. Unwritten devices read the release-clock expression under either clock.
    for model in ("xb6", "xb12"):
        cell = load_lifecycle(CELLS / f"nohwa-drift-20261119-{model}-hwa-base.json")
        token = load_lifecycle(TOKEN / f"drift-token-budget-20261116-{model}-m4.json")
        for name in ("network", "data", "devices", "deployment", "program_verify", "defects"):
            assert cell.section_digest(name) == token.section_digest(name)
        a, b = cell.fault_tuning, token.fault_tuning
        assert (a.assignment_seed, a.fault_rewrite, a.development, a.source_sha256) == \
            (b.assignment_seed, b.fault_rewrite, b.development, b.source_sha256)
        da, db = a.drift.to_dict(), b.drift.to_dict()
        assert da.pop("write_clock") == "reset_absorb" and db.pop("write_clock") == "continues"
        assert da == db and a.drift.mapping is None and b.drift.mapping is None

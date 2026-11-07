"""P0 unit tests for exp-001 (run on a GPU host from the code snapshot root).

python labs/domain_ft/test_domain_ft.py <run_root>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import torch  # noqa: E402

import model as M  # noqa: E402
import train as T  # noqa: E402

METHODS = ["full", "mlp", "digital", "ln", "bitfit", "ia3", "lora_fc16", "padapter64", "lora_attn16",
           "lora_fc4", "lora_fc64", "lora_fc256", "padapter16", "padapter256", "padapter1024"]


def main(root: Path):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    opt_path = root / "model" / "pytorch_model.bin"
    report = {}
    files = T.data_files(root)
    hs = torch.load(files["hellaswag"], weights_only=False)
    cs = T.ChoiceSet(hs, "development", 20)
    tok, mask = T.pad_rows(cs.rows[:16], cs.ctx[:16], dev)

    base, _ = M.build_model(opt_path, "opt", None, dev)
    base.eval()
    with torch.no_grad():
        ref = base.head(base.hidden(tok))
    counts = {}
    for m in METHODS:
        mod, _ = M.build_model(opt_path, "opt", None, dev)
        params = M.apply_method(mod, m, 41)
        n = sum(p.numel() for p in params)
        exp = M.expected_params(m)
        if m == "full":
            exp = sum(p.numel() for p in base.parameters())
        assert n == exp, (m, n, exp)
        assert all(p.requires_grad for p in params)
        assert sum(p.numel() for p in mod.parameters() if p.requires_grad) == n, m
        mod.eval()
        with torch.no_grad():
            out = mod.head(mod.hidden(tok))
        neutral = bool(torch.equal(out, ref))
        assert neutral, (m, float((out - ref).abs().max()))
        counts[m] = n
        del mod
    report["param_counts"] = counts
    report["init_neutral"] = "all methods bitwise identical logits at init"

    # scorer equality with hwa networks.tasks + networks.opt on 20 HellaSwag dev examples
    import networks.tasks as tasks
    from networks.opt import PINNED, from_hugging_face
    tasks.check_task(hs, "facebook/opt-125m")
    state = torch.load(opt_path, map_location="cpu", weights_only=True)
    hwa = from_hugging_face(state, PINNED["facebook/opt-125m"]["config"]).to(dev).eval()
    c = tasks.cohort(hs, "development", 20, 2048)["choices"]
    sums_h = torch.zeros(len(c["offsets"]) - 1, dtype=torch.float64)
    with torch.no_grad():
        for rows in tasks.batches(c):
            b = tasks.padded(c, rows)
            mk = tasks.scored_mask(b, dev)
            t = b["tokens"].to(dev)
            logits = hwa(t)[:, :-1]
            _, tgt = tasks.continuation_log_probs(logits, t, mk)
            sums_h[rows] = tasks.row_sums(tgt, mk).cpu()
    sums_m, cnt = T.eval_rows(base, cs.rows, cs.ctx, dev)
    diff = float((sums_m - sums_h).abs().max())
    dec_h = tasks.decision_scores(sums_h.view(-1, 4), c["chars"].view(-1, 4), "acc_norm").argmax(1)
    dec_m = (sums_m.view(-1, 4) / cs.chars.view(-1, 4)).argmax(1)
    assert diff < 2e-3 and torch.equal(dec_h, dec_m), diff
    report["scorer_vs_hwa"] = {"examples": 20, "max_abs_sum_diff": diff, "decisions_equal": True}

    # existing task files still validate; PIQA rebuild from source is identical with the extended registry
    for name in ("hellaswag", "sst2", "piqa"):
        tasks.check_task(torch.load(files[name], weights_only=False), "facebook/opt-125m")
    src = root / "data" / "hwa" / "sources" / "piqa"
    if src.exists():
        import pandas
        from transformers import AutoTokenizer
        tok_ = AutoTokenizer.from_pretrained(str(root / "model"))
        tok_.model_max_length = 1 << 62
        recs = {n: pandas.read_parquet(src / f"{n}.parquet").to_dict("records") for n in ("train", "validation")}
        piqa = torch.load(files["piqa"], weights_only=False)
        doc = tasks.build_task(task="piqa", model="facebook/opt-125m", train=recs["train"], validation=recs["validation"],
                               encode=lambda s: tok_(s, add_special_tokens=False)["input_ids"],
                               development_examples=piqa["split"]["development_examples"],
                               split_seed=piqa["split"]["seed"], sources={}, tokenizer={})
        for sp in tasks.CHOICE_SPLITS:
            for k, v in piqa["splits"][sp].items():
                assert torch.equal(v, doc["splits"][sp][k]), (sp, k)
        report["piqa_rebuild_identical"] = True

    # std3 clean dense model == OptMlpSuffix forward of the source
    std3 = Path.home() / "explore-20261101/inputs/crossbar-std3.pt"
    if std3.exists():
        from networks.opt import OptMlpSuffix
        teacher = hwa.requires_grad_(False)
        net = OptMlpSuffix(teacher, 6, 256, "per_output_channel_absmax", "model").to(dev)
        net.load_state_dict(torch.load(std3, map_location=dev, weights_only=False)["model"])
        net.eval()
        clean, rec = M.build_model(opt_path, "std3_clean", std3, dev)
        clean.eval()
        x = hs["splits"]["development"]["context_ids"][:255]
        t = torch.cat([torch.tensor([2]), x])[None].to(dev)
        with torch.no_grad():
            a = net(t)
            b = clean.head(clean.hidden(t))[:, :-1]
        d = float((a - b).abs().max())
        assert d < 1e-2, d
        report["std3_clean_vs_OptMlpSuffix_max_logit_diff"] = d
        report["std3_sha256"] = rec["std3_sha256"]
        dep, rec1 = M.build_model(opt_path, "std3_d1", std3, dev)
        report["std3_d1_programming"] = rec1["programming"]
    print(json.dumps(report, indent=2))
    (root / "unit_tests.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

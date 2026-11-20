"""Build the shared OPT-1.3B synthetic corpus or the shared recovery targets."""
import argparse
import json
import time
from pathlib import Path

import torch
from core.artifacts import sha256_file
from core.serialization import save
from methods.fault_tuning import cohort, raw_hash
from networks.opt import CORPUS_SCHEMA, PINNED, load_teacher

MODEL = "facebook/opt-1.3b"
COHORT = "c919329dbe0d545d7614daf3d8f790881e944cd80ff53fff88fc19d9697f7762"
SPLITS = {"train": (195200, 20261120), "validation": (256, 20261121), "test": (384, 20261122)}


@torch.inference_mode()
def corpus(args):
    from transformers import AutoTokenizer, OPTConfig, OPTForCausalLM

    directory = args.teacher.parent
    expected_files = {
        "config.json": "f1aaceefb7d34d346e9b0400d95fdb4970567bcbe1d4aba1068c5402099b644f",
        "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        "vocab.json": "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab",
        "special_tokens_map.json": "0f3cd5f307a2735cfee31ddf33be88655e8be165ee2cdc26ced966f04258adf0",
        "tokenizer_config.json": "d1e4185b43a0a06421ab7218d5c1a775350bd13024507b34fbed25dcf6c14c6b",
    }
    for name, digest in expected_files.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Pinned {name} digest mismatch")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    prompt = tokenizer.encode("The", add_special_tokens=False)
    config = OPTConfig.from_pretrained(directory, local_files_only=True)
    model = OPTForCausalLM(config)
    model.load_state_dict(torch.load(args.teacher, weights_only=True, map_location="cpu"), strict=True)
    model.generation_config.eos_token_id = None
    model.to(args.device).eval()
    files = {name: digest for name, digest in expected_files.items() if name != "config.json"}
    result = {"schema": CORPUS_SCHEMA, "model": MODEL, "tokenizer": {"files": files},
              "teacher_sha256": PINNED[MODEL]["sha256"], "revision": PINNED[MODEL]["revision"],
              "generation": {"prompt": "The", "chunk_length": 512, "temperature": 0.9, "top_k": 50,
                             "batch_size": args.batch_size, "torch": str(torch.__version__),
                             "config_sha256": sha256_file(directory / "config.json")},
              "sources": {}, "splits": {}}
    splits = SPLITS if args.kind == "corpus" else {"probe": (4*args.batch_size, SPLITS["train"][1])}
    if args.kind == "canary-corpus":
        splits = {name: (min(256,n), seed) for name, (n, seed) in SPLITS.items()}
    for split, (windows, seed) in splits.items():
        torch.manual_seed(seed)
        chunks = windows // 2
        pieces = []
        for i in range(0, chunks, args.batch_size):
            batch = min(args.batch_size, chunks-i)
            ids = torch.tensor([prompt]*batch, device=args.device)
            started = time.monotonic()
            draw = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                  temperature=0.9, top_k=50, max_new_tokens=512-len(prompt), eos_token_id=None,
                                  pad_token_id=tokenizer.pad_token_id)
            if draw.shape != (batch, 512):
                raise RuntimeError("Unexpected generation length")
            pieces.append(draw.cpu().flatten())
            if i == 0 or (i // args.batch_size) % 100 == 0 or args.kind == "generation-probe":
                print(json.dumps({"split": split, "chunks_done": i+batch,
                                  "tokens_per_second": draw.numel()/(time.monotonic()-started)}), flush=True)
        tokens = torch.cat(pieces)
        result["splits"][split] = tokens
        result["sources"][split] = {"seed": seed, "sha256": raw_hash(tokens), "tokens": tokens.numel()}
    if args.kind == "generation-probe":
        result["schema"] = "opt13b.generation_probe.v1"
    return result


@torch.inference_mode()
def recovery(args):
    from ebl.schema.lifecycle import parse_lifecycle

    probe = Path(__file__).resolve().parents[1] / "pilots/20261118-opt1p3b-port/opt1p3b-cmo-l12-probe.json"
    lc = parse_lifecycle(json.loads(probe.read_text()))
    tokens = cohort(lc, {"corpus": lc.data.corpus, "corpus_sha256": lc.data.corpus_sha256,
                         "split": "train", "sequences": 1024, "seed": 20261104, "cohort_sha256": COHORT})
    teacher, _ = load_teacher(args.teacher, lc.network, args.device)
    generator = torch.Generator(device=args.device).manual_seed(20261104 + 104729)
    draws = []
    for batch in tokens.split(16):
        probs = teacher(batch.to(args.device))[:, :-1].float().softmax(-1)
        draws.append(torch.multinomial(probs.flatten(0, 1), 8, replacement=True,
                                       generator=generator).reshape(len(batch), 255, 8).cpu())
    return torch.cat(draws)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("corpus", "recovery", "generation-probe", "canary-corpus"))
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists() or args.batch_size < 1:
        p.error("Specify an unused output path and a positive generation batch")
    if sha256_file(args.teacher) != PINNED[MODEL]["sha256"]:
        p.error("Teacher digest differs from pinned OPT-1.3B")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    value = corpus(args) if args.kind != "recovery" else recovery(args)
    save(args.output, value)
    print(json.dumps({"file": str(args.output), "sha256": sha256_file(args.output),
                      "contents": value["sources"] if args.kind != "recovery" else raw_hash(value)}))

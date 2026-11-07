"""Build exp-001 task files, LM corpora and probe files from the pinned sources.

Usage: python labs/domain_ft/build_data.py <data_root> <opt_snapshot_dir> <hwa_data_dir> [keys...]

Choice tasks use networks.tasks (choice_task.v1, extra ``test_mismatched`` split for
MNLI). LM corpora use token_corpus.v1 (BOS + documents joined by id 2). Every
output gets a receipt in <data_root>/receipts/<key>.json.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import networks.tasks as tasks  # noqa: E402
from networks.opt import CORPUS_SCHEMA  # noqa: E402

SEED = 20261108
MAX_LEN = 256
MODEL = "facebook/opt-125m"
TRAIN_CAP = 100_000  # choice-task train examples kept (after the seeded shuffle); 1,000 x 32 = 32k are used


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class Builder:
    def __init__(self, root: Path, snapshot: Path):
        self.root = root
        self.tok = AutoTokenizer.from_pretrained(str(snapshot))
        self.tok.model_max_length = 1 << 62
        files = ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json")
        self.tok_receipt = {"class": type(self.tok).__name__,
                            "special_tokens": "none from the tokenizer; BOS (2) prepended to every scored row",
                            "files": {n: sha256(snapshot / n) for n in files if (snapshot / n).exists()}}
        self.sources = json.loads((root / "receipts" / "sources.json").read_text())

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def src(self, key: str, name: str) -> Path:
        return self.root / self.sources[key]["files"][name]["path"]

    def src_receipt(self, key: str) -> dict:
        s = self.sources[key]
        return {"repo": s["repo"], "revision": s["revision"],
                "files": {n: {"source": f["source"], "sha256": f["sha256"]} for n, f in s["files"].items()}}

    def save(self, name: str, value: dict, receipt: dict) -> None:
        path = self.root / "built" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(value, tmp)
        tmp.replace(path)
        receipt = {"file": f"built/{name}.pt", "sha256": sha256(path), "bytes": path.stat().st_size, **receipt}
        (self.root / "receipts" / f"{name}.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps({k: receipt[k] for k in receipt if k in ("file", "examples", "tokens", "blocks")}), flush=True)

    # ---- choice tasks -------------------------------------------------------------
    def truncation(self, split: dict, k: int) -> dict:
        n = len(split["labels"])
        ctx = split["context_offsets"].diff()
        cont = split["continuation_offsets"].diff().view(n, k).amax(1)
        over = (1 + ctx + cont) > MAX_LEN
        return {"examples": n, "truncated": int(over.sum()), "rate": float(over.double().mean()),
                "max_row_tokens": int((1 + ctx + cont).max())}

    def choice_doc(self, task: str, splits: dict[str, list[tuple[int, dict]]], extra: dict) -> None:
        spec = tasks.TASKS[task]
        encoded = {name: tasks._encode_split(spec, self.encode, items) for name, items in splits.items()}
        doc = {"schema": tasks.CHOICE_SCHEMA, "model": MODEL, "task": task, "choices": spec.choices,
               "template": spec.template, "bos": tasks.BOS, "pad": tasks.PAD, "tokenizer": self.tok_receipt,
               "sources": self.src_receipt(task if task != "boolq" else "boolq"), "split": extra, "splits": encoded}
        tasks.check_task(doc, MODEL)
        labels = {n: torch.bincount(s["labels"], minlength=spec.choices).tolist() for n, s in encoded.items()}
        dev = encoded["development"]["labels"]
        receipt = {"task": task, "template": spec.template, "split": extra,
                   "examples": {n: len(s["labels"]) for n, s in encoded.items()},
                   "label_counts": labels,
                   "majority_dev_percent": 100 * float(torch.bincount(dev).max()) / len(dev),
                   "truncation_left_context": {n: self.truncation(s, spec.choices) for n, s in encoded.items()},
                   "sources": self.src_receipt(task), "tokenizer": self.tok_receipt}
        self.save(f"{task}-opt125m-dft1", doc, receipt)

    def shuffled(self, rows: list[int], seed: int) -> list[int]:
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
        return [rows[i] for i in order]

    def mnli(self):
        spec = tasks.TASKS["mnli"]
        train = pq.read_table(self.src("mnli", "train")).to_pylist()
        train = [r for r in train if int(r["label"]) in (0, 1, 2)]
        parts = tasks.split_train(spec, train, 2000, SEED)
        kept = parts["train"][:TRAIN_CAP]
        vm = pq.read_table(self.src("mnli", "validation_matched")).to_pylist()
        vmm = pq.read_table(self.src("mnli", "validation_mismatched")).to_pylist()
        extra = {"seed": SEED, "group_key": "premise", "development_examples": 2000,
                 "development_groups": parts["development_groups"], "train_available": len(parts["train"]),
                 "train_cap": TRAIN_CAP, "excluded_train_examples": len(parts["excluded"]),
                 "test": "validation_matched", "test_mismatched": "validation_mismatched"}
        self.choice_doc("mnli", {"train": [(i, train[i]) for i in kept],
                                 "development": [(i, train[i]) for i in parts["development"]],
                                 "test": list(enumerate(vm)), "test_mismatched": list(enumerate(vmm))}, extra)

    def jsonl(self, path):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def rct(self):
        spec = tasks.TASKS["rct"]
        train, dev, test = (self.jsonl(self.src("rct", n)) for n in ("train", "dev", "test"))
        kept = self.shuffled(list(range(len(train))), SEED)[:TRAIN_CAP]
        parts = tasks.split_train(spec, dev, 2000, SEED)
        extra = {"seed": SEED, "group_key": "abstract_id", "development_examples": 2000,
                 "development_source": "official dev (grouped by abstract)",
                 "development_groups": parts["development_groups"], "train_available": len(train),
                 "train_cap": TRAIN_CAP}
        self.choice_doc("rct", {"train": [(i, train[i]) for i in kept],
                                "development": [(i, dev[i]) for i in parts["development"]],
                                "test": list(enumerate(test))}, extra)

    def boolq(self):
        spec = tasks.TASKS["boolq"]
        stats = {"truncated": 0, "examples": 0}

        def fit(record):
            # Tail-truncate the passage so BOS + context + longest continuation fits 256 tokens.
            stats["examples"] += 1
            tail = f"\nQuestion: {record['question']}?\nAnswer:"
            budget = MAX_LEN - 1 - 2 - len(self.encode(tail)) - 2
            ids = self.encode(record["passage"])
            if len(ids) > budget:
                stats["truncated"] += 1
                record = {**record, "passage": self.tok.decode(ids[:budget]).rstrip()}
            return record

        train = [fit(r) for r in pq.read_table(self.src("boolq", "train")).to_pylist()]
        val = [fit(r) for r in pq.read_table(self.src("boolq", "validation")).to_pylist()]
        parts = tasks.split_train(spec, train, 1000, SEED)
        extra = {"seed": SEED, "group_key": "passage", "development_examples": 1000,
                 "development_groups": parts["development_groups"],
                 "excluded_train_examples": len(parts["excluded"]),
                 "passage_tail_truncation": {**stats, "rate": stats["truncated"] / stats["examples"]}}
        self.choice_doc("boolq", {"train": [(i, train[i]) for i in parts["train"]],
                                  "development": [(i, train[i]) for i in parts["development"]],
                                  "test": list(enumerate(val))}, extra)

    def casehold(self):
        spec = tasks.TASKS["casehold"]
        out = {}
        dropped = {}
        for n in ("train", "validation", "test"):
            rows = pq.read_table(self.src("casehold", n)).to_pylist()
            out[n] = [r for r in rows if "<HOLDING>" in r["context"]]
            dropped[n] = len(rows) - len(out[n])
        parts = tasks.split_train(spec, out["validation"], 2000, SEED)
        extra = {"seed": SEED, "group_key": "context", "development_examples": 2000,
                 "development_source": "official validation", "dropped_without_holding_marker": dropped,
                 "source_note": "casehold/casehold is script-only; coastalcph/lex_glue case_hold parquet used"}
        self.choice_doc("casehold", {"train": list(enumerate(out["train"])),
                                     "development": [(i, out["validation"][i]) for i in parts["development"]],
                                     "test": list(enumerate(out["test"]))}, extra)

    # ---- probes -------------------------------------------------------------------
    def arc_easy(self):
        rows = pq.read_table(self.src("arc_easy", "test")).to_pylist()
        contexts, conts, chars, labels, example = [], [], [], [], []
        for e, r in enumerate(rows):
            query = f"Question: {r['question']}\nAnswer:"
            texts = list(r["choices"]["text"])
            labels.append(list(r["choices"]["label"]).index(r["answerKey"]))
            for k, text in enumerate(texts):
                head, tail = tasks.encode_pair(self.encode, query, " " + text)
                if k == 0:
                    contexts.append(head)
                conts.append(tail)
                chars.append(len(text))
                example.append(e)
        doc = {"schema": "domain_ft.variable_choice.v1", "task": "arc_easy", "bos": 2, "pad": 1,
               "contexts": contexts, "continuations": conts, "chars": chars, "example": example, "labels": labels}
        self.save("arc_easy-test-opt125m-dft1", doc, {"examples": len(rows), "rows": len(conts),
                                                       "template": "lm_eval_v0.4_arc_easy+bos",
                                                       "sources": self.src_receipt("arc_easy")})

    def lambada(self):
        rows = pq.read_table(self.src("lambada", "test")).to_pylist()
        pick = sorted(torch.randperm(len(rows), generator=torch.Generator().manual_seed(SEED))[:2000].tolist())
        contexts, targets = [], []
        for i in pick:
            text = rows[i]["text"]
            context, word = text.rsplit(" ", 1)
            head, tail = tasks.encode_pair(self.encode, context, " " + word)
            contexts.append(head)
            targets.append(tail)
        doc = {"schema": "domain_ft.lambada.v1", "bos": 2, "rows": pick, "contexts": contexts, "targets": targets}
        self.save("lambada-2000-opt125m-dft1", doc, {"examples": len(pick), "available": len(rows), "seed": SEED,
                                                     "template": "lm_eval_v0.4_lambada_openai+bos (greedy last word)",
                                                     "sources": self.src_receipt("lambada")})

    # ---- LM corpora ---------------------------------------------------------------
    def stream(self, texts: list[str]) -> torch.Tensor:
        ids = self.tok(texts, add_special_tokens=False)["input_ids"]
        flat = [2]
        for x in ids:
            flat.extend(x)
            flat.append(2)
        return torch.tensor(flat[:-1], dtype=torch.int64)

    @staticmethod
    def words(text: str) -> list[str]:
        text = re.sub(r"[0-9]+(?:[.,][0-9]+)*", " @ ", text.lower())
        return re.findall(r"[a-z@]+", text)

    def pubmed(self):
        def clean(a: str) -> str:
            return " ".join(a.replace("<S>", " ").replace("</S>", " ").split())

        # RCT test sentences with >= 10 words; matched on normalized words (lowercase, digits -> @,
        # punctuation dropped) because RCT replaces numbers with @ and the corpora tokenize differently.
        rct = self.jsonl(self.src("rct", "test"))
        index: dict[tuple, list[tuple]] = {}
        n_sent = 0
        for r in rct:
            if len(r["text"].split()) < 10:
                continue
            w = tuple(self.words(r["text"]))
            if len(w) < 6:
                continue
            n_sent += 1
            index.setdefault(w[:6], []).append(w)
        train, dropped = [], 0
        for i in range(5):
            for a in pq.read_table(self.src("pubmed", f"train{i}"), columns=["abstract"]).column(0).to_pylist():
                a = clean(a)
                if not a:
                    continue
                w = self.words(a)
                hit = False
                for p in range(len(w) - 5):
                    cands = index.get(tuple(w[p:p + 6]))
                    if cands and any(tuple(w[p:p + len(c)]) == c for c in cands):
                        hit = True
                        break
                if hit:
                    dropped += 1
                else:
                    train.append(a)
        val = [clean(a) for a in pq.read_table(self.src("pubmed", "validation"), columns=["abstract"]).column(0).to_pylist()]
        test = [clean(a) for a in pq.read_table(self.src("pubmed", "test"), columns=["abstract"]).column(0).to_pylist()]
        val, test = [a for a in val if a], [a for a in test if a]
        splits = {"train": self.stream(train), "validation": self.stream(val), "test": self.stream(test)}
        corpus = {"schema": CORPUS_SCHEMA, "model": MODEL,
                  "tokenizer": {**self.tok_receipt, "special_tokens": "BOS (2) at the start and between documents"},
                  "sources": self.src_receipt("pubmed"), "splits": splits}
        self.save("pubmed-opt125m-dft1", corpus, {
            "field": "abstract (sentence markers stripped, whitespace collapsed; source is lowercased and pre-tokenized)",
            "documents": {"train": len(train), "validation": len(val), "test": len(test)},
            "decontamination": {"rct_test_sentences_ge10_words": n_sent, "train_abstracts_dropped": dropped,
                                "match": "normalized word sequence (lowercase, digit runs -> @, punctuation dropped)"},
            "tokens": {k: len(v) for k, v in splits.items()},
            "blocks": {k: len(v) // MAX_LEN for k, v in splits.items()},
            "dev_blocks_used": 128, "test_blocks_used": 256, "sources": self.src_receipt("pubmed")})

    def pubmed2(self):
        """Natural-case PubMed abstracts from PubMedQA pqa_artificial (main decision 2026-11-07 22:40):
        abstract = " ".join(contexts) + " " + long_answer; split by pubid (seeded): test 2,000, dev 1,000."""
        rows = pq.read_table(self.src("pubmedqa", "artificial"), columns=["pubid", "context", "long_answer"]).to_pylist()
        by_id = {}
        for r in rows:
            by_id.setdefault(int(r["pubid"]), r)
        ids = sorted(by_id)
        order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(SEED)).tolist()
        test_ids = [ids[i] for i in order[:2000]]
        dev_ids = [ids[i] for i in order[2000:3000]]
        train_ids = [ids[i] for i in order[3000:]]

        def text(r):
            return " ".join((" ".join(r["context"]["contexts"]) + " " + (r["long_answer"] or "")).split())

        rct = self.jsonl(self.src("rct", "test"))
        index: dict[tuple, list[tuple]] = {}
        n_sent = 0
        for r in rct:
            if len(r["text"].split()) < 10:
                continue
            w = tuple(self.words(r["text"]))
            if len(w) < 6:
                continue
            n_sent += 1
            index.setdefault(w[:6], []).append(w)

        def contaminated(a):
            w = self.words(a)
            for p in range(len(w) - 5):
                cands = index.get(tuple(w[p:p + 6]))
                if cands and any(tuple(w[p:p + len(c)]) == c for c in cands):
                    return True
            return False

        train, dropped = [], 0
        for i in train_ids:
            a = text(by_id[i])
            if contaminated(a):
                dropped += 1
            else:
                train.append(a)
        val = [text(by_id[i]) for i in dev_ids]
        test = [text(by_id[i]) for i in test_ids]
        dev_cont = sum(contaminated(a) for a in val)
        test_cont = sum(contaminated(a) for a in test)
        splits = {"train": self.stream(train), "validation": self.stream(val), "test": self.stream(test)}
        corpus = {"schema": CORPUS_SCHEMA, "model": MODEL,
                  "tokenizer": {**self.tok_receipt, "special_tokens": "BOS (2) at the start and between documents"},
                  "sources": self.src_receipt("pubmedqa"), "splits": splits}
        self.save("pubmed-opt125m-dft2", corpus, {
            "field": "PubMedQA pqa_artificial: ' '.join(context.contexts) + ' ' + long_answer (whitespace collapsed; natural case)",
            "split": "by pubid (seeded 20261108): test 2,000, dev 1,000, rest train",
            "rows": len(rows), "unique_pubids": len(ids),
            "documents": {"train": len(train), "validation": len(val), "test": len(test)},
            "decontamination": {"rct_test_sentences_ge10_words": n_sent, "train_abstracts_dropped": dropped,
                                "dev_abstracts_matching_(kept)": dev_cont, "test_abstracts_matching_(kept)": test_cont,
                                "match": "normalized word sequence (lowercase, digit runs -> @, punctuation dropped)"},
            "tokens": {k: len(v) for k, v in splits.items()},
            "blocks": {k: len(v) // MAX_LEN for k, v in splits.items()},
            "dev_blocks_used": 128, "test_blocks_used": 256, "sources": self.src_receipt("pubmedqa")})

    def code(self):
        repos: dict[str, list[str]] = {}
        licenses: dict[str, int] = {}
        with gzip.open(self.src("code", "all"), "rt") as f:
            for line in f:
                r = json.loads(line)
                repos.setdefault(r["repo_name"], []).append(r["content"])
                licenses[r["license"]] = licenses.get(r["license"], 0) + 1
        names = sorted(repos)
        order = torch.randperm(len(names), generator=torch.Generator().manual_seed(SEED)).tolist()
        # Dev/test: one seeded file per repository, first 1,024 tokens each, so 128/256 blocks span
        # many repositories. Train: whole files of the remaining repositories until >= 20M tokens.
        targets = {"test": 256 * MAX_LEN + 8192, "validation": 128 * MAX_LEN + 8192, "train": 20_000_000}
        assign = {k: [] for k in targets}
        tokens = {k: 0 for k in targets}
        texts = {k: [] for k in targets}
        current = iter(("test", "validation", "train"))
        split = next(current)
        g = torch.Generator().manual_seed(SEED + 1)
        for i in order:
            files = repos[names[i]]
            if split != "train":
                pick = files[int(torch.randint(len(files), (1,), generator=g))]
                ids = self.tok(pick, add_special_tokens=False)["input_ids"][:1024]
                if len(ids) < 64:
                    continue
                files = [self.tok.decode(ids)]
                ids = [ids]
            else:
                ids = self.tok(files, add_special_tokens=False)["input_ids"]
            assign[split].append(names[i])
            texts[split] += files
            tokens[split] += sum(len(x) + 1 for x in ids)
            if tokens[split] >= targets[split]:
                split = next(current, None)
                if split is None:
                    break
        splits = {k: self.stream(texts[k]) for k in ("train", "validation", "test")}
        corpus = {"schema": CORPUS_SCHEMA, "model": MODEL,
                  "tokenizer": {**self.tok_receipt, "special_tokens": "BOS (2) at the start and between files"},
                  "sources": self.src_receipt("code"), "splits": splits}
        self.save("code-opt125m-dft1", corpus, {
            "split": "by repository (seeded shuffle of repo names; test, then dev (one seeded file per repo, first 1,024 tokens), then train whole files until >= 20M tokens)",
            "repos": {k: len(v) for k, v in assign.items()}, "repos_available": len(names),
            "files": {k: len(v) for k, v in texts.items()}, "licenses_all_files": licenses,
            "repo_overlap": len(set(assign["train"]) & (set(assign["test"]) | set(assign["validation"]))),
            "tokens": {k: len(v) for k, v in splits.items()},
            "blocks": {k: len(v) // MAX_LEN for k, v in splits.items()},
            "dev_blocks_used": 128, "test_blocks_used": 256, "sources": self.src_receipt("code")})


def main():
    root, snapshot = Path(sys.argv[1]), Path(sys.argv[2])
    keys = sys.argv[4:] or ["boolq", "casehold", "arc_easy", "lambada", "mnli", "rct", "code", "pubmed"]
    b = Builder(root, snapshot)
    for k in keys:
        t = time.time()
        getattr(b, k)()
        print(k, "done", round(time.time() - t, 1), "s", flush=True)


if __name__ == "__main__":
    main()

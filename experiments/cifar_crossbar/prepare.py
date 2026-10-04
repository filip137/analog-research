"""Download public inputs without executing remote model code."""

from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "cifar10": (
        "cifar10_resnet32-ef93fc4d.pt",
        "ef93fc4d3ea83c08d75bfd6fc31af56047cfa241608c9ba3a1f6dc11cca20dd9",
        93.53,
    ),
    "cifar100": (
        "cifar100_resnet32-84213ce6.pt",
        "84213ce69c556d1e3b983dc13d154a92d4132113592eae55215a656f62bfb896",
        70.16,
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--datasets", nargs="+", choices=tuple(SOURCES), default=list(SOURCES)
    )
    p.add_argument("--download-data", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "sources.json"
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for dataset in args.datasets:
        filename, prefix, accuracy = SOURCES[dataset]
        url = (
            "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/"
            + filename
        )
        path = args.output / filename
        if not path.exists():
            temporary = path.with_suffix(".download")
            urllib.request.urlretrieve(url, temporary)
            digest = sha256(temporary.read_bytes()).hexdigest()
            if not digest.startswith(prefix):
                raise ValueError(
                    f"Expected checkpoint SHA-256 prefix {prefix}; got {digest}."
                )
            temporary.replace(path)
        digest = sha256(path.read_bytes()).hexdigest()
        if not digest.startswith(prefix):
            raise ValueError("Expected the pinned published checkpoint digest prefix.")
        receipts[dataset] = {
            "file": filename,
            "url": url,
            "sha256": digest,
            "reported_accuracy_percent": accuracy,
            "architecture": "resnet32_16_32_64_projection",
            "model_source_revision": "786c162",
            "source": "https://github.com/chenyaofo/pytorch-cifar-models",
        }
        if args.download_data:
            from torchvision.datasets import CIFAR10, CIFAR100

            cls = CIFAR10 if dataset == "cifar10" else CIFAR100
            for train in (True, False):
                cls(str(args.output / "data"), train=train, download=True)
        print(json.dumps({dataset: receipts[dataset]}), flush=True)
    (args.output / "sources.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()

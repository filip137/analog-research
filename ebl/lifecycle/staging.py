"""Prepare a minimal, verifiable source payload; never copy it to a host."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from ebl.runtime.store import atomic_write_json


def snapshot(output,extra_files=()):
    root=Path(__file__).resolve().parents[1]
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Expected a new, empty snapshot destination.')
    for name in ('ebl.cli','ebl.runtime.opt_cmo_drn.runtime','ebl.runtime.opt_cmo_drn.recovery',
                 'ebl.lifecycle.study','ebl.lifecycle.__main__','ebl.lifecycle.tuning','ebl.lifecycle.recovery'):
        importlib.import_module(name)
    files=set()
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.is_relative_to(root) and path.suffix=='.py':files.add(path.relative_to(root))
    files.add(Path('ebl/__main__.py'))
    for filename in extra_files:
        path=Path(filename).resolve()
        if not path.is_relative_to(root) or path.suffix!='.json':
            raise ValueError('Only explicit in-worktree JSON configs can supplement the Python dependency closure.')
        files.add(path.relative_to(root))
    digests={}
    for relative in sorted(files):
        target=output/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,target)
        digests[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    receipt={'schema':'opt_cmo_drn.source.v1','commit':commit,'files':digests,
             'bytes':sum((output/name).stat().st_size for name in digests)}
    atomic_write_json(output/'source-snapshot.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,action='append',default=[])
    args=parser.parse_args()
    receipt=snapshot(args.output,args.config)
    print(json.dumps({'files':len(receipt['files']),'bytes':receipt['bytes'],'path':str(args.output.resolve())},indent=2))


if __name__=='__main__':main()

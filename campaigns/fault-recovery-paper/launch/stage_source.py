"""Make a Jean Zay staging pair for an hwa commit: a git bundle of the branch and a source receipt.

Usage: python stage_source.py <hwa worktree> <branch> <out dir>
The receipt lists the sha256 of every tracked regular file (symlinks skipped) at the
worktree's HEAD, which must be clean. The launch worker checks it before running.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    worktree, branch, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    git = ['git', '-C', str(worktree)]
    if subprocess.run(git + ['status', '--porcelain'], capture_output=True, text=True, check=True).stdout.strip():
        raise SystemExit('The worktree is not clean.')
    sha = subprocess.run(git + ['rev-parse', '--short', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(git + ['bundle', 'create', str(out / f'hwa-{sha}.bundle'), branch], check=True, capture_output=True)
    names = subprocess.run(git + ['ls-files', '-z'], capture_output=True, check=True).stdout.decode().split('\0')
    files = {}
    for name in filter(None, names):
        path = worktree / name
        if path.is_symlink() or not path.is_file():
            continue
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {'schema': 'fault_tuning.source_snapshot.v1', 'origin_worktree': str(worktree),
               'origin_commit': sha, 'files': files}
    (out / 'source_snapshot.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'commit': sha, 'files': len(files), 'out': str(out)}))


if __name__ == '__main__':
    main()

"""Write source_snapshot.json (sha256 of every regular file) for a ``git archive`` export.

Usage: python3 make_receipt.py <exported source dir> <commit>
The runtime checks this receipt when OPT_DRN_SOURCE_SNAPSHOT names it.
"""
import hashlib
import json
from pathlib import Path
import sys

root, commit = Path(sys.argv[1]).resolve(), sys.argv[2]
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(root.rglob('*')) if p.is_file() and not p.is_symlink() and p.name != 'source_snapshot.json'}
(root / 'source_snapshot.json').write_text(json.dumps({'schema': 'drn.source_snapshot.v1', 'origin_commit': commit,
                                                       'files': files}, indent=1) + '\n')
print(json.dumps({'commit': commit, 'files': len(files)}))

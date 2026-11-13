#!/bin/bash -l
# On Jean Zay: clone a staged hwa bundle into source-<sha>/hwa, detach at <sha>, install and verify the receipt.
# Usage: bash jz_setup_source.sh <staging root> <sha>
set -euo pipefail
root=$1
sha=$2
module load git/2.39.1 >/dev/null 2>&1
cd "$root"
test ! -e "source-$sha"
mkdir "source-$sha"
git clone -q "hwa-$sha.bundle" "source-$sha/hwa" 2>/dev/null || true
cd "source-$sha/hwa"
git checkout -q --detach "$sha"
mv "$root/launch-$sha/source_snapshot.json" .
echo source_snapshot.json >> .git/info/exclude
test -z "$(git status --porcelain)"
python3 - <<'EOF'
import hashlib, json
d = json.load(open('source_snapshot.json'))
bad = [f for f, h in d['files'].items() if hashlib.sha256(open(f, 'rb').read()).hexdigest() != h]
print(json.dumps({'commit': d['origin_commit'], 'files': len(d['files']), 'mismatch': len(bad)}))
assert not bad
EOF

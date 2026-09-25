#!/usr/bin/env python3
"""Validate campaign Markdown and generate a deterministic navigation ledger.

No third-party dependencies, scientific execution, remote reads, or inference of
hypothesis verdicts. Frontmatter is a deliberately small YAML-compatible subset:
each nonempty line contains a key and a JSON value.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

STATES = {'planned', 'blocked', 'ready', 'running', 'partial', 'complete',
          'failed', 'cancelled', 'superseded', 'imported'}
NEEDS_RESULT = {'partial', 'complete', 'failed', 'imported'}
EVIDENCE = {'imported-summary', 'validated-local'}
VERDICTS = {'supports', 'contradicts', 'inconclusive', 'not-tested'}


class CampaignError(ValueError):
    """An actionable structural or metadata error."""


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise CampaignError(f'{path}: {exc}') from exc


def metadata(path: Path, expected: set[str]) -> dict[str, Any]:
    lines = read_text(path).splitlines()
    if not lines or lines[0] != '---':
        raise CampaignError(f'{path}: missing frontmatter')
    try:
        end = lines.index('---', 1)
    except ValueError as exc:
        raise CampaignError(f'{path}: unclosed frontmatter') from exc
    out: dict[str, Any] = {}
    for number, line in enumerate(lines[1:end], 2):
        if not line.strip():
            continue
        key, sep, value = line.partition(':')
        key = key.strip()
        if not sep or not re.fullmatch(r'[a-z_]+', key) or key in out:
            raise CampaignError(f'{path}:{number}: invalid/duplicate metadata key')
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CampaignError(f'{path}:{number}: values must be single-line JSON') from exc
    if set(out) != expected:
        raise CampaignError(f'{path}: expected keys {sorted(expected)}, got {sorted(out)}')
    return out


def text_value(meta: dict[str, Any], key: str, path: Path) -> str:
    value = meta[key]
    if not isinstance(value, str) or not value.strip() or '\n' in value or '\r' in value:
        raise CampaignError(f'{path}: {key} must be nonempty single-line text')
    return value


def record_id(meta: dict[str, Any], prefix: str, path: Path) -> str:
    value = text_value(meta, 'id', path)
    if not re.fullmatch(re.escape(prefix) + r'-[0-9]{3}', value) or value.endswith('-000'):
        raise CampaignError(f'{path}: expected positive three-digit {prefix} ID')
    if not path.name.startswith(value + '-'):
        raise CampaignError(f'{path}: filename must begin {value}-')
    text_value(meta, 'title', path)
    return value


def cell(value: str) -> str:
    return value.replace('\\', '\\\\').replace('|', '\\|').replace('\n', ' ')


def mdlink(root: Path, path: Path, label: str) -> str:
    return f'[{cell(label)}]({path.relative_to(root).as_posix()})'


def render(root: Path) -> str:
    """Validate one campaign and return its ledger without changing any files."""
    if not root.is_dir():
        raise CampaignError(f'{root}: campaign directory missing')
    for name in ('README.md',):
        if not read_text(root / name).strip():
            raise CampaignError(f'{root / name}: empty required document')
    # Reject links out of the campaign in file discovery; ordinary Markdown links
    # to historical evidence are intentionally not opened by this program.
    for path in root.rglob('*'):
        if path.is_symlink():
            raise CampaignError(f'{path}: symlink not allowed in campaign artifacts')
    xfiles = sorted((root / 'explorations').glob('*.md'))
    xids: set[str] = set()
    for path in xfiles:
        match = re.match(r'(X-[0-9]{3})-.+\.md$', path.name)
        if not match or match[1] == 'X-000' or match[1] in xids:
            raise CampaignError(f'{path}: invalid/duplicate exploration ID')
        xids.add(match[1])
        if not read_text(path).strip():
            raise CampaignError(f'{path}: empty exploration')
    hypotheses: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted((root / 'hypotheses').glob('*.md')):
        meta = metadata(path, {'id', 'title'})
        hid = record_id(meta, 'H', path)
        if hid in hypotheses:
            raise CampaignError(f'{path}: duplicate hypothesis {hid}')
        hypotheses[hid] = (path, meta)
    board = root / 'ideas.md'
    if board.exists():
        board_ids = set(re.findall(r'\bH-[0-9]{3}\b', read_text(board)))
        if not board_ids <= set(hypotheses):
            raise CampaignError(f'{board}: unknown hypothesis reference')
    series = sorted(path for path in (root / 'series').glob('*') if path.is_dir())
    experiments: dict[str, tuple[Path, dict[str, Any]]] = {}
    result_paths: list[Path] = []
    series_ids: set[str] = set()
    for folder in series:
        match = re.match(r'([0-9]{3})-.+$', folder.name)
        if not match or match[1] == '000' or match[1] in series_ids:
            raise CampaignError(f'{folder}: invalid/duplicate series ID')
        series_ids.add(match[1])
        if not read_text(folder / 'README.md').strip():
            raise CampaignError(f'{folder}: empty series README')
        for name in ('experiments', 'results'):
            if not (folder / name).is_dir():
                raise CampaignError(f'{folder}: missing {name}/')
        for path in sorted((folder / 'experiments').glob('*.md')):
            meta = metadata(path, {'id', 'title', 'status', 'hypotheses'})
            eid = record_id(meta, 'exp', path)
            if eid in experiments:
                raise CampaignError(f'{path}: duplicate experiment {eid} across series')
            if text_value(meta, 'status', path) not in STATES:
                raise CampaignError(f'{path}: invalid status')
            hs = meta['hypotheses']
            if (not isinstance(hs, list) or not all(isinstance(h, str) for h in hs)
                    or len(hs) != len(set(hs)) or not set(hs) <= set(hypotheses)):
                raise CampaignError(f'{path}: invalid/unknown/duplicate hypothesis reference')
            experiments[eid] = (path, meta)
        result_paths.extend(sorted((folder / 'results').glob('*.md')))
    results: dict[str, list[tuple[Path, dict[str, Any]]]] = {eid: [] for eid in experiments}
    for path in result_paths:
        meta = metadata(path, {'experiment', 'evidence', 'summary', 'verdicts'})
        eid = text_value(meta, 'experiment', path)
        if eid not in experiments:
            raise CampaignError(f'{path}: unknown experiment {eid}')
        epath, exp = experiments[eid]
        if path.parent.parent != epath.parent.parent:
            raise CampaignError(f'{path}: result and experiment must be in the same series')
        if text_value(meta, 'evidence', path) not in EVIDENCE:
            raise CampaignError(f'{path}: invalid evidence class')
        text_value(meta, 'summary', path)
        verdicts = meta['verdicts']
        if (not isinstance(verdicts, dict) or not set(verdicts) <= set(exp['hypotheses'])
                or not all(isinstance(v, str) and v in VERDICTS for v in verdicts.values())):
            raise CampaignError(f'{path}: invalid verdict or undeclared hypothesis')
        if exp['status'] == 'imported' and meta['evidence'] != 'imported-summary':
            raise CampaignError(f'{path}: imported experiment cannot claim fresh validation')
        results[eid].append((path, meta))
    for eid, (path, meta) in experiments.items():
        if meta['status'] in NEEDS_RESULT and not results[eid]:
            raise CampaignError(f'{path}: {meta["status"]} needs a result note')
    counts = Counter(meta['status'] for _, meta in experiments.values())
    lines = ['# Campaign ledger', '', '<!-- GENERATED by experiment-loop/scripts/ledger.py; do not edit. -->', '',
             'Sources: campaign experiment metadata and reviewer-written result notes. '
             'No raw evidence or live job state is inferred.', '',
             'Counts: ' + ', '.join(f'{state}={count}' for state, count in sorted(counts.items())) + '.', '',
             '## Experiments', '', '| Experiment | State | Hypotheses | Recorded observations |',
             '|---|---|---|---|']
    for eid, (path, meta) in sorted(experiments.items()):
        hlinks = ', '.join(mdlink(root, hypotheses[h][0], h) for h in meta['hypotheses']) or 'None declared'
        obs = '<br>'.join(mdlink(root, rp, rm['evidence']) + ': ' + cell(rm['summary'])
                         for rp, rm in results[eid]) or 'No result recorded.'
        lines.append(f'| {mdlink(root, path, eid + " — " + meta["title"])} | {meta["status"]} | {hlinks} | {obs} |')
    lines += ['', '## Hypothesis evidence', '',
              'Verdicts below are scoped judgments from individual notes, not a generated global status. '
              'Imports are retrospective; conflicting notes remain visible.', '',
              '| Hypothesis | Result-note judgments |', '|---|---|']
    for hid, (path, meta) in sorted(hypotheses.items()):
        judgments = [mdlink(root, rp, eid) + f': {rm["verdicts"][hid]} ({rm["evidence"]})'
                     for eid in sorted(results) for rp, rm in results[eid] if hid in rm['verdicts']]
        lines.append(f'| {mdlink(root, path, hid + " — " + meta["title"])} | '
                     + ('<br>'.join(judgments) or 'No result-note judgment recorded.') + ' |')
    lines += ['', '## Refresh', '',
              'From the repository root, run the experiment-loop ledger script on this campaign path. '
              'Use `--check` to detect stale output without writing. The checker validates structure, '
              'not scientific truth or the underlying run bundles.', '']
    return '\n'.join(lines)


def write_atomic(path: Path, content: str) -> None:
    if path.is_symlink():
        raise CampaignError(f'{path}: refusing to replace symlink')
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix='.ledger-', delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(content)
        tmp.chmod(0o644)
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate and fail on stale/missing ledger; do not write')
    parser.add_argument('campaigns', nargs='+', type=Path)
    args = parser.parse_args(argv)
    try:
        # Validate all requested campaigns before writing any of them.
        rendered = [(root, render(root)) for root in args.campaigns]
        stale = []
        for root, content in rendered:
            target = root / 'ledger.md'
            if args.check:
                if not target.exists() or read_text(target) != content:
                    stale.append(str(target))
            elif not target.exists() or read_text(target) != content:
                write_atomic(target, content)
        if stale:
            print('Stale or missing ledger(s):\n' + '\n'.join(stale), file=sys.stderr)
            return 1
        print(('Checked' if args.check else 'Generated') + f' {len(rendered)} campaign ledger(s).')
        return 0
    except (CampaignError, OSError) as exc:
        print(f'Campaign error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

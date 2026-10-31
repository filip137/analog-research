#!/usr/bin/env python3
"""Semi-autonomous experiment loop. Each tick advances every experiment by at
most one step and is safe to rerun:

    authorized note on base, no run  -> launch (envelope and provenance checks)
    run finished, no results PR      -> validity guards + fresh reviewer -> results PR
    all results merged, no proposal  -> fresh proposer -> proposal PR
    proposal PR merged by APPROVER   -> its note becomes launchable

Agents only read and answer in a fixed schema. This script writes files, opens
PRs and launches runs. It never merges proposal PRs; that merge is the launch
authorization.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get('LOOP_BASE', 'loop-test')
RUNS = Path(os.environ.get('LOOP_RUNS', ROOT.parent / 'loop-runs'))
APPROVER = os.environ.get('LOOP_APPROVER', 'filip137')
MODEL = os.environ.get('LOOP_MODEL')
AGENT_BUDGET_USD = os.environ.get('LOOP_AGENT_BUDGET_USD', '1.00')
AUTOMERGE_RESULTS = os.environ.get('LOOP_AUTOMERGE_RESULTS', '1') == '1'
TRAILER = 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
PR_FOOTER = '🤖 Generated with [Claude Code](https://claude.com/claude-code)'


# ---------------------------------------------------------------- plumbing

# A locked GNOME keyring makes gh wait forever for an unlock prompt over SSH.
# Unsetting the address is not enough: D-Bus clients fall back to
# $XDG_RUNTIME_DIR/bus, so point them at a socket that does not exist.
ENV = {**os.environ, 'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/nonexistent'}


def sh(*argv, cwd=ROOT, input=None):
    p = subprocess.run([str(a) for a in argv], cwd=cwd, input=input, text=True, capture_output=True, env=ENV)
    if p.returncode:
        raise RuntimeError(f'{" ".join(map(str, argv[:3]))} failed: {p.stderr.strip()[-2000:]}')
    return p.stdout.strip()


def git(*argv, cwd=ROOT):
    return sh('git', *argv, cwd=cwd)


def repo():
    url = git('remote', 'get-url', 'origin')
    return re.search(r'github\.com[:/](.+?)(?:\.git)?$', url).group(1)


def gh(*argv):
    return sh('gh', *argv, '-R', repo())


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_note(path):
    _, front, body = Path(path).read_text().split('---\n', 2)
    return yaml.safe_load(front), body


def notes(campaign):
    return sorted((campaign / 'experiments').glob('exp-*.md'))


def result_path(campaign, eid):
    return campaign / 'results' / eid / 'result.md'


def prs(branch):
    return json.loads(gh('pr', 'list', '--state', 'all', '--head', branch, '--json', 'number,state,url,mergedBy'))


def open_pr(branch, files, title, body):
    """Commit files on top of origin/BASE in a throwaway worktree and open a PR."""
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / 'wt'
        git('worktree', 'add', '--detach', wt, f'origin/{BASE}')
        try:
            for name, text in files.items():
                (wt / name).parent.mkdir(parents=True, exist_ok=True)
                (wt / name).write_text(text)
            git('add', '-A', cwd=wt)
            git('-c', 'user.name=experiment-loop', 'commit', '-m', f'{title}\n\n{TRAILER}', cwd=wt)
            git('push', 'origin', f'HEAD:refs/heads/{branch}', cwd=wt)
        finally:
            git('worktree', 'remove', '--force', wt)
    return gh('pr', 'create', '--base', BASE, '--head', branch, '--title', title,
              '--body', f'{body}\n\n{PR_FOOTER}')


# ---------------------------------------------------------------- envelope

def check_value(value, spec, path):
    if value is None:
        return [] if spec.get('nullable') else [f'{path}: required']
    if spec['type'] == 'list':
        if not isinstance(value, list):
            return [f'{path}: expected a list']
        lo, hi = spec.get('min_len', 0), spec.get('max_len', math.inf)
        errs = [] if lo <= len(value) <= hi else [f'{path}: length {len(value)} outside [{lo}, {hi}]']
        for i, item in enumerate(value):
            errs += check_fields(item, spec['items'], f'{path}[{i}]')
        return errs
    ok = {'int': lambda v: isinstance(v, int) and not isinstance(v, bool),
          'float': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
          'str': lambda v: isinstance(v, str)}[spec['type']](value)
    if not ok:
        return [f'{path}: expected {spec["type"]}, got {value!r}']
    if ('min' in spec and value < spec['min']) or ('max' in spec and value > spec['max']):
        return [f'{path}: {value} outside [{spec.get("min")}, {spec.get("max")}]']
    return []


def check_fields(obj, fields, path='config'):
    if not isinstance(obj, dict):
        return [f'{path}: expected an object']
    errs = [f'{path}.{k}: not allowed by the envelope' for k in obj if k not in fields]
    for k, spec in fields.items():
        errs += check_value(obj.get(k), spec, f'{path}.{k}')
    return errs


# ---------------------------------------------------------------- agents

REVIEW_SCHEMA = {
    'type': 'object', 'required': ['verdict', 'rule_applied', 'summary', 'concerns'],
    'properties': {
        'verdict': {'enum': ['supported', 'contradicted', 'inconclusive', 'invalid']},
        'rule_applied': {'type': 'string', 'description': 'The decision rule computed step by step with the numbers used.'},
        'summary': {'type': 'string', 'description': 'Two to four sentences for the result note.'},
        'concerns': {'type': 'string', 'description': 'Validity or interpretation concerns; "none" if none.'}}}

REVIEWER = """You are the Reviewer for experiment {eid}. The packet on stdin holds the
experiment note exactly as it was when the run launched, and the run's metrics.
Apply the note's decision rule to the metrics exactly as written. Do not relax,
reinterpret or add criteria. If the rule cannot be applied as written, answer
inconclusive or invalid and say why. Answer only from the packet."""

PROPOSE_SCHEMA = {
    'type': 'object', 'required': ['action', 'rationale'],
    'properties': {
        'action': {'enum': ['next_experiment', 'stop', 'ask_human']},
        'rationale': {'type': 'string'},
        'experiment': {
            'type': 'object',
            'required': ['slug', 'title', 'question', 'hypothesis', 'config', 'decision_rule', 'outcomes'],
            'properties': {
                'slug': {'type': 'string'}, 'title': {'type': 'string'},
                'question': {'type': 'string'}, 'hypothesis': {'type': 'string'},
                'config': {'type': 'object', 'description': 'Runner config; must satisfy envelope.json params.'},
                'decision_rule': {'type': 'string', 'description': 'Markdown; metric, comparison and thresholds giving supported/contradicted/inconclusive.'},
                'outcomes': {'type': 'string', 'description': 'Markdown; what each verdict would mean for the campaign.'}}}}}

PROPOSER = """You are the Explorer for the campaign in the packet on stdin: its README,
envelope, experiment notes, result notes and earlier proposals. Choose one:
- next_experiment: the single most informative next experiment. Its config must
  satisfy envelope.json. Give a falsifiable hypothesis and a decision rule fixed
  before data, with an explicit metric, comparison and thresholds. Do not repeat
  a finished experiment unless it is a confirmation with new seeds.
- stop: the README's question is answered for this envelope, or no allowed
  experiment would change the conclusion.
- ask_human: the next useful step needs something outside the envelope, such
  as new code, a different device model or more budget.
You may read files under the working directory, but base your judgment on the
recorded results."""


def ask(instruction, packet, schema):
    text = '\n\n'.join(f'===== {name} =====\n{body}' for name, body in packet.items())
    argv = ['claude', '-p', instruction, '--output-format', 'json', '--json-schema', json.dumps(schema),
            '--tools', 'Read,Grep,Glob', '--strict-mcp-config', '--no-session-persistence',
            '--max-budget-usd', AGENT_BUDGET_USD]
    if MODEL:
        argv += ['--model', MODEL]
    out = json.loads(sh(*argv, input=text))
    if out.get('is_error') or out.get('structured_output') is None:
        raise RuntimeError(f'agent failed: {out.get("subtype")} {str(out.get("result"))[:500]}')
    answer = out['structured_output']
    answer['_cost_usd'] = out.get('total_cost_usd')
    return answer


# ---------------------------------------------------------------- steps

def authorization(campaign, path):
    meta, _ = read_note(path)
    if meta.get('proposed_by') != 'loop':
        who = meta.get('authorized_by')
        return (True, f'authorized_by: {who}') if who else (False, 'human-written note has no authorized_by')
    merged = [p for p in prs(f'proposal/{campaign.name}/after-{meta["follows"]}') if p['state'] == 'MERGED']
    if not merged:
        return False, 'proposal PR not merged yet'
    by = (merged[0].get('mergedBy') or {}).get('login')
    return by == APPROVER, f'proposal PR #{merged[0]["number"]} merged by {by}'


def launch(campaign, path, envelope, why):
    meta, _ = read_note(path)
    errs = check_fields(meta['config'], envelope['params'])
    if errs:
        return 'waiting', f'launch refused, envelope: {"; ".join(errs)}'
    if git('status', '--porcelain'):
        return 'waiting', 'launch refused: worktree is dirty'
    run = RUNS / campaign.name / meta['id']
    run.mkdir(parents=True)
    shutil.copy(campaign / envelope['runner'], run / 'runner.py')
    (run / 'config.json').write_text(json.dumps(meta['config'], indent=1))
    script = (f'timeout {envelope["max_seconds"]} {sys.executable} runner.py --config config.json --out . '
              f'> stdout.log 2>&1; echo "{{\\"exit\\": $?}}" > done.json')
    proc = subprocess.Popen(['bash', '-c', script], cwd=run, start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    record = {'experiment': meta['id'], 'note': rel(path), 'note_sha256': sha256(path),
              'commit': git('rev-parse', 'HEAD'), 'authorization': why, 'pid': proc.pid,
              'python': sys.executable, 'started_utc': now()}
    (run / 'launch.json').write_text(json.dumps(record, indent=1))
    return 'acted', f'launched in {run} ({why})'


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def review(campaign, path, run):
    meta, _ = read_note(path)
    launched = json.loads((run / 'launch.json').read_text())
    done = json.loads((run / 'done.json').read_text())
    metrics = (run / 'metrics.json').read_text() if (run / 'metrics.json').exists() else None
    guards = {'exit code 0': done['exit'] == 0,
              'metrics.json present': metrics is not None,
              'metrics finite': metrics is not None and finite(json.loads(metrics)),
              'note unchanged since launch': sha256(path) == launched['note_sha256']}
    passed = all(guards.values())
    if passed:
        frozen = git('show', f'{launched["commit"]}:{launched["note"]}')
        answer = ask(REVIEWER.format(eid=meta['id']),
                     {'experiment note (as launched)': frozen, 'metrics.json': metrics}, REVIEW_SCHEMA)
    else:
        log = (run / 'stdout.log').read_text()[-1500:] if (run / 'stdout.log').exists() else ''
        answer = {'verdict': 'invalid', 'rule_applied': 'Not applied: a validity guard failed.',
                  'summary': 'The run did not produce reviewable evidence.',
                  'concerns': f'```\n{log}\n```' if log else 'No log.', '_cost_usd': 0}
    guard_lines = '\n'.join(f'- {"pass" if ok else "**FAIL**"}: {name}' for name, ok in guards.items())
    result = f"""# {meta['id']} result: {answer['verdict']}

Experiment: [{meta['title']}](../../experiments/{path.name})

## Summary

{answer['summary']}

## Decision rule applied

{answer['rule_applied']}

## Concerns

{answer['concerns']}

## Validity guards

{guard_lines}

## Provenance

- Launch commit: `{launched['commit']}`; note sha256 `{launched['note_sha256'][:16]}…`
- Authorization: {launched['authorization']}
- Run directory: `{run}`
- Reviewer: fresh `claude -p`, read-only tools, cost ${answer['_cost_usd'] or 0:.3f}; reviewed {now()}
"""
    out = f'{rel(campaign)}/results/{meta["id"]}'
    files = {f'{out}/result.md': result, f'{out}/metrics.json': metrics or '{}\n'}
    title = f'{campaign.name} {meta["id"]} results: {answer["verdict"]}'
    body = f'{answer["summary"]}\n\n**Guards**\n{guard_lines}'
    url = open_pr(f'results/{campaign.name}/{meta["id"]}', files, title, body)
    if AUTOMERGE_RESULTS and passed and answer['verdict'] != 'invalid':
        gh('pr', 'merge', url, '--merge', '--delete-branch')
        return 'acted', f'reviewed ({answer["verdict"]}), results PR merged: {url}'
    return 'acted', f'reviewed ({answer["verdict"]}), results PR needs you: {url}'


def propose(campaign, envelope, last):
    existing = notes(campaign)
    if len(existing) >= envelope['max_experiments']:
        answer = {'action': 'ask_human', '_cost_usd': 0,
                  'rationale': f'Checkpoint: {len(existing)} experiments reached the envelope limit of '
                               f'{envelope["max_experiments"]} before human review.'}
    else:
        sources = [campaign / 'README.md', campaign / 'envelope.json', *existing,
                   *sorted(campaign.glob('results/*/result.md')), *sorted(campaign.glob('results/*/metrics.json')),
                   *sorted(campaign.glob('proposals/*.md'))]
        packet = {rel(p): p.read_text() for p in sources}

        def envelope_errors(a):
            if a['action'] != 'next_experiment':
                return []
            return check_fields(a.get('experiment', {}).get('config'), envelope['params'])

        answer = ask(PROPOSER, packet, PROPOSE_SCHEMA)
        if errs := envelope_errors(answer):
            packet['envelope check of your previous answer'] = '\n'.join(errs)
            answer = ask(PROPOSER, packet, PROPOSE_SCHEMA)
            errs = envelope_errors(answer)
        if errs:
            answer = {'action': 'ask_human', '_cost_usd': answer['_cost_usd'],
                      'rationale': 'The proposed config failed the envelope twice:\n' + '\n'.join(f'- {e}' for e in errs)}
    files = {}
    head = f'# Proposal after {last}: {answer["action"]}\n\n{answer["rationale"]}\n'
    if answer['action'] == 'next_experiment':
        x = answer['experiment']
        eid = f'exp-{len(existing) + 1:03d}'
        front = {'id': eid, 'title': x['title'], 'proposed_by': 'loop', 'follows': last,
                 'proposed_utc': now(), 'config': x['config']}
        files[f'{rel(campaign)}/experiments/{eid}.md'] = (
            f'---\n{yaml.safe_dump(front, sort_keys=False)}---\n## Question\n\n{x["question"]}\n\n'
            f'## Hypothesis\n\n{x["hypothesis"]}\n\n## Decision rule (fixed before data)\n\n{x["decision_rule"]}\n\n'
            f'## Outcomes\n\n{x["outcomes"]}\n')
        head += f'\nProposed experiment: [{eid}](../experiments/{eid}.md)\n'
        title = f'{campaign.name} proposal after {last}: {eid} {x["title"]}'
        gate = 'Envelope check: pass. **Merging this PR authorizes the loop to launch it.** Edit the note in this PR first if you want changes.'
    else:
        title = f'{campaign.name} after {last}: {answer["action"]}'
        gate = 'Merging records this decision; nothing launches.'
    files[f'{rel(campaign)}/proposals/after-{last}.md'] = head + f'\nProposer cost ${answer["_cost_usd"] or 0:.3f}, {now()}\n'
    url = open_pr(f'proposal/{campaign.name}/after-{last}', files, title, f'{answer["rationale"]}\n\n{gate}')
    return 'acted', f'proposal ({answer["action"]}) needs you: {url}'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def tick():
    if git('rev-parse', '--abbrev-ref', 'HEAD') != BASE:
        raise SystemExit(f'{ROOT} must be on {BASE}')
    git('fetch', '--prune', 'origin')
    git('merge', '--ff-only', f'origin/{BASE}')
    events = []
    for campaign in sorted(p.parent for p in ROOT.glob('campaigns/*/envelope.json')):
        envelope = json.loads((campaign / 'envelope.json').read_text())
        finished, pending = [], []
        for path in notes(campaign):
            eid = read_note(path)[0]['id']
            run = RUNS / campaign.name / eid
            tag = f'{campaign.name}/{eid}'
            if result_path(campaign, eid).exists():
                finished.append(eid)
                continue
            pending.append(eid)
            if not run.exists():
                ok, why = authorization(campaign, path)
                events.append((tag, *(launch(campaign, path, envelope, why) if ok else ('waiting', why))))
            elif not (run / 'done.json').exists():
                pid = json.loads((run / 'launch.json').read_text())['pid']
                events.append((tag, 'busy', 'running') if alive(pid) else (tag, 'waiting', f'run lost (pid {pid} gone, no done.json): {run}'))
            elif pr := prs(f'results/{campaign.name}/{eid}'):
                events.append((tag, 'waiting', f'results PR {pr[0]["state"].lower()}: {pr[0]["url"]}'))
            else:
                events.append((tag, *review(campaign, path, run)))
        if finished and not pending:
            last, tag = finished[-1], f'{campaign.name}/after-{finished[-1]}'
            if pr := prs(f'proposal/{campaign.name}/after-{last}'):
                events.append((tag, 'waiting', f'proposal PR {pr[0]["state"].lower()}: {pr[0]["url"]}'))
            elif (campaign / 'proposals' / f'after-{last}.md').exists():
                events.append((tag, 'waiting', 'campaign paused by a recorded stop/ask_human decision'))
            else:
                events.append((tag, *propose(campaign, envelope, last)))
    return events


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['tick', 'run'], help='tick: one pass; run: tick until only human steps remain')
    p.add_argument('--interval', type=float, default=5.0)
    a = p.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    with open(RUNS / '.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another tick is running')
        while True:
            events = tick()
            for tag, kind, msg in events:
                print(f'{now()} {kind:7} {tag}: {msg}', flush=True)
            if a.command == 'tick' or all(kind == 'waiting' for _, kind, _ in events):
                break
            time.sleep(a.interval)


if __name__ == '__main__':
    main()

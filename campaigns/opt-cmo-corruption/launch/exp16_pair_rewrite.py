"""exp-016 fault-aware pair rewrite over the exp-013 held P0s and new gmin/gmax/open 3% deployments.

Prepares, probes and collects native runs only. ``random`` jobs reuse the exp-013
materialized P0s and run one on-chip stage with all five arms. ``kinds`` jobs
deploy gmin, gmax and open at 3% from the frozen exp-007 HWA sources and then run
the four held arms; they must run on the GPU model that deployed the exp-013 P0s
(RTX 5090), so that healthy devices share their programming realization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
from exp1e_prepare_host import command, checked_input, sha256, DEFAULT_BASE
from exp7_recovery_regimes import ARRAYS, CATALOG, TAGS, lifecycle as exp13_lifecycle
from slice_campaign import write_frozen
from workflow.lifecycle import parse_lifecycle, verify_provenance

PREFIX = 'opt125m-cmo-l6-exp16-pair-rewrite'
NOTE = 'campaigns/opt-cmo-corruption/series/002-onchip-recovery/experiments/exp-016-pair-rewrite.md'
FAMILIES = ('random', 'kinds', 'random3')
KINDS = ('gmin', 'gmax', 'open')
KIND_RATE_PPM = 30000
# Shared by every rewrite arm, so a device receives the same partner draw in each arm.
REWRITE_SEED = 241016
ORACLE = {'detection': 'oracle', 'seed': REWRITE_SEED}
VERIFY = {'detection': 'verify', 'roles': 'pair', 'seed': REWRITE_SEED, 'threshold_sigma': 5.0}
HOLD_ARMS = [
    {'id': 'none', 'weights': 'hold', 'calibration': False},
    {'id': 'active_rewrite_oracle', 'weights': 'hold', 'calibration': False, 'fault_rewrite': {**ORACLE, 'roles': 'active'}},
    {'id': 'pair_rewrite_oracle', 'weights': 'hold', 'calibration': False, 'fault_rewrite': {**ORACLE, 'roles': 'pair'}},
    {'id': 'pair_rewrite_verify', 'weights': 'hold', 'calibration': False, 'fault_rewrite': VERIFY},
]
# Arm 5: the verify rewrite, then the exp-013 cap-4 recipe (random kind only).
RECOVERY_ARM = {'id': 'pair_rewrite_verify_recovery', 'weights': 'learn', 'calibration': False, 'fault_rewrite': VERIFY}
PHASE = 'opt-cmo-exp16-pair-rewrite-20261013'
# Recorded only from the user's explicit launch approval; queue() refuses without it.
AUTHORIZATION = ''
RANDOM3_AUTHORIZATION = ('User assigned implementation and execution of the reduced exp-016 plan: '
    'std3/cdt3/std9, assignments 271002-271004, held exp-013 nominal and random 3% P0s, '
    'all five declared arms including cap-4 recovery after verify rewriting. '
    'Six GPU-hours, 200 GB across hosts, 24 hours; only idle standing-authorized GPUs, '
    'with the repository 08:00-22:00 two-5090 limit. Other rates and kinds are deferred.')
CANARIES = ('std3-deploy', 'std3-271002-random', 'std3-271002-random3')
PEAK_MIB, HEADROOM_MIB = 20500, 2048
# From the October 13 probe on an RTX 3090 (42.6 s per held arm, 191 s for the
# learning arm, per case); RTX 5090 times scale by exp-013's measured ratio.
EXPECTED_SECONDS = {'random': {'3090': 2300, '5090': 1000}, 'kinds': {'3090': 650, '5090': 300},
                    'deploy': {'3090': 900, '5090': 900}, 'random3': {'3090': 900, '5090': 450}}
BUDGET_SECONDS = 4800
STORAGE_BYTES = {'random': 24_000_000_000, 'kinds': 9_000_000_000, 'deploy': 7_000_000_000,
                 'random3': 9_000_000_000}
# Provisional: the note's caps (6 GPU-hours, 200 GB) split across the two queue
# phases. The probe projects about 300 GB in all, so the user must decide storage
# before launch; update these with that decision.
BUDGETS = {'main': {'gpu_hours': 5.0, 'storage_bytes': 160_000_000_000, 'window_seconds': 86400},
           'kinds': {'gpu_hours': 1.0, 'storage_bytes': 40_000_000_000, 'window_seconds': 86400},
           'random3': {'gpu_hours': 6.0, 'storage_bytes': 200_000_000_000, 'window_seconds': 86400}}
QUESTION = ('How much fault-induced teacher-KL loss does a one-shot fault-aware pair rewrite remove, against '
            'active-only rewriting, HWA-only deployment and noisy on-chip recovery?')
DECISION = ('Frozen R0-R6 rules on assignments 271002-271004; oracle arms are non-physical bounds; '
            'no automatic follow-up. ' + NOTE)


def lifecycle(tag, family, *, probe_case=None):
    if tag not in TAGS or family not in FAMILIES:
        raise ValueError('Unknown exp-016 lifecycle')
    doc = exp13_lifecycle(f'{tag}-{ARRAYS[0]}-cap4').to_dict()
    doc['lifecycle_id'] = f'{PREFIX}-{tag}-{family}'
    doc['question'].update(text=QUESTION, decision=DECISION)
    doc['onchip']['arms'] = HOLD_ARMS + ([RECOVERY_ARM] if family in ('random', 'random3') else [])
    if family == 'random3':
        doc['onchip']['case_labels'] = ['nominal', 'random_30000ppm']
        doc['question']['decision'] = ('First phase: healthy and random 3% only. ' + DECISION)
    if family == 'kinds':
        doc['defects']['cases'] = [{'kind': kind, 'rate_ppm': KIND_RATE_PPM, 'die_concentration': None,
                                    'array_concentration': None} for kind in KINDS]
    if probe_case is not None:
        # A cost probe: one held P0 case, every arm; it is not evidence.
        doc['lifecycle_id'] += '-probe'
        doc['question'].update(text='Cost probe for exp-016 (not evidence): ' + QUESTION,
                               decision='Measure stage time, memory and state storage only. ' + NOTE)
        doc['onchip']['case_labels'] = [probe_case]
    return parse_lifecycle(doc)


def generate(phase='full'):
    out = []
    for tag in TAGS:
        for family in (('random3',) if phase == 'random3' else ('random', 'kinds')):
            lc = lifecycle(tag, family)
            path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
            write_frozen(path, lc.to_dict()); out.append(str(path))
    return out


def reused_contract(base, reused, tag, array, lc):
    """Inputs of a random job: the exp-013 materialized P0 and its frozen upstream bundles."""

    from experiments.cifar_crossbar.runtime import load
    group = f'{tag}-{array}'
    ref = json.loads(CATALOG.read_text())[group]
    receipt = json.loads((reused/group/'portable.json').read_text())
    expected_apparent = {k: v['apparent_sha256'] for k, v in ref['cases'].items()}
    if (receipt['parent_sha256'] != ref['deployment']['sha256'] or not receipt['physical_state_preserved']
            or receipt['new_device_writes'] or receipt['apparent_sha256'] != expected_apparent):
        raise ValueError('Wrong frozen deployment receipt')
    paths = {'teacher_weights': base/'inputs/opt-125m-27dcfa74/pytorch_model.bin',
             'device_data': reused/tag/'cache.pt', 'device_model': reused/tag/'devices.pt',
             'device_state': reused/group/'portable.pt'}
    digests = {k: v['sha256'] for k, v in ref['inputs'].items()}
    digests['device_state'] = receipt['sha256']
    artifacts = {k: checked_input(p, digests[k]) for k, p in paths.items()}
    p0 = load(paths['device_state'])
    if (p0['array_seed'] != array or p0['source_sha256'] != ref['source_sha256']
            or p0['materialized_from_sha256'] != ref['deployment']['sha256']):
        raise ValueError('Wrong source or array in frozen P0')
    for role, kind, arm in [('device_data', 'feature_cache', None), ('device_model', 'device_bundle', None),
                            ('device_state', 'deployment', ref['hwa_arm'])]:
        bundle = p0 if role == 'device_state' else load(paths[role])
        verify_provenance(bundle['provenance'], lc, kind, hwa_arm=arm)
    return {'artifacts': artifacts, 'array_seed': array, 'hwa_arm': ref['hwa_arm'],
            'source_sha256': ref['source_sha256'], 'parent_p0_sha256': ref['deployment']['sha256'],
            'apparent_sha256': expected_apparent}


def single_stage(lc, inputs, root, stage_id, env, *, allow_dirty=False):
    """Plan ``lc`` and keep one native stage bound to explicit inputs; dry-run it."""

    cfg = root/'lifecycle.json'
    write_frozen(cfg, lc.to_dict())
    command([sys.executable, '-m', 'workflow', 'plan', str(cfg), '--teacher-weights',
             inputs['artifacts']['teacher_weights']['path'], '--output', str(root/'plan'), '--device', 'cuda:0',
             '--cpu-threads', '1', '--python', sys.executable], cwd=ROOT)
    raw = json.loads((root/'plan/campaign.json').read_text())
    stage = next(s for s in raw['stages'] if s['id'] == stage_id)
    stage['config'] = str((root/'plan'/stage['config']).resolve()); stage['depends_on'] = []
    stage['inputs'] = {k: {'path': v['path']} for k, v in inputs['artifacts'].items()}
    raw['stages'] = [stage]; raw['campaign_id'] = lc.lifecycle_id
    manifest = root/'campaign.json'; write_frozen(manifest, raw)
    command([sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', str(manifest), '--output-dir',
             str(root/'dry-run'), '--dry-run', '--fail-fast'] + (['--allow-dirty'] if allow_dirty else []),
            cwd=ROOT, env=dict(os.environ, **env), logs=root/'preparation/dry-run')
    return manifest


def probe(base, reused, output, case):
    """Local cost probe of one random P0 case with every arm; outputs stay outside the evidence roots."""

    tag, array = 'std3', ARRAYS[0]
    lc = lifecycle(tag, 'random', probe_case=case)
    path = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
    write_frozen(path, lc.to_dict())
    inputs = reused_contract(base, reused, tag, array, lc)
    env = {'EBL_CORPUS_ROOT': str(base/'inputs/corpora'), 'EBL_CAMPAIGN_STREAM': '1',
           'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1'}
    output.mkdir(parents=True, exist_ok=True)
    write_frozen(output/'inputs.json', inputs)
    manifest = single_stage(lc, inputs, output, f'onchip__{inputs["hwa_arm"]}__{array}', env, allow_dirty=True)
    argv = [sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', str(manifest), '--output-dir',
            str(output/'runs'), '--fail-fast', '--allow-dirty']
    return {'lifecycle': str(path), 'manifest': str(manifest), 'argv': argv, 'env': env,
            'free_bytes_home': shutil.disk_usage(ROOT).free}


GPU_MODELS = {'riri': '5090', 'loulou': '5090', 'trex': '5090', 'fifi': '5090',
              'nom-cool-1': '3090', 'local': '3090'}


def labels(phase='full'):
    """Native jobs: one deploy campaign per source, then one on-chip slice per source, assignment and family."""

    if phase == 'random3':
        return [f'{tag}-{array}-random3' for tag in TAGS for array in ARRAYS]
    if phase != 'full':
        raise ValueError('Unknown exp-016 phase')
    return ([f'{tag}-deploy' for tag in TAGS]
            + [f'{tag}-{array}-{family}' for family in ('random', 'kinds') for tag in TAGS for array in ARRAYS])


def parts(label):
    if label not in labels() + labels('random3'):
        raise ValueError('Unknown exp-016 job')
    if label.endswith('-deploy'):
        return label.removesuffix('-deploy'), None, 'deploy'
    tag, array, family = label.split('-')
    return tag, int(array), family


def eligible(label, host):
    """Deployments run where the exp-013 P0s were deployed (RTX 5090), so healthy endpoints match."""

    _, _, family = parts(label)
    if host not in GPU_MODELS:
        raise ValueError('Unassigned host')
    return family != 'deploy' or GPU_MODELS[host] == '5090'


def source_reference(sources, tag):
    """The frozen exp-007 HWA source of ``tag``: the model deployed in every exp-013 P0 of that tag."""

    group = next(g for g in json.loads(CATALOG.read_text()).values() if g['tag'] == tag)
    path = next((sources/tag).glob('*/checkpoints/source.pt'))
    return {'path': path, 'result': path.parent.parent/'result.json', 'sha256': group['source_sha256'],
            'hwa_arm': group['hwa_arm']}


def deploy_campaign(lc, ref, root, env):
    """exp1g's deployment-only slice: prepare, devices and every declared deployment from the frozen source."""

    from exp1g_four_arm import deployment_manifest
    from workflow.lifecycle import provenance
    from workflow.runtime import validate_source_fit
    inputs = [checked_input(ref['path'], ref['sha256'])]
    original = parse_lifecycle(json.loads(ref['result'].with_name('config.resolved.json').read_text())['lifecycle'])
    verify_provenance(provenance(original, 'hwa_source', hwa_arm=ref['hwa_arm']), lc, 'hwa_source', hwa_arm=ref['hwa_arm'])
    validate_source_fit(lc, lc.hwa.arm(ref['hwa_arm']), json.loads(ref['result'].read_text())['metrics']['fit'])
    cfg = root/'lifecycle.json'; write_frozen(cfg, lc.to_dict())
    command([sys.executable, '-m', 'workflow', 'plan', str(cfg), '--teacher-weights',
             str(DEFAULT_BASE/'inputs/opt-125m-27dcfa74/pytorch_model.bin'), '--output', str(root/'plan'),
             '--device', 'cuda:0', '--cpu-threads', '1', '--python', sys.executable], cwd=ROOT, env=dict(os.environ, **env))
    manifest = root/'campaign.json'
    write_frozen(manifest, deployment_manifest(root/'plan/campaign.json', ref['path'], ref['hwa_arm']))
    command([sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest', str(manifest), '--output-dir',
             str(root/'dry-run'), '--dry-run', '--fail-fast'], cwd=ROOT, env=dict(os.environ, **env),
            logs=root/'preparation/dry-run')
    return manifest, inputs


def materialize_kinds(tag, deploy_root, reused, output):
    """Portable P0s of one kinds deployment campaign, as exp-013 did for its random P0s (no new writes)."""

    from workflow.portable_deployment import materialize, validate
    lc = lifecycle(tag, 'kinds'); reports = []
    runs = sorted(deploy_root.glob('runs/*/stages/*/local/deploy__*/attempt-*/runs/*'))
    for array in ARRAYS:
        run = next(r for r in runs if f'__{array}/' in str(r))
        target = output/f'{tag}-{array}'/'portable.pt'; target.parent.mkdir(parents=True, exist_ok=True)
        receipt = validate(target) if target.exists() else materialize(run, reused/tag/'devices.pt', target, 'cuda:0')
        if set(receipt['apparent_sha256']) != {c.label for c in lc.defects.cases}:
            raise ValueError('Materialized kinds P0 lacks a declared case')
        reports.append({'array_seed': array, **receipt})
    write_frozen(output/f'{tag}-materialized.json', {'tag': tag, 'deployments': reports})
    return reports


def kinds_contract(base, reused, kinds, tag, array, lc):
    """Inputs of a kinds on-chip slice: the materialized new P0 and the reused upstream bundles."""

    from experiments.cifar_crossbar.runtime import load
    ref = json.loads(CATALOG.read_text())[f'{tag}-{array}']
    receipt = json.loads((kinds/f'{tag}-{array}'/'portable.json').read_text())
    if not receipt['physical_state_preserved'] or receipt['new_device_writes']:
        raise ValueError('Wrong materialized kinds deployment')
    paths = {'teacher_weights': base/'inputs/opt-125m-27dcfa74/pytorch_model.bin',
             'device_data': reused/tag/'cache.pt', 'device_model': reused/tag/'devices.pt',
             'device_state': kinds/f'{tag}-{array}'/'portable.pt'}
    digests = {k: v['sha256'] for k, v in ref['inputs'].items()}; digests['device_state'] = receipt['sha256']
    artifacts = {k: checked_input(p, digests[k]) for k, p in paths.items()}
    p0 = load(paths['device_state'])
    if p0['array_seed'] != array or p0['source_sha256'] != ref['source_sha256']:
        raise ValueError('Wrong source or array in kinds P0')
    for role, kind, arm in [('device_data', 'feature_cache', None), ('device_model', 'device_bundle', None),
                            ('device_state', 'deployment', ref['hwa_arm'])]:
        bundle = p0 if role == 'device_state' else load(paths[role])
        verify_provenance(bundle['provenance'], lc, kind, hwa_arm=arm)
    return {'artifacts': artifacts, 'array_seed': array, 'hwa_arm': ref['hwa_arm'],
            'source_sha256': ref['source_sha256'], 'parent_p0_sha256': receipt['parent_sha256'],
            'apparent_sha256': receipt['apparent_sha256']}


def prepare_host(host, base, reused, *, sources=None, kinds=None, phase='full'):
    """Prepare every eligible job whose inputs exist here: deploy campaigns, random slices, kinds slices."""

    if command(['git', 'status', '--porcelain'], cwd=ROOT).strip():
        raise ValueError('Worker checkout must be clean')
    revision = command(['git', 'rev-parse', 'HEAD'], cwd=ROOT).strip()
    env = {'EBL_CORPUS_ROOT': str(base/'inputs/corpora'), 'EBL_CAMPAIGN_STREAM': '1',
           'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1'}
    cases = []
    for label in labels(phase):
        if not eligible(label, host):
            continue
        tag, array, family = parts(label)
        lc = lifecycle(tag, 'kinds' if family == 'deploy' else family)
        jid = PREFIX+'-'+label; root = ROOT/'results/lifecycles'/jid
        if family == 'deploy':
            if sources is None or not (sources/tag).is_dir():
                continue
            manifest, extra = deploy_campaign(lc, source_reference(sources, tag), root, env)
            native = root/'runs'/lc.lifecycle_id
            validation = [sys.executable, '-m', 'campaigns.lifecycle_collect', 'validate', str(native), '--deployment-only']
            files, artifacts = [manifest], extra
        else:
            if family in ('random', 'random3'):
                if not (reused/f'{tag}-{array}/portable.pt').exists():
                    continue
                inputs = reused_contract(base, reused, tag, array, lc)
            else:
                if kinds is None or not (kinds/f'{tag}-{array}/portable.pt').exists():
                    continue
                inputs = kinds_contract(base, reused, kinds, tag, array, lc)
            write_frozen(root/'inputs.json', inputs)
            manifest = single_stage(lc, inputs, root, f'onchip__{inputs["hwa_arm"]}__{array}', env)
            native = root/'runs'/lc.lifecycle_id
            validation = [sys.executable, '-m', 'campaigns.recovery_collect', 'validate', str(native),
                          '--inputs', str(root/'inputs.json')]
            files, artifacts = [manifest, root/'inputs.json', CATALOG], list(inputs['artifacts'].values())
        cfg = ROOT/'campaigns/opt-cmo-corruption/lifecycles'/(lc.lifecycle_id+'.json')
        if parse_lifecycle(json.loads(cfg.read_text())).digest() != lc.digest():
            raise ValueError('Lifecycle changed')
        receipt = base/'worker-receipts'/revision/jid; receipt.parent.mkdir(parents=True, exist_ok=True)
        worker = root/'tmux-worker.json'
        write_frozen(worker, {'case_id': jid, 'state_root': str(receipt), 'cwd': str(ROOT),
            'timeout_seconds': 2100 if family == 'random3' else 7200,
            'lease_seconds': 900, 'env': env, 'argv': [sys.executable, '-m', 'ebl', 'campaign', 'run', '--manifest',
            str(manifest), '--output-dir', str(root/'runs'), '--resume', '--fail-fast']})
        cases.append({'id': jid, 'label': label, 'family': family, 'lifecycle_sha256': lc.digest(), 'dry_run_passed': True,
            'placement': {'workdir': str(ROOT), 'run_dir': str(receipt), 'gpu_models': [GPU_MODELS[host]],
                'argv': [sys.executable, '-m', 'campaigns.tmux_worker', 'bridge', '--spec', str(worker)],
                'validate_argv': validation, 'campaign_python': sys.executable, 'campaign_worker_spec': str(worker),
                'campaign_result_root': str(native),
                **({'recovery_inputs': str(root/'inputs.json'), 'recovery_input_contract': inputs}
                   if family != 'deploy' else {}),
                'prepared_files': [{'path': str(p), 'sha256': sha256(p)} for p in [cfg, worker, *files]] + artifacts}})
    return {'host': host, 'source': str(ROOT), 'source_revision': revision, 'base': str(base),
            'phase': phase, 'cases': cases}


def check_realization(tag, array, teacher, portable, device='cuda:0'):
    """Healthy devices of each gmin/gmax/open 3% deployment equal those of the exp-013 random 3% P0.

    Run on the 5090 host of the kinds job. Each kinds array is rebuilt exactly as
    the deploy stage programs it; ``portable`` is the materialized exp-013 P0,
    whose held conductances are compared at every device healthy in both maps.
    """

    import torch
    from experiments.cifar_crossbar.runtime import load
    from workflow import deployment, program_verify
    from workflow import devices as device_stage
    lc = lifecycle(tag, 'kinds'); p0 = load(portable)
    group = json.loads(CATALOG.read_text())[f'{tag}-{array}']
    network = deployment.build_network(deployment.load_teacher(teacher, lc.network, device)[0], lc, group['hwa_arm'])
    network.load_state_dict(p0['model'])
    target = deployment.targets(network).to(device)
    if not torch.equal(target, p0['target'].to(device)):
        raise ValueError('Expected the exp-013 deployment targets')
    reference = p0['cases'][f'random_{KIND_RATE_PPM}ppm']['array']
    old_g = torch.cat([child['g'] for child in reference['children']], 1).to(device)
    rows = []
    for case in lc.defects.cases:
        fresh = program_verify.fresh(lc, network.layout, array, case, None, device)
        fresh.program(target)
        mask, _ = device_stage.hidden_faults(fresh)
        new_g = torch.cat([child.g for child in fresh.children], 1)
        healthy = ~mask
        rows.append({'case': case.label, 'compared_devices': int(healthy.sum()),
                     'differing_devices': int((new_g[healthy] != old_g[healthy]).sum())})
    return {'tag': tag, 'array_seed': array, 'reference_case': f'random_{KIND_RATE_PPM}ppm',
            'gpu': torch.cuda.get_device_name(0) if device.startswith('cuda') else 'cpu', 'cases': rows,
            'identical': all(r['differing_devices'] == 0 for r in rows)}


def campaign_manifest(phase, labels_, worker, evidence, revision, hosts, budget):
    from exp3_lora import thresholds
    cases = []
    for label in labels_:
        canary = label in CANARIES
        cases.append({'id': PREFIX+'-'+label, 'stage': 'readiness' if canary else 'training',
            'dependencies': [] if canary else [PREFIX+'-'+x for x in CANARIES if x in labels_],
            'required_gates': ['preflight'] if canary else ['handoff'],
            'expected_gpu_seconds': EXPECTED_SECONDS[parts(label)[2]]['3090'],
            'expected_storage_bytes': STORAGE_BYTES[parts(label)[2]],
            'progress': {**thresholds(EXPECTED_SECONDS[parts(label)[2]]['3090']),
                         'phases': {'onchip': thresholds(EXPECTED_SECONDS[parts(label)[2]]['3090']),
                                    'validation': thresholds(300), 'collection': thresholds(300)}}})
    doc = {'version': 1, 'id': phase, 'poll_seconds': 300, 'required_adapters': ['tmux'],
           'budgets': budget, 'cases': cases,
           'storage_roots': [{'target': h, 'path': str(worker)} for h in hosts]
             + [{'target': h, 'path': str(DEFAULT_BASE/'worker-receipts'/revision)} for h in hosts]
             + [{'target': 'local', 'path': str(evidence)}]}
    from campaigns.execution_service import validate_manifest
    validate_manifest(doc)
    return doc


def queue(receipts, output, controller, evidence, phase):
    """Queue the main phase (deploy campaigns and random slices) or, after materialization, the kinds slices."""

    authorization = RANDOM3_AUTHORIZATION if phase == 'random3' else AUTHORIZATION
    if not authorization:
        raise ValueError('exp-016 has no recorded launch authorization')
    hosts = {json.loads(p.read_text())['host']: json.loads(p.read_text()) for p in receipts}
    if len({h['source_revision'] for h in hosts.values()}) != 1 or len({h['source'] for h in hosts.values()}) != 1:
        raise ValueError('Require prepared hosts at one frozen revision and worker path')
    families = {'main': ('deploy', 'random'), 'kinds': ('kinds',), 'random3': ('random3',)}[phase]
    selected = [label for label in labels('random3' if phase == 'random3' else 'full')
                if parts(label)[2] in families]
    deadline = json.loads((evidence/'readiness/execution-window.json').read_text())['deadline']
    jobs = []
    for label in selected:
        tag, _, family = parts(label)
        lc = lifecycle(tag, 'kinds' if family == 'deploy' else family)
        prepared = {host: c for host, h in hosts.items() for c in h['cases'] if c['label'] == label}
        if not prepared:
            raise ValueError('Scientific case has no validated placement: '+label)
        if any(not c['dry_run_passed'] or c['lifecycle_sha256'] != lc.digest() for c in prepared.values()):
            raise ValueError('Unprepared scientific case')
        job = {'id': PREFIX+'-'+label, 'project_dir': str(controller), 'handoff': NOTE, 'authorization': authorization,
               'peak_memory_mib': PEAK_MIB, 'headroom_mib': HEADROOM_MIB,
               'expected_seconds': EXPECTED_SECONDS[family]['3090'],
               'budget_seconds': 2400 if phase == 'random3' else BUDGET_SECONDS,
               'deadline': deadline, 'priority': 120 if label in CANARIES else 100,
               'startup_grace_seconds': 600, 'stale_seconds': 900, 'recovery': False, 'retry_safe': False,
               'placements': {h: c['placement'] for h, c in prepared.items()},
               'completion': {'argv': [sys.executable, str(controller/'campaigns/opt-cmo-corruption/launch/exp16_pair_rewrite.py'),
                    'collect', '--queue-config', str(Path.home()/'.config/run-watch/queue.json'), '--evidence', str(evidence)],
                    'workdir': str(controller), 'timeout_seconds': 900, 'retry_safe': True}}
        path = output/'jobs'/(job['id']+'.json'); write_frozen(path, job); jobs.append(str(path))
    revision = next(iter(hosts.values()))['source_revision']
    budget = BUDGETS[phase]
    manifest = campaign_manifest(f'{PHASE}-{phase}', selected, Path(next(iter(hosts.values()))['source']),
                                 evidence, revision, sorted(hosts), budget)
    if phase == 'random3' and 'fifi' in hosts:
        # Inputs newly staged for this phase also consume its storage allowance.
        manifest['storage_roots'].append({'target': 'fifi', 'path': str(DEFAULT_BASE/'exp16-reused')})
    write_frozen(output/'campaign.json', manifest)
    return {'manifest': str(output/'campaign.json'), 'jobs': jobs}


SPLITS = ('development', 'transfer_generated', 'transfer_wikitext')


def checked_report(label, rows, records, collection):
    """Validate one on-chip slice and extract per-arm metrics and rewrite receipts."""

    import math
    tag, array, family = parts(label); lc = lifecycle(tag, family)
    expected = {(c.label, a.arm_id) for c in lc.onchip_cases for a in lc.onchip.arms}
    if {(r['case'], r['onchip_arm']) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError('Missing or duplicate case/arm coverage')
    values = []
    ref = json.loads(CATALOG.read_text())[f'{tag}-{array}']
    for case in lc.onchip_cases:
        arms = {r['onchip_arm']: r for r in rows if r['case'] == case.label}
        hold = arms['none']
        if hold['final_apparent_sha256'] != hold['initial_apparent_sha256'] or hold['final'] != hold['initial']:
            raise ValueError('Held control changed')
        if family in ('random', 'random3') and hold['initial_apparent_sha256'] != ref['cases'][case.label]['apparent_sha256']:
            raise ValueError('Held control differs from the frozen exp-013 P0')
        for arm in lc.onchip.arms:
            row = arms[arm.arm_id]; rewrite = row.get('fault_rewrite')
            if (row['array_seed'] != array or row['hwa_arm'] != ref['hwa_arm']
                    or row['initial_apparent_sha256'] != hold['initial_apparent_sha256']):
                raise ValueError('Arms do not share their P0')
            if (rewrite is None) != (arm.fault_rewrite is None):
                raise ValueError('Missing or unexpected rewrite receipt')
            if rewrite:
                reads = rewrite['cost']['verify_reads']
                if reads != (2*lc.network.analog_weights if rewrite['detection'] == 'verify' else 0):
                    raise ValueError('Unexpected verify-read count')
                if not arm.trains and row['final_apparent_sha256'] != rewrite['rewritten_apparent_sha256']:
                    raise ValueError('A held rewrite arm changed after its rewrite')
            if arm.trains and rewrite['rewritten_apparent_sha256'] != arms['pair_rewrite_verify']['final_apparent_sha256']:
                raise ValueError('Learning did not start from the verify rewrite')
            for split in SPLITS:
                kl = row['final'][split]['teacher_kl']
                if not math.isfinite(kl) or kl < 0:
                    raise ValueError('Invalid KL')
            values.append({'tag': tag, 'array_seed': array, 'family': family, 'case': case.label,
                           'kind': case.kind, 'rate_ppm': case.rate_ppm, 'arm': arm.arm_id,
                           'final': {s: row['final'][s] for s in SPLITS}, 'cost': row['cost'],
                           'fault_rewrite': rewrite, 'initial_apparent_sha256': hold['initial_apparent_sha256'],
                           'final_apparent_sha256': row['final_apparent_sha256']})
    memory = max((m['peak_reserved_bytes'] for m in records if m.get('stage') == 'onchip_memory'), default=0)
    if memory > PEAK_MIB*1024**2: raise ValueError('Measured CUDA peak exceeds reservation')
    elapsed = next(m['elapsed_seconds'] for m in records if m.get('stage') == 'onchip' and m.get('terminal'))
    return {'label': label, 'rows': values, 'peak_reserved_bytes': memory, 'onchip_seconds': elapsed,
            'collection': str(collection)}


def held_reproduction(reports, exp13):
    """The ``none`` arm reproduces exp-013's held metrics (within 1e-6 relative) for every random P0."""

    previous = json.loads((exp13/'reviews/summary.json').read_text())
    reference = {(r['tag'], r['array_seed'], r['case']): r for r in previous['rows'] if r['cap'] == 'cap4'}
    worst, mismatched = 0.0, []
    for report in reports:
        for row in report['rows']:
            if row['family'] not in ('random', 'random3') or row['arm'] != 'none':
                continue
            ref = reference[row['tag'], row['array_seed'], row['case']]
            if row['initial_apparent_sha256'] != ref['initial_apparent_sha256']:
                mismatched.append({**{k: row[k] for k in ('tag', 'array_seed', 'case')}, 'field': 'p0'})
            for split in SPLITS:
                old = ref['effects'][split]['deployment']['teacher_kl']; new = row['final'][split]['teacher_kl']
                worst = max(worst, abs(new - old)/old)
    return {'max_relative_kl_difference': worst, 'p0_mismatches': mismatched,
            'reproduced': not mismatched and worst <= 1e-6}


def summarize(reports, exp13, phase='full'):
    """Frozen decision inputs: D, D0, every arm's KL and gap closed per unit; exp-013 references for R4."""

    expected = {l for l in labels(phase) if parts(l)[2] != 'deploy'}
    if len(reports) != len(expected) or {r['label'] for r in reports} != expected:
        raise ValueError('Incomplete or duplicate scientific coverage')
    rows = {(r['tag'], r['array_seed'], r['case'], r['arm']): r for report in reports for r in report['rows']}
    expected_rows = {(tag, array, c.label, a.arm_id)
                     for label in expected for tag, array, family in [parts(label)]
                     for c in lifecycle(tag, family).onchip_cases for a in lifecycle(tag, family).onchip.arms}
    if set(rows) != expected_rows or len(rows) != sum(len(r['rows']) for r in reports):
        raise ValueError('Incomplete or duplicate scientific coverage')
    reproduction = held_reproduction(reports, exp13)
    if not reproduction['reproduced']:
        raise ValueError('Held controls do not reproduce exp-013')
    previous = json.loads((exp13/'reviews/summary.json').read_text())
    noisy = {(r['tag'], r['array_seed'], r['case'], r['cap']): r for r in previous['rows']}
    units = []
    for (tag, array, case, arm), row in sorted(rows.items()):
        if arm == 'none' or case == 'nominal':
            continue
        for split in SPLITS:
            d = rows[tag, array, case, 'none']['final'][split]['teacher_kl']
            d0 = rows[tag, array, 'nominal', 'none']['final'][split]['teacher_kl']
            x = row['final'][split]['teacher_kl']
            unit = {'source': tag, 'array_seed': array, 'case': case, 'kind': row['kind'], 'rate_ppm': row['rate_ppm'],
                    'arm': arm, 'domain': split, 'D': d, 'D0': d0, 'X': x,
                    'gap_closed': (d - x)/(d - d0) if d - d0 > 0 else None}
            if row['kind'] == 'random':
                for cap in ('uncapped', 'cap4'):
                    unit[f'exp013_{cap}'] = noisy[tag, array, case, cap]['effects'][split]['onchip']['teacher_kl']
            units.append(unit)
    return {'coverage_complete': True, 'phase': phase, 'full_study_complete': phase == 'full',
            'starting_conditions': len({k[:3] for k in rows}), 'endpoints': len(rows),
            'rows': list(rows.values()), 'units': units, 'held_reproduction': reproduction,
            'selection': 'Frozen R0-R6 inputs only; classification and verdict belong to the review.'}


def collect(queue_config, evidence):
    import fcntl
    from experiments.artifacts import atomic_write_json
    label = os.environ['RUN_WATCH_JOB_ID'].removeprefix(PREFIX+'-'); _, _, family = parts(label)
    phase = 'random3' if family == 'random3' else 'full'
    exp13 = evidence.parents[1]/'opt125m-cmo-l6-exp7-recovery-regimes' if phase == 'random3' else evidence.parent/'opt125m-cmo-l6-exp7-recovery-regimes'
    if family == 'deploy':
        from campaigns.lifecycle_collect import collect as deployment_collect
        result = deployment_collect(queue_config, evidence/'collected', deployment_only=True)
        return {'collected': result, 'label': label}
    from campaigns.recovery_collect import collect as native_collect
    result = native_collect(queue_config, evidence/'collected'); folder = Path(result['collected'])
    report = checked_report(label, json.loads((folder/'measurements.json').read_text()),
        [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()],
        folder/'collection.json')
    reproduction = held_reproduction([report], exp13)
    if not reproduction['reproduced']:
        raise ValueError('Held controls do not reproduce exp-013')
    with (evidence/'.completion.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_frozen(evidence/'reviews'/(label+'.json'), report)
        expected = [l for l in labels(phase) if parts(l)[2] != 'deploy']
        if phase == 'random3' and label in CANARIES:
            from exp1f_cdt_lr import service, transition
            from campaigns.execution_service import digest
            manifest = json.loads((evidence/'prepared/queue/campaign.json').read_text())
            gate = evidence/'readiness/handoff.json'
            write_frozen(gate, {'manifest_sha256': digest(manifest), 'passed': True,
                               'evidence': [str(evidence/'reviews'/(label+'.json'))],
                               'held_reproduction': reproduction, 'onchip_seconds': report['onchip_seconds']})
            service(queue_config, 'gate', manifest['id'], 'handoff', gate)
            for name in expected:
                if name not in CANARIES:
                    transition(queue_config, 'release', PREFIX+'-'+name)
        completed = [json.loads(p.read_text()) for l in expected if (p := evidence/'reviews'/(l+'.json')).exists()]
        atomic_write_json(evidence/'reviews/progress.json', {'completed_jobs': len(completed),
                          'expected_jobs': len(expected), 'phase': phase,
                          'completed_endpoints': sum(len(r['rows']) for r in completed),
                          'reports': [r['label'] for r in completed]})
        if len(completed) == len(expected):
            write_frozen(evidence/'reviews/summary.json',
                         summarize(completed, exp13, phase))
    return {'collected': str(folder), 'label': label}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True)
    q = sub.add_parser('generate'); q.add_argument('--phase', choices=('full', 'random3'), default='full')
    q = sub.add_parser('probe'); q.add_argument('--base', type=Path, default=DEFAULT_BASE)
    q.add_argument('--reused', type=Path, required=True); q.add_argument('--output', type=Path, required=True)
    q.add_argument('--case', default='random_30000ppm')
    q = sub.add_parser('prepare-host'); q.add_argument('--host', required=True)
    q.add_argument('--base', type=Path, default=DEFAULT_BASE); q.add_argument('--reused', type=Path, required=True)
    q.add_argument('--sources', type=Path); q.add_argument('--kinds', type=Path)
    q.add_argument('--phase', choices=('full', 'random3'), default='full')
    q = sub.add_parser('materialize'); q.add_argument('--tag', required=True)
    q.add_argument('--deploy-root', type=Path, required=True); q.add_argument('--reused', type=Path, required=True)
    q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('queue'); q.add_argument('receipts', type=Path, nargs='+')
    q.add_argument('--output', type=Path, required=True); q.add_argument('--controller', type=Path, required=True)
    q.add_argument('--evidence', type=Path, required=True); q.add_argument('--phase', choices=('main', 'kinds', 'random3'), required=True)
    q = sub.add_parser('collect'); q.add_argument('--queue-config', type=Path, required=True)
    q.add_argument('--evidence', type=Path, required=True)
    q = sub.add_parser('check-realization'); q.add_argument('--tag', required=True)
    q.add_argument('--array', type=int, required=True); q.add_argument('--teacher', type=Path, required=True)
    q.add_argument('--portable', type=Path, required=True)
    a = p.parse_args()
    if a.mode == 'generate': result = generate(a.phase)
    elif a.mode == 'probe': result = probe(a.base, a.reused, a.output, a.case)
    elif a.mode == 'prepare-host': result = prepare_host(a.host, a.base, a.reused, sources=a.sources, kinds=a.kinds, phase=a.phase)
    elif a.mode == 'materialize': result = materialize_kinds(a.tag, a.deploy_root, a.reused, a.output)
    elif a.mode == 'queue': result = queue(a.receipts, a.output, a.controller, a.evidence, a.phase)
    elif a.mode == 'collect': result = collect(a.queue_config, a.evidence)
    else: result = check_realization(a.tag, a.array, a.teacher, a.portable)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__': main()

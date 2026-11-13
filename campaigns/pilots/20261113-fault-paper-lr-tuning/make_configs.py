"""Write the strict configs of the fault-recovery paper LR screen (DRN half).

Usage: python campaigns/pilots/20261113-fault-paper-lr-tuning/make_configs.py [--check]

Writes ``examples/opt_cmo_drn/fault_paper_tuning/*.json``: the 3 % / 271001 deployment
(the October deploy stage, unchanged except the fault case), the job configs (arms packed
per Jean Zay job) and two small canary configs.
"""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'examples' / 'opt_cmo_drn' / 'fault_paper_tuning'
BASE = ROOT / 'examples' / 'opt_cmo_drn' / 'corruption_levels' / 'random02-deploy.json'
DEPLOYMENT = {'assignment_seed': 271001, 'fault_kind': 'random', 'fault_rate': 0.03}
LOCAL_SAMPLES = '6bc43ffaa92134c39308b844aba562f97c396229976ab416cd2a452d9c338b4f'
WRITE_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
LORA_RATES = (1e-3, 3e-3, 1e-2)
SETTINGS = {
    'corpus': 'wikitext-2-raw-v1',
    'corpus_sha256': '33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01',
    'sequence_length': 256,
    'recovery_split': 'train', 'recovery_seed': 20261104, 'recovery_sequences': 1024,
    'recovery_sha256': 'c919329dbe0d545d7614daf3d8f790881e944cd80ff53fff88fc19d9697f7762',
    'samples_file': 'teacher-samples8-c919329d-df73ce66.pt',
    'samples_sha256': 'df73ce6615be3cdd3005d2ca0b97c03131f9236ba481bc7250edde5ff66fb01a',
    'samples_count': 8, 'samples_seed': 20365833,
    'development_split': 'validation', 'development_start': 0, 'development_sequences': 64,
    'development_sha256': '7411cf50b625e4c62f65d8e12bac538fd83c6dad3e4c74b29a9f2a6f5ad83191',
    'steps': 300, 'batch': 8, 'microbatch': 1, 'evaluation_microbatch': 1,
    'order_seed': 20261104, 'schedule': 'cosine_to_zero', 'development_every': 100,
    'child_fraction': 0.1, 'fisher_batch': 8, 'lora_alpha': 'rank', 'lora_seed': 20261113,
}


def arm(method, rate=None, rank=None):
    label = method if rate is None else f'{method}{"" if rank is None else rank}-lr{rate:g}'
    return {'label': label, 'method': method,
            **({} if rate is None else {'learning_rate': rate}), **({} if rank is None else {'rank': rank})}


def arms():
    return {
        'holds': [arm('naive_hold'), arm('hold')],
        'noisy': [arm('noisy', r) for r in WRITE_RATES],
        'noiseless': [arm('noiseless', r) for r in WRITE_RATES],
        'child_d': [arm('child_d', r) for r in WRITE_RATES],
        'lora16': [arm('lora', r, 16) for r in LORA_RATES],
        'lora64': [arm('lora', r, 64) for r in LORA_RATES],
    }


def documents(jobs):
    base = json.loads(BASE.read_text())
    deploy = copy.deepcopy(base)
    deploy['deployment'] = dict(DEPLOYMENT)
    deploy['device'] = 'cuda:0'
    out = {'deploy.json': deploy}
    tuning = copy.deepcopy(deploy)
    tuning['stage'] = 'fault_tuning'
    tuning['recovery'].update(arm='learning', update_rule='bounded_thresholded_adam', write_noise_factor=1.0,
                              threshold_sigma=2.0, max_write_attempts=None, target_bounds='gmin_gmax',
                              conductance_floor='none', development_every=0)
    tuning['training']['max_wall_seconds'] = 6900
    tuning['cpu_threads'] = 10
    a = arms()
    for name, groups in jobs.items():
        doc = copy.deepcopy(tuning)
        doc['fault_tuning'] = {**SETTINGS, 'arms': [x for g in groups for x in a[g]]}
        out[f'{name}.json'] = doc
    canary = copy.deepcopy(tuning)
    canary['training']['max_wall_seconds'] = 1200
    canary['cpu_threads'] = 4
    canary['fault_tuning'] = {**SETTINGS, 'steps': 3, 'development_every': 2, 'recovery_sequences': 1024,
                              'arms': [arm('naive_hold'), arm('hold'), arm('noisy', 1e-2), arm('noiseless', 1e-3),
                                       arm('child_d', 1e-2), arm('lora', 1e-2, 16)]}
    out['canary.json'] = canary
    # Local RTX 3090 canary only: the same draw procedure gives a different tensor there.
    local = copy.deepcopy(canary)
    local['fault_tuning'].update(samples_file='teacher-samples8-c919329d-6bc43ffa-rtx3090.pt',
                                 samples_sha256=LOCAL_SAMPLES)
    out['canary-rtx3090.json'] = local
    return out


# Packing per Jean Zay job (see the pilot note for the wall-time estimate).
JOBS = {
    'job-a-holds-noisy': ['holds', 'noisy'],
    'job-b-noiseless': ['noiseless'],
    'job-c-child-d': ['child_d'],
    'job-d-lora16': ['lora16'],
    'job-e-lora64': ['lora64'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, doc in documents(JOBS).items():
        text = json.dumps(doc, indent=2) + '\n'
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
    if stale:
        raise SystemExit(f'Stale configs: {stale}')
    print(json.dumps({'configs': sorted(documents(JOBS))}))


if __name__ == '__main__':
    main()

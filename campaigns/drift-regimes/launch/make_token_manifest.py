"""Create launch groups from immutable token-backward lifecycle candidates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_token_backward_lifecycles import MODES, RATES, name
from make_manifest import ROOT, JZ


def manifest(source,output,rates=None):
    root=ROOT/'campaigns/drift-regimes/lifecycles'
    paths=sorted(root.glob('*-tokens-*.json'))
    configs={'tb-canary':[str(p.relative_to(ROOT)) for p in paths if p.stem.endswith('-tokens-canary')]}
    for mode in MODES:
        configs['tb-tune-'+name(mode)]=[str(p.relative_to(ROOT)) for p in paths
                                      if p.stem.endswith('-tokens-tune-'+name(mode))]
    if rates is not None:
        if set(rates)!=set(MODES) or any(lr not in RATES for lr in rates.values()):
            raise ValueError('Expected a predeclared grid LR for each mode.')
        for seed in (271001,271002,271003):
            selected=[]
            for p in paths:
                if f'-{seed}-tokens-eval-' not in p.stem: continue
                if p.stem.endswith('-tokens-eval-reference') or any(
                    p.stem.endswith('-tokens-eval-'+name(m)+'-lr'+format(lr,'g')) for m,lr in rates.items()):
                    selected.append(str(p.relative_to(ROOT)))
            configs['tb-eval-'+str(seed)]=selected
    if any(not v for v in configs.values()): raise ValueError('Missing immutable lifecycle candidate.')
    return {
      'schema':'drift_regimes.screen.launch.v1','run':'token-backward',
      'campaign_prefix':'opt125m-token-backward-', 'source':source,'output':output,
      'teacher':JZ+'/opt-cmo-exp1/inputs/opt-125m-27dcfa74/pytorch_model.bin',
      'teacher_sha256':'2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9',
      'weights':JZ+'/opt-drift-times-20261028/inputs/crossbar-std3.pt',
      'weights_sha256':'86522e230dfac9646beffbd84d244fa7cdf7ecdbac4b9f3db2de29dcbab2b028',
      'corpus_root':JZ+'/opt-cmo-exp1/inputs/corpora',
      'corpora':{'opt125m-synthetic-50m':'ce58085af0ae1add6415372a50bf20ba4238ee8cae5ee55093c83e6a9a36d984',
                 'wikitext-2-raw-v1':'33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01'},
      'configs':configs,'budget_gpu_seconds':28800,
      'resources':{'account':'umg@v100','partition':'gpu_p13','qos':'qos_gpu-dev',
                   'constraint':'v100-32g','gpus':1,'cpus':10,'memory':'site-managed; no --mem flags'}}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source');parser.add_argument('output');parser.add_argument('destination')
    parser.add_argument('--rates')
    a=parser.parse_args()
    rates=None if a.rates is None else json.loads(Path(a.rates).read_text())['selected']
    m=manifest(a.source,a.output,rates)
    Path(a.destination).write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps({'groups':list(m['configs'])}))

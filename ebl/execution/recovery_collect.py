"""Collect native recovery-only slices with explicitly pinned upstream artifacts.

Each slice contains one onchip stage. All four inputs are absolute frozen paths;
the native runtime checks their lifecycle provenance. This collector checks the
manifest, full artifact inspection, exact trajectory coverage, replay, upstream
hashes and portable deployment lineage again before publishing compact evidence.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from core.artifacts import content_hash, sha256_file
from ebl.execution.lifecycle_collect import (CONTROL_FILES, exact_attempt, finite, text_sha,
                                        write_frozen, collection_lock)
from ebl.schema.lifecycle import parse_lifecycle
from ebl.lifecycle.rows import _measurement_rows


def checked(payload, inputs):
    if payload['schema'] != 'lifecycle.reused_recovery.v1':
        raise ValueError('Unexpected recovery export schema')
    root = Path(payload['root']); run = Path(payload['run']); files = payload['files']
    if not root.is_absolute() or not run.is_relative_to(root):
        raise ValueError('Run escaped native campaign root')
    contract = json.loads(payload['campaign_resolved'])
    source = json.loads(payload['source_manifest'])
    from ebl.execution.schema import CampaignSpec
    from ebl.execution.runner import _resolved_campaign_contract
    parsed = CampaignSpec.parse(source, base_dir=Path(contract['source_manifest']['path']).parent)
    resolved = _resolved_campaign_contract(parsed)
    if (resolved != {k:v for k,v in contract.items() if k not in {'source_manifest','resolved_contract_sha256'}}
            or content_hash(resolved) != contract['resolved_contract_sha256']
            or text_sha(payload['source_manifest']) != contract['source_manifest']['sha256']
            or root.name != contract['campaign_id'] or len(contract['stages']) != 1):
        raise ValueError('Frozen campaign contract mismatch')
    planned = contract['stages'][0]
    aggregate = json.loads(payload['aggregate'])['stages']
    if len(aggregate) != 1 or aggregate[0]['status'] != 'complete' or aggregate[0]['stage_id'] != planned['id']:
        raise ValueError('Incomplete native coverage')
    result = json.loads(files['result.json']); config = json.loads(files['config.resolved.json'])
    manifest = json.loads(files['manifest.json']); inspection = json.loads(files['inspection.json'])
    lc = parse_lifecycle(config['lifecycle']); selection = config['stage']
    if (result['status'] != 'complete' or selection['kind'] != 'onchip'
            or config != json.loads(files['stage-config.json'])
            or manifest['config']['sha256'] != content_hash(config)
            or result['metrics']['stage'] != selection or result['metrics']['lifecycle_sha256'] != lc.digest()
            or aggregate[0]['run_result'] != str(run / 'result.json')
            or aggregate[0]['run_result_sha256'] != text_sha(files['result.json'])):
        raise ValueError('Native result/configuration identity mismatch')
    if (len(inspection) != 1 or inspection[0]['valid'] is not True
            or inspection[0]['process_complete'] is not True
            or inspection[0]['validation_mode'] != 'full_artifact_hashes'
            or inspection[0]['result_sha256'] != text_sha(files['result.json'])
            or inspection[0]['path'] != str(run)):
        raise ValueError('Native full artifact inspection failed')
    bound = {r['role']:r for r in manifest['inputs'] if r['role'] != 'staged_source_snapshot'}
    if set(bound) != set(inputs['artifacts']) or set(planned['inputs']) != set(bound):
        raise ValueError('External input coverage mismatch')
    for role, expected in inputs['artifacts'].items():
        if any(bound[role].get(k) != expected[k] for k in ('path','sha256')) or planned['inputs'][role].get('path') != expected['path']:
            raise ValueError('Frozen external input identity mismatch')
    kinds = {a['kind']:a for a in result['artifacts']}
    rows = json.loads(files['measurements.json']); finite(rows); finite(result['metrics'])
    if kinds['measurements']['sha256'] != text_sha(files['measurements.json']):
        raise ValueError('Measurements changed')
    expected_rows = {(c.label,a.arm_id) for c in lc.onchip_cases for a in lc.onchip.arms}
    if len(rows) != len(expected_rows) or {(r['case'],r['onchip_arm']) for r in rows} != expected_rows or result['metrics']['exact_replays'] != len(rows):
        raise ValueError('Trajectory coverage or exact replay mismatch')
    arms = {a.arm_id:a for a in lc.onchip.arms}
    for row in rows:
        a = arms[row['onchip_arm']]
        epochs = lc.onchip.epochs if a.trains else 0
        if (row['lifecycle_id'] != lc.lifecycle_id or row['hwa_arm'] != selection['hwa_arm']
                or row['array_seed'] != inputs['array_seed'] or row['array_seed'] != selection['array_seed']
                or row['p0_sha256'] != inputs['artifacts']['device_state']['sha256']
                or row['weights'] != a.weights or row['calibration'] != a.calibration
                or row['sequence_presentations'] != epochs * lc.recovery_examples
                or [r['epoch'] for r in row['curve']] != list(range(1,epochs+1))
                or not row['replay_exact']
                or row['state_sha256'] != kinds[f"onchip_state.{row['case']}.{a.arm_id}"]['sha256']
                or row['initial_apparent_sha256'] != inputs['apparent_sha256'][row['case']]):
            raise ValueError('Recovery identity, state, schedule or cohort mismatch')
        if lc.recovery_views:
            from networks.context_views import accounting, extra_evaluation_names
            if any(row.get(k) != v for k,v in accounting(lc, epochs).items()):
                raise ValueError('Recovery context token/update budget mismatch')
            if any(name not in row['initial'] or name not in row['final'] for name in extra_evaluation_names(lc)):
                raise ValueError('Missing fixed context evaluation')
        if a.digital_adapter:
            meta = row.get('digital_adapter', {})
            if (any(meta.get(k) != v for k,v in a.digital_adapter.to_dict().items())
                    or meta.get('base_unchanged') is not True or meta.get('array_unchanged') is not True
                    or row['final_apparent_sha256'] != row['initial_apparent_sha256']
                    or any(row['cost'][k] != 0 for k in ('device_writes','device_programs','verify_reads','stuck_write_attempts'))):
                raise ValueError('Digital-only LoRA contract violated')
        if a.weights == 'learn' and lc.onchip.max_write_attempts_per_device is not None:
            cap = lc.onchip.max_write_attempts_per_device
            cost = row['cost']
            if (not 0 <= cost['max_onchip_device_writes'] <= cost['max_onchip_device_write_attempts'] <= cap
                    or cost['issued_write_attempts'] != cost['device_writes'] + cost['stuck_write_attempts']):
                raise ValueError('Thresholded per-device write-attempt cap violated')
    flat = [{**r,'duplicate_coverage':False} for r in _measurement_rows(rows,run)]
    if json.loads(payload['rows_json']) != flat:
        raise ValueError('Native collect differs from measurements')
    fields = sorted({k for row in flat for k in row})
    if list(csv.DictReader(io.StringIO(payload['rows_csv']))) != [
            {k: '' if row.get(k) is None else str(row[k]) for k in fields} for row in flat]:
        raise ValueError('Collected CSV differs from validated rows')
    for line in files['metrics.jsonl'].splitlines():
        if line.strip(): finite(json.loads(line))
    return payload


def export(root, inputs):
    contract_text = (root / 'campaign.resolved.json').read_text(); contract=json.loads(contract_text)
    aggregate_text = (root / 'aggregate/results.json').read_text(); records=json.loads(aggregate_text)['stages']
    if len(records) != 1 or records[0]['status'] != 'complete': raise ValueError('Recovery incomplete')
    run = Path(records[0]['run_result']).parent
    command = [sys.executable,'-m','ebl','runs','inspect',str(run),'--verify-artifacts','--require-complete','--json']
    inspection = subprocess.run(command,check=True,capture_output=True,text=True,timeout=180).stdout
    files = {name:(run/name).read_text() for name in (*CONTROL_FILES,'measurements.json')}
    files['inspection.json']=inspection
    files['stage-config.json']=Path(contract['stages'][0]['config']).read_text()
    with tempfile.TemporaryDirectory(prefix='recovery-collect-') as tmp:
        subprocess.run([sys.executable,'-m','workflow','collect',str(run),'--output',tmp],check=True,capture_output=True,text=True,timeout=180)
        payload = {'schema':'lifecycle.reused_recovery.v1','root':str(root),'run':str(run),
            'campaign_resolved':contract_text,'source_manifest':Path(contract['source_manifest']['path']).read_text(),
            'aggregate':aggregate_text,'files':files,'rows_json':(Path(tmp)/'rows.json').read_text(),
            'rows_csv':(Path(tmp)/'rows.csv').read_text()}
    return checked(payload,inputs)


def collect(queue_config, output):
    job_id=os.environ['RUN_WATCH_JOB_ID']; cfg=json.loads(queue_config.read_text())
    dbpath=Path(cfg.get('state_dir','~/.local/state/run-watch/queue')).expanduser()/'queue.sqlite3'
    job,attempt=exact_attempt(dbpath,job_id); placement=json.loads(job['spec'])['placements'][attempt['target']]
    target=cfg['targets'][attempt['target']]
    argv=[placement['campaign_python'],'-m','ebl.execution.recovery_collect','export',placement['campaign_result_root'],
          '--inputs',placement['recovery_inputs']]
    if target.get('transport') == 'local': command,cwd=argv,placement['workdir']
    else:
        command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',*target.get('ssh_options',[]),target['host'],
                 'cd '+shlex.quote(placement['workdir'])+' && '+shlex.join(argv)]; cwd=None
    completed=subprocess.run(command,cwd=cwd,capture_output=True,text=True,check=True,timeout=600)
    payload=json.loads(completed.stdout)
    inputs=placement['recovery_input_contract']
    checked(payload,inputs)
    with collection_lock(output):
        _, current=exact_attempt(dbpath,job_id)
        if current['token'] != attempt['token']: raise ValueError('Attempt changed during collection')
        dest=output/job_id/attempt['token']; dest.mkdir(parents=True,exist_ok=True)
        values={**payload['files'],'rows.json':payload['rows_json'],'rows.csv':payload['rows_csv'],
                'export.json':json.dumps(payload,sort_keys=True)}
        for name,value in values.items(): write_frozen(dest/name,value)
        receipt={'job':job_id,'attempt':attempt['token'],'target':attempt['target'],'run':payload['run'],
                 'files':{name:text_sha(value) for name,value in values.items()}}
        write_frozen(dest/'collection.json',json.dumps(receipt,sort_keys=True)+'\n')
    return {'collected':str(dest)}


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    for mode in ('export','validate'):
        a=sub.add_parser(mode); a.add_argument('root',type=Path); a.add_argument('--inputs',type=Path,required=True)
    a=sub.add_parser('collect'); a.add_argument('--queue-config',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='collect': result=collect(a.queue_config,a.output)
    else:
        result=export(a.root,json.loads(a.inputs.read_text()))
        if a.mode=='validate': result={'valid':True,'rows':len(json.loads(result['rows_json']))}
    print(json.dumps(result,allow_nan=False))

if __name__=='__main__': main()

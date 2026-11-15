"""CPU artifact verification/collection for the prepared two-case scaling run."""
import argparse,json,os,subprocess,sys
from pathlib import Path


def validate(root):
    terminal=json.loads((root/'result.json').read_text())
    if terminal['state']!='complete':raise RuntimeError('Supervisor did not complete')
    rows={};bundles=[]
    for case in ['drn','digital']:
        candidates=[r for r in terminal['cases'] if r['case']==case]
        if len(candidates)!=1 or candidates[0]['state']!='complete':raise RuntimeError(f'Missing completed case {case}')
        original=Path(candidates[0]['bundle'])
        # Native result identity remains unchanged when copied to the controller.
        bundle=root/case/original.name;bundles.append(str(bundle))
        result=json.loads((bundle/'artifacts/summary.json').read_text())
        if result['updates']<=0 or result['test']['count']!=10000 or result['selected_validation']['count']!=5000:
            raise RuntimeError(f'Incomplete evaluation in {case}')
        rows[case]={'bundle':str(bundle),'summary':result}
    with (root/'terminal-inspection.json').open('w') as output:
        subprocess.run([sys.executable,'-m','ebl','runs','inspect',*bundles,'--verify-artifacts','--require-complete','--json'],stdout=output,check=True)
    (root/'collected-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--remote');p.add_argument('--remote-root',default='/tmp/pseudo-drn-large-20261115')
    p.add_argument('--thread');args=p.parse_args();root=args.root
    if args.remote:
        if args.remote!='nom-cool-1':raise ValueError('Only the authorized fallback is supported')
        root.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a',f'{args.remote}:{args.remote_root}/',str(root)+'/'],check=True)
    rows=validate(root)
    if args.remote:
        os.environ.setdefault('MPLCONFIGDIR','/tmp/pdl-matplotlib')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for name,row in rows.items():
            metrics=[json.loads(line) for line in (Path(row['bundle'])/'metrics.jsonl').read_text().splitlines()]
            metrics=[r for r in metrics if r.get('stage')=='validation']
            ax.plot([r['epoch'] for r in metrics],[r['accuracy'] for r in metrics],label=name)
        ax.set(xlabel='Epoch',ylabel='Validation accuracy',title='CIFAR-10: six attention blocks, one seed')
        ax.legend();ax.grid(alpha=.2);fig.savefig(root/'learning.png',dpi=180)
    if args.thread:
        message=(f'The authorized large attention–DRN CIFAR job has finished execution and artifact collection. '
                 f'Read {root}/collected-summary.json, {root}/terminal-inspection.json and issue53 handoff. '
                 'Review achieved epoch coverage, accuracy and gradient checks honestly; finish the existing experiment records and issue/worktree closeout. '
                 'Do not infer new GPU budget or launch additional experiments. This is the existing task continuation, not a new task.')
        try:
            delivered=subprocess.run(['/home/filip/.local/bin/codex','queue','--thread',args.thread,'--resume','--message',message],capture_output=True,text=True,timeout=30)
            callback={'returncode':delivered.returncode,'stdout':delivered.stdout[-2000:],'stderr':delivered.stderr[-2000:]}
        except (OSError,subprocess.TimeoutExpired) as error:
            callback={'error':str(error),'delivered':False}
        (root/'callback.json').write_text(json.dumps(callback,indent=2))
    print(json.dumps({n:r['summary'] for n,r in rows.items()}))


if __name__=='__main__':main()

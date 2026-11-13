"""Plot a completed native MOS attention bundle without running a model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('output_dir',type=Path)
    args=parser.parse_args()
    root=args.run_dir/'artifacts';out=args.output_dir
    manifest=json.loads((args.run_dir/'manifest.json').read_text())
    if manifest['experiment_id']!='mos_attention_learning.v1': raise ValueError('Wrong experiment family')
    summary=json.loads((root/'summary.json').read_text())
    history=json.loads((root/'history.json').read_text())
    initial=np.load(root/'initial-test.npz');final=np.load(root/'final-test.npz')
    ablation=np.load(root/'no-attention-test.npz');data=np.load(root/'data.npz')
    for batch in (initial,ablation):
        np.testing.assert_array_equal(batch['pairs'],final['pairs'])
        np.testing.assert_array_equal(batch['target'],final['target'])
    out.mkdir(parents=True,exist_ok=False)
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(10,4.1),constrained_layout=True)
        epochs=[row['epoch'] for row in history];losses=[row['train_mse'] for row in history]
        axes[0].semilogy(epochs,losses,color='#355C9B',lw=2,label='Measured SPICE EP')
        axes[0].axhline(np.mean(data['train_pairs'].mean(1)**2),color='#909090',ls='--',label='Zero predictor on training pairs')
        axes[0].set(xlabel='Completed weight updates',ylabel='Training MSE',title='Circuit training')
        axes[0].legend(frameon=False,fontsize=8)
        x=final['target'];axes[1].plot([-.4,.4],[-.4,.4],color='#808080',lw=1,ls='--')
        for batch,label,color,marker in ((initial,'Initial','#999999','o'),
                                         (final,'Trained','#355C9B','o'),
                                         (ablation,'Trained; attention removed','#C47435','x')):
            axes[1].scatter(x,batch['predictions'],s=24,c=color,marker=marker,label=label,alpha=.85)
        axes[1].set(xlabel='Target mean of the two input tokens',ylabel='Masked-token output',
                    title=f"Held-out inputs ({len(x)} pairs)",xlim=(-.42,.42),ylim=(-.42,.42))
        axes[1].set_aspect('equal',adjustable='box');axes[1].legend(frameon=False,fontsize=8)
        fig.suptitle('EP learning with native MOS attention',fontsize=13)
        fig.savefig(out/'learning.png',dpi=180)
        fig.savefig(out/'learning.svg')
        plt.close(fig)
    sources=['summary.json','history.json','data.npz','initial-test.npz','final-test.npz','no-attention-test.npz']
    metadata={'run_dir':str(args.run_dir.resolve()),'source':manifest['source'],
              'inputs':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources},
              'result':{name:summary[name] for name in ('passed','initial_test_mse','final_test_mse','no_attention_test_mse')},
              'claim_boundary':summary['claim_boundary']}
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__': main()

"""Draw one shared V100 target tensor, requiring the exact historical1024 prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASELINE = 'df73ce6615be3cdd3005d2ca0b97c03131f9236ba481bc7250edde5ff66fb01a'
TEACHER = '2d74da6615135c58cf3cf9ad4cb11e7c613ff9e55fe658a47ab83b6c8d1174a9'


def tensor_hash(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def validate_prefix(samples):
    if tuple(samples.shape[:2]) != (1024,255) or samples.shape[-1] != 8 or tensor_hash(samples) != BASELINE:
        raise ValueError('Historical teacher target prefix changed; do not publish or train')
    return BASELINE


def draw_chunks(tokens, logits, generator, batch=16):
    """Preserve generator order while retaining only target IDs on CPU."""
    for chunk in tokens.split(batch):
        probs = logits(chunk).float().softmax(-1)
        shape = probs.shape[:-1]
        samples = torch.multinomial(probs.flatten(0,1),8,replacement=True,generator=generator)
        yield samples.view(*shape,8).cpu()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', type=Path)
    ap.add_argument('teacher', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite previous target outputs')
    spec = json.loads(Path(__file__).with_name('cohorts.json').read_text())
    if file_hash(args.corpus) != spec['corpus_sha256'] or file_hash(args.teacher) != TEACHER:
        raise ValueError('Teacher or corpus differs from pinned input')
    corpus = torch.load(args.corpus,map_location='cpu',weights_only=False)
    stream=corpus['splits']['train']; length=spec['sequence_length']
    blocks=stream[:len(stream)//length*length].reshape(-1,length)
    order=torch.randperm(len(blocks),generator=torch.Generator().manual_seed(spec['selection_seed']))
    tokens=blocks[order[:8192]].contiguous()
    for c in spec['cohorts']:
        if tensor_hash(tokens[:c['sequences']]) != c['sha256']:
            raise ValueError('Recovery cohort identity mismatch')
    from networks.opt.data import pinned_load
    from networks.opt.decoder import from_hugging_face
    from ebl.runtime.opt_cmo_drn.runtime import MODEL
    from methods.drift_regimes.forward import teacher_logits,teacher_prefix
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    teacher=from_hugging_face(pinned_load(args.teacher,TEACHER),MODEL).eval().requires_grad_(False).to(args.device)
    generator=torch.Generator(device=args.device).manual_seed(20365833)
    def logits(chunk):
        return teacher_logits(teacher,6,teacher_prefix(teacher,chunk.to(args.device),6))
    parts=[]
    for i,part in enumerate(draw_chunks(tokens,logits,generator),start=1):
        parts.append(part)
        if i*16==1024:
            validate_prefix(torch.cat(parts))
            print(json.dumps({'event':'original_prefix_verified','sha256':BASELINE}),flush=True)
        if i%32==0:
            print(json.dumps({'event':'target_progress','sequences':i*16}),flush=True)
    samples=torch.cat(parts).contiguous()
    validate_prefix(samples[:1024])
    if list(samples.shape)!=[8192,255,8]:
        raise ValueError('Unexpected teacher target shape')
    args.output.mkdir(parents=True,exist_ok=False)
    digest=tensor_hash(samples); name=f'teacher-samples8-n8192-{digest[:12]}.pt'; path=args.output/name
    torch.save(samples,path)
    c=next(c for c in spec['cohorts'] if c['sequences']==8192)
    receipt={'schema':'fault_token_targets.v1','status':'complete','baseline_prefix_sha256':BASELINE,
             'teacher_sha256':TEACHER,'corpus_sha256':spec['corpus_sha256'],'samples_seed':20365833,
             'draw_batch':16,'count':8,'gpu':torch.cuda.get_device_name(args.device),'torch':torch.__version__,
             'samples':[{'sequences':8192,'cohort_sha256':c['sha256'],'shape':list(samples.shape),
                         'file':name,'file_sha256':file_hash(path),'tensor_sha256':digest}]}
    (args.output/'targets-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()

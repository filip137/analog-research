"""Prepare pinned CAD sentiment pairs; no test split is read for this pilot."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REVISION = '6f232a1d2a11462a30ce08fb4825b734ab30828e'
BLOBS = {'train_paired.tsv':'637196cbf29f7b57a7f064bf1e43df7534852a97',
         'dev_paired.tsv':'26c3fbb6d492d2a4152e3cd72c184c00c2768c1e',
         'train_orig.tsv':'fe4c24bb8f89919dc2a7784697f1fed254f6e643',
         'dev_orig.tsv':'fc74d83fc2a580514b4ebeb0b51bcd7dd24d0cfa'}

def read_pairs(root, split):
    def rows(name):
        return list(csv.DictReader((root/name).open(), delimiter='\t'))
    # Distribution files differ in whitespace; never use fuzzy semantic matching.
    originals = {' '.join(r['Text'].split()) for r in rows(f'{split}_orig.tsv')}
    groups = defaultdict(list)
    for row in rows(f'{split}_paired.tsv'):
        groups[row['batch_id']].append(row)
    out = []; excluded=[]
    for key, pair in groups.items():
        if len(pair)!=2 or {r['Sentiment'] for r in pair}!={'Positive','Negative'}:
            raise ValueError(f'Invalid counterfactual pair {split}/{key}')
        orig = [r for r in pair if ' '.join(r['Text'].split()) in originals]
        if len(orig)!=1:
            excluded.append({'pair_id':f'{split}/{key}','reason':'ambiguous_original_identity'})
            continue
        revised = next(r for r in pair if r is not orig[0])
        out.append((f'{split}/{key}', orig[0], revised))
    return out,excluded

def prepare(root, tokenizer_dir):
    from transformers import GPT2Tokenizer
    for name, expected in BLOBS.items():
        data=(root/name).read_bytes()
        actual=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if actual!=expected:
            raise ValueError(f'Pinned Git blob mismatch: {name}')
    tok=GPT2Tokenizer.from_pretrained(tokenizer_dir,local_files_only=True)
    label_ids=[tok.encode(' negative',add_special_tokens=False),tok.encode(' positive',add_special_tokens=False)]
    if any(len(v)!=1 for v in label_ids):
        raise ValueError('Expected single-token sentiment labels.')
    prompt_a=tok.encode('Review: ',add_special_tokens=False)
    prompt_b=tok.encode('\nSentiment:',add_special_tokens=False)
    doc={'schema':'sparse-equilibrium.cad.v1','revision':REVISION,'label_ids':[v[0] for v in label_ids],
         'max_tokens':128,'splits':{},'excluded':{},'files':{}}
    seen=set()
    for split in ('train','dev'):
        pairs,excluded=read_pairs(root,split)
        kept=[]; current=set()
        for key,orig,rev in pairs:
            texts=[' '.join(r['Text'].split()) for r in (orig,rev)]
            if any(t in seen for t in texts):
                excluded.append({'pair_id':key,'reason':'text_overlap_with_training'});continue
            current.update(texts)
            encoded=[]
            for row in (orig,rev):
                text_ids=tok.encode(row['Text'],add_special_tokens=False)
                ids=[2]+prompt_a+text_ids[:128-1-len(prompt_a)-len(prompt_b)]+prompt_b
                encoded.append({'tokens':ids,'label':int(row['Sentiment']=='Positive'),
                                'text_sha256':hashlib.sha256(row['Text'].encode()).hexdigest(),
                                'truncated':len(text_ids)>128-1-len(prompt_a)-len(prompt_b)})
            if encoded[0]['tokens']==encoded[1]['tokens']:
                excluded.append({'pair_id':key,'reason':'identical_prompt_tokens'}); continue
            kept.append({'pair_id':key,'original':encoded[0],'revised':encoded[1]})
        seen.update(current)
        doc['splits'][split]=kept; doc['excluded'][split]=excluded
    for name in BLOBS:
        doc['files'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    (root/'sentiment.json').write_text(json.dumps(doc))
    print(json.dumps({'pairs':{k:len(v) for k,v in doc['splits'].items()},
                      'excluded':{k:len(v) for k,v in doc['excluded'].items()},'label_ids':doc['label_ids']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--tokenizer',required=True)
    a=p.parse_args(); prepare(a.root,a.tokenizer)

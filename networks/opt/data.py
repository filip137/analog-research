"""Hash-pinned token streams and upstream-compatible cohort ordering."""

import os
from pathlib import Path
import torch
from core.artifacts import sha256_file
from hardware.faults import tensor_hash

TRANSFER_TESTS=(
    ('generated','wikitext2-train-synthtest','8e73b6b43aafb92b8b52e11cdd060d573331fae410350cf2427f3eeba640e224',384),
    ('wikitext','wikitext-2-raw-v1','33e958349a446c09be3768491db7ac98eea45e38b6b884c59eb970d8b2e50e01',1024),
)


def pinned_load(path,digest):
    path=Path(path)
    actual=sha256_file(path)
    if actual!=digest:
        raise ValueError(f'Expected {path} SHA-256 {digest}; received {actual}.')
    return torch.load(path,map_location='cpu',weights_only=True)


def load_corpus(name,digest):
    root=os.environ.get('EBL_CORPUS_ROOT')
    if not root: raise ValueError('Expected EBL_CORPUS_ROOT for explicit pinned token corpora.')
    payload=pinned_load(Path(root)/f'{name}.pt',digest)
    if payload.get('schema')!='crossbar_lifecycle.token_corpus.v1' or payload.get('model')!='facebook/opt-125m':
        raise ValueError('Expected pinned OPT token corpus schema.')
    for value in payload['splits'].values():
        if value.dtype!=torch.int64 or value.ndim!=1 or len(value)==0 or value.min()<0 or value.max()>=50272:
            raise ValueError('Expected nonempty int64 OPT token streams.')
    return payload['splits']


def blocks(stream,length,count):
    available=len(stream)//length
    if count>available: raise ValueError(f'Expected {count} sequences; only {available} available.')
    return stream[:available*length].reshape(-1,length)[:count]


def cohorts(config):
    d=config.data
    splits=load_corpus(d.corpus,d.sha256)
    available=len(splits['train'])//d.sequence_length
    train=blocks(splits['train'],d.sequence_length,available)
    order=torch.randperm(available,generator=torch.Generator().manual_seed(d.seed))
    if d.sequences>available: raise ValueError('Training cohort exceeds pinned corpus.')
    train=train[order[:d.sequences]]
    development=blocks(splits['validation'],d.sequence_length,d.development_sequences)
    return train,development,{'training_sha256':tensor_hash(train),'development_sha256':tensor_hash(development),
                              'training_sequences':len(train),'input_tokens_per_epoch':train.numel()}

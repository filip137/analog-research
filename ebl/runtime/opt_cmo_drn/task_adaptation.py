"""Native task-adaptation stages on the same mapped CMO DRN bank as fault recovery."""
from dataclasses import asdict
from pathlib import Path
import os
import time

import torch

from ebl.runtime.opt_cmo_drn import runtime
from ebl.runtime.opt_cmo_drn.fault_tuning import _store, restored_mapping_check
from ebl.runtime.store import atomic_write_json, sha256_file
from methods import drn_task_adaptation as learn
from methods.programming.cmo import apply_mapping
from methods.task_adaptation import count, evaluate
from networks.opt.physical import CmoDrnBank


HWA_DIGESTS = {6: 'dfb9f22fa6a922952aeddd2f1c8f29b4a40e129fe0d2296f8343decbc309bcb9',
               12: '6805b62950743838de89ed356cbaf5e92a98710cf87c3f7b11093be89c4a8a1a'}


def _bundle(name, digest):
    root = Path(os.environ['EBL_CORPUS_ROOT'])
    path = root / (name + '.pt')
    if sha256_file(path) != digest:
        raise ValueError(f'Changed task data: {name}.')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data['schema'] != 's7-task.v1' or data['task'] != name:
        raise ValueError(f'Incorrect task data: {name}.')
    return data, {'role': name, 'path': str(path.resolve()), 'sha256': digest}


def run(request):
    c = request.spec
    s = c.task_adaptation
    if (not request.teacher_weights or not request.weights or any(getattr(request, k, None) for k in
            ('base_weights', 'device_data', 'device_model', 'deployment', 'resume'))):
        raise ValueError('Task adaptation needs explicit OPT teacher and task-agnostic HWA weights.')
    if bool(request.device_state) != (s.information in ('proxy','recipient')):
        raise ValueError('Clean fits consume only a mapped source. Proxy/recipient stages consume their own bank.')
    if bool(request.task_fit) != (s.action == 'receive'):
        raise ValueError('Receiving a trained adapter or sparse/full updates requires --task-fit.')
    if bool(request.task_masks) != (s.method == 'child' and s.action in ('fit','learn')):
        raise ValueError('Child-Tuning training requires its matching --task-masks artifact.')
    runtime.configure_runtime(c)
    torch.backends.cudnn.allow_tf32 = False
    device = c.device
    task, task_id = _bundle(s.task, s.task_sha256)
    wiki, wiki_id = _bundle('wikitext', s.wiki_sha256)
    probes = wiki['splits']['train']['tokens'][:s.calibration_items].long()
    if len(probes) != s.calibration_items or count(task['splits']['train']) < s.fisher_items:
        raise ValueError('Insufficient calibration or task data.')
    receipts = [task_id, wiki_id]
    for key in ('task_fit','task_masks'):
        path = getattr(request,key)
        if path:
            receipts.append({'role':key, 'path':str(path.resolve()), 'sha256':sha256_file(path)})
    store = _store(request, receipts)
    started = time.monotonic()

    def event(name, **fields):
        elapsed = time.monotonic() - started
        store.append_metric({'event': name, 'elapsed_seconds': elapsed, **fields})
        if elapsed > c.training.max_wall_seconds:
            raise TimeoutError('Task-adaptation walltime exceeded.')

    try:
        source_digest = sha256_file(request.weights)
        if source_digest != HWA_DIGESTS[c.analog_layers]:
            raise ValueError('Task stage requires the pinned task-agnostic HWA checkpoint for this depth.')
        teacher = runtime.teacher_model(request.teacher_weights,c)
        model = runtime.from_mapped(teacher,c,torch.load(request.weights, map_location='cpu', weights_only=True))
        model.to(device).eval().requires_grad_(False)
        teacher.to(device)
        split = task['splits'][s.split]
        limit = s.limit or None
        def task_metrics(forward):
            return evaluate(forward, split, device, s.eval_microbatch, limit=limit)

        baseline = {}
        if s.action == 'controls':
            baseline['fp32_teacher'] = task_metrics(lambda ids: teacher(ids)[:,:-1])
            baseline['ideal_source'] = task_metrics(model)
        ideal_blocks = [{k:t.detach().to('cpu',copy=True) for k,t in block.state_dict().items()} for block in model.blocks]
        mapping = apply_mapping(model,3.)
        if s.action == 'controls':
            baseline['mapped_target'] = task_metrics(model)
        bank = None
        bank_digest = None
        if request.device_state:
            physical = torch.load(request.device_state,map_location='cpu',weights_only=True)
            if (physical['schema'] != 'opt_cmo_drn.deployment.v1' or physical['source_weights_sha256'] != source_digest
                    or physical['deployment'] != asdict(c.deployment)):
                raise ValueError('Mismatch between requested source, mapping, fault case and bank.')
            restored_mapping_check(model,physical,mapping)
            bank = CmoDrnBank(physical['bank'],device)
            if bank.held_hashes() != physical['start_held_hashes']:
                raise ValueError('Physical bank changed during restoration.')
            del physical
            bank_digest = sha256_file(request.device_state)
        identity = {'task_sha256':s.task_sha256, 'wiki_sha256':s.wiki_sha256, 'task':s.task,
                    'source_sha256':source_digest, 'teacher_sha256':c.teacher_sha256,
                    'analog_layers':c.analog_layers, 'mapping_scale':c.deployment.mapping_scale,
                    'seed':s.seed,'calibration_items':s.calibration_items}
        training_identity = {**identity,'information':s.information,'bank_sha256':bank_digest,
                             'fisher_items':s.fisher_items}
        event('task_inputs_verified', bank_sha256=bank_digest, task=s.task, action=s.action, method=s.method)
        artifacts = []

        held_view = bank.view(model) if bank else None
        def retention(forward):
            return evaluate(forward, wiki['splits'][s.split], device, s.eval_microbatch,
                            teacher_fn=lambda ids: teacher(ids)[:,:-1], limit=limit)
        if s.action == 'controls':
            raw_forward = lambda ids: model(ids, held_view)
            baseline['fault_aware_hold'] = {'task':task_metrics(raw_forward),'retention':retention(raw_forward)}
        correction = learn.output_calibration(model,held_view,probes,device,s.eval_microbatch,ideal_blocks)
        del ideal_blocks
        correction.attach(model)
        event('task_calibration', probe_blocks=len(probes), chips_specific=(bank is not None))
        def view_fn():
            if s.information == 'proxy':
                return learn.proxy_view(model,bank)
            if s.action == 'learn' and s.method == 'child':
                return bank.view(model)
            return held_view
        def forward(ids):
            return model(ids,view_fn())
        extra = {}
        adapters = None
        if s.action == 'rank':
            # Ranking uses an exactly fixed initial view across the entire Fisher pass.
            ranked_view = view_fn()
            ranking = learn.rank_masks(model,bank,lambda ids: model(ids,ranked_view),task['splits']['train'],s,
                                       lambda i,n: event('task_fisher',batch=i,batches=n))
            ranked = {'schema':'drn.task_masks.v1', 'identity':training_identity, **ranking}
            path = store.run_dir/'checkpoints/task_masks.pt'
            runtime.save(path, ranked)
            artifacts.append(store.artifact_record(path, kind='task_masks'))
            extra = {'eligible':ranking['eligible'], 'counts':{k:v['count'] for k,v in ranking['fractions'].items()}}
        elif s.action in ('fit','learn'):
            masks = None
            if request.task_masks:
                ranked = torch.load(request.task_masks,map_location='cpu',weights_only=True)
                if ranked['schema'] != 'drn.task_masks.v1' or ranked['identity'] != training_identity:
                    raise ValueError('Fisher mask came from a different task, chip or selection budget.')
                masks = learn.unpack_masks(ranked,s.fraction,model)
            adapters, extra = learn.train(model,bank,view_fn,task,s,masks,s.action=='learn',
                                          lambda i,n,loss: event('task_update',step=i,steps=n,loss=loss))
            if s.action == 'fit':
                fit = {'schema':'drn.task_fit.v1','identity':identity,'training':training_identity,
                       'method':s.method,'fraction':s.fraction,'lr':s.lr,'steps':s.steps,
                       'training_costs':extra, 'parameters':learn.export_fit(model,s.method,adapters,masks)}
                path = store.run_dir/'checkpoints/task_fit.pt'
                runtime.save(path,fit)
                artifacts.append(store.artifact_record(path,kind='task_fit'))
        elif s.action == 'receive':
            fit = torch.load(request.task_fit,map_location='cpu',weights_only=True)
            if (fit['schema'] != 'drn.task_fit.v1' or fit['identity'] != identity or fit['method'] != s.method
                    or fit['fraction'] != s.fraction or fit['lr'] != s.lr or fit['steps'] != s.steps
                    or fit['training']['bank_sha256'] == bank_digest):
                raise ValueError('Unmatched fit, or the offline trainer saw the recipient.')
            adapters, extra = learn.transfer(model,bank,fit['parameters'],s.method,s.seed)
            extra['offline_training_costs'] = fit['training_costs']
            extra['training_information'] = fit['training']['information']
            del fit
        if s.action in ('controls','receive','learn'):
            held_view = bank.view(model)
            endpoint = lambda ids: model(ids,held_view)
            row = {'task':task_metrics(endpoint),'retention':retention(endpoint)}
            if s.action == 'controls':
                baseline['calibrated_hold'] = row
            extra['endpoint'] = row
            extra['bank_costs'] = bank.costs()
            if s.method == 'lora' and bank.costs()['write_attempts'] != 0:
                raise RuntimeError('Digital LoRA issued an analog write.')
            for array in bank.arrays:
                array.check()
        if adapters:
            adapters.detach_hooks()
        correction.detach_hooks()
        summary = {**identity, **extra, 'settings':asdict(s), 'deployment':asdict(c.deployment),
                   'bank_sha256':bank_digest, 'baselines':baseline,
                   'seconds':time.monotonic()-started,
                   'peak_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0}
        path = store.run_dir/'summary.json'
        atomic_write_json(path,summary)
        artifacts.append(store.artifact_record(path,kind='task_summary'))
        store.complete(metrics={'seconds':summary['seconds'],'action':s.action,'task':s.task,
                                'method':s.method, 'split':s.split},artifacts=artifacts)
        return 0
    except BaseException as error:
        store.fail(error)
        raise

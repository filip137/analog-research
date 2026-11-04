"""Bounded architecture-migration replay captured before code relocation."""
import hashlib
import json
import torch


def digest(value):
    def signature(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return ["tensor", str(item.dtype), list(item.shape), hashlib.sha256(data).hexdigest()]
        if isinstance(item, dict):
            return ["dict", [[k, signature(v)] for k, v in sorted(item.items())]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [signature(v) for v in item]]
        return [type(item).__name__, item]
    return hashlib.sha256(json.dumps(signature(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def opt_traces():
    from test_opt_cmo_drn import tiny_model
    from networks.opt.adapter import DrnAdapter
    from methods.hwa.opt import hardware_view
    torch.set_num_threads(1)
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    traces = {}
    for checkpointed in (False, True):
        for learning in (False, True):
            for physical in (False, True):
                torch.manual_seed(43)
                model, _ = tiny_model(checkpointed)
                model.train(learning)
                if physical:
                    adapter = DrnAdapter(model, "cpu")
                    held, groups, roles, names = adapter.physical_layout()
                    held = [g.clone().requires_grad_(learning) for g in held]
                    correction = [(torch.tensor(1.03), torch.tensor(.01)) for _ in model.blocks]
                    out, captures = adapter.forward(tokens, held, correction, capture=True)
                    if learning:
                        out.square().mean().backward()
                    extra = dict(captures=captures, held=held, gradients=[g.grad for g in held],
                                 groups=groups, roles=roles, names=names)
                else:
                    view, projected = hardware_view(model, .03, torch.Generator().manual_seed(91))
                    out = model(tokens, view)
                    if learning:
                        out.square().mean().backward()
                    extra = dict(view=view, projected=projected, gradients=[p.grad for p in model.parameters()])
                value = dict(logits=out, state=model.state_dict(), **extra)
                traces[str((checkpointed, learning, physical))] = {k: digest(v) for k, v in value.items()}
    return traces

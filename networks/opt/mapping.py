"""Network-declared physical layouts and fault domains."""
from hardware.faults import MatrixSpec

def fault_group(name: str) -> str:
    """The module owning an analog matrix, e.g. ``decoder.layers.11`` for its fc1 and fc2."""

    return name.rsplit(".", 1)[0]

def programming_layout(model):
    layout, masters, domains = [], [], {}
    offset = 0
    for k, block in enumerate(model.blocks):
        group = f'decoder.layers.{model.start+k}'
        for name, u in zip(('fc1', 'fc2'), (block.u1, block.u2)):
            key = f'{group}.{name}'
            layout.append(MatrixSpec(key, (u.shape[1], u.shape[0]), offset, u.numel()))
            masters.append(u)
            domains[key] = group
            offset += u.numel()
    return layout, masters, domains

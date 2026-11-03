"""Physical grouping of the supported network families' named matrices."""
from hardware.mapping import FaultDomain


def fault_domains(layout):
    """Keep base MLP groups stable when separately grouped adapters are added.

    This derives the existing OPT/CIFAR naming contract without changing the
    serialized MatrixSpec fields, receipts or random stream namespaces.
    """
    return {matrix.name: FaultDomain(matrix.name.rsplit('.', 1)[0],
                                     matrix.name.endswith(('.lora_a', '.lora_b')))
            for matrix in layout}

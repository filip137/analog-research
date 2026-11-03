"""One-MLP device-state resource probe, executed only inside a GPU allocation."""
import json
import os
import time
from pathlib import Path
import pytest
import torch
from training.flowers_drift import FlowersLaw, FlowersState


@pytest.mark.skipif(not torch.cuda.is_available() or 'FLOWERS_PROBE_MODEL' not in os.environ,
                    reason='Explicit staged Flowers model and allocated GPU required')
def test_one_mlp_state_footprint():
    law = FlowersLaw(json.loads(Path(os.environ['FLOWERS_PROBE_MODEL']).read_text()), 'cuda:0')
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    n = 2*768*3072
    anchor = torch.full((2,n),48.6,device='cuda:0')
    state = FlowersState(law,anchor,2.,(391030,271001,'resource-probe'))
    state.read(62.)
    mask = torch.zeros_like(anchor,dtype=torch.bool);mask[:,:n//5]=True
    state.rewrite(anchor,mask,64.)
    physical,_=state.read(300.)
    assert torch.isfinite(physical).all() and (physical>=9).all() and (physical<=88.2).all()
    torch.cuda.synchronize()
    report={'schema':'flowers.device_state_probe.v1','logical_mlp_layers':1,
            'physical_devices':anchor.numel(),'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.monotonic()-started,'torch_version':str(torch.__version__)}
    Path(os.environ['FLOWERS_PROBE_REPORT']).write_text(json.dumps(report,indent=2)+'\n')

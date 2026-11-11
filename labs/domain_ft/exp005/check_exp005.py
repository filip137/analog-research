"""exp-005 GPU checks on the real std3 source (run once on a Jean Zay node before the screen):

python labs/domain_ft/exp005/check_exp005.py <root>

1. The on-chip base (build_model "std3_cp<c>_d<draw>") equals, bit for bit, the std3 base deploy.py programs at
   (c_p, draw) -- the chip L2 trains on is the chip it is evaluated on, and the chip B0/L0/L1 are evaluated on.
2. The per-update HWA law in train.py (--hwa-cp, GPU generator) has the same statistics as model.program_weights at
   that c_p: per matrix, the std of (W_noisy - W_clean)/s_j agrees within 1.5 % and the mean within 0.02 sigma.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import deploy as D  # noqa: E402
import model as M  # noqa: E402


def main(root: Path):
    dev = torch.device("cuda")
    opt_path, s3 = root / "model" / "pytorch_model.bin", root / "inputs" / "crossbar-std3.pt"
    scales = M.std3_scales(s3)
    clean, _ = M.build_model(opt_path, "std3_clean", s3, dev)
    base_state = {k: v.detach().clone() for k, v in clean.state_dict().items()}
    for cp, draw in ((1.0, 271002), (10.0, 271004), (3.0, 271003)):
        onchip, rec = M.build_model(opt_path, f"std3_cp{cp:g}_d{draw}", s3, dev)
        dep, _ = D.deployed_state(base_state, scales, cp, draw, "rescale")
        sd = onchip.state_dict()
        bad = [k for k in sd if not torch.equal(sd[k], dep[k])]
        assert not bad, (cp, draw, bad[:4])
        print(f"on-chip base == deploy B0 at c_p={cp:g} draw={draw}: all {len(sd)} tensors equal "
              f"(oor active {rec['programming']['out_of_range_active']})")
    gen = torch.Generator(device=dev).manual_seed(31)
    for cp in (1.0, 10.0):
        worst = 0.0
        for name, sc in scales.items():
            w = base_state[f"{name}.weight"]
            q, s, _ = M.deploy_targets(w, sc)
            gas = (M.G_MID + M.G_SPAN * q).float().to(dev)
            e = torch.randn(w.shape, generator=gen, device=dev)
            ga = (gas + cp * M.sigma_pv(gas) * e).clamp_(M.G_MIN, M.G_MAX)
            e.normal_(generator=gen)
            gr = (M.G_MID + cp * M.sigma_pv(M.G_MID) * e).clamp_(M.G_MIN, M.G_MAX)
            z_gpu = ((ga - gr) / M.G_SPAN - q.float().to(dev)).double()
            wh, _ = M.program_weights(w, sc, cp, 999, name)
            z_ref = (wh.double().cpu() / s[:, None] - q).to(dev)
            rs = float(z_gpu.std() / z_ref.std())
            dm = float((z_gpu.mean() - z_ref.mean()) / z_ref.std())
            worst = max(worst, abs(rs - 1))
            assert abs(rs - 1) < 0.015 and abs(dm) < 0.02, (cp, name, rs, dm)
        print(f"HWA law c_p={cp:g}: per-matrix std ratio within {worst:.4f} of program_weights")
    print("exp005 checks passed")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

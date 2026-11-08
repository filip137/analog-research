"""exp-002 CPU checks of the deployment law: python labs/domain_ft/exp002/test_exp002.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as M  # noqa: E402


def main():
    torch.manual_seed(0)
    w = torch.randn(3072, 768) * 0.05
    s = w.abs().amax(1).double() * 1.3
    s[:5] = w[:5].abs().amax(1).double() * 0.8
    w0, i0 = M.program_weights(w, s, 0.0, None, "layers.6.fc1")
    assert torch.equal(w0, w) and i0["channels_rescaled"] == 5
    s2 = s.clone()
    s2[:5] = w[:5].abs().amax(1).double()
    q = w.double() / s2[:, None]
    for cp in (1.0, 10.0):
        w1, _ = M.program_weights(w, s, cp, 271002, "layers.6.fc1")
        w1b, _ = M.program_weights(w, s, cp, 271002, "layers.6.fc1")
        assert torch.equal(w1, w1b)
        sig = cp * torch.sqrt(M.sigma_pv(M.G_MID + M.G_SPAN * q) ** 2 + M.sigma_pv(M.G_MID) ** 2) / M.G_SPAN
        z = (w1.double() / s2[:, None] - q) / sig
        assert abs(float(z.std()) - 1) < 0.01 and abs(float(z.mean())) < 0.01, (cp, float(z.std()))
        print(f"cp={cp}: z std {float(z.std()):.4f} mean {float(z.mean()):+.5f}")
    print("exp002 law tests passed")


if __name__ == "__main__":
    main()

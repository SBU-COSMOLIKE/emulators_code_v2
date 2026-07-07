#!/usr/bin/env python3
"""GE-C: eval_val partition invariance + eval-batch timing (torch-only).

Transcribed faithfully from the home note eval-bs-decoupling.md:202-300
(the ready-to-paste GE-C script, the spec of record). The harness runs
it on the workstation as ``python gates/checks/ge_c_eval_bs.py``: Part 1
must print "Part 1: PASS" (per-row chi2 from eval_val agrees across eval
batch sizes to rtol 1e-6), and Part 2 reports the eval-batch launch
saving on CUDA. Exits nonzero if Part 1 fails.

This is a verbatim transcription of a note's tested script, so the house
90-column / naming style is NOT retrofitted onto it (the same rule the
verbatim compute_data_vectors import follows). Only the exit status at
the end is added so the runner reads a pass/fail code.
"""
import time
import numpy as np
import torch
from emulator.training import eval_val, derive_eval_bs, _EVAL_BS_TARGET

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_VAL, N_IN, N_OUT = 5000, 8, 780
RTOL = 1e-6

C = torch.randn(N_VAL, N_IN, device=DEV)
DV = torch.randn(N_VAL, N_OUT, device=DEV)
model = torch.nn.Linear(N_IN, N_OUT).to(DEV).eval()


class ToyChi2:
    needs_params = False

    def chi2(self, pred, target):
        return ((pred - target) ** 2).sum(dim=1)   # per-row, (bs,)


loss = ToyChi2()
# thresholds stay on CPU: eval_val moves c to CPU before the frac
# comparison (as the real pipeline does with DEFAULT_THRESHOLDS), so a
# device tensor here would mismatch on CUDA (D-E1).
thresholds = torch.tensor([0.2, 0.5, 1.0])


def toy_data():
    return {"load_C": lambda rows: C[rows],
            "load_dv": lambda rows: DV[rows],
            "idx": np.arange(N_VAL)}


def per_row_chi2(bs):
    """eval_val's batch+pad loop, but returning the full (N_VAL,) c."""
    out = []
    with torch.no_grad():
        for s in range(0, N_VAL, bs):
            xb, yb = C[s:s + bs], DV[s:s + bs]
            n = xb.shape[0]
            if n < bs:
                xb = torch.cat([xb, C[:1].expand(bs - n, -1)], dim=0)
                yb = torch.cat([yb, DV[:1].expand(bs - n, -1)], dim=0)
            out.append(loss.chi2(model(xb), yb)[:n].clone())
    return torch.cat(out).cpu()


print(f"device {DEV}, target {_EVAL_BS_TARGET}, derived bs "
      f"{derive_eval_bs(n_val=N_VAL, target=_EVAL_BS_TARGET, load=N_VAL)}")

print("\n=== Part 1: partition invariance (rtol 1e-6) ===")
ref_c = per_row_chi2(N_VAL)
ref_med, ref_mean = ref_c.median().item(), ref_c.mean().item()
ref_frac = (ref_c[:, None] > thresholds.cpu()[None, :]).float().mean(0)
ok = True
for bs in (32, 517, 1000, 2048):
    med, mean, frac = eval_val(model=model, lossfn=loss, data=toy_data(),
                               load=N_VAL, bs=bs, thresholds=thresholds)
    c = per_row_chi2(bs)
    row_ok = torch.allclose(c, ref_c, rtol=RTOL, atol=1e-6)
    met_ok = (abs(med - ref_med) <= RTOL * abs(ref_med) + 1e-6 and
              abs(mean - ref_mean) <= RTOL * abs(ref_mean) + 1e-6 and
              torch.allclose(frac.cpu(), ref_frac, rtol=RTOL, atol=1e-6))
    ok &= row_ok and met_ok
    print(f"  bs={bs:5d}  per-row allclose={row_ok}  metrics={met_ok}")
print("Part 1:", "PASS" if ok else "FAIL")

print("\n=== Part 2: eval timing, training bs vs derived bs ===")
if DEV.type == "cuda":
    derived = derive_eval_bs(n_val=N_VAL, target=_EVAL_BS_TARGET, load=N_VAL)

    def timed(bs, reps=50):
        for _ in range(3):
            eval_val(model=model, lossfn=loss, data=toy_data(),
                     load=N_VAL, bs=bs, thresholds=thresholds)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            eval_val(model=model, lossfn=loss, data=toy_data(),
                     load=N_VAL, bs=bs, thresholds=thresholds)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    t32, td = timed(32), timed(derived)
    print(f"  bs=32     : {t32*1e3:7.2f} ms  ({-(-N_VAL // 32)} batches)")
    print(f"  bs={derived:<4d}  : {td*1e3:7.2f} ms  "
          f"({-(-N_VAL // derived)} batches)")
    print(f"  saving    : {(t32 - td)*1e3:7.2f} ms  (x{t32/td:.1f} launches)")
else:
    print("  (CPU: timing meaningless; run Part 2 on the 3060.)")

print("\nReal check: one `train_single ... bs: 32` run vs the pre-change "
      "build: epoch time drops ~0.3 s, metrics match to rtol 1e-6.")

# Added for the harness: the runner reads this exit code (Part 1 is the
# hard acceptance; Part 2 is informational timing).
import sys
sys.exit(0 if ok else 1)

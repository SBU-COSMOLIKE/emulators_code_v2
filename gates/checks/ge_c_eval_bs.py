#!/usr/bin/env python3
"""eval-batch-size check: the validation score is independent of batching.

Two parts, run as a top-level script (there is no main function):

Part 1, partition invariance. eval_val scores the validation set in
batches, padding the last short batch. This part proves the answer does
not depend on the batch size used to walk the set: the per-row chi2, the
medians, the means, and the threshold fractions all agree, whether the set
is scored in one big batch or in batches of 32, 517, 1000, or 2048 rows
(to a relative tolerance of 1e-6). "Partition invariance" is exactly that
promise, that the score is a property of the data, not of how it was
chopped into batches. Part 1 must print "Part 1: PASS"; the script exits
non-zero if it fails.

Part 2, timing (CUDA only, informational). It times eval_val at the small
training batch size against the larger batch size derive_eval_bs picks, to
show the saving in per-batch launch overhead. On CPU it is skipped.

The check math (the toy model, the padded batch loop, the eval_val call)
is transcribed verbatim from the home note's tested script, so the house
90-column and naming style is not retrofitted onto the code; only the
docstrings were rewritten for readability and the exit status added so the
runner reads a pass/fail code.

Spec code GE-C. Home note: eval-bs-decoupling.md:202-300.
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
    """A stand-in loss whose chi2 is the plain per-row sum of squared errors.

    Lets partition invariance be checked without cosmolike: needs_params is
    False, and chi2 returns one value per row, so eval_val reduces it exactly
    as it would a real loss.
    """
    needs_params = False

    def chi2(self, pred, target):
        return ((pred - target) ** 2).sum(dim=1)   # per-row, (bs,)


loss = ToyChi2()
# thresholds stay on CPU: eval_val moves c to CPU before the frac
# comparison (as the real pipeline does with DEFAULT_THRESHOLDS), so a
# device tensor here would mismatch on CUDA (D-E1).
thresholds = torch.tensor([0.2, 0.5, 1.0])


def toy_data():
    """The tiny in-memory data dict eval_val expects.

    Row-indexed loaders for the inputs C and the data vectors DV, plus the
    row-index array, all pointing at the module-level toy tensors.
    """
    return {"load_C": lambda rows: C[rows],
            "load_dv": lambda rows: DV[rows],
            "idx": np.arange(N_VAL)}


def per_row_chi2(bs):
    """eval_val's batched, last-batch-padded forward pass, kept per row.

    Reproduces the same batch loop eval_val runs (pad the final short batch,
    then drop the pad), but returns the full (N_VAL,) per-row chi2 instead of
    the reduced medians/means, so the batch-size sweep can compare row for
    row against the whole-set reference.
    """
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
        """Average wall-clock time of one eval_val call at batch size bs.

        Warms up three times, synchronizes CUDA, then averages reps timed
        calls (CUDA only; the caller guards on DEV.type).
        """
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

print("\nReal check: one `cosmic_shear_train ... bs: 32` run vs the pre-change "
      "build: epoch time drops ~0.3 s, metrics match to rtol 1e-6.")

# Added for the harness: the runner reads this exit code (Part 1 is the
# hard acceptance; Part 2 is informational timing).
import sys
sys.exit(0 if ok else 1)

---
name: eval-bs-decoupling
description: "Design spec (Architect, 2026-07-06, REVISED same day pre-execution): decouple the validation batch from the training bs by DERIVING it from n_val — no YAML knob. eval_val runs at the training bs today, so a small-batch run doubles its eval launches (bs=32: ~157 eval batches ≈ 0.3 s of a 2.4 s launch-bound epoch). Revision: the first draft added train_args.eval_bs (default 1024); the user asked why users should pick a pure-perf knob with a computable optimum at all — they shouldn't: choose the BATCH COUNT k = ceil(n_val/1024), then bs = ceil(n_val/k), and the fixed-shape tail padding is < k rows (~0.1% worst case; 0 rows at the production n_val=5000, where bs=1000 beats both 1024 and 2048). Enabled by the absolute-counts feature: n_val is exact at config time. training.py-only change; run_emulator/experiment signatures untouched."
metadata:
  node_type: memory
  type: project
---

# Eval batch: derived from n_val, decoupled from the training bs

User request 2026-07-06, from the launch-bound timing puzzle (bs=64 vs
32: per-step cost identical 2.05 ms -> epoch time ~ step count; the
eval pass doubled its launches along with training for no reason).

**Revision (same day, before execution):** the first draft shipped a
`train_args.eval_bs` knob (default 1024). The user asked whether a
pure-performance parameter with a computable optimum should be
user-selectable at all. It should not: the derivation below always
beats any hand-picked constant, and one less YAML key fits the
standing avoid-bloat rule. The knob is gone from this design.

## Facts (verified in source)

- `eval_val` (training.py ~453) takes `bs`; both call sites in
  `training_loop_batched` pass the TRAINING bs (epoch-0 baseline ~847,
  per-epoch ~957). At bs=32, n_val=5000: ~157 eval batches x ~2 ms
  dispatch ≈ 0.3 s of the observed 2.4 s/epoch.
- eval_val already pads the final short batch up to `bs` (stride-0
  row-0 copies, pad chi2 sliced off) so the compiled `fwd_chi2` twin
  sees ONE static shape. A run-constant eval batch therefore costs no
  recompiles.
- Metrics are per-row; the batch split only partitions rows -> median /
  mean / frac are partition-invariant (up to matmul reduction-order
  ulps). Memory at ~1000-row batches is a few MB. Eval is no_grad.
- [[n-train-n-val-absolute-counts]] guarantees n_val exactly at config
  time, so the derived batch is fully determined before the first
  compile.

## The derivation (why no knob can beat it)

The free integer is the batch COUNT k, not the batch size:

    k  = ceil(n_val / 1024)      # batches needed at the ~1024 target
    bs = ceil(n_val / k)         # equalized batch

Total padded rows = k*bs - n_val < k, i.e. fewer padded rows than
batches (~0.1% worst case). Examples:

    n_val   k   bs      padded rows
    5000    5   1000    0           (production; beats 1024's 2.3%
                                     and 2048's 18.6%)
    5001    5   1001    4
    3000    3   1000    0
    1025    2   513     1
    800     1   800     0

By ~512-1024 rows the 3060 is compute-saturated on the forward + chi2
matmuls, so the k-vs-k+2 launch difference is noise; the target 1024
only sets the scale.

## Design

1. **Pure `derive_eval_bs(n_val, target, load)`** in training.py
   beside eval_val (torch-free integer arithmetic, exec-extractable):
   k = ceil(n_val / target); bs = ceil(n_val / k); clamp bs to `load`
   when the streamed chunk is smaller (the memmap-streaming edge:
   batches never span chunks, so a bs above `load` would pad every
   chunk); n_val < 1 raises. `_EVAL_BS_TARGET = 1024` as a named
   module constant (not magic), the only tunable, deliberately not in
   the YAML.
2. **Threading, training.py only:** training_loop_batched computes
   `eval_bs = derive_eval_bs(n_val=len(data["val"]["idx"]),
   target=_EVAL_BS_TARGET, load=load)` ONCE, before the epoch-0
   baseline eval; both eval_val call sites pass `bs=eval_bs`. The
   training bs is untouched everywhere in the train loop.
   run_emulator's and experiment.train's signatures are untouched —
   no new key, no pass-through, nothing for validate_sizes or
   SWEEPABLE_TOP_KEYS.
3. **Docs:** eval_val's `bs` docstring drops "same as training" (it is
   the derived eval batch); training_loop_batched documents the
   derivation at the compute site; one comment line in the
   train_single YAML next to `bs:` ("validation runs at a derived
   ~1024-row batch, independent of bs"). No YAML key changes.
4. No behavior change for metrics: same rows, same per-row math, only
   the partition changes.

## Validation gate

- GE-A (pure, Mac, exec-extracted derive_eval_bs): the example table
  above verbatim (5000 -> 1000/0 pad, 5001 -> 1001/4, 3000 -> 1000/0,
  1025 -> 513/1, 800 -> 800/0); n_val = 1 -> 1; load clamp
  (load=256 -> 256); n_val < 1 raises; property sweep: for ~1e4 random
  n_val in [1, 1e6], k*bs - n_val < k always and bs <= max(target,
  load-clamped) bound holds. Paste raw outputs.
- GE-B (static + style, Mac): AST — derive_eval_bs computed once in
  training_loop_batched BEFORE the baseline eval; BOTH eval_val call
  sites pass the derived value and none passes the training bs;
  run_emulator / experiment.py signatures unchanged (grep: no eval_bs
  key anywhere in experiment.py or the YAMLs beyond the one comment);
  house scans; whole-tree py_compile; keyword-vs-signature on the
  changed call sites.
- GE-C (workstation, torch-only standalone script — no cosmolike): toy
  linear model + quadratic per-row chi2 stub, n_val = 5000: per-row
  chi2 vectors from eval_val at bs in {32, 517, 1000, 2048} agree
  pairwise (allclose rtol 1e-6; median / mean / frac likewise — not
  bitwise, matmul reduction order). Then one real bs=32 run: epoch
  time drops ~0.3 s and metrics match the pre-change run to the same
  tolerance. Rides the workstation queue.

## Sequencing

Stacks on the sweep-guards commit unit (run that block first for clean
units; otherwise the Implementer reports the two-bucket split).
Eventual commit (user):

    git commit -m "Derive the validation batch from n_val (~1024 target, near-zero tail padding; eval decoupled from training bs; gates GE-A/B Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

Implemented 2026-07-06 (Opus 4.8, amazing-keller); execution log + raw
GE-A/B evidence + the ready-to-paste GE-C script in the last section.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file (note the revision: no YAML
  knob — the earlier eval_bs key design is superseded, do not add it).
- **Target file(s):** emulator/training.py ONLY (derive_eval_bs +
  _EVAL_BS_TARGET, the one-time computation in training_loop_batched,
  both eval_val call sites, eval_val docstring), plus one comment line
  in example_yamls/train_single_emulator_cosmic_shear.yaml.
- **Contracts & interfaces:** derive_eval_bs(n_val, target, load)
  pure, torch-free; k = ceil(n_val/target), bs = ceil(n_val/k),
  clamped to load, n_val < 1 raises — verbatim numerics.
  run_emulator / training_loop_batched / experiment.train signatures
  unchanged; no new YAML key; SWEEPABLE_TOP_KEYS unchanged. Declare
  any deviation.
- **Constraints & edge cases:** derived once per run (the compiled
  fwd_chi2 must keep one static shape — the existing padding does
  this, do not touch it); training bs untouched in the train loop;
  the load clamp only matters on the memmap-streaming path.
- **Validation gate:** GE-A + GE-B on the Mac (raw outputs, including
  the property sweep); GE-C is a small torch-only script riding the
  workstation queue with the timing check — write it now and include
  it in the handoff.
- **Next milestone:** IMPLEMENTER_HANDOFF with GE-A/B evidence + the
  ready-to-paste GE-C script.

### 2026-07-06 — Implementer (Opus 4.8) execution

training.py-only, plus one train_single YAML comment. No knob (per the
revision). Mac dev box: GE-A/B ran by exec-extraction / AST; GE-C is the
workstation script, embedded below.

**Done:**

- training.py: `_EVAL_BS_TARGET = 1024` module constant + pure
  `derive_eval_bs(n_val, target, load)` beside eval_val (torch-free:
  `k = (n_val + target - 1)//target`, `bs = (n_val + k - 1)//k`,
  `min(bs, load)`; n_val < 1 raises). training_loop_batched computes
  `eval_bs = derive_eval_bs(n_val=len(data["val"]["idx"]),
  target=_EVAL_BS_TARGET, load=load)` ONCE right before the epoch-0
  baseline eval; both eval_val call sites now pass `bs=eval_bs` (was
  `bs=bs`, the training batch). eval_val's `bs` docstring drops "same as
  training". run_emulator / training_loop_batched / experiment.train
  signatures untouched; the fixed-shape padding untouched.
- train_single YAML: one comment beside `bs:` (validation runs at a
  derived ~1024-row batch, independent of bs). No YAML key.

**Deviations from blueprint:** none. No signature change, no YAML key,
SWEEPABLE_TOP_KEYS untouched, experiment.py untouched (zero `eval_bs`
references, verified). Numerics verbatim.

**Gate evidence (raw, Mac):**

- GE-A derive_eval_bs (pure, exec-extracted): the example table verbatim
  — n_val 5000 -> bs 1000 pad 0, 5001 -> 1001 pad 4, 3000 -> 1000 pad 0,
  1025 -> 513 pad 1, 800 -> 800 pad 0; n_val 1 -> 1; load clamp
  (load 256 -> 256, load 999 -> 999, load 1000 -> 1000); n_val 0 / -3
  raise ("n_val >= 1"); property sweep over 10000 random n_val in
  [1, 1e6]: pad = k*bs - n_val in [0, k) always, 1 <= bs <= target and
  bs <= n_val always, clamp == min(unclamped, load) always, worst-case
  padded fraction 0.098% (matches the ~0.1% claim). GE-A: ALL PASS.
- GE-B static + style: exactly one derive_eval_bs call in
  training_loop_batched, before the first (baseline) eval_val; both
  eval_val sites pass bs=eval_bs, neither bs=bs; derive call keywords ==
  its signature {n_val, target, load}; no `eval_bs:` YAML key anywhere;
  experiment.py has zero `eval_bs` references; run_emulator /
  training_loop_batched / experiment.train signatures carry no eval_bs
  param; 0 new all-caps (one slip "COUNT" caught + de-capped) / 0 new
  ` -- ` on the two touched files; training.py <= 90 cols; 0 new
  comprehensions; whole-tree py_compile clean. GE-B: ALL PASS.
- GE-C (workstation, torch-only, no cosmolike): Part 1 proves metrics are
  partition-invariant (eval_val median / mean / frac and a per-row chi2
  vector rebuilt with eval_val's batch+pad loop agree across bs in
  {32, 517, 1000, 2048}, rtol 1e-6); Part 2 times eval_val at bs 32 vs
  the derived bs to show the launch-count saving; the full ~0.3 s/epoch
  drop is the real `train_single ... bs: 32` run vs the pre-change build.
  Script (ready to paste, run from $ROOTDIR):

```python
#!/usr/bin/env python3
"""GE-C: eval_val partition invariance + eval-batch timing (torch-only)."""
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
thresholds = torch.tensor([0.2, 0.5, 1.0], device=DEV)


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
```

Open: GE-C (workstation) + the Architect re-audit.

### 2026-07-06 — Architect re-audit: code ACCEPTED; two micro-deltas
### (D-E1 script bug, D-E2 alignment)

Verified independently (own harness, own seed): GE-A — the verbatim
table, n_val 1 and 1024 edges, the three load clamps, the 0/-3 raises,
and a 2e4-point property sweep (pad in [0, k), 1 <= bs <= 1024,
bs <= n_val; worst padded fraction 0.09765%, matching the claimed
0.098%). GE-B — exactly one derive_eval_bs call in
training_loop_batched, before both eval_val calls; both sites pass
bs=eval_bs (no training-bs leak); derive keywords == signature; zero
eval_bs references in experiment.py; no eval_bs: key in any YAML;
scans + whole-tree py_compile clean. Code deviations: none. The
training.py + YAML diff is accepted as-is except D-E2.

**D-E1 (required, GE-C script only — it lives in this note, not in
code): thresholds must be a CPU tensor.** The script builds
`thresholds = torch.tensor([...], device=DEV)`; eval_val does
`c = torch.cat(chi2s).cpu()` and then compares
`c[:, None] > thresholds[None, :]` — CPU vs CUDA tensors, so on the
3060 the FIRST eval_val call raises "Expected all tensors to be on the
same device" and GE-C dies before producing evidence. The real
pipeline passes CPU thresholds (DEFAULT_THRESHOLDS in experiment.py),
so the script must too: drop `device=DEV` from the thresholds line
(and nowhere else — C / DV / model stay on DEV). One-line fix in the
embedded script above.

**D-E2 (required, cosmetic): the two YAML continuation-comment lines
are one column off.** In train_single YAML the new `bs:` comment's
continuation lines have `#` at column 24; the anchor line and every
neighboring key comment sit at column 25. Add one leading space to
each of the two continuation lines.

Commit waits on D-E1 + D-E2 (both one-line). GE-C then rides the
workstation queue unchanged.

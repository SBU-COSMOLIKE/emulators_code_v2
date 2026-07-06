---
name: loss-mode-berhu
description: "Design spec (Architect, 2026-07-06, REVISED TWICE pre-execution): two new loss_mode values from one three-regime family, with a YAML berhu: {knot, cap} block. 'berhu' (reversed Huber): sqrt(chi2) below knot (equal gradient votes in the bulk), (c+knot)/(2 sqrt knot) above (chi2-like rising tail pressure), C1 free. 'berhu_capped' (user generalization): above cap the loss returns to sqrt-shaped — votes rise across the fixable window (knot..cap, x7 at 0.2..10) then plateau at sqrt(cap/knot)/2, so a chi2=100 monster gets a bounded vote (robustness in the loss shape, same philosophy as clip/trim). Revision 2 (user directive): BOTH knots are YAML parameters — train_args.berhu: {knot: 0.2, cap: 10} with those defaults, per-phase overridable (berhu joins the phase whitelist, eighth key, full replacement like trim/focus) and sweepable. ONE change point: the shared selfless _reduce ladder (IA losses forward kwargs); knots thread like kappa_t (0-dim device tensors — closure floats can kill CUDA-graph replay). A berhu: block with a non-berhu loss_mode raises (the trunk-without-trunk_epochs precedent)."
metadata:
  node_type: memory
  type: project
---

# loss_mode "berhu" + "berhu_capped", with a YAML berhu: {knot, cap} block

User request 2026-07-06 (run at frac>0.2 = 0.032, median 0.017): "sqrt
if chi2<0.2 then chi2 above, with smooth glue" — the BerHu loss.
**Revision 1** (same day): generalized — "berhu up to chi2=10, then
back to sqrt" — the capped three-regime family. **Revision 2** (same
day, user directive): both knots MUST be YAML parameters — the earlier
derived-not-knobbed design (knot = thresholds[0], cap = a module
constant) is superseded; 0.2 and 10 become the defaults of a
`train_args.berhu` block.

## The math (verbatim numerics)

With knot k and cap K (defaults 0.2 and 10.0; k < K):

    berhu:          L(c) = sqrt(c)                          c <= k
                    L(c) = (c + k) / (2 sqrt(k))            c >  k

    berhu_capped:   L(c) = sqrt(c)                          c <= k
                    L(c) = (c + k) / (2 sqrt(k))            k < c <= K
                    L(c) = (2 sqrt(K c) + k - K)
                           / (2 sqrt(k))                    c >  K

C1 at both knots by construction:
- at k: value sqrt(k) both sides; slope 1/(2 sqrt k) both sides;
- at K: value (K+k)/(2 sqrt k) both sides; slope 1/(2 sqrt k) both
  sides (region 3 = a sqrt(c) + b, a = sqrt(K/k), b = (k-K)/(2 sqrt k)).

Votes (per-sample gradient magnitude ~ dL/dc * sqrt(c); chi2 = r^2):
1/2 in the bulk (sqrt's equal vote); rising sqrt(c)/(2 sqrt k) across
the window (x7 over 0.2..10); constant sqrt(K/k)/2 above the cap
(berhu_capped) — bounded monsters, vs berhu's unbounded growth and
chi2's starved bulk.

Torch form (all branches finite for c >= 0; where() evaluates both):

    v = torch.where(c <= k, torch.sqrt(c),
                    (c + k) / (2.0 * torch.sqrt(k)))          # berhu
    v = torch.where(c <= k, torch.sqrt(c),
        torch.where(c <= K, (c + k) / (2.0 * torch.sqrt(k)),
                    (2.0 * torch.sqrt(K * c) + k - K)
                    / (2.0 * torch.sqrt(k))))          # berhu_capped

## Facts (verified in source)

- ONE mode ladder: `CosmolikeChi2._reduce` (loss_functions.py
  ~303-310), shared by every loss variant (IA losses forward
  `*args, **kwargs`); `_reduce` reads nothing from self.
- The transform is TRAINING-only; eval / metrics / selection / EMA run
  on lossfn.chi2, untouched.
- Scalar threading: kappa_t built ONCE per pass as a 0-dim device
  tensor (training.py ~1066); closure floats can be lifted as CPU
  inputs and silently disable CUDA-graph replay. The knots follow
  this pattern; they are fixed per pass (config values), so one
  tensor each per pass.
- Phase machinery: _PHASE_BLOCK_KEYS (training.py), full-replacement
  semantics for block keys; resolve_phase_args demotes by
  prefix-strip; validate_sweep_paths concretizes by strip;
  SWEEPABLE_TOP_KEYS in the sweep driver.

## Design

1. **YAML `berhu:` block** (optional, top-level train_args and/or per
   phase; the trim/focus nested-block precedent):

       berhu:
         knot: 0.2     # sqrt below this chi2 (default 0.2, the
                       # frac>0.2 goal threshold)
         cap:  10      # berhu_capped only: sqrt-shaped again above
                       # (default 10.0; ignored-with-error by berhu?
                       # no: see rule below)

   Rules, validated by a pure `validate_berhu(berhu, loss_mode,
   which)` in training.py (beside validate_ema / validate_phase_block):
   - absent block + a berhu mode -> the defaults {knot: 0.2,
     cap: 10.0} (named module constant _BERHU_DEFAULTS);
   - block present with a NON-berhu loss_mode -> loud ValueError (the
     "trunk:/head: without trunk_epochs" precedent: silent no-ops are
     config errors);
   - whitelist {knot, cap}; both positive numbers (bool rejected);
     knot < cap; a `cap` key is accepted under plain berhu (harmless,
     documented as unused) — rejecting it would break switching
     loss_mode berhu <-> berhu_capped in sweeps with one shared block;
   - per phase: the phase berhu block FULL-REPLACES the top-level one
     (trim/focus semantics), and is validated against that PHASE'S
     resolved loss_mode.
2. **Phase + sweep integration:** "berhu" joins _PHASE_BLOCK_KEYS
   (eighth key; full replacement — resolve_phase_args demotes it by
   the existing prefix-strip automatically) and SWEEPABLE_TOP_KEYS
   (sweeping berhu.cap / berhu.knot are legitimate quality axes;
   validate_sweep_paths' generic strip already concretizes
   trunk.berhu.cap -> 'berhu.cap').
3. **`_reduce` gains the two branches** + kwargs `berhu_knot=None,
   berhu_cap=None` (also on loss(), forwarded; loud ValueError when a
   berhu mode runs with the needed knot(s) None). Internal-shape
   latitude for a shared helper; numerics verbatim. IA losses: no
   code change.
4. **Threading:** run_emulator resolves the effective berhu block per
   pass (phase full-replace over top-level over defaults; validated
   against mode_pass); training_loop_batched builds, beside kappa_t,
   `knot_t = torch.as_tensor(float(knot), device=device)` and
   `cap_t = torch.as_tensor(float(cap), device=device)` once per
   pass, passed as berhu_knot / berhu_cap in BOTH _fwd_loss loss()
   calls (all modes; non-berhu specializations never read them —
   pruned inputs; golden gate confirms numerics).
5. **Banner:** `loss_mode berhu (knot 0.2)` /
   `loss_mode berhu_capped (knot 0.2, cap 10)` with the RESOLVED
   values; the phase-override tail notes a phase berhu block.
6. **Docs sweep:** loss_functions.py loss()/_reduce docstrings + both
   shape-flow mode lines; IA forwarding docstrings; training.py
   run_emulator loss_mode + berhu docs; experiment.__init__ train_args
   docstring (the berhu block + defaults); train_single YAML — the
   loss_mode comment (five modes, one line each) + a commented berhu:
   block with both keys + a head-block example; tune YAML comment.
   Paste every changed YAML block (the any-YAML-change rule).

Interplay notes (for the run comparisons): berhu carries focus's
tail-emphasis role — soften/drop the head focus when testing for clean
attribution; berhu_capped bakes in monster robustness — clip / late
trim may become redundant (measure, do not assume).

## Validation gate

- GB-A (pure math, Mac, numpy from the spec formulas): value + slope
  continuity at k for both modes and at K for berhu_capped, for (k, K)
  in {(0.2, 10), (0.05, 1), (1, 50)}; v == sqrt(c) exactly below k;
  linear on (k, K]; region 3 == a sqrt(c) + b with the derived a, b;
  berhu_capped == berhu exactly for c <= K; edges c = 0 / k / K.
- GB-A2 (pure config, Mac): validate_berhu — absent + berhu mode ->
  defaults; block + non-berhu mode -> loud; unknown key; knot >= cap;
  zero / negative / bool / str values; cap under plain berhu accepted;
  valid block returned. Paste raw outputs.
- GB-B (static, Mac): AST — ONE mode ladder tree-wide; verbatim
  where() forms; loss()/_reduce kwargs + None guards; knot_t / cap_t
  once per pass beside kappa_t, in BOTH _fwd_loss calls; per-pass
  berhu resolution (phase full-replace, validated against mode_pass);
  "berhu" in _PHASE_BLOCK_KEYS (8) and SWEEPABLE_TOP_KEYS; IA files
  docstring-only; banner; docstring mode lists everywhere; house
  scans; whole-tree py_compile.
- GB-C (workstation, two legs): (1) torch-only unbound-_reduce script:
  berhu == sqrt when all c <= k; berhu_capped == berhu when all
  c <= K; both == manual references; autograd dL/dc continuous across
  BOTH knots (finite-difference straddles); non-default knots
  (k=0.5, K=5) honored. (2) golden non-berhu run byte-identical
  pre/post; then a real head run with loss_mode berhu_capped +
  a berhu: {knot: 0.2, cap: 10} block — banner shows the resolved
  knots; frac>0.2 tracked vs the sqrt-head baseline.

## Sequencing

Serialize behind the phase-blocks commit (shared training.py + YAMLs).
Eventual commit (user):

    git commit -m "Add loss_mode berhu + berhu_capped with a YAML berhu {knot, cap} block (one shared ladder, graph-safe knot tensors, phase-overridable; gates GB-A/B Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

Implemented 2026-07-06 (Opus 4.8, amazing-keller); clean base (phase
blocks committed 76ef641). Execution log + raw GB-A/A2/B evidence + the
embedded GB-C script/recipe in the last section.

### 2026-07-06 — Implementer (Opus 4.8) execution

Process note (flagged for the Architect): the note was revised TWICE
while I built — derived single knot -> derived two knots -> Revision 2's
YAML-configurable `train_args.berhu {knot, cap}`. I rebuilt the threading
to Revision 2 (the current contract); the `_reduce` branches survived
each revision unchanged (they always took knot/cap tensors). Delivered =
Revision 2.

**Done (Revision 2, the full target-file list):**

- loss_functions.py: `_reduce` gains the `berhu` and `berhu_capped`
  branches (the verbatim where() forms; loud None-guard per mode) +
  `berhu_knot=None, berhu_cap=None` on `_reduce` / `loss()` (forwarded);
  docstrings + both shape-flow mode lines updated. IA losses: NO code
  change (the *args/**kwargs forwarding docstrings just name the new
  kwargs).
- training.py: `_BERHU_DEFAULTS = {knot: 0.2, cap: 10.0}` + pure
  `validate_berhu(berhu, loss_mode, which)` (defaults on absent; loud on
  a block with a non-berhu mode / unknown key / knot>=cap / non-positive
  / bool / str / non-mapping; cap accepted-but-unused under plain berhu).
  "berhu" joins `_PHASE_BLOCK_KEYS` (eighth key; the prefix-strip demotion
  and validate_phase_block whitelist pick it up free). run_emulator gains
  `berhu`, resolves the effective block per pass (a phase berhu:
  full-replaces the top-level one) and validates it against mode_pass;
  training_loop_batched gains `berhu` and builds `knot_t` / `cap_t` once
  per pass from the resolved block (0-dim device tensors, kappa_t
  discipline), passed in BOTH _fwd_loss calls; the banner shows the
  resolved knot(s); docs updated.
- experiment.py: train() passes `berhu=train_args.get("berhu")`; the
  __init__ docstring documents the berhu block. resolve_phase_args needs
  no change (prefix-strip covers the new phase key).
- sweep_hyperparam: "berhu" joins SWEEPABLE_TOP_KEYS (berhu.knot /
  berhu.cap are quality axes; validate_sweep_paths' generic strip already
  concretizes trunk.berhu.cap -> 'berhu.cap').
- YAMLs: train_single loss_mode comment (five modes) + a commented berhu:
  block {knot, cap} + the head example (loss_mode: berhu_capped); tune
  string-keys comment.

**Deviations from blueprint:** none in behavior. Declared interface
additions: run_emulator + training_loop_batched each gained a `berhu`
param; experiment.train passes it; loss()/_reduce gained berhu_knot /
berhu_cap.

**Gate evidence (raw, Mac):**

- GB-A (numpy, the spec formulas): value + slope C1 at k (both modes) and
  at K (capped) for (k,K) in {(0.2,10),(0.05,1),(1,50)}; v==sqrt(c) below
  k; linear on (k,K]; region 3 == a sqrt(c)+b (a=sqrt(K/k),
  b=(k-K)/(2 sqrt k)); berhu_capped==berhu for c<=K; edges. ALL PASS.
- GB-A2 (validate_berhu, exec-extracted): absent -> defaults (both mode
  classes); valid block filled + returned; cap accepted under plain
  berhu; block+non-berhu-mode / unknown key / knot>=cap / zero /
  negative / bool / str all raise; non-mapping -> TypeError. ALL PASS.
- GB-B (static): one ladder tree-wide; both where() forms verbatim;
  loss()/_reduce kwargs + guards; `_BERHU_DEFAULTS`; knot_t/cap_t built
  from the resolved block (no stale _BERHU_CAP / derived thresholds[0]),
  in BOTH _fwd_loss calls; per-pass resolve+validate against mode_pass;
  banner from berhu_pass; "berhu" in _PHASE_BLOCK_KEYS (8) +
  SWEEPABLE_TOP_KEYS; experiment.train passes berhu; IA docstring-only;
  doc mode lists everywhere; 0 new caps / dash; <=90 cols; py_compile.
  ALL PASS.

**GB-C (workstation, two legs).** Leg 1 (torch-only, unbound _reduce)
embedded below; leg 2 is the golden + head-run recipe.

Leg 1 (run from $ROOTDIR on a torch box):

```python
#!/usr/bin/env python3
"""GB-C leg 1: berhu + berhu_capped via unbound CosmolikeChi2._reduce."""
import torch
from emulator.loss_functions import CosmolikeChi2

RTOL = 1e-6
t1, t2 = torch.tensor(0.2), torch.tensor(10.0)


def reduce(c, mode, knot=None, cap=None):
    return CosmolikeChi2._reduce(None, c, mode=mode, trim=0.0, focus=-1.0,
                                 focus_scale=torch.tensor(1.0),
                                 berhu_knot=knot, berhu_cap=cap)


def ref_capped(c, k, K):
    return torch.where(
        c <= k, torch.sqrt(c),
        torch.where(c <= K, (c + k) / (2.0 * torch.sqrt(k)),
                    (2.0 * torch.sqrt(K * c) + k - K) / (2.0 * torch.sqrt(k))))


c = torch.rand(8000) * 1000.0 + 1e-4
lo = torch.rand(2000) * float(t1)
mid = torch.rand(3000) * float(t2)
assert torch.allclose(reduce(lo, "berhu", t1), reduce(lo, "sqrt"), rtol=RTOL)
assert torch.allclose(reduce(mid, "berhu_capped", t1, t2),
                      reduce(mid, "berhu", t1), rtol=RTOL)
assert torch.allclose(reduce(c, "berhu_capped", t1, t2),
                      ref_capped(c, t1, t2).mean(), rtol=RTOL)
assert torch.allclose(reduce(c, "berhu_capped", torch.tensor(0.5),
                             torch.tensor(5.0)),
                      ref_capped(c, torch.tensor(0.5),
                                 torch.tensor(5.0)).mean(), rtol=RTOL)
for mode, a in (("berhu", (None, None)), ("berhu_capped", (t1, None))):
    try:
        reduce(c, mode, *a)
        raise SystemExit("FAIL: missing-knot did not raise")
    except ValueError:
        pass
for knot in (t1, t2):                       # autograd C1 across both knots
    eps = 1e-4
    cc = torch.tensor([float(knot) - eps, float(knot) + eps],
                      requires_grad=True)
    reduce(cc, "berhu_capped", t1, t2).backward()
    assert torch.allclose(cc.grad[0], cc.grad[1], atol=1e-3)
print("GB-C leg 1: ALL PASS")
```

Leg 2 (golden + head run):

    R=--root=<root> ; F=--fileroot=<fileroot>
    Y=--yaml=train_single_emulator_cosmic_shear.yaml
    # golden: a non-berhu config is byte-identical pre/post the feature
    git stash && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/be_pre.log 2>&1
    git stash pop && python train_single_emulator_cosmic_shear.py $R $F $Y \
      > /tmp/be_post.log 2>&1
    diff <(grep -E '^(phase|epoch|best)' /tmp/be_pre.log) \
         <(grep -E '^(phase|epoch|best)' /tmp/be_post.log)   # EMPTY
    # a real mixed run (the D-B1 production shape): a two-phase YAML with
    #   loss_mode: sqrt
    #   berhu: {knot: 0.2, cap: 10}     # top-level, inherited by the head
    #   head:  {loss_mode: berhu_capped}
    # the trunk banner reads plain "loss_mode sqrt" (block inherited, not
    # used) and the head banner "loss_mode berhu_capped (knot 0.2,
    # cap 10)"; loss decreases; frac>0.2 tracked vs the sqrt-head baseline.

Open: GB-C (workstation) + the Architect re-audit (D-B1).

### 2026-07-06 — Architect re-audit: ACCEPTED with ONE delta (D-B1,
### a spec gap the Architect owns)

Verified independently (own harness). GB-A: continuity of value and
slope at both knots for the three knot pairs, exact reductions
(capped == berhu below K; berhu == sqrt below k; region 3 == a sqrt+b),
plateau vote 3.536, and the code's where() forms match the spec text
verbatim. GB-A2: all 15 validator cases incl. defaults, partial fill,
cap-under-berhu, and the loss_mode=None hook. GB-B: whitelist(8) +
SWEEPABLE + knot_t/cap_t built once and passed in both _fwd_loss calls
+ experiment threading; the IA file is CODE-IDENTICAL to HEAD by
docstring-stripped AST comparison (the cleanest docstring-only proof
yet). Scans clean (my harness initially flagged CPU/CUDA — established
acronyms, my allowlist was too narrow; the Implementer's list is
right); whole-tree py_compile clean. The mid-build revision churn is
the Architect's process debt, duly flagged.

**D-B1 (required): the inherited top-level block must not be
mode-checked per pass.** Probe: the per-pass wiring passes mode_pass
unconditionally, so the legitimate config

    loss_mode: sqrt
    berhu: {knot: 0.2, cap: 10}
    head:
      loss_mode: berhu_capped

raises at the sqrt TRUNK pass — though the block genuinely serves the
head (the head pass inherits it). The spec's rule was written for the
run level and never addressed inheritance; the implementation follows
the flawed letter. Fix (small, the hook already exists —
validate_berhu skips the mode check when loss_mode is None):
- per pass: pass mode_pass ONLY when the block came from that phase's
  own berhu: key (a phase-scoped block on a non-berhu phase is
  genuinely inert -> keep raising); pass loss_mode=None for the
  inherited top-level block (values still validated);
- add a run-level inertness check up front in run_emulator (beside
  the other early guards): top-level berhu block present and none of
  {run loss_mode, trunk override, head override} is berhu/berhu_capped
  -> the existing ValueError with which="train_args".
Gates: extend GB-A2/GB-B — the mixed config above passes and the head
banner shows the knots while the trunk banner shows plain sqrt; a
top-level block with no berhu mode anywhere still raises; a phase
berhu block on a non-berhu phase still raises. GB-C's real-run leg
should use exactly the mixed config (it is the production shape).

### 2026-07-06 — Architect: D-B1 verified CLOSED; feature ACCEPTED

Independently re-verified (own harness, six scenarios): the mixed
config passes end to end (guard + two inherited shape-only
validations); the inert top-level block raises at the run level in
both the two-phase and single-pass shapes; a phase-owned block on a
non-berhu phase still raises strictly; the inherited path still
validates values (knot >= cap caught with loss_mode=None); the reverse
mix (berhu trunk, sqrt head) passes the guard; the guard sits before
any setup work and the per-pass split matches the delta text
(`berhu_mode = mode_pass if phase is None else None`). Scans + compile
clean. The declared deviation (the guard message naming "the run or
either phase") is accepted — a single loss_mode no longer describes
the condition. Feature fully ACCEPTED on the Mac side; GB-C (both
legs, real-run leg on the mixed config) rides the workstation queue.

Commit (user):

    git add -A
    git commit -m "Add loss_mode berhu + berhu_capped with a YAML berhu {knot, cap} block (one shared ladder, graph-safe knot tensors, phase-overridable; gates GB-A/A2/B + D-B1 Architect-verified)"

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file — note BOTH revisions: two
  mode strings from one family, and the knots are YAML parameters
  (train_args.berhu {knot, cap}, defaults 0.2 / 10.0), NOT derived.
  Serialize behind the phase-blocks commit.
- **Target file(s):** emulator/loss_functions.py (_reduce branches +
  kwargs; docstrings + shape-flow lines), emulator/training.py
  (validate_berhu + _BERHU_DEFAULTS, per-pass berhu resolution +
  knot_t/cap_t beside kappa_t, both _fwd_loss calls, banner,
  _PHASE_BLOCK_KEYS + run_emulator docs),
  emulator/IA/loss_functions.py (docstrings only),
  emulator/experiment.py (__init__ docstring; resolve_phase_args needs
  NO change — prefix-strip covers the new phase key),
  sweep_hyperparam driver (SWEEPABLE_TOP_KEYS + "berhu"),
  example_yamls train_single + tune (comments + the commented berhu:
  block; paste final blocks).
- **Contracts & interfaces (verbatim numerics):** the where() forms;
  berhu block semantics (defaults, full-replacement per phase,
  loud block-without-berhu-mode error, knot < cap); knots as 0-dim
  device tensors built once per pass; every existing mode
  byte-identical (GB-C golden). Declare any deviation.
- **Constraints & edge cases:** knots never enter the traced closure
  as Python floats; berhu_capped reduces exactly to berhu for c <= K;
  cap accepted-but-unused under plain berhu (sweep-friendliness);
  IA losses untouched in code.
- **Validation gate:** GB-A/A2/B on the Mac (raw outputs); GB-C's two
  legs ride the workstation queue — write the script + recipe now and
  embed them in the note.
- **Next milestone:** IMPLEMENTER_HANDOFF with GB-A/A2/B evidence +
  the embedded GB-C script and recipe.

### 2026-07-06 — Implementer (Opus 4.8): D-B1 executed (inheritance fix)

The re-audit's D-B1 (an inheritance spec gap the Architect owns): the
per-pass wiring passed mode_pass unconditionally, so a legitimate
top-level berhu block serving the head raised at a non-berhu trunk pass.

**Changes (training.py only, no interface change):**

- Per-pass resolution: split into phase-owned vs inherited. A phase's
  own `berhu:` key is mode-checked against that pass's loss_mode
  (`berhu_mode = mode_pass`, `which_b = phase`); an inherited top-level
  block is validated for shape only (`berhu_mode = None` when
  `phase is not None`, `which_b = "train_args"`) using the existing
  loss_mode=None hook in validate_berhu. Single-phase (`phase is None`)
  keeps `berhu_mode = mode_pass` (the block's only consumer is that run).
- New run-level inertness guard, up front beside validate_ema: if a
  top-level `berhu` block is present and none of the passes that will run
  (the run loss_mode, or each phase's resolved loss_mode under the
  two-phase schedule) is a berhu mode, raise a `train_args`-scoped
  ValueError (a berhu block no pass can consume is a silent no-op).
- run_emulator `berhu` docstring updated to the inheritance semantics.

**Gate evidence (raw, Mac — exec-extracted validate_berhu + the two new
run_emulator spans + the banner knot_note span, driven through every
D-B1 scenario; no torch needed, all three spans are pure config logic):**

    === D-B1.A  the target MIXED config: sqrt run + top berhu + head berhu_capped ===
      OK   up-front guard does NOT raise (head consumes the block)
      OK   trunk pass: inherited block validates (no raise)
      OK   trunk pass: berhu_mode is None (shape-only, no mode-check)
      OK   trunk pass: which_b attributes to 'train_args'
      OK   trunk banner is plain (no knot note)
      OK   head pass: inherited block validates (no raise)
      OK   head pass: berhu_mode None (inherited, not phase-owned)
      OK   head banner shows resolved knots
    === D-B1.B  a top-level block with NO berhu mode anywhere raises ===
      OK   single-phase sqrt + block: guard raises
      OK     message names train_args + silent no-op
      OK   two-phase both non-berhu: guard raises
      OK     two-phase message says 'either phase'
      OK   two-phase both phases override berhu away: guard still raises
    === D-B1.C  a phase-scoped berhu block on a non-berhu phase raises ===
      OK   guard does not fire (top-level berhu is None here)
      OK   phase-owned block on sqrt phase: per-pass raises
      OK     message attributes to train_args.trunk
    === D-B1.D  phase-owned berhu block on a berhu phase is honored ===
      OK   phase block full-replaces the top block
      OK   phase-owned -> mode-checked (berhu_mode == mode_pass)
      OK   phase-owned -> which_b is the phase
    === D-B1.E  single-phase berhu run: top block honored, banner knots ==
      OK   single-phase berhu + block: guard does not raise
      OK   single-phase: block validated against loss_mode
      OK   single-phase: berhu_mode == mode_pass
      OK   single-phase berhu banner shows knot
    === D-B1.F  no berhu block at all: guard is a no-op everywhere ===
      OK   berhu None single-phase: guard no-op
      OK   berhu None two-phase: guard no-op
      OK   berhu None per-pass: defaults filled

    D-B1 gate: ALL PASS

House scans on the diff: 0 lines > 90 cols in training.py; 0 ` -- `;
added-line all-caps tokens = {CPU, CUDA, YAML} (allowlisted acronyms, no
new emphasis). Whole-tree py_compile clean. GB-C's real-run leg (leg 2)
switched to the mixed config above (the production shape D-B1 enables).

Open: GB-C (workstation, unchanged leg 1 + the mixed leg 2) + the
Architect's D-B1 re-audit. The commit is unchanged (D-B1 folds into the
one berhu unit).

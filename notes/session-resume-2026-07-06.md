---
name: session-resume-2026-07-06
description: "Post-compact READ-FIRST pointer for the long 2026-07-05/06 Opus (Implementer) session on branch amazing-keller. Consolidates the commit chain (audit + ten features committed, tip 76ef641), the ONE uncommitted unit (loss_mode berhu/berhu_capped, Revision 2, awaiting the user's commit + an Architect re-audit), the full WORKSTATION-DEFERRED validation queue across every feature, and the re-audits owed. Per-feature detail + raw gate evidence live in each feature's own note; MEMORY.md indexes them all as IMPLEMENTED."
metadata:
  node_type: memory
  type: project
---

# Session resume (2026-07-05/06, Opus Implementer on amazing-keller)

One long dual-agent session (Fable = Architect, Opus = Implementer). The
Architect relayed feature handoffs; the Implementer executed each with Mac
gate evidence and emitted IMPLEMENTER_HANDOFF blocks. Every feature has its
own `notes/` entry with resume state + raw gate outputs; this is the
consolidated pickup after a compaction.

## Commit chain (git log on amazing-keller, newest first)

- `76ef641` phase blocks (nested lr + per-phase scheduler) — [[phase-blocks-nested-lr-scheduler]]
- `7ebc061` weight EMA (folds eval-bs D-E1/D-E2 + D-M1) — [[weight-ema-snapshot-coupled]]
- `46ec5e1` eval batch from n_val — [[eval-bs-decoupling]]
- `4471539` sweep-guards + D-P1 — [[driver-audit-phase-sweep-guards]]
- `2b7a2af` resolve_phase_args — [[resolve-phase-args-single-phase]]
- `9950692` D-1 n_keep>=1 guard; `906528c` n_train/n_val — [[n-train-n-val-absolute-counts]]
- `dba7588` param_cuts nesting + triangle shading; `a0cd132` window cuts;
  `e2394ed` package audit (the early features).

All the above are Architect-verified and committed. `aaa02f1` / `f67e9c3`
are Architect note commits.

## THE ONE UNCOMMITTED UNIT: loss_mode berhu / berhu_capped

[[loss-mode-berhu]], Revision 2 (YAML-configurable knots). IMPLEMENTED +
GB-A/A2/B all PASS on the Mac; NOT yet committed and NOT yet re-audited.
The note was revised twice mid-build (derived knot -> derived two knots ->
Revision 2's `train_args.berhu {knot, cap}`); delivered = Revision 2.
Files + the commit command are in the note's execution section (one clean
unit: loss_functions.py, training.py, IA/loss_functions.py, experiment.py,
sweep_hyperparam.py, the 2 train_single/tune YAMLs, the note, MEMORY.md).
The user commits (never the Implementer).

## Owed to the Architect (re-audits)

- Re-audit **berhu D-B1** on amazing-keller — the inheritance fix (a
  phase-owned berhu block is mode-checked against its pass; an inherited
  top-level block is shape-only via the loss_mode=None hook; a new
  run-level inertness guard rejects a top-level block no pass consumes).
  Berhu Revision 2 itself was already Architect-ACCEPTED (with D-B1 as
  the one delta); D-B1 is now executed + Mac-gated (see the berhu note's
  D-B1 section). Every other feature is Architect-ACCEPTED.

## WORKSTATION-DEFERRED validation queue (torch + cosmolike; none on the
## Mac). Most fold into a few train_single runs.

- **GB-C** (berhu): the embedded unbound-`_reduce` torch script (berhu==sqrt
  below knot, capped==berhu below cap, non-default knots, autograd C1 at
  both knots) + a golden non-berhu byte-identity run + a real `berhu_capped`
  head run. Script embedded in [[loss-mode-berhu]].
- **GH-E** (phase blocks): a head `scheduler.patience: 10` cadence run +
  a no-blocks golden `diff`.
- **GM-C/D** (EMA): off-mode byte-identity golden + on-mode smoke
  (post-rewind EMA jumps with raw).
- **GE-C** (eval-bs): the partition-invariance + timing script (embedded).
- **GP-D** (resolve_phase_args): the failing `name: resmlp` + two-phase YAML
  now trains; a rescnn+nla control is a no-op.
- **GS-D** (n_train/n_val): banner shows the `sizes:` line + `used N of P
  cut rows`.
- **Older, still open:** G-F (window smoke), GN-F (param_cuts load),
  GT-B/GT-C (triangle render + PDF), G1 runtime `import emulator...` leg,
  and **item-27** (audit): the duplicate `ci.init_probes` in
  geometries_output.py, resolved by a chi2 A/B (identical -> delete; else a
  why-comment). See [[audit-package-style-2026-07-05]] P6.

The Architect closes the remaining gates on the pasted outputs in one
workstation pass; each recipe is in its feature note.

## Method that made Mac-side gating possible

The Mac dev python has numpy + stdlib but NOT torch / cosmolike /
matplotlib / getdist / pyyaml. Gates ran by EXEC-EXTRACTING the pure
functions from source (ast-parse, take the def/const nodes by name, exec
their span into a numpy-only namespace) + AST/tokenize scans + py_compile,
and — for functions using torch/psutil internally — exec'ing the whole
module with numpy-backed torch/psutil stubs in sys.modules. Captured in
[[dev-machine-mac-m2-32gb]]. The scan/harness scripts lived in the session
scratchpad (ephemeral); the method carries over, not the files.

## Superseded

The [[session-status-2026-07-06]] note and the earlier body of this file
described only the first four features; this consolidation replaces them.

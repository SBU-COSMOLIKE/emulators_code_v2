---
name: driver-audit-phase-sweep-guards
description: "Architect audit (2026-07-06) of the four non-train_single drivers + example YAMLs against the recent features (window cuts, param_cuts nesting, n_train/n_val, resolve_phase_args). Structure is sound: every driver funnels through exp.train() so the single-phase demotion applies uniformly; zero stale key references (divisor / omegabh2_cut / flat cuts) anywhere. ONE behavioral regression found: resolve_phase_args removed the loud failure that used to stop a hyperparam sweep or Optuna search over trunk_epochs / head.* on a single-phase model — those now train N identical points (head/trunk_epochs demoted away) or waste search dimensions, silently. Fix: a pure validate_sweep_paths(paths, two_phase) in experiment.py, called at startup by sweep_hyperparam + tune; plus stale-comment parity fixes (set_by_path docstring, sweep YAML 'guards still apply', tune YAML 'two-phase keys work here too' beside a shipped name: resmlp)."
metadata:
  node_type: memory
  type: project
---

# Driver audit: phase-key sweeps + doc parity after resolve_phase_args

User question 2026-07-06: were sweep_hyperparam / sweep_ntrain / tune /
bakeoff and the example YAMLs audited against the new flags? Answer:
only the lines the features touched. This note is the full audit
(Architect-run, per the audit-is-Fable-only rule) + the fix plan.

## Audit findings (Architect, raw greps + source reads)

Clean (no action):

- Zero references to train_divisor / val_divisor / omegabh2_cut / flat
  cut keys in any driver or YAML (grep, whole tree minus notes).
- All four drivers train exclusively through exp.train(...) (tune 187 /
  346, sweep_ntrain 160 / 438, sweep_hyperparam 254, bakeoff 149 / 399;
  no direct run_emulator calls) — resolve_phase_args applies uniformly.
- bakeoff + sweep_ntrain: reuse the train_single YAML, pass explicit
  stage_train(n_train=int(N)) capped by pool_size(), stage_val() reads
  the new n_val — transparent to every recent feature; headers carry no
  stale schema text. With the D-1 guard an N grid that ever produced 0
  now fails loudly instead of staging an empty source.
- set_by_path (sweep_hyperparam) deep-copies per point — the
  resolve_phase_args no-mutation contract is not violated by sweeps.

Findings (action needed):

1. **Behavioral regression — silent phase-key sweeps on single-phase
   models.** Before resolve_phase_args, sweeping `head.lr_base` (or
   `trunk_epochs`) with `name: resmlp` died loudly at the first point on
   run_emulator's guards. Now train() demotes the keys first, so:
   - `head.*` or `trunk_epochs` sweep -> N IDENTICAL training runs (the
     axis is dropped), a whole GPU sweep wasted with only a quiet-gated
     notice (workers run quiet=True);
   - `trunk.*` sweep -> functional but disguised: it sweeps the merged
     top-level key (e.g. trunk.loss_mode sweeps loss_mode);
   - tune: an Optuna range leaf inside trunk:/head:/trunk_epochs on a
     single-phase model = a dead (or disguised) search dimension
     sampled every trial — silently degraded search. The shipped tune
     YAML says "trunk_epochs / head two-phase keys work here too" right
     beside `name: resmlp`.
   SWEEPABLE_TOP_KEYS deliberately lists trunk_epochs / trunk / head
   (valid for two-phase sweeps), so the whitelist cannot catch this;
   the check must know the model's capability.
2. **Stale comments** (all assert the pre-demotion loudness):
   - sweep_hyperparam set_by_path docstring: "run_emulator's
     trunk_epochs guard still applies";
   - sweep YAML lines ~37-39 ("guards still apply (trunk/head overrides
     need trunk_epochs > 0)") and ~82-83 ("head.lr_base ... needs
     trunk_epochs > 0");
   - tune YAML line ~14 ("trunk_epochs / head two-phase keys work here
     too (fixed scalars)") — needs the single-phase caveat, since the
     shipped model is resmlp.

## Design

1. **Pure `validate_sweep_paths(paths, two_phase)`** in experiment.py,
   beside resolve_phase_args (one source of truth for both drivers; no
   torch). `paths` = iterable of dotted train_args paths a search will
   vary. two_phase=True -> returns silently. two_phase=False -> for any
   path whose FIRST segment is:
   - "head" or "trunk_epochs": raise ValueError — the demotion makes
     every point identical; name the path, say the model is
     single-phase (no set_train_phase), and that the axis is dropped
     by resolve_phase_args;
   - "trunk": raise ValueError — sweep/search the top-level key
     directly instead (on a single-phase model trunk: merges into the
     top level, so `trunk.loss_mode` is `loss_mode` in disguise);
   - anything else: fine.
   Multiple offending paths -> one error listing all of them.
2. **sweep_hyperparam call site:** in main, right after
   `exp = EmulatorExperiment.from_config(cfg, ...)` (line ~342) and
   before any dispatch (serial loop AND gpu pool):
   `validate_sweep_paths(paths=[param], two_phase=hasattr(
   exp.model_cls, "set_train_phase"))`. Also reword the set_by_path
   docstring line (finding 2) to name this guard + the demotion.
3. **tune call site:** in main, where the searched leaves are known
   (`ranges = search_defaults(train_args=raw_ta)` region, line ~304):
   collect the dotted paths of every range leaf in raw_train_args
   (the [default, min, max, kind] leaves) and pass them all in one
   validate_sweep_paths call, before any worker spawns. If
   search_defaults does not expose paths, a small local walk of
   raw_ta collecting them is fine (pure, few lines).
4. **YAML comment parity** (with the D-P1-corrected wording: single
   phase = any model class without set_train_phase, every name: resmlp
   including ia: nla / tatt):
   - sweep YAML ~37-39 + ~82-83: keep the two-phase statement, add the
     single-phase sentence (head/trunk_epochs sweeps are rejected at
     startup; trunk.* means the top-level key);
   - tune YAML ~14: same caveat sentence.
5. No behavior change for two-phase models anywhere; bakeoff and
   sweep_ntrain untouched.

## Validation gate

- GD-A (pure, Mac, exec-extracted validate_sweep_paths): head.lr_base /
  trunk_epochs / trunk.loss_mode each raise their tailored message when
  two_phase=False (trunk's says "sweep loss_mode directly"); a mixed
  list raises once naming all offenders; nepochs / lr.lr_base / model.*
  pass; everything passes when two_phase=True. Paste raw outputs.
- GD-B (static, Mac): AST — sweep_hyperparam calls validate_sweep_paths
  after from_config and before the serial loop AND the pool dispatch;
  tune calls it before spawning workers, with the collected range
  paths; both probe hasattr(exp.model_cls, "set_train_phase").
- GD-C style: house scans on the four touched files (two .py, two
  .yaml); whole-tree py_compile.
- GD-D: none owed — the change is an early config error; no new runtime
  behavior for valid configs, workstation queue unchanged.

## Sequencing

Same Implementer session as the pending D-P1 (three doc lines). Commit
units after acceptance (user runs): D-1 (already given, ready),
resolve_phase_args (after D-P1), then this:

    git commit -m "Reject phase-key sweeps on single-phase models (validate_sweep_paths in sweep/tune) + stale two-phase comments (gates GD-A-C Architect-verified)"

then `cd ../../..` and `git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

Implemented 2026-07-06 (Opus 4.8, amazing-keller); D-P1 closed first,
then this. Execution log + raw gate evidence in the last section.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Notes entry (read first):** this file (audit findings + design).
  Do D-P1 (notes/resolve-phase-args-single-phase.md re-audit section)
  in the same session, first.
- **Target file(s):** emulator/experiment.py (pure validate_sweep_paths
  beside resolve_phase_args), sweep_hyperparam_emulator_cosmic_shear.py
  (call site + set_by_path docstring reword),
  tune_single_emulator_cosmic_shear.py (call site + range-path
  collection), example_yamls/sweep_hyperparam_*.yaml +
  example_yamls/tune_single_*.yaml (comment parity, D-P1 wording).
- **Contracts & interfaces:** validate_sweep_paths(paths, two_phase)
  pure, raises ValueError naming every offending path with the
  head/trunk_epochs vs trunk.* distinction; returns None. Capability
  probe stays hasattr(model_cls, "set_train_phase") at the call sites.
  Declare any interface deviation.
- **Constraints & edge cases:** two-phase models: zero behavior change;
  trunk.* message must name the concrete top-level key to sweep
  instead; tune must validate ALL searched paths in one call before
  any GPU work; bakeoff / sweep_ntrain untouched.
- **Validation gate:** GD-A/B/C, raw outputs pasted; GD-D none.
- **Next milestone:** one IMPLEMENTER_HANDOFF covering D-P1 + this,
  with the per-commit file split for the three pending commit units.

### 2026-07-06 — Implementer (Opus 4.8) execution

D-P1 of [[resolve-phase-args-single-phase]] closed first (three doc
spots), then this feature. Mac dev box: GD-A/B/C ran by exec-extraction /
AST; GD-D none owed (an early config error, no new runtime behavior).

**D-P1 (doc-only):** the three "single-phase = resmlp / nla" spots now
read "single-phase = any name: resmlp, including ia nla / tatt (no
set_train_phase); two-phase = rescnn / restrf": experiment.py __init__
train_args docstring, the train() call-site comment, and the train_single
YAML trunk:/head: comment. No code change; scans + py_compile re-run
clean, and a tree-wide grep confirms the stale phrases are gone from code
and YAML (the only survivors are in this note's + the resolve note's
audit prose, quoting the old wording deliberately).

**Done (this feature, exactly the target-file list):**

- experiment.py: new standalone pure `validate_sweep_paths(paths,
  two_phase)` beside resolve_phase_args (the one source of truth both
  drivers call). two_phase -> returns None. Single-phase -> collects
  every offending path and raises one ValueError: a first segment
  `head` / `trunk_epochs` names the path and says the axis is dropped by
  resolve_phase_args (every point identical); a first segment `trunk`
  names the concrete top-level key to sweep instead (`trunk.lr_base` ->
  `lr.lr_base`, else the second segment); a mixed list lists all
  offenders in one message.
- sweep_hyperparam: imports validate_sweep_paths; calls it right after
  from_config and before any dispatch (the serial `_hyper_job` loop AND
  run_gpu_pool), guarded by `not act_mode` (the activation-family sweep
  is not a train_args path); probes
  `hasattr(exp.model_cls, "set_train_phase")`. set_by_path docstring
  reworded (the stale "run_emulator's trunk_epochs guard still applies"
  now names validate_sweep_paths + the demotion).
- tune: imports validate_sweep_paths; calls it right after
  `ranges = search_defaults(...)` and before any worker spawns
  (serial stage_train AND the mp spawn), passing `paths=list(ranges)`
  (search_defaults returns {dotted-path: default}, so its keys ARE the
  searched leaves; no extra walk needed).
- YAML comment parity (D-P1 wording): sweep YAML lines ~37-39 (two-phase
  sweeps as written; single-phase head. / trunk_epochs rejected at
  startup, trunk.X = top-level X) and the commented head.lr_base example
  (~82-83, "two-phase only; rejected at startup on single-phase"); tune
  YAML line ~14 (the two-phase keys are fixed scalars on a two-phase
  model; a range over head. / trunk_epochs / trunk. on a single-phase
  model is rejected at startup).

**Deviations from blueprint:** none. validate_sweep_paths is the one
added public function (declared). bakeoff / sweep_ntrain untouched;
two-phase models unaffected.

**Gate evidence (raw, Mac):**

- GD-A validate_sweep_paths (pure, exec-extracted): `head.lr_base` and
  `trunk_epochs` raise with "drop this axis" / "every sweep point would
  be identical"; `trunk.loss_mode` -> "sweep 'loss_mode' directly",
  `trunk.lr_base` -> "sweep 'lr.lr_base' directly" (the lr nesting);
  head-vs-trunk wording is distinct (head says "drop this axis" not
  "directly", trunk the reverse); a mixed list
  [head.lr_base, trunk.loss_mode, nepochs] raises ONE error naming both
  offenders and not nepochs; nepochs / lr.lr_base / bs / model.cnn.* /
  trim.* / focus.* all pass; two_phase=True is silent even for the phase
  axes. GD-A: ALL PASS.
- GD-B static wiring (AST within each main): sweep_hyperparam calls
  validate_sweep_paths after from_config and before both the serial
  `_hyper_job` loop and run_gpu_pool; tune calls it after search_defaults
  and before both serial stage_train and the ctx.Process spawn, passing
  list(ranges); both probe hasattr(exp.model_cls, "set_train_phase").
  GD-B: ALL PASS.
- GD-C style: 0 new all-caps / 0 new ` -- ` on all six touched files
  (one caps slip, "AND" in a sweep comment, caught by the gate and
  de-capped); experiment.py + both drivers <= 90 cols; 0 new
  comprehensions in experiment.py; D-P1 stale phrases gone from code +
  YAML tree-wide; whole-tree py_compile clean. GD-C: ALL PASS.
- GD-D: none owed (early config error; valid configs unchanged;
  workstation queue untouched).

Open: nothing on the Mac. Architect re-audit of D-P1 + this feature owed.

### 2026-07-06 — Architect re-audit: ACCEPTED (D-P1 + sweep guards)

Verified independently (own harness, own cases): validate_sweep_paths
rejects head.lr_base / trunk_epochs with the axis-dropped wording,
concretizes trunk.loss_mode -> "sweep 'loss_mode' directly" and
trunk.lr_base -> "'lr.lr_base'" (the nesting), handles a bare "trunk",
raises ONE error for a mixed list naming both offenders and not the
clean axis, matches exact segments only (heads_of_state passes), and is
silent for two_phase or empty paths. Wiring (AST, line-ordered):
sweep from_config(343) < guard(352) < serial job(397) < pool(477), with
act_mode assigned before the guard (the skip is correct: activation
sweeps are not train_args paths); tune ranges(304) < guard(310) <
serial staging(331) < Process(436), paths=list(ranges) exactly the
searched leaves. D-P1: all six stale phrases gone from the code + YAML
tree (my sweep); the added caps tokens in the diff live only in notes
status markers, code/YAML added lines are caps-clean; whole-tree
py_compile clean. Deviations: none (the trunk.lr_base concretization
was declared and is within the message contract).

Commit sequencing accepted as the Implementer proposed: 2b7a2af shipped
resolve_phase_args without D-P1, and D-P1's experiment.py hunks share
the file with validate_sweep_paths, so both land as ONE commit:

    git commit -m "Reject phase-key sweeps on single-phase models (validate_sweep_paths in sweep/tune) + doc delta D-P1 (gates GD-A-C Architect-verified)"

Nothing remains open on this feature (GD-D was never owed). The
workstation queue is unchanged: G-F, GN-F, GT-B/GT-C, GS-D, GP-D,
item-27, G1 import leg.

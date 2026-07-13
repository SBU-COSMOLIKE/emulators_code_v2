# Red-team batch — Implementer handoff for Architect review (2026-07-13)

Full inventory of every 45M-tagged change the Implementer (Opus 4.8) landed
across the red-team batch, plus the 45M-tagged queue units that preceded it.
Written as the review brief for the Architect (Fable 5) audit pass. The
per-unit records and the audited-rollout spec live in
[gates-and-board.md](gates-and-board.md); this note is the single-page index
into them, with the "audit hardest here" guidance the raw commit log does not
carry.

## State at handoff

- All units landed on `origin/main` via merge `8ce72a9` (parents `d67eaba` +
  branch tip `f139bb2`). The merge carried 5 notes conflicts, every one a
  clean keep-both (the Architect's spec sections + the 45M-72 anchor
  sections); zero code conflicts.
- Board grew 33 -> 40 gates; `board-selftest` 33/33; `run_board --list`
  validates the evidence map (rc 0); `compileall` clean across emulator +
  gates + generators + drivers.
- All acceptance below is Mac/CPU-only. The live GPU / cosmolike gate runs
  are workstation-owed (listed under OPEN).
- Convention: `45M-xx` codes appear only as commit-message provenance and in
  `notes/`; per 45M-85 they are stripped from all Python prose.

Reproduce the CPU acceptance from the repo root:

```
cd gates && python3 run_board.py --list                  # rc 0 => evidence map validates
PYTHONPATH=.. python3 gates/checks/board_selftest.py     # 33/33 ALL PASS
```

## 0. 45M-tagged queue units (foundation, not the board-truth batch)

| Unit | Commit | Files | Acceptance |
|---|---|---|---|
| 45M-22 (queue 43): CMB amplitude is the order-one factor `f=(A_s_ref/A_s)*exp(2(tau-tau_ref))`, ==1 at fiducial; law `as_exp2tau_ref`; retired `as_exp2tau` refused; refs persisted 0-d float64 | `4a19a17` | `emulator/losses/cmb.py`, `geometries/cmb.py`, `experiment.py`, `inference.py`, `results.py` | cmb-identity legs (known-answer + retired-law / missing-ref refusals + staging cross-check) |
| 45M-61 (unit 14h): one shared `screen_chi2` score-domain boundary at every chi2 consumer | `3f47d86` | `emulator/losses/core.py`, `training.py`, `diagnostics.py` | diagnostics-domain (DIAG-A) 20/20 (new gate) |
| 45M-57 / 45M-58 (units 60 + 14f): `ordinary_median` (quantile 0.5, not lower-middle) + every published reduction a finite float64 + post-reduction guard | `4846fdd` | `emulator/training.py` | `ge_c_eval_bs` Part 1b (even/odd median, batch invariance, mutation arm) |

## 1. Red-team board-truth + integrity (correctness fixes)

| Unit | Commit | Files | Gate |
|---|---|---|---|
| 45M-73 / 77 / 82: board reports the truth — dep-skip exits nonzero and runs no body; unknown selector id = usage error nonzero; selectors mutually exclusive; compile-lane skip is a distinct non-green code mapped to non-PASS | `d786975` | `run_board.py`, `board.py`, `checks/finite_contract.py`, `checks/board_selftest.py` (new) | board-selftest (BRD-A) |
| 45M-71 / 74: resume trusts BOTH code-digest + input-digest; RUNNING record persisted before gate code; per-attempt immutable logs; atomic (temp + os.replace) status / BOARD.md publication | `5947a05` | `run_board.py` | board-selftest (extended) |
| 45M-76: `_read_native_bool` parses `transfer_refined` by type (rejects truthy `"False"`) | `53334f0` | `emulator/results.py` | artifact-readback (ARB-A, new) |
| 45M-84: `stage_source` counts BOTH compact copies (params + target) + reindex, keeps the disk branch when the pair exceeds budget | `0ec1879` | `emulator/data_staging.py` | stage-ram (SRM-A, new) |
| 45M-79: scalar driver stamps `rescale="none"` so its own artifact is a valid fine-tune source | `48aac94` | `scalar_train_emulator.py` | census leg in artifact-readback |
| 45M-80: every cosmic_shear driver owns an explicit `family="cosmolike"` identity and rejects a wrong-family YAML (`family=None` no longer skips the check) | `e9943bc` | `cosmic_shear_train_emulator.py` + 3 siblings | family-first (FAM-A, new) |
| 45M-78: all 8 public entry points parse with strict `parse_args` (no `parse_known_args`) — a misspelled flag exits nonzero before any expensive work | `0139b1a` | 8 drivers (cosmic_shear train / sweep_ntrain / sweep_hyperparam / bakeoff / tune, scalar_train, generator_core, compute_cmb_covariance) | cli-strict (CLI-A, new) |
| 45M-75 schema half: `_validate_optimizer_opts` rejects eps=0 / non-finite, negative / non-finite weight_decay, non-positive lr, out-of-range betas before build | `7b4e4ec` | `emulator/training.py` | finite-contract Part J. Post-step-finite half = workstation-owed |
| 45M-81: required integer `--seed` owns a numpy Generator threaded through the 4 sampling sites + emcee, written to the chain header | `80315c3` | `compute_data_vectors/generator_core.py` | generator-seed (GEN-A, new) |

## 2. Red-team capstone — 45M-72 structured evidence map (FOUNDATION only)

Commit `1155c81`. Files: `board.py`, `run_board.py`, `checks/board_selftest.py`,
5 notes. New `Assertion(aid, anchor)` + `Gate.evidence`; `validate_evidence`
runs on every invocation (exit 2 on an orphaned anchor or a duplicate id); 7
red-team gates migrated with resolving `<a id>` anchors in their home notes;
board-selftest 26/26 -> 33/33 proving the 4 mutation arms (bad anchor / missing
note / duplicate id / malformed anchor) with the shipped board as the live
control. Additive: the other 33 gates are untouched and still run on their
`maps=` prose.

The full rollout — per-leg aids across all 58 `ctx.expect` sites + 27
check-script leg manifests + the runner's declared-vs-executed reconciliation +
a note anchor per leg + reconciling each gate's `home=` with the note that
documents it — is specified in gates-and-board.md ("The audited rollout") and
HELD FOR ARCHITECT AUDIT. It is a codebase-wide refactor of the verification
harness itself, deliberately not mass-edited unsupervised.

Design decisions the Architect should rule on (defaults chosen, none costly to
reverse):

1. Additive, not replacing: `maps=` prose kept alongside `evidence=`. Retire
   `maps` during rollout, or keep both (prose = human, evidence = machine)?
2. Validate-on-every-invocation: a `--list` now fails (exit 2) if any migrated
   anchor is orphaned by a note edit. Correct gate point, or should it move to
   a preflight / `--check` step so a note-only edit does not block a listing?
3. Explicit `<a id>` markers over heading slugs (chosen for stability against
   rewording). Confirm.
4. aid scheme `brd-a.exit-truth` (spec-code prefix + leg). Confirm before the
   rollout mints ~90 of them.
5. `home=` vs anchor-note: for the 7 migrated gates the anchor section lives in
   each gate's own `home=` note (they agree). The rollout must reconcile this
   for all 40.

## 3. Red-team didactics (doc-only; AST-with-docstrings-stripped identical per file)

| Unit | Commit | File | Note |
|---|---|---|---|
| 45M-85: strip all 84 internal `45M-*` codes from `emulator/` + `gates/` Python prose | `2807d3f` | 14 files | 11/14 AST-identical, 3 differ only in a human-facing string |
| 45M-83: staging row-coordinate glossary (disk / compact / loader rows + the `dump_rows` invariant + the `[9,2,9,5]` example) | `4dc0779` | `emulator/data_staging.py` | AST byte-identical |
| 45M-88: "How a gate teaches its evidence" + glossary terms | `bf62114` | `gates/board.py` (module docstring) | mine |
| 45M-86: family run lifecycle diagram + family decision table | `a9834fe` | `emulator/experiment.py` | sub-agent draft, re-verified |
| 45M-87: `transfer_state_dict` tensor-by-tensor shape-flow diagram | `b37b5d2` | `emulator/warmstart.py` | sub-agent draft, re-verified |
| 45M-89: diagnostics estimator-vs-verdict split (only 2/7 carry an in-code verdict) | `b1a375a` | `emulator/diagnostics.py` | sub-agent draft, re-verified |
| 45M-90: save<->rebuild reversible map | `1c3821c` | `emulator/results.py` | sub-agent draft, re-verified |

Audit hardest here. 86 / 87 / 89 / 90 were drafted by sub-agents under a strict
AST-identity gate (so no code could change), then independently re-verified by
the Implementer (compile + AST-identity + falsifiable-fact spot-checks). Two
reasons to scrutinize:

- They were built from one-line descriptions, and the Architect's own fuller
  specs for 83 / 85 / 86 / 89 / 90 are now on `main` beside them (the merge
  preserved both). Audit each implementation against its spec:
  - 45M-90 (`results.py`): the save<->rebuild reversible-map table is present;
    the spec (artifacts-inference-warmstart.md) also wants `state_dict` / HDF5
    group / dataset / attribute definitions + the ordinary / NPCE / transfer /
    refined ownership trees. Those definitional bullets are NOT yet done.
    Partial.
  - 45M-86 (`experiment.py`): module-docstring lifecycle diagram + family
    decision table present; the spec (conventions-and-workflow.md) also asks
    for per-method "state before / after" docs on `from_config` /
    `build_geometry` / `train`. Module-level only. Partial.
  - 45M-89 (`diagnostics.py`): verify the claim that only 2 of 7 diagnostics
    carry an in-code verdict (coverage_diagnostic's `coverage_limited`,
    local_linear_floor's floor rule).
  - 45M-87 (`warmstart.py`): verify the diagram legend completeness and the
    tensor-op accuracy of `transfer_state_dict` against the real code
    (matched vs grown key, zero-pad, rank-3 FiLM case).
- Voice is the Architect's domain. The Implementer checked <=90 col, legends
  present, no all-caps / double-dash, but the didactic register and the
  shape-flow diagram discipline are what the Architect gates on. Treat 86 / 87
  / 89 / 90 as first-pass drafts.

## New gates (board 33 -> 40)

board-selftest (BRD-A), artifact-readback (ARB-A), stage-ram (SRM-A),
family-first (FAM-A), cli-strict (CLI-A), generator-seed (GEN-A),
diagnostics-domain (DIAG-A).

## EXPLICITLY OPEN (not claimed done)

- 45M-72 full rollout — held for Architect audit.
- 45M-75 post-optimizer-step finite half — workstation confirmation-request
  (eps=0 AdamW repro).
- Workstation-owed live proofs: 45M-81 append-replay / worker-invariance,
  45M-79 live finetune parity, 45M-76 live save / forge / rebuild, full board
  rerun on a compile + cosmolike box.
- Pre-red-team queue tail (unrelated to 45M): 52 -> 55 -> 22(+20) -> 13(+01).

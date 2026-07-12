# The acceptance board: gates, harness, run history, lessons

Consolidated 2026-07-11 from gates-harness-user-run.md,
workstation-board-2026-07.md, gates-id-translation.md,
gates-checks-docs-plain-language.md, probe-generalization-bugs.md
(retired; full texts in git history).

## What the board is

Every feature lands with executable acceptance gates the USER runs on
the GPU workstation — no Claude session there. One command, zero
babysitting:

    git pull
    git checkout -- gates/board_config.json    # drop local edits
    python gates/run_board.py --check          # preflight only
    python gates/run_board.py                  # the whole board
    git add gates/logs && git commit && git push

The Architect then audits the RAW logs (full tee'd stdout+stderr per
gate + the computed acceptance values + an effective-config JSON dump)
— never summaries; a summary-only log fails the audit by construction.
Verdicts are written into the topic notes by the Architect, never by
the harness.

## Layout and operation

- `gates/run_board.py` — CLI + runner. Selectors: `--check`, `--list`,
  `--dry-run`, `--gate <id>`, `--tier`, `--from <id>`,
  `--force-rerun <id>` (flag BEFORE the id), `--debug`.
- `gates/board.py` — the registry: one Gate per entry (id, tier,
  spec_code, title, home note, run fn, needs). COUNT GATES BY
  ENUMERATING THIS REGISTRY, never by note arithmetic (a +1
  double-count survived several sessions once).
- `gates/checks/` — numeric check scripts (print acceptance VALUES,
  exit nonzero on any failed leg); `gates/configs/` — smoke YAMLs;
  `gates/logs/` — `<GATE_ID>.log` + `BOARD.md` + `board_status.json`.
- RESUME by default: PASS gates skip on rerun (skip lines + a green
  summary IS a pass); `--force-rerun` overrides; a crash loses only
  the in-flight gate.
- Preflight guards: base commit is an ancestor of HEAD; clean tree in
  the code dirs (board_config.json excluded); the cocoa env imports;
  data paths resolve; `driver_fileroot` is loud on placeholders.
- Portable config: board_config.json ships `"rootdir": null`, resolved
  from `$ROOTDIR` at load IN MEMORY (the file is never rewritten); an
  explicit file value wins; the resolved value + source is printed and
  recorded in every log. The golden config is
  `cosmic_shear_train_emulator.yaml` (renamed 2026-07-11).
- Golden byte-identity legs run the pinned base in a TEMPORARY
  `git worktree` (never checkout-in-place); the comparator strips
  exactly one machine-noise field (the trailing wall-clock column). A
  golden claim against a distant base is a CHAIN claim — bisect with
  intermediate bases before blaming the named feature.
- Harness hygiene rules born from reds: `sys.executable` never bare
  "python"; check scripts get `PYTHONPATH=<repo root>`; encode each
  gate FROM its home note (the spec of record), bending the encoding
  to the note; quiet terminal by default (one header + one verdict
  line per gate; `--debug` restores).
- cobaya-run must run FROM `$ROOTDIR` (the YAMLs' ./ paths anchor to
  the cwd — a cocoa-wide convention).

## The gate philosophy

- Two species per feature: an IDENTITY gate (feature off / golden base
  → byte-identical or bitwise-equal output — the strongest claim, and
  it doubles as a refactor-transparency proof) and a SMOKE gate
  (feature ON → the named behavior actually fires). A default-off knob
  left unset tests nothing (the ema-rewind lesson).
- A learning smoke must FAIL on a dead network: test OFF the training
  mean (a mean-predictor is then visibly wrong) and set the collapse
  bar BELOW the mean-predictor's value (the D-SPE2-5 rule) — the one
  gate defect a green board cannot surface.
- Evidence tiers: the Mac proves structure (py_compile, AST proofs,
  import smokes, numeric probes — see conventions-and-workflow.md);
  the workstation is the real acceptance. The first workstation run of
  anything new is DIAGNOSTIC by design: reds come back as raw logs and
  are fixed in the repo, never hand-patched on the box.
- Reds triage by layer — harness bug / check bug / config bug /
  library bug / contract bug — and fixing layer N unmasks layer N+1.
  The entire board history: ONE production-library bug (a lazy h5py
  import), ZERO physics bugs. The byte-identity discipline is why.
- Never-trust-defaults binds the harness too: every smoke YAML carries
  the FULL required train_args block set; drift proofs monkeypatch the
  sharpest code default (`make_activation.__defaults__`) to prove
  artifacts carry resolved values; the one self-reading check script
  resolves `$ROOTDIR` itself (D-GBC-1 — an audit that probes the
  loader only misses check-script self-reads).
- The generalization review axis: hunt code that works only because xi
  is block 0 of the 3x2pt vector (global `dest_idx` vs block-local
  indices; never hardcode 780/1560; derive from `total_size`) — xi
  survives block-0 coincidences, ggl/wtheta will not.

## The 32 gates (2026-07-11)

Human ids everywhere (CLI/logs/README); the legacy two-letter codes
survive only as each Gate's `spec_code`, printed once per log header.
The original translation table (ema-off-identity=GM-C, ...,
scalar-smoke=SPE-B) is in git history; today's registry:

- Training-stack gates (homes → training-stack.md): ema-off-identity,
  ema-smoke, production-diagnostic, single-phase-demotion,
  head-scheduler-override, eval-batch-invariance, berhu-loss,
  loss-schema-equivalence, berhu-anneal, ema-anneal, joint-training,
  weight-decay-census.
- Model gates (→ models-and-designs.md): head-activation-pin,
  relu-tanh-norm, npce-training.
- Data-cut gates (→ data-generation-and-cuts.md): param-window-cuts,
  triangle-shading.
- Artifact/adapter/warm-start gates (→
  artifacts-inference-warmstart.md): save-rebuild-drift,
  cobaya-adapter, finetune-identity, finetune-smoke,
  transfer-identity, transfer-smoke, geo-paths.
- Family gates (→ families-scalar-cmb.md /
  families-background-mps.md): scalar-identity, scalar-smoke,
  cmb-identity, cmb-smoke, bsn-identity, bsn-smoke, mps-identity,
  mps-smoke.

Notable current facts: geo-paths asserts the OLD flat geometry module
paths are dead (D-GEO5) — a pre-GEO test artifact fails rebuild loudly
by design; mps-smoke includes the MPS-DIAG pages leg; cmb-smoke is the
slow one (~400 serial CAMB calls); scalar-identity/scalar-smoke need a
`--force-rerun` on the next run (they gained legs after their last
green); cmb-identity and mps-identity carry the D-CM13 head legs
(ResTRF + n_tokens on the CMB fixture, ResCNN on the grid2d fixture:
attach, identity basis, epoch-0 identity, and the head
save->rebuild->predict bitwise round-trip — first run after the lift,
never yet green); save-rebuild-drift's cosmic-shear rescnn head
variant + pre-persistence refusal went GREEN on run 2b (07-11 night,
CUDA). The full regression pass is `python gates/run_board.py
--force-rerun-all` (added 07-11: reruns every SELECTED gate, composes
with --gate/--tier/--from, never deletes the resume map — an
interrupted pass re-run WITHOUT the flag resumes from whatever it
already re-proved).

## Run history (compressed; the full ledger is in git history)

| Run / date | What was new | Outcome → lesson |
|---|---|---|
| build, 07-07 | harness + 19 gates, Mac-gated | D-GH1..8: dry-run bypasses dep-skip; PYTHONPATH; sys.executable; encode from home notes |
| 1, 07-08 | first workstation board | wiring reds: --yaml resolution, a grep matching its own pattern, a check testing an impossible C1 tolerance (verify checks on a known-good case first), missing train_args masked by earlier failures |
| 2, 07-08 | run-1 fixes | the ONE library bug: results.py used h5py without importing it |
| 3, 07-08 | run-2 fixes | strip wall-clock in golden compares; the smoke's feature was OFF (rewind opt-in); assert on the right stream |
| 4–8, 07-08 | the evaluate leg | eight peeled layers: dep-edge ≠ artifact existence; `python_path` not `path:` (a legacy adapter in cocoa's cobaya fork shadows the class); cobaya `force: True`; pin param names from the covmat header; the dv-shape contract → `dv_return` + persisted `section_sizes` (GCT-D) |
| 10–11, 07-08 | GCT-D; post-GRF rerun | GREEN 18/18 twice; a fresh green after a refactor IS the refactor's transparency proof |
| MCMC smoke, 07-08 | manual cobaya-run | PASS, 500 steps; emulator 776 evals/s (1.2 ms/call warm) vs cosmolike 2270/s in that config |
| 12–12d, 07-10 | FTW gates (21) | D-FTW-1 (stamp the resolved rescale attr), D-GBC-1 (self-reading check), D-FTW-2 (finetune YAMLs carry no model: block) → GREEN 21/21 |
| triangle, 07-10 | first triangle-shading execution | D-GTB-1: classify plot layers by design contract (zorder 0), not rendering heuristics — first-execution risk of a gate authored blind |
| 13, 07-10 | TPE gates | 24/24 green FIRST TRY |
| SPE runs 1–5, 07-10 | scalar gates (25) | one REAL library bug caught (the getdist chain-root sidecar pairing); evaluate readback redesigned from stdout; 25/25 green |
| 32-gate run 1, 07-11 | the four family gate pairs + geo-paths (at the STALE HEAD 295d0fa, no --force-rerun) | 29/32: all four identity gates + geo-paths green FIRST TRY; three smoke reds, three REAL causes — (a) the cmb-smoke fixture wrote cobaya {value: X} params to the plain-numbers covariance script; (b) the background generator missed the wants-Cl quirk → stale cached CAMBdata → every dump row one cosmology (now a LOAD-BEARING comment + a dump-variance tripwire); (c) D-MP9: the boost's low-k law-space columns are PHYSICALLY constant (base exact) → pinned, not rejected. Lessons: a smoke's first execution tests the FIXTURE as much as the library; a cache loop with fixed input params serves stale physics silently; a guard built for "degenerate = bug" must carve out "degenerate = the base is exact here" |
| 2b, 07-11 night | the force-rerun trio, at HEAD 08cfc41 (pre-triage-fixes) | save-rebuild-drift ALL PASS on CUDA incl. the NEW head variant (bin-split persistence proven end to end) + the pre-persistence refusal; scalar-identity green with the reworded guard; scalar-smoke red on its FIRST diagnostics-leg execution — the hand-built fracs row (4-wide) vs DEFAULT_THRESHOLDS (5 entries) indexed out of bounds, the SAME landmine in all four smoke gates' legs, fixed by sizing fracs to exp.thresholds. Lesson: a hand-built fixture that mirrors a REAL run value (exp.thresholds) must derive every coupled width from it, not hardcode |
| 3, 07-11 late | first run at the FIXED code (HEAD 86577ff) | scalar-smoke GREEN (the fracs-width fix proven); cmb-smoke peeled to the covariance script's SECOND validation layer — the gate wrote CAMB's ombh2/omch2 where the script's whitelist wants its own omegabh2/omegach2 names (fixed; the lesson sharpened: a gate fixture MIRRORS the shipped example YAML, never re-types its keys from memory — two of the example's conventions were each re-invented wrong once); bsn/mps results pending at export |
| 4, 07-11 late | HEAD 2568f15 (the run-3 fix) | cmb-smoke FULLY GREEN (all legs incl. provider.get_Cl bitwise + 2 pages) — CME accepted end to end. bsn-smoke: the tripwire fired at spread EXACTLY 0.0 WITH the wants-Cl quirk in place — hypothesis FALSIFIED by the instrument built to test it. Standing fix: the bsn generator evaluates through model.logposterior(point, cached=False) (the standard lifecycle; cobaya routes the dropped-omegam/lambda params itself), Cl requirement removed again. Lesson: when a mechanism hypothesis about third-party internals fails once on the real machine, stop patching around it and switch to the documented API path |
| 5, 07-11 late | HEAD a491eb3 (the lifecycle fix) | THE LIFECYCLE FIX PROVEN: the bsn tripwire passed at spread 6.67e-2 (the predicted healthy H0-prior value) and BOTH trainings collapsed below their mean predictors (Hubble 0.101 vs 80.7; D_M 0.00949 vs 14). New failure one leg later: the in-process camb_truth reference used cocoa's relative "./external_modules/code/CAMB", which resolves only from $ROOTDIR — the check runs from the repo dir ("camblib.so not found"). Fixed in BOTH twins (bsn + mps camb_truth: absolute path from $ROOTDIR). Lesson: the "cobaya-run from $ROOTDIR" rule extends to IN-PROCESS get_model with cocoa's relative theory paths — resolve absolutely in check scripts, the subprocess legs (cwd=rootdir) never hit it |
| NEXT | re-run after pull | bsn-smoke past camb_truth (cobaya-vs-CAMB + diagnostics legs, first execution) + mps-smoke (D-MP9 + its own camb_truth fix); full green = GEO's acceptance + the baseline for D-CM12; then the user's full regression pass via `--force-rerun-all` |

## Check-script documentation rule (live)

Check scripts are what a person opens WHEN A GATE FAILS; they must
read as plain English under stress: every term of art defined at first
use or dropped; spec codes + note line-ranges confined to one header
line; main() and every helper carry substantive docstrings. The
2026-07-08 sweep (zero logic change, AST-proven) set the standard;
every new check is written under it from birth.

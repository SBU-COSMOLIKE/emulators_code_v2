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
| 6+7, 07-11 night | bsn-smoke GREEN (run 6, full gate); mps-smoke at the sped-up code (run 7, 12 min) | the k_max derivation worked (dumps in minutes) and pklin trained green; the BOOST training hit D-MP9's law-none error — the gate deliberately trains boost under law "none" and B = 1 at low k is the SAME physics under any law. D-MP9 AMENDED law-agnostic: pin constant columns for every law, keep only the wholly-constant dead-dump guard loud. Lesson: when a guard needs a carve-out, carve on the PHYSICS (partial-constant = flat physics; whole-constant = dead dump), not on a config axis (the law) that the physics does not respect |
| 8, 07-11 night | HEAD a03c410 (the law-agnostic pin) | D-MP9 PROVEN LIVE: the boost trained through the pinned low-k region (collapse 13.7 vs 319), pklin green, the MPS-DIAG pages landed. One leg further: camb_truth's Pk_interpolator request carried only the 3 probe redshifts — cobaya's spline demands >= 4 (the EXACT recorded first-run risk). Fixed: the request now carries a support containing the probes as nodes + range padding. Lesson: the recorded first-run risks are a checklist — grep them into the gate's own requests when authoring |
| 9, 07-12 00:12 | HEAD 4c65331 (the z-support fix) | **THE FIRST FULL 32/32 GREEN BOARD.** mps-smoke ALL PASS: P_lin / P_nl vs CAMB's OWN P(k, z) at rel 0.93% / 0.93% against the 5% bar (a 200-row 40-epoch smoke emulator), the range guard loud. This green is simultaneously GEO's acceptance and the board baseline for D-CM12 + the science thread. Eight runs, every red root-caused and recorded: 2 gate-fixture format bugs, 1 real generator bug (stale cached background -> the logposterior lifecycle), 1 falsified hypothesis (the wants-Cl quirk), 2 design amendments (D-MP9 born + made law-agnostic), 1 path-resolution class ($ROOTDIR), 1 cobaya constraint (>= 4 spline redshifts), 1 fixture-width landmine (fracs vs DEFAULT_THRESHOLDS) |
| 10, 07-12 00:19-00:55 | the `--force-rerun-all` regression pass, run at HEAD 4c65331 (BEFORE the overnight-batch merge — none of the NPCE/transfer/eq-6/phase legs existed in the executed tree) | 29/32. The cosmic-shear era + all four smokes + the D-CM13 head legs + D-MP9 pins re-proven green (mps-smoke rel 0.70%/1.05%, bsn-smoke rel 0.16%, tripwire spread 6.4e-2). THREE reds, all harness classes, none physics: (a) mps-identity's pre-amendment "un-standardizable" leg still expected the partial-constant raise D-MP9's law-agnostic amendment removed — its first execution since run 1 (DELETED; the D-MP9 legs assert the pin, the wholly-constant leg keeps the dead-dump raise); (b) bsn-identity's piecewise leg FLAKED: unseeded torch RNG (the synthetic nets differ every run) x a cross-METHOD comparison (the adapter's cubic interp1d vs the check's linear np.interp at z=1090, rtol 1e-3) — fixed like-for-like (the check builds the same cubic), per-assertion reports, and torch.manual_seed(0) in the bsn/mps/scalar identity mains; (c) ema-off-identity's golden leg invoked TODAY'S driver filename inside the pinned pre-rename worktree (46ec5e1 carries train_single_emulator_cosmic_shear.py) — run_driver now resolves by existence against a legacy-name map, loud when neither exists. Lessons: a stale leg is invisible until a force pass re-executes it; cross-path legs compare like with like; unseeded fixtures make reds unreproducible; a golden leg must run the pinned tree's OWN names |
| 11, 07-12 ~10:42-11:17 | the post-merge pass at merged HEAD 4e783fa (the user merged the seven commits and reran) | **30/32.** The three run-10 fixes PROVEN (ema-off-identity, bsn-identity, mps-identity all green), check_npce FIRST-EXECUTED GREEN in all four family identity gates, the two-phase phase-discipline legs green, transfer-smoke + npce-training green. TWO first-execution reds: (a) cmb-smoke leg 2b (eq-6): compute_cmb_covariance built the clpp re-lensing array truncated at lens_lmax+1 = 351 — CAMB's get_lensed_cls_with_spectrum REFUSES anything shorter than Params.max_l+1 ("clpp must go to at least Params.max_l", verified in CAMB's results.py; the solve ran lmax 500). rc=1 with an EMPTY stderr because cobaya's exception hook logs to STDOUT — and the leg's failure detail carried stderr only, so the red arrived blind. FIXED both: clpp now taken full-length from cambdata.get_lens_potential_cls(raw_cl=False)[:, 0] (which also stops silently DELENSING every L above lens_lmax at fiducial — a correctness bug the crash masked), and all three run_tool failure details in cmb_smoke.py now carry stdout+stderr tails. Lessons: a subprocess red must capture BOTH streams (cobaya hooks swallow stderr); an API executed for the first time is a first-execution risk even when the algebra was probe-proven — the probes never called real CAMB. (b) transfer-identity's check_diagonal: red, log not yet inspected (the harness line names only the exit code) — root cause OPEN, needs gates/logs/transfer-identity.log |
| 12, 07-12 12:53-13:04 | the three-gate force-rerun at HEAD 7f455e6 (pre-delta: the THREE-leg eq-6 oracle in the executed tree) | **31/32.** cmb-identity GREEN — the eq-6 legs first-executed under torch: truth 5.89e-14, discrimination ~1e16, band 3.02e-14. cmb-smoke GREEN — the FIRST full eq-6 execution on real CAMB: blocks symmetric/PSD/off-diag alive (\|off\|max 7.28e-07), the step study AND the fractional-amplitude weight keys asserted in provenance (coord fractional_band_amplitude, policy smooth-response band projection). The eq-6 normalization fix is workstation-proven; the run-11 clpp-length fix proven too. transfer-identity: 58/59 legs green, ONE red — "cross-family transfer base raises": a GATE-FIXTURE bug, root-caused on the Mac. The leg points the grid2d run's transfer.from at the diag_transfer artifact saved WITH a transfer_base group two legs earlier, so warmstart.load_source's chaining refusal ("no chaining") fires inside _load_diag_transfer BEFORE the family-kind check, and the needle test ("never" + "families") fails on the wrong message. The library rule is correct and correctly ordered (experiment.py _load_diag_transfer: kind check right after load, message "a transfer never crosses families"); the fixture hands it an artifact that trips a different, equally-correct refusal first. Lesson: a loud-error leg must use a fixture that is INVALID ONLY in the way under test — one extra defect and the wrong guard answers | 
| NEXT | fixture fixes + one rerun | Two gate-file-only changes land together: (a) the transfer-identity cross-family leg gets a PLAIN grid base artifact (no transfer_base group) as its transfer.from; (b) cmb-identity's five-leg oracle delta (convention-honest fake, already Architect-accepted at 86db0b4) reaches the workstation via the merge. Then `python gates/run_board.py --force-rerun cmb-identity transfer-identity` (cmb-smoke needs no rerun: the delta touches only cmb_identity.py and smoke is green at the same producer). Then: the bounded grid2d staging unit, the five artifacts, EMUL2, and the dense-covariance audit. |

## Check-script documentation rule (live)

Check scripts are what a person opens WHEN A GATE FAILS; they must
read as plain English under stress: every term of art defined at first
use or dropped; main() and every helper carry substantive docstrings.
SUPERSEDED 2026-07-12 on codes: spec codes appear NOWHERE outside
notes/ — not even in a header line (the user ruling in
conventions-and-workflow.md "In-file documentation"; the gate registry's
spec_code field stays as the notes-ledger key and is never rendered).
Note-file pointers ("spec: notes/x.md:lines") are the crosswalk. The
2026-07-08 sweep (zero logic change, AST-proven) set the standard;
every new check is written under it from birth.

## The eval-batch check must stop executing at import (red-team 45M-04, 2026-07-12, Architect-VERIFIED; queue 32)

gates/checks/ge_c_eval_bs.py is the one production exception to the
no-global-data rule (found by the red team's free-name scan over all
92 Python files; verified by reading). C and DV (5000 x 780, CUDA when
available), the model, the loss, and the thresholds all allocate at
module scope (:33-:60); toy_data / per_row_chi2 / timed close over
them; the complete test executes at import time and ends in sys.exit
(:149). The module docstring says so itself (:4 "there is no main
function") — that transcription waiver is now contrary to the standing
house rule (functions take data/state as parameters; a rare
unavoidable global read carries the visible warning marker), and these
globals are avoidable. The board survives only because it runs the
file as a subprocess script (board.py:369, ctx.run_check); importing
the module from anywhere else allocates a device tensor, runs the
whole test, and kills the importing process, and the helpers cannot be
probed in isolation.

Contract (Implementer):

1. main() owns seeding, device selection, fixture allocation,
   execution, printing, and the final status;
   `raise SystemExit(main())` only beneath
   `if __name__ == "__main__":`. Importing performs no allocation,
   timing, output, or exit.
2. toy_data takes the input and target tensors explicitly;
   per_row_chi2 takes model, loss, tensors, row count, and batch size;
   timed becomes a top-level helper with every dependency in its
   signature.
3. Numeric controls stay as named constants or main locals with
   definitions: validation rows, input width, output width, tolerance,
   warmup repetitions, timing repetitions.
4. The tensor-mechanics prose states: slicing returns a view; expand
   is a repeated view without copying; torch.cat allocates the padded
   batch; clone detaches the saved per-row result from temporary batch
   storage.
5. Numbers preserved: the partition-invariance values and the CUDA
   timing behavior are unchanged.

Acceptance: importing the module is side-effect free; the free-name
scan reports no data-global reads; direct script execution still
returns nonzero on a forced partition mismatch; the board continues to
execute the real check, not a helper-only substitute.

## UNIT 4 EXTENDED (45M-69, seventeenth batch, 2026-07-12): the board reuses PASS verdicts after executable code changes

Finding (red team, CONFIRMED by read — the mechanism is plain):
_passed is status-only (run_board.py:866-868,
`status.get(gate_id, {}).get("status") == "PASS"`); the resume
path skips any stored PASS (:952) and dependency resolution
accepts it (:960); board_status.json records only
status/detail/ts (:965-967). Nothing ties a verdict to the
executable tree that produced it: HEAD-at-run goes into the raw
log but is never read back; the base-notes ancestor check passes
for every descendant; --force-rerun-all is manual recovery, not
automatic truth. Reachable wrong result: gate G passes at commit
A; commit B regresses the code; a normal board run at B skips G
"already PASS" and BOARD.md certifies a tree that never executed
the gate — a nominally green board can combine verdicts from
different executable versions.

Contract (the red team's clauses adopted, plus a scope
alignment and propose-first):

1. A deterministic executable-surface digest covers tracked
   emulator code, root Python drivers, gate
   definitions/checks/configs, and the other executable inputs.
   SCOPE ALIGNMENT (Architect): the digest surface is the SAME
   surface entry 4's dirty-tree-watch fix expands to —
   compute_data_vectors/, cobaya_theory/, and syren/ INCLUDED —
   one executable-surface definition for both mechanisms.
   Excluded: generated gate logs, BOARD.md/board_status.json,
   the machine-local board_config.json override, notes/ and
   documentation-only files.
2. The digest is stored with every PASS or FAIL record.
3. A PASS is reusable only when its stored digest equals the
   current digest.
4. A legacy record without a digest is STALE, never implicitly
   current.
5. Dependency checks accept only current-digest PASS records.
6. --list and BOARD.md display stale verdicts as STALE, not
   PASS.
7. --force-rerun / --force-rerun-all are preserved as explicit
   rerun controls, not correctness patches.
8. The raw log's commit identifier stays provenance and never
   substitutes for the digest: committing logs or notes must not
   invalidate science gates.
9. The digest definition (hash construction, file enumeration,
   ordering stability) is design-sensitive: PROPOSE-FIRST in the
   unit-4 landing, per the large-unit rule.

Red legs (CPU-only harness testing):

- PASS under digest A followed by executable digest B reruns
  instead of skipping;
- a stale prerequisite produces dependency-stale behavior, never
  downstream execution under an old PASS;
- a legacy PASS lacking a digest is stale;
- a documentation-only or generated-log commit leaves the digest
  unchanged;
- changing one byte in emulator code, a driver, a gate body, or
  a committed gate fixture changes the digest;
- a same-digest interrupted run still resumes;
- --force-rerun executes even when the digest matches;
- mutation arm: restore status-only _passed — the
  executable-change leg must fail.

Interim rule, recorded: until unit 4 lands, --force-rerun on
every gate whose surfaces changed remains the manual truth
control (the current owed rerun list already follows it).
Placement: unit 4 (harness/CLI truth), where queued — no new
number. USER-VISIBLE: BOARD.md gains STALE; a green board
certifies the current tree only.

## RED-TEAM BOARD-TRUTH + INTEGRITY BATCH — Opus IMPLEMENTED (2026-07-12, Architect asleep, user-authorized direct implement)

Seven reproduced red-team defects implemented on the branch (batch grant;
merge/push to main stays user-only). Each self-committed with a board-listed
CPU gate; all green on the Mac (no torch for the harness ones, Cocoa torch for
the module ones). Board gate count 33 -> 38 (+board-selftest, +artifact-readback,
+stage-ram, +family-first, and the earlier diagnostics-domain).

- 45M-73 / 45M-77 / 45M-82 (commit d786975): the board runner reports the truth
  about what ran. run_selection returns a categorized summary (passed / resume /
  failed / skipped_dep) + "incomplete"; main exits nonzero unless EVERY selected
  gate is current PASS (a dependency-skipped gate that ran no body no longer
  exits 0). select_gates validates every --gate / --from id (SelectionError with
  suggestions, not a warning-then-subset); main validates --force-rerun ids and
  refuses an empty real-run selection; --gate/--tier/--from are mutually
  exclusive; the resolved selection is printed. finite_contract returns exit
  code 2 (LANE_UNAVAILABLE) when its mandatory torch.compile lane cannot run,
  and the board wrapper maps any nonzero to non-PASS. Gate: board-selftest
  (BRD-A) 17/17.
- 45M-76 (commit 53334f0): results.py read transfer_refined with
  bool(f.attrs.get(...)) -> the string "False" is truthy and would load drifted
  weights. New _read_native_bool parses by type (native bool accepted, absent ->
  default, string/int refused naming the file + native-boolean schema). Gate:
  artifact-readback (ARB-A). Live save/forge/rebuild is workstation-owed.
- 45M-84 (commit 0ec1879): stage_source counted only the dv[rows] bytes but
  materializes BOTH C[rows] and dv[rows]; a narrow-output scalar dump chose RAM
  when the two copies didn't fit. Now sums params + dv (each own dtype/width) +
  the reindex array; prints the predicted bytes + branch. Gate: stage-ram
  (SRM-A) 6/6 (mocked memory).
- 45M-79 (commit 48aac94): scalar_train_emulator omitted the rescale attr, so
  warmstart.load_source refused the scalar driver's own artifact as a fine-tune
  source. Now stamps rescale="none" (resolved fact); load_source unchanged.
  Census leg in artifact-readback. Fuller shared-provenance-assembler is the
  unit-24 amendment.
- 45M-80 (commit e9943bc): the direct cosmic_shear drivers passed family=None,
  skipping require_family_block; a wrong-family YAML trained under the wrong
  identity (a scalar YAML died later at run_tag KeyError). The cosmolike family
  now has an explicit "cosmolike" validator identity (owns no data-block key,
  rejects any other family's block naming its driver); the four cosmic_shear
  drivers default family="cosmolike" and always check; dispatcher prose deleted.
  Gate: family-first (FAM-A) 15/15.

STILL OPEN from the red-team batch (queued, not yet implemented, each large):
45M-71 (resume input-digest + atomic RUNNING), 45M-74 (atomic per-attempt log +
temp-file status/BOARD.md publication) -- one board state-machine visit; 45M-72
(structured assertion-ID -> note-anchor evidence map, only 2/33 spec_codes occur
in their home notes); 45M-78 (strict parse_args across 8 CLI entry points); 45M-81
(generator sampling seed + replayable RNG); the documentation batch 45M-85 (strip
audit codes from Python prose -- note: the code committed this session ADDED some,
so that sweep must include gates/checks/board_selftest.py, diagnostics_domain.py,
etc.) + 45M-83 (row-coordinate glossary) + 45M-86..90 (lifecycle / warmstart /
gates-teach / diagnostics / save-rebuild didactics). 45M-75 is a workstation
confirmation-request (post-optimizer-step finite boundary; eps=0 AdamW).

## RED-TEAM BATCH continued — Opus (2026-07-12, user-authorized "do them ALL")

Second wave of the red-team batch implemented on the branch (batch grant;
merge/push to main stays user-only). Board now 40 gates. All CPU-verifiable
legs green on the Mac (Cocoa torch where the module needs it).

- 45M-71 + 45M-74 (commit 5947a05): board resume trusts BOTH an
  executable-surface digest and an input digest; a config change / mutated
  referenced YAML / interrupted attempt reruns and never satisfies a
  dependency. A RUNNING record is persisted (atomically) before any gate code;
  each attempt writes its own immutable per-attempt log (temp + os.replace);
  board_status.json + BOARD.md go through temp + os.replace; --list / BOARD.md
  distinguish current PASS / stale-code / stale-input / interrupted; a
  status/log digest mismatch is loud. Gate board-selftest (BRD-A) now 26/26
  (33/33 after the 45M-72 evidence-map foundation adds seven legs; 39/39 after
  the raw-log-trust repair below adds six).
- 45M-75 schema half (commit 7b4e4ec): _validate_optimizer_opts rejects a
  zero / non-finite Adam eps, a non-finite / negative weight_decay, a
  non-positive / non-finite lr, and a beta outside [0,1) before the optimizer
  is built (the 0/0 zero-gradient trap). finite-contract Part J. The post-step
  finite boundary remains the workstation confirmation half.
- 45M-78 (commit 0139b1a): all eight public entry points parse with strict
  parse_args (no parse_known_args) -- a misspelled flag exits nonzero before
  any data / artifact / CAMB / worker / output-root work. Gate cli-strict
  (CLI-A) 14/14.
- 45M-80 (commit e9943bc): the direct cosmic_shear drivers own an explicit
  "cosmolike" family identity and reject a wrong-family YAML naming the right
  driver (family=None no longer skips the check). Gate family-first (FAM-A).
- 45M-79 (commit 48aac94): the scalar driver stamps rescale="none" so its own
  artifact is a valid fine-tune source. Census in artifact-readback.
- 45M-76 (commit 53334f0): _read_native_bool parses transfer_refined by type
  (the truthy "False" no longer loads drifted weights). Gate artifact-readback
  (ARB-A).
- 45M-84 (commit 0ec1879): stage_source counts BOTH compact copies (params +
  target). Gate stage-ram (SRM-A).
- 45M-81 (commit 80315c3): required integer --seed owns a numpy Generator
  threaded through the four sampling sites + emcee; recorded in the chain
  header. Gate generator-seed (GEN-A). Append-replay + worker-invariance +
  full RNG-state manifest are the workstation remainder.
- 45M-85 (commit 2807d3f): all 84 internal 45M-* audit codes stripped from
  emulator/ + gates/ Python (comments / docstrings / gate descriptions /
  printed headings); 11 of 14 files AST-identical, the other 3 differ only in a
  human-facing runtime string; zero 45M remain.
- 45M-83 (commit 4dc0779): the data_staging row-coordinate glossary (disk /
  compact / loader rows + the dump_rows[j] invariant + the [9,2,9,5] example +
  the discarded param_stats scale + grid2d moment order); AST byte-identical.

STILL OPEN:
- 45M-72: FOUNDATION LANDED this session (the Assertion schema + Gate.evidence
  field, validate_evidence run on every invocation, the seven red-team gates
  migrated with one headline assertion each + their home-note <a id> anchors,
  and the board-selftest evidence-map legs proving the validator rejects a bad
  anchor / missing note / duplicate id / malformed anchor; board-selftest
  26/26 -> 33/33). Additive: the other 33 gates are untouched and still run on
  their maps= prose. The AUDITED ROLLOUT (per-leg ids threaded through all 58
  ctx.expect sites + 27 check-script leg manifests + the runner's declared-vs-
  executed reconciliation + note anchors per leg + reconciling each gate's
  home= with the note that documents it) is specified in "The audited rollout"
  subsection above, held for Architect audit before it lands (a codebase-wide
  refactor of the verification harness itself).
- 45M-86 / 87 / 88 / 89 / 90: DONE (the didactic documentation batch; each
  doc-only, py_compile clean + AST-with-docstrings-stripped identical per
  file). 86 = experiment.py module docstring gains the six-stage family run
  lifecycle diagram (with legend) + the family decision table (keys +
  validators read from from_config) (a9834fe). 87 = warmstart.py
  transfer_state_dict gains the tensor-by-tensor shape-flow diagram, matched
  vs grown keys, zero-padding, the rank-3 FiLM case (b37b5d2). 88 = board.py
  module docstring gains "How a gate teaches its evidence" (the four records
  a reviewer reads) + the assertion / evidence glossary terms (bf62114). 89 =
  diagnostics.py gains the estimator-vs-verdict split (only 2 of 7 carry an
  in-code verdict: coverage_limited + local_linear_floor) (b1a375a). 90 =
  results.py save_emulator gains the reversible map pairing every saved value
  with its rebuild read site + the labelled provenance-only keys (1c3821c).
  The four experiment/warmstart/diagnostics/results units were drafted by
  gated sub-agents under a strict AST-identity check, then independently
  re-verified (compile + AST + falsifiable-fact spot-checks) before commit.

## Structured evidence map — gate contract anchors (45M-72 foundation)

The board carries a structured evidence map (`Gate.evidence`, a tuple of
`Assertion(aid, anchor)`): each migrated gate names, in code, the stable
assertion id it proves and the home-note anchor that documents it. The
runner validates the whole map statically before any gate runs — every
anchor must resolve to an explicit `<a id="...">` marker in `notes/`, and
no two assertions anywhere on the board may share an id — so the free-form
`maps=` prose can no longer drift from the note it cites (the gap this
increment closes: before it, almost none of the board's tests named a note
passage the note still carried). Each anchor below is that marker; a gate
whose home note is elsewhere carries its anchor in that note, not here.

This foundation migrates the seven red-team gates with one headline
assertion each; the per-acceptance-leg ids (every `ctx.expect` and every
external check leg emitting a unique id the runner reconciles against the
declared set) are the audited rollout specified below.

<a id="brd-a-board-truth"></a>
**board-selftest (BRD-A) — the wrapper reports the truth about what ran.**
A dependency-skipped selected gate exits nonzero and runs no test body; an
unknown `--gate` / `--from` / `--force-rerun` id is a usage error with a
suggestion and a nonzero exit; the run selectors are mutually exclusive;
and the finite-contract compile-lane skip is a distinct non-green exit code
the board wrapper maps to non-PASS. Proven by `gate_board_selftest` /
`gates/checks/board_selftest.py`.

### The audited rollout (45M-72 remainder, for Architect audit)

The foundation above lands the schema (`Assertion` + `Gate.evidence`), the
static validator (`validate_evidence` in `run_board.py`, run on every
invocation), the seven migrated gates, and the mutation leg in
board-selftest that proves the validator rejects a missing anchor and a
duplicate id. It is additive: a gate with no `evidence` is untouched, so
the other 33 gates still run on their `maps=` prose. The remaining work,
which touches all 40 gates + 58 `ctx.expect` sites + 27 check scripts + the
home notes and so warrants an Architect audit before it lands:

1. **Per-leg ids.** Give `RunContext.expect` an `aid=` argument, and give
   every gate one `Assertion` per acceptance leg (not one headline). The
   context records each executed aid.
2. **Executed-vs-declared reconciliation.** After a gate runs, the runner
   compares the set of aids the body actually emitted against the set the
   gate declared in `evidence`; a leg that was declared but never executed
   (a silently-dropped check) fails the gate.
3. **External-script leg manifests.** Each `gates/checks/*.py` prints a
   machine-readable manifest of the leg ids it ran (one line per leg); the
   runner parses it and folds those ids into the executed set, so an
   external check's dropped leg is caught the same way.
4. **Note anchors for every leg.** Each home note gains an `<a id>` marker
   per acceptance leg, and each gate's `home=` is reconciled with the note
   that actually documents it (several red-team gates are written up here in
   `gates-and-board.md` while their `home=` names a domain note).

## Audit repair queue — Opus Implementer (2026-07-13)

Working the ordered Implementer queue in
[red-team-audit-and-didactics-2026-07-13.md](red-team-audit-and-didactics-2026-07-13.md).

### Queue 1a DONE: resume trusts the raw log's digest (critical reopen)

The audit's critical reopen: `_resume_state` checked the code and input
digests but never the log digest, and `_log_digest_mismatch` only annotated
`BOARD.md`, so deleting, truncating, or editing a raw log left the status a
current PASS and a normal rerun skipped it.

Repair (commit pending, `gates/run_board.py` + `gates/checks/board_selftest.py`):

- One shared predicate `_log_stale(record)`: a stored PASS is unverifiable
  (and therefore stale) when it names no log, stored no log digest, the log
  file is gone, or the bytes no longer hash to the stored digest.
- `_resume_state` returns a new non-green `stale-log` state on that predicate,
  after the code/input digest checks. The runner's skip, the rerun message,
  the `--list` / `BOARD.md` state column, and `_dep_current_pass` all consume
  the one `_resume_state`, so the display and the skip cannot disagree (the
  audit's "same state" requirement). The old `_log_digest_mismatch` annotation
  stays as extra detail.
- `board-selftest` gains a `check_log_trust` section driving the REAL
  `_resume_state`: valid-control PASS, then truncation, edit, deletion, and a
  missing stored digest each read `stale-log`; a stale-log record is not a
  current dependency; a load-bearing mutation arm flips PASS -> stale-log on a
  byte edit. `pass_record` now cites a real seed log that `drive_main`
  materializes, so a seeded current PASS carries verifiable log evidence.
  Board-selftest 33/33 -> 39/39, ALL PASS on the Mac; `--list` still validates.

### Queue 1c DONE: preflight watches the whole executable surface

The audit's preflight hole: the dirty-tree watch covered `emulator/`,
`gates/`, and root drivers, but not `compute_data_vectors/`, `cobaya_theory/`,
or `syren/`, so a dirty generator, adapter, or vendored formula could pass the
reproducibility check. `run_board.py` now defines the executable surface once
as `_EXECUTABLE_DIRS = (emulator, gates, compute_data_vectors, cobaya_theory,
syren)`, and preflight (b) watches that plus the root drivers. Compile clean,
`--list` rc 0, board-selftest 39/39 unchanged.

STILL OPEN in queue 1: the reviewed executable/input **manifest** replacing the
two coarse digests (`_gate_code_digest` omits shared helpers / runner / imported
production modules; `_gate_input_digest` hashes every YAML in `yaml_dir`). Per
the audit this is a proposal first; `_EXECUTABLE_DIRS` is the shared-surface
seed it will build on.

### Queue 4 DONE: optimizer + CMB schemas validate by type, not coercion

The audit's reopen: `_validate_optimizer_opts` and `validate_cmb` converted
public numeric controls with `float(...)`, so a bool (a subclass of int,
`float(True)` is 1.0) and a numeric string (`float("0.1")` is 0.1) were
silently admitted as learning rate, eps, weight decay, betas, `as_ref`, or
`tau_ref`.

Repair: one shared predicate `_is_finite_real(value)` (a finite, non-boolean
`int`/`float`) added to both `emulator/training.py` and `emulator/experiment.py`
(the same predicate the other public scientific controls already use inline).
`_validate_optimizer_opts` and `validate_cmb`'s `as_ref`/`tau_ref` now validate
through it before any range check, keeping the documented domains (positive lr
and eps, nonnegative weight decay, betas in [0,1), positive `as_ref`).

Gates: finite-contract Part J gains the bool / string red legs (lr, eps, weight
decay, betas) plus an int-lr control; `cmb-identity` gains `check_cmb_ref_schema`
(a valid control + `as_ref`/`tau_ref` as True / False / numeric string / NaN /
inf / zero / negative each refused). Verified on the cocoa-torch interpreter:
the optimizer logic via a direct `_validate_optimizer_opts` import probe (18
red legs reject, controls accept) and the CMB legs live in a full green
`cmb-identity` run; the finite-contract Part J live run stays workstation-owed
(that gate imports cosmolike). The post-optimizer-step finite check on the
parameters and optimizer state each step remains the workstation companion.

### Queue 6a DONE: README current-state factual repair (the false statements)

The audit listed READMEs that contradict the landed tree. Corrected the
unambiguous false statements:

- `gates/README.md`: the "known gaps" paragraph claimed the dirty-tree watch
  covered only `emulator/` + `gates/` + drivers and that an unknown selector
  "warns but proceeds (exit 0)". Both are now false (queue 1c and the board's
  45M-73/77). Rewritten to the current behavior: the watch covers the whole
  executable surface, and an unknown `--gate` / `--from` / `--force-rerun` id
  is a nonzero usage error rejected before any test runs.
- `gates/README.md` "## The 32 tests" and `emulator/README.md` "the acceptance
  board (32 gates)": de-hardcoded. `board.py` `BOARD` is named as the
  authoritative registry (40 at this writing); the reader is pointed at
  `run_board.py --list`, and the table is flagged as not yet listing the newest
  gates (avoids re-introducing enumeration rot).
- `emulator/README.md`: the two `45M-12` public-prose leaks (the Simpson rule)
  replaced with the executed rule + a `notes/families-background-mps.md`
  pointer; the stale grid2d "open production blocker because it materializes
  unthinned float64 selections" replaced with the landed bounded-staging
  behavior (its remaining review item is lifecycle/evidence).

Verified: an untruncated grep for `45M-12`, `32 test`/`32 gate`, "warns but
proceeds", and the retired grid2d claim returns nothing in either README.
STILL OPEN in queue 6: the two-layer gate-mechanism description + the
declared-vs-executed honesty in `gates/README.md`; the root/package README
restructuring (stable role sentence + a separate dated "current limitations"
table); and the broader in-file first-time-ML didactic campaign (the priority
list in the audit spec), each doc-only with AST-identity + compile + an
untruncated stale-pattern scan over the full pattern family (`unit \d`, `(f)`,
"Architect", "ruling", not just `45M`).

Queue items 2 (evidence rollout), 3 (staging seeded-order truth), 5
(workstation evidence), and the rest of 6 remain.

## Architect batch audit of the overnight batch (2026-07-13, Fable)

Scope: everything landed on main after `05d4937` (queue 43 `4a19a17`,
14(h) `3f47d86`, 60+14(f) `4846fdd`, the red-team wave 45M-71..90,
repair queue 1a `b6cfd87` + 1c `7f69c35`) plus the two then-unmerged
commits on this branch (queue 4 `c9ace04`, queue 6a `dce3d69`).
Evidence: the Architect's OWN gate reruns from this branch's tree on
the cocoa-torch interpreter — `run_board --list` rc 0, board-selftest
ALL PASS, diagnostics-domain, cmb-identity (incl. the queue-4 schema
legs), stage-ram, artifact-readback, family-first, cli-strict,
generator-seed all green, ge_c_eval_bs rc 0 (Part 2 timing stays
workstation-owed) — plus independent code reads of every contract
surface. Green claims were re-executed, not trusted.

Verdicts:

- Queue 43 / 14(h) / 60+14(f): PASS against the adjudicated contracts.
  The order-one factor, its 0-d float64 persistence, the retired-law
  and missing-ref refusals, the shared screen_chi2 consumer census,
  and the float64 ordinary median are all contract-true.
- Red-team wave 45M-71..90: implementations ACCEPTED post-hoc (they
  landed without Architect adjudication while the Architect was
  offline; the persisted handoff + honest OPEN lists + holding the
  45M-72 rollout are the right mitigations, and this audit closes the
  gap). The red-team audit's reopens are CONFIRMED independently:
  - stage_source order defect CONFIRMED in the code: the resident
    branch returns sorted-unique rows + arange while the disk branch
    returns the original unsorted selection (duplicates included), so
    the same seed trains different minibatches depending on host RAM;
    the module's [9,2,9,5] example misstates the disk branch; the
    stage-ram gate sorts before comparing, hiding exactly this. A
    green stage-ram run is NOT evidence against the reopen.
  - the resume digests are the narrow surfaces the audit described
    (gate body + literal check paths; whole yaml_dir), honestly
    framed as the seed of the queue-1b manifest (proposal-first).
- Repair queue: 1a VERIFIED (stale-log is a first-class state of the
  one `_resume_state` consumed by skip, display, and dependencies;
  selftest arms rerun green); 1c VERIFIED (`_EXECUTABLE_DIRS` matches
  the unit-4-EXTENDED scope ruling exactly); queue 4 and queue 6a
  AUDITED PASS (typed non-bool non-string schema; README stale-claim
  grep re-verified empty by the Architect) — both CLEARED TO MERGE.

Evidence-map rulings, now BINDING (adjudication authority stays with
the Architect; the red team's five answers are ratified on the merits,
so there is no practical conflict — recorded so "binding" framing does
not drift the role file):

1. Keep BOTH `maps=` (human promise) and `evidence=` (machine legs);
   the rollout makes them agree.
2. Validate the evidence registry on EVERY invocation incl. `--list`
   (listing an invalid registry is false advertisement; the failure
   is loud and cheap to fix).
3. Explicit `<a id>` anchors, stable under heading rewording.
4. Assertion ids are `<gate-id>.<plain-leg-name>`
   (e.g. `board-selftest.exit-truth`) — NO spec-code prefixes; the
   raw log must teach a reader, internal codes stay in notes/.
5. Each gate's executable legs anchor in its one declared `home=`
   note; other links are prose only.
6. ADOPTED as rollout clause: declared-vs-executed reconciliation
   with exactly one terminal result per aid per run (PASS / FAIL /
   explicit non-green UNAVAILABLE); external check scripts emit a
   machine-readable per-leg manifest; a missing, duplicate, unknown,
   or conditionally-omitted aid is a red gate. This is the
   dead-network rule applied to the harness itself.

The audited-rollout spec (45M-72 remainder, four items above) is
APPROVED under these six rules.

Sequencing ruling: queue 3 (staging seeded-order truth) runs FIRST —
it is the one open science-affecting defect. Then the queue-1b
manifest PROPOSAL (Opus proposes, Architect reviews before landing),
then the evidence rollout (queue 2), then the pre-red-team tail
50 -> 52 -> 55 -> 22(+20) -> 13(+01). Unit 50 (epoch chunking) must
preserve queue 3's canonical order under chunking and reuses its
invariance legs — another reason 3 lands first. Queue 6 didactics
continue doc-only alongside; queue 5 workstation evidence rides the
user's next workstation session.

Queue-3 contract (pinned):

- The canonical selected-row order is the ORIGINAL seeded selection
  order. Compact storage may be sorted internally, but stage_source
  then returns an explicit local-coordinate array mapping the original
  selection order into that storage — never a bare arange.
- Duplicate indices in the selection REFUSE loudly (the selection is
  a unique permutation prefix; a duplicate reaching staging is
  corruption). A caller census precedes the refusal clause.
- The banner prints all three named byte terms (params, target,
  reindex), their exact total, the budget, the comparison operator,
  and the chosen branch — no arithmetic that does not sum.
- The fit comparison's equality policy is explicit and gated with
  below / equal / above legs (keep strict `<`: equal-to-budget
  streams from disk, the conservative side).
- The gate drives the REAL loader and epoch permutation in BOTH
  storage regimes and proves parameters, targets, sibling-dump rows,
  minibatch membership, AND minibatch order identical under the same
  seeds; a mutation arm restoring arange must FAIL; no sorting or
  dedup inside any assertion.
- The module's row-coordinate example is corrected in the same
  commit (the disk branch returns the original sequence).

Didactics verdicts: 45M-87 (warmstart diagram) PASSES voice and
accuracy (legend complete, rank-3 FiLM case correct). 45M-89: the
red team is right and the handoff's "2 of 7" is wrong BY THE MODULE'S
OWN TABLE — exactly one function (coverage_diagnostic) returns an
in-code verdict; the "just two compute a verdict" prose contradicts
the table it introduces; fix rides queue 6. The Architect's own
untruncated residue scan over the FULL pattern family (45M, unit N,
increment, Architect/Implementer, red team, ruling, adjudicat-, D-*,
POL-*) counts 108 lines across ~25 Python files — larger than the
audit's 48/16 `45M`-adjacent count; the queue-6 completion gate must
record its pattern family + identifier allowlist and scan untruncated.

Governance record: (1) the wave-without-adjudication gap is CLOSED by
this audit — protocol resumes (all code changes via ARCHITECT_HANDOFF,
red-team findings adjudicated before implementation now that the
Architect is back); (2) the 45M-69 propose-first deviation (interim
resume digests landed unproposed in `5947a05`) is noted, self-corrected
by the audit into queue 1b, no rework — the interim is honest and
strictly better than nothing; (3) the earlier (h)-then-43 resequencing
flag is CLOSED — both landed green the same night, no interaction
materialized.

### Queue 3 DONE: staging seeded-order truth (Opus, 2026-07-13)

The pinned queue-3 contract and every rider are satisfied in one landing.

Code (`emulator/data_staging.py::stage_source`): the resident branch now
returns `np.searchsorted(rows, idx)` — the local coordinates that walk the
sorted compact copy in the run's seeded selection order — instead of a bare
`np.arange(rows.size)`. The disk branch already returned the global selection
order, so both branches now present the selected rows in one canonical order;
under the training loop's shared epoch permutation (`perm = idx_src[randperm]`)
the same seed trains the same cosmology at the same step in either storage
regime. The loader's own `slots()` remap (`searchsorted(used_rows, rows)` in
`batching.py`) made the disk branch correct under every loader regime already;
the defect was solely the resident branch's arange, so the fix is confined to
`stage_source`. A duplicate selection row now raises loudly (the selection is a
unique permutation prefix by construction — census: the only two callers,
`load_source` and the scalar loader, both pass `phys[:keep]` from one
`torch.randperm` draw). The banner prints all three named terms
(`params + dv + idx = total`), the comparison operator that held, the budget,
and the branch; the strict `<` boundary is documented as a deliberate policy
(exact fill streams from disk to keep working headroom). The module docstring's
row-coordinate story and the `[9,2,9,5]` example were rewritten around a unique
`[9,2,5]` selection with corrected coordinate definitions (storage vs loader).

Gate (`gates/checks/stage_ram.py`, 21 legs, rebuilt): the byte-accounting legs
are preserved; new legs are the duplicate refusal + unique control, the
exact-fit boundary (need below / equal / above budget via a pinned integer
allowance), the honest three-term banner (parsed and summed, operator vs
branch), and the canonical-order proof — it drives the REAL per-source loader
`_build_loaders_one` in both storage regimes (resident gather from an ndarray,
disk stream from a real `.npy` memmap) with identity geometry/loss stand-ins,
draws one shared epoch permutation, and asserts the executed parameters,
targets, `dump_rows` alignment, and bs=2 minibatch membership and order match
row for row against a selection-order anchor, with a mutation arm restoring
`arange` that must (and does) diverge. No sorting or dedup inside any
assertion. Texnotes: `texnotes/emulator_code_guide.tex` §"Worked staging
example" and §"Worked memory calculation" updated from the cross-regime
divergence + pending-banner limitations to the landed behavior, in this same
landing. Home-note spec (`data-generation-and-cuts.md#srm-a-stage-ram`) and the
board `maps` / gate docstring rewritten to the broadened claim.

Verification (Mac, cocoa-torch interpreter): `stage-ram` 21/21 ALL PASS;
`compileall emulator gates` clean; `run_board --list` rc 0 (evidence anchors
validate); `board-selftest` ALL PASS. The full 40-gate board run stays
workstation-owed (queue 5). NOTE for unit 50 (epoch chunking, queue tail): it
must preserve this canonical order under chunking and can reuse these
invariance legs.

Queue items 2 (evidence rollout), the 1b manifest PROPOSAL, 5 (workstation
evidence), and the rest of 6 (README + didactic campaign) remain.

## RT-2026-07-13-01 adjudication (Fable, 2026-07-13)

The queue-3 landing (`2c26c34`) is AUDITED PASS first: every pinned
clause and every deep-pass rider is honored — searchsorted local
coordinates in seeded selection order, the in-code duplicate refusal,
the three-term banner naming the comparison that actually held, the
below/equal/above budget legs, the texnotes passage and module example
corrected in the SAME commit — and the rebuilt stage-ram gate reran
ALL PASS under the Architect's own hands from a clean checkout of the
merged main (order-recovery leg + the arange mutation arm included).

### RT-IMPL-02 CONFIRMED — queue 1c REOPENED (1c-bis)

The red team's claim is right, with a sharper mechanism than stated:
the preflight pathspec is fine and `_dirty_lines` DOES carry the
exclusion — but `_git` returns `proc.stdout.strip()`, so the FIRST
porcelain line loses its leading status column and `line[3:]`
misparses that line only. `gates/board_config.json` therefore escapes
the exclusion exactly when it is the only (or alphabetically first)
dirty entry — the documented local-deploy-override use case. Proved
live in a clean worktree of merged main: a config-only edit returned
`['M gates/board_config.json']` as an offender (the false red), while
a config + board.py edit correctly excluded the config and reported
only board.py (proving the filter works when the config is not the
head line). The failure direction is false-red only; no false-green
path exists (a garbled first-line path can never EQUAL the exclusion).

Contract (1c-bis):

- Per-line parsing immune to the transport: either `_dirty_lines`
  parses each line without relying on column alignment a global strip
  can break, or the porcelain consumer stops using the stripped
  transport (`_git`'s strip stays for its other callers).
- ONE owner for the executed watch: the pathspec
  (`_EXECUTABLE_DIRS` + root drivers) and the exclusion path live
  together, and the displayed surface text (the [ok]/[FAIL] lines and
  the log-header note) derives from that owner, never restated by
  hand.
- Legs, driving the REAL preflight helpers: a config-ONLY edit stays
  clean (the head-line case this reopen proves); a config + neighbor
  edit reds ONLY the neighbor; a neighbor-only edit reds; the valid
  clean-tree control; a mutation arm restoring the head-line misparse
  must fail.

Small and board-side; may ride the next board-harness commit, ahead
of the evidence rollout.

### BLOAT-01 CONFIRMED — queue-4 rider

`_is_finite_real` is defined twice (training.py + experiment.py, both
born in the queue-4 commit). One owner: `emulator/training.py` — the
import direction is already pinned by the in-repo constraint that
training must not import experiment, and experiment.py already
imports from `.training`. The experiment copy is deleted, its call
sites import the one definition, the explanatory docstring is kept,
and the existing finite-contract Part J + cmb-identity schema legs
rerun green unchanged.

### BLOAT-04 CONFIRMED — queue-6 rider (binding completion evidence)

The generator modules gain real module docstrings defining, in plain
language at first use: MPI rank, worker, memmap, checkpoint, append,
and sidecar. The Cobaya adapters gain module docstrings defining:
construction, requirement negotiation, sample calculation, getter
readback, and what Cobaya's state means to an adapter. Completion
evidence for the whole queue-6 campaign is now BINDING as: an
untruncated zero-hit scan over the Architect's full audit-pattern
family (45M, unit N, increment, Architect/Implementer, red team,
ruling, adjudicat-, D-*, POL-* — the 108-line baseline), against a
REVIEWED identifier allowlist recorded in the gate output.

### New units from this batch

BLOAT-02 -> UNIT 64 (generator storage consolidation, proposal-first):
spec in data-generation-and-cuts.md "UNIT 64". BLOAT-03 -> UNIT 65
(adapter mechanics consolidation): spec in families-background-mps.md
"UNIT 65". Sequencing: 64 AFTER the ingress cluster (same files); 65
lands with the typed adapter contract the wave-4 adapter visits
establish. Staging truth stays ahead of the evidence rollout
(landed); 45M-89 retains exactly one in-code verdict (ratified).

### Queue 1b: Architect constraints for the manifest proposal (2026-07-13)

Queue 3 is closed (landing `2c26c34` AUDITED PASS). The 1b proposal is
GO, and the Implementer's three open questions are pre-answered so the
proposal can converge in one review round. Order of work: the 1c-bis +
BLOAT-01 rider commit first (small, already contracted above), the 1b
proposal note in the same session (design only), the rollout (queue 2)
strictly after 1b review — both edit the same runner files, and the
rollout must build on the final resume machinery, not race it.

1. Declared, then checked — never walked into existence. The manifest
   is a DECLARED set (inspectable, note-auditable). Reconciliation is
   a STATIC repo-local import scan (AST-level over the executable
   surface dirs), chosen because it must run on the Mac without
   importing the modules — a cosmolike-importing check script cannot
   be imported here, but its import lines can be read. A repo-local
   import the scan finds missing from the declaration is a validation
   error (exit 2, the evidence-map pattern). Third-party drift (torch,
   numpy, cobaya) stays environment territory — preflight and the
   queue-5 capability lanes — never a per-gate digest member.
2. A NEW Gate field. evidence= is the note-anchor trust chain; needs=
   is capability lanes; overloading either muddles two validators.
   The new field defaults to None = the conservative fallback below.
3. Composition with the rollout: the aid declared-vs-executed
   reconciliation (queue 2) and the import declared-vs-found
   reconciliation (1b) SHARE the reporting shape (one "declared vs
   observed" error format a reader learns once); whether they share
   code is the proposal's choice.
4. Persisted inspectable membership: at PASS time the status record
   stores the manifest as RESOLVED members, each with its own digest
   (resolved paths, materialized facts — never re-derivable
   declarations), so --list names WHICH member went stale, not just
   that something did.
5. The shared harness surface (run_board.py, board.py, every invoked
   check script) is hashed for every gate — the audit's "a digest with
   no inspectable membership is not evidence" requirement stands.
6. Conservative fallback + honest legacy: an undeclared gate keeps
   today's dual-digest behavior AND is displayed manifest-less; on 1b
   landing, legacy PASS records become a non-green "pre-manifest"
   resume state. The too-narrow digest surface is exactly the false
   currency the audit proved, and the queue-5 full board rerun is
   already owed, so the honest staleing costs nothing (consistent
   with the digestless-is-STALE ruling of the unit-4 extension).
7. Input side: a declared gate names its SPECIFIC files (its YAMLs
   and data / covmat / axis / artifact inputs), resolved against
   board_config at run time; the whole-yaml_dir hash retires for
   declared gates and survives only inside the fallback.

Texnotes PDF: a tracked build product beside its source is stale by
construction (it is stale now, third time this cycle). Preferred
repair, red-team's call in their build pipeline: stop tracking the
PDF and build on demand (pdflatex exists on the dev Mac), or add a
freshness check (PDF vs .tex digest) to the board's doc lane. Until
one lands, the rebuild is owed whenever the .tex changes.

### 1c-bis + BLOAT-01 DONE (Opus, 2026-07-13)

1c-bis (the first-porcelain-line strip misparse): `_git` gains a
`strip` parameter (default True, so every single-value caller -- commit
hash, ancestor check -- is unchanged); preflight (b) reads the watch
with `strip=False`, so `_dirty_lines` receives every porcelain line with
its two-column status prefix intact and `line[3:]` is uniform. One owner
for the executed watch: `_WATCH_EXCLUDE = "gates/board_config.json"`
lives beside `_EXECUTABLE_DIRS`; `_watched_paths()` owns the pathspec;
`_dirty_lines` excludes `_WATCH_EXCLUDE`; and the preflight `[ok]` surface
text prints that same constant -- pathspec, exclusion, and surface text
can no longer drift. No production behavior changes for a clean tree; the
fix only stops the config from false-red-ing when it is the head entry.

Legs (`board_selftest.check_dirty_watch`, 7, driving the REAL
`_dirty_lines` / `_git` / `_watched_paths`): config-only head-line stays
clean; config + neighbor reds only the neighbor; neighbor-only reds; the
empty-porcelain clean control; a mutation arm that restores the global
strip (feeds the stripped head line) and must -- and does -- false-red the
config; `_git(strip=False)` proven to preserve the raw transport while
`strip=True` trims; and the one-owner pathspec/exclusion coverage. Live
proof on a real dirty tree (pathspec pinned to the config so it is the
head line): `strip=False` excludes it (clean), the retired global strip
leaks `M gates/board_config.json` (false red); the config's bytes were
restored and the tree re-verified clean.

BLOAT-01 (queue-4 dedup): `emulator/experiment.py`'s `_is_finite_real`
copy is deleted and imported from the `emulator/training.py` owner (the
import direction the in-repo constraint already pins; experiment already
imports from `.training`). Verified one object
(`experiment._is_finite_real is training._is_finite_real`), the predicate
still rejects bool / numeric-string / non-finite, and `cmb-identity`
(which exercises `validate_cmb` -> `_is_finite_real`) reran all green.
finite-contract Part J (optimizer schema) rides the workstation, unchanged
by a pure move.

Verification (Mac, cocoa-torch): `compileall emulator gates` clean;
`board-selftest` ALL PASS (now with the clean-tree-watch legs);
`run_board --list` rc 0; `cmb-identity` all green; the live head-line
proof PASS. Next: the queue-1b manifest PROPOSAL (design only), under the
seven pre-answered constraints above.

## Queue 1b: executable/input manifest PROPOSAL (Opus, 2026-07-13, DESIGN ONLY)

For Architect review. No code lands from this section; it names the
`Gate` field schema, the static-scan mechanics, the persisted-manifest
JSON shape, and the pre-manifest legacy transition, and stops there.
Numbering follows the seven pre-answered constraints above.

### The problem this closes (recap)

`_gate_code_digest` hashes only the gate body plus the `gates/checks/*.py`
paths named literally in it; it misses `board.py` shared helpers, the
`run_board.py` runner, and the production modules a check imports. A change
to a shared helper or an imported library module can leave a stored PASS
falsely current. `_gate_input_digest` hashes the WHOLE `yaml_dir`, so an
unrelated YAML edit falsely stales every gate, and it never establishes
which specific data / covmat / axis / artifact files a gate consumes. The
audit's rule: "a digest with no inspectable membership is not evidence."

### (A) Gate-field schema

One NEW optional `Gate` field (constraint 2 -- not folded into `evidence=`,
the note-anchor trust chain, nor `needs=`, the capability lanes):

```
@dataclass(frozen=True)
class Manifest:
  # repo-relative production-module ROOTS this gate's checks depend on,
  # BEYOND the always-hashed shared harness and the check scripts the gate
  # body names (those are added automatically). Each must live inside the
  # executable surface (_EXECUTABLE_DIRS).
  code:   Tuple[str, ...] = ()
  # board_config KEYS (dotted) whose resolved values are the specific
  # external files this gate consumes: its own smoke YAML(s) and any data /
  # covmat / axis / artifact inputs. Resolved against board_config at run
  # time, never stored as raw deploy paths.
  inputs: Tuple[str, ...] = ()

# on Gate:
manifest: Optional[Manifest] = None   # None = the conservative fallback (D)
```

Example (illustrative, not a build list): `stage-ram` declares
`code=("emulator/data_staging.py", "emulator/batching.py")`,
`inputs=()` (it fabricates its own arrays). A data-driven gate such as
`cmb-smoke` declares its generator / covariance modules in `code` and its
YAML + dump + covmat config keys in `inputs`.

The always-hashed shared harness (constraint 5) -- `run_board.py`,
`board.py`, and every check script the gate invokes -- is added by the
digester for EVERY gate, declared or not; a gate never has to (and cannot
usefully) re-declare it.

### (B) Static-scan reconciliation mechanics (constraint 1)

The manifest is DECLARED, then CHECKED -- never walked into existence.
A new `validate_manifests(BOARD)` runs on every invocation (including
`--list`), the same shape as `validate_evidence`, and exits 2 on a
mismatch.

The check is a STATIC, repo-local, AST-level import scan -- it must run on
the Mac without importing the modules, because a cosmolike-importing check
script cannot be imported here, but its `import` lines CAN be read
(`ast.parse` on the file bytes, never `import`/`exec`):

1. Seed = the gate's check scripts (auto-discovered from the gate body, as
   today) + its declared `code` roots + the shared harness.
2. For each seed file, `ast.parse` and collect `import X` / `from X import`
   targets. Resolve each to a repo-relative path ONLY if it lands inside
   `_EXECUTABLE_DIRS`. Third-party targets (torch, numpy, cobaya,
   cosmolike_*) are IGNORED -- environment drift stays preflight + the
   queue-5 capability lanes, never a per-gate digest member.
3. Transitively close over the repo-local hits to a fixpoint. The closure
   is the "found" set.
4. RECONCILE. Under the approved terse ruling (direct roots + derived
   closure) the closure is derived from the declared roots, so ordinary
   repo-local imports are covered by construction -- a plain "found vs
   declared" check is vacuous for them (the deriver already swallowed every
   import it could see, and it declared nothing separately to compare
   against). The under-declaration that still bites is the dependency the
   scan is blind to, and it is caught by two censuses instead:

   (a) a literal repo-relative `.py`-path census over the gate body and its
       check scripts -- the subprocess targets: the `run_driver` driver
       names, the `_DRIVER` constant, and the sweep-driver constants. Every
       such literal `.py` path must be covered by a declared root or an
       auto-discovered check script; an uncovered one is a validation error
       (exit 2). A covered driver then becomes a seed, so the closure walk
       swallows its imports the same as any other seed.
   (b) a dynamic-import census -- every `importlib` / `__import__` site
       inside the derived closure is flagged. The gate must either declare
       roots covering the dynamically-reachable modules (a rebuild-shaped
       gate declares `emulator/designs/`, the directory the model-recipe
       strings resolve into) or the site must appear in a reviewed waiver
       table kept in this note. This is where "declared-then-checked" stays
       meaningful: exactly where the scan is blind, the human declaration
       (or a reviewed waiver) is what the census reconciles against.

   The digest then hashes the full derived closure, so a change to any
   transitively-reachable repo-local module -- including a driver pulled in
   by census (a) -- staled the gate.

Blind spots (what the scan cannot see, and the mitigation). A static scan
reads `import` / `from ... import` nodes; it cannot resolve a module whose
name is a runtime value. It is blind to:

- Dynamic imports: `__import__(name)` and `importlib.import_module(name)`.
  The target is a string or variable computed at runtime, so no repo-local
  path can be recovered from the source alone.
- importlib string / computed names: any import whose module name is built
  from a string or a variable rather than written as a literal statement.
- Function-local (and conditional) imports: an `import` inside a function
  body or an `if` branch is missed if the scan reads only module-level
  statements. Walking the WHOLE tree (`ast.walk`) recovers the literal ones
  (a function-local `from emulator.x import y` is a visible node); only the
  runtime-named forms above stay invisible even then. The proposal
  recommends the whole-tree walk so the blind spot shrinks to the dynamic
  forms alone.

These forms are not hypothetical; the live in-repo instances (verified to
exist) are the reason the scan is built the way it is:

- The function-local `import importlib` at `emulator/results.py:546` is the
  concrete motivator for walking the whole tree (`ast.walk`) rather than only
  the module level -- it sits inside a function body, so a module-level-only
  scan would miss it. Walking the whole tree sees it as an ordinary literal
  import node.
- The string-target dynamic imports at `emulator/results.py:602` and `:672`,
  and `emulator/warmstart.py:368` and `:410`, are the "model-recipe" pattern:
  `getattr(importlib.import_module(mod), qual)` resolving a design / loss
  class from a string that a saved artifact recorded. The module name is a
  runtime value read out of the recipe, so the scan cannot resolve it to a
  repo-local path -- these are the true blind spots that survive the
  whole-tree walk.

There is a second blind spot the source-level import scan cannot cross at
all: the subprocess boundary. A run-shaped gate does not `import` its work; it
launches a driver in a child process through `ctx.run_driver` (the `_DRIVER`
constant `"cosmic_shear_train_emulator.py"`, plus the sweep and legacy driver
constants). That driver `.py` file, and everything it imports, appears in no
check script's import graph -- the check script only spawns it -- so an import
scan is structurally blind to the whole driver subtree. Its mitigation is not
another import walk (there is nothing to walk from) but delta 2's literal-path
census: the driver names written as literal `.py` strings in the gate body are
enumerated and required to be covered, which makes each driver a scan seed in
its own right. See the reconciliation below.

Because the reconciliation cannot flag a dependency it cannot see, a gate
whose checks reach a repo-local module only through one of these forms must
either declare that module explicitly in `manifest.code` or (for the driver
subtree) have the driver enumerated by the literal-path census so it becomes a
seed; that resolution rests on the declaration and the Architect's review, not
on the import scan alone. The manifest never claims to have found every
dependency automatically -- it claims to hash every declared-or-found
repo-local module and to error on any found import the declaration leaves out.
A dynamic-import dependency is the one lane where a silent under-declaration is
possible; the honest cost is named here rather than papered over
(per the Architect review, d4d2136, and the crossing note, 416f821).

Ruling (approved): direct roots + derived closure. `code` names only the
direct roots; the digester derives the transitive closure from them. This was
chosen over a full-closure declaration (which would be verbose and rot on
every refactor) because the note stays readable and the scan is the single
source of the closure. Never-trust-defaults is still satisfied: the closure is
a derived value, but the persisted manifest (section C) materializes that
resolved closure with per-member digests, so the artifact is the whole truth
while the declaration is only intent / config. The two censuses above are how
under-declaration is still caught once transitivity is delegated to the
deriver -- the reconciliation bites exactly at the driver subtree and the
dynamic-import sites the closure walk cannot reach on its own.

### (C) Persisted-manifest JSON shape (constraint 4)

At PASS time the status record stores the manifest as RESOLVED members --
materialized paths and their own digests, never re-derivable declarations
-- so `--list` / `BOARD.md` can name WHICH member went stale, not just
that something did:

```
"stage-ram": {
  "status": "PASS",
  "code_digest":  "<sha256 over the ordered code members>",
  "input_digest": "<sha256 over the ordered input members>",
  "manifest": {
    "code": [
      {"path": "gates/run_board.py",          "sha256": "..."},
      {"path": "gates/board.py",              "sha256": "..."},
      {"path": "gates/checks/stage_ram.py",   "sha256": "..."},
      {"path": "emulator/data_staging.py",    "sha256": "..."},
      {"path": "emulator/batching.py",        "sha256": "..."}
    ],
    "inputs": []
  },
  "log": "stage-ram.log", "log_digest": "..."
}
```

The overall `code_digest` / `input_digest` stay as the fast resume
comparison; the per-member list is the inspectable evidence behind them.
`_resume_state` gains a member-level compare: when an overall digest
differs, it walks the persisted members against the freshly-resolved ones
and reports the first changed `path` (a `stale-code` / `stale-input` with a
named member, not an opaque flip).

Determinism. The members are sorted by their repo-relative path before the
overall digest is taken, so the same set of members always produces the same
digest regardless of the order the closure walk discovered them. The fixpoint
closure result is likewise independent of traversal order -- it is the set of
files reachable from the seeds, and a set has no order. A `(path, digest)`
parse cache is an allowed optimization (it avoids re-hashing a file two seeds
both reach), but the cache is never itself persisted as evidence: only the
resolved, sorted member list is written to the status record, so nothing about
the run's traversal or caching can leak into the stored truth.

### (D) Pre-manifest legacy transition (constraint 6)

- An undeclared gate (`manifest is None`) keeps today's `_gate_code_digest`
  / `_gate_input_digest` (whole-`yaml_dir`) behavior AND is displayed
  manifest-less. The too-narrow surface is exactly the false currency the
  audit proved, so this fallback is honest-but-weak, never the goal.
- On 1b landing, every existing PASS record predates the manifest and has
  no persisted `manifest` block. A DECLARED gate whose stored record lacks
  that block reads a NEW non-green `pre-manifest` resume state (beside
  `stale-code` / `stale-log`), forcing a rerun -- the same digestless-is-
  stale ruling as the unit-4 extension. The queue-5 full-board rerun is
  already owed, so the honest staleing costs nothing.

### Input side (constraint 7)

A declared gate's `inputs` name its SPECIFIC files, resolved against
board_config at run time (its smoke YAML by its `gate_configs` key, its
data / covmat / axis / artifact inputs by their keys). The whole-`yaml_dir`
hash RETIRES for declared gates and survives only inside the (D) fallback.

### Shared reporting shape (constraint 3)

The 1b import "declared vs found" reconciliation and the queue-2 aid
"declared vs executed" reconciliation emit ONE "declared vs observed" error
line a reader learns once, e.g.
`[manifest] <gate>: declared {..} but the import scan found {..} (undeclared: X)`
alongside the rollout's
`[evidence] <gate>: declared {aids} but executed {aids}`.
Whether they share a helper is the queue-2 implementation's choice; the
format is fixed here.

### Sequencing

1b lands before the queue-2 evidence rollout (both edit the runner's resume
machinery; the rollout must build on the final manifest, not race it).
Suggested build order once approved: the `Manifest` dataclass + `Gate`
field + `validate_manifests` (fixpoint scan) first, gated by
`board-selftest` arms driving the real scan on a fabricated under-declared
gate (the mutation arm); then the digest + persisted-member rewrite with
the `pre-manifest` state; then per-gate `manifest=` population, gate by
gate, each a rerun. Awaiting Architect review before any of it.

### Architect review of the 1b proposal — APPROVED WITH DELTAS (Fable, 2026-07-13)

The proposal satisfies all seven constraints as written: the Manifest
dataclass and optional Gate field, the AST fixpoint scan validated on
every invocation, the persisted resolved members with member-level
stale naming, the always-hashed shared harness, the pre-manifest
transition, the yaml_dir retirement, and the fixed "declared vs
observed" error format. The suggested build order is approved
(dataclass + validate_manifests with the under-declaration mutation
arm first; then the digest/persist rewrite with the pre-manifest
state; then per-gate population, each a rerun).

RULING on the flagged decision: DIRECT ROOTS + DERIVED CLOSURE (the
Implementer's recommendation). The declaration states intent and
stays readable; a full-closure declaration rots on every refactor and
invites blanket copy-paste. Never-trust-defaults is satisfied because
the PERSISTED manifest materializes the resolved closure with
per-member digests — the artifact is the whole truth, the declaration
is config. The closure deriver itself lives in the always-hashed
shared harness, so a deriver bug stales every gate honestly.

Three DELTAS, all required before implementation:

1. BLIND-SPOTS PARAGRAPH (required by the review handoff; see the
   crossing note below for its status). The scan walks the ENTIRE AST (ast.walk), so
   function-local import statements are seen — results.py:546 is a
   live function-local "import importlib". What the scan CANNOT see,
   documented with the live instances: string-target dynamic imports
   (getattr(importlib.import_module(mod), qual) at results.py
   :602/:672 and warmstart.py :368/:410 — the model-recipe pattern
   resolving design classes from saved artifact strings) and
   subprocess-invoked files (ctx.run_driver / the _DRIVER constants —
   a run-shaped gate's driver and everything it imports never appear
   in any check script's import graph).
2. RECONCILIATION REDEFINED under the terse ruling. With a derived
   closure, ordinary repo-local imports are covered by construction,
   so "found vs declared" is vacuous for them. The under-declaration
   check that still bites, and is REQUIRED: (a) a literal
   repo-relative .py path census over the gate body + its check
   scripts (the subprocess targets: run_driver names, _DRIVER and the
   sweep-driver constants) — every hit must be covered by declared
   roots or auto-discovered scripts, and a covered driver is then a
   SEED so the closure swallows its imports; (b) a dynamic-import
   census — every importlib / __import__ site inside the derived
   closure is flagged, and the gate either declares roots covering
   the dynamically-reachable modules (rebuild-shaped gates declare
   emulator/designs/) or the site appears in a REVIEWED waiver table
   in this note. Declared-then-checked stays meaningful exactly where
   the scan is blind.
3. DETERMINISM: members sorted by repo-relative path before the
   overall digest; the fixpoint result independent of traversal
   order; a (path, digest) parse cache is fine but is never itself
   persisted as evidence.

With the deltas incorporated into the proposal text (a short edit,
not a re-review — cite this section), implementation may begin in the
approved order. Queue 2 remains blocked until 1b lands.

### Crossing note + delta-1 status correction (Fable, 2026-07-13)

Relay crossing, recorded for the honest history: the Implementer wrote
a blind-spots passage into the proposal section CONCURRENTLY with the
Architect's review, as an uncommitted edit in this shared worktree.
The Architect's review commit (`d4d2136`) staged the whole notes file
and therefore CARRIED that passage — one commit, two authors' hunks.
The review text's original claim that the passage was "absent from the
proposal" was true of the reviewed commit (`cd6ac7a` contains none of
it) but is false of the file as committed; the delta-1 opener above is
corrected accordingly.

Delta-1 status after crediting the Implementer's passage: the
dynamic-import half is DONE and good (whole-tree ast.walk shrinking
the blind spot to runtime-named forms; the honest
declaration-plus-review mitigation; "the one lane where a silent
under-declaration is possible" named rather than papered over). Still
REQUIRED to close delta 1: (a) the live in-repo instances named in
the passage (results.py :602 / :672 and warmstart.py :368 / :410, the
model-recipe pattern; the function-local import at results.py:546 as
the ast.walk motivator), and (b) the SUBPROCESS boundary — ctx.run_driver
and the _DRIVER constants appear nowhere in the proposal section, and
a run-shaped gate's driver is invisible to ANY import scan; its
mitigation is delta 2's literal-path census, which the blind-spots
passage should point at. Deltas 2 and 3 stand unchanged.

Workflow rule from this crossing (shared-worktree hygiene): before
committing a notes file, `git diff` it and confirm only your own
hunks are present; if the other agent's uncommitted hunks are in the
file, either hold the commit or record the shared authorship in the
message. A whole-file `git add` in a shared worktree commits whatever
is there, not what you wrote.

## RT-2026-07-13-02..06 adjudication (Fable, 2026-07-13)

All five findings independently verified by the Architect (code
chains read end to end + live probes on the cocoa interpreter; the
red team's curved-distance numbers reproduce EXACTLY under the sinh
mapping at H0 = 70.0, Omega_k = 0.1). Verdicts and placement:

- RT-02 CONFIRMED -> NEW UNIT 66 (public return-value ownership),
  spec in artifacts-inference-warmstart.md. On CPU,
  .detach().cpu().numpy() on the persistent axes shares storage
  (inference.py:520, :529-530); Architect probe: a caller edit of the
  returned array changed the tensor [0,1] -> [99,1]; the SAME code on
  MPS copies, so public ownership semantics are device-dependent
  today. Persistent-state views also leave through diagnostic
  dictionaries (diagnostics.py :542, :585, :669, :765, :790, :901,
  :917 — geometry sigma/ell/scale/z/k).
- RT-03 CONFIRMED -> NEW queue 1d (board child environment
  identity), contract below. The whole mechanism is in code:
  rootdir() resolves config-override-else-$ROOTDIR
  (run_board.py:406-420); sh() copies os.environ verbatim (:255) and
  nothing injects ROOTDIR; run_check adds only PYTHONPATH
  (:301-307); emulator/cocoa.py and the generators read $ROOTDIR
  directly; the :526 comment states the assumption ("rootdir equals
  $ROOTDIR") that nothing enforces. With a board_config rootdir
  override B and shell $ROOTDIR=A, the board certifies B while every
  child executes against A.
- RT-04 CONFIRMED -> rider on the 1d landing. rc_w is captured at
  board.py:705 and never consumed (single occurrence in the file);
  the warning expectation (:712-717) is substring-only, so a
  print-the-warning-then-crash run passes the leg.
- RT-05 CONFIRMED (CRITICAL — wrong science) -> NEW UNIT 67, spec in
  families-background-mps.md. The flat-only refusal keys on emulator
  input names only (emul_baosn.py:161, "omk" in req) — a global
  Cobaya omk that is not an emulator input bypasses it; the producer
  stores chi as D_M with the flat assumption in a comment only
  (dataset_generator_background.py:343-347); the untruncated omk
  grep over compute_data_vectors shows the ONLY curvature
  enforcement in the tree is compute_cmb_covariance's
  LCDM_FIXED_ONLY — the background generator has none.
- RT-06 CONFIRMED -> NEW UNIT 68, spec in
  data-generation-and-cuts.md. Architect probe on the real Cobaya: a
  parameter declared {prior: {min: 0, max: 1}} yields
  model.info()['params'] WITHOUT a 'latex' key, and
  generator_core.py:808 indexes it unconditionally — after sampling
  finished and after the chain .txt was written. RULING on the fix:
  the parameter NAME becomes the display label when latex is absent
  (GetDist's own convention); presentation metadata is NOT promoted
  to a required key and is NOT a refusal surface.

### Queue 1d contract (RT-03 + the RT-04 rider): the executed child environment equals the certified root

- ONE owner: sh() injects ROOTDIR = str(effective resolved rootdir)
  into child_env for EVERY child. The landing includes the census
  proving all child launches (drivers, check scripts, Cobaya
  subprocesses, golden runs) route through that owner. If rootdir is
  unresolved, the refusal fires BEFORE any child launch.
- The injected value is recorded in the per-run log header alongside
  the existing metadata — the recorded value IS the executed value,
  never a restatement from another variable.
- Legs (board_selftest, driving the REAL sh / run_driver /
  run_check): inherited shell $ROOTDIR=A + board rootdir B -> the
  child observes exactly B on driver, check, and Cobaya paths;
  $ROOTDIR absent entirely -> the child still observes B; board
  rootdir unresolved -> refusal before launch; a mutation arm
  restoring the uninjected inherit-only environment must FAIL.
- RT-04 rider, same landing: gate_gha_f's warning leg becomes
  (rc_w == 0) AND the warning substring, with a board_selftest
  fake-ctx warning-then-nonzero-rc arm that must fail the leg; the
  invalid-license leg's required rc_l != 0 stays the separate
  negative control.
- Sequencing: after the in-flight 1b phases (same files), BEFORE
  queue 2 and queue 5 — the workstation certification run must
  execute with a trustworthy environment. Queue 5 now DEPENDS on 1d.

Sequencing after this batch (amended, binding): 1b (in flight) ->
1d(+RT-04) -> 2 -> 66 -> 5 -> rest of 6 -> 50 -> 52 -> 55 ->
22(+20) -> 13(+01). Unit 67 rides the wave-4 background visit
(15 + 58 + 62 + 67); unit 68 rides the generator-ingress cluster;
64 and 65 unchanged. Guide-custody note: any Current-gap paragraphs
these five defects deserve, and their later closure, are RED-TEAM
edits only (conventions-and-workflow.md custody rule); the landings
NAME affected paragraphs and stop there.

## 1b phase 1 — post-merge Architect audit (Fable, 2026-07-13): PASS, phase-2 GO with three binding riders

Commit `5e4fded` (on main via `907064f`) audited post-merge (it reached
main before this audit — queue-4/6a precedent, gap closed here).
Evidence, re-executed by the Architect from a CLEAN detached checkout
of merged main `060a150` on the cocoa interpreter (the shared worktree
already carries the Implementer's uncommitted phase-2 edit, so it was
NOT used): board-selftest 55 legs ALL PASS (rc 0) including the nine
manifest legs; `run_board --list` rc 0.

Contract conformance, verified in the diff: frozen roots-only
`Manifest(code, inputs)` with the intent-vs-truth story;
`Gate.manifest=None` conservative fallback, rolling migration;
`validate_manifests(BOARD)` on every invocation before `--list`,
exit 2; the closure deriver walks the WHOLE AST (the function-local
import motivator is covered), resolves repo-local only (third-party
= preflight territory), and is a set-based fixpoint (delta 3); both
censuses bite exactly where the import scan is blind — the
literal-path census adds `_DRIVER` and `driver=` literals, a covered
driver joins the closure seeds; the dynamic-import census walks the
DERIVED closure against the reviewed `_DYNAMIC_IMPORT_WAIVERS`
(results.py + warmstart.py, the delta-1 live instances); the selftest
mutation arms catch an uncovered subprocess target AND an unwaived
dynamic-import site (the pinned rider, honored).

Architect adversarial probes (run live against the real
`validate_manifests` on clean main; both are validation HOLES, not
false evidence — no live gate declares a manifest yet):

- P1/P2: a BARE DIRECTORY declared as the covering root
  (`emulator/designs`) VALIDATES (the cover check accepts `r == c`)
  while `_derive_closure({"emulator/designs"})` returns the
  unparseable directory string as its only member — the designs tree
  never enters the closure, so a phase-2 digest built on it would
  hash nothing real for the dynamically-loaded modules.
- P3: a typo'd extra root (`emulator/desings/typo.py`) validates
  silently — it seeds nothing, covers nothing, and errors nothing;
  a misspelled declaration is exactly the lie the manifest exists to
  prevent.

BINDING PHASE-2 RIDERS (before any phase-3 population):

1. Root schema totality: every declared code root must exist as a
   repo `.py` file or a directory; anything else is a validation
   error (kills P3).
2. Directory-root expansion: a directory root expands recursively to
   its `.py` members as closure seeds and digest members; a root
   resolving to ZERO members is a validation error (makes the cover
   check's directory acceptance correct, kills P1/P2).
3. Input-side validation: phase 2 validates every `inputs=` dotted
   key resolves against board_config (phase 1 validates nothing on
   the input side).

Awareness note (not a defect; direction is false-red): the
literal-path census reads the whole gate source including docstrings,
so a `.py` path named in a gate docstring becomes a required
declaration at population time — reword or declare when phase 3 hits
one. The determinism leg is a same-process smoke; the structural
guarantee is the set-based fixpoint, which the diff shows.

Phase 2 is GO under these riders; its notes entry names guide
~:4723's digest half as the narrowed Current-gap (red-team custody).

## 1b phase 2 DONE (Opus, 2026-07-13): digest consumes the closure, members persisted, pre-manifest state, three riders honored

A declared gate now digests its RESOLVED code manifest -- the derived
transitive repo-local closure of its roots + check scripts + shared harness,
member by member -- instead of the legacy gate-body-plus-literal-checks
digest; a manifest-less gate keeps that legacy digest as the conservative
fallback. The input digest branches the same way: a declared gate hashes only
the specific files its `inputs=` keys resolve to (the whole-`yaml_dir` hash
retires for it), a manifest-less gate keeps the broad hash. At PASS time the
status record persists the manifest as sorted resolved members (each code
member `{path, sha256}`, each input member `{key, path, sha256}`), so `--list`
/ `BOARD.md` name WHICH member staled (`_stale_member`), not just that the
digest moved. `_resume_state` gains the non-green `pre-manifest` state: a gate
that now declares a manifest but whose stored PASS predates it (no `manifest`
block) reruns -- digestless-is-stale, the unit-4 extension. The run loop and
the board display consume the new state.

The three BINDING phase-1-audit riders are honored and gated:

- r1 root schema totality: a declared code root must be an existing repo `.py`
  file or a directory, else a validation error (kills P3 -- a misspelled root
  can no longer pass while seeding and hashing nothing).
- r2 directory-root expansion: `_expand_root` expands a directory root
  recursively to its `.py` members as closure seeds AND digest members; a
  directory that expands to zero `.py` files is a validation error (makes the
  dynamic-import cover check's `r == c` directory acceptance correct -- kills
  P1/P2, the bare-directory hole; declaring `emulator/designs` really pulls the
  design tree into the closure).
- r3 input-side validation: `validate_manifests(gates, cfg)` now takes cfg and
  errors on any `inputs=` dotted key that does not resolve against
  board_config.

Selftest: `board-selftest` 55 -> 67 legs. New: `check_manifest_persistence`
(6 -- sorted resolved members, digest binds membership, input-key resolution,
pre-manifest, stale-code-with-named-member, undeclared-untouched) and
`check_manifest_riders` (6 -- r1 non-existent root reds, r2 dir expands + dir
covers-and-enters-closure + empty-dir reds, r3 unresolvable key reds + resolving
key clears). The phase-1 reconciliation fixtures were reworked to real repo
modules (a non-existent fixture root now correctly reds under r1) and pass cfg.

Guide currency (red-team custody, I name -- do not edit): phase 2 closes the
DIGEST half of the Current-gap at `texnotes/emulator_code_guide.tex` ~:4723
("the gate-code digest covers the gate function and check scripts named
literally inside it, not the complete transitive imported production surface
... a change only in an imported adapter or generator can still require an
explicit force-rerun") -- for a DECLARED gate, whose digest now covers that
transitive surface. The dirty-tree-watch half of the same paragraph was closed
by 1c/1c-bis. The paragraph narrows to: the fallback (manifest-less) gates
still carry the legacy narrow digest until phase-3 population; route the guide
update accordingly.

Verification (Mac, cocoa-torch): `board-selftest` ALL PASS (67 legs, 0 fail);
`run_board --list` rc 0; `compileall emulator gates` clean; manifest-less real
gates unchanged (legacy digest path). Full 40-gate board run stays
workstation-owed (queue 5).

Remaining: phase 3 (per-gate `manifest=` population, each a rerun, gate by
gate); queue 2 (evidence rollout) stays blocked until 1b fully lands.

### 1b phase 2 — post-merge Architect audit (Fable, 2026-07-13): PASS; phase-3 flow approved

Commit `24ed07a` (on main via `6f3f54f`) audited post-merge. Evidence,
re-executed by the Architect from a clean detached checkout of merged
main on the cocoa interpreter: board-selftest 67 legs ALL PASS (rc 0),
`run_board --list` rc 0.

Contract-true on the phase-2 spec AND all three phase-1-audit riders:

- The digest rewrite: a declared gate's code digest is the digest of
  its RESOLVED sorted members ({path, sha256} over the derived
  closure); its input digest hashes only the specific files its
  inputs= keys resolve to (the whole-yaml_dir hash retires for it);
  a manifest-less gate keeps both legacy digests. One structural
  improvement beyond the spec, VERIFIED: `_manifest_seeds` is the ONE
  owner feeding validation and digest, so they can never disagree
  about a gate's dependency set.
- Riders: r1 typo'd root reds (Architect probe re-run: reds); r2
  directory roots expand recursively (probe: emulator/designs
  contributes 5 members to the closure) and a zero-member directory
  reds (probe with a genuinely empty in-repo dir: reds — the
  Architect's first "texnotes should red" probe was a FALSE ALARM,
  texnotes/make_figures.py legitimately expands); r3 an unresolvable
  inputs key reds and a resolving one clears (probe: both
  directions). r3's semantics are deliberately resolve-not-exist:
  the key must navigate board_config to a string, but the file may
  legitimately be absent on the Mac (workstation deploy data) — the
  None-sha member records the absence honestly instead of breaking
  cross-machine --list. Verified and RIGHT.
- Pre-manifest: the REAL `_resume_state` returns "pre-manifest" for a
  declared gate whose stored PASS carries no manifest block
  (Architect probe: confirmed), the run loop reruns it, and the
  BOARD.md log cell carries the state — one `_resume_state`, every
  consumer (the 1a lesson held).
- Member-level naming: `_stale_member` names the first changed
  member (probe: a perturbed stored sha256 named
  "code:emulator/activations.py"); surfaced in the BOARD.md detail.
- Determinism (delta 3): two independent resolutions byte-identical,
  members sorted by path (probe: confirmed).

Minor notes, no rework: `_stale_member` names only members present in
the STORED block — a newly-ADDED member (a new import) yields a
generic stale-code with no member named; best-effort naming, the
digest verdict is still correct. The persisted manifest is computed
at launch, consistent with the verdict-bound digests + the clean-tree
preflight.

Phase-3 flow (RULING): propose-first, approved as the Implementer
suggested — the population ORDER plus the FIRST gate's declared
roots/inputs come to the Architect as a notes proposal before any
population lands; each populated gate is a rerun. PLACEMENT AMENDED
for queue 1d: population may begin on Mac-runnable gates once the
proposal is approved, but 1d (child-environment identity, with the
RT-04 rider) must land BEFORE the first WORKSTATION rerun — a
populated gate re-certified on the box must execute under an injected,
recorded ROOTDIR, not an inherited ambiguous one. Queue 2 stays
blocked until 1b fully lands (population complete). Guide gap
handling verified: the phase-2 notes entry NAMES ~:4723's digest half
(narrowed), no guide edit — custody honored.

## Queue 1d + RT-04 DONE (Opus, 2026-07-13): the child environment equals the certified root

`RunContext.sh` is now the ONE owner of the child environment: every child a
gate launches (driver via `run_driver`, check script via `run_check`, the
golden-run `git worktree` ops, and the Cobaya subprocess a driver spawns as a
grandchild that inherits this) starts from the current environment with
`ROOTDIR` forced to `str(self.cfg["rootdir"])` -- the board's resolved,
certified root -- never the inherited shell value. If that rootdir is
unresolved, `sh()` raises `GateFailure` BEFORE launching anything (defence in
depth beside the preflight rootdir check). The per-run log header records
`child ROOTDIR (injected into every child)` from the SAME `cfg["rootdir"]`
value `sh()` injects, so the recorded root IS the executed root, not a
restatement. Census: exactly one `subprocess.Popen` in `run_board.py` (inside
`sh()`); `_git` / `_probe_import` use `subprocess.run` and are harness-internal
(not gate children), so every gate child routes through the one owner.

RT-04 rider (same landing): `gate_gha_f`'s flag-vs-pin warning leg is now
`ok=(rc_w == 0 and <warning substring>)` -- a warning printed on a driver run
that then exited nonzero no longer passes. The invalid-license leg's
`rc_l != 0` stays the separate negative control.

Selftest: 67 -> 74 legs. `check_child_env` (5, driving the REAL sh over a child
that echoes its ROOTDIR): inherited A + board B -> child sees B; $ROOTDIR
absent -> child still sees B; a mutation arm (inherit-only env) makes the child
see A not B; an unresolved rootdir refuses before launch; the one-owner Popen
census. `check_gha_f_warning` (2, driving the REAL `gate_gha_f` with a fake ctx,
the golden/smoke helpers stubbed): a warning on a FAILED flag run fails the leg;
a warning on a SUCCESSFUL flag run passes (control).

Verification (Mac, cocoa-torch): `board-selftest` ALL PASS (74 legs, 0 fail);
`run_board --list` rc 0; `cmb-identity` green (the regression control the 1d
gate required); `compileall emulator gates` clean. Queue 5 now DEPENDS on 1d
(the workstation certification run executes under the injected, recorded root).

Guide custody: I checked the guide for a Current-gap paragraph describing the
harness child-environment inheritance; the ROOTDIR discussion at
`emulator_code_guide.tex` ~:351 is descriptive setup (what ROOTDIR is), not a
labeled gap about the board injecting it, and none of the labeled Current-gap
paragraphs covers the gates-harness env. So 1d narrows no guide Current-gap;
none named. (If the red team judges one is owed, it is their edit.)

## Queue 1b phase 3: per-gate manifest population -- PROPOSAL / analysis (Opus, 2026-07-13)

Phase 3 populates each gate's `manifest=`; its VALUE (a populated gate's
`pre-manifest`/stale record reruns with the closure digest) is realized only by
the workstation reruns (queue 5, which now waits on 1d). It is also
design-sensitive per gate. So this section records the analysis + a population
plan for review rather than a piecemeal population -- a code-only manifest with
`inputs=()` on a gate that has data inputs would RETIRE its whole-yaml_dir hash
and hash no data, a regression worse than the fallback. Population is
all-or-nothing per gate (code AND inputs together).

Gate categories (from the closure/dynamic/driver scan over BOARD):

- Pure-Python board-integrity checks, no external inputs, dynamic-clean:
  board-selftest, stage-ram, generator-seed, family-first. These declare
  `Manifest(code=(), inputs=())` -- the check auto-seed already closes over
  their surface, and the closure digest is strictly broader than the legacy
  gate-body digest. FULLY Mac-validatable now; the natural first increment.
- Clean-closure checks that consume a config/data (eval-batch-invariance,
  diagnostics-domain, weight-decay-census, triangle-shading): declare
  `code=()` (+ any driver they run) and the SPECIFIC input keys -- needs the
  board_config key per gate.
- Dynamic-cover gates whose closure reaches the model-recipe (results.py /
  warmstart.py): the family identity + smoke gates, save-rebuild-drift,
  cobaya-adapter, artifact-readback, finetune/transfer-identity, geo-paths,
  finite-contract. These MUST declare `emulator/designs` (+ `emulator/losses`)
  to satisfy the dynamic-import census, plus their data/covmat/axis/artifact
  input keys.
- Driver-running gates with no check (ema-*, *-smoke, param-window-cuts,
  joint-training, npce-training, head-activation-pin, relu-tanh-norm, ...):
  the driver is launched through a shared `_smoke_leg`/`_golden_leg`/
  `_smoke_driver` helper, so the literal-path census (over the gate body) does
  not see it -- these declare the driver root explicitly (`_DRIVER`, or the
  tune/sweep driver the gate's helper passes) + `emulator/designs`+`losses`
  (the driver's closure reaches the model-recipe) + input keys. Identifying the
  exact driver per gate is the per-gate design call.

TWO BLOCKERS found in the scan, needing an Architect decision before those
gates populate:

- `cli-strict` and `geo-paths` have an `importlib`/`__import__` site in their
  own CHECK SCRIPT (`gates/checks/cli_strict.py`, `gates/checks/geo_paths.py`),
  not in production. The dynamic-import census would red them as unwaived. A
  check-script dynamic import is a test-harness construct, not a production
  model-recipe; options: (a) waive these two check-script sites in
  `_DYNAMIC_IMPORT_WAIVERS` with a covering-root convention, or (b) exempt
  `gates/checks/` sites from the census (a check script is already a hashed
  seed; its dynamic import loads a test double, not a science module). Rec: (b),
  a narrow "the census applies to the production closure, not the check-script
  seeds themselves" carve-out -- but this is a contract call for the Architect.

Recommended order: land the first increment (the four pure board-integrity
gates, `Manifest(code=(), inputs=())`, Mac-validated) to prove the wiring
end-to-end; then the dynamic-cover + driver gates in family batches, each
needing its input keys (board_config) and, for driver gates, its driver -- best
done with the board_config in hand and reruns on the box. Population is
intertwined with queue 5 (each populated gate reruns there). Awaiting the
Architect's call on the cli-strict/geo-paths census carve-out and the
input-key/driver population approach before the non-trivial batches.

### Full population order + the first gate's complete declaration (for review)

Population order (each batch a set of small design calls; each populated gate
reruns on the box, so batches land as the workstation cycles through them):

1. Pure board-integrity, `Manifest(code=(), inputs=())` (no roots, no external
   inputs): board-selftest, stage-ram, generator-seed, family-first. Mac-fully
   validatable; the wiring-proof first increment.
2. Clean-closure checks with a config YAML (no dynamic sites, no model-recipe):
   eval-batch-invariance, diagnostics-domain, weight-decay-census,
   triangle-shading -- `code=()` (+ the driver root if the check runs one) and
   `inputs=(gate_configs.<key>,)`.
3. Dynamic-cover, no driver (the closure reaches the model-recipe): finite-
   contract, artifact-readback, finetune-identity, transfer-identity,
   scalar-identity, cmb-identity, bsn-identity, mps-identity, save-rebuild-drift,
   cobaya-adapter -- `code=("emulator/designs", "emulator/losses")` + their
   board_config-resolvable inputs.
4. Driver gates (a helper launches the driver, invisible to the literal
   census): ema-* / *-smoke / param-window-cuts / joint-training /
   npce-training / head-activation-pin / relu-tanh-norm -- `code=(<the driver
   .py the gate's helper runs>, "emulator/designs", "emulator/losses")` +
   `inputs=(gate_configs.<key>, ...)`.
5. AFTER the census carve-out ruling: cli-strict, geo-paths (their check
   scripts carry the flagged dynamic import).

The FIRST gate (board-selftest), complete declaration:

    manifest=Manifest(code=(), inputs=())

  - code is empty: board-selftest names its own check script
    (gates/checks/board_selftest.py) in its gate body, so that check is an
    automatic closure seed; its transitive repo-local closure (run_board.py +
    board.py, the shared harness) is dynamic-clean and needs no declared root.
    The resulting digest is strictly broader than the legacy gate-body digest
    (it now covers run_board.py / board.py changes), which is the whole point.
  - inputs is empty: board-selftest is pure Python and reads no data, covmat,
    YAML, or artifact -- there is no board_config-resolvable file to name.

A representative NON-TRIVIAL exemplar (save-rebuild-drift), validated by
validate_manifests (30-member closure, 0 errors):

    manifest=Manifest(code=("emulator/designs", "emulator/losses"), inputs=())

  - emulator/designs is a root because the rebuild path resolves a saved
    artifact's design class from its stored string via results.py's model-recipe
    (getattr(importlib.import_module(mod), qual)); that dynamic import is
    invisible to the static scan, so the designs tree is declared to cover it
    (the dynamic-import census requires it) AND to hash every design class the
    rebuild can instantiate.
  - emulator/losses is a root for the same reason on the loss side: warmstart.py
    and results.py rebuild the loss class from its stored string, so the losses
    tree is declared to cover that dynamic import and hash the loss classes.
  - inputs is empty: save-rebuild-drift FORGES its emulator artifacts in-process
    and rebuilds them; it consumes no board_config-named external file (there is
    no save-rebuild-drift config key).

Input-side limitation to note at population: board_config exposes the smoke
YAML (gate_configs.<key>), evaluate_yaml, and saved_emulator_root as resolvable
keys, but a gate's raw data / covmat / axis dumps are named INSIDE its YAML, not
as board_config keys -- so the input manifest hashes the YAML (catching a config
edit) but not the dv file the YAML points at. The dirty-tree preflight does not
watch the data trees either. If per-gate data-file currency is wanted, that is a
board_config extension (explicit data keys per gate) -- flagged here as a
decision, not silently dropped.

Awaiting the Architect's approval of this order + the first-gate declaration,
and the cli-strict/geo-paths census carve-out, before landing the increments.

### Queue 1d + RT-04 — pre-merge Architect audit (Fable, 2026-07-13): PASS; phase-3 proposal APPROVED with two rulings

Commit `28d4207` audited BEFORE merge (rhythm restored). Evidence,
re-executed by the Architect from a clean detached checkout of the
branch tip on the cocoa interpreter: board-selftest 74 legs ALL PASS
(rc 0) including the five child-environment legs (fake roots A and B:
injected B wins over inherited A; B holds with $ROOTDIR absent; the
inherit-only mutation arm shows the child seeing A; unresolved root
refuses BEFORE launch; the one-owner census is itself a leg — exactly
one subprocess.Popen, inside sh()) and the two RT-04 legs
(warning-on-failed-run fails; warning-on-clean-run passes);
`run_board --list` rc 0.

Contract conformance: sh() is the one owner (child_env built from
os.environ with ROOTDIR forced to cfg["rootdir"]); the log header
records the same cfg value the injection reads (recorded = executed);
gate_gha_f's warning leg now requires rc_w == 0 AND the warning, with
the honest detail naming rc_w. The out-of-owner subprocess calls
(_git at :598 and the import-capability probe at :609) query the repo
and the interpreter, not the certified root — legitimately outside
the injection. MINOR (no rework, pin at the next harness touch):
sh() layers the caller's env OVER the injected ROOTDIR, so a future
caller passing env={"ROOTDIR": ...} could override the owner; no
current caller does and the census leg pins the routing — add a
refusal for a caller-supplied ROOTDIR when the file is next edited.

Phase-3 population proposal (`5d1b065`): APPROVED — the batch order
(wiring-proof empty manifests -> clean-closure + inputs ->
dynamic-cover -> driver gates -> the flagged pair), the board-selftest
first declaration (empty/empty, with the correct strictly-broader
argument), and the save-rebuild-drift exemplar (designs + losses roots
covering the model-recipe dynamic imports) are all contract-true. Two
RULINGS close its open questions:

1. Batch 5 is NOT a carve-out: cli-strict / geo-paths go through the
   designed lane — their check scripts' dynamic-import sites enter
   _DYNAMIC_IMPORT_WAIVERS as reviewed entries naming the covering
   roots (which those gates then declare). A site whose targets are
   genuinely unbounded at run time is reworked to static imports
   instead. No third category exists.
2. The flagged input-side limitation (data files named inside a
   gate's YAML, not as board_config keys) is ACCEPTED for batches
   1-3 and CLOSED at batch 4: each driver-gate population increment
   adds explicit board_config data keys for the dv / covariance /
   axis files its YAML names, so the gates whose science consumes
   deploy data carry them as input members. The YAML member alone is
   the fingerprint of intent, not of the data bytes.

Sequencing unchanged: this landing carries 1d, so the
1d-before-workstation-rerun dependency is satisfied at the merge;
population batches follow the approved order; queue 2 after
population completes.

## 20M-01/02 adjudication (Fable, 2026-07-13): both CONFIRMED — units 69 + 70

Both verified by the Architect against the code and the installed
Cobaya. 20M-01: the adapter declares nonlinear=False defaults
(emul_mps.py:411-412, :426-428) while the installed Cobaya base
signatures default nonlinear=True on BOTH getters (live
inspect.signature probe on cobaya 3.6.2) — an omitted-argument call
silently serves the linear spectrum where the protocol promises the
nonlinear one; the MPS gates always spell the argument, so no leg
exercises the public default. 20M-02: cmb_residual_diagnostic uses
x_enc for encode/decode but drops it at the chi2 call
(diagnostics.py:537), and CmbFactoredChi2.chi2 falls back to the
mutable training stash (losses/cmb.py:486, stashed at :529) — the
validation rows are scored with the LAST TRAINING BATCH's amplitude
factors; the red team's two-row analytic example ([3,3] correct vs
[12,0.75] shipped) checks out arithmetically, and a batch-length
mismatch is a shape crash instead. The cmb-smoke mean-predictor bar
(cmb_smoke.py:394) calls chi2 the same parameterless way, and
diagnostics-domain builds only law="none" so it cannot see any of
this. Placement: 20M-01 -> NEW UNIT 69 (families-background-mps.md);
20M-02 -> NEW UNIT 70 (families-scalar-cmb.md). Both contracts ratify
the red team's required clauses and legs; the requested gates are
board-listed (small-tensor CPU legs, Mac-validatable; nothing needs
CUDA). Sequencing: 69 + 70 land together as one small increment IN
PARALLEL with the phase-3 population batches (disjoint files), before
queue 2 — 69 is EMUL2-critical (a real Cobaya consumer gets wrong
science by default today), 70 corrupts worst-row selection and
overlays in the shipped CMB example.

## 20M-03..06 adjudication (Fable, 2026-07-13): all four CONFIRMED — the real-consumer protocol cluster, units 71-74; EMUL2 acceptance now formally blocked on it

All four verified by the Architect (code reads + probes on the
installed cobaya 3.6.2). The sharpest probe result: BoltzmannBase's
get_Cl signature is (ell_factor=False, units='FIRASmuK2') — so even a
DEFAULT-argument get_Cl() call fails against emul_cmb's muK2-only
refusal (:271-282); the adapter advertises generic "Cl" (:192-193)
and the README promises "serves get_Cl to any cobaya likelihood"
(:1961). emul_scalars.calculate writes state[name] = value for every
artifact-defined output (:222) with no string/reserved-name check in
validate_scalar (:648-651). emul_mps defines NO must_provide (class
:193; only get_requirements :330) — it inherits the accept-everything
base. syren_base holds the hidden mnu=0.06 default (:94, documented
"legacy fixed") and the adapter never requests a global mnu, so the
red team's fixed-mnu 0.12 substitution serving an unchanged spectrum
(with 6.76% real base sensitivity) is mechanically exact.

RULING (20M-03, the requested choice): OPTION 1 — honor the generic
Cl contract. The method name, the advertised product, the README
promise, and EMUL2's purpose (drop-in serving to the bundled real
likelihoods, two of which are the cited consumers) all commit to the
protocol; implementing the DOCUMENTED conversions from PERSISTED
artifact facts is not silent conversion, it is the contract. The
startup-green/runtime-red middle state is inadmissible either way.

Placement: 20M-03 -> UNIT 71 (families-scalar-cmb.md); 20M-04 ->
UNIT 72 (families-scalar-cmb.md); 20M-05 -> UNIT 73
(families-background-mps.md); 20M-06 -> UNIT 74 (CRITICAL,
families-background-mps.md). One unifying delta across the cluster:
71's temperature/unit convention, 74's fixed cosmology facts, and
67's flat-only fact are ONE persisted "fixed scientific facts" block
in the artifact schema — defined once on the producer side
(coordinates with units 37 + 62), read back and compared by every
adapter; not three ad-hoc mechanisms. The BAOSN one-verdict precedent
(must_provide and getters share one capability helper) is now a
program-wide adapter law and folds into the typed adapter contract
(unit 65's neighborhood).

Sequencing: units 71-74 join the wave-4 adapter visits (CMB/scalar
visits gain 71+72; the MPS visit gains 73+74 beside 16+63+the sigma8
half). EMUL2 ACCEPTANCE IS FORMALLY BLOCKED on 67 + 69 + 71 + 72 +
73 + 74 — no real-Cobaya acceptance claim while a bundled consumer
can pass startup and fail (or silently mis-serve) at evaluation.
Population, 69+70, and queue 2 stay ahead of the cluster as
sequenced.

## 20M-07/08 adjudication (Fable, 2026-07-13): both CONFIRMED — unit 66 AMENDED, new unit 75

20M-07 CONFIRMED in code: emul_cmb.get_Cl returns
self.current_state["Cl"] directly (:283, read this session),
emul_cosmic_shear.get_cosmic_shear returns
self.current_state["cosmic_shear"] (:197), and emul_mps.get_Pk_grid
returns the cached tuple (:420-421) — a destructive first consumer
edits the product every later consumer at the same cosmology reads.
This is exactly the derived-cached-array attack class the Architect
flagged for unit 66, and the red team is right that a .numpy()-token
census cannot see it (these getters leak NumPy objects through
container returns). RULING: UNIT 66 AMENDED (spec updated in
artifacts-inference-warmstart.md) — the ownership surface is every
public exit (predictor, diagnostics, AND Cobaya getters), copies at
the getter boundary, alias-aware census. Sequencing unchanged (after
queue 2, before queue 5); the amendment folds in BEFORE the landing.

20M-08 CONFIRMED in code: emul_mps.initialize accepts any
pklin+boost pair with equal z,k axes, UNIONS both predictors'
names into one requirement mapping (:231-232), and the comment
claiming the axes prove "one generator run" (:267, :279) is a false
implication — the red team's probe composed an LCDM pklin with a
w-carrying identity boost, published w, and served the w=-1 surface
at w=-0.5 (74.5% max deviation from the real base). Distinct from
unit 74 (single-artifact fixed facts): this is CROSS-ARTIFACT domain
identity. RULING: NEW UNIT 75 (families-background-mps.md),
extending the artifact-pair integrity campaign and READING THE SAME
fixed-facts block as units 71/74 — pair equality is proven on the
persisted scientific-domain binding, never manufactured by a union.
EMUL2 acceptance blocklist grows to 67 + 69 + 71 + 72 + 73 + 74 +
75.

### 1b phase-3 batch-1 — pre-merge Architect audit (Fable, 2026-07-13): PASS on Mac scope; family-first re-categorization APPROVED

Commit `bb370cf` audited before merge. Evidence re-executed by the
Architect from a clean detached checkout on the cocoa interpreter:
board-selftest 74 legs ALL PASS (rc 0; includes the pre-manifest and
persisted-member legs), `run_board --list` rc 0 over the now-populated
BOARD (validate_manifests runs on every invocation, so the three
declarations are live-validated). A real single-gate run attempt
confirms the Mac boundary: preflight refuses on cosmolike / CUDA /
the deploy yaml_dir — workstation facts, exactly as designed — so the
populated gates' REAL reruns (pre-manifest firing, verdicts
republished with persisted members) ride the next real board
invocation and are the first thing queue 5 must show. The three
resolved closures match the landing notes and are strictly broader
than the legacy digests (the shared harness is now hashed for all
three; stage-ram additionally hashes data_staging.py + batching.py —
the surface it tests).

The family-first re-categorization is APPROVED: the three probe-found
reasons are real (the docstring literal is the phase-1 awareness case
where DECLARE is right; root-driver imports are outside
_EXECUTABLE_DIRS by design and enter as declared roots; open().read()
dependencies are invisible to every code census — the right lane for
those is the INPUT side, which is exactly what the batch-4
board_config data keys provide). family-first populates in a declared
batch with its driver root and input keys, not with code=(). The
open().read() lane is hereby added to the documented blind-spot list
beside dynamic imports and subprocess targets: code censuses cannot
see file reads; input keys carry them.

## 20M-09 adjudication (Fable, 2026-07-13): CONFIRMED — unit 76, recipe schema totality; the load-side never-trust-defaults violation

CONFIRMED at emulator/results.py:683: `if kwargs.get("head_act") is
not None` conflates an ABSENT head_act with the explicitly persisted
None, omits the keyword, and lets the constructor's Python default
choose the head activation — the red team's end-to-end CPU probe
(parameter-free ReLU trunk + pinned tanh head; recipe field deleted;
strict=True load necessarily blind because parameterless activations
contribute no state-dict keys) changed the prediction from
-1.7615941763 to -1.0 with no warning. The Architect's census adds
the sibling lane: :677's outer `if "block_opts" in kwargs` silently
skips the whole block for a class that requires it — same fallback
class; the signature-derived key census adjudicates every `.get(`
site on the rebuild path (:605, :643-644, :715 included). This is a
direct load-side violation of the house never-trust-defaults rule
(persist resolved values; load with NO code-default fallbacks) and
the artifact gates' schema-v2 claim is currently overbroad.
Placement: NEW UNIT 76 (artifacts-inference-warmstart.md), the
saved-recipe schema-totality side of the artifact campaign with the
pair-integrity item as interlock (a pair digest authenticates bytes;
it cannot prove a recipe complete). EMUL2 blocklist grows to
67 + 69 + 71..76 — a rebuild that silently changes architecture
poisons everything served above it. Acknowledged: the red team
continues into the loss/geometry-algebra and generator
completion/publication seams.

## 20M-10/11/12 adjudication (Fable, 2026-07-13): all three CONFIRMED — units 77, 78, 79

- 20M-10 CONFIRMED (CRITICAL) at the cited lines; the code's own
  docstrings state the convention that produces the defect:
  _unwhiten_templates returns center-free templates ("the
  constant-coefficient GG template carries it"), _composed_physical
  adds the center "once after the combine" — so the factored physical
  GAIN multiplies T0 without its center and the general error is
  -c*r0, with zero gain leverage exactly where GG equals the center
  (the zero-leverage failure the physical space exists to avoid).
  The witness (c=10, GG=12, r=1: current 14 vs advertised 24)
  is arithmetically exact; transfer-identity is green because
  zero-correction parity coincides for both formulas and base_decode
  repeats the same convention (self-comparison blind). -> UNIT 77
  (artifacts-inference-warmstart.md), the transfer campaign; blocks
  the science thread's factored-transfer runs.
- 20M-11 CONFIRMED (CRITICAL): the driver persists rescale as a root
  attribute; grep over results.py shows "rescale" ONLY in comments
  and docstrings — rebuild_emulator has no functional read — while
  warmstart.py:216-217 loudly refuses a non-"none" source; the
  public predictor is the one boundary without the check, and the
  red team's end-to-end save->predict repro (max abs error 28.236,
  all values finite) executes the omission. RULING: refusal-first,
  as recommended — public inference accepts rescale == "none" ONLY,
  refusing "rescaled"/"residual" with prose explaining the
  parameter-dependent inverse transform is not reconstructible from
  the artifact; full analytic-rescale inference is a SEPARATE
  user-gated schema/design unit if ever wanted, never a silent
  blessing of existing files. -> UNIT 78
  (artifacts-inference-warmstart.md), and it RIDES the unit-76
  landing: the rescale fact is one more required native-string read
  in the same recipe-totality validation pass. The note's
  "needs cosmolike" imprecision is corrected per the finding.
- 20M-12 CONFIRMED: pce.py:333 PCEResidualDiagChi2(CmbDiagonalChi2)
  leaks configure_roughness into every NPCE diagonal family, and
  training.py:2733's hasattr test — whose own comment claims it
  identifies "a CMB loss" — accepts them; the alternating
  seven-coordinate witness (objective 7 -> 13.4512 with redshift
  bins smoothed as multipoles) is reproduced through the shipped
  implementation. -> UNIT 79 (training-stack.md): roughness
  eligibility becomes an explicit family capability; only data.cmb
  carries the block; refusal at configuration validation.

EMUL2 blocklist grows by 78: now 67 + 69 + 71..76 + 78. Unit 77
gates the factored-transfer science thread (D-TP9 side) rather than
EMUL2. Unit 79 is a training-truth refusal, small, CPU legs.

## 20M-13 adjudication (Fable, 2026-07-13): CONFIRMED — unit 80, the physical-contraction dtype boundary

CONFIRMED at all four anchors: geometries/output.py documents AND
recommends float64 for stiff 3x2pt directions (:76-79) and stores
basis/Cinv in the geometry dtype (:264); PCERatioChi2 casts the
packed physical truth to float32 (pce.py:266) and contracts the
float32 residual directly against geo.Cinv / Cinv_sq
(:310-311); TransferChi2's physical branch does the same
(transfer.py:291, :294). A float64 geometry therefore crashes both
public physical-composition paths (RuntimeError, reproduced by the
red team through the shipped classes on kept AND full contractions,
sum AND gain) while the whitened route — which passes through the
dtype-aware unwhitening — works, proving the geometry itself valid.
Fail-loud, not silent science, but a documented recommended
configuration is unusable with two public training choices.
-> NEW UNIT 80 (training-stack.md). Sequencing: lands in the
transfer campaign WITH unit 77 — the shared physical-contraction
owner unit 80 creates is exactly where unit 77's one-composition
owner contracts, so the two one-owner clauses land as one algebra
increment (pce.py + transfer.py together). Compatibility note: the
unit-14 chi2 screen already carries compute-dtype provenance; a
float64 contraction yields a float64 chi2 and the band follows the
actual compute dtype — no double work, but the unit-80 legs assert
the returned dtype so the provenance stays honest.

## 20M-14 adjudication (Fable, 2026-07-13): CONFIRMED — unit 81, amplitude-law role distinctness

CONFIRMED at every anchor: validate_cmb checks presence of the four
law keys (experiment.py:806) with no distinctness or native-string
requirement; configure_law does two membership checks and two
independent names.index resolutions (losses/cmb.py:392-393); the
staging repeats the SAME parallel mapping (experiment.py:3899-3900)
— two independent definitions of one resolved mapping, the exact
anti-pattern — so a single column can hold both physical roles
consistently on producer AND consumer sides, and ordinary parity
agrees on the wrong science. The witnesses are exact
(exp(-0.108) = 0.8976275921 for tau_name aliased to As;
3.8889e-8 for as_name aliased to tau). The red team's clause 7
also formally closes the Architect's own deep-pass minor: the
staging banner's "f at fiducial" was computed from the identity
(definitionally 1.0); it becomes an evaluation through the RESOLVED
roles, which detects aliasing at the fiducial. -> NEW UNIT 81
(families-scalar-cmb.md). Not EMUL2-blocklisted (the shipped
configuration is correct; the readback refusal protects serving),
but the artifact-readback refusal clause is binding before any CMB
production training.

## Unit 69 post-merge audit + the 20M-backlog sequencing ruling (Fable, 2026-07-13)

UNIT 69 (`371b0bd`): AUDITED PASS on Mac scope. The Architect reran
gates/checks/mps_identity.py from a clean detached checkout on the
cocoa interpreter: all five unit-69 legs PASS (omitted == explicit
True != False on both getters; the interpolator three-arm at node +
interior; the separated-sentinel catch-power leg; the
branches-differ mutation control; the BoltzmannBase signature-pin
protocol guard), and the diff satisfies clauses 1-4 (defaults
flipped, refusal/serving branches untouched so the explicit paths
are byte-identical, docstrings name the omitted-call behavior, the
board maps= text updated). Leg 5 (the real-Cobaya provider-routed
call on mps-smoke) is workstation-owed as declared and joins the
queue-5 manifest. The check's ONE red leg ("bounded staging:
streamed mean equals the known answer") fails IDENTICALLY at the
parent commit — pre-existing, owned by the already-binding
bounded-grid2d amendment (the confirmed cancellation /
float32-payload finding), not unit 69; recorded here so the next
reader does not re-attribute it.

SEQUENCING RULING (answers the Implementer's two questions; the unit
specs in the notes ARE the handoffs — no separate batched handoff is
coming):

1. UNIT 70 NEXT, immediately (69's pair; contract + legs already in
   hand from the 20M-01/02 adjudication).
2. Then UNIT 79 (small CPU training-truth refusal, independent).
3. Then the 77 + 80 ALGEBRA INCREMENT as one landing — the
   composition owner and the contraction owner are born together
   with both mutation arms; CPU-implementable, its Torch legs are
   board-listed and execute at queue 5. D-TP9 stays blocked until
   this lands.
4. Then the 76 + 78 ARTIFACT-TOTALITY INCREMENT (results.py
   read-side: recipe schema totality + the rescale refusal) — fully
   Mac-runnable, and it clears two EMUL2 blockers WITHOUT needing
   the adapter contract.
5. THEN the FIXED-FACTS BLOCK PROPOSAL (the shared persisted
   scientific-facts schema serving 67 + 71 + 74 + 75) comes to the
   Architect for review — propose-first, it spans the producer
   schema and all five adapters. The adapter cluster (66-amended,
   71-75, 81) implements ONLY after that approval, inside the wave-4
   visits under the typed adapter contract. This is why
   blocklist-first-strictly is WRONG: guess-implementing 71/74/75
   now would build the block three times.
6. Population batches 2-5 INTERLEAVE as filler while blocked on
   Architect reviews: batch-2 Mac-validatable halves may land any
   time; batch 4 waits for its board_config data-key additions;
   reruns accumulate for queue 5.
7. Queue 2 after population completes. Queue 5 (the user's
   workstation session) now carries: the populated gates' real
   reruns, unit 69 leg 5, the 77/80 Torch legs, the 45M-75
   post-step half, Part J / Part 2 timing, and the full board.

## 20M-15/16 + 20M-14 addendum + the tracked-PDF ruling (Fable, 2026-07-13): all adjudicated

- 20M-15 CONFIRMED: __load_chk loads the failure sidecar
  (generator_core.py:619-640) and every loader validates STRUCTURE
  only (existence, row counts, nbytes/rank at :503-521, :540,
  :580-594) — a corrupt payload row under a false success bit loads,
  prints "Loaded models", schedules nothing, and republishes; the red
  team executed the shipped loader body with only unavailable imports
  stubbed. -> BINDING READ-SIDE AMENDMENT TO UNIT 56 (its
  checkpoint-ingress half; spec in data-generation-and-cuts.md):
  every row whose failure bit is false revalidates through THE SAME
  family-specific stored-payload predicate used at publication,
  BEFORE any print / flag / scheduling; an invalid-but-successful row
  makes checkpoint and sidecar mutually inconsistent and REFUSES with
  nonzero status touching neither file. "Valid when written" and
  "valid when resumed" share one predicate so they cannot drift.
- 20M-16 CONFIRMED: the fresh path writes the float64 chain with
  fmt="%.9e" (:798-801) and INDEPENDENTLY casts self.samples to
  float32 (:804, self.dtype at :244), while the append path reloads
  the written chain as float32 before computing (:365-368) — decimal
  and binary rounding do not commute at midpoint-adjacent values, so
  fresh generation computes at a row one float32 ULP away from the
  row it publishes (~700 per million uniform draws). The parameter
  row is the scientific LABEL of the data vector; a payload may not
  belong to the neighboring representable cosmology. -> NEW UNIT 82
  (row authenticity), data-generation-and-cuts.md: one canonical
  published parameter table materialized once before any science
  call; bitwise identity between producer rows, staged rows, and the
  training loader's recovered rows across fresh / resume / append /
  serial / MPI; lnp/chi2 either recomputed at the canonical row or
  explicitly labeled pre-canonical sampling diagnostics; the manifest
  records the canonical dtype/representation.
- 20M-14 ADDENDUM ACCEPTED -> UNIT 81 AMENDED (semantic roles):
  distinctness is necessary, not sufficient — swapped roles
  (as_name="tau", tau_name="As") give the finite factor 3.4907e-8 at
  the fiducial (Architect re-derivation exact: 3.889e-8 x
  exp(-0.108)) with producer/consumer in perfect agreement. The
  amended contract requires explicit persisted column-role semantics
  (a canonical role registry; aliases legal only when registered to
  the scientific role) AND the executable semantic check: f at the
  RECORDED fiducial, through the RESOLVED roles, equals the law's
  identity value within its declared numerical contract — which
  refuses swaps and arbitrary pairs without needing the registry to
  enumerate the world. Added legs adopted (swapped pair; unrelated
  existing pair; registered-alias control; fiducial unity; the
  presence+distinctness-only mutation must fail).
- TRACKED-PDF RULING (red team, RATIFIED — closes the last custody
  item): the PDF stays tracked (the user wants it browsable); the
  documentation lane gains a BUILD-MANIFEST gate -> NEW UNIT 83
  (queue-6 documentation lane): a generated manifest of cryptographic
  digests over every included .tex source, figure-generation script,
  figure/vector asset, texnotes-owned bibliography/style input, and
  the tracked PDF itself; written ONLY after a successful build; any
  edit, regeneration, missing input, unrecorded include, or
  independently replaced PDF turns the lane non-green until rebuild +
  manifest regeneration. NO modification times (checkouts and merges
  make them lies); the input census derives from the manuscript's
  actual include/graphics dependency closure with an explicit
  reviewed list only where TeX discovery cannot be mechanical.
  Custody split at implementation: the red team owns the .tex side;
  the Implementer builds the gate/helper code via handoff.

Also this turn: CLAUDE.md's writer split aligned with the USER RULE
(Architect owns notes/*.md, red team owns texnotes/, the Implementer
writes neither — everything travels in the handoff block); Opus's
retraction commit 0040bc5 was correct under that rule.

## 20M-13 addendum + 20M-17/18 adjudication (Fable, 2026-07-13): unit 80 extended; units 84 + 85 born; two more EMUL2 blockers

- 20M-13 ADDENDUM ACCEPTED -> UNIT 80 AMENDED (training-stack.md):
  the structured heads build W_fd / W_df from geometry-dtype tensors
  with no cast (plain.py:577-594 — the construction is right in the
  docstring — mirrored in ia.py), so a documented/recommended
  float64 output geometry crashes every structured head family
  (ResCNN / ResTRF / Template*) at y @ W_fd before any loss runs.
  The amendment lands IN the unit-80 increment: one end-to-end owner
  of "supported geometry precision" from model head through physical
  contraction — trunk stays the model compute dtype, head basis
  buffers cast at their owned boundary, the loss geometry keeps its
  requested precision, forward/inverse basis transforms get an
  independent known answer, float32 bitwise identical, all four
  head families complete forward+backward under float64, and a
  mutation keeping float64 buffers beside a float32 trunk reds.
- 20M-17 CONFIRMED -> NEW UNIT 84 (physical input-domain identity),
  artifacts-inference-warmstart.md. ParamGeometry.state() persists
  exactly {names, center, evecs, sqrt_ev} — no support fact —
  _as_row builds the tensor uncheckedand all five adapters cross
  that boundary; the red team's tanh witness serves finite,
  well-shaped answers 23.84% / 90% wrong far outside the trained
  box. Distinct from units 20/21/46/58/74 exactly as filed; unit 46
  (NPCE domain policy) becomes the PCE-specific instance of this
  general contract and must consume the same persisted block.
  EMUL2-blocking.
- 20M-18 CONFIRMED -> NEW UNIT 85 (canonical dark-energy resolver),
  families-background-mps.md. syren_params_from documents and
  implements wa = 0.0 when absent (syren_base.py:54, :88) and
  nothing in the adapter requests or derives wa from a sampled
  w0pwa — the generator meanwhile computes through Cobaya's resolved
  input mapping, so the stored correction belongs to wa = 0.5 while
  the served base uses wa = 0 (12.987% miss on the shipped-adapter
  probe). The SAMPLED-coordinate sibling of unit 74's fixed-fact
  lane in the same function. EMUL2-blocking.

Open EMUL2 blocklist after this batch: 67, 71-76, 78, 84, 85
(69 closed by audit; 70 in flight, never blocklisted). The
fixed-facts block proposal (already queued for Architect review)
now ALSO carries 85's persisted dark-energy parameterization/role
identity and 84's domain block placement question — the proposal
must present all the persisted-identity members together.

## 20M-18 addendum + the notes-ownership correction (Fable, 2026-07-13)

20M-18 ADDENDUM ACCEPTED — the defect is publicly reachable with no
invented parameterization: EXAMPLE_EMUL2_EVALUATE.yaml samples
w0pwa (prior [-5, -0.01], dropped), defines the dynamic bridge
wa: 'lambda w0pwa, w: w0pwa - w' with derived: false (:126-128), and
its ONE shipped evaluation point sets w0pwa == w == -0.9 (:220-221)
— wa = 0 exactly, masking the omission. The red team's full-grid
magnitudes on the real vendored base: 1.7774% in P_lin, 3.0239% in
the combined pklin x boost multiplier within z<=2, k<=10 (where the
low-k blend weight is numerically ONE — this is the final served
nonlinear error under a perfect network), 6.8977% within z<=10,
k<=100, and 121.5% over the full shipped grid. UNIT 85 is AMENDED
with the addendum (spec updated in families-background-mps.md):
public-YAML reachability recorded, the spy-gate leg added (the
canonical Syren tuple observed at generation and at serving must be
EQUAL on the public w0pwa YAML at a nonzero-wa point; removing the
wa requirement reproduces the miss), and the ownership refinement —
wa-under-w0pwa is a DYNAMIC per-point fact under unit 74's
consumer-side resolved-facts mechanism, never pinned as fixed;
unit 7 stays the shared alias/coordinate resolver. RECORD
CORRECTION executed: artifacts-inference-warmstart.md's unit-7
passage claimed the adapter hands a "full calculate(**params)
mapping" — false; real Cobaya routing supplies only
required/supported inputs; the sentence is corrected in place with
the addendum cited.

NOTES-OWNERSHIP USER CORRECTION (2026-07-13, supersedes this
morning's stricter reading): the Implementer MAY edit notes/*.md
(resume state appended to the handoff's entry, the original
CLAUDE.md split); the ONLY carve-out is texnotes/ TeX sources
(red-team-owned). CLAUDE.md is restored accordingly; Opus's
retraction 0040bc5 was an over-correction made in good faith — its
deleted resume blocks survive in history and in the Architect's
audit records, and Opus resumes notes writes from now on.

## 20M-18 reachability correction + final proof + 20M-19 adjudication (Fable, 2026-07-13)

20M-18 REACHABILITY: the red team retracted its own shipped-YAML
claim after forward-walking the real Cobaya 3.6.2 routing — with
drop: true, a component whose artifact requires w0pwa fails at model
construction ("Requirement w0pwa of probe is not provided by any
component, nor sampled directly"), so the shipped example is
STARTUP-RED for w0pwa-storing artifacts, not the silent witness. The
silent lane is the valid NON-drop configuration, now proven at the
FULL reachability standard: a complete real Cobaya model (real
emul_mps + vendored Syren + deterministic schema-shaped artifacts +
a real likelihood reading Pk_grid through the provider) initialized,
assigned the theory exactly ['w0pwa','w','As','ns','H0','omegab',
'omegam'], ran logposterior, and served a finite positive spectrum
at the wa = 0 base — 0.1298745470923877 max relative error against
the requested wa = 0.5 cosmology. RULING: UNIT 85 stands, RECORD
CORRECTED (families-background-mps.md "UNIT 85 REACHABILITY
CORRECTION"); the shipped drop YAML's startup incompatibility is a
SEPARATE defect folded into unit 85's scope — the example may not be
claimed as working EMUL2 routing today and is either repaired or
refused with a migration message; the acceptance suite carries BOTH
branches. The Architect's earlier addendum record ("publicly
reachable via the shipped YAML") is superseded by this correction —
appended, not rewritten. The red team's self-correction applied the
standing reachability rule to its own filing (trace from the public
boundary before accepting the internal mechanism); that is the
standard working as designed and is recorded as such.

20M-19 CONFIRMED -> NEW UNIT 86 (families-background-mps.md, beside
unit 63). The untruncated grep is decisive: const_mask exists ONLY
in geometries/grid2d.py — no loss file consumes it — so the pin is
enforced in decode (torch.where(const_mask, center, out)) while
every diagonal metric sums all encoded coordinates: the executed
witness scores chi2 = 0 vs 10000 for two predictions that decode
BIT-IDENTICALLY. Training, validation, best-epoch selection, and
diagnostics optimize and select a DIFFERENT FUNCTION from the one
the program serves, and gradients flow through a coordinate the
public path always discards.

### Unit 70 — pre-merge Architect audit (Fable, 2026-07-13): PASS, one improvement beyond contract

Commit `5d22634` audited before merge. Evidence re-executed by the
Architect from a clean detached checkout on the cocoa interpreter:
cmb-identity 78 legs ALL PASS (rc 0) — the [2, 0.5] control factors,
the analytic params-passing [3.0000, 3.0000], the omitted-params
stale-stash demonstration reproducing [12.0000, 0.7500] exactly, the
stash-invariance leg (byte-identical), and the wrong-length-stash
no-crash leg — and diagnostics-domain green (the law="none"
control). Contract-true on all four clauses: every needs_params
branch in cmb/grid/grid2d residual diagnostics passes the batch's
own x_enc to chi2 (diagnostics.py :537, :760, :897); the cmb-smoke
mean-predictor bar scores with explicit validation params; the
caller rule is uniform across the family. BEYOND CONTRACT, verified
and approved: loss() now clears the stash in a finally block, so the
private-stash rule is structural — an omitting public caller gets
the loud ValueError refusal, never ANY stale factor; the roughness
penalty still reads the stash inside the reduction, before the
clear. The Implementer's resume-state notes entry
(families-scalar-cmb.md) is back under the corrected ownership rule
and is accurate. The cmb-smoke stale-cache mutation arm rides the
workstation smoke run (queue 5), as scoped.

## 20M-18 coupling clause + 45M-81 executed failure (Fable, 2026-07-13)

20M-18 CROSSING RESOLVED: the red team's rejection notice crossed
with the Architect's correction commit — the ledger ALREADY records
the non-drop configuration as the silent lane, the shipped drop YAML
as independently startup-red, both branches in the gate, and the
unit-7 prose correction (2cd7ecf). ONE clause in their notice is new
and is ADOPTED into unit 85 (recorded below in
families-background-mps.md): a repair that merely removes drop: true
is INSUFFICIENT — it converts the loud startup failure into silent
wrong physics unless the dynamic-wa routing (the canonical resolver)
is repaired IN THE SAME UNIT. The example repair and the resolver
are one landing, never two. The magnitude ladder and spy-tuple leg
run on the non-drop branch; the drop branch is the
startup/migration leg.

45M-81 CONVERTED: what was the owed append acceptance proof is now a
CONFIRMED CURRENT FAILURE, re-executed by the Architect
(200/200 appended rows duplicate originals at the public minimum
N = M = 200; fresh+append != one-shot; append row 0 == fresh row 0)
with the mechanism complete in code: self.rng =
default_rng(self.seed) at generator_core.py:270, NO rng state in the
checkpoint save/load path, the same rng.uniform at :748 drawing the
restarted stream on append, and the :792 header implying seed-only
provenance. Binding amendment spec in data-generation-and-cuts.md.
CAUTION recorded: until the fix lands, ANY dataset produced with
--loadchk 1 --append 1 under the same seed silently duplicates its
original rows — no production dataset has been generated by append
yet (production generation is queued), and none may be until this
lands. Unit 82's canonical serialization is orthogonal and cannot
repair it (it would faithfully canonicalize the duplicates); the
queue-5 emcee rider (sampler._random private-attr seeding) joins
this amendment's Gaussian-branch clause.

## 20M-21 adjudication (Fable, 2026-07-13): CONFIRMED — unit 87, the chain writes GetDist's column with the wrong sign

CONFIRMED at every anchor AND by the Architect's own live GetDist
probe: sampler.get_log_prob supplies lnp (:727), uniform mode
fabricates lnp = 1 (:751), chi2 = -2*lnp is computed separately
(:760), and BOTH fresh (:798) and append (:844) publish
[weights, lnp, params..., chi2*] — raw log posterior in the reserved
column GetDist documents and reads as MINUS log posterior. The probe:
two rows with log posteriors [-1, -10] written in the current
convention make loadMCSamples select the logpost = -10 row as best
fit (argmin over loglikes); the minus_logpost convention selects -1.
Likelihood shading, cooling, and every loglikes consumer reverse the
same way. The standard chain file currently tells its standard
reader that the least probable sample is the best one. -> NEW UNIT
87 (data-generation-and-cuts.md), the generator
publication/provenance campaign beside the 45M-81 amendment and
units 68/82.

## 1b phase-3 population batches 2 + 3 + geo-paths DONE (Opus, 2026-07-13): 15 gates declared; batch 4 + cli-strict await two Architect calls

Following the approved order (batch 1 landed at bb370cf, audited PASS), the
autonomous population window added:

- Batch 2 (2674c23) -- clean check-only gates, `Manifest(code=(), inputs=())`:
  eval-batch-invariance, diagnostics-domain, triangle-shading. No driver, no
  golden base, no gate_configs key; the empty manifest gives the rich real
  closure (training/losses/plotting). diagnostics-domain's docstring named
  "losses/core.py" as prose -> reworded to "the losses core module" (the
  literal-path awareness case; the dep is already in the import closure, so
  reword not declare).
- Batch 3 (724bade) -- the 8 dynamic-cover identity gates,
  `Manifest(code=("emulator/designs","emulator/losses"), inputs=())`:
  finite-contract, artifact-readback, finetune-identity, transfer-identity,
  scalar-identity, cmb-identity, bsn-identity, mps-identity. Each closure reaches
  the results.py/warmstart.py model-recipe dynamic import; designs+losses covers
  it and hashes every rebuildable design/loss class. inputs=() per ruling 2
  (these identity gates forge fixtures in-process).

Verified each batch on the Mac (cocoa-torch): validate_manifests(BOARD, cfg)
ok=True / 0 errors (14 declared gates); board-selftest ALL PASS; compileall clean.
Every populated gate reruns to bind its digest (queue 5).

The full empty-manifest probe categorized all remaining gates. TWO Architect
design calls block batches 4 + 5:

1. Batch 4 (driver gates: ema-*, *-smoke, param-window-cuts, joint-training,
   npce-training, head-activation-pin, relu-tanh-norm, weight-decay-census,
   berhu-loss, single-phase-demotion, production-diagnostic, family-first).
   Each runs a driver (cosmic_shear_train_emulator.py etc.) -- some named in the
   gate body (literal census flags them), most launched via a shared
   _golden_leg/_smoke_driver helper (the driver escapes the census, so an empty
   manifest validates but hashes nothing for the driver). The code side is
   `code=(<the driver>, "emulator/designs", "emulator/losses")`. The BLOCKER is
   ruling 2's input side: batch 4 must add explicit board_config data keys for
   the dv/covariance/axis files each YAML names -- a board_config.json schema
   extension (which keys, how threaded through _config_key_value). That schema is
   a design call, proposed-not-landed.
2. Batch 5 geo-paths DONE (this landing); cli-strict deferred. geo-paths'
   gates/checks/geo_paths.py:170 imported RETIRED legacy module names expecting
   ModuleNotFoundError -- a non-existence test ruling 1's binary (waiver-with-
   covering-roots, else static-rework) does not fit: a static import of a
   deleted module cannot be written, and there is no module to hash. Resolved by
   reworking that site to importlib.util.find_spec (returns None for an absent
   module, is neither import_module nor __import__, so the census never sees it
   -- a clean non-import existence probe; the check still reds if a legacy path
   is resurrected). geo-paths then populates as a dynamic-cover gate,
   code=(designs, losses). Flagged for audit: this is my resolution of a
   ruling-1 gap, trivially reverted if the Architect prefers a waiver/carve-out.
   cli-strict is DEFERRED: its cli_strict.py:58 import loads bounded entry-point
   drivers, which genuinely needs a _DYNAMIC_IMPORT_WAIVERS entry -- and that
   table is the reviewed integrity whitelist, so the entry (cover =
   cosmic_shear_train_emulator + scalar_train_emulator; declaration =
   the driven drivers + designs+losses) is proposed-not-landed for Architect
   review.

So batches 1-3 + geo-paths are complete (15 gates, Mac-validated, workstation
reruns owed); batch 4 awaits the board_config data-key schema, cli-strict awaits
the reviewed waiver entry.

## 20M-22/23 adjudication (Fable, 2026-07-13): both CONFIRMED — unit 21 gains the family-sign/PSD amendment; unit 8 gains the run-control state machine

- 20M-22 CONFIRMED: emul_cmb.calculate copies each decoded row into
  the shared Cl dict and publishes it (:244-250) with NO
  spectrum-family validity check between decode and state — the red
  team's real-lifecycle probe (real component, deterministic artifact
  doubles) published finite tt = -10 / ee = -2 / pp = -3, and a
  TT = EE = 1, TE = 2 joint control whose 2x2 temperature/
  polarization covariance has determinant -3: finite, impossible,
  invisible to the queued shape/finite boundary. -> UNIT 21 AMENDED
  (families-scalar-cmb.md): TT/EE/pp physically nonnegative at every
  stored ell (exact-zero policy decided and documented; NEVER clipped
  or absolute-valued), TE stays signed (consistent with unit 56's
  generator-side semantics), the joint PSD bound
  (TE^2 <= TT*EE) within a representation-derived rounding band that
  covers storage arithmetic ONLY, failures name
  spectrum/triplet/multipole/values/bound and leave NO partial
  state["Cl"], and the proof lives in the board-listed cmb-identity
  gate. Distinct from unit 11 (per-artifact transform
  authentication): this is the physical covariance ASSEMBLED from
  independently predicted spectra at the public consumer boundary.
- 20M-23 CONFIRMED: the append/loadchk relation is documented in the
  CLI help itself (generator_core.py:150) and validated NOWHERE —
  the flags are copied independently (:238, :255), __load_chk
  returns False whenever loadchk != 1, and :699 routes
  loadedfromchk == False to the FRESH branch, whose savetxt replaces
  the existing chain: the red team's live reproduction destroyed the
  sentinel dataset under an accepted --append 1 --loadchk 0 command.
  -> UNIT 8 AMENDED (data-generation-and-cuts.md): the run-control
  state machine validates BEFORE any path mutation (append == 1
  requires loadchk == 1; legal append additionally requires the
  manifest-authenticated prior unit 8 already specifies); the
  illegal pair raises a teaching error preserving EVERY byte of the
  existing bundle; the three legal states (0/0 fresh, 1/0 resume,
  1/1 authenticated append) are exhaustive; 45M-81's RNG
  continuation stays an independent requirement of legal append.
  Distinct from the corrupt-resume fall-through clause: here the
  prior bundle is HEALTHY and the user's explicit append intent is
  bypassed by an unrelated accepted flag.

### 1b phase-3 batches 2 + 3 — pre-merge Architect audit (Fable, 2026-07-13): PASS; one proposal correction ratified

Commits `2674c23` (batch 2: eval-batch-invariance, diagnostics-domain,
triangle-shading — Manifest(code=(), inputs=())) and `724bade`
(batch 3: the 8 dynamic-cover identity gates —
Manifest(code=("emulator/designs", "emulator/losses"), inputs=()))
audited before merge. Evidence, re-executed by the Architect from a
clean detached checkout of the branch tip: board-selftest 74 legs
ALL PASS (rc 0); `run_board --list` rc 0 over the now-14-gate
populated board (validate_manifests live-validates every declaration
and the designs+losses waiver coverage on each invocation).

Batch-2 deviation RATIFIED as a correction: the approved proposal
described batch 2 as "clean-closure checks WITH a config YAML
(inputs=(gate_configs.<key>,))" — the Architect's own census through
the REAL registry (inspect.getsource over every populated gate body)
plus untruncated greps over the three check scripts
(diagnostics_domain.py, ge_c_eval_bs.py, gt_b_triangle.py) found
ZERO yaml / require_config / rootdir usage: these gates run pure
synthetic fixtures, so inputs=() is the NARROWER TRUTH and the
proposal's description was wrong about them — the same honest
correction pattern as batch 1's family-first re-categorization.
Batch-3 census: all eight declared gate bodies clean of
config/driver/yaml/rootdir tokens; the designs+losses roots match
the approved exemplar reasoning (the model-recipe dynamic imports).
Fourteen gates now carry manifests; their board reruns (pre-manifest
firing -> persisted members) remain the first queue-5 exhibit,
Mac preflight refusing on workstation facts as designed.

## 20M-24 adjudication (Fable, 2026-07-13): CONFIRMED — unit 88, GPU-pack tokens must precede the allocations they budget

CONFIRMED at every anchor: _lane_main runs setup_fn
(scheduling.py:242) BEFORE the job loop reads work or acquires
tokens (:252-259, released only around job_fn), while
estimate_train_vram_fraction's own docstring charges the resident
data term the setup allocates — _hyper_setup builds the experiment,
stages train AND validation, and builds the geometry
(cosmic_shear_sweep_hyperparam_emulator.py:131-138) outside the
gate. The red team's live instrumentation through the REAL
run_gpu_pool (4 lanes, four-token "exclusive" jobs): maximum
simultaneous setup allocations 4, maximum simultaneous executions 1
— the semaphore serializes the small region while four charged
resident states coexist; a four-token job can OOM during setups
before its exclusive region begins. Distinct from worker
liveness/invalid tokens: every lane terminates normally while the
scheduler violates its own capacity model. -> NEW UNIT 88
(training-stack.md); the multi-GPU sweep surface (the program's
production tuning path) is blocked on it for any --gpu-pack use.

### Batch 4/5 design calls — RULED (Fable, 2026-07-13); plus a mixed-commit record note

The Implementer's batches-2+3 resume block (committed inside
`d852b1a`, removed by their own later edit inside `5d46e8d` — both
mixed-author commits; the content is intact in history and the
diff-staged-hunks hygiene rule is re-affirmed for BOTH sessions)
raised two blocking design calls. Both are now ruled:

1. GEO-PATHS NON-EXISTENCE TEST — option (a) APPROVED, and the
   ruling taxonomy is refined: a non-existence probe expressed as
   importlib.util.find_spec is NOT a dynamic import (it is neither
   import_module nor __import__, imports nothing, and has no module
   to hash), so the census correctly never sees it. The refined
   batch-5 lane set: a dynamic-import site either (i) enters the
   reviewed waiver table with covering roots the gate declares
   (cli-strict: the bounded entry-point drivers), or (ii) is
   reworked to static imports, or (iii) — the refinement — a
   NON-EXISTENCE test is reworked to find_spec, the sanctioned
   non-import probe. No fourth category. The geo-paths gate keeps
   its purpose: a live spec for a retired module name is the
   failure, and the rework must not weaken that (the leg still reds
   when a legacy module returns).
2. BATCH-4 DATA KEYS — the schema is a `gate_data` block parallel
   to `gate_configs`, dotted keys `gate_data.<gate-id>.<label>`,
   values resolved by the EXISTING `_resolve_config_path` ladder
   (absolute -> rootdir-relative -> yaml_dir-relative) with the
   phase-2 resolve-not-exist semantics (a workstation-only file is
   an honest None-sha member on the Mac). A driver gate's increment
   declares `inputs=("gate_configs.<id>", "gate_data.<id>.<label>",
   ...)` for the dv / covariance / axis files its YAML names. Shape,
   paste-ready (board_config.json):

       "gate_data": {
         "ema-smoke": {
           "train_dump": "projects/lsst_y1/gates_board/dump_train.h5",
           "covariance": "projects/lsst_y1/gates_board/cov.txt"
         }
       }

   Labels are per-gate free names; the docs block gains one
   `_comments` line stating that gate_data values are data files a
   declared gate's manifest hashes as input members.

## 20M-25 adjudication (Fable, 2026-07-13): CONFIRMED — unit 89, loss-object state is established completely on every training invocation

CONFIRMED: run_emulator configures roughness only when the resolved
block is non-null (training.py:2732-2741) and the null path clears
NOTHING; CmbDiagonalChi2 keeps _rough/_rough_lam as persistent
instance state (losses/cmb.py:205-223); the hyperparameter driver
deliberately reuses one staged experiment + chi2fn per lane — so an
enabled->disabled sweep's second point optimizes the FIRST point's
penalty (executed: 13.4511995 vs the fresh-object 7.0; silent
inherited penalty 6.4511995, the unit-79 witness arithmetic
surfacing on a new lane). The in-code comment "configured once on
the run's loss object" was written for the single-run world the
per-lane repeated-training path no longer inhabits. Sweep order and
lane assignment can change the scientific comparison. The
Architect's census widens the hazard family: configure_law
(cmb.py:342), configure_rescaling (core.py:668), and transfer's
configure_roughness (transfer.py:754) are further conditional
configure_* setters on loss objects — the unit audits all of them
under one discipline. -> NEW UNIT 89 (training-stack.md), the
loss-object sibling of unit 55's repeated-training isolation class,
distinct from unit 79's family eligibility; blocks production
hyperparameter sweeps over loss blocks.

### Batch 5a (geo-paths) — pre-merge Architect audit (Fable, 2026-07-13): PASS; batches 4 + 5b are GO — population completes on their landing

Commit `fddf05f` audited (it landed just before the find_spec ruling
committed — the crossing resolved correctly, the code matches the
approved option (a) exactly). Evidence, re-executed from a clean
checkout of the branch tip: geo-paths rc 0 with the find_spec probe
live ("all 6 absent from the import system") and the census leg
green; board-selftest 74 legs ALL PASS; `--list` rc 0 over the
15-gate populated board. Declaration truthfulness verified beyond
sufficiency: the Architect derived geo-paths' closure WITHOUT its
declared roots — it reaches BOTH waived files (results.py +
warmstart.py), so the designs+losses roots are REQUIRED by the
dynamic-import census, not gratuitous.

GO (answers the crossed handoff; both rulings were issued in
`39ee40e`): batch 4 (the 12 driver gates) proceeds under the
gate_data schema exactly as ruled — the paste-ready block shape,
dotted gate_data.<gate-id>.<label> keys, the existing resolution
ladder, resolve-not-exist semantics; batch 5b (cli-strict) proceeds
under the reviewed-waiver lane — _DYNAMIC_IMPORT_WAIVERS entries for
cli_strict.py's bounded entry-point driver imports, naming the driver
roots the gate then declares. Their landing COMPLETES the 1b
population; queue 2 (the evidence rollout) unblocks at that merge.
The populated board's real reruns remain the first queue-5 exhibit.

## DIDACTICS-01/02 adjudication (Fable, 2026-07-13): both ACCEPTED into queue 6 — navigation truth first, then the file-by-file teaching campaign

Factual verification (Architect): gates/checks/__init__.py says "the
other six files" while the directory holds 27 (25 substantive check
modules + __init__ + logscan); the README carries the prose count
INSIDE the very sentence warning against prose counts ("never from a
number in prose (40 gates at this writing...)", :102) and admits its
table is knowingly incomplete (:107); the broken phrase "never a
storage upcast (the):" sits at finite_contract.py:58; the Manifest
docstring carries "phase 1"/"later phases" rollout history. The
representative teaching gaps cited (bsn_identity :102-293,
mps_smoke :110-399, finite_contract :17-64 and body) are consistent
with the census counts on spot-check.

RULINGS:

1. DIDACTICS-02 (navigation truth) lands FIRST — it is small,
   factual, and unblocks every new reader: the file-by-file
   __init__ index; the README inventory MECHANICALLY VALIDATED
   against BOARD (validation, not generation — the prose around the
   table stays human-written per the house didactic register, but a
   board-selftest LEG proves exact set equality of the table, the
   __init__ index, and the live files/BOARD, with an explicit
   helper-file allowlist); no numeric gate count anywhere in prose;
   the beginner-first runner walkthrough (BOARD row -> run function
   -> ctx.run/ctx.expect -> raw log -> status record -> BOARD.md)
   BEFORE the evidence/audit vocabulary; the worked miniature gate
   with all six artifacts; the maps/evidence/manifest distinction
   taught on that example. TIMING: the Manifest/Gate docstring
   cleanup happens AFTER batches 4/5b land, so the docstrings
   describe the completed two-regime behavior once, with the queue
   phases and dates moved to notes/ where they belong.
2. DIDACTICS-01 (the teaching campaign) is the large queue-6 item,
   landed FILE-BY-FILE per gate family as the red team's follow-up
   batches isolate them — never one mega-commit. Its whole required
   contract is ratified: the per-file pipeline diagram + glossary;
   the four paragraphs (Production rule / Fixture / Pass condition /
   Catch-power) on every scientific leg owner; test-double honesty
   statements; Python mechanics taught at first use; named constants
   with derivations replacing magic values; mutation arms showing
   the faulty algorithm and the numerical difference; the Cobaya
   lifecycle taught once and referenced; internal audit codes and
   history removed from executable documentation (history lives in
   notes/); tests never shortened to pay for prose.
3. The mechanical acceptance is BINDING per landing: the AST
   docstring census with a reviewed one-liner allowlist; the
   untruncated internal-history scan over gates/**/*.py — this
   MERGES with the existing queue-6 108-line pattern-family
   completion evidence, whose scope now names gates/ explicitly;
   compileall + board-selftest green; and the DOCUMENTATION-ONLY
   AST-EQUIVALENCE rule: the executable AST unchanged except
   docstrings and named documentation constants whose values are
   byte-identical. The human spot-check standard (the second-year
   physics student answering the four questions from the file alone)
   is the campaign's acceptance voice, consistent with
   user-didactics-and-python-voice.md.

Sequencing: doc-only, interleaves with the unit queue as
review-blocked filler; does NOT preempt batches 4/5b, queue 2, or
unit 79. The Implementer reads
notes/user-didactics-and-python-voice.md BEFORE writing any of it.

## DIDACTICS-03..09 adjudication (Fable, 2026-07-13): the per-family teaching batches ratified; two factual corrections jump the queue; one new gate unit

03 (family identity/smoke gates as laboratory exercises), 04 (staging
coordinate map + Python-memory mechanics, with the shared nine-term
glossary), 06 (selftest/harness mechanics box + the four-state worked
example + digest-is-byte-identity-not-correctness), and 07
(finite-contract/diagnostics/eval_bs PyTorch + numerical-analysis
teaching) are RATIFIED AS FILED — they are exactly the per-family
DIDACTICS-01 batches the campaign anticipated, each landing under the
binding mechanical acceptance already ruled (AST docstring census,
untruncated history scan, compileall + selftest, documentation-only
AST equivalence, the reviewed one-liner allowlist). ONE named
exception to the doc-only rule: DIDACTICS-04's tiny coordinate table
must be EXECUTABLE — the table lives in the documentation, and its
mechanical check lands as one new named stage-ram leg reproducing
both storage regimes' identical row/minibatch order; that single leg
is the only executable-AST change the didactics campaign may make,
recorded here so the AST-equivalence census can exempt it by name.

05 (factual falsehoods) CONFIRMED, and SHARPENED by the Architect's
verification: bsn_identity's "closed-form flat LCDM reference"
(:13-14) does not merely share the Simpson ALGORITHM — it imports
the production cumulative_simpson function itself (:50), so the
reference has ZERO catch power against a defect in that function;
it tests grid-resolution behavior only. RULINGS: (a) the honest
narrowing (README all-family CoCoA SONIC wording, no agent/session
history, the BSN comparison described exactly) is a FACTUAL
correction that rides the first queue-6 landing with DIDACTICS-02;
(b) the independence upgrade is accepted as gate work -> NEW UNIT 90
(below). 08 (batching.py teaches the wrong row-coordinate system)
CONFIRMED — the universal global-rows claim (:41-45 and siblings)
is true only for disk-backed staging after the queue-3 landing, and
the "no shared rows, no leakage" implication (:403-406) is not
proven by separate filenames; the two-regime table, the four-concept
glossary, the continued [9,2,5] example, and the
assumption-not-yet-enforced leakage sentence (until the queued
unit-28 disjointness check lands, then naming the real comparison)
are RATIFIED as an immediate factual correction alongside 05a.

09 (team shorthand census: ~149 leg / 89 parity / 157 identity /
56 mutation / 58 pin ...) RATIFIED with the allowlist emphasized:
gate ids and machine identifiers keep their names (cmb-identity
stays cmb-identity; the queue-2 aid scheme keeps "leg" as its
defined term); every retained technical term is defined once in
gates/README.md and locally where sharper; the three plain-language
questions accompany every deliberately broken comparison. The
queue-2 rollout's log vocabulary must use the taught terms.

UNIT 90 (from DIDACTICS-05b): the BSN distance gate gains an
ALGORITHMICALLY INDEPENDENT reference — an adaptive-quadrature
distance (scipy.integrate.quad at tight tolerance, which shares no
code with cumulative_simpson) or an exact analytic form, compared at
a documented tolerance derived from the quadrature's own error
estimate — plus a mutation arm corrupting the Simpson WEIGHTS that
the new leg must catch and the old fine-grid leg provably cannot.
The fine-grid leg is retained under its honest narrowed description.
Board-listed, small, rides the BSN family visit (with units 58/62);
spec home families-background-mps.md via this entry.

Sequencing: 05a + 08 join DIDACTICS-02's first landing (factual
corrections first); 03/04/06/07/09 land per gate family as the red
team's batches isolate them; unit 90 rides the BSN visit. All
doc-only interleave except the two named executable exceptions (the
04 table leg; unit 90); nothing preempts batches 4/5b, queue 2, or
unit 79.

## DIDACTICS-10..19 adjudication (Fable, 2026-07-13): the inference/adapter didactics wave — three lanes, one documentation-examples gate (unit 91)

(DIDACTICS-12 was filed twice; counted once.) Factual verification
by the Architect, all four confirmed: inference.py:4 says "the h5
alone" and :31-33 "all come from the h5" while the trained weights
live in <root>.emul (rebuild requires both); :3-6 frames the
predictor as cosmic-shear while it executes five family branches;
analytics.py promises the input's dtype while its NumPy branch
coerces to float64 (:86-90); syren_base.py:40 says "the six
syren-base arguments" and returns seven; emul_mps.py:4-5 and
emul_baosn.py:4-5 claim the adapters "re-derive no physics" while
they compose P_lin/boost/blends, sigma8, distance integrals, and
flat-universe relations. All ten contracts are RATIFIED, organized
into three lanes so prose is written once against final behavior:

LANE 1 — FACTUAL CORRECTIONS (doc-only, land with the queue-6
factual bundle beside 02/05a/08): 10a (the five-family return table
mechanically matched to every executed predict branch; the two-file
ownership truth — .emul = trained state_dict, .h5 = recipe +
geometries + facts + histories, both required; the common
prefix-then-branch lesson; the no_grad/detach/cpu/numpy definitions);
11a (the TRUE dtype contract: NumPy float32 in -> float64 out, Torch
preserves dtype/device — no same-dtype claim survives); 12 (seven
arguments; the quantity/units table incl. kref = 1e-4 1/Mpc
explained; the positional-call order mapped in comments; the h^3
dimensional cancellation shown); 14 (the ownership tables replacing
"re-derives no physics": EmulatorPredictor owns learned inference;
emul_mps owns the deterministic composition P_lin = e^r Pbase,
B = e^rB Bbase, P_nl = B P_lin + the low-k rule + sigma8; emul_baosn
owns c/H, the distance integral, and the flat relations —
"thin adapter" may describe code volume, never erase owned science).

LANE 2 — UNIT 91, THE DOCUMENTATION-EXAMPLES GATE: the campaign's
acceptance repeatedly demands small EXECUTABLE examples (11's dtype
controls; 12's h != 1 unit-table check; 13's five-point Simpson
walkthrough executed against the hand calculation; 15's instrumented
real lifecycle order; 16's overlapping-requirement example; 17's
three interpolator return-shape cases + the typed range error; 18's
CMB scatter and BAOSN mixed-query fixtures). Rather than scattering
named AST-equivalence exceptions through production files, ALL of
them live in ONE new board-listed check module (the documentation-
examples gate, docs lane, unit-83 neighborhood): production files
stay strictly documentation-only under the stripped-AST census, and
the gate proves every worked example in the prose actually computes
what the prose says. DIDACTICS-04's coordinate-table leg moves under
this gate too (superseding its stage-ram-leg placement — one home
for executable documentation, not two).

LANE 3 — FAMILY DIDACTICS RIDE THEIR PROTOCOL VISITS: the full
adapter teaching (15's four-phase Cobaya lifecycle; 16's
requirement/root/device mechanics with one shared canonical
explanation; 17's PowerSpectrumInterpolator 100% public-method
census; 18's running coordinate examples; 19's cosmic-shear/scalar
outputs + the annotated example YAMLs) lands per family WITH or
AFTER the wave-4 protocol units (71-75, 73's must_provide, 85's
resolver), so the taught call order and state semantics describe the
FINAL behavior, never a to-be-replaced one. 13 (cumulative_simpson
walkthrough + history removal) rides the BSN visit with unit 90
(same family, one visit). 10b (the returned-container ownership
sentences) rides UNIT 66's landing — the prose states what 66
implements, written once. Two deferral rules: 12's
positional-to-keyword conversion is an AST change and rides unit 85
(the same file's resolver landing); nothing in lane 3 starts before
its protocol unit.

The BLOAT-04 queue-6 rider (adapter/generator module docstrings) is
SUBSUMED by 15/16 — one campaign, not two overlapping ones; the
binding completion evidence (the 108-line pattern-family untruncated
scan + reviewed allowlist) now covers emulator/ + cobaya_theory/ +
gates/ didactics landings alike.

## DIDACTICS-20..26 adjudication (Fable, 2026-07-13): registry prose rides queue 2; two activation design calls ruled; the temp-dir leak is a behavioral fix; quantifier discipline joins the acceptance battery

Spot-verification (Architect): board.py:634 "This week's legs." and
:1370's malformed "parity + +" exist; FIVE tempfile.mkdtemp sites
across gsv_bitwise_drift/gct_parity/finetune_identity/
transfer_identity with ZERO rmtree/TemporaryDirectory in gsv — the
leak is real; geo_paths:16-20 still claims "importing ... raises
ModuleNotFoundError" AFTER the approved find_spec rework — stale
prose that survived the Architect's own fddf05f audit (code and legs
were checked; the module docstring's mechanism claim was not —
recorded as an audit-scope lesson: gate audits now read the module
preamble against the executed mechanism); activations.py:27 states
the gamma-to-1 gate limits unconditionally while sigmoid(beta x)
reverses them for beta < 0 — the red team's tail mathematics is
correct.

RULINGS:

- 20 (board.py registry prose): the stable-section rewrite of every
  registry entry RIDES QUEUE 2 — the maps= prose is rewritten into
  the required sections in the same visit that mints the structured
  evidence maps, one pass over that surface, with the
  non-executed-evidence rule aligned to binding ruling 6: a prose
  claim resting on Architect/visual confirmation is labeled
  non-green/UNAVAILABLE until reconciliation executes it. The
  stale/malformed lines (:634, :1370) and their siblings are
  FACTUAL-BUNDLE fixes now.
- 21 (activation tail slopes): the documentation error is CONFIRMED;
  the prose is rewritten to the EXECUTED unconstrained model with
  the sign-conditional index sets and the p != 1 asymptotic form.
  DESIGN CALL: beta is NOT constrained — a reversed or blended gate
  orientation is learnable modeling freedom, not a defect; no
  evidence motivates changing training dynamics. The analytic
  positive/negative/zero-beta tail tests land in UNIT 91.
- 22 (power-activation bounds): the midpoint formula and the p = 0
  division hole are CONFIRMED. DESIGN CALL: option (a) — the
  constructors VALIDATE finite, strictly ordered, zero-excluding
  bounds (refusing non-default bounds would be a capability
  regression; validation matches the config-totality philosophy);
  the general midpoint initialization is documented, with p(0) = 1
  scoped to centered bounds; "no NaN" survives only alongside that
  domain validation. The validation code change RIDES UNIT 40 (the
  power-activation unit, same class family); the constructor tests
  (default, non-centered, reversed, nonfinite, zero-containing) land
  with unit 40's legs and UNIT 91.
- 23 (finetune/transfer preambles): RATIFIED — exact-copy vs
  tolerance comparison separated; delegated evidence labeled
  delegated; the overview mechanically compared with main's calls;
  the stale "refine not yet implemented" is a FACTUAL-BUNDLE fix;
  "runs once" becomes "cached after encode, not rerun by the
  measured path" unless the assertion becomes == 1. The QUANTIFIER
  DISCIPLINE is adopted campaign-wide: bitwise / exactly once /
  strict-loadable / all / every / no-remaining survive in prose only
  with an adjacent assertion mechanically proving the quantifier.
- 24 (structural scans as proofs): RATIFIED — every structural-scan
  label states file surface, pattern, and blind spots; "census" is
  reserved for mechanically complete surfaces; the HDF5-attributes-
  are-typed correction (the real risk is a wrong-typed legacy value
  and Python truthiness on "False") is a FACTUAL-BUNDLE fix; the
  emcee two-RNG handoff teaching coordinates with the 45M-81
  amendment (same surface, one lesson).
- 25 (parity label + temp leak): the metric is named by its actual
  stabilized-relative-error formula with the 1e-8 floor and the
  derived 1e-6 bar. The mkdtemp leak is a BEHAVIORAL gate-hygiene
  fix — TemporaryDirectory / guaranteed finally cleanup at all five
  sites, landing WITH the 25 didactics batch as a small executable
  change (gates test code; the production doc-only rule is not
  violated), with the injected-mid-gate-failure acceptance ratified:
  no unowned directory survives failure or success.
- 26 (lexical vs executed): geo_paths' prose is corrected to the
  find_spec truth ("not discoverable"; no import is re-added — the
  approved rework stands); cli_strict and family_first labels are
  NARROWED to their lexical evidence by default (the live
  boundary-ordering upgrades are board-listed options for the family
  visits, not obligations now).

Routing: factual items -> the queue-6 factual bundle; executable
tests -> unit 91; the bounds validation -> unit 40; the temp-dir
cleanup -> the 25 batch; the registry rewrite -> queue 2; everything
else -> the per-family lane-3 batches under the standing acceptance.

## DIDACTICS-27..32 adjudication (Fable, 2026-07-13): the smoke-fixture disjointness defect is real and blocks queue 5's smoke evidence; five documentation/gate-strength rulings around it

THE HEADLINE — DIDACTICS-31 CONFIRMED (a gate-fixture CORRECTNESS
defect, not didactics): the Architect verified the full seed path —
one --seed 1234 command shape generates BOTH the train and val files
in bsn_smoke (:109), mps_smoke (:126), and cmb_smoke (:181);
split_seed: 0 in all three configs; equal seed + equal inputs
reproduce the same parameter table, and the same split seed selects
the same 180 of 200 rows in the same order. Validation overlap is
100% in all three real-generator smokes: their validation-collapse /
best-epoch numbers measure MEMORIZATION of training cosmologies, not
generalization. The production comment at experiment.py:3255-3256
("the val file differs, so the same seed gives an independent
selection") is FALSE for same-seed files. -> UNIT 28 AMENDED (the
validation-leakage unit gains the smoke-fixture half): distinct
explicit RECORDED generator seeds for train and val (or one table
with a persisted, proven disjoint partition); a zero-overlap
comparison on canonical physical parameter rows BEFORE training;
independent parameter<->data-vector row-alignment proof in both
files; printed generator seeds, split seed, row counts, and overlap
count; the taught distinction (generator seed chooses cosmologies;
split_seed permutes rows INSIDE a file and can never make two
identical files independent); a mutation arm restoring the same
generator seed must fail before training. The false experiment.py
comment is a FACTUAL-BUNDLE fix. PRIORITY: the fixture repair lands
BEFORE queue 5 — the workstation smoke evidence must mean
generalization; until it lands, no prose may call the smoke
validation numbers generalization evidence. (The dead-network bar
still catches total non-learning; the identity gates are untouched.)

The other five:

- 27 (wrapper vs child): RATIFIED — the wrapper-child reconciliation
  (required files, executed subprocesses, exact-vs-tolerant metric,
  check count/names, asserted-vs-logged evidence, owed work) EXTENDS
  the queue-2 registry rewrite; a wrapper cannot upgrade a lexical
  scan into runtime proof, a logged instruction into an executed
  test, or ctx.log into PASS — binding ruling 6's UNAVAILABLE label
  covers the gct MCMC instruction and the ftw/tpe provenance echoes
  until reconciliation executes them. The wrapper falsehoods (gsv's
  "h5 alone" echo, the rtol echo, the bitwise echo, tpe's dated
  prose) join the factual bundle.
- 28 (subprocess semantics + diagnostics): RATIFIED — every
  subprocess owner gains the ten-field contract; failure paths print
  LABELED bounded tails of BOTH stdout and stderr plus cwd/argv/
  expected files (Cobaya errors live on stdout); the scalar readback
  prose distinguishes missing file / missing header / unusable
  header; board_selftest's FOUR unmanaged mkdtemp sites join the
  DIDACTICS-25 behavioral cleanup batch (one cleanup landing, all
  nine sites).
- 29 (triangle + logscan): gate-STRENGTHENING ruled, not narrowing —
  the triangle gate identifies each intended panel's axis, checks
  its actual filled collection, verifies grey RGBA within a stated
  tolerance, and identifies the omegamh2 marginal specifically; the
  wrong-panel + unrelated-patch mutation must fail (small
  board-listed executable work in the 29 batch). logscan's
  comparison is renamed to what it is — selected/normalized text
  equality — and "byte-identical" is RESERVED for raw
  bytes/undecoded tensors; registry prose inherits the honest term.
- 30 (two false explanations): the weight-decay conclusion is
  NARROWED to "the explicit decay term is disabled in every group"
  (no trajectory-equality test ordered; adding one is optional
  family-visit work); the BerHu dimensional prose is corrected
  branch by branch with the t1 = 0.2 hand calculation; the
  lazy-iteration / identity-set / reference-ownership / autograd /
  unbound-method teaching is ratified. Both falsehoods join the
  factual bundle.
- 32 (raw-log authority): RULING = OPTION 1 — complete, separately
  labeled child stdout and stderr are PERSISTED into the parent gate
  log (the raw-log-authority promise is kept, not retracted), with
  concise bounded tails in terminal failure details. Timeout policy:
  NO timeout is added — a workstation CAMB/generator call has no
  safe universal bound and interrupted attempts are already a
  first-class resume state — but the documentation states explicitly
  that subprocess.run waits indefinitely and captured mode shows no
  live progress. The stream-marker acceptance (both markers land in
  durable evidence) is ratified.

## DIDACTICS-33 adjudication (Fable, 2026-07-13): the row-coordinate correction extends into training.py — one landing with the 08 fix; errors name original dump rows in both regimes

(DIDACTICS-28..32 were re-filed this relay; they are already
adjudicated — see "DIDACTICS-27..32 adjudication" above.)

33 CONFIRMED, and WIDER than filed: beyond the cited anchors
(training.py:23-25 loader definition; eval_val ~:1570; order_rows
~:1606), the Architect's grep finds the same false-universal
"global rows" claim at :1788 ("global rows into the full dump") and
:1863; the ".cpu() = the one D2H copy" line sits at :1634 and is
conditional exactly as filed (a CPU tensor's .cpu() is normally a
no-op; torch.cat allocates BEFORE the transfer).

RULINGS (a rider on the DIDACTICS-08 factual correction — batching.py
and training.py are ONE landing sharing the two-regime coordinate
table): (1) "active source-array coordinate" replaces every
universal global-row claim, with the two-regime table carried into
eval_val; (2) the error-coordinate contract goes BEYOND the honest
label: in the resident regime, staging already holds the
original-row mapping (the sorted rows array), so a validation error
message TRANSLATES the local compact position and names the ORIGINAL
dump row — zero new persistence; the disk regime's coordinates are
already original rows; therefore a bad-row report names the original
dump row in BOTH regimes, and the prose says so; (3) torch.cat's
allocation is described separately from the conditional .cpu()
transfer, and every detach/clone/cpu/numpy chain states gradient,
ownership, device, and copy/view behavior separately.

Acceptance ratified: both regimes exercised with the [9,2,5]
example; a deliberately bad row reported as its original dump row in
both regimes; CPU and accelerator controls verify the transfer
wording; no universal global-row or unconditional-D2H claim survives
the scan.

## DIDACTICS-34..41 adjudication (Fable, 2026-07-13): eight model/loss/geometry documentation falsehoods, all CONFIRMED — two Architect widenings and one campaign-wide census upgrade

All eight anchors verified on HEAD; the red team's mathematics
checks out where checkable (35's joint-to-separable ratio
Ck/(gk+C) = 330/41 = 8.05 for C=30, g=1, k=11, against the claimed
~k/2 = 5.5; 38's capped-BerHu upper branch grows like sqrt(c) —
the VALUE is unbounded, the residual INFLUENCE is what plateaus).
All eight are documentation-only repairs routed into the didactics
campaign's model/loss/geometry family batches under the standing
acceptance (stripped-AST identity, compileall, the stale-phrase
scans); their worked examples (34's one-multipole encode->residual->
law-removal->chi2 trip, 38's three-row sorted-prefix-mask example,
41's ResBlock shape trace, 35's exact parameter counts) land as
UNIT 91 documentation fixtures.

Verdicts + Architect deltas:

- 34 (CMB law direction): CONFIRMED — encode forms f*C/sigma,
  decode DIVIDES by f (consistent with the queue-43 law: the chi2
  divides f back out); "already physical" mislabels a whitened
  residual. WIDENED: the reversed phrase has two siblings the filing
  did not cite — inference.py:180 ("multiplies the imposed amplitude
  law back") and :532-534 ("decode multiplies the amplitude law
  back") — the stale-phrase census covers all three.
- 35 (Conv1d bank): CONFIRMED with the ratio formula verified; the
  free-standing ~k/2 rule is deleted, the executed
  (C_out, C_in/g, k) bank taught, parameter counts shown or
  evaluated from the resolved run.
- 36 (shared_mlp): CONFIRMED — the executed branch is a
  fixed-width shared-across-token MLP ablation (dim -> dim every
  layer, activation after the last), not the dim -> d_ff -> dim
  textbook FFN; renamed honestly with the shape trace.
- 37 (PCE OMP-vs-LARS): CONFIRMED — the executed algorithm is
  OMP-style greedy selection with per-term ridge refits and
  PRESS/LOO + patience; "least-angle" and "one pass" retire from
  prose (public names may stay for compatibility).
- 38 (trim + BerHu): CONFIRMED — "topk" becomes
  sort-then-fixed-prefix-mask (with the compiled-shape rationale);
  bounded INFLUENCE (the sqrt(c) dL/dc vote) is distinguished from
  the unbounded loss value.
- 39 (loss study map): CONFIRMED — the package preamble teaches the
  geometry PROTOCOL (encode/decode/dest_idx/total_size as the
  relied-upon surface, not one concrete class) and the study map
  gains scalar.py/cmb.py/transfer.py with a mechanical
  file-completeness census.
- 40 (grid.py direction reversal): CONFIRMED as the amendment it
  claims to be — this instance SURVIVED the existing single-line
  direction census precisely because the false clause spans lines.
  CAMPAIGN-WIDE UPGRADE: every stale-phrase census in the didactics
  acceptance battery becomes multiline/semantic-capable (recorded
  scan patterns; single-line grep is no longer sufficient evidence
  for a zero-hit claim on prose).
- 41 (ResBlock skip location): CONFIRMED — the invariant moves to
  the module introduction (width-preserving (B,D)->(B,D); internal
  path Linear->norm->act->...->final Linear->ADD SKIP->final
  norm->final act; rectangular projections outside the block) with
  the named shape trace.

## DIDACTICS-42..57 adjudication (Fable, 2026-07-13): five gate-truth defects become one repair batch before queue 5; unit 92 born; the doc items routed

Architect verification: the vacuous-truth mechanisms are live —
logscan.byte_identity('', '', pattern) and noise-vs-noise BOTH return
(True, '') through the real helper (46); mps_smoke:144's
(arr[arr.sum(axis=1) != 0] > 0).all() filters the rows the claim must
reject (47); diagnostics_domain:304-305 joins its two absence
predicates with OR (52) — the sharpened form of the Architect's own
deep-pass ROLLOUT RIDER 3, whose when-next-touched deferral is hereby
UPGRADED to now; experiment.py:88-89 states ".h5 weights + .emul
recipe", the exact reversal (49).

GATE-TRUTH REPAIR BATCH (46 + 47 + 48 + 52 + 53): five false-green /
overclaim lanes in existing gates land as ONE repair increment
BEFORE queue 5, beside the unit-28 fixture repair — the workstation
evidence comes from repaired gates or it means less than it claims.
- 46: selection success separated from selection equality; nonempty
  both sides at minimum, caller-declared line classes preferred;
  zero-match failures name side/pattern/count; the label becomes
  "character equality of a declared subset", never byte identity;
  the empty/one-sided/noise mutants red through the REAL _golden_leg
  board path.
- 47 (rider on the unit-56 payload contract): exact declared row
  count asserted; the canonical publication predicate proves every
  row; finiteness after the storage cast; strict positivity on every
  stored P_lin element; NO pre-filtering of suspected failures
  before a whole-dataset verdict; the empty-array.all() convention
  taught.
- 48: the diagnostics PDF is resolved from the producer's own path
  helper, asserted to exist, nonempty, parseable, with the expected
  pages; the visual claim stays explicitly manual; "produced /
  opened / contains pages / looks correct" are four different
  claims; the rc-zero-no-PDF fake is the load-bearing mutation.
- 52: the OR verdict is replaced by the AST/data-flow census over
  every diagnostic chi2 producer (compute-dtype score reaches the
  screen before any float64 conversion), with single-mutation red
  legs for each forbidden path alone; De Morgan taught in place.
- 53 RULING: EXTEND, not narrow — the six-block covariance is
  science-bearing (the Motloch-Hu off-diagonal work), so every block
  gains finiteness, every same-spectrum block symmetry + PSD, every
  cross block its placement/known-answer relation, per-block
  evidence in the raw log; the corrupt-only-cov_ee and
  corrupt-one-cross mutations must red.

UNIT 92 (54, device-audit totality): audit_devices promises every
tensor but walks only vars() one level deep and demotes mismatches to
silent-gated performance prints. Contract ratified: cycle-safe
recursive traversal (nested modules, containers, composed
geometry/loss owners), alias deduplication by identity with ownership
paths, a REAL mismatch raises regardless of silent (only formatting
is quiet-gated), device type and index documented separately; CPU
traversal legs + the workstation CUDA lane proving a nested CPU
tensor cannot enter a CUDA loss silently; the shallow-vars mutation
must red.

Documentation routing: 42 (the package intro still calls the
five-family library cosmic-shear-only), 43 (set_device does NOT
retarget default-device ops; worker streams CAN interleave), 49 (the
artifact reversal + the stage_train loader claim + the cosmic-shear
headline on the five-family class), and 50 (factored encode does NOT
increase width — the property computes n_param exactly) join the
FACTUAL BUNDLE. 44, 55, 56, 57 join the family batches with their
censuses (whitening names its random quantity and covariance
assumption; association language replaces causal/optimal claims;
state_dict membership = registered parameters incl. frozen + persistent
buffers, one canonical definition reused; decode is the inverse on
the kept subspace only, with the three precise identities). 51 is
ratified and its AST census (public signature names vs Arguments:
entries; declared returns vs the executed tuple/dict surface) JOINS
THE STANDING ACCEPTANCE BATTERY. 45 duplicates the already-ruled
DIDACTICS-11 (_analytic_R dtype contract — same anchors, same
repair): MERGED, no second unit. The 40-ADDENDUM is accepted: the
direction census covers docstrings, comments, AND exception strings
(printed errors are user documentation), adding the cmb.py:207/:245
and scalar.py:121-128 reversal sites to the multiline scan.

## DIDACTICS-58 adjudication (Fable, 2026-07-13): the canonical harness-Python preamble — a rider on the DIDACTICS-02 __init__ landing

CONFIRMED with a precise census: the Architect's own count finds
EXACTLY 24 gate scripts carrying both the module-global FAILURES
accumulator and the __main__ executable footer (full intersection of
the two patterns). RATIFIED as filed, PLACED as a rider on the
DIDACTICS-02 gates/checks/__init__.py landing — the file-by-file
index and the canonical beginner preamble are one edit to one file.
The preamble teaches all seven mechanics (module-load-time list
creation; name lookup finding the module-level object; in-place
.append needing no global; rebinding needing global; per-subprocess
isolation via fresh module execution; __name__ semantics; main() ->
sys.exit -> shell/board exit status), each standalone script points
to it without team shorthand, per-file report docstrings stay
descriptive, and — per the standing C-coder rule — the repeated
pattern is NOT compressed into a clever framework to save lines.

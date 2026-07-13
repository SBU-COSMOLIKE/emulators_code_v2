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

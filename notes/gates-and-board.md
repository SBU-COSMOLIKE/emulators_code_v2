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

<a id="board-selftest-exit-truth"></a>
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
landing. Home-note spec (`data-generation-and-cuts.md#stage-ram-both-copies`) and the
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

## DIDACTICS-62/63 adjudication (Fable, 2026-07-13): banner-as-truth — prose narrowed now, behavioral evidence in a second gate-truth increment

Both CONFIRMED as the banner-as-truth class: five gates
(head-scheduler-override, berhu-anneal, ema-anneal, joint-training,
relu-tanh-norm) advertise behavioral proofs (lr cadence, exact
hold/ramp values and continuity, trunk-parameter movement, loss
descent) while their assertions check exit codes and banner
substrings; param-window-cuts and production-diagnostic accept ANY
"used <n> of <m> cut rows" line without parsing the integers or
comparing against an independent reference — a driver hard-printing
"used 1 of 1 cut rows" greens both legs. Point 7 is adopted as
doctrine and recorded beside binding ruling 6: the evidence rollout
proves an assertion EXECUTED; it can never supply an observation the
assertion never made — reconciliation and observation are orthogonal
axes of gate truth.

RULINGS (narrow-now / strengthen-next):

1. PROSE NARROWING is immediate (joins the queue-6 factual bundle):
   each of the seven gates' titles/maps= describe exactly the
   startup/schema/banner evidence executed today, with the
   behavioral claims moved to explicitly-owed status — so the
   queue-5 run's claims are honest even where power is still owed,
   and the queue-2 aid minting uses the NARROWED claims.
2. DIDACTICS-63 (independent-reference repair for param-window-cuts
   + production-diagnostic's banner leg) JOINS THE GATE-TRUTH REPAIR
   BATCH (same class as 46-53, small): the deterministic table with
   an independently computed physical mask; used/total parsed as
   integers and checked against the independent count, the requested
   selection, AND the staged-index identities (identity comparison,
   not count); 0 <= used <= total with nontrivial shrinkage; the
   hardcoded-plausible-banner mutation must fail; the lesson
   recorded — a banner is an observation to compare against an
   independent reference, never its own evidence.
3. DIDACTICS-62's behavioral strengthening is GATE-TRUTH INCREMENT
   2: where the evidence already exists in driver output, parse and
   assert it pre-queue-5 — scheduler: phase-tagged lr events at the
   patience cadence (ignore-the-override mutation reds); anneals:
   the real schedule evaluated before the hold, at the boundary,
   during the ramp, at the endpoint, exact values + continuity
   (constant-schedule mutation reds); activations: two finite loss
   observations + an explicit descent criterion (dead/mean-only
   nets red). Joint-training's trunk-movement proof needs evidence
   the driver does not yet emit: the driver gains ONE
   phase-boundary trunk-parameter digest line (essential-only
   compatible — per phase, not per epoch), and the gate asserts a
   finite nonzero change in the joint run and EXACT ZERO in the
   frozen control; that half rides the training-truth trio
   (79/88/89) if increment 2's timing is tight, with its narrowed
   prose honest in the interim.
4. Banner/schema evidence and numerical/optimization evidence are
   reported separately in every repaired gate.

## DIDACTICS-59/60 adjudication (Fable, 2026-07-13): the copied-reference gate and the stale baseline — both join gate-truth increment 2, with one sequencing trap caught

(Filed twice in one relay; counted once.)

59 CONFIRMED (the copied-reference problem): ge_c_eval_bs computes
its own per-row chi2 array while production eval_val publishes only
reductions — the gate can certify its local helper while the real
aggregation, row association, or ragged-batch handling is wrong.
Contract RATIFIED: the REAL eval_val entry point exercised across
the three partitions (one batch / equal batches / ragged final);
every published production value compared (mean, median, threshold
fractions, ranking-history values) against an independent float64
reference; the distinct-scores row-permutation fixture; the
drop-or-reorder-a-batch mutation that leaves the copied helper
unchanged and must red. On clause 7, the RULING is reuse-not-add:
per-row scores are already produced on the production diagnostics
path (the _screen_diag_chi2 producers) — the gate consumes that
existing production-owned surface; no new boundary and no duplicated
formula. The board prose names exactly which production outputs are
compared.

60 CONFIRMED (the stale baseline + the unsupported cause): the
smoothness-causes-two-epoch-convergence claim and the stale 0.455
are removed NOW (factual bundle); two epochs is described as the
bounded runtime of a deterministic smoke fixture — a budget, not a
scientific explanation. Contract RATIFIED: the mean-only baseline
recomputed from the EXACT staged rows; the four evidence values
printed (val-row count, mean-only median, trained median,
threshold); the trained model beats BOTH the fixed threshold and the
recomputed baseline; 0.3 retained only with recorded empirical
margin; the mean-only/dead-network mutation arm (the dead-network
law, formalized for scalar-smoke). SEQUENCING TRAP CAUGHT: the
recomputed 0.4401868977 was measured on the CURRENT
100%-overlapping fixture — the unit-28 disjointness repair changes
the staged rows, so 60's baseline recomputation and margin recording
land WITH OR AFTER that repair, never before; the number in this
filing is evidence of staleness, not the new reference.

Placement: both join GATE-TRUTH INCREMENT 2 (the behavioral-evidence
batch, pre-queue-5), 60 ordered after the unit-28 fixture repair
within it.

## DIDACTICS-61 + 64..71 adjudication (Fable, 2026-07-13): the durable register adopted; nine confirmations; UNIT 93 minted; 65 merges into 29; the 46 golden-leg addendum absorbed

THE REGISTER: the red team's findings now live durably in
notes/red-team-audit-and-didactics-2026-07-13.md ("Durable DIDACTICS
handoff register: 42--71", on main). The register copies of 42-58,
59/60, and 62/63 were reconciled against the rulings already issued
in this file — they MATCH; no reopened item. The register's 46 entry
carries ONE addendum beyond my original filing, verified and
absorbed below. The durable-record rule is ADOPTED both ways: every
future red-team filing is appended to a topic/audit note before its
chat copy, and adjudications cite the file, never the chat.

Verification (Architect, worktree at 976c739; untruncated greps):

- 61: stage_ram.py:139's evidence string says "the one-wide float32
  target" while the fixture is dv = np.zeros((n, 2)) (:129) — the
  byte arithmetic uses 2 correctly; only the prose is false.
  logscan.decreasing already ENFORCES the two-point minimum
  (len < 2 -> False, "need at least two points") while its Returns
  block still says "non-empty" — a stale docstring, not a code hole.
  The REAL behavioral hole: a +Inf first value with a finite last
  gives drop = +Inf > tol and PASSES — a numerically exploded run
  reads as a descent. (NaN endpoints already fail closed through the
  comparison.)
- 64: gsv_bitwise_drift.py:328-335 patches
  training.DEFAULT_COMPILE_MODE around a rebuild that is called with
  compile_model=False (:251 default) AND whose fixture recipe
  persists "compile_mode": None (:148) — rebuild reads the persisted
  recipe, so the patched global provably cannot reach the compared
  output. The compile-mode drift arm asserts nothing.
- 65: the global-count mechanism was verified at the 29
  adjudication; the refile adds the exact acceptance (below).
- 66: gct_parity.py:141-147 derives masked from the SAME rebuilt
  geometry's dest_idx and accepts nonzero_masked == 0 — an
  all-covering dest_idx makes the masked selection empty (vacuous
  green), and a lost cut lets the object under test redefine its own
  expected mask.
- 67: warmstart.py:78 "loaded once", :80 "builds this once", :271
  "Load and validate the source emulator once" — while load_source
  executes rebuild_emulator (:309; opens the .h5 and torch.loads the
  .emul) and then opens h5py.File(root + ".h5") AGAIN (:335) for the
  recipe/resolved facts. The FinetuneSource attribute docstring ends
  at dataset; the live .ia attribute (:131) is absent from it.
- 68: _require_parity_finite guards exactly the four finetune
  baseline tensors (:945-951) and three transfer tensors
  (:1113-1117); BOTH perturbed arms (finetune :957-965, transfer
  :1125-1134) feed torch.equal unguarded, so a NaN/Inf born only on
  the perturbation is mislabeled "the extra parameters leaked" /
  "moved the epoch-0 prediction" — the developer is sent hunting
  zero-padding instead of the overflow.
- 69: the untruncated grep finds EXACTLY the six claimed sites
  (warmstart:34, diagnostics:88, losses/ia:16, designs/ia:18,
  designs/plain:60, geometries/output:52) — no seventh.
- 70: results.py:150 says "(a cuda-saved state needs the saving GPU
  visible)" while the executed save runs detach().cpu() on every
  persisted tensor (:292, :311, :370, :395) and rebuild loads with
  map_location=device (:722). gsv_bitwise_drift has NO CPU-residency
  assertion on the serialized tensors — the gate clause is
  add-not-cite.
- 71: warmstart.py:20-21's headline promises "the source emulator's
  own function, bit for bit" while the function-reproduction
  comparison is tolerance (_PARITY_TOL = 1.0e-5, :967); only the
  extras-independence half is torch.equal (:961), and transfer's
  epoch-0 bitwise base equality (:1119) is a separate invariant.
- 46 ADDENDUM: _golden_leg discards BOTH child return codes
  (board.py, "_, cur" / "_, pre") before byte_identity — combined
  with the empty-selection green already live-reproduced, two
  crashed children after their last matching lines, or a one-sided
  crash, or a nothing-matching pattern all PASS.

RULINGS:

1. 61 splits: the stage_ram evidence string and the decreasing
   docstring ("non-empty" -> the executed two-point strict rule) are
   FACTUAL-BUNDLE fixes; the finiteness requirement is a small
   BEHAVIORAL addition joining the GATE-TRUTH BATCH — both endpoints
   must be finite, a nonfinite endpoint returns False with a named
   detail (house finite-contract doctrine: a nonfinite loss is a
   broken run, not a descent) — with the five named controls (empty,
   one-value, NaN, equal, +Inf-first) as selftest legs.
2. 64 splits: the vacuous compile-mode arm is REMOVED NOW (gate-truth
   increment 2's honest-evidence class — a drift claim its own patch
   cannot reach may not stand); compile-mode persistence becomes NEW
   UNIT 93: a CUDA lane that rebuilds with compile_model=True,
   instruments torch.compile to OBSERVE the persisted recipe mode,
   and reds when rebuild ignores or loses that field. Board-listed
   accelerator leg, workstation-owed (the queue-5 exhibit family
   with 77/80's Torch legs). Never-trust-defaults: a persisted
   recipe field must be provably honored by the consuming code.
3. 65 MERGES INTO 29 — one home, the sharper acceptance adopted: the
   exact expected (x_parameter, y_parameter, window) set constructed
   independently, axis identities extracted from each Axes object,
   the exact set/count compared, the omegamh2 marginal identified
   specifically, _CUT_GREY checked on each expected artist, and the
   move-a-correct-artist-to-a-wrong-axis mutation that leaves global
   counts unchanged must red.
4. 66 joins the GATE-TRUTH BATCH (the same vacuous-pass class as
   47): the fixture declares an independent, NONEMPTY expected
   masked-index set; the gate asserts observed mask identity and
   count against it, exact zeros there, and compares kept
   coordinates between section and full outputs; the all-kept and
   deleted-expected-index mutations must fail.
5. 67 + 68 land as ONE WARMSTART VISIT (same file, prose + guard
   together, after the gauntlet with the family visits). 67 ruling:
   TEACH, don't consolidate — the second metadata pass is stated
   with its reason (rebuild_emulator owns network reconstruction;
   the follow-up open reads recipe/resolved facts), object count is
   distinguished from file-open count, and ia = nla / tatt / None is
   defined; the instrumented h5py.File/torch.load open-count census
   that pins the final prose is a UNIT 91 executable example. 68
   ruling: the shared finite predicate is applied to the perturbed
   encode and output tensors in BOTH arms (a production fix); board
   legs mint NaN and Inf only on the perturbation and require the
   finite-contract error naming quantity/side/row; removing either
   perturbed-arm guard must red.
6. 69: ONE canonical sentence — "decorrelated and unit variance
   (equal numerical scale); this does not guarantee equal
   learnability" — replaces all six sites, a FACTUAL-BUNDLE fix
   whose completion evidence is a multiline repo scan leaving zero
   "equally hard"/"equally easy" claims in emulator/ (the campaign's
   multiline-census law; single-line grep insufficient).
7. 70 splits: the results.py:150 correction (this library persists a
   CPU-normalized checkpoint, distinct from a raw CUDA checkpoint;
   map_location selects the load destination) is a FACTUAL-BUNDLE
   fix; the CPU-residency leg (torch.load the .emul and assert every
   tensor's device.type == "cpu") joins gate-truth increment 2
   (verified absent today — add, not cite).
8. 71: the headline is rewritten to the TWO invariants — the widened
   network reproduces the source function within _PARITY_TOL
   (different matrix widths change floating-point reduction order),
   and perturbing only the zero-connected extras on the SAME widened
   network is bit-identical (torch.equal) — with transfer's separate
   bitwise base-composition requirement untouched. FACTUAL-BUNDLE
   fix; the bit-for-bit census (every "bit for bit" / "bitwise" /
   "bit-identical" occurrence mapped to the comparator actually
   used) joins the quantifier-discipline battery.
9. The 46 golden-leg addendum is ABSORBED into 46's gate-truth
   repair: identity requires BOTH return codes zero AND a nonempty /
   minimum selected-line count AND equality; both rcs and both
   counts reported; the three mutations (empty selection; both
   children rc 1 after matching lines; tip-only rc 1) must red.

Sequencing: the gauntlet is UNCHANGED — batches 4/5b -> the unit-28
fixture repair -> the gate-truth batch (46+addendum/47/48/52/53/63
+ 61-finiteness + 66) -> gate-truth increment 2 (62/59/60 +
64-narrowing + 70-leg). Unit 93 is workstation-owed; 67+68 ride the
family visits; the factual bundle grows by 61/69/70/71's prose.

## The consolidated DIDACTICS execution handoff (Fable, 2026-07-13): one plan for findings 01--71

Every DIDACTICS finding is adjudicated; this section is the ONE
execution plan the Implementer works from, consolidating the
per-batch adjudications above into ordered deliverables. The full
filings live in notes/red-team-audit-and-didactics-2026-07-13.md
(the durable register, 42--71) and the per-batch sections of this
file (01/02, 03..09, 10..19, 20..26, 27..32, 33..41, 42..57, 58,
62/63, 59/60, 61+64..71). On any tension between this summary and a
per-batch section, the per-batch section wins.

THE SPINE (binding order; nothing below preempts it): batches 4/5b
(1b population) -> the unit-28 smoke-fixture repair -> the
gate-truth batch -> gate-truth increment 2 -> queue 5. Doc-only
deliverables interleave as review-blocked filler.

- D1, navigation truth (DIDACTICS-02 + 58): the gates/checks
  __init__ file-by-file index; the README inventory mechanically
  validated against BOARD by a board-selftest set-equality leg with
  a helper-file allowlist; no numeric gate count anywhere in prose;
  the beginner-first runner walkthrough before the evidence
  vocabulary; the worked miniature gate with all six artifacts; the
  seven-mechanics harness-Python preamble (58, census: 24 files)
  rides this landing. Manifest/Gate docstring cleanup AFTER 4/5b.
- D2, the factual bundle (queue 6; falsehoods jump the queue; first
  landing = 02 + 05a + 08 + 33 per their sections): every item a
  per-batch section marks "factual bundle", headline members — the
  package/experiment/inference cosmic-shear-only intros (42/49/10a);
  the .h5/.emul ownership reversals; "re-derives no physics" (14);
  the dtype contract truths (11a/45); six-vs-seven syren args (12);
  board.py :634/:1370; the stale "refine not yet implemented" (23);
  the HDF5-typed correction (24); geo_paths' import-raises prose
  (26); scalar-smoke's "smooth"/0.455 (60-prose); experiment.py's
  false "independent selection" comment (31); the wrapper falsehood
  echoes (27); weight-decay narrowing + BerHu units (30); the
  batching.py two-regime rows correction (08); the README
  all-family + BSN honest narrowing (05a); scheduling device/worker
  scope (43); CMB whitening weighting (44); factored-width (50);
  state_dict frozen params (56); masked-decode inverse (57);
  association-not-causation (55); 61's two prose halves; 69's
  six-site canonical whitening sentence (multiline census); 70's
  CPU-normalization correction; 71's two-invariant split (+ the
  bit-for-bit census).
- D3, the unit-28 smoke-fixture repair (31): distinct RECORDED
  generator seeds (or one proven-disjoint partition), zero-overlap
  refusal before training, row-alignment proof, printed
  seeds/counts/overlap, the same-seed mutation arm. BEFORE queue 5.
- D4, the gate-truth batch: 46 (+ the golden-leg rc addendum: both
  rc==0, nonempty/minimum selected-line count, both rcs+counts
  reported, three mutations), 47 (payload validated before
  filtering), 48 (the PDF verified), 52 (both absences, per-pattern
  mutations), 53 (all six blocks or the narrowed contract), 63
  (parse-and-verify the used-n-of-m banner against an independent
  mask), 61-finiteness (decreasing: both endpoints finite, five
  named controls), 66 (independent nonempty expected masked-index
  set + two mutations).
- D5, gate-truth increment 2: 62's five behavioral gates
  (parse-and-assert; the joint-training driver's ONE phase-boundary
  trunk digest line), 59 (the real eval_val across three
  partitions; reuse the production per-row surface), 60 (baseline
  recomputed AFTER D3, both bars, dead-network arm), 64's vacuous
  compile-arm removal, 70's CPU-residency leg.
- D6, behavioral hygiene: the mkdtemp cleanup (all NINE sites — the
  five gate sites of 25 + board_selftest's four of 28 — one
  landing, injected-failure acceptance); the triangle strengthening
  (29 with 65's exact acceptance: expected (x,y,window) set, axis
  identities, per-artist _CUT_GREY, moved-artist mutation).
- D7, UNIT 91 (the documentation-examples gate): ONE board-listed
  module holding every executable worked example — 11's dtype
  controls, 12's unit-table check, 13's five-point Simpson, 15's
  instrumented lifecycle order, 16's overlapping-requirement
  example, 17's interpolator shape/range cases, 18's scatter+mixed
  fixtures, 04's coordinate-table leg, 21's activation tail tests,
  22's constructor tests (validation code rides unit 40), 67's
  open-count census. Production files stay stripped-AST-identical.
- D8, lane-3 family visits (each rides its protocol unit, never
  before): 15-19 adapter teaching with units 71-75/73/85; 12's
  keyword-arg conversion with 85; 13 + unit 90 with the BSN visit;
  10b with unit 66; 03/04/06/07/09 per gate family; 23's
  finetune/transfer preambles; 28's subprocess ten-field contract +
  both-stream failure tails; 32's persisted labeled child streams +
  the explicit no-timeout statement; 67+68 as ONE warmstart visit
  (teach the double open, define .ia, finite guards on BOTH
  perturbed parity arms + board legs).
- D9, queue-2 riders: 20's registry stable-section rewrite, 27's
  wrapper-child reconciliation, UNAVAILABLE labeling per binding
  ruling 6, and 09's taught vocabulary in the rollout's logs.
- D10, minted units on the board: 90 (BSN independent quadrature +
  Simpson-weights mutation), 91 (D7), 92 (device-audit totality),
  93 (compile-mode persistence, CUDA instrumented lane) — 93 and
  the accelerator halves are workstation-owed queue-5 exhibits.

ACCEPTANCE, binding per landing: the AST docstring census with the
reviewed one-liner allowlist; the untruncated 108-line
history-pattern scan (emulator/ + cobaya_theory/ + gates/);
compileall + board-selftest green; documentation-only AST
equivalence with the NAMED executable exceptions only (04's table
leg, units 90-93, the D4-D6 gate repairs); the quantifier
discipline (bitwise / exactly-once / all claims carry adjacent
mechanical proof); multiline stale-phrase censuses;
scope-and-blind-spot labels ("census" = complete surfaces only);
the signature-vs-Arguments AST census (51); the second-year physics
student voice — the Implementer reads
notes/user-didactics-and-python-voice.md BEFORE writing any prose.

Handed off 2026-07-13; this section is the citation for the
relayed ARCHITECT_HANDOFF covering the whole campaign.

## Registry-persistence audit (Fable, 2026-07-13): the retroactive durable-record pass — GO for merge

The red team applied the durable-record rule retroactively to every
earlier bug-hunt series on codex/architect-docs-static-audit (commit
57dfe26, notes-only; main e6bcaf3 merged in as 35f0137, no
conflicts). Architect verification, each claim checked at source:

- 57dfe26's stat touches EXACTLY five notes files — no production,
  gate, generator, or adapter code. The "doc-only" claim is true.
- Tombstones: an untruncated repo-wide grep finds 45M-10, 45M-18,
  and 20M-20 ONLY as the registry's explicit "not issued" lines —
  the number gaps are recorded, not backfilled.
- Retractions 45M-05/43/44 remain visible (registry + the
  training-stack VOID analysis + the state ledger) and are not
  rewritten as live defects; 45M-08's provenance limitation
  (index-item-only receipt) is stated rather than reconstructed.
- The dataset-readiness contract's LIVE-STATUS claims verified
  against code: generator_core.py's unconditional MPI.Finalize() +
  exit(0) tail (:1183-1184), the silent boundary rewrite (:241 —
  their text even correctly notes boundary == 1 takes the fallback
  branch as the valid unchanged endpoint), and data_staging.py's
  missing fail-file consumer. All still present at the audited HEAD.
- The scalar-driver closure (Original unit 5(a)) matches recorded
  history (the pre-existing driver, 45M-80's executable refusal at
  e9943bc) and is correctly scoped — it closes only the stale
  routing claim, explicitly not the artifact-integrity units.
- The scheduler execution protocol (45M-25) is a coherent
  domain-owner contract consistent with house law (one warmup
  owner; resolved pass records = never-trust-defaults; counting
  schedulers + analytic LR sequence as the board gate), marked open.
- MEMORY.md's 17 deleted lines remove exactly the stale numeric
  snapshots ("32/32 at run 9", "the 32 gates") in favor of
  board.py/--list as the authoritative gate census — the
  DIDACTICS-02 no-counts-in-prose doctrine applied to the index —
  and the new cold-start order routes readers through the registry.
  The ledger's demotion to "chronology and routing, not a live
  queue; current sequencing comes from the latest persisted
  Architect handoff" is CORRECT and now points at the consolidated
  DIDACTICS execution handoff above.

One naming equivalence recorded, not a defect: the crosswalk says
"the 20M-15 amendment owns checkpoint-ingress revalidation" where
this file's ledger phrased the same object as "unit 56's
checkpoint-ingress amendment" (the amendment was minted during the
20M-15 adjudication and attached to unit 56). The registry's own
scope warning — a bare number is historical shorthand; cite the
finding name and owner heading — covers exactly this case.

VERDICT: GO. Landing order: the codex branch first (main
fast-forwards to 35f0137), then claude/amazing-keller-e798b6
(2f25450; disjoint file sets, clean merge), then both sessions run
the resync ritual. The registry becomes the canonical identifier
crosswalk; adjudications keep citing topic-note headings.

## Registry alias closure (d00358c) — audited, GO (Fable, 2026-07-13)

The red team's d00358c (notes-only, 3 files) closes the one aliasing
seam from the registry audit: the crosswalk now reads "15 (the
checkpoint-ingress amendment to unit 56)" and the dataset contract
"the 20M-15 checkpoint-ingress amendment to that same unit 56 owns
resumed-row revalidation" — both labels joined in place, verified at
the anchors. MEMORY.md's cold-start order now names the consolidated
D1-D10 handoff as sequencing authority. The handoff's "main dirty"
hold was STALE at audit time: the user had concluded the vi-stuck
merge as 7743fc6, MERGE_HEAD gone, status clean. d00358c's base
83c8624 is an ancestor of main; file sets disjoint from 7743fc6's —
the refresh-first step is unnecessary; merge directly.

## Batch-4/5b checkpoint audit + the deploy_data shape ruling (Fable, 2026-07-13)

CHECKPOINT AUDITED (the three leftover no-schema gates, uncommitted
in the shared worktree; diff read pre-commit): exactly three
Manifest insertions, five lines. weight-decay-census code=()
inputs=() is CORRECT for a reasoned cause, not by accident: the gate
works purely through subprocess drivers, so its IN-PROCESS check
closure never reaches the waived dynamic-import surface — child
executables are the trunk digest's jurisdiction. save-rebuild-drift
and cobaya-adapter rebuild IN-PROCESS (their closures reach
results.py's waived imports), so designs+losses roots are required
— the geo-paths exemplar's logic, applied consistently.
validate_manifests: 18 declared, ok=True. Commit custody: the
Implementer commits its own board.py work at the assembled landing;
Architect commits stay notes-only. At the assembled-landing audit I
will independently re-derive gwd-c's empty closure
(beyond-sufficiency, as done for geo-paths).

THE DEPLOY_DATA SHAPE RULING (the one open design decision):
APPROVED as recommended — ONE shared top-level deploy_data block in
board_config.json, logical-key -> rootdir-relative path:

  "deploy_data": {
    "w0wa_takahashi_train_cs_16":
      "projects/lsst_y1/chains/w0wa_takahashi_dvs_train_cs_16.npy",
    "lsst_y1_dataset":
      "external_modules/data/lsst_y1/lsst_y1_M1_GGL0.05.dataset"
  }

Binding constraints:
1. Logical keys are SEMANTIC (name the fixture, never a gate id).
2. Lane rule by PROVENANCE, not consumer count: deploy_data holds
   deploy-machine data (external_modules/data, projects/*/chains);
   gate_data keeps gate-owned configs/fixtures (ruling 2's lane). A
   single-consumer deploy path still lives in deploy_data — one lane
   for deploy data, chosen by what the path IS.
3. Gates reference dotted deploy_data.<logical-key> in inputs=,
   resolved through the existing ladder with resolve-not-exist
   semantics (rider r3; cross-machine None-sha members).
4. A referenced-but-absent logical key is STARTUP RED at
   validate_manifests (exit 2, the dotted path named) — no fallback,
   no default (never-trust-defaults).
5. No literal path may appear twice: gates sharing a fixture MUST
   share the key. The sign-off proposal proves this by presenting
   the FULL derived block as a paste-ready JSON block plus every
   consuming gate's inputs= — values derived from the shipped
   configs, never guessed.
6. board_config.json remains the local deployment override; machine-
   varying paths belong exactly there.

Sequencing confirmed: BATCH-3-STYLE gates + cli-strict land
immediately under the standing GOs; the deploy_data proposal returns
for sign-off; then the full-board Mac validation and the review
handoff. The deferred resume block lands with the assembled landing.

## DIDACTICS-72..92 adjudication (Fable, 2026-07-13): the README wave — four file visits, the factual heads jump the queue, one aliasing rule

The register (notes/red-team-audit-and-didactics-2026-07-13.md,
"README-focused DIDACTICS handoff register: 72--92", e6256a1) read in
full; anchors sample-verified at the audited HEAD, ALL CONFIRMED:

- 79: the canonical generator command (README.md:2425-2427) omits
  --seed, which generator_core.py:157-166 declares required=True
  ("an unrecorded seed cannot be replayed"); --unif is required=True
  choices [0,1] (:125-130), not a default; and the command passes
  --boundary 1.0 two paragraphs after the README's own "pass only
  0 < --boundary < 1" (:2387-2389). The copyable command CANNOT RUN
  as printed — the repo's highest-visibility falsehood class.
- 86: both same-file contradictions verbatim — emulator/README.md
  :128 "the grid2d law transform currently defeats the memory
  ladder" vs :419 "Bounded grid2d staging has landed ... the retired
  full-materialization concern"; :83 still names the retired
  external_modules/code/emulators/emultrfv2/ path.
- 87: gates/README.md:3-7 "cosmic-shear ... runs every ... one raw
  log per test"; :46/:147 promise fixed logs/<test>.log against the
  landed immutable per-attempt timestamped logs.
- 90: syren/README.md:3-7 universalizes log(P/P_base) against the
  three-law registry (none = raw rows; syren_linear = log(P/P_base);
  syren_halofit = log(B/B_base)).
- 91, SHARPENED beyond the filing: ":34-35 byte-verbatim
  (AST-verified in the vendoring probe)" — an untruncated grep over
  gates/, syren/, emulator/, compute_data_vectors/ finds NO retained
  vendoring probe (the three "vendor" hits are folder-reference
  prose). The parenthetical cites a NONEXISTENT artifact: binding
  ruling 6's never-executed-evidence class, in a README.
- 81: emulator/README.md:97-100 "the same per-sample chi2" verbatim.

RULINGS:

1. THE FACTUAL HEADS JUMP THE QUEUE (join the D2 factual bundle):
   79's command repair (a reproducible explained seed; --unif stated
   required with its 0/1 meanings; an interior boundary value or the
   flag omitted); 86's two contradictions + the retired path; 87's
   false scope/every-test/log-name claims; 89's edit-the-config
   instruction (a standard clone resolves rootdir: null from
   $ROOTDIR) and the "stale git tip" mischaracterization (preflight
   proves base-notes ANCESTRY of HEAD; it never fetches origin);
   81 -> "same per-row score interface, shape (B,)"; 82's narrowing
   (the square-dense/two-projection rule is the ResMLP trunk's, not
   the library's — Conv1d, attention, FiLMGenerator are
   counterexamples at blocks.py:350-425); 83's precise claim
   (non-cosmic-shear training does not import CosmoLike — not "pure
   PyTorch"); 90's law table heads; 91's honest narrowing
   ("vendoring prevents an unreviewed package-manager upgrade";
   "byte-verbatim" reserved for byte/hash comparison; the
   nonexistent-probe citation removed); 76's three data-boundary
   corrections (generator writes PHYSICAL values, ParamGeometry
   whitens after selection; the dv .npy is memmapped while the
   parameter .txt parses eagerly; "in-memory source" vs file-backed
   payload distinguished).
2. FOUR FILE VISITS, one per README, queue-6 lane (doc-only,
   review-blocked filler): root (72-80), package map (81-86), gates
   (87-89 MERGED with DIDACTICS-02's D1 landing — gates/README.md
   gets ONE visit: walkthrough + completed table + operator path
   together, never two passes), syren (90-92, landing WITH or AFTER
   unit 75 since 92's domain paragraph sources its ranges there
   rather than inventing them).
3. Riders honored as filed — no duplicate mechanisms: 86 + 88
   sharpen the queue-6 restructuring and the DIDACTICS-62/ruling-6
   acceptance surfaces; 88's completed table runs through
   DIDACTICS-02's mechanical set-equality leg, extended to the
   six-field rows and preceded by the evidence ladder (banner/schema
   < parity/round-trip < independent known answer with mutation
   catch power); 91's identity half rides the artifact
   implementation-identity work; 92 teaches the honest evidence
   split (stub-base assembly identity vs the law-none real-CAMB
   smoke; neither proves the real Syren fit's accuracy).
4. THE ALIASING RULE (binding, program-wide): DIDACTICS labels now
   collide with unit numbers (DIDACTICS-90/91/92 vs units 90/91/92).
   Every artifact writes the DIDACTICS- prefix in full; a bare
   number means a UNIT. The register's own scope note anticipated
   exactly this.
5. Acceptance additions for this wave: every repaired command is
   EXECUTED once on its appropriate machine before it is printed
   (Mac-safe now; workstation commands at the queue-5 window, run
   recorded) — command-to-output truth is the wave's theme and a
   printed command is a quantifier claim about reality; the 86 link
   check covers ALL READMEs (clickable ../notes/... paths); no
   numeric gate count anywhere (DIDACTICS-02 law); untruncated
   stale-phrase scans per file; the voice note read before writing.

Sequencing: doc-only; nothing preempts the spine (4/5b -> D3 -> D4
-> D5). The consolidated plan absorbs this wave without a new spine
position: D1 gains 72 + the merged gates-README visit; D2 gains the
factual heads; the four visits join the queue-6 lane.

## 25M-01..06 adjudication (Fable, 2026-07-13): all six CONFIRMED — one new unit (94), four amendments, one extension; two immediate production advisories

Durable register at fafc122 (entries carry contracts + red legs);
every anchor independently verified in code:

- 25M-01 CONFIRMED (generator_core.py:746-747): the uniform-branch
  safety margin multiplies the ABSOLUTE endpoint (1.0001*lo /
  0.9999*hi, sign-reversed for negatives) — proportional to the
  coordinate's distance from zero, not the interval width. The
  [70.0, 70.02] witness retains 29.99% of its width; [1000.0,
  1000.01] inverts and rng.uniform raises. Translation-dependent
  physics. -> NEW UNIT 94, the boundary-interior owner: ONE named
  helper working in interval coordinates (nextafter(low,high) /
  nextafter(high,low) preferred, or a named width-relative margin);
  finite, ordered, representably nonempty interior validated BEFORE
  sampling; requested AND resolved per-name support persisted in
  dataset identity — the generation-side half of the support story
  whose inference-side half is unit 84 (the artifact fact 84
  consumes is the one 94 records). Red legs as filed, including the
  endpoint-times-constant restoration mutation.
- 25M-02 CONFIRMED (:369-375 stretch infinite hard-prior endpoints
  by temp*width/5; :426-429 name all three uniform outputs only
  _<probe>_unifs while the non-unif branch embeds _<temp>): two
  healthy temperatures produce different supports at identical
  paths. -> UNIT 8 AMENDMENT: dataset identity includes sampling
  mode, requested temperature, resolved per-name bounds, the
  boundary-interior policy (94's), and seed/RNG; resume/append
  require exact identity before mutation; different identity =
  refusal or a digest-derived distinct path.
- 25M-03 CONFIRMED (:221-223: rank-0 __run_mcmc() runs
  UNCONDITIONALLY, then chain==1 merely skips
  __generate_datavectors): chain-only mode rewrites the parameter
  chain + .paramnames/.ranges/.covmat at the same stems and leaves
  old dv/failure files — a successful TWO-COMMAND corruption pairing
  new cosmology row i with old physics row i, all row counts
  agreeing. -> UNIT 8 + UNIT 82 JOINT AMENDMENT: chain-only owns a
  distinct identity/location or refuses before mutation when any
  full-dataset member exists; the manifest records mode; full
  generation cannot adopt a colliding chain-only bundle. The
  run-control state machine (unit 8) gains --chain as a mode axis
  beside --append/--loadchk.
- 25M-04 CONFIRMED (cosmic_shear_tune_emulator.py:217 family=
  "cosmolike" default; :290 study_name = STUDY_NAME if family is
  None else prog; the adjacent comment still promises the historic
  name): STUDY_NAME is now UNREACHABLE — the documented direct
  command forks its Optuna resume from cosmic_shear_tune to
  cosmic_shear_tune_emulator. A REGRESSION INTRODUCED BY A REPAIR
  (e9943bc) — the forward-walk law's second strike (FTW model-block
  precedent). -> UNIT 53 AMENDMENT: one pure resolver maps family
  identity to a stable study name (direct cosmolike -> the retained
  historic constant; wrappers -> their pinned tags); resolved name
  in the study manifest and final report; renames are explicit
  migrations, never validator collateral. The false comment is
  corrected in the SAME landing.
- 25M-05 CONFIRMED (both sweep writers persist raw args.activation
  — ntrain :493, hyperparam :444 outside the swept axis — while
  from_config resolves None -> YAML -> "H" at experiment.py:
  2181-2189): the shipped default runs H under "# activation=None".
  -> UNIT 41 EXTENSION (resolved-record truth reaches sweep
  products): metadata assembled from ONE immutable resolved run
  record, never raw optional CLI fields; activation-family sweeps
  record "swept" + ordered values; serial and pooled paths share the
  record; banner/artifact/table/figure agree.
- 25M-06 CONFIRMED (:786 .ranges writes {:.5e} while :799 chain rows
  write %.9e): float32-distinct bounds 70.00001/70.00002 both
  publish as 7.00000e+01 — a false zero-width support the manifest
  would faithfully digest (file identity cannot substitute for
  representation truth). -> UNIT 82 EXTENSION (canonical
  representation, coupled with unit 87's one-decimal-contract): one
  decimal policy derived from the owned dtype shared by
  chain/header/ranges; publication refuses before mutation if
  conversion collapses a valid interval; both witness pairs
  round-trip distinct.

TWO IMMEDIATE PRODUCTION ADVISORIES (effective now, until landings):
(1) uniform generation with NARROW or LARGE-OFFSET intervals is
unsafe (the margin eats width or inverts); shipped broad priors are
negligibly affected (margin << width). (2) --chain 1 at stems
holding a full dataset is the two-command corruption — do not run it
there. These join the standing 45M-81 no-append advisory.

Sequencing: production-code repairs in the unit queue (not the
pre-queue-5 gate-truth gauntlet); 25M-02/03 land with unit 8's
run-control machine; nothing preempts batches 4/5b -> D3 -> D4 ->
D5. The 25M series joins the registry namespace with the
durable-record rule honored (fafc122 landed before the chat copy).

## Phase-3 landing audit + deploy_data sign-off + the census word-boundary ruling (Fable, 2026-07-13)

LANDING 774bf3d AUDITED — GO. Nine no-schema gates populated
(15 -> 24 declared, ok=True; selftest ALL PASS with and without
$ROOTDIR; py_compile clean). Both flagged deviations ratified:

- The board_selftest live-board fix is an UPGRADE, not merely a
  repair: the old leg validated the live BOARD against a stub cfg
  and was vacuous by construction ("all manifest-less ... no-op");
  the new leg reconciles every declared manifest against the REAL
  board_config via _load_config (safe with $ROOTDIR unset) and
  prints the declared count. The Implementer caught a latent
  vacuous-pass in OUR OWN acceptance machinery — the exact class the
  gate-truth campaign hunts; recorded as such.
- The three smoke-gate docstring rewords (bare generator .py tokens
  -> prose) follow the diagnostics-domain precedent.

DEPLOY_DATA SIGN-OFF — APPROVED WITH ONE AMENDMENT. The census claim
was mechanically re-verified: an untruncated uniq -c over ALL gate
configs finds exactly 22 occurrences of each of the six file names —
every batch-4 config names the identical six fixtures, and the val
role genuinely consumes the cs_8 files (role-named keys over
file-named paths is honest and ratified). Paths match the resolution
rule (train/val under --root/chains; the dataset under
$ROOTDIR/external_modules/data/<cosmolike_data_dir>).
cosmolike_data_dir itself is correctly omitted (a directory, not a
leaf member). THE AMENDMENT — a recorded blind spot: the .dataset
file is a POINTER; the sibling files it references (data vector,
covariance, mask, n(z)) are NOT in the hash surface, so a changed
mask leaves a green manifest. The _help entry gains one sentence
naming this boundary, and a board-listed HARDENING OPTION is queued
for the queue-5 window: measure the hashing cost there, then either
r2-expand the data directory or pin the referenced members. V1
lands as proposed; the blind spot is documented, not silent
(scope-and-blind-spot law).

CENSUS WORD-BOUNDARY RULING — APPROVED, WIDENED TO BOTH UNANCHORED
SITES. The phantom was live-reproduced by the Architect:
re.findall(r"[\w./-]+\.py", "cmd=[ctx.python, ...]") captures
"ctx.py". The hardened form [\w./-]+\.py(?!\w) captures nothing
there, still captures a real sentence-final mention
("... gates/run_board.py."), and stops the latent .pyc/.pyx false
captures. Apply (?!\w) to run_board.py:1054 (the general census)
AND :975 (the gates/checks-prefixed census — same .pyc class);
:1057's quoted driver form is bounded by quotes and stays. Pinned
acceptance (selftest legs): production-diagnostic validates (the
phantom is gone); a control keeps a real .py mention captured
including sentence-final; "x.pyc" yields nothing; "ctx.python"
yields nothing. This is the one authorized edit to the
validate_manifests machinery; the Implementer lands it with
production-diagnostic's population.

param-window-cuts' ctx.log reword (geometries.output.py -> "the
geometries output module") is approved at its population
(diagnostics-domain precedent).

Result: the next increment lands the deploy_data block + 15 gate
inputs= + the census hardening + production-diagnostic, completing
population 40/40 — queue 2 opens at that merge. The full-board Mac
validation and the review handoff remain the acceptance.

## 25M-07..13 adjudication (Fable, 2026-07-13): six confirmed, one retraction accepted; units 95 + 96 minted; three unit-13 amendments under the user's reasonableness ruling; one new advisory

Durable register at 652dad9. THE RETRACTION FIRST: 25M-07 was
retracted IN THE REGISTER before this adjudication — the chat copy
still filed it live, and the durable record overrode the chat
exactly as the durable-record rule intends. The executed
observation stands (the -2*s_step arm goes negative for s_step >
0.5); the conclusion did not follow — the user's h = 0.7 remark
exposed a notation collision (numerical step fraction vs
cosmological Hubble parameter), and no CAMB known-answer proved
signed formal arms wrong. Identifier retired, no code or gate
change owed; a future restriction needs an independent CAMB
known-answer, not the category assumption that every stencil arm
must be a realizable cosmology.

THE USER RULING (recorded in the register, ratified here):
covariance reasonableness is anchored to the Planck-LCDM fiducial
(example_yamls/cmb_covariance_lcdm.yaml: H0=67.36, As=2.1e-9,
ns=0.9660, omegabh2=0.02237, omegach2=0.1200, tau=0.0544,
mnu=0.06), which stays a byte-identical known-answer control in
EVERY covariance-validator change. 25M-08/11/12 are SCHEMA-TOTALITY
and catch-power claims, not claims that the shipped calculation is
wrong.

Verified and placed:

- 25M-08 CONFIRMED (compute_cmb_covariance.py:562: clpp *= (1.0 +
  eps); [1e-20, 2e-20] rounds every factor to exactly 1.0 — the
  executed run saw ONE unique relensing payload, returned exactly
  zero non-Gaussian covariance, and called it converged) -> UNIT 13
  AMENDMENT: ordered representable factors 1-2s < 1-s < 1 < 1+s <
  1+2s derived via nextafter, both-sign changed-value counts
  persisted, the false-green fixture refuses before relensing.
- 25M-11 CONFIRMED (noise_spectrum squares amplitudes independently
  :185-188; gaussian_blocks assembles TT/TE/EE with no PSD check;
  executed witness delta 1/10/1 -> joint eigenvalues
  [-2.86365772e-11, 1.43097071e-11, 2.86537506e-11] while every
  scalar check greens) -> UNIT 13 AMENDMENT: the 2x2 noise PSD
  inequality delta_te^2 <= delta_tt*delta_ee at the config boundary
  (representation-derived band), the per-ell signal+noise check,
  assembled-covariance PSD within one owned tolerance before
  publication; no clipping/loading/abs repairs.
- 25M-12 CONFIRMED (:188 exponentiates the beam factor; executed:
  ell 5000 + 60-arcmin beam -> inf; 32 arcmin -> noise finite
  ~4.11e162 while its covariance square overflows; savez writes
  without a postcompute finite check) -> UNIT 13 AMENDMENT: derive
  the largest beam exponent from resolved lmax and prove noise AND
  covariance products representable (named formulas, no guessed
  cap); postcompute finiteness on every array before output
  mutation, first key/ell/value named on failure.
- 25M-09 CONFIRMED (results.py:348 writes pce only when non-None;
  :617-625 rebuild infers composition from group PRESENCE;
  _read_native_bool("transfer_refined") only runs inside the
  if-transfer_base branch; executed witness: deleting f['pce'] from
  a valid NPCE artifact strict-loads and moves H0 by 2.7999344) ->
  NEW UNIT 96, the artifact composition mode: a native REQUIRED
  enum (plain/npce/transfer + refined as a separate native fact)
  persisted from the executed run; two-way required/forbidden group
  validation before model construction; mutual exclusion; absence
  NEVER means plain on schema v2; legacy presence-only artifacts
  refuse with a migration instruction; config_yaml corroborates but
  never substitutes for the enum. Interlocks units 3 (pair
  identity), 76 (recipe totality), 41 (resolved record). HIGH.
- 25M-10 CONFIRMED (stage_source prints unconditionally with no
  verbosity parameter; CMB geometry raw prints at experiment.py:
  3493/3504/3548; two worker failure paths bypass the quiet logger)
  -> NEW UNIT 95, the output-channel owner: ONE owned emit channel
  threaded through staging/geometry/workers; --quiet either honest
  or its help narrowed to what it controls; the house
  terminal-output rule (essential-only + debug switch) is the
  design frame. The captured-stdout witness becomes the gate leg.
- 25M-13 CONFIRMED (emul_baosn.py:126-127 unions every predictor
  name with only quantity/units checks; executed witness: w moved
  H(z=1) while D_M(z=1050) stayed bit-identical at
  13999.394531250002 — an independent flat-wCDM integral moved
  7.506%) -> UNIT 75 EXTENSION (the BAOSN half): equal canonical
  generator/dataset/domain binding, compatible sampled coordinates
  and domains, identical fixed facts (74/67's machinery), a
  parameter sampled by one half = refusal; requirement union never
  creates compatibility. NEW ADVISORY (joins the standing list):
  never serve a Hubble + D_M pair from different runs until the
  pair binding lands.

Sequencing: production repairs join the unit queue (13's three
amendments ride its existing 45M-01 slot; 96 beside 76's landing;
95 with the training/driver campaign; 75-BAOSN with the family
visits). Nothing preempts population completion -> the gauntlet.
The registry namespace: 25M-07 retired-tombstone, 08-13 live.

## 25M-16 (Red Team CONFIRMED, awaiting Architect adjudication): runtime-loaded Python is absent from populated gate manifests, so adapter edits retain a current PASS

The phase-3 scanner recognizes literal imports plus
`importlib.import_module` and `__import__` call sites
(`run_board.py:858-885`). It does not recognize Python executed through
`importlib.util.spec_from_file_location(...); loader.exec_module(...)`, a
Cobaya `python_path`, or executable `.py` source opened as ordinary data.
Those are all live mechanisms in populated gates, not theoretical dynamic
imports.

All four identity checks load their real adapter through
`spec_from_file_location` (`scalar_identity.py:271-297`,
`cmb_identity.py:552-572`, `bsn_identity.py:381-399`, and
`mps_identity.py:1185-1218`). All four family smokes execute the adapter
through Cobaya's declared `python_path`. Yet their populated manifests omit
`cobaya_theory`: the real derived manifests contain 31, 31, 32, and 31 members
for scalar/CMB/BAOSN/MPS identity and 34, 36, 34, and 37 for the corresponding
smokes, with zero `cobaya_theory/` members in every case. Executed through the
real manifest validator and resume-state path, all eight declarations validate
and a stored current PASS remains PASS when the omitted adapter is the only
changed executable. This directly falsifies `_gate_code_digest`'s claim at
`run_board.py:1419-1424` that an imported adapter necessarily stales the gate.

The same blind class exists without a module loader:
`artifact_readback.py:83-101` opens `scalar_train_emulator.py` as text and
asserts that it stamps `rescale='none'`. Its populated 23-member manifest
omits that driver. A driver edit can therefore invalidate what the check
would assert on rerun while its stored PASS stays current. This is one runtime
executable-dependency class, not a second finding.

The live inventory is broader because the `.py`-path census scans only the
gate-body source (`run_board.py:1047-1058`), not the check-script closure.
`family-first` omits the three family drivers it opens at
`family_first.py:82-101`; `generator-seed` has only three manifest members and
omits `compute_data_vectors/generator_core.py`, whose source supplies its
science assertions (`generator_seed.py:24-58`); and `board-selftest` omits
`gates/checks/finite_contract.py`, which it reads to certify the compile-lane
verdict (`board_selftest.py:266-277`). Each real declaration validates today,
and each named source member is absent. These are additional acceptance
fixtures for the same closure defect.

There is also one bounded ordinary-import hole. `gct_parity.py:43` uses the
bare sibling import `from gsv_bitwise_drift import ...` because
`gates/checks/` is placed on `sys.path`. `_module_to_repo_paths` rejects an
absolute import whose first component is not one of `_EXECUTABLE_DIRS`
(`run_board.py:804-815`), so the real static closure contains
`emulator/inference.py` but omits
`gates/checks/gsv_bitwise_drift.py`. `cobaya-adapter` therefore stays current
when its executed `train_save`/`tiny_config` helper changes. A repo-wide AST
census found this is the sole current bare sibling-check import; it is a
bounded resolver case, not permission for a path heuristic.

Required contract: derive one complete executable closure for every gate.
In addition to literal imports, it recognizes and resolves
`spec_from_file_location`/`exec_module`, Cobaya `python_path` plus component
module names, subprocess targets, and executable `.py` paths opened/read by
the check closure. A runtime-named site that cannot be resolved statically is
covered by one reviewed direct-root declaration and reconciled against the
site; it never disappears merely because the call is not named
`import_module`. Persist each adapter/driver as a manifest member of every
gate that executes or inspects it. The help text names any irreducible blind
spot honestly.

Board-selftest acceptance: census the current board and prove the executed
adapter is a member of each of the eight family identity/smoke manifests and
`scalar_train_emulator.py` is a member of `artifact-readback`; removing one
direct/derived root makes validation red; a temporary byte edit to each
runtime-loaded member changes the digest and makes `_resume_state` stale-code;
the unchanged current board validates. The catch-power mutation restores
today's two-call dynamic-import scanner and must green the false stored PASS.
The current `family-first`, `generator-seed`, and board-selftest source-read
fixtures are included so a repair limited to Cobaya adapters cannot pass.
The sibling-import leg proves `gsv_bitwise_drift.py` joins
`cobaya-adapter`'s closure; changing it makes that gate stale-code even if its
dependency reruns. Restoring the `_EXECUTABLE_DIRS`-only absolute resolver is
a required mutation failure.

`geo-paths` is the largest current source-read fixture: its check walks the
repository and reads every Python file to reject retired geometry imports
(`geo_paths.py:180-204`), but its populated manifest contains only 31 members
and omits executable files across `emulator/`, root drivers, generators, and
adapters. Its correct dependency surface is the exact Python census it reads,
with the same explicit exclusions—not merely the current designs/losses
roots. The repaired acceptance inventory includes this whole-scope gate.

## 25M-18 (Red Team CONFIRMED, awaiting Architect adjudication): a child file is accepted as coverage for an arbitrary dynamic-import tree

The waiver validator accepts a required cover when
`root == cover` **or `root.startswith(cover + "/")`**
(`run_board.py:1070-1081`). The second direction is backwards for a dynamic
class name: declaring one child such as `emulator/designs/blocks.py` does not
hash sibling model implementations under the required `emulator/designs`
tree. The shipped board selftest explicitly constructs that invalid manifest
and reports it as “designs root declared”
(`board_selftest.py:637-640`). The validator returns green.

This has a concrete executable consequence. `results.py:664-672` imports the
model class named by an artifact recipe. A manifest containing only
`blocks.py` can validate while omitting `designs/plain.py` and
`designs/ia.py`; a ResMLP or factored-IA artifact then executes unhashed code,
so an edit can leave a stored PASS current. Current populated gates happen to
declare the full trees, but the reusable validator and its own catch-power
test certify an unsafe declaration. This is distinct from 25M-16: that
finding discovers missing dependency mechanisms; this one accepts an
insufficient declaration for a dependency mechanism it already found.

The validator also treats the waiver's cover tuple as alternatives by using
one `any(...)`. That is false for `gates/checks/cli_strict.py`: its reviewed
tuple lists eight entry points, and the check loops over/imports all eight.
Declaring any one driver satisfies the site even though the other seven are
unhashed. The current populated gate declares them all only because manual
population review caught it; validation does not enforce that truth.

Required contract: a declared root covers a waiver requirement only when it
is equal to or an ancestor of every required cover. For the current recipe
waiver, require exact `emulator/designs` and `emulator/losses` unless the
waiver schema explicitly enumerates a reviewed finite subset. A child of the
cover never stands for its siblings. Census every waiver entry for a live
call site and its exact required covers so an obsolete entry cannot retain a
gratuitous tree. The waiver schema distinguishes required-all roots from true
alternatives; no current tuple is silently interpreted as alternatives.

Board-selftest legs: the full design/loss roots pass; the current
`blocks.py`-only fixture moves to a must-red mutation; two artifacts selecting
classes in `plain.py` and `ia.py` prove edits to either member change the
digest/stale the dependent gate; an ancestor-root control is accepted only
when its expansion contains the full required covers; reversing the
ancestor test restores today's false green and must fail.
For `cli_strict`, delete each of the eight entry-point roots one at a time and
require validation to red; an `any-cover` mutation must fail.

## 25M-19 (Red Team CONFIRMED, awaiting Architect adjudication): input manifests hash a different path than the gate executes

`RunContext.evaluate_yaml()` defines a relative `evaluate_yaml` as
repository-relative (`run_board.py:412-420`), and `cobaya-adapter` executes
that path. `_resolve_config_path`, used by the input manifest, instead tries
the same string process-CWD-first, then ROOTDIR-relative, then
`yaml_dir`-relative, and never tries repository-relative
(`run_board.py:1347-1371`). The manifest and consumer therefore have different
path owners.

Live reproduction with the shipped value
`gates/configs/cobaya-adapter-evaluate.yaml`: launched from the repository,
the manifest records the file and digest; launched from an unrelated temporary
directory, it records `{path: None, sha256: None}` and still passes
`validate_manifests`, while `RunContext.evaluate_yaml()` continues to execute
the real repository file. A first PASS from that directory authenticates no
YAML bytes, so later edits do not stale it. Conversely, merely changing the
launch directory can change the digest without changing the executed file.
This violates the runner's any-working-directory contract and is a concrete
false-green plus false-stale pair.

The same mismatch now reaches all 15 batch-4 driver manifests. Their
`gate_configs.*` consumers execute `yaml_dir/value`, while the generic
manifest resolver first accepts a launch-CWD collision (or records no member
outside the repository) before trying the owner-specific base. The
rootdir-relative `_CS_DEPLOY_DATA` members are consistent; the bespoke gate
YAMLs are not. Acceptance therefore covers both `evaluate_yaml` and at least
one populated batch-4 `gate_configs.*` input rather than repairing a single
key by special case.

Required contract: each input-key namespace has one canonical resolver shared
by its consumer and manifest writer. `evaluate_yaml` resolves relative to the
repository; `gate_configs.*` resolves through `yaml_dir`; `gate_data.*` uses
its declared data owner. No generic process-CWD candidate precedes those
rules. Manifest validation requires the canonical file to exist and refuses a
resolved string with `{path: None, sha256: None}`.

Board-selftest legs: from two unrelated temporary working directories, the
same config yields the identical absolute member path and digest; the path
executed by `RunContext` equals the path hashed by the manifest; a colliding
file in the launch directory is ignored; editing the canonical YAML changes
the input digest and makes resume stale-input; a missing canonical file reds
validation. Restoring `Path(value)` as the first candidate or omitting the
repository candidate must fail the catch-power legs.

## 25M-20 (Red Team CONFIRMED, immediate unit-4 reopen): resume trusts a downstream PASS before checking whether its dependency is current

`run_selection` computes the downstream gate's resume state and returns early
on a current stored PASS (`run_board.py:1862-1871`). Only after that early
return does it check whether every prerequisite is a current PASS
(`:1880-1891`). Dependency currency is therefore absent from the reusable-PASS
predicate, despite the existing unit-4 contract explicitly requiring it.

Live reproduction through the real public `main`/`run_selection` path with
the shipped board-selftest helpers: the prerequisite has a stored PASS whose
code digest is stale (`deadbeef`), while the downstream gate has a current
stored PASS. Selecting the downstream gate prints
`[skip] downstream: already PASS`, executes zero gate bodies, and exits 0.
The current selftest covers a resumed child only when the prerequisite is
current (`board_selftest.py:212-216`), so it encodes no counterexample. In the
real board, `cobaya-adapter` can remain green while `save-rebuild-drift` is
stale. Even a full run can rerun the prerequisite and then skip the child,
never re-proving it against the newly produced artifact.

Required contract: dependency currency is part of a gate's reusable-PASS
state. Check dependencies before the resume return, or make `_resume_state`
dependency-aware. A stale-code, stale-input, stale-log, pre-manifest, failed,
or skipped prerequisite makes the downstream verdict non-green and the
requested command nonzero. When a prerequisite reruns, any child whose proof
consumes its output reruns too; persist the dependency verdict/digest or
artifact identity needed to bind that currency. No manual `--force-rerun`
instruction substitutes for this ordering.

CPU board-selftest legs: a current child paired with each stale prerequisite
state executes no false resume and returns nonzero; FAIL/SKIP/RUNNING
prerequisites do the same; a full sequence reruns the prerequisite and then
the dependent child; an unchanged current pair still resumes; an independent
gate remains resumable. The mutation restoring today's resume-before-deps
ordering must reproduce return code 0 with zero calls and therefore red.

## 25M-21 (Red Team CONFIRMED, awaiting Architect adjudication): documentation inside board_config invalidates every gate input digest

`_gate_input_digest` hashes the complete effective config after excluding only
`debug` and derived `rootdir_source` (`run_board.py:1471-1492`). It therefore
includes top-level `_help`, whose nested strings are prose for humans and are
never consumed by gate execution. A documentation-only change can mark every
populated gate stale-input and cause expensive GPU reruns, contradicting
unit 4's existing requirement that documentation-only changes leave evidence
currency unchanged.

Live digest reproduction on populated `stage-ram`: appending the words
`prose only` to `_help.what` changes the input digest from prefix `471a30...`
to `eed28b...`; changing `debug` correctly leaves it unchanged. This is an
input-identity false red, not a security concern. It wastes workstation time
and makes the board claim the scientific inputs changed when only explanatory
text did.

Required contract: one named canonical projection contains exactly the
configuration fields that can affect execution or science. `_help` and any
future documentation/annotation namespace are excluded; every consumed value
remains included. The same projection drives the input digest and the logged
resolved-execution record so the displayed facts and authenticated facts do
not drift.

CPU board-selftest legs: mutate every `_help` leaf independently and prove all
gate input digests unchanged; mutate actual `driver_root`, `driver_fileroot`,
`yaml_dir`, `evaluate_yaml`, golden-base, gate-config, and gate-data values and
prove the relevant digest changes; `debug` remains excluded. A mutation that
hashes the raw config must reproduce the prose-only stale-input and red.

## 25M-22 (Red Team CONFIRMED; Architect-CONFIRMED 2026-07-13): `saved_emulator_root` is a documented control that no code reads

`board_config.json` promises that a non-null `saved_emulator_root` selects an
already-saved schema-v2 artifact for `save-rebuild-drift` and
`cobaya-adapter`, while null makes the gate train its own tiny emulator. An
untruncated Python census finds zero reads of that key. `gsv_bitwise_drift`
reads only `rootdir` and `driver_root` (`:73-104`) and then unconditionally
trains/persists `chains/gates_emul_evaluate` (`:301-309`); the evaluate YAML
loads that fixed product. A user can set a valid different artifact and the
gate silently ignores it.

The dead control is nevertheless included in every effective-config digest,
so changing it makes gates stale-input while changing no execution. This is a
research-workflow correctness defect, not a security issue: the board says it
tested the selected emulator but always tests its internally trained one.

Recommended contract: remove the unused key and its help text; the gate's tiny
artifact is intentionally owned by `save-rebuild-drift`, and an external-root
mode adds no current evidence. If the Architect instead keeps the capability,
make it a typed, path-validated explicit mode, bind both `.h5` and `.emul` as
input-manifest members, and route every consumer to that root. It may never
remain an ignored digest-only value.

CPU acceptance: a config census proves every non-documentation public key has
an execution reader; after removal, changing no dead key can stale gates. If
the keep-capability branch is chosen, valid and missing sentinel roots prove
selection/refusal respectively, the executed root equals the hashed pair, and
the old unconditional-training mutation must red.

Architect adjudication: CONFIRMED and the recommended removal branch is
binding. Remove `saved_emulator_root` and its help text; do not create an
external-artifact mode merely to preserve a dead option. The config census is
promoted from one repair leg to a permanent board-selftest invariant: every
public, non-documentation board-config key has an execution reader. This joins
25M-16/18/19/21 in the single manifest-hardening landing before queue 2, so
the landing makes one deliberate digest transition.
## 25M-14/15 + DIDACTICS-93 adjudication (Fable, 2026-07-13): both defects confirmed, both Torch gates commissioned; the naming ruling ratified

Durable register at efddf98 (latest main merged as 97e9acb).

- 25M-14 CONFIRMED (the width-one transformer demotion):
  plain.py:866-870 bounds n_tokens by 2..total, ADMITTING T == total
  (one scalar per token); blocks.py:602-604's only constructor guard
  is dim % n_heads == 0, which width 1 + n_heads 1 passes; both
  TRFBlock branches are pre-norm LayerNorm(dim) (:609, :621) — at
  dim = 1 the normalized value is identically zero before the
  learned affine, so with film: false every branch output is a
  learned constant and the ResTRF correction t - t0 is
  input-independent FOR EVERY POSSIBLE TRAINED WEIGHT SET, while the
  ResMLP trunk still learns and hides the demotion from aggregate
  bars. -> AMENDMENT to unit 29 (model-block value schema,
  models-and-designs.md): the active-model validator derives real
  token widths (the padded max_bin) from the geometry BEFORE
  construction and refuses max token width < 2 for BOTH plain and
  factored TRF paths, naming output length, token count, resolved
  width, and the LayerNorm degeneracy; no padding or artificial
  embedding silently repairs a requested design. TORCH GATE
  COMMISSIONED (board-listed, workstation): the five filed legs —
  the N=4/n_tokens=4/n_heads=1/film-false refusal before
  construction; the bypassed-validation behavioral witness
  (deterministic nonzero head weights, identical corrections for
  two distinct t0 rows, zero correction Jacobian w.r.t. t0);
  adjacent n_tokens=3 constructs input-dependent; plain and
  factored share the verdict; the divisibility-only mutation greens
  construction but reds the behavioral witness.
- 25M-15 CONFIRMED (the packed-target planner undercount):
  batching.py:124-127's own comment asserts the target "matches the
  output's shape/dtype, so another out_bytes" — false for
  PCERatioChi2 / TransferDiagChi2 (2*n_keep per row) and factored
  TransferChi2 ((n_templates+1)*n_keep); the resident path resolves
  tgt_dim (:276-289) but BOTH streaming calls (:373, :387) invoke
  batches_per_load without it, and :214's max(1, ...) converts a
  resident-plus-one-batch deficit into permission. The 84-byte
  witness arithmetic (3*(14-7)*4) is exact. Distinct from 45M-84
  (permanent host copy vs transient device memory), as filed. ->
  AMENDMENT to the live-resource-sizing unit (second-wave local
  list 5, training-stack.md owner): ONE owner computes input +
  model-output + ACTUAL-target bytes with width/dtype threaded from
  the loader boundary that stages the target; ordinary
  target_dim == out_dim stays byte-identical; the planner report
  names each term; a deficit REFUSES with required/available/terms
  — never max(1, ...). TORCH LEGS COMMISSIONED (board-listed):
  the pure 84-byte and two-batches-vs-one boundary legs (CPU); one
  real packed-target streaming integration leg + ordinary control
  (workstation); both mutations (restore 2*out_bytes; restore
  max(1, ...)) red.
- DIDACTICS-93 RATIFIED (the user's naming ruling, recorded in
  conventions-and-workflow.md as binding): cosmological h is
  RESERVED for H0/100 program-wide; the covariance numerical step
  becomes step_frac in Python and s_step in prose; the untruncated
  census also disambiguates hidden-state, hardness, horizon, and
  local-Hubble uses. This is an AST-CHANGING rename, not doc-only:
  it lands as a bounded rename increment whose acceptance is
  byte-identical outputs (the Planck-LCDM control among them) and
  the untruncated census; arithmetic and persisted values
  unchanged. The 25M-07 retraction was CAUSED by this collision —
  the ruling closes the class, not just the instance.

Both commissioned gates join the queue-5 workstation exhibit
family. Sequencing unchanged: population completion -> the gauntlet
first; the two amendments join their units' queue slots.

## 1b phase-3 population COMPLETE — 40/40 declared (Opus, 2026-07-13)

The manifest population is finished; queue 2 opens at the merge. Three
branch commits carry it, each Mac-validated (validate_manifests ok=True,
board-selftest ALL PASS with and without $ROOTDIR, py_compile clean):

- 774bf3d — nine no-schema gates (15 -> 24): weight-decay-census (code=(),
  its in-process check never reaches the waived model-recipe surface),
  save-rebuild-drift + cobaya-adapter (designs+losses; adapter inputs
  evaluate_yaml), cli-strict (the reviewed _DYNAMIC_IMPORT_WAIVERS entry ->
  the eight entry-point drivers + designs+losses), family-first,
  scalar-smoke, and cmb/bsn/mps-smoke (designs+losses + their
  compute_data_vectors generators as declared subprocess-dep roots;
  bare-generator .py docstring tokens reworded per the diagnostics-domain
  precedent). The board_selftest live-board leg was upgraded from a vacuous
  stub-cfg no-op to a real board_config reconciliation (cobaya-adapter's
  evaluate_yaml was the first real input key that exposed it).

- 9154e2e — batch 4 (24 -> 39): the signed-off deploy_data block (one shared
  top-level block, six semantic fixture keys, values derived from the shipped
  configs -- census: all 22 configs name the identical six, no path twice)
  plus its _help; two board.py constants (_CS_TRAIN_CODE, _CS_DEPLOY_DATA) so
  gates sharing a fixture share the key; 15 driver-gate manifests;
  param-window-cuts' ctx.log reworded.

- 2108c89 — census (?!\w) hardening + production-diagnostic (39 -> 40): the one
  authorized validate_manifests edit, at both unanchored census sites (:975,
  :1054; the quoted driver= form stays), so cmd=[ctx.python, ...] no longer
  reads as a phantom ctx.py and a .pyc never false-matches while a real
  sentence-final path is still caught; production-diagnostic populates; the
  deploy_data _help gains the .dataset-pointer blind spot (siblings outside the
  hash surface -- the queued queue-5 hardening option); two pinned selftest
  legs (the first immediately caught a standalone ctx.py in its own fixture
  comment). The populated reruns are the first queue-5 exhibit (deploy_data
  resolves to strings on the Mac, hashes on the box).

Next spine increments (fully specced via parallel subagent investigation):
D3 (unit-28 smoke-fixture disjointness -- distinct recorded seeds train 1234 /
val 5678, a zero-overlap refusal before training, a same-seed mutation arm, in
cmb/bsn/mps_smoke.py) and D4 (gate-truth 46/47/48 -- selected_subset_equal with
both-rc==0 and a nonempty minimum count; the vacuous all-zero P_lin positivity
fix in mps_smoke; the unchecked-PDF assertion in gate_diag).

## 25M-16..21 adjudication (Fable, 2026-07-13): the board-machinery batch — unit-4 reopened NOW, a 1b hardening increment minted, queue 2 resequenced behind it; an Architect audit lesson

Durable register at a3b19f4. The chat relay carried 16..20; the
register additionally holds 25M-21 (complete, with an executed
digest witness) — adjudicated with the batch, the register being
the source of record. Scope note recorded: scientific correctness,
MCMC reproducibility, and test truth only.

ALL SIX CONFIRMED (anchors verified in code):

- 25M-20, THE IMMEDIATE UNIT-4 REOPEN: run_selection's resume
  early-continue on a current stored PASS runs BEFORE the
  dependency loop, so _dep_current_pass is never consulted for a
  resumed gate — the existing "dependencies accept only current
  PASS" clause is bypassed by resume. Live witness: stale-code
  prerequisite + current child -> "[skip] ... already PASS", zero
  bodies, exit 0. RULING: dependency currency joins the
  reusable-PASS predicate (check deps before the resume return or
  make _resume_state dependency-aware); every stale/FAIL/SKIP/
  RUNNING/pre-manifest prerequisite state makes the child
  non-green; a rerun prerequisite reruns its artifact-consuming
  children. The full state matrix + the resume-before-deps mutation
  land as selftest legs. BINDING TIMING: this fix lands with or
  immediately after the population-completing increment — the
  40/40 handoff is not accepted green without it.
- 25M-16 (closure truth): _dynamic_import_sites recognizes only
  importlib.import_module/__import__; the four identity gates load
  adapters via spec_from_file_location (scalar_identity.py:294,
  cmb_identity.py:569 and :1043's cov oracle, bsn/mps siblings),
  the four smokes load them via Cobaya python_path, and NONE of the
  eight manifests contains a cobaya_theory member (red-team
  executed counts: 31/31/32/31 and 34/36/34/37). artifact-readback,
  family-first, generator-seed, board-selftest, and geo-paths read
  executable source AS DATA that their manifests omit; the .py
  census scans gate-body source only, not the check closure.
  gct_parity.py:43's bare sibling import is dropped by
  _module_to_repo_paths' _EXECUTABLE_DIRS filter (the sole such
  import, per their repo-wide census). RULING: one whole-check
  executable closure (runtime loaders, Cobaya python_path +
  component names, subprocess targets, source-opened-as-data, bare
  sibling imports); unresolvable runtime-named sites get reviewed
  direct-root declarations reconciled against the site; the
  acceptance battery as filed, including the two-call-scanner
  restoration mutation and the geo-paths whole-scope fixture.
- 25M-18 (waiver direction): run_board.py:1076-1077 accepts
  r.startswith(cover + "/") — a CHILD satisfies a tree requirement,
  and board_selftest.py:637-640 blesses the blocks.py-only fixture
  as "designs root declared". Current populated gates happen to
  declare full trees (which is why the population audits passed:
  the declarations are sufficient, the VALIDATOR is permissive).
  RULING: a declared root covers a waiver only when equal to or an
  ANCESTOR of every required cover; the blessing fixture becomes a
  must-red mutation; every waiver entry censused against its live
  call site.
- 25M-19 (resolution ownership): the evaluate_yaml consumer
  resolves repo-relative (:412-420) while _resolve_config_path
  tries Path(value) process-CWD-FIRST and never tries the repo
  (:1355-1360); the same CWD-first generic reaches all 15 batch-4
  gate_configs manifests. Live witness: from an unrelated cwd the
  manifest records {path: None, sha256: None} and validates while
  the gate executes the real file. RULING: one owner-specific
  resolver per input namespace, shared by consumer and manifest
  writer (evaluate_yaml -> repo; gate_configs.* -> yaml_dir;
  gate_data.* -> its data owner; NO generic CWD candidate);
  REPO-OWNED inputs must resolve and refuse None-sha — rider r3's
  resolve-not-exist stays for deploy_data ONLY (machine-dependent
  by design). Two-cwd identity, collision-ignored, executed-path ==
  hashed-path legs as filed.
- 25M-21 (register-only filing; input-digest scope):
  _gate_input_digest hashes the effective config excluding only
  debug/rootdir_source, so a _help prose edit stales EVERY declared
  gate (witness: 471a30.. -> eed28b..) — a false-red contradicting
  unit 4's documentation-currency requirement. RULING: one named
  canonical projection of execution-relevant fields; _help and any
  documentation namespace excluded; a projection-change leg proves
  prose edits leave digests fixed while value edits stale.
- 25M-17 (const_mask presence inference): grid2d state() omits the
  mask when None, from_state treats absence as None (:119-120),
  decode clamps only when non-None — deleting dv_geometry/
  const_mask from a valid pinned artifact strict-loads and serves
  1.25 where the intact artifact serves 1.00 (their executed
  boost/none witness). REOPENS unit 63's "pre-pin absence stays
  legal" precision clause: current saves ALWAYS persist the mask
  (explicit all-false when unpinned); the required geometry-state
  member set is validated before from_state; anonymous legacy
  absence refuses with a migration instruction; no version integer
  (the mask itself is the fact). Interlocks unit 96; lands with the
  MPS visit beside 96's group-validation legs. The
  PRESENCE-INFERENCE CENSUS I requested is CLOSED: beyond 96's
  composition groups, const_mask was the only silent scientific
  reinterpretation site.

STRUCTURE: 16 + 18 + 19 + 21 form THE 1B HARDENING INCREMENT — one
machinery batch landing AFTER population completes and BEFORE
queue 2. SEQUENCING CHANGE (explicit): queue 2 (the evidence
rollout) now opens after the hardening increment, not at population
completion — the registry rewrite must mint aids from truthful
machinery. Expected and correct side effect: the completed closure
repair stales stored PASSes whose manifests omitted real
dependencies; those reruns are the system working.

ARCHITECT AUDIT LESSON (recorded against my own passes): the
population audits verified declarations THROUGH the machinery now
shown permissive — validate_manifests green plus my geo-paths
closure re-derivation used the same scanner and so inherited its
blind spots. Declarations-vs-machinery consistency is not machinery
truth. Adopted: any audit of a validation system includes at least
one adversarial probe AGAINST the machinery (a runtime-loader
fixture, a wrong-direction waiver, a cwd flip) — audit the
validator, not only through it.

## 25M-23 (Red Team CONFIRMED, awaiting Architect adjudication): the board-listed finite-contract check has an unresolved helper name and crashes late in Part H

`gates/checks/finite_contract.py:87` imports `CosmolikeChi2` and
`_chi2_neg_band` from `emulator.losses.core`, but
`check_chi2_band_dtype_provenance` calls `_chi2_domain` at :1255. There is no
module definition and no import of that name. The call is not dead: `main`
invokes this function at :1389 after the earlier finite-contract sections.
`py_compile` succeeds because Python resolves global names only when the line
executes, so the repository's compile acceptance cannot catch this class.

The failure is deterministic on every Torch environment: once the check
reaches the dtype-provenance mutation arm it raises `NameError`, aborting the
script before Part J and before the check prints its final PASS/FAIL summary.
The board wrapper then correctly sees a nonzero check process, but the
workstation run is wasted and the gate can never certify the landed finite
contract. A `symtable` scan over `emulator/`, `compute_data_vectors/`,
`cobaya_theory/`, and `gates/` found this as the only unresolved production
name after excluding Python's injected `__file__` and the two deliberate
gate-mutation sentinels (`_does_not_exist_ce` and the source-string `_spawn`).

Required contract: gate-only repair; import `_chi2_domain` from the same
single owner as `_chi2_neg_band` (or call an equally direct shared owner), and
do not change producer arithmetic to make the fixture pass. Run the complete
board-listed `finite-contract` check on the Torch workstation and require it
to reach Part J plus its final summary. The gate log must contain the
dtype-provenance mutation PASS and the optimizer-schema section, proving the
script did not merely exit before this line. Add a mechanical binding leg for
the touched helper (with explicit allowlisting of deliberate undefined-name
mutation fixtures); `py_compile` remains necessary but is not sufficient.
Deleting the repaired binding must reproduce the late `NameError` and red the
gate. No production-code change is requested by 25M-23.

## 25M-24 (Red Team CONFIRMED, awaiting Architect adjudication): action-mode early returns bypass the board's selector-truth contract

`build_parser` makes `--list` and `--check` independent booleans and makes
only `--gate` / `--tier` / `--from` mutually exclusive
(`gates/run_board.py:2004-2046`).  In `main`, the `--list` return at
:2097-2100 and the `--check` return at :2102-2104 occur before the unknown-id
checks and `select_gates` at :2108-2112.  The board therefore enforces its
selector contract only in run mode, although the BRD-A evidence anchor and
`board_selftest.py` state without that qualification that an unknown
`--gate`, `--from`, or `--force-rerun` id is a nonzero usage error.

This is live through the real command line on current HEAD:

```
python3 gates/run_board.py --list --gate definitely-not-a-gate
python3 gates/run_board.py --list --from definitely-not-a-gate
python3 gates/run_board.py --list --force-rerun definitely-not-a-gate
```

All three print the ordinary 51-line full-board listing, print no error, and
exit 0.  `--list --check` also exits 0 and silently chooses the listing; on a
workstation where preflight is green, `--check --gate definitely-not-a-gate`
would analogously return the preflight verdict without validating the named
id.  A pasted command can therefore contain a misspelled requested gate or
rerun while reporting success and executing none of the named work.

Required contract: define one unambiguous action before any early return.
`--list` and `--check` are mutually exclusive with each other and with
run-only selectors, force controls, and `--dry-run` (recommended), or every
named id is
validated before either action and the supported composition is documented;
there must be no silent precedence.  The parser/error names the conflicting
flags and exits nonzero before preflight or a gate body.  Plain `--list` stays
the portable no-Torch listing and plain `--check` stays preflight-only.
Board-selftest drives the real `main` boundary for list+unknown gate,
list+unknown from, list+unknown force-rerun, check+unknown from, list+dry-run,
and list+check; every ambiguous/unknown command is nonzero, while the two plain
controls retain their existing behavior.  A mutation that restores either
early return above argument/selector validation must red.  This is a
gate-harness-only repair; no scientific producer changes.

## 25M-25 (Red Team CONFIRMED, awaiting Architect adjudication): `--from` silently omits an optional gate when that gate is the named start

`select_gates` documents `--from` as “starts the board at a gate” and first
locates the named registry index, but its append condition is
`index >= start and not gate.optional` (`gates/run_board.py:1720-1732`).
Consequently an explicitly named optional start is discarded by the same
default filter intended for optional gates the user did not request.

The current board has a real witness.  Calling the production selector with
`from_gate="triangle-shading"` returns 20 gates whose first id is
`joint-training`; `triangle-shading` is absent.  The control
`--gate triangle-shading` includes it.  Thus a workstation command
`run_board.py --from triangle-shading` can finish green after running every
later non-optional gate while never executing the gate named as its starting
point.

Required contract: an optional gate named as the `--from` start is explicit
user intent and is included as the first selected gate.  Later optional gates
may remain excluded, preserving the default-board policy; ordinary
non-optional starts and `--gate` behavior stay byte-for-byte equivalent in
selection/order.  Board-selftest uses a small board containing an optional
start, a later non-optional gate, and a later optional gate: the named start
plus later non-optionals are returned in board order, the unrelated later
optional is absent, and an unknown start still raises.  A mutation restoring
the unconditional `not gate.optional` filter must red.  This is also
gate-harness-only.

## Population 40/40 pre-merge audit (Fable, 2026-07-13): GO — and the first audit under the new law catches an any-vs-all waiver hole

The four commits (774bf3d / 9154e2e / 2108c89 / dd4bf85) audited;
all code changes scoped to gates/ + notes. Independent verification
on the cocoa interpreter (not a re-read of the Implementer's
numbers): board-selftest ALL PASS rerun by the Architect;
validate_manifests re-executed — 40 of 40 declared, ok=True, 0
errors; deploy_data byte-matches the signed-off block (six distinct
paths, role-named keys) with the dataset-pointer blind-spot
sentence present in _help; the (?!\w) hardening present at BOTH
unanchored census sites (:978, :1060) with the quoted driver= form
untouched; spot-checked manifests exact (berhu-loss and
production-diagnostic: one gate_configs key + the six deploy_data
keys; production-diagnostic's code roots include the driver +
designs + losses). The Implementer's report that the first pinned
census leg caught a ctx.py phantom in its own fixture comment is
credited — the leg earned its keep on day one.

THE PROBE FINDING (the first audit run under "probe against the
machinery"): stripping emulator/designs from cobaya-adapter's
declaration IN MEMORY still validates ok=True. Cause: the waiver
coverage clause is any()-quantified over (roots x covers) — ANY
declared root matching ANY cover clears the whole waiver, so a
multi-cover waiver (designs AND losses) is satisfied by either
alone. This is the SECOND permissiveness direction in the clause
25M-18 indicted (child-satisfies-parent was the first). NOT a
landing blocker — every shipped declaration carries both roots — but
it is now a witnessed ADDENDUM to the 25M-18 contract in the 1b
hardening increment: every required cover must be covered by some
declared root (all-quantified over covers); the strip-designs-keep-
losses fixture becomes a must-red selftest leg beside the
child-as-cover mutation.

VERDICT: GO for the merge. The 1b population is COMPLETE. Standing
conditions unchanged and restated: the 40/40 landing is accepted
GREEN only with 25M-20's dependency-currency fix landing
immediately after (acknowledged by the Implementer); queue 2 opens
after the 1b hardening increment (16/18+addendum/19/21), not at
this merge; the populated reruns remain the first queue-5 exhibit.
D3/D4 prep noted and consistent with the spine (D4's full roster:
46/47/48/52/53/63 + 61-finiteness + 66).

## 25M-22 adjudication + the 16..21 refile reconciliation (Fable, 2026-07-13)

The seven-finding relay is the 16..21 batch already adjudicated
(rulings stand unchanged; counted once) plus ONE new finding and
ONE new clause:

- 25M-18's NEW CLAUSE (tuples-as-alternatives) CONVERGES with my
  40/40-audit probe: the red team observed that cli_strict's waiver
  lists eight entry points, ALL imported, while the any() quantifier
  treats them as alternatives; my in-memory strip-designs-keep-
  losses probe witnessed the same quantifier hole on the recipe
  waiver. ONE merged contract in the 1b hardening increment:
  coverage is all-quantified over required covers AND over
  required-all tuple members; must-red fixtures = child-as-cover,
  strip-one-of-two-covers, any-one-of-eight-entry-points.
- 25M-22 CONFIRMED (verified by untruncated grep: the key exists
  ONLY at board_config.json:8 (_help) and :20 (null value); zero
  Python reads; gsv_bitwise_drift reads rootdir/driver_root and
  unconditionally trains chains/gates_emul_evaluate at :305-307).
  The board promises it tests a selected emulator while always
  testing its internally trained one, and the dead value still
  enters every input digest. RULING: REMOVE — the recommended
  option. A documented control no code reads is a standing lie
  (never-trust-defaults applied to config surfaces); the tiny
  self-trained fixture is deliberate gate ownership; an
  external-root mode adds no current evidence. Remove the key AND
  its _help entry; the Implementer presents the resulting
  board_config block paste-ready. LANDING: with the 1b hardening
  increment beside 25M-21's digest projection, so the digest
  transition happens ONCE. PERMANENT ACCEPTANCE (class, not
  instance): the config census proving every non-documentation
  public key has an execution reader becomes a standing selftest
  leg — no future dead key can accumulate.

## 25M-20 DONE — resume no longer bypasses dependency currency (Opus, 2026-07-13)

Commit cc85aa9 (the immediate unit-4 reopen, landed right after population
completion per the binding timing). run_selection's resume early-continue on a
current stored PASS ran before the dependency loop, so a gate whose own PASS was
current resume-skipped as green with a stale / failed / interrupted prerequisite.
Fix: dependency currency joins the reusable-PASS predicate — a gate resumes only
when its own PASS is current AND every dependency is a current PASS that was not
itself rerun this run (a `reran` set tracks this-run executions; a reran
prerequisite reruns its artifact-consuming child). board_selftest gains
check_dependency_currency driving the real run_selection over the state matrix:
both-current-PASS resumes (0/0 bodies); a stale-code prerequisite reruns its
current-PASS child (the resume-before-deps mutation, red-capable — child bodies 0
without the fix); an interrupted RUNNING prerequisite reruns the child; a FAILED
prerequisite skip-deps it (nonzero exit). board-selftest ALL PASS; py_compile
clean.

Next: the 1b hardening increment (25M-16 whole-check closure, 25M-18 waiver
direction + the all-quantified-coverage addendum, 25M-19 owner-specific
resolvers, 25M-21 digest projection) — one machinery batch before queue 2, with
the expected-and-correct side effect that the closure repair stales stored
PASSes whose manifests omitted real dependencies. Then D3 and D4 per the specs.

## 25M-20 pre-merge audit (Fable, 2026-07-13): GO — the 40/40 population is accepted GREEN; one topology rider joins the hardening increment

cc85aa9 audited (run_board.py +29/-2, board_selftest.py +55). The
implementation matches the ruling exactly: the resume predicate now
requires own-PASS currency AND every dependency a current PASS AND
no dependency in the this-run reran set; a current-PASS child with
reran prerequisites prints the honest rerun line and executes; a
gate joins reran only when its body actually runs (a SKIP-DEP'd
gate never does, so skip cascades stay non-green). The reran set is
TRANSITIVE by construction (A stale -> A reruns -> B reruns -> C
reruns). An unselected stale prerequisite denies the child's resume
and falls to the dependency-skip path: non-green, nonzero — the
ruling's semantics precisely. Independent verification: the
dependency-currency selftest legs rerun green on the cocoa
interpreter (skip-cascade matrix, FAIL prerequisite, rc 1).

ONE RIDER (Architect probe, non-blocking): the fix relies on BOARD
listing dependencies before their children — TRUE today (probe: for
every gate, each dep's BOARD index precedes the gate's; OK) but
unenforced. The topological assertion becomes a permanent selftest
leg in the 1b hardening increment: authoring order is now a
correctness invariant of the resume machinery, so the board asserts
it.

VERDICT: GO. Per the binding timing, the 40/40 population landing
is hereby ACCEPTED GREEN. The 1b hardening increment is GO as
scoped WITH TWO SCOPE CORRECTIONS: (1) 25M-22's key removal + the
every-key-has-a-reader config census leg belong to this increment
(beside 21's projection — one digest transition), absent from the
Implementer's scope list; (2) the topology-assertion leg above.
Increment contents, final: 16 closure truth; 18 merged
all-quantified coverage (three must-red fixtures); 19 owner
resolvers; 21 digest projection; 22 removal + census leg; the
topology leg. Queue 2 opens at that increment's merge.

## 25M-24/25 adjudication (Fable, 2026-07-13): both selector-truth defects CONFIRMED — the hardening increment gains items 7 and 8

Durable register at 97e8802. Verified at source:

- 25M-24 CONFIRMED: main() returns for --list (rc 0) and --check
  BEFORE the selector/force-rerun validation — whose own adjacent
  comment promises "an unknown id is a usage error with a nonzero
  exit, never a warning followed by a successful run". The promise
  exists in the code and is bypassed for action modes: --list with
  an unknown --gate/--from/--force-rerun prints the full board and
  exits 0; --list --check silently prefers listing. RULING as
  filed: action modes are standalone and mutually exclusive;
  ignored or incompatible run controls are a usage error (exit 2);
  the legs run the REAL main() (unknown-id-under---list nonzero;
  --list --check nonzero; each restoration mutation red).
- 25M-25 CONFIRMED: the --from branch indexes the FULL board
  (optional gates included) for the start position, then filters
  `not gate.optional` in the slice — an explicitly named OPTIONAL
  start is silently dropped while everything after it is selected
  (--from triangle-shading -> 20 gates beginning at joint-training,
  the named gate absent). RULING as filed: an explicitly named
  optional start is INCLUDED and first; unrelated later optional
  gates remain excluded; the selector fixture pins the exact
  expected id list and the restoration mutation must red.

Placement: both amend the unit-4 harness/CLI-truth surface (the
45M-77 selector lineage). They join THE 1B HARDENING INCREMENT as
items 7 and 8 (same file, one machinery landing, selector truth is
prerequisite to trusting queue-5 CLI runs). If the Implementer has
already frozen the increment scope mid-execution, they land as an
immediate sibling commit reviewed in the same audit — one landing
or two commits, ONE audit.

Recorded honestly: the red team's observation that board-selftest
remains green against both defects is the point — the current legs
exercise the selector function, not main()'s action-mode ordering.
Consistent with the standing lesson; the new legs run the real
main().

Red Team coverage rider on 25M-24 (2026-07-13, awaiting Architect
adjudication): the ignored-control class also exists inside an ordinary run
selection.  The real command

```
python3 gates/run_board.py --dry-run --gate family-first \
  --force-rerun triangle-shading
```

validates both ids, prints `selected 1 gate(s): family-first`, never mentions
or runs `triangle-shading`, and exits 0.  `forced` may contain ids outside
`selection`, while `run_selection` loops only over `selection`.  This does not
need a new repair unit: extend 25M-24's “ignored controls are usage errors”
contract so every named `--force-rerun` id must belong to the resolved
selection (or the command refuses the mismatch with exit 2 and names both
surfaces).  A real-main dry-run leg and a non-dry fake-gate leg must restore
the current subset mismatch as their catch-power mutation.  Do not silently
union the forced gate into the selection: `--gate` / `--tier` / `--from` own
what is tested; force changes resume behavior only.

## 25M-26 (Red Team CONFIRMED, awaiting Architect adjudication): dependency identity is lost between board invocations, so an old child PASS survives a newly produced prerequisite artifact

The 25M-20 landing fixes dependency currency only inside one call to
`run_selection`.  Its `reran = set()` is allocated afresh for each invocation
(`gates/run_board.py:1852-1856`), and a successful child's status record stores
its own code/input/log/attempt identity but no identity of the prerequisite
result it consumed (`:1974-1985`).  On the next process, `_dep_current_pass`
asks only whether the prerequisite is a current PASS *now*.  It cannot tell
that this is a different PASS from the one against which the child was proved.

Executed two-invocation reproduction through the real `main` and
`run_selection`, sharing one status map and log directory:

1. Seed `prereq` and its dependent `child` as genuine current PASS records.
2. Invoke `--gate prereq --force-rerun prereq`.  The prerequisite executes,
   publishes new attempt `20260713-144604-705330`, and exits 0.
3. In a separate `main` invocation, select `--gate child`.  The runner prints
   `[skip] child: already PASS ... dependencies current`, calls the child body
   zero times, reports one resumed current PASS, and exits 0.  The child's
   surviving record is still the seed record proved against the old
   prerequisite result.

This is a reachable science-evidence error on the real board, not status-only
cosmetics.  `save-rebuild-drift` overwrites the persistent
`gates_emul_evaluate.h5` artifact (`gsv_bitwise_drift.py:305-307`).
`cobaya-adapter`, `finetune-smoke`, and `transfer-smoke` all declare it as
their prerequisite (`gates/board.py:2079-2112`) and consume the saved emulator.
A user can force only `save-rebuild-drift`, then later invoke one child; the
old child PASS is reported current even though it did not exercise the newly
written artifact.

The publication surfaces expose the same missing owner.  `_resume_state`
looks only at the named gate's own record (`run_board.py:1531-1562`), and both
`cmd_list` (`:1999-2026`) and `_write_board_md` (`:1593-1628`) call it without
dependency lineage.  With a stale-code prerequisite plus an own-current
child, public `--list` prints the prerequisite `stale-code` and the child
`PASS`, exit 0, even though `_dep_current_pass` returns false.  When a forced
prerequisite rerun fails, the emitted `BOARD.md` similarly publishes the
prerequisite as `FAIL` and its unselected child as `PASS`.

Required contract (the unimplemented persisted-identity clause already
present in the original 25M-20 ruling):

1. Every dependent gate PASS persists a direct-dependency snapshot, keyed by
   dependency id, containing the dependency's terminal result identity (a
   unique successful attempt plus its immutable log digest, or an equivalently
   strong single result-generation digest).  Capture it only after dependency
   currency has been validated and only when the child itself executes and
   passes.
2. A dependent PASS is current only when every dependency is independently
   current **and** its present result identity exactly equals the snapshot the
   child consumed.  A later dependency rerun, even a successful byte-identical
   rerun in another process, makes the child `stale-dependency` until the child
   re-executes.
3. One dependency-aware resume-state owner supplies execution resume,
   dependency acceptance, `--list`, and `BOARD.md`.  These surfaces may not
   reconstruct separate approximations.  A stale dependency changes the
   displayed/computed state without erasing the child's historical result or
   immutable log.
4. A legacy PASS for a gate with `deps` but no dependency snapshot predates
   this contract and is non-green (`pre-dependency` or equivalent).  Never
   bless it by copying today's dependency identity into the old child record;
   only an executed child can establish what it consumed.
5. Direct snapshots compose transitively: if A changes, B becomes stale; C's
   snapshot of B can no longer certify C.  Preserve the current same-process
   rerun behavior, but persisted lineage—not an in-memory set—is the authority
   across invocations.

Pure-Python board-selftest legs, all using the real runner paths: unchanged
current parent+child with a persisted matching snapshot resumes; two
sequential `main` calls force the parent then select the child and prove the
child body executes; a legacy dependent PASS without a snapshot is non-green;
parent stale-code/stale-input/stale-log/FAIL/SKIP/RUNNING states make both
`--list` and `BOARD.md` show the child non-green; a successful new parent
attempt also makes the old child `stale-dependency`; an A -> B -> C chain
proves transitive invalidation; an independent gate remains resumable.  The
catch-power mutation deletes or ignores the stored dependency snapshot and
must reproduce the false second-command resume with return code zero and zero
child calls.  Census the real BOARD's dependent gates and assert every
dependent PASS written by the harness carries the snapshot.

This reopens the accepted 25M-20 implementation; it does not dispute the
same-invocation `reran` repair, which is correct but insufficient.  It also
corrects the Red Team's earlier RT-IMPL-01 wording that execution, dependency
acceptance, `--list`, and `BOARD.md` already shared a complete verdict: they
share own-gate currency, not persisted dependency-result identity.

### 25M-26 Architect adjudication (Fable, 2026-07-13): CONFIRMED; hardening item 9; clause reconciliation becomes standing audit law

The finding and repair contract are adopted whole.  The two-invocation
witness is the permanent cross-process acceptance leg; `stale-dependency` is
a first-class resume state; and a legacy dependent PASS without a persisted
snapshot is non-green, following the pre-manifest rule rather than receiving
a retroactive identity.  This is item 9 in the 1b hardening increment.

Process lesson (binding on every future implementation audit): reconcile the
issued ruling's clause list against the delivered diff one clause at a time.
It is insufficient to verify that the implementation's own design is
internally coherent.  25M-20 passed a pre-merge audit while its explicit
persisted dependency-identity clause was absent; the accepted in-memory
`reran` design was correct but only a subset of the ruling.  The audit report
must therefore carry a clause checklist with code and gate evidence for every
item, or mark the item unimplemented.  The hardening increment is closed to
unrelated new scope; only a finding that reopens one of its existing clauses
may join it.  Other machinery findings wait for the next machinery window.

## 25M-27 (Red Team CONFIRMED, awaiting Architect adjudication): deleting a tracked root driver escapes queue 1c's dirty-tree watch

Queue 1c promises that preflight watches the five executable directories
**plus the root drivers**.  `_watched_paths` derives the latter from
`_REPO.glob("*.py")` (`gates/run_board.py:615-629`).  A deleted tracked driver
no longer exists for that glob, so it is omitted from the pathspec handed to
`git status` (`:1137-1142`).  The dirty tree can therefore be certified clean
because the path used to ask Git the question was built from the already-
damaged filesystem.

Live isolated-clone witness against the real helpers: move the tracked root
driver `cosmic_shear_train_emulator.py` out of the clone, then evaluate
`git status --porcelain -- <_watched_paths()>`.  `_watched_paths()` does not
contain the deleted filename, Git returns rc 0 with empty stdout, and
`_dirty_lines` returns `[]`.  A second witness using the manifest-uncovered
thin driver `cmb_tune_emulator.py` leaves `validate_manifests(BOARD, cfg)`
green as well, so manifest validation does not happen to mask this preflight
defect.  An unrelated selected gate may proceed while a tracked executable
entry point is deleted.

Required repair: derive the root-driver watch from tracked identity or a
stable complete registry, not only from entries that currently exist.  It
must also retain newly added/untracked root Python drivers, so the natural
surface is the union of the tracked root-`*.py` set and current root-`*.py`
entries (or an equivalent top-anchored Git pathspec).  Treat nonzero `git
status` as preflight failure rather than an empty clean result.  Keep one
owner for the watched surface and its printed description.

Pure Git/selftest legs using a temporary repository and the real helpers:
clean control; modified tracked root driver reds; deleted tracked root driver
reds and is named; newly added root Python driver reds; unrelated root text
file remains outside this code surface; config-only exclusion remains clean;
an uncovered thin family driver proves the manifest cannot satisfy the test
for preflight.  The mutation restoring the existence glob must reproduce an
empty offender list after deletion.

This reopens queue 1c's “root drivers” clause.  It is not permission to expand
the closed nine-item hardening landing unless the Architect places the reopen
there; otherwise it waits for the next machinery window.

## 25M-28 (Red Team CONFIRMED, awaiting Architect adjudication): 1b's inspectable stale-member promise is absent from `--list`, and input relocation loses the member name

The persisted-manifest contract says both `--list` and `BOARD.md` name which
resolved member staled (`notes/gates-and-board.md`, section C), and the phase-2
DONE record repeats that claim.  `_write_board_md` calls `_stale_member` and
adds `[stale member: ...]` (`run_board.py:1621-1624`).  `cmd_list` only prints
the state returned by `_resume_state` (`:1999-2026`); it never calls
`_stale_member` and has no member-detail field.

Live synthetic status using a real manifested gate: `_stale_member` returned
`code:cosmic_shear_train_emulator.py`, while `cmd_list` printed only
`ema-off-identity ... stale-code ...` with no filename.  Thus the operator
surface named explicitly by the ruling cannot inspect the persisted evidence.

There is a second omission in the same comparison owner.  Persisted input
members are triples `{key, path, sha256}`, and the overall input digest changes
when the resolved path changes.  `_stale_member` reduces fresh inputs to
`{key: sha256}` (`:1580-1583`).  Repointing one key from `a.yaml` to a
byte-identical `b.yaml` produces `stale-input`, but `_stale_member` returns the
empty string because the hashes match; the changed path is never named.

Required repair: one state-detail formatter is consumed by `cmd_list` and
`BOARD.md`; it reports the same first stale member on both surfaces.  Compare
the full persisted input identity `(key, path, sha256)`, not the hash alone,
and name at least the input key plus old/new path when relocation is the
cause.  Preserve the phase-2 audit's accepted best-effort exception for a
newly introduced member absent from the old manifest: a generic stale state
is honest there because the old record cannot name what it never stored.

Selftest legs: real `main --list` names a changed code member; generated
`BOARD.md` names the same member; byte-identical input relocation is
`stale-input` and names `input:<key>` plus the path change; content-only input
change names the key; unchanged members remain current.  Mutations removing
the list detail and reducing inputs back to `{key: sha}` must each red.

## DIDACTICS-94 adjudication (Fable, 2026-07-13): CONFIRMED with one sharpening — the decoupled anchor is live for refine, queued for finetune

Durable register at c020c84. Verified: README.md:1748 carries the
compressed "(a pull back toward the saved weights) is planned"
one-liner; the decoupled post-step mechanism is REAL and documented
in the code itself (training.py:324-356: "decoupled L2-SP anchor: a
post-step pull toward a reference W_0 ... AdamW-decoupling argument
... kept OUT of the loss"); the masked zero-init extra columns
story is executed (warmstart.py anchor_masks: ones on source
columns, zeros on the n_extra carriers; training.py:343). The
scalar example's arithmetic checks: 2.4 - 0.01*0.5*(2.4-2.0) =
2.398.

RATIFIED as filed, with ONE SHARPENING the prose must carry: the
decoupled anchor is a LIVE mechanism for the transfer REFINE stage
(refine.anchor is a validated key, experiment.py:1488-1501); it is
the ordinary-FINETUNE train_args.finetune.anchor key that is queued
and currently refused. The README teaches: (1) the conceptual L2-SP
form R(W) = (lambda/2) sum_j ||M_j o (W_j - W_j0)||_F^2 with every
symbol defined and the weight-decay distinction (decay pulls toward
ZERO and fights the warm start; the anchor pulls toward the SOURCE
weights); (2) the executed decoupled update W+ = Wopt - eta*lambda*
M o (Wopt - W0), kept out of the reported scientific loss, with the
recorded scalar example; (3) which weights anchor and why the
zero-initialized new-parameter columns are masked FREE; (4) the
current-state adjacency: live for refine, queued for finetune
(unit 24's contract), so ordinary fine-tuning is currently
unanchored.

Placement per campaign law: the teaching lands with the ROOT README
visit (the 72-80 batch's fine-tuning section); the 2.398 scalar
example is an executable documentation fixture -> UNIT 91; the
current-state sentence follows the stable prose per the
limitations-adjacent rule.

## 25M-26 adjudication (Fable, 2026-07-13): CONFIRMED — the persisted-lineage half of my own 25M-20 ruling, unimplemented and un-caught by my audit

Durable register at d890cab. The finding is verified against the
code I audited: cc85aa9's reran set is allocated per run_selection
call, and a child's status record persists its own code/input/log/
attempt identity but NO identity of the prerequisite result it
consumed; _dep_current_pass asks only whether the prerequisite is a
current PASS NOW. The red team's two-invocation witness (force-rerun
the prerequisite; a separate invocation resumes the child with zero
bodies, exit 0) follows necessarily.

THE HONEST PART FIRST: the 25M-20 ruling ALREADY contained this
clause — "persist the dependency verdict/digest or artifact
identity needed to bind that currency." cc85aa9 implemented the
in-process half only, the handoff's scope list did not mention
persistence, and my audit GO'd it as "exactly to the ruling"
without reconciling the ruling's clause list against the delivered
mechanism. AUDIT LESSON (recorded beside the probe-the-machinery
law): a pre-merge audit walks the RULING'S CLAUSES as a checklist
against the diff — "matches the described design" is not "matches
the ruling"; the described design may be the ruling minus a clause.

RULING (completes 25M-20; the contract as filed): each successful
child PASS persists, per direct dependency, the identity of the
successful result it consumed (the dependency's attempt id + verdict
/log digest). _resume_state compares the snapshot against each
dependency's CURRENT successful attempt; a mismatch is the new
non-green state STALE-DEPENDENCY, rerun on selection, published as
such by --list and BOARD.md (a child never displays PASS beneath a
stale/failed prerequisite). Legacy dependent PASSes without
snapshots are non-green and rerun — never retroactively blessed
(the pre-manifest precedent applied to lineage). Cross-process
selftest legs: the exact two-invocation witness; the
snapshot-refresh control (child reruns once, snapshots the new
attempt, then resumes); the legacy-record refusal; the mutation
restoring snapshot-free records must reproduce exit-0-zero-bodies
and red. The resume-state list grows to include stale-dependency —
the gates/README teaching (DIDACTICS-87's list) and the state
machine prose update with it in the D1 visit.

The 25M-24 RIDER is ratified into item 7's contract: --gate A
--force-rerun B with B outside the selected surface is a usage
error (exit 2), never a silent discard.

Placement: 25M-26 = ITEM 9 of the 1b hardening increment (one
machinery landing, one audit; the same status-record surface 21's
projection already touches). Queue 2 remains behind the increment.

## 1b hardening increment: 5 of 9 items landed (Opus, 2026-07-13)

Five self-contained / low-blast-radius items landed on the branch, each
Mac-validated (board-selftest ALL PASS, validate_manifests 40/40 ok, py_compile
clean); reviewed together in one audit (the Architect's "one landing or two
commits, ONE audit"):

- 9e30860 (items 21+22+topology): 25M-21 the named _config_execution_projection
  (documentation namespaces excluded from the input digest; _help prose edits no
  longer stale gates, value edits still do); 25M-22 saved_emulator_root removed
  (key + _help, zero readers verified) with the config-census standing leg
  (every non-documentation board_config key has a Python reader); the
  topology-assertion leg (deps precede children in BOARD, the 25M-20 rider).
- 05334f3 (items 24+25): 25M-24 main() validates selectors + rejects
  --list/--check-together BEFORE any action-mode return (real-main() legs);
  25M-25 select_gates' --from includes an explicitly named optional start first
  (id-list-pinned leg + restoration mutation).

REMAINING (the interdependent census-core trio + the cross-invocation item), all
fully specced (parallel subagent investigation captured in the handoff):
- 25M-16 closure truth: extend the census for spec_from_file_location loaders +
  Cobaya python_path adapters (reviewed _RUNTIME_LOADER_COVERS table), source-
  opened-as-data (a _data_read_sites scanner hashing-as-file, NOT closure-
  seeding -- the finite_contract.py leaf lesson), and the bare-sibling import
  (gct_parity.py:43 -> resolve against the importer's dir); re-populate the 8
  identity/smoke manifests with their cobaya_theory root; the two-call-scanner
  mutation + the geo-paths whole-scope fixture. Per-gate covering-root table in
  the handoff.
- 25M-18 all-quantified coverage: census (b) requires a declared root equal-to-
  or-ancestor-of EVERY required cover, all-quantified over covers AND over
  required-all tuple members; must-red fixtures child-as-cover, strip-one-of-two,
  any-one-of-eight (interlocks 16's new tables).
- 25M-19 owner resolvers: one resolver per input namespace (evaluate_yaml->repo,
  gate_configs.*->yaml_dir, gate_data.*->owner, deploy_data->machine-dependent
  resolve-not-exist); repo-owned inputs refuse None-sha; two-cwd + executed==
  hashed legs; CWD-first mutation.
- 25M-26 cross-invocation lineage: per-dependency snapshots persisted + a
  STALE-DEPENDENCY resume state (the cross-run persistence clause of 25M-20).

These four are best landed together with fresh focus -- they change the
validation core (census, coverage, resolver, resume-persistence) all 40 gates
depend on, and 16's tables consume 18's coverage rule. Queue 2 opens at the
full increment's merge.

## Hardening 5/9 pre-merge audit (Fable, 2026-07-13): GO for the merge; item 7 is INCOMPLETE by two clauses — the checklist law's first two catches

Commits 9e30860 + 05334f3 audited under both new laws (clause
checklist + live probes against the real main() on the cocoa
interpreter).

CLEAN — verified by probe, not by report:
- 25M-21: _config_execution_projection excludes ANY
  underscore-prefixed documentation namespace (a correct
  generalization beyond the ruling's literal _help); live probe —
  prose edit leaves the projection fixed, value edit moves it.
- 25M-22: zero saved_emulator_root occurrences remain; the
  config-census standing leg present.
- Topology-assertion leg present (deps precede children in BOARD).
- 25M-24, the landed clauses: --list with an unknown --gate exits 2;
  --list --check exits 2 (live real-main() probes).
- 25M-25: --from triangle-shading returns the named optional start
  FIRST (count 21, zero later optionals) — exactly to ruling.

TWO CLAUSE GAPS, both in item 7 (25M-24), both found by walking the
ruling's clause list:
1. IGNORED run controls: --list --force-rerun <valid-id> exits 0 —
   the ruling requires "incompatible or IGNORED run controls exit
   nonzero"; the mutual exclusion covers --check and unknown ids
   but a valid ignored control passes silently.
2. The bcf4ce2 RIDER: --gate A --force-rerun B with B outside the
   selected surface reached preflight instead of exiting 2 as a
   usage error (live probe: cmb-identity + force-rerun bsn-identity
   ran preflight).

RESOLUTION: the merge proceeds NOW — nothing landed is wrong; the
gaps are absent clauses, not broken ones, and the increment is
explicitly open at 5/9. Item 7's two missing clauses join the
remainder batch (a small commit alongside or before the census-core
trio). QUEUE 2 opens only at 9/9 PLUS item-7 completion.

GO for the remainder in the Implementer's proposed dependency
order, ratified: 18's all-quantified coverage rule first (16's
tables consume it), then 16's closure truth (the
_RUNTIME_LOADER_COVERS reviewed table; the hashes-as-file
data-read rule is APPROVED — a source file read as data joins the
digest surface without seeding the import closure, the
finite_contract leaf lesson), then 19's owner resolvers, then 26's
lineage snapshots + STALE-DEPENDENCY. The deliberate stop at the
census-core boundary was the right call and is credited: the four
remaining items rewrite the validation core all 40 gates depend on
and deserve a fresh session.

## Consolidation check + 25M-23 and 25M-27..35 adjudication (Fable, 2026-07-13): the merge on main is SOUND; ten findings placed across four homes

CONSOLIDATION VERDICT — SOUND. 16bea64 is a proper two-parent merge
(81183e7 red-team line + 47e5f59 my branch tip); my branch is an
ancestor of main (nothing dropped); zero conflict markers repo-wide;
my three newest sections and the HARDENING 5/9 ledger paragraph are
present verbatim; the landing is notes-only (11 files, all notes/);
the working tree is clean; all four cited anchors resolve. The
red team now appends to the shared ledger directly — my future
appends anchor on their tail, which is the shared-file protocol
working.

THE BATCH: the register's 27..35 plus 25M-23, which was filed
DURABLY but never relayed in chat — the register-is-the-record rule
surfaces it here; adjudicated with the batch. This is the red
team's first clause-checklist wave: each item cites an issued
clause and its missing implementation or discriminating leg.

Verified by my own probes (conclusive):
- 23: finite_contract.py:1255 calls _chi2_domain, which is DEFINED
  NOWHERE (the only def is check_chi2_domain, :885) — Part H dies
  on NameError when reached. A broken gate today; no production
  change requested.
- 27: _watched_paths builds the root-driver pathspec from
  _REPO.glob("*.py") — existing files only — so a DELETED tracked
  driver vanishes from the very pathspec that asks git the
  question; preflight certifies clean. (_EXECUTABLE_DIRS covers
  directory deletions; only root drivers have the hole — exactly as
  filed.)
- 28 (mechanism): _stale_member reports the first hash mismatch by
  stored path/key lookup; the --list omission and the relocation
  name-loss stand per the register witness.
- 29 (absence proven): eval_val is driven 42 times in
  finite_contract.py, but every 1e38 payload lives in the TRAINING
  epoch-reduction leg (:812-858); no extreme-scale eval Part I
  exists — the issued 14(f) clause is unimplemented.
- 35 (shape): mps_identity.py:788 calls
  exp.release_train_staging() manually as "the sweep lane's
  cleanup" — the lifecycle leg substitutes a manual call for
  driving the real _sweep_job.

CONFIRMED per the register's executed witnesses (30/31/32/33/34),
with the standing condition: each entry's witness reproduces as the
FIRST leg of its repair landing.

PLACEMENTS (four homes):
1. 23 joins the GATE-TRUTH BATCH (D4) immediately — a crashing Part
   H is a broken gate now; the repair is the helper's real
   definition or the call's correction, plus a leg that executes
   Part H to completion.
2. 27 + 28 form a small BOARD-MACHINERY FOLLOW-UP right behind the
   census-core remainder (the hardening increment stays CLOSED at
   nine): 27 = the watched pathspec derives from the git-tracked
   inventory (git ls-files), never the existing-file glob, with the
   deleted-driver leg; 28 = --list surfaces the stale member, and
   member identity is path-aware so a byte-identical relocation is
   still named. Both before queue 5.
3. 29 + 30 + 31 join GATE-TRUTH INCREMENT 2 (the unit-14 acceptance
   completions): the extreme-scale eval Part I as specified in the
   issued 14(f) clause; 14(h)'s grid/grid2d corrupt-score refusals
   live-driven, not AST-censused; Parts A/C asserting CURRENT error
   prefixes (the false-red repair), each with its witness leg.
4. 32 + 33 + 34 join the MPS/STAGING VISIT (with 17 and 96's
   artifact legs): 32 is the heaviest — a QUEUE-3 REOPEN
   (_grid2d_law_rows overwrites the canonical seeded order with
   sorted-compact arange, and the MPS gate asserts the defect as
   truth — the gate flips with the fix, one landing); 33 = exact
   raw/base row-count equality in bounded staging, not
   covers-the-max; 34 = the discriminating payload where a pre-cast
   accumulator changes the pin mask. 35 joins the gate-truth work
   (test-double honesty: drive the real _sweep_job).

The 25M series stands at 01..35, one tombstone (07), no unrelayed
remainder — 23's surfacing closes the register-vs-chat gap.

## 1b hardening remainder: 18 / 16-runtime / item-7 / 19 / 26 landed; 16's data-read half is the flagged remainder (Opus, 2026-07-13)

Executed the ARCHITECT_HANDOFF: READY FOR EXECUTION remainder in the
ratified order. Four commits on the branch, each Mac-validated
(validate_manifests 40/40 ok, board-selftest ALL PASS growing 102 ->
120, py_compile clean); one audit.

- 23cc78a (25M-18 + 25M-16-runtime): waiver coverage is ALL-quantified
  over required covers, and a root covers a cover only by being that
  cover or an ANCESTOR (`_root_is_ancestor`) — both permissiveness
  directions (child-as-cover, any-of-many) killed; the pre-25M-18
  blessing selftest fixture flipped to must-red, plus child-as-cover /
  strip-one-of-two / any-one-of-eight (live cli_strict 8-cover) +
  full-trees green. NEW runtime-loader census (c): `_RUNTIME_LOADER_COVERS`
  reviewed table + `_runtime_loader_sites` scanner (spec_from_file_location
  Call sites + python_path detected STRUCTURALLY as a dict key / yaml-line
  via the `_COBAYA_PP` constant so the harness's own prose never
  self-matches); coverage is TABLE-DRIVEN (positive), the scanner is the
  negative catch on unlisted files. The eight identity/smoke manifests
  re-populated with their cobaya_theory root(s) (cmb also
  compute_cmb_covariance). Bare-sibling imports resolve against the
  importer's dir in `_module_to_repo_paths` (gct_parity.py ->
  gsv_bitwise_drift, now in cobaya-adapter's closure; a third-party
  top-level still -> []). check_runtime_loader_census added.
- 7d8566d (item-7 completion): the two 25M-24 clause gaps the checklist
  audit found. An action mode (--list/--check) now rejects ANY paired run
  control (selection / force / dry) as a usage error (exit 2) — "ignored
  controls exit nonzero", a valid ignored control failing like an unknown
  one; and an explicit --force-rerun id OUTSIDE the selected surface is a
  usage error (the bcf4ce2 rider), --force-rerun-all exempt. Five
  real-main() legs, red-capable against the un-fixed exit-0.
- b1835ac (25M-19): input resolution is a function of the reviewed OWNER
  of each namespace (`_input_owner`: evaluate_yaml -> repo/_REPO,
  gate_configs.* -> yaml_dir, deploy_data/gate_data -> machine), NO
  process-CWD candidate; the RunContext.evaluate_yaml consumer delegates
  to the SAME resolver (executed == hashed). A repo-owned input that fails
  to resolve reds. check_input_owner_resolution: owner dispatch, two-cwd
  identity, collision-ignored, executed==hashed, repo-owned refuse-None-sha.
  DEVIATION FLAGGED: the ruling's "resolve-not-exist for deploy_data ONLY"
  is honored as "repo-owned must resolve; yaml_dir AND machine may be
  absent on a dev box" — yaml_dir is itself a deploy path (this numpy-only
  Mac has no external_modules/code checkout), so requiring gate_configs to
  resolve would break the Mac-green validate the increment depends on; the
  security property is preserved for the true repo-owned input
  (evaluate_yaml, resolved under _REPO, present on every machine). Awaiting
  ratification.
- 65c5bec (25M-26): completes 25M-20's persistence clause. Each child PASS
  persists per-dependency `deps` snapshots (attempt id + log digest,
  `_dep_snapshot`); `_resume_state` gains `_dependency_lineage_state` ->
  the new non-green "stale-dependency" state when a dependency's CURRENT
  attempt differs from the snapshot (a separate invocation reran it) or
  when a dependent PASS carries no snapshot (legacy, never blessed). Rerun
  message + BOARD.md log-cell updated. check_cross_invocation_lineage over
  ONE shared status file + log dir across several real main() invocations:
  the two-invocation witness, snapshot-refresh control, and the
  snapshot-free mutation reproducing exit-0-zero-bodies as a rerun.
  DEFERRED to the D1 visit (per ruling): the gates/README resume-state list
  (DIDACTICS-87) and the state-machine prose gain "stale-dependency".

THE FLAGGED REMAINDER — 16's source-opened-as-data (`_data_read_sites`)
half is NOT landed. The runtime-loader census closed the ADAPTER digest
hole (the scientific one: a changed emul_*.py now reruns its identity /
smoke gate). The ruling's 16 ALSO names a data-read mechanism (the
"two-call-scanner" + "geo-paths whole-scope fixture"): geo-paths,
board-selftest, artifact-readback, family-first, generator-seed OPEN
executable .py AS DATA their manifests omit (verified sites: geo_paths.py
os.walk whole-repo; board_selftest.py rglob whole-repo + board.py /
run_board.py / finite_contract.py fixed reads; artifact_readback.py:76/97
results.py + scalar_train_emulator.py; family_first.py:88/99 four drivers;
generator_seed.py:39/57 generator_core.py; also cli_strict.py:82 eight
entry points, diagnostics_domain.py:283 diagnostics.py). Per the ratified
rule they must HASH-as-file WITHOUT closure-seeding (the finite_contract
leaf lesson). I deliberately stopped rather than guess this design at the
tail of a long session: it changes the DIGEST SURFACE for meta-gates
(geo-paths' digest would become the whole repo -> it reruns on any repo
change) and I lack the red team's FILED geo-paths whole-scope expected
member set. PROPOSAL for audit: a `_DATA_READ_COVERS` reviewed table (per
source-census file -> a fixed .py list OR a WHOLE_REPO marker) mirroring
`_RUNTIME_LOADER_COVERS`; `_data_read_sites` scanner detecting open /
read_text / read_bytes / ast.parse on a .py + os.walk/rglob/glob for .py;
`_gate_code_digest` folds the resolved data-read targets as HASH LEAVES
(no _derive_closure expansion), a WHOLE_REPO marker hashing the sorted
repo .py set. OPEN QUESTIONS for the Architect: (1) reviewed-table vs
static path resolution; (2) is geo-paths' digest = the whole repo the
intended semantics; (3) the exact geo-paths whole-scope fixture member set
(exclusions: __pycache__ / notes / .git / self). QUEUE 2 opens only at the
FULL 9/9 including this half.

LANDING BLOCK (main is the user's; clean fast-forward):
  cd /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
  git merge claude/amazing-keller-e798b6 && git push

## Hardening 8.5/9 pre-merge audit + the 19 narrowing and data-read rulings (Fable, 2026-07-13): GO on all four commits; the last half is fully specified

AUDIT — GO (clause walk + live probes on the cocoa interpreter):
- 25M-18 (23cc78a): my strip-designs-keep-losses witness now REDS
  ("reaches the waived dynamic import emulator/results.py:672 ...");
  the all-quantified ancestor-or-equal coverage holds; baseline
  40/40 stays green.
- 25M-16 runtime half: stripping cobaya_theory/emul_cmb.py from
  cmb-identity REDS with the new census naming the runtime-loader
  site — the adapter digest hole is CLOSED; the eight
  identity/smoke manifests carry adapter + generator roots
  (cmb-identity and mps-smoke inspected). One audit note: my first
  tamper dropped the directory name "cobaya_theory" and stayed
  green because the declared root is the FILE — a reminder that
  tamper probes must strip EXACT declared roots (recorded so the
  next auditor doesn't misread such a green).
- Item-7 completion (7d8566d): --list --force-rerun <valid> now
  exits 2; --gate A --force-rerun B-outside-selection exits 2
  before preflight — both prior clause gaps closed, live-probed.
- 25M-19 (b1835ac) + 25M-26 (65c5bec): clause lists walked against
  the diffs; the two-cwd/executed==hashed/refuse-None-sha and
  two-invocation/snapshot-refresh/legacy legs all present;
  board-selftest ALL PASS independently rerun (102 -> 120 legs).

RULING 1 — the 25M-19 NARROWING: APPROVED. Per-owner refusal
semantics are the correct reading: repo-owned (evaluate_yaml,
resolved under the repository) must resolve on every machine;
yaml_dir-owned (gate_configs) and machine-owned (deploy_data) are
resolve-not-exist on boxes lacking their trees — yaml_dir is itself
a deploy path, so the Mac's numpy-only checkout legitimately lacks
it. ONE BALANCING CLAUSE, binding: None-sha is a VALIDATION-time
allowance, never a RUN-time one — at gate run time an unresolvable
declared input refuses before the gate body executes, so a
workstation run hashes every member or does not run. If run-time
refusal is not already implicit in the input-digest path, it lands
as a small leg with the data-read half.

RULING 2 — the DATA-READ DESIGN (the three open questions):
1. REVIEWED TABLE (_DATA_READ_COVERS) over static resolution —
   consistent with _RUNTIME_LOADER_COVERS and the waiver
   philosophy; the _data_read_sites scanner is the negative catch
   (an unwaived data-read site = validation error). Hash-as-file,
   never closure-seed (already approved; the finite_contract leaf
   lesson).
2. GEO-PATHS GOES WHOLE-SCOPE, and the missing member set is a
   non-problem: the gate's manifest enumerator IS the gate's own
   scan enumerator — ONE shared function with the same exclusions,
   so the hashed surface and the asserted surface can never
   disagree. A gate whose verdict quantifies over every repo .py
   MUST stale when any changes; geo-paths is a cheap text scan, so
   frequent reruns are correct, affordable, and honest.
3. THE WHOLE-SCOPE FIXTURE: set equality between the gate's
   enumerated scan set and its manifest members, plus one
   byte-edit-any-file -> stale-code control; the five data-readers
   (geo-paths, board-selftest, artifact-readback, family-first,
   generator-seed) each gain their reviewed cover entry and a
   member-present leg.

Queue 2 opens at the full 9/9 (this half + the balancing clause);
the 27/28 machinery follow-up lands right behind it, before
queue 5. The deliberate stop-rather-than-guess on a digest-surface
design question was correct and is credited — the whole-repo
semantics needed an Architect ruling, and now it has one.

## DIDACTICS-95..100 adjudication (Fable, 2026-07-13): the current-state ruling ratified as campaign law; the CMB rewrite's four factual heads verified; SONIC corrected

Durable register at 9f00106. Verification: the standing user ruling
is recorded verbatim in conventions-and-workflow.md (README =
current library, never a development diary; limitations =
scope/consequence/user-action only) — RATIFIED as binding campaign
law, superseding and generalizing the queue-6 stable-map/
current-limitations restructuring with user authority. The
acceptance stands as filed: an untruncated history-vocabulary scan
(board run / workstation-proven / fixture / rerun / queued / landed
/ ruling / retired / dated status ...) with HUMAN adjudication of
every retained hit, the reviewed reasons living in audit evidence,
never in the README.

- 96 CONFIRMED (the CMB rewrite): all four factual corrections
  verified or physically ratified — (a) the displayed formula is
  the TT/EE auto-spectrum form; the covariance code's TE expression
  differs and pp carries no instrumental noise (the :229-242 region
  read during 25M-11/12); the README states the executable
  interface (.npz supplies ell / cl_<spectrum> / sigma_<spectrum>;
  the trainer divides the centered residual by the stored sigma)
  rather than reproducing derivations; (b) dense non-Gaussian
  cov_* blocks are written but the trainer reads only
  ell/sigma/cl — geometries/cmb.py's own preamble says "no
  rotation and no dense matrix"; (c) as_exp2tau_ref removes the
  DOMINANT primary amplitude dependence (As e^-2tau), not all
  shape-independent content — "shape only" is an overclaim, and
  the shipped example uses As without the generator always
  sampling As directly; (d) roughness is a twice-applied boxcar
  smoother penalizing the removed remainder — the LOSS CODE'S OWN
  DOCSTRING (losses/cmb.py:52-62) is honest; the README overclaim
  contradicts its own implementation's documentation. Plus the
  serving-units sentence: raw C_ell; TT/TE/EE in muK^2; pp
  dimensionless. The four heads + units jump the queue (D2); the
  six-step rewrite rides the root README visit.
- 95/97/98/99 RATIFIED and FOLDED: no new landing slots — 95's
  diary sweep + 97's parenthetical rule (short
  definitions/symbols/units/acronyms only) + 98's family chapters
  ride the ROOT README visit (72-80); 99 rides the package/gates/
  syren visits (81-86/87-89/90-92). One visit per file stands.
- 100 CONFIRMED: README.md:21 expands SONIC as "Surrogates and
  Operators ..." while the exact name is "Simulated Observables for
  Numerical Inference in Cosmology" — the root README correction is
  Implementer-assigned and joins the FACTUAL BUNDLE immediately
  (the project's own name, line 21). The red team's TeX/PDF
  correction stays in their custody — the texnotes rule honored
  exactly as designed.

## 1b hardening COMPLETE — 9/9 + item-7 + the 27/28 machinery follow-up (Opus, 2026-07-13)

Executed the two READY-FOR-EXECUTION handoffs after the 8.5/9 audit GO.
Two commits since the 8.5/9 batch, each Mac-validated (validate_manifests
40/40 ok, board-selftest ALL PASS 129 -> 142, py_compile clean):

- 9fec3fa (16 DATA-READ HALF -> 9/9): a check that OPENS .py source AS DATA
  hashes it as a DIGEST LEAF, never closure-seeded (finite_contract leaf
  lesson). `_DATA_READ_COVERS` reviewed table (the five named readers +
  the scanner-forced diagnostics-domain; cli-strict dropped -- its eight
  drivers are already code roots) drives the digest via `_data_read_targets`
  (POSITIVE fold); `_data_read_sites` is the NEGATIVE catch (os.walk /
  glob-of-.py / open|read_text|read_bytes|getsource with a .py literal;
  ast.walk and ast.parse are NOT reads, excluded; the shared harness is
  excluded). census (d) reds an unreviewed data-read site. geo-paths goes
  WHOLE-SCOPE via ONE shared enumerator: `run_board.repo_py_files()` is BOTH
  the gate's manifest data surface AND geo_paths.py's own folder-census scan
  (geo_paths imports run_board the board_selftest way) -- proven by the
  set-equality leg (101 == 101), so the scanned set and the hashed set are
  one function, not two lists. board-selftest is likewise whole-scope.
  family-first's data cover closes a real hole (its three sweep/tune drivers
  were read as data but not code roots). 25M-19 BALANCING CLAUSE: a None
  input sha is a VALIDATION-time allowance only -- at RUN time run_selection
  refuses a declared input that does not resolve/hash BEFORE the body.
  EXPECTED on the first workstation run and CORRECT (per the ruling):
  geo-paths and board-selftest read stale-code and rerun ONCE, because their
  whole-repo digest is now computed -- a cheap text scan, honest, affordable.
- 1eb0e8d (27+28 machinery follow-up): 25M-27 -- the clean-tree watch's root
  drivers are the UNION of git-TRACKED root *.py (git ls-files) and current
  root *.py, so a DELETED tracked driver stays watched and reds instead of a
  glob built from the damaged filesystem certifying it clean; preflight
  treats a nonzero git status as FAILURE, not empty-clean. 25M-28 --
  _stale_member compares an input by its FULL identity (key, path, sha256),
  so a byte-identical RELOCATION names key + old->new path; a shared
  `_state_detail` formatter now serves BOTH --list and BOARD.md, so the
  operator surface the ruling names can inspect the stale member on --list.
  board-selftest gains check_watch_tracked_drivers (pure-git temp-repo legs)
  and check_stale_member_surface.

DEFERRED to the D1 visit (per the 26 ruling): the gates/README resume-state
list (DIDACTICS-87) and the state-machine prose gain "stale-dependency".

QUEUE 2 now opens (9/9 + item-7 complete, 27/28 landed). Per the spine:
queue 2's evidence rollout -> D3 -> D4 -> queue 5, whose first exhibit is the
populated board's real reruns (including geo-paths / board-selftest staling
once on the first workstation run, as expected).

LANDING BLOCK (main is the user's; clean fast-forward):
  cd /Users/vivianmiranda/data/COCOA/june2026/emulators_code_v2
  git merge claude/amazing-keller-e798b6 && git push

## USER REASSIGNMENT: the didactics campaign moves to the red team (Fable, 2026-07-13)

The user reassigned the DIDACTICS campaign's implementation from the
Implementer to the red team, to relieve the Implementer's ~30-40
landing backlog. Precedent: the red team implemented the 2026-07-12
defect batch (audited). The split, binding:

RED TEAM NOW OWNS (implementation, not just filing):
- The D2 FACTUAL BUNDLE (every accumulated head: the falsehoods,
  contradictions, the SONIC expansion, 79's command repair, 96's
  four CMB heads + units sentence, 61/69/70/71's prose, ...).
- D1 NAVIGATION TRUTH's prose half now (the gates/checks __init__
  index, README walkthrough, the 58 preamble); its EXECUTABLE
  board-selftest set-equality leg WAITS for the 27/28 machinery
  landing (board_selftest.py is Implementer-owned until then).
- The FOUR README FILE VISITS under the current-state user ruling:
  root (72-80 + 94-98), package map (81-86), gates (87-89 + 99,
  merged with D1), syren (90-92 + 99).
- UNIT 91, the documentation-examples gate (one NEW board-listed
  module; ~15 fixtures, most from their own filings).
- The DIDACTICS-93 rename increment (step_frac/s_step + the
  ambiguous-name census; byte-identical outputs + the Planck-LCDM
  control as acceptance).
- Lane-3 family teaching PROSE — but only AFTER each protocol unit
  lands (the wave-4 units stay with the Implementer); nothing
  lane-3 starts before its unit, unchanged.

STAYS WITH THE IMPLEMENTER: the lane-3 AST pieces inside protocol
landings (10b with unit 66; 12's keyword-arg conversion with unit
85; 04's stage-ram table leg); gate units 90/92/93 (minted from
didactics findings but gate work, not campaign doc work); and
everything else in the queue (machinery, queue 2, the gauntlet,
production units).

COLLISION RULES (binding): the red team's campaign surfaces are the
four READMEs, the production docstring surfaces the campaign names,
the NEW unit-91 module, and texnotes; they do NOT touch
gates/checks/*.py teaching until D4 + D5 land on any file both
would touch, and do NOT touch gates/run_board.py / board.py /
emulator production code. The Implementer's surfaces are unchanged
minus the campaign. Both continue landing on main with pre-merges;
the both-append ledger conflict resolves as filing-first,
adjudication-second, both retained.

GOVERNANCE UNCHANGED: every campaign landing is Architect-audited
pre-merge under the full binding battery (the voice note read
BEFORE writing; AST docstring census + documentation-only AST
equivalence with named exceptions; untruncated history-vocabulary
scans with human-adjudicated retentions; quantifier discipline;
commands executed before printed; the link check; no numeric
counts; the DIDACTICS- prefix aliasing rule; the current-state
ruling). The red team keeps filing rights, but campaign
adjudications remain exclusively mine, and the Implementer may
spot-review campaign landings (the role swap, for adversarial
coverage). The anti-self-audit rule: the red team never certifies
its own campaign work green — my audit is the acceptance.

## USER REASSIGNMENT, EXTENDED: three more transfers to the red team (Fable, 2026-07-13)

The user directed moving everything SAFELY movable beyond the
didactics campaign. Three items pass the safety test (standalone
surface, red-team-executable locally, no machinery or one-owner
coupling); each was red-team-filed with executed witnesses:

1. THE UNIT-13 COVARIANCE PACKAGE: the 25M-08/11/12 trio
   (stencil-factor representability; the 2x2 noise PSD inequality +
   assembled-PSD check; beam-exponent representability + postcompute
   finiteness) PLUS the 45M-01 params-block schema amendment — the
   whole compute_cmb_covariance.py validator campaign, under the
   recorded Planck-LCDM byte-identity control. Pure-CPU legs, all
   executable in the red team's environment. Constraint: the legs
   live in a covariance-owned check module (new or existing
   covariance-specific); touching any SHARED gates/checks file
   needs my coordination first.
2. UNIT 90 (the BSN independent quadrature reference +
   Simpson-weights mutation): a bounded addition to bsn_identity —
   a file no gauntlet batch touches, collision-safe. The torch-
   dependent execution is workstation-owed as always; an
   Implementer spot-run on the Mac cocoa interpreter may
   pre-validate.
3. UNIT 94's BOUNDARY-INTERIOR HALF (25M-01): the named
   interval-coordinate helper (nextafter interior), the
   pre-sampling validation and refusal, and the CPU red legs in
   generator_core.py's uniform branch. THE SEAM, binding: the
   requested+resolved support PERSISTENCE clause stays with unit 8
   (the Implementer's run-control/identity machine); the red team
   lands the helper FIRST and unit 8 rebases on it — same file,
   disjoint regions, that landing order.

EXAMINED AND KEPT WITH THE IMPLEMENTER, with reasons: unit 92
(emulator/training.py production — collision rule); unit 93 (CUDA-
only execution, pairs with 77/80's Torch legs); unit 95 (spans
staging/experiment/sweep drivers — design-wide); unit 96 (artifact
core, interlocks 3/76/41); the 25M-06 .ranges fix (unit 82's ONE
canonical decimal representation must not gain a second owner); the
D3/D4/D5 gauntlet (critical path, torch-dependent fixtures,
Implementer momentum); 27/28 + queue 2 (machinery); the queue-3
reopen 32/33/34 (emulator/data_staging production); unit-21's
publication PSD (couples to the D3/56 publication predicate); the
wave-4/EMUL2 chain and both design proposals.

The red team's implementation surfaces now total: the four READMEs
+ named docstring surfaces + unit 91's new module + texnotes
(campaign), compute_data_vectors/compute_cmb_covariance.py + its
check module (unit 13), gates/checks/bsn_identity.py (unit 90), and
generator_core.py's uniform branch (unit 94 half, landing before
unit 8). Everything else remains Implementer territory. Governance
unchanged: every landing Architect-audited pre-merge; the red team
never self-certifies; adjudications stay mine.

## 1B COMPLETE: the 9/9 + 27/28 pre-merge audit (Fable, 2026-07-13): GO — queue 2 formally opens

Commits 9fec3fa (data-read half -> 9/9) and 1eb0e8d (27/28) audited
under the clause-checklist law with fresh adversarial probes:

- 40/40 validates, 0 errors; the full selftest reruns ALL PASS
  independently on the cocoa interpreter.
- The whole-scope ruling is implemented EXACTLY as designed:
  geo_paths.py:197 iterates run_board.repo_py_files() — the same
  function that builds its manifest surface; the live set equality
  holds at 101 == 101. One enumerator, not two lists.
- The census-(d) negative catch is real: removing
  artifact_readback's reviewed cover entry in memory REDS with the
  scanner naming the exact unreviewed data-read site
  (artifact_readback.py:76).
- Clause walks clean on both rulings, including the 25M-19
  balancing clause (run-time refusal of an unresolvable declared
  input, landed + legged) and 28's full-identity member comparison
  with the shared _state_detail formatter (one owner for --list and
  BOARD.md).

Three Implementer refinements RATIFIED as improvements over the
ruled minimum: cli-strict dropped from the data-read covers (its
drivers are already code roots — correct de-duplication); the
27 pathspec is the UNION of git-tracked and currently existing
root drivers (covers untracked new drivers beyond my git-ls-files
minimum); a nonzero git status is now itself a preflight failure
(fail-closed). The expected first-workstation-run staling of
geo-paths and board-selftest (their whole-repo digests computed
fresh) is honest, affordable, and recorded in advance.

VERDICT: GO. THE 1B PROGRAM IS COMPLETE — population 40/40,
hardening 9/9 with item-7 and the 27/28 follow-up, every clause of
every ruling implemented and probed. QUEUE 2 (the evidence rollout)
FORMALLY OPENS at this merge. Per the standing coordination point,
D1's executable board-selftest set-equality leg is hereby RELEASED
to the red team (the 27/28 landing quieted board_selftest.py). The
spine: queue 2 -> D3 -> D4 -> D5 -> queue 5, whose first exhibit is
the populated board's real reruns with persisted members.

## PARALLELIZATION WAVE 2 (user pre-authorization, trigger table) (Fable, 2026-07-13)

The user pre-authorized further transfers to the red team as
collision constraints clear. Binding mechanism: each row fires when
its trigger lands (reported in an Implementer handoff); the
Architect then issues the transfer handoff WITHOUT a fresh user
ruling — the authorization is this section. Governance unchanged
(Architect pre-merge audit; no self-certification; the seams rule).

| Transfer | Trigger | Why safe then |
- UNIT 93 (compile-mode CUDA lane): trigger NONE — transferable on
  red-team request at any time. Writing is machine-independent;
  execution is workstation-owed for either team. Moves on their
  capacity, not on a seam.
- UNIT 92 (device-audit totality, emulator/training.py): trigger =
  D5 landed. The unit-14 acceptance completions are the last
  gauntlet work near the training/eval surface; after D5,
  training.py has no in-flight Implementer increment until the
  late-unit tail.
- UNIT 21 (CMB publication PSD): trigger = D3 + unit 56 landed.
  Its publication-predicate coupling is to the disjointness repair
  and checkpoint-ingress; once both land, the generator publication
  surface is quiet and the legs are pure CPU.
- UNITS 41 + 53 (sweep-header truth; study-name resolver): trigger
  = queue-2 rollout plan APPROVED (both touch driver surfaces the
  rollout may relabel; after the plan freezes the vocabulary, the
  drivers are safe to edit). Pure-python table/resolver work,
  red-team CPU-executable.
- UNIT 95 (output-channel owner): trigger = the wave-4 protocol
  chain landed (experiment.py quiets — 95 spans
  staging/experiment/sweep drivers and CANNOT move while the
  adapter-protocol units are in flight there). The largest wave-2
  item; re-confirm the seam with the Implementer at transfer time.
- gates/checks teaching (the campaign's 03/04/06/07 batches):
  trigger = D4 + D5 landed (the standing collision rule lifting is
  itself the trigger; already implied by the campaign transfer,
  restated here for completeness).

NOT pre-authorized (would need a fresh user ruling): unit 96 (the
artifact core), units 74/76/77/78/80/82/84/85 (fixed-facts +
recipe/artifact one-owner chain), unit 8 (run-control machine),
queue 2 itself, and the queue-3 staging reopen — these are the
Implementer's deep-context core, and moving them would trade
correctness risk for throughput.

Standing expectation: at each trigger the Implementer's handoff
names the freed surface; my transfer handoff to the red team cites
this section and the unit's contract; the ledger records each
firing.

## Queue 2 rollout plan — PROPOSAL for Architect review (Opus, 2026-07-13)

Propose-don't-guess for a design-sensitive surface: this is the shape
BEFORE any code lands. It builds on the 45M-72 foundation already on the
branch — `Assertion(aid, anchor)`, `Gate.evidence`, and `validate_evidence`
(run_board.py:707, runs on every invocation incl. `--list`, checks every
anchor resolves to a real `<a id>` and no two aids collide) — and on the
six binding evidence-map rulings + Fable's one-verdict constraint (the
"Architect batch audit" section above; the red-team note's "Binding rulings
for the structured evidence-map rollout"). Nothing here re-proposes the
foundation; it specifies the four owed pieces (per-leg ids, executed-vs-
declared reconciliation, external-script manifests, per-leg anchors) plus
the three campaign riders the D9 handoff attached (DIDACTICS-20 stable
sections, DIDACTICS-27 wrapper-child, ruling-6 UNAVAILABLE, 09 vocabulary).

Substrate census (grounds the "~90 aids"): board.py carries 57 `ctx.expect`
sites across 40 gates and calls 26 `gates/checks/*.py` scripts; 7 gates
already carry one headline `Assertion` each. The ~90 acceptance legs =
the 57 in-process expects + ~33 reviewed check-script legs (the numeric
scripts each assert a handful; board_selftest's 142 internal `report()`
calls are the HARNESS's self-proof, not board legs — brd-a mints ~7 legs,
not 142). Minting is per-reviewed-leg; the landing enumerates the 90 names,
this proposal fixes the RULE that generates them.

### Deliverable A — the registry stable-section rewrite shape (all 40 entries)

The drift DIDACTICS-20 names: today `maps=` is free-form prose — some entries
carry note LINE NUMBERS ("148-153 (leg 1: ...)", which rot exactly as the
"32 tests" count did, DIDACTICS-24's enumeration-rot lesson), some carry long
descriptive sentences nothing checks. The rewrite kills the drift by moving
the machine truth OUT of the prose and INTO the structured map, and demoting
`maps=` to one checkable-by-a-human sentence.

Proposed post-rewrite shape of every registry entry:

- `title=` — one line, the README name (unchanged).
- `maps=` — ONE sentence: the human promise (ruling 1 keeps it), in taught
  vocabulary (Deliverable B), NO note line numbers, NO leg enumeration. It
  says what the gate proves; the `evidence=` tuple says which legs prove it.
- `evidence=` — the stable section: one `Assertion("<gate>.<leg>", "<home>.md#<gate>-<leg>")`
  PER acceptance leg. THIS is the drift-proof surface — `validate_evidence`
  already guarantees each anchor resolves and each aid is board-unique, so a
  reworded note can never silently orphan a leg (ruling 3).
- `home=` — reconciled to the ONE note that actually carries the anchors
  (ruling 5 + item 4). Several red-team gates are written up in
  gates-and-board.md while `home=` names a domain note; each such gate's
  `home=` is corrected in this pass so its legs anchor where they live.

Where the six wrapper-child dimensions live (the "required sections" of
DIDACTICS-20, enumerated by DIDACTICS-27: required files, executed
subprocesses, exact-vs-tolerant metric, check count/names, asserted-vs-logged
evidence, owed work): in the home note, at the gate's headline anchor, as ONE
labeled block per gate — not duplicated into board.py (prose in board.py is
what drifts). Proposed block, carried once per gate in its home note:

```
<a id="<gate-id>-<headline>"></a>
**<gate-id> — <one line: what the whole gate proves>.**
- files:      <inputs read / outputs written, or "none (in-process)">
- subprocess: <the driver/child executed, or "none — asserted in-process">
- metric:     <exact-byte | stabilized-rel-error <= <bar> | set-equality | selected-text-equality | lexical-scan>
- legs:       <count>, named <gate>.<leg> ...  (or, for a census leg: surface + pattern + blind-spots)
- evidence:   asserted <X> ; logged-only <Y> -> UNAVAILABLE until reconciled (ruling 6)
- owed:       <workstation / queue-5 legs, or "none">
```

DESIGN FORK A1 (for your ruling): anchor GRANULARITY. Item 4 says "each home
note gains an `<a id>` marker per acceptance leg", and ruling 5 anchors each
leg. Two readings:
  (A1-i) one full six-field block PER LEG — ~90 six-field blocks across the
    home notes. Maximum locality; heavy note surface; much of it repeats the
    gate-level files/subprocess.
  (A1-ii) [RECOMMENDED] one six-field block per GATE at its headline anchor,
    plus a LIGHTWEIGHT per-leg anchor — `<a id="<gate>-<leg>"></a>` followed
    by a one-line taught-vocabulary claim. Satisfies "anchor per leg" and
    ruling 5 without exploding each leg into six fields; the wrapper-child
    six dimensions are stated once per gate where they are actually uniform.
I recommend A1-ii; the difference is ~90 one-line anchored claims + 40 six-
field blocks (ii) vs ~90 six-field blocks (i). Your call fixes the note
surface the landing writes.

### Deliverable B — the aid-minting scheme (~90 aids)

- NAME: `<gate-id>.<plain-leg-name>` (ruling 4) — no spec-code prefix, lower
  kebab, taught vocabulary. The raw log must teach a reader; internal codes
  (45M-*, DIDACTICS-*) stay in notes/.
- ANCHOR MARKER = the aid with `.`->`-`: aid `cli-a.strict-parse` anchors at
  `#cli-a-strict-parse`. This makes item-4 reconciliation a pure string
  transform and lets `validate_evidence` check the 1:1 mechanically. NOTE:
  the 7 existing foundation anchors were hand-picked and do NOT all follow
  this (`brd-a.exit-truth` -> `#brd-a-board-truth`); this pass NORMALIZES the
  7 so every aid maps to its anchor by the one transform. (Small, mechanical,
  and it removes a second naming convention.)
- ONE STRING, THREE PLACES: the aid appears verbatim in the board.py
  `Assertion`, in the `ctx.expect(aid=...)` / check-script manifest line that
  emits it, and (dot->dash) in the note anchor. The reconciliation is exactly
  the check that these three agree.
- MINT FROM THE NARROWED CLAIMS (DIDACTICS-62/63 point 1: "the queue-2 aid
  minting uses the NARROWED claims"). Each aid names what is EXECUTED TODAY,
  not the aspirational behavioral claim. A banner-only leg mints
  `param-window-cuts.banner-present`, never `.rows-selected`, until the
  independent-reference strengthening (DIDACTICS-63, red-team gate-truth
  batch) lands. Behavioral claims not executed today are NOT minted green —
  they mint UNAVAILABLE (Deliverable D) or wait for their strengthening.
- TAUGHT VOCABULARY (09 + the reserved-term rulings): `.selected-text-equality`
  not `.byte-identical` unless the leg compares raw undecoded bytes
  (DIDACTICS-29); `.stabilized-rel-error` for the parity metric with its 1e-8
  floor / 1e-6 bar (DIDACTICS-25); honest quantifiers — a leg named `.all-*`
  or `.exactly-once` carries an adjacent assertion mechanically proving the
  quantifier or it is renamed (DIDACTICS-23 quantifier discipline). `census`
  in a leg name is reserved for a mechanically complete surface (DIDACTICS-24).

### Deliverable C — wrapper-child reconciliation order

DIDACTICS-27's rule: a wrapper (the board.py gate body) cannot upgrade a
lexical scan into runtime proof, a logged instruction into an executed test,
or `ctx.log` into PASS. Reconciling a gate = walking its six dimensions and
classifying every claimed leg as one of:
  - EXECUTED (a real comparison, in-process or in the child) -> aid mints PASS/FAIL;
  - LEXICAL / LOGGED-ONLY / visual-or-Architect-confirmed -> aid mints
    UNAVAILABLE (ruling 6) until a real assertion replaces it;
  - OWED (workstation / queue-5) -> aid mints UNAVAILABLE with an owed reason.

Proposed migration ORDER (each gate is one rerun, per the note's "gate by
gate" pattern), simplest machinery first so the reconciliation is proven
before the hardest gates use it:
  1. The 7 already-migrated gates (brd-a, gen-a, cli-a, fam-a, srm-a, arb-a,
     diag-a): expand each headline into per-leg aids; normalize the 7 anchors.
  2. Pure in-process gates (no subprocess): reconciliation is only the expect
     sites — the smallest surface, builds `expect(aid=)` + the executed-set
     recorder.
  3. Wrapper+child gates (board.py body + a `gates/checks/*.py`): adds the
     check-script `##AID` manifest mechanism (Deliverable D).
  4. Subprocess-driver gates LAST (gct, gsv, ftw, tpe): the ones carrying the
     wrapper falsehoods DIDACTICS-27 named — gsv's "h5 alone" echo, the rtol
     echo, the bitwise echo, tpe's dated prose, the gct MCMC instruction, the
     ftw/tpe provenance echoes. Their PROSE narrowing is the red team's
     factual bundle; the MACHINE disposition (minting those echo/instruction
     legs at UNAVAILABLE so the wrapper cannot green a logged instruction) is
     queue-2's. COORDINATION: the two must agree — I mint the aid non-green,
     the red team narrows the prose; neither alone is sufficient.

### Deliverable D — UNAVAILABLE labeling per binding ruling 6

Ruling 6: exactly one terminal result per aid per run — PASS / FAIL /
explicit non-green UNAVAILABLE; a missing, duplicate, unknown, check-script-
crashed, or conditionally-omitted aid reds the gate. The dead-network rule
turned on the harness itself. Mechanism:

- EXECUTED-SET RECORDER. `RunContext` grows an ordered `self._executed`
  (list of (aid, result)). `expect(*, aid, label, ok, detail="")` appends
  (aid, PASS/FAIL) and keeps its current raise-on-FAIL. A NEW
  `ctx.unavailable(*, aid, label, reason)` appends (aid, UNAVAILABLE) and
  does NOT raise — it is an honest declared non-green leg, not a gate failure.
- RECONCILIATION. After a gate body returns without GateFailure, the runner
  compares `declared = {a.aid for a in gate.evidence}` against
  `executed = {aid emitted in-process} | {aid folded from check-script
  manifests}`. Any declared-not-executed, executed-not-declared, or duplicate
  emission reds the gate via the pinned line (note ~1174, verbatim):
  `[evidence] <gate>: declared {aids} but executed {aids}`.
- CHECK-SCRIPT MANIFEST. Each `gates/checks/*.py` prints, per leg, a machine-
  readable `##AID <aid> <PASS|FAIL|UNAVAILABLE>` line; `run_check` already
  returns (rc, output), so the runner parses `##AID` lines from output and
  folds them into the executed set. A script that CRASHES before its manifest
  leaves its declared aids un-emitted -> reconciliation reds the gate (ruling
  6's "crash before its manifest"). The per-script `report()` helper gains the
  aid argument and prints the `##AID` line — one shared convention, 26 scripts.
- ONE-VERDICT CONSTRAINT (Fable's addition): where a check script already
  aggregates its own PASS/FAIL, the `##AID` lines ARE the verdict record and
  the script exit code stays the single aggregate — no second parallel verdict
  is manufactured from the same execution.

DESIGN FORK D1 (for your ruling): does an UNAVAILABLE leg make the GATE
non-green?
  (D1-i) strictest: any UNAVAILABLE leg -> the gate reports a distinct
    non-green "partial" state. Maximum honesty; but every gate with a
    workstation-owed leg goes yellow on the Mac, changing the board's colour
    meaning broadly.
  (D1-ii) [RECOMMENDED] a gate PASSES on its environment-available legs;
    UNAVAILABLE legs are recorded and displayed (the log + BOARD.md show
    "N legs owed") but do not fail the gate. The case ruling 6 targets — a
    leg DECLARED executable-here but SILENTLY dropped — still reds, because
    silence != an explicit `ctx.unavailable`. UNAVAILABLE is the honest
    "owed/lexical" terminal; a missing aid is the red case. This matches the
    existing `needs=`/SKIPPED discipline (the Mac already can't run cosmolike
    legs) without repainting the board.
I recommend D1-ii. This is the one genuine judgment call in the plan; your
ruling fixes whether the aggregate board colour changes.

### Rollout sequencing + its own validation gate

Build order (probe-the-machinery law — the mutation arms prove the
reconciliation bites BEFORE any real gate is migrated):
  1. `expect(aid=)` + `ctx.unavailable(aid=)` + the executed-set recorder +
     the reconciliation predicate in run_board.py, gated by board-selftest
     arms on a FABRICATED gate: declared-not-executed reds, double-emit reds,
     unknown-emit reds, explicit-unavailable greens-on-executed, control green.
  2. The `##AID` manifest helper + the runner's parse-and-fold, gated by a
     board-selftest arm driving a fake check script whose manifest drops a leg
     / crashes before the manifest -> red.
  3. Per-gate migration in the Deliverable-C order, each a rerun: expand
     evidence into per-leg aids from the narrowed claims, thread `aid=`, add
     the home-note anchor block(s), classify owed/lexical legs UNAVAILABLE.
  4. The Deliverable-A `maps=` one-sentence rewrite + `home=` reconciliation,
     same visit per gate (DIDACTICS-20: "one pass over that surface").

Validation gate (the handoff's "per the rollout plan once approved"):
  - `validate_evidence` green on the shipped board on every invocation incl.
    `--list`; every anchor resolves; all ~90 aids board-unique.
  - board-selftest gains the six reconciliation mutation arms above; each
    reds its fabricated gate, every control greens.
  - every migrated gate reruns green (or honest-non-green with UNAVAILABLE
    legs per D1) on the Mac where `needs=` allow; workstation-owed legs marked
    UNAVAILABLE, never silently dropped.
  - the pinned `[evidence]` line used verbatim.
  - compileall clean; the taught-vocabulary scan over the new aid names +
    `maps=` sentences (no `.byte-identical` outside raw-bytes legs; honest
    quantifiers carry adjacent proof).

### The second owed artifact — fixed-facts proposal sequencing

Per the handoff ("sequence it where your context is freshest"): my context
right now is the evidence-map machinery — board.py registry, run_board
reconciliation, the check-script manifest. The fixed-facts proposal lives in
a DIFFERENT surface (the canonical-representation / `.ranges` / units 71/74/82
artifact-schema context). I therefore sequence the fixed-facts proposal at the
HEAD of the production-units work — written when I open units 71/74/82, where
its context is freshest — NOT interleaved with this rollout. It is deferred by
sequencing, not dropped; flagging so it is not read as forgotten.

### What is NOT in scope here (red-team owned)

The wrapper-falsehood PROSE narrowing (gsv/gct/ftw/tpe echoes, tpe dated
prose) and the DIDACTICS-20 stale/malformed board.py lines (:634, :1370) are
FACTUAL-BUNDLE fixes, red-team custody. Queue 2 owns the MACHINE disposition
(minting those legs at UNAVAILABLE); the prose and the disposition must agree,
the coordination point in Deliverable C. `gates/checks/` stays exclusively the
Implementer's through D4+D5, so the `##AID` helper edits and the red team's D1
set-equality leg do not collide.

AWAITING Architect review — no code lands until the two forks (A1 anchor
granularity, D1 UNAVAILABLE-gate-colour) are ruled and the plan is approved.

## Queue-2 rollout plan adjudication (Fable, 2026-07-13): APPROVED — A1-ii and D1-ii adopted with riders; the census corrected to 57/25; the 41+53 trigger fires

The proposal above audited clause-by-clause against the six binding
evidence-map rulings + the one-verdict constraint (the checklist
law). Every clause either verifies or is corrected below; nothing
re-opens the foundation.

Verification (probe against the machinery, untruncated):

- 57 `ctx.expect(` sites in board.py — CONFIRMED exactly.
- Check-script census CORRECTED: 25, not the proposal's 26. There
  are exactly 25 literal `ctx.run_check("gates/checks/*.py")`
  sites, one per script, no constructed paths; gates/checks/ holds
  27 files but `__init__.py` is the package marker and `logscan.py`
  is an IMPORTED helper (`from checks import logscan`), never a
  child. The older rollout-spec text's "58 expects + 27 scripts"
  was the pre-hardening tree plus a directory count; current truth
  is 57 + 25. The ~90-aid estimate stands (57 in-process + ~33
  reviewed check-script legs); the landing enumerates the exact
  names, per the proposal's own rule.
- The 7 foundation aids all carry the SHORT BOARD ID prefix
  (brd-a.exit-truth, gen-a.owned-rng, cli-a.strict-parse, ...) and
  the 7 anchors are confirmed non-uniform (#brd-a-board-truth,
  #gen-a-generator-seed, #cli-a-strict-cli) — the proposed
  normalization is real work, not tidying.
- validate_evidence today enforces resolvable + unique ONLY; the
  aid->anchor dot-to-dash transform the proposal leans on ("a pure
  string transform ... checked mechanically") is a convention until
  the validator rider below lands.
- The pinned `[evidence]` line is cited verbatim correctly (:1174).

FORK A1 RULING: A1-ii ADOPTED — one six-field block per gate at its
headline anchor, plus a lightweight `<a id="<gate>-<leg>">` per leg
followed by a one-line taught-vocabulary claim. The six wrapper-child
dimensions are gate-level facts; ~90 six-field copies would recreate
the duplication drift this rollout exists to kill. RIDER (mixed
metrics): the block's `metric:` field is a quantifier over legs and
carries the quantifier discipline — where legs differ in metric it
says "per-leg" and each leg one-liner names its own; a single metric
may be stated at gate level only when every leg actually uses it.

FORK D1 RULING: D1-ii ADOPTED — a gate PASSES on its
environment-available legs; UNAVAILABLE legs are recorded and
displayed; a DECLARED leg that is silently dropped (no expect, no
explicit ctx.unavailable) still reds via reconciliation. TWO RIDERS:

1. ZERO-EXECUTED GUARD: a gate whose executed set is EMPTY (every
   declared leg UNAVAILABLE) may not PASS — it reports a distinct
   non-green state. The dead-network doctrine applied to the harness
   itself: a gate that executed nothing must not read as proof.
   Board-selftest arm: a fabricated all-unavailable gate must not
   report PASS.
2. DISPLAY TRUTH: the per-gate result line is pinned —
   `PASS (executed N/M; UNAVAILABLE K: <aid> <aid> ...)` — and
   appears in the gate log, --list, and BOARD.md alike. A bare PASS
   while the recorder holds UNAVAILABLE legs is forbidden and
   selftest-armed. UNAVAILABLE reasons (owed-workstation / lexical /
   environment) travel with the aid in the log.

RULING-4 CLARIFICATION (aid prefix, flagged to the red team as
register owners): the `<gate-id>` in an aid is the gate's BOARD ID —
the exact string `--gate` accepts (brd-a, gsv, gct, ...) — not the
README title. Ruling 4's example (`board-selftest.exit-truth`) was
illustrative; its substance (no internal audit-code prefixes; plain
leg names; the raw log must teach) is untouched. Rationale: a red
aid line is directly actionable (`run_board.py --gate brd-a` reruns
exactly the failing gate), and the short id is already the board's
log-public primary key (BOARD.md rows, status keys, selectors). The
landed foundation used this convention and was audited GO; this
clarification records it as deliberate.

VALIDATOR RIDER: validate_evidence gains invariant (3) — every
Assertion's anchor fragment equals its aid with "." -> "-". The
7 foundation anchors normalize in the AID->ANCHOR direction (the
note markers are renamed to the transform; 5 notes touched,
mechanical). Selftest mutation arm: a non-transform anchor exits 2
on --list.

APPROVED AS PROPOSED: the Deliverable-A entry shape (`maps=` one
human sentence, `evidence=` per-leg, `home=` reconciled); the
Deliverable-B minting rule (narrowed claims only; unexecuted
behavior never mints green; the reserved-term vocabulary); the
Deliverable-C order (7 migrated -> in-process -> wrapper+child ->
subprocess-driver last) with its red-team coordination clause
(machine disposition = queue 2, prose narrowing = the factual
bundle, the two must agree); the Deliverable-D mechanism (recorder,
ctx.unavailable, ##AID manifests, crash-before-manifest reds, the
one-verdict constraint); the rollout sequencing (fabricated-gate
mutation arms prove the reconciliation bites BEFORE any real gate
migrates — the probe-the-machinery law, correctly internalized);
the validation gate, EXTENDED by the two new arms from the D1
riders and the transform arm; and the fixed-facts proposal
sequenced at the head of the production-units work — ACCEPTED as
flagged (deferred by sequencing, not dropped; unit 74 stays
CRITICAL there).

TRIGGER FIRES: this approval is the Wave-2 trigger for UNITS 41+53;
the transfer handoff is issued this turn. Scope boundary made
explicit: unit 41 transfers ONLY its 25M-05 sweep-products
extension (metadata from ONE immutable resolved run record;
activation-family sweeps record "swept" + ordered values; serial
and pooled paths share the record; banner/artifact/table/figure
agree). Unit 41's transfer-refine artifact core (the 45M-33
drift-metric amendment) belongs to the fixed-facts/artifact chain —
NOT pre-authorized, stays with the Implementer. Unit 53 transfers
WHOLE (the 25M-04 study-name resolver amendment: one pure resolver,
family identity -> stable study name, resolved name in the study
manifest and final report, renames are explicit migrations; the
false comment corrected in the same landing).

The Implementer builds in the proposed order (machinery + mutation
arms, then ##AID, then per-gate migration with the maps=/home=
rewrite per visit), then D3 -> D4 -> D5. Every landing gets the
standard pre-merge audit; the riders above are acceptance clauses,
not suggestions.

## Aid-prefix RE-RULING (Fable, 2026-07-13): gate.id, not the spec code — the ruling-4 "clarification" is RETRACTED

The Implementer blocked on the aid-prefix clause of the adjudication
above and filed three factual claims against its premise. All three
verified by me against the machinery (the raw evidence, not the
filing):

- `--gate brd-a` is REJECTED live: "error: --gate names unknown
  gate id(s): 'brd-a'" (the 45M-77 strict refusal), while
  `--gate board-selftest` passes id validation (it proceeds to the
  action-mode check). My "run_board.py --gate brd-a reruns exactly
  the failing gate" was not a runnable command.
- brd-a/gen-a/gsv/gct are the LOWERCASED SPEC CODES:
  `Gate(id="board-selftest", spec_code="BRD-A", ...)`, and
  board.py's own field docstring defines spec_code as "the key of
  this test's audit-history entry inside its home" — an internal
  audit code, exactly the class ruling 4 excludes from assertion
  identifiers.
- The board's log-public primary key is gate.id: BOARD.md rows
  concatenate gate.id (run_board.py:2143), `_registry_ids()` — "the
  selector validation authority" — collects gate.id, and an
  untruncated grep finds spec_code NOWHERE in run_board.py.

RE-RULING (replaces the "RULING-4 CLARIFICATION" paragraph of the
adjudication above; everything else there stands): the aid prefix is
`gate.id` — `board-selftest.exit-truth`, `save-rebuild-drift.<leg>`.
Ruling 4's example was LITERALLY CORRECT as written; it was the
landed foundation that deviated by keying to the spec code, and my
adjudication preserved the deviation under an unverified rationale.
The rationale itself survives the flip: under gate.id a red aid line
IS directly actionable (`--gate save-rebuild-drift`) and DOES match
the BOARD.md primary key — the two properties I wanted were only
ever true of gate.id.

CONSEQUENCE (supersedes the anchor-only normalization): the 7
foundation aids are RE-KEYED to gate.id, and the 7 anchors follow
via invariant (3)'s transform (e.g. brd-a.exit-truth ->
board-selftest.exit-truth, anchor #board-selftest-exit-truth; 5
notes touched). The proposal's `<gate-id>` placeholders now read as
gate.id everywhere, including the six-field block template.

THE ERROR, recorded: I probed every clause of the proposal EXCEPT
the one I ruled on from memory — counts, anchors, validator,
pinned line all verified; the selector vocabulary asserted from the
notes prose pattern "Gate board-selftest (BRD-A)". The
probe-against-the-machinery law applies to MY OWN premises exactly
as it applies to filings: a ruling clause that names a runnable
command must have run it. The Implementer's block — refusing to
build on a premise it could falsify — is the protocol working.

RETRACTION rider: the previous handoff asked the red team to record
a register note softening ruling 4's example. WITHDRAWN — the
example stands as written; no register edit is needed.

CONFIRMED to the Implementer: proceed with increment 1 as scoped in
the blocking handoff — the reconciliation machinery (expect(aid=),
ctx.unavailable, executed-set recorder, PASS-on-available +
zero-executed guard + pinned display), validate_evidence invariant
(3), the 7 foundation aids re-keyed to gate.id + anchors
normalized, and the board-selftest mutation arms — Mac-verified
(--list rc 0 + board-selftest green) before handoff; ##AID
manifests and per-gate migration follow as separate increments.

## D2 factual-increment audit (Fable, 2026-07-13): 9b59e0e — GO for merge

The red team's first campaign landing (DIDACTICS D2, the factual
bundle) on codex/architect-docs-static-audit, 29 files: four READMEs
(root, emulator/, gates/, syren/), 22 Python doc surfaces under
emulator/ + cobaya_theory/, and their register. Every evidence claim
re-derived independently, not accepted:

- AST (my own comparator, parent 238d774 vs 9b59e0e, all 24 .py):
  20 files AST-identical with docstrings stripped; 4 files differ
  ONLY in human-facing runtime strings (losses/cmb.py, results.py,
  training.py, warmstart.py — voice/punctuation plus warmstart's
  clearer anchor refusal "train_args.finetune.anchor is not
  available. Remove the anchor key ..."); ZERO structural changes.
  Matches the register's own four-file disclosure exactly.
- compileall (emulator, cobaya_theory, syren, compute_data_vectors,
  gates/checks) clean and board_selftest ALL PASS — MY runs, not
  the filing's.
- SONIC expands exactly to "Simulated Observables for Numerical
  Inference in Cosmology" (README:21); an untruncated scan finds no
  retired expansion.
- Residual-block order: the rewritten docstring (final Linear ->
  skip addition -> normalization -> activation) matches the executed
  forward (blocks.py: the i == n-1 branch adds xskip after
  layers[i], before acts[i](norms[i](out))). The shared-MLP
  "textbook" overclaim is narrowed to the fixed-width dim->dim
  truth.
- losses/cmb.py rewrite VERIFIED as the 44/95-class corrections
  with the teaching preserved: the amplitude-law transform (f =
  (A_s_ref/A_s) * exp(2(tau - tau_ref))), encode/decode/score
  directions, AMPLITUDE_LAWS registry all survive; "cosmic-variance
  scale" is corrected to "the positive scale stored in the
  covariance product", and "the amplitude generalizes for free" is
  narrowed to "removes the dominant primary-CMB amplitude trend but
  does not make the remaining target independent of all amplitude
  information". The retired-law diary paragraph is gone per the
  current-state ruling.
- warmstart.py:20-33 now carries EXACTLY the 71 two-invariant
  split (_PARITY_TOL numerical reproduction with the
  reduction-order reason; torch.equal on the zero-connected-extras
  arm), the 56 state_dict frozen-parameters fact, and the 69
  whitening sentence; the 69 census is CLEAN (zero "equally
  hard"/"equally easy" in emulator/ at the commit).
- results.py teaches the executed CPU normalization (detach + CPU
  before torch.save; map_location selects the destination) — the 70
  prose fix.
- The README CMB workflow now shows the independent
  dataset_generator_cmb.py / compute_cmb_covariance.py branches.
- gates/README.md's one row (bsn-identity) is an HONEST NARROWING
  consistent with unit 90's confirmed finding: "vs closed-form flat
  LCDM ... save/rebuild bitwise" becomes "numerical consistency of
  the production distance integrator on analytic fixtures ...
  save/rebuild identity" — the row now claims what is executed;
  the independent-quadrature strengthening remains unit 90 (their
  custody).
- Ruling 4 restored BYTE-EXACT in the register (the withdrawn
  brd-a.exit-truth clarification note removed) — closing my
  retraction's loop.
- DIDACTICS-79 HOLD ACCEPTED as filed: the false generator command
  is removed, --unif/--seed requiredness was proven by executing
  the parser in isolation, and NO replacement command was invented
  because none can be executed in their environment — the
  executed-before-printed rule applied correctly, the item stays
  open with a durable-record explanation. The register also
  explicitly refuses to bless the remaining history-vocabulary in
  the READMEs (later current-state visits).
- COLLISION CHECK: no gates/board.py, gates/run_board.py, or
  gates/checks/*.py edits — the Implementer's exclusive surfaces
  untouched while increment 1 is in flight (confirmed in flight as
  UNCOMMITTED working-tree edits on amazing-keller; the branch tip
  is still my audited dbc7f16, so the landing order below sweeps in
  nothing unaudited).

VERDICT: GO. Landing order: the codex branch first, then
claude/amazing-keller-e798b6 (disjoint files, no conflict — the
codex commit touches no note my branch touches). Units 41 + 53 may
start at the red team on this landing, per their handoff's own
sequencing. Unit 93 remains unclaimed and available.

## THROUGHPUT REBALANCE (user ruling, 2026-07-13): red-team speed is now an explicit division criterion

USER RULING (after the D2 GO): the red team's demonstrated pace is
taken into account when dividing work — the backlog must finish
faster. Calibration recorded with the ruling: D2 was doc-only under
an AST-identity constraint (intrinsically faster than machinery
increments), so the acted-on signal is QUALITY-PER-AUDIT — the red
team is ~60-for-60 on verified filings and its first code-adjacent
landing passed audit clean on the first pass. Governance is
unchanged: Architect pre-merge audit on every landing, no
self-certification, one owner per file at any moment.

TRANSFERS EFFECTIVE NOW (zero collision with increment 1/2):

1. QUEUE-2 NOTE-SIDE EVIDENCE BLOCKS. The red team drafts the
   home-note evidence surface for the board per the approved A1-ii
   template: one six-field block per gate (files / subprocess /
   metric / legs / evidence / owed) at the headline anchor plus a
   lightweight per-leg anchor + one-line claim, leg names minted by
   the frozen Deliverable-B rule (gate.id prefix, narrowed claims,
   taught vocabulary, reserved terms). Rationale: this is the
   rollout's largest WRITING surface, most narrowed claims trace to
   the red team's own filings, extra <a id> markers are inert until
   a gate declares them (validate_evidence checks only declared
   anchors), and the notes/ side collides with nothing the
   Implementer's increments touch except the seven foundation
   anchors ALREADY being re-keyed — those seven are EXCLUDED from
   the draft (the Implementer lands them in increment 1). Division
   inside the surface: the red team drafts every gate EXCEPT the
   four subprocess-driver gates (gct / gsv / ftw / tpe wrapper
   gates), whose leg classification is entangled with the
   wrapper-falsehood machine disposition — the Implementer names
   those legs during migration. I audit the drafted blocks; the
   Implementer treats audited names as the spec and files a naming
   objection per gate where migration disagrees.
2. ENVIRONMENT PROBE (gates further transfers). The red team
   reports whether the cocoa interpreter
   (Cocoa/.local/bin/python, torch 2.6.0 CPU+MPS) executes in
   their sandbox — evidence: one torch import + one small forward
   pass printed. The answer decides whether torch-executing units
   (D3's smoke-fixture repair, most D4/D5 legs) can transfer with
   local verification or only in write-here/verify-there form (the
   unit-93 precedent).

PRE-AUTHORIZED AT THE INCREMENT-2 SEAM (fires when the Implementer
lands the ##AID manifest increment and gates/checks/ goes quiet;
same mechanism as the Wave-2 table — no fresh ruling needed):

3. D6 (mkdtemp hygiene, all nine sites) — whole.
4. 61-FINITENESS — whole (checks/logscan.py decreasing: both
   endpoints finite + the five named controls as board-selftest
   legs); pure-CPU, runnable in their environment.
5. D3 (the unit-28 smoke-fixture disjointness repair) — IF the
   environment probe says torch runs locally; otherwise it stays
   with the Implementer (a gauntlet-critical repair must be
   executed by its author, not verified blind).
6. D4 SPLIT DECISION at the same seam: check-script legs whose
   files the ##AID sweep has finished may transfer file-by-file
   (one owner per file at any moment); board.py legs (46's
   _golden_leg + addendum) stay with the Implementer.

UNCHANGED — THE IMPLEMENTER'S CORE (kept deliberately; the reason
is capability and stakes, not habit): the queue-2 machinery +
board.py migration (one file, the harness itself), the fixed-facts/
artifact chain (74/76/77/78/80/82/84/85 — production artifact truth
whose legs execute torch save/rebuild), unit 8, unit 96, the
queue-3 staging reopen. Stripping everything else away is what
shortens that queue; moving its members would trade the most
expensive class of error for throughput.

The gauntlet's critical path is unchanged (increments 1-2 -> D3 ->
D4 -> D5 -> queue 5); this rebalance shortens the TAIL behind it
and parallelizes the note surface under it.

## Queue 2 increment 1 COMPLETE — reconciliation machinery + the 7-aid re-key (Opus, 2026-07-13)

The approved increment 1 is landed and Mac-verified (cocoa-torch not needed —
board-selftest is pure Python). Scope exactly as ruled; ##AID manifests and the
per-gate migration are the following increments.

**What landed:**

- `RunContext` (run_board.py): `self._executed` recorder; `expect(*, ..., aid=None)`
  records `(aid, PASS/FAIL, detail)` when an aid is given (un-migrated legs stay
  invisible — rolling migration); new `unavailable(*, aid, label, reason)` — the
  honest non-green terminal that does NOT raise (fork D1-ii).
- `validate_evidence` invariant (3): every anchor fragment must equal its aid
  with `.`->`-`. Runs on every invocation incl. `--list` (exits 2 on violation).
- `_reconcile_evidence(gate, executed)` + `_evidence_cell` + `_fmt_set`: binding
  ruling 6 — declared-not-executed, executed-not-declared, and emitted-twice red
  the gate via the pinned `[evidence] <gate>: declared {..} but executed {..}`
  line; fork D1-ii PASS-on-available; the ZERO-EXECUTED GUARD (all-UNAVAILABLE
  may not PASS). Wired into `run_selection` (reconciles only an otherwise-passing
  gate), persisted as `verdict["evidence"]`, and rendered by the shared
  `_state_detail` so the pinned `PASS (executed N/M; UNAVAILABLE K: <aid> ...)`
  line shows in the gate log, `--list`, and BOARD.md alike.
- The 7 foundation aids RE-KEYED to the gate.id prefix (per the aid-prefix
  re-ruling): `board-selftest.exit-truth`, `generator-seed.owned-rng`,
  `cli-strict.strict-parse`, `family-first.family-owned`, `stage-ram.both-copies`,
  `artifact-readback.typed-bool`, `diagnostics-domain.score-boundary`; their 7
  note anchors renamed to the transform (5 notes touched + one prose spec pointer
  in gates-and-board.md). Each gate body's result `expect` now carries its
  headline `aid=`, so the declared leg actually executes and reconciles.
- board-selftest: `check_evidence_map` mutations transform-isolated (1/2 anchors
  made transform-correct so they isolate the unresolved / missing-note defect; 3
  uses a real re-keyed anchor so the duplicate is isolated) + new mutation 5
  (non-transform anchor rejected) + the CLI arm (a non-transform anchor makes the
  REAL `main --list` exit 2). Two new check functions: `check_evidence_reconciliation`
  (drives the REAL `_reconcile_evidence` — control-green, declared-not-executed,
  emitted-twice, executed-not-declared, mixed-PASS+UNAVAILABLE-green, zero-executed
  guard) and `check_evidence_gate_verdict` (drives the REAL `run_selection` via
  `drive_main`, validate_evidence patched off: a mixed gate ends PASS with the
  persisted block, an all-UNAVAILABLE gate ends FAIL, a silently-dropped leg reds).

**Verification (Mac):** `py_compile` + `compileall gates` clean; board-selftest
**155 PASS / 0 FAIL** (was 142 — +13 legs); `run_board --list` rc 0 (all 7
re-keyed anchors resolve + satisfy invariant 3, all aids unique). Every mutation
arm reds its fabricated gate; every control greens.

**Two implementation choices flagged for the pre-merge audit:**

1. A reconciliation failure (mismatch OR the zero-executed guard) sets the gate's
   `outcome = "FAIL"` with a distinguishing DETAIL (the `[evidence]` line), rather
   than minting a NEW named status. Rationale: FAIL is already a fully-plumbed
   non-green terminal (resume, exit code, display, counts); a new state would need
   threading through `_resume_state`, `cmd_list`, `_write_board_md`, and the exit
   rule for no behavioral gain. The ruling's "distinct non-green state" is met by
   the detail. If the audit prefers a distinct label, it is a small follow-up.
2. Reconciliation runs ONLY on an otherwise-passing gate: a gate that already
   FAILED (a raised `expect`) is non-green for a real reason, and its raised leg
   left `_executed` deliberately short — reconciling it would report spurious
   declared-not-executed noise on an already-red gate.

**Trigger already fired:** the rollout-plan approval freed UNITS 41 (25M-05 slice
only) + 53; recorded here so the transfer is on the ledger.

**Next:** increment 2 = the `##AID` check-script manifests (parse-and-fold in
`run_check`, crash-before-manifest reds) with its board-selftest arm; then
increment 3 = per-gate migration in the Deliverable-C order (7 migrated ->
in-process -> wrapper+child -> subprocess-driver last) with the `maps=` one-
sentence rewrite + `home=` reconciliation per visit.

## Queue-2 increment-1 audit (Fable, 2026-07-13): GO — audited PRE-commit, landed as d1896ce

The Implementer delivered increment 1 uncommitted and asked for a
commit; I audited the working tree BEFORE any commit existed (a
strictly stronger ordering than the session's usual land-then-audit)
and committed it myself on GO. 8 files, 467 insertions / 42
deletions. The ruling walked as a checklist against the diff:

- expect(aid=) records (aid, PASS/FAIL, detail), keeps raise-on-FAIL;
  aid=None stays invisible, so the migration is rolling and the 33
  evidence-less gates are untouched.
- ctx.unavailable(aid, label, reason) is the explicit non-green
  terminal; the reason travels into the log, and the docstring
  teaches exactly ruling 6 + D1-ii.
- _reconcile_evidence: declared-not-executed / executed-not-declared
  / emitted-twice each red an otherwise-passing gate through the
  pinned line, which appears VERBATIM with an additive parenthetical
  taxonomy (declared-not-executed: ... ; emitted-twice: ...) --
  RATIFIED as an enhancement, the fixed prefix intact. A FAILED gate
  is left unreconciled (already red for a real reason).
- Zero-executed guard present: n_pass == 0 reds with "a gate that
  proved nothing may not PASS".
- Pinned display in ALL THREE surfaces: the footer prints
  "PASS (executed N/M; UNAVAILABLE K: <aids>)", verdict["evidence"]
  persists the block, and the SHARED _state_detail (the 25M-28
  helper) appends "[evidence: ...]" so --list and BOARD.md cannot
  disagree with the log.
- validate_evidence invariant (3): anchor fragment must equal the
  aid with "." -> "-"; clear error message naming both.
- All 7 foundation aids re-keyed to gate.id with leg names retained
  (board-selftest.exit-truth, generator-seed.owned-rng,
  cli-strict.strict-parse, family-first.family-owned,
  stage-ram.both-copies, artifact-readback.typed-bool,
  diagnostics-domain.score-boundary); all 7 note anchors renamed to
  the exact transforms (5 notes); aid= threaded into each foundation
  wrapper's rc==0 expect so every declared leg is EXECUTED --
  reconciliation is coherent on the shipped board from day one.
- Selftest arms EXCEED the ruled minimum: beyond the predicate arms
  (clean control, declared-not-executed, emitted-twice,
  executed-not-declared, mixed-passes, all-UNAVAILABLE-must-not-
  PASS, non-transform anchor red + --list exit 2), the increment
  adds REAL-RUNNER arms driving fabricated gates through
  run_selection itself (mixed ends PASS with the persisted evidence
  block; all-UNAVAILABLE ends FAIL; a silently-dropped leg reds) --
  the probe-the-machinery law applied by the author.

My own verification (not the filing's): compileall clean;
PYTHONPATH=. python3 gates/run_board.py --list rc 0 (the validator
green over the re-keyed board, so all 7 transforms resolve); board
selftest ALL PASS; and a direct in-process tamper probe -- a
fabricated gate declaring a RESOLVABLE but non-transform anchor --
reds validate_evidence with the exact expected message (no tree
mutation; the probe ran against the machinery, not through it).

VERDICT: GO. Landed as d1896ce (Architect-committed, Opus
co-authored). INCREMENT 2 (##AID manifests) FIRES NOW -- its landing
is the pre-authorized transfer trigger (D6, 61-finiteness,
D3-if-torch, D4 file-by-file).

COMMIT-AUTHORITY NOTE for the user: the Implementer asked whether it
may self-commit board-code landings to keep throughput up. That
grant is the user's, not mine (branch commits are user-owned by
default, with explicit time-boxed authorizations on record). Interim
protocol used here -- audit-then-Architect-commit -- keeps increments
moving without a new grant. My recommendation if speed matters: a
time-boxed self-commit grant through the gauntlet (increments 2..n +
D3/D4/D5), since every landing still gets my pre-merge audit before
any merge to main.

## COMMIT GRANT (user, 2026-07-13): Implementer self-commits branch landings through the gauntlet

The user granted the commit authorization requested in the
increment-1 audit. Scope as recommended and now binding:

- The Implementer SELF-COMMITS its landings on
  claude/amazing-keller-e798b6, TIME-BOXED through the gauntlet:
  queue-2 increments 2..n, the per-gate migration, D3, D4, D5. The
  box closes when D5 lands; anything after needs a fresh grant.
- Unchanged safety: every landing is still handed off for my
  pre-merge audit BEFORE any merge to main, and merge/push to main
  remains user-only. The grant removes the Architect-as-committer
  serialization point, not the audit.
- A landing commit that my audit then REDS is repaired by a
  follow-up commit on the branch (never a rewrite of a pushed/
  merged commit); the audit records both shas.
- I keep committing my own notes (adjudications, audits, ledger),
  as throughout.

## Root README cleanup audit (Fable, 2026-07-13): 46efa6d + cbdd49e — GO

Red-team follow-up on branch codex/readme-code-map-dedup (a NEW
worktree; lineage verified: D2's 9b59e0e is an ancestor, so no
collision with the pending D2 merge). Two bounded README changes +
register records, both verified at the diff:

- 46efa6d removes the DUPLICATE "Code map" bullet from Contents;
  the explanatory sentence with the same emulator/README.md link
  remains directly above (README:79). My untruncated scan confirms
  exactly one code-map navigation reference remains (the other hit
  is the verb "maps" in prose).
- cbdd49e replaces the saved-artifact paragraph's development-ledger
  language ("Known integrity gap (fix queued)... until the binding
  lands", plus a notes/ pointer in the public README) with the
  direct user action: keep the .h5/.emul pair together, never mix
  path roots, never replace only one member. The current-state
  ruling applied correctly: consequence + user action stay, the
  queued-fix diary goes, the engineering contract remains in its
  owner note (their register says so explicitly). The retained
  claims are true (the .emul holds best-epoch weights; the .h5
  carries the resolved config; rebuild_emulator reads the stored
  recipe, never later defaults). The units-77/80 production
  advisory is UNAFFECTED — it lives in notes, and the user-facing
  warning that motivated it is preserved.

VERDICT: GO. The register carries both records with the
no-self-certification line. Landing: merge
codex/readme-code-map-dedup (it contains D2 + this cleanup), then
amazing-keller as usual.

## Queue 2 increment 2 COMPLETE — the ##AID check-script manifest fold (Opus, 2026-07-13)

Increment 2 is landed and Mac-verified. It builds the MECHANISM that lets an
external check script's per-leg legs reconcile like in-process ones; it does NOT
migrate the 25 scripts to emit their real aids (that is increment 3, which wires
to the red team's audited naming drafts). No real script emits `##AID` yet, so the
live board is unchanged (`--list` rc 0).

**What landed (run_board.py + board_selftest.py only):**

- `_parse_aid_manifest(output)` + the `_AID_MANIFEST_PREFIX = "##AID"` constant:
  scans a check script's captured output for reserved
  `##AID <aid> <PASS|FAIL|UNAVAILABLE> [reason...]` lines, returns
  `(records, malformed)` — records are `(aid, result, reason)` (the reason keeps
  its spaces for an UNAVAILABLE leg); non-manifest output is ignored so a script
  keeps its human CHECK lines.
- `run_check` now FOLDS the parsed records into `ctx._executed` after the
  subprocess returns, so a script's per-leg legs join the executed set and
  reconcile exactly like in-process `expect(aid=)` legs. An UNPARSEABLE `##AID`
  line raises `GateFailure` immediately (a machine-unreadable manifest is a
  check-script contract violation), regardless of `allow_fail`.
- Binding ruling 6 for external checks is now closed: a script that DROPS a
  declared leg (the aid is never folded) reds via reconciliation; a script that
  CRASHES before its manifest emits nothing, so its declared leg is missing and
  reds; a malformed line reds. The one-verdict constraint is enforced by
  reconciliation's duplicate rule — a leg counted by BOTH the script's `##AID`
  and a gate-body `expect(aid=)` is emitted-twice and reds (so a migrated gate
  records ONE verdict per leg, never a second parallel one).

**board-selftest `check_aid_manifest`** drives the REAL parser, the REAL
`run_check` subprocess, and the REAL `run_selection` over tiny temp check scripts
(auto-cleaned via `TemporaryDirectory`): the parser unit legs (records + reason;
malformed flagged); a both-legs script folds green; a dropped-leg / crash /
malformed / double-counted-leg script each reds. **162 PASS / 0 FAIL** (was 155,
+7 legs); `--list` rc 0; compile clean.

**Increment 2's landing is the pre-authorized WAVE-2 TRIGGER** (throughput
rebalance f0aa2a9): D6, 61-finiteness, D3 (pending the red team's torch probe),
and the D4 file-by-file split leave the Implementer queue for the red team on
this landing; **46's `_golden_leg` repair (board.py) stays with the Implementer.**

**Next:** increment 3 = per-gate migration in the Deliverable-C order (7 migrated
-> in-process -> wrapper+child -> subprocess-driver last), each gate's per-leg
aids wired to the red team's AUDITED home-note naming draft (per-gate objection
where migration disagrees; the 4 subprocess-driver wrappers + the 7 foundation
re-keys are Implementer-named), with the `maps=` one-sentence rewrite + `home=`
reconciliation per visit. Increment 3 waits on the audited drafts.

## Queue-2 increment-2 audit (Fable, 2026-07-13): ONE REQUIRED REPAIR — a manifest FAIL under a zero exit code reconciles GREEN

The ##AID fold layer audited in the working tree (run_board.py +
board_selftest.py + the notes resume; no check-script or board.py
edits — the step-2 scope exactly). VERIFIED GOOD: _parse_aid_manifest
(reserved-prefix scan, reason kept with internal spaces, malformed
lines collected); run_check raising on an unparseable ##AID line
regardless of allow_fail (a contract violation is never tolerated);
the fold into ctx._executed; and five real-runner selftest arms
(fold-green control, dropped-leg red, crash-before-manifest red,
unparseable-line red, script+body double-count red under the
one-verdict constraint). My own runs: compileall clean, --list rc 0,
board-selftest ALL PASS.

THE DEFECT (found by walking the two increments' COMPOSITION, then
reproduced against the machinery): _reconcile_evidence's passing
path still assumes "a FAIL raised GateFailure and never reaches
this path" — true for in-process expects, FALSIFIED by the fold,
which appends FAIL records without raising. A check script whose
own aggregation is buggy — it prints ``##AID <aid> FAIL`` yet exits
0 (exactly the D-SPE2-5/DIDACTICS-62 verdict-aggregation class this
program exists to catch) — passes the wrapper's rc==0 expect, and
reconciliation RELABELS the FAIL as UNAVAILABLE. Reproduced
in-process: declared {good-leg, bad-leg}, executed [(good-leg,
PASS), (bad-leg, FAIL)] -> ok=True, display "executed 1/2;
UNAVAILABLE 1: probe-fail.bad-leg". A recorded failure absorbed as
an owed leg; the harness quietly reinterprets contradictory
evidence — the precise anti-pattern binding ruling 6 kills.

REQUIRED REPAIR (blocks the increment-2 commit):

1. _reconcile_evidence's passing path reds on ANY record whose
   result is FAIL, with a named line, e.g. ``[evidence] <gate>: leg
   <aid> recorded FAIL while the gate body passed (the check's
   manifest contradicts its exit code)``. The scan runs BEFORE the
   n_pass / unavailable split, so a FAIL is never counted into
   either bucket.
2. A selftest arm through the real runner: a fabricated script
   prints one PASS leg and one ``##AID <aid> FAIL`` leg and exits 0;
   the wrapper's rc==0 expect passes; the gate must end FAIL with
   the named line. Control: the same script with both legs PASS
   stays green.

Under the commit grant: repair, self-commit the corrected increment
(one commit, the whole layer), and hand off the sha for the delta
audit. The four transfer triggers announced on this landing HOLD —
they fire when the repaired increment commits, not before.

Note for the record: increments 1 and 2 were each internally
consistent; the hole lives in their composition (a comment in 1
made an assumption 2 invalidated). Composition re-walks — re-reading
the earlier increment's assumptions against the new one's channels —
join the audit checklist alongside the clause walk.

### Increment-2 follow-up repair — the folded-FAIL composition hole (Opus, 2026-07-13)

Architect audit ab07a2e caught a composition hole in `e193097`: the `##AID` fold
is a NON-raising channel, so a check that prints `##AID <aid> FAIL` while exiting
0 passes the wrapper's `rc==0` expect and reaches `_reconcile_evidence`'s passing
path with a FAIL in its executed set — where increment 1's PASS/UNAVAILABLE split
silently relabeled it UNAVAILABLE (and the gate could PASS). Increment 1's "a FAIL
always raised GateFailure" assumption is falsified by the fold channel.

Repair (follow-up commit, per the self-commit grant): `_reconcile_evidence`'s
passing path now scans every declared leg for a FAIL record BEFORE the
PASS/UNAVAILABLE split and reds on any, with the named line
`[evidence] <gate>: leg <aid> recorded FAIL while the gate body passed (the
check's manifest contradicts its exit code)`. board-selftest `check_aid_manifest`
gains the real-runner arm: a fabricated script emitting one PASS + one FAIL leg at
exit 0 (wrapper expect passes) ends the gate FAIL; an all-PASS control stays green.
**164 PASS / 0 FAIL** (was 162); `--list` rc 0; compile clean.

## Increment-2 delta audit (Fable, 2026-07-13): 14c88a3 GO — increment 2 COMPLETE; the four transfers FIRE

The follow-up repair audited at the diff and re-probed with the
ORIGINAL reproduction: declared {good-leg, bad-leg}, executed
[(good, PASS), (bad, FAIL)] now reds with exactly the specified
line ("recorded FAIL while the gate body passed (the check's
manifest contradicts its exit code)"); the PASS + explicit
UNAVAILABLE control stays green. The FAIL scan runs BEFORE the
PASS/UNAVAILABLE split as required; the two real-runner arms landed
(##AID FAIL at exit 0 reds; all-PASS control green); the
increment-1 comment is rewritten to teach both emission channels.
My runs: --list rc 0, board-selftest ALL PASS (164).

VERDICT: GO. Increment 2 is COMPLETE (e193097 + 14c88a3). THE FOUR
PRE-AUTHORIZED TRANSFERS FIRE with this record: D6 (all nine
mkdtemp sites, whole), 61-finiteness (checks/logscan.py decreasing
+ its five control legs, whole), D3 (the unit-28 smoke-fixture
repair — conditional on the red team's still-owed torch probe), and
D4 file-by-file. D4 transfer protocol: the red team CLAIMS a
check-script file by naming it in a handoff BEFORE editing (one
owner per file at any moment; the Implementer may veto a
collision); 46's _golden_leg (board.py) stays with the Implementer
and is already approved to build. Where verification needs torch
the red team's environment lacks, the write-here/verify-there form
applies (the unit-93 precedent) until the probe answers.

## SUBAGENT RULE (user, 2026-07-13): Implementer handoffs request subagent fan-outs where the work parallelizes

USER RULING: when the Architect hands the Implementer new work, the
handoff requests subagents where the work admits them. Precedent:
the 45M-86..90 didactics units were drafted by gated sub-agents
under a strict AST-identity check, then independently re-verified
before commit — that discipline is the template, now standing:

- The handoff NAMES which deliverables parallelize (e.g. increment
  3's per-gate migration fans out per gate once the pattern is
  proven serially on the first two or three gates) and which stay
  serial (shared-file machinery edits; anything where increments
  interlock).
- Every subagent draft passes the SAME per-landing acceptance as
  first-hand work (selftest/compile/AST gates as applicable), and
  the Implementer independently re-verifies before self-committing —
  a subagent is a drafting tool, never a verification substitute.
- My pre-merge audit is unchanged and does not care which hands
  drafted a diff.

## Unit 46 COMPLETE — the golden-leg both-rc + non-empty-selection repair (Opus, 2026-07-13)

DIDACTICS-46 + the rc addendum, landed serially (single-file board.py change per
the subagent rule). `_golden_leg` discarded BOTH child return codes (`_, cur` /
`_, pre`) and compared whatever the pattern selected, so a child that crashed
after its last matching line, or a pattern matching nothing on both sides, passed
byte-identity vacuously (the empty-selection green was live-reproduced).

Repair (board.py `_golden_leg`): capture both child rcs (`cur_rc, cur` /
`pre_rc, pre`); require BOTH rc == 0 AND a non-empty selected-line count (via
`logscan.matching_lines`, so `byte_identity`'s signature is unchanged for its
other callers) AND equality; the verdict detail always reports both rcs + both
counts (`rc pre=.. cur=..; selected pre=.. cur=..`), and names the reason(s) on
failure (nonzero child rc / empty selection / the byte divergence).

board-selftest `check_golden_leg` drives the REAL `board._golden_leg` via a stub
ctx (`_GoldenCtx`) feeding controlled child (rc, output) pairs: a clean rc0/rc0
identical non-empty selection greens (control); a diverging line, an empty
selection, both-children-rc-1-after-matching-lines, and a tip-only-rc-1 each red.
**169 PASS / 0 FAIL** (was 164, +5); `--list` rc 0; compile clean. ("Minimum"
selected-line count is read as non-empty / >= 1; a higher floor is a one-line
change if the audit wants it.)

## README figures + didactic rewrite audit (Fable, 2026-07-13): 701d6f9 — GO

The red team's root-README visit (codex/readme-code-map-dedup):
1249 insertions / 939 deletions on top of D2, three manuscript
figures + a preview renderer + the register record. Verified:

- SONIC expansion SURVIVES as the subtitle (bold-initial form); the
  cbdd49e pair warning survives; the DIDACTICS-95-class overclaim
  phrases remain absent (zero hits).
- The three figures verified VISUALLY (Read on the PNGs):
  colorblind-safe throughout (blue/orange/purple + distinct line
  styles; no red+green pairing), no cropped labels or overlaps; the
  ownership chain teaches the true eight-state pipeline (strict
  load, artifact pair, per-arrow validating boundary); the
  activation panels carry the gamma/beta, gates, and p=1.4 tail
  content truthfully.
- Every figure symbol defined in place beside fig01 (C, V, N, B, P,
  P_enc, D, K, the memmap sentence, the box-color semantics); each
  PNG links to its vector PDF; both the PDF owner
  (texnotes/make_figures.py) and the new renderer are linked.
- PNG dimensions verified with sips: 1800x550 / 1800x525 / 1800x700
  — the register's exact numbers. The byte-identical second-render
  claim is REGISTER-WITNESSED only (pdftoppm is not on my PATH);
  the renderer's determinism is structurally plausible (fixed dpi,
  -singlefile, one tool) and the dimensions cross-check.
- render_readme_previews.py: stdlib + Poppler, no absolute paths,
  house Arguments: blocks. Style note (not a blocker): f-strings +
  type annotations are texnotes-local idiom, outside the production
  surface the C-readable rule guards.
- BOARD IMPACT of the new tracked texnotes .py: geo_paths run with
  the cocoa torch interpreter in their tree — PASS (the whole-scope
  enumerator absorbs the new file on both sides of its set
  equality). No production or gate file touched (commit stat).
- Em dashes: zero in the rewritten README. The register attributes
  the prose rules to the user's ruling in publicly-stateable form.
- The frontispiece image the README embeds is pre-existing and
  tracked (cb2ee8e).

VERDICT: GO. 46efa6d + cbdd49e + 701d6f9 land together with the
codex branch merge.

## Queue-2 note-side evidence drafts audit (Fable, 2026-07-13): 8236417 — GO; 25M-36/37 CONFIRMED; one normalization follow-up

The red team's naming drafts (codex/queue2-note-evidence, 1667
insertions-only across six home notes + the register). MY OWN
mechanical reconciliation (a fresh parser, not theirs): 27 complete
six-field blocks; 137 unique aids, every prefix equal to its
block's board gate id; 164 new anchors = 137 leg transforms + 27
headlines, no duplicates, every dot-to-dash anchor present; legs
counts match the parsed aid lists in all 27 blocks. The exclusion
set is the 7 foundation gates + SIX wrapper-family gates
(cobaya-adapter, save-rebuild-drift, finetune-identity/-smoke,
transfer-identity/-smoke) — a defensible superset of my four-
surface minimum, erring toward leaving entangled legs with the
Implementer; RATIFIED. Content spot-reads (ema-off-identity,
ema-smoke, single-phase-demotion, finite-contract): the
narrowed-claims discipline is exemplary — blind spots stated in the
leg one-liners (the empty-selection acceptance, the broad demotion
pattern, "no current whole-gate PASS" on finite-contract with every
known defect mapped into evidence/owed).

ONE REQUIRED FOLLOW-UP (non-blocking for increment 3): the draft
carries TWO headline-anchor conventions — `<gate>-evidence` in four
notes, bare `<gate>` in the two families notes (8 blocks). Leg
anchors (what the code wires) are uniform; the headline form must
be too. RULING: normalize the 8 bare headlines to `<gate>-evidence`
in a follow-up commit.

25M-36 CONFIRMED at gates/checks/mps_identity.py:275-276: the
check's reference computes want.mean(axis=0) in float64 THEN casts
to float32, while the producer means the STORED float32 payload —
a representation-order defect in the gate's reference (the producer
is correct). Repair: compute the reference in the producer's order
(cast first, then mean) or justify an explicit tolerance;
gate-truth class, red-team claimable under D4's claim-before-edit.

25M-37 CONFIRMED at emulator/geometries/output.py:50-51:
cosmolike_lsst_y1_interface and getdist.IniFile import at MODULE
level, so any gate rebuilding a persisted cosmic-shear geometry
pays both before its first assertion — an import-time death is not
a declared UNAVAILABLE disposition. Repair AT THE PRODUCTION
BOUNDARY (defer both imports into from_cosmolike / their use
sites), NOT four gate-local stubs — Implementer custody (emulator
production), a small standalone landing before queue 5; the four
affected blocks then update their evidence lines (bounded, red
team). The disclosed bounded omission (the older 25M-37 narrative
naming only CosmoLike) is accepted as filed.

MERGE MECHANICS: their branch forked pre-increment-1; git
merge-tree found ONE conflict (the register's both-append tail).
Resolved on THIS branch by merging codex/queue2-note-evidence in,
both sections retained — the user's main merge is then clean.

VERDICT: GO. INCREMENT 3 IS UNBLOCKED for the 27 drafted gates: the
audited aid names are the spec; per-gate objections where migration
disagrees; the six excluded wrapper-family gates + 7 foundation
legs remain Implementer-named.

## Unit 46 audit (Fable, 2026-07-13): b9835a2 GO — the golden leg is no longer vacuously green

The D4 _golden_leg repair walked clause-by-clause against the 46
ruling + addendum: both child return codes are now captured and
required zero ("a golden run must complete"); the compared
selection must be NON-EMPTY on both sides; equality still required;
and the always-printed status reports both rcs and both selected
counts beside the verdict. Selftest arms: the ruled three mutations
(empty selection; both children rc 1 after matching lines; tip-only
rc 1) plus a diverging-selection arm and the clean control — all
red/green as required. My runs: board-selftest ALL PASS, --list
rc 0. The inline comment records the pre-46 vacuous-pass honestly.
VERDICT: GO. D4's board.py half is closed; the check-script D4 legs
remain with their claim-before-edit owners.

## TEX-PROSE-01..08 adjudication (Fable, 2026-07-13): all eight CONFIRMED/ACCEPTED; the repair order approved with two riders

The red team's independent manuscript pass (register on
codex/tex-prose-audit, c9b6f64; no TeX/figure/PDF edits — filed,
not fixed). My verification: 01 CONFIRMED AT SOURCE (the guide's
:4969-4974 carries a broken \rm — literally "{ m km..." after a
lost backslash-r — compiling but rendering stray m's); 05 SAMPLED
(h_j half-width at :2815 beside cosmological h; kernel half-width q
at :2697 beside q's other meaning); 03's census cross-checked (my
single-line grep finds 48 Current gap / Required closure / Current
deviation paragraphs vs their ~58 — their multiline census governs,
non-material); 02's register framing verified publicly-stateable
with correct exclusions (CLI -- syntax, mathematical negation,
refusal rules) and exact line anchors. 04/06/07/08 are
register-witnessed with the same line-anchor discipline that has
run ~60-for-60 this campaign — ACCEPTED as filed.

RULINGS (all repairs red-team custody; texnotes is theirs):

1. 01 + 02 first, as they propose — mechanical hard failures.
2. 03: the current-state doctrine EXTENDS from the README to the
   guide — current behavior, consequence, safe user action. RIDER:
   a still-open defect NEVER loses its user-facing warning; each
   removed diary paragraph gets a register map entry (where its
   warning now lives, or why none is needed). The standing
   gap-closure route continues: an Implementer landing that closes
   a gap names file+line, and the guide update routes to the red
   team.
3. 04: APPROVED with the SCHEMA-ALIGNMENT RIDER — the refactored
   per-gate appendix uses the six-field evidence-block vocabulary
   (files / subprocess / metric / legs / evidence / owed) and the
   audited aid names where it names legs, so the manuscript and the
   home notes teach ONE schema, not two.
4. 05: APPROVED — rename the colliding half-width symbol,
   disambiguate q, define epoch/composition/TATT symbols before
   first use.
5. 06: APPROVED — authoritative citations for external algorithm
   claims, or the explicit "as implemented here" narrowing.
6. 07: APPROVED — name the owning class/function/adapter/driver at
   every state transition.
7. 08: APPROVED — gate-claim verbs narrowed to what the executed
   fixtures establish; the audited evidence blocks are the truth
   source for what is executed TODAY.
8. ACCEPTANCE per landing: recompile, render, visual inspection,
   and the FULL prose census repeated — evidence in the register.

## 25M-37 COMPLETE — geometry-output imports deferred to their use sites (Opus, 2026-07-13)

From the naming-drafts audit: `emulator/geometries/output.py` imported
`cosmolike_lsst_y1_interface` and `getdist.IniFile` at MODULE level, so a missing
dependency was an import-time death for every consumer of the module (inference,
the board, tests) — not a declared disposition of the one training-path call that
needs them. Both imports are deferred into their use sites: `from_cosmolike`
(cosmolike + getdist) and `build_shear_angle_map` (getdist only; that path reads
the ini + n(z) file, no cosmolike), with a module-top breadcrumb pointing there.

Verified on the cocoa-torch interpreter: `import emulator.geometries.output`
succeeds with `cosmolike present in env: False` (the module previously died at the
module-level cosmolike import); `compileall emulator` clean. Small standalone
production landing, before queue 5; no taught-behavior guide passage identified
(an import-location change). Independent of the increment-3 naming drafts.

## 25M-37 repair audit (Fable, 2026-07-13): 3ba8588 GO — the import boundary probed live

The production deferral audited at the diff and PROBED against the
machinery: the module-level cosmolike_lsst_y1_interface + getdist
imports are gone (a teaching comment explains the 25M-37 reason in
place); from_cosmolike carries both at its head, build_shear_angle_
map carries its own IniFile import; every ci.* reference (:315-366)
verified inside from_cosmolike (:265-:384). LIVE PROBE (cocoa
interpreter, a meta_path blocker refusing both packages): the
module IMPORTS CLEANLY with both dependencies blocked, and calling
from_cosmolike fails at exactly its one deferred import with the
blocked ImportError — an import-time death became a declared
failure of the one call that needs the dependency, as ruled. My
runs: compileall clean, board-selftest ALL PASS. The Implementer's
guide-route note is correct: any guide passage documenting the old
import structure is the red team's update (the gap-closure route).
VERDICT: GO. The four Torch-declared gates can now reach their
first assertion on a cosmolike-less box; the drafted evidence
blocks' eager-dependency lines update at the red team (bounded,
announced).

## Headline-anchor normalization audit (Fable, 2026-07-13): ab82b20 GO — the drafts follow-up is closed

The ruled follow-up verified at the diff: EXACTLY eight
one-line anchor swaps (bsn/mps-identity+smoke in
families-background-mps.md, scalar/cmb-identity+smoke in
families-scalar-cmb.md), each bare `<gate>` headline becoming
`<gate>-evidence`; 8 insertions / 8 deletions, nothing else; an
untruncated scan finds no reference to any retired bare anchor.
The headline surface now carries ONE convention across all 27
blocks; leg anchors were already uniform. Merged into this branch
(no conflict — the register untouched this time). VERDICT: GO. The
drafts package (8236417 + ab82b20) is fully closed; increment 3
wires against it unchanged (code references leg anchors only).

## Queue 2 increment 3 — serial-proof gate 1: geo-paths migrated (Opus, 2026-07-13)

First gate of the serial proof (a check-script gate) wired to the audited naming
draft (56661f2). No objection: the drafted names fit the code exactly (each of
geo_paths.py's three legs is one report() call).

- board.py geo-paths: `evidence=` = the three drafted Assertions
  (`geo-paths.fresh-save-uses-folder-paths`, `.legacy-flat-paths-absent`,
  `.legacy-reference-census`, anchored in artifacts-inference-warmstart.md);
  `maps=` rewritten to one taught-vocabulary sentence (the draft headline);
  `home=` already `artifacts-inference-warmstart` (no change). The gate body's
  rc `expect` stays aid-less: the child exit is the aggregate, not a fourth leg
  (the draft's one-verdict clause).
- geo_paths.py: `report()` gains `aid=` and emits one `##AID <aid> <result>`
  line per leg; the three leg report() calls carry their drafted aid. The board
  folds those three records; reconciliation matches declared == executed.

Verified (Mac): compile clean; `--list` rc 0 (the three anchors resolve, satisfy
the aid->anchor transform, and are board-unique); board-selftest 169 PASS
(the fold/reconcile machinery unchanged); geo_paths.py on the cocoa-torch
interpreter exits 0 and prints EXACTLY the three declared `##AID ... PASS` lines.
The full `run_board --gate geo-paths` clean-tree rerun is the end-to-end
confirmation (below / workstation).

## THROUGHPUT REBALANCE, WAVE 3 (Fable, 2026-07-13): three further transfers + one D4 scope clarification

The user asked for further red-team transfers. Inventory result:
most of the remaining Implementer queue is either trigger-gated
(21 at D3+56, 92 at D5, 95 at wave-4, checks-teaching at D4+D5) or
the kept core (the artifact chain 74/76/77/78/80/82/84/85, unit 8,
unit 96, the queue-3 staging reopen, board.py). Three items move
NOW — each small, self-contained, red-team-filed originally, off
the artifact chain and off board machinery, colliding with nothing
in increment 3 (which touches board.py + checks + notes only):

1. UNIT 29's 25M-14 AMENDMENT (the width-one transformer
   refusal): designs/plain.py + designs/blocks.py refuse max token
   width < 2 BEFORE construction on both ResTRF paths (n_tokens ==
   n_out makes width-one tokens; LayerNorm(1) is identically zero
   pre-affine, so the correction becomes input-independent).
   Contract: the 25M-14 adjudication. Torch verification:
   write-here/verify-there until the probe answers.
2. THE SIZING UNIT (25M-15): batching.py's packed-target byte
   estimate drops the resolved tgt_dim (the 84-byte witness
   undercount), and max(1, ...) turns a memory deficit into a wrong
   batch instead of a refusal. Contract: the 25M-15 adjudication —
   the CPU boundary legs (the 84-byte witness) are plain arithmetic,
   fully testable in their environment; the packed-target streaming
   leg stays workstation-owed.
3. THE TRIANGLE STRENGTHENING (DIDACTICS-29 merged with
   DIDACTICS-65): clarifying that D6 transferred WHOLE — the
   earlier firing named only the nine mkdtemp sites; the D6
   bundle's triangle half (the exact expected (x_parameter,
   y_parameter, window) set built independently, axis identities
   per Axes object, per-artist _CUT_GREY, the moved-artist mutation
   that leaves global counts unchanged) moves with it.
   gt_b_triangle.py is not wrapper-family; claim it like any D4
   file.

D4 SCOPE CLARIFICATION: the file-by-file claims EXCLUDE the
wrapper-family check scripts — gct_parity.py (DIDACTICS-66's
masked-index repair) and gsv_bitwise_drift.py (the 64-narrowing and
70-leg, which are D5 anyway) stay with the Implementer, because
increment 3 wires those very files for the six wrapper-family
gates and one file has one owner at a time. Claimable D4 surfaces
include cmb_identity.py (53's six blocks), scalar_smoke.py (63's
used-n-of-m banner), gwd_census.py (52's both-absences +
per-pattern mutations), and gt_b_triangle.py (48's PDF
verification + the triangle half above); 47 claims by naming its
file first, as ruled.

The Implementer's queue after this wave: increment 3 (in flight),
D5, unit 56, the fixed-facts proposal + artifact chain, units 8 /
24 / 63-reopen / 66 / 68-guards / 96, the queue-3 reopen — the
deep-context core, plus trigger-gated items until their seams
clear.

## 25M-37 evidence readback audit + D3 FIRES (Fable, 2026-07-13): de4bdd5 GO; the torch probe is positive

de4bdd5 audited: the two red-team-owned evidence blocks
(scalar-identity, finite-contract) updated to the 3ba8588 deferred
reality — the stale eager-dependency text is gone, the
wrapper-reduction truth is retained ("no logged-only claim is
promoted"), and finite-contract's account now records the repaired
import boundary while keeping the four Parts A/C message-prefix
false reds and the Part F crash as CURRENT execution truth. The
wrapper-family boundary was respected: finetune-identity and
transfer-identity blocks untouched (their children's new results
are recorded in the register for the Implementer's future blocks
to carry). MY OWN RUN reproduced the headline claim: the
scalar_identity child, cocoa Torch interpreter, ends
"PASS: scalar-identity all checks green". Merged into this branch
conflict-free (they merged our branch first).

THE TORCH PROBE IS POSITIVE: cocoa Python imports Torch 2.6.0 and
a CPU nn.Linear(2,1) forward pass with input [1,2], weight [3,4]
prints [[11.0]] — the arithmetic itself checks (1*3 + 2*4 = 11).
No CUDA claim made or needed. D3's pre-authorized environment
condition (THROUGHPUT REBALANCE item 5) is satisfied:

D3 TRANSFERS to the red team NOW — the unit-28 smoke-fixture
disjointness repair, gauntlet-critical, BEFORE queue 5: distinct
RECORDED generator seeds (train 1234 / val 5678, or one
proven-disjoint partition), a zero-overlap refusal BEFORE training,
the row-alignment proof, printed seeds/counts/overlap, and the
same-seed mutation arm (D3's contract in the consolidated DIDACTICS
handoff). Coordination: scalar_smoke.py hosts both D3 fixture work
and DIDACTICS-63's banner leg — ONE claim covers the file, both
repairs may land in one visit. D5's DIDACTICS-60 baseline
recomputation happens AFTER D3, as ruled. The gauntlet is now:
increment 3 (Implementer) in parallel with D3 (red team) -> D4
(split by file) -> D5 (Implementer).

## Increment-3 serial-proof gate 1 audit (Fable, 2026-07-13): 7cbcd7a (geo-paths) GO — the migration pattern is RATIFIED

The first migrated gate audited as the pattern the fan-out
replicates. Verified:

- evidence= carries EXACTLY the three drafted aids
  (fresh-save-uses-folder-paths, legacy-flat-paths-absent,
  legacy-reference-census) with exact transform anchors in
  artifacts-inference-warmstart.md; home= already equals the anchor
  note (ruling 5).
- maps= is one human sentence, no line numbers, no leg enumeration.
- geo_paths.py's report() gains aid= and prints one reserved
  ``##AID <aid> <result>`` line per declared leg; the child's exit
  status stays the single aggregate (the draft block's own "not a
  fourth leg" honored — one-verdict).
- MY RUNS: --list rc 0 (validator green over the new tuple); the
  child under the cocoa interpreter emits EXACTLY the three
  declared ##AID lines, all PASS, aggregate PASS; board-selftest
  ALL PASS.

PATTERN CLAUSE RATIFIED FOR THE FAN-OUT (so subagents replicate it
correctly): when ALL of a gate's declared legs are child-emitted,
the wrapper's rc==0 expect stays AID-LESS (invisible to
reconciliation, which then sees exactly the folded child legs); the
seven foundation gates' wrapper-level headline aids are the OTHER
correct shape, for gates whose one declared leg IS the wrapper's
rc expect. A subagent must pick the shape from the draft block's
evidence field, never add both.

VERDICT: GO. Serial-proof gate 2 (a pure in-process gate) proceeds;
the 2-gate batch audit will be delta-only against this ratified
pattern.

## Queue 2 increment 3 — serial-proof gate 2: single-phase-demotion migrated (Opus, 2026-07-13)

Second gate of the serial proof — a PURE IN-PROCESS (wrapper-asserted) gate,
completing the two aid-shapes the fan-out will use. No objection: the drafted
names fit the body's three ctx.expect calls 1:1.

- board.py single-phase-demotion: `evidence=` = the three drafted Assertions
  (`single-phase-exit-zero`, `demotion-text-present`, `two-phase-control-exit-zero`,
  anchored in training-stack.md); `maps=` -> one taught-vocabulary sentence;
  `home=` already `training-stack`.
- gate_gp_d body: each of the three ctx.expect calls carries its drafted aid
  (wrapper-emitted — there is no child ##AID here; the two driver subprocesses
  are asserted BY the wrapper, per the draft's "asserted by the wrapper"). Leg 2's
  detail narrowed from the stale "EXACT notice string to confirm" to the honest
  broad-presence claim the draft ratified (matches the `demotion-text-present`
  aid — DIDACTICS-62/63 banner-as-truth discipline).
- All three legs are workstation-owed (needs torch+cosmolike+gpu); the gate is
  capability-skipped on the Mac, so there is no per-leg ctx.unavailable (the
  whole gate is skipped, not partially owed).

Verified (Mac): compile clean; `--list` rc 0 (the three anchors resolve, satisfy
the transform, board-unique); a static check confirms the three declared evidence
aids EQUAL the three aids emitted in gate_gp_d; board-selftest 169 PASS. The live
green rerun is workstation-owed (this gate needs cosmolike+gpu).

**Serial-proof batch COMPLETE**: gate 1 geo-paths (7cbcd7a, child-emitted ##AID,
GO'd) + gate 2 single-phase-demotion (wrapper-emitted ctx.expect(aid=)). The two
aid-shapes for the fan-out are both demonstrated. Awaiting the delta audit; on GO,
fan the remaining drafted gates to subagents (increment-sized self-commits;
wrapper-family + foundation legs kept out of subagent hands).

## Increment-3 serial-proof gate 2 audit (Fable, 2026-07-13): da2ac6a GO — both aid shapes proven; FAN OUT

The wrapper-emitted shape audited delta-only against the ratified
pattern: evidence= carries exactly the three drafted aids with
transform anchors in training-stack.md (home= already
"training-stack"); the three wrapper expects thread aid= and
nothing else does; the stale line-number maps= ("110-113 ...") is
one honest sentence; and the migration PROPAGATED the draft's
narrowing into the wrapper detail — the demotion-text expect no
longer promises "EXACT notice string to confirm against
training-stack.md:111" but states the broad pattern truthfully.
My runs: --list TRUE rc 0 (an earlier rc=1 reading was my own
shell-pipeline artifact, re-measured cleanly), board-selftest ALL
PASS. The gate itself is workstation-owed at runtime (two training
children) — reconciliation exercises at queue 5.

VERDICT: GO. The serial batch is complete with both shapes proven
(7cbcd7a child-emitted; da2ac6a wrapper-emitted). THE FAN-OUT IS
GREEN-LIT per the subagent rule: one gate per subagent, the audited
draft block is the spec, the two ratified shapes chosen from the
block's evidence field, independent re-verification before
self-commit, increment-sized batches; the six wrapper-family gates
+ seven foundation legs stay first-hand.

## Root README public-prose hardening audit (Fable, 2026-07-13): 2d49984 — content GO; ONE REQUIRED REDACTION in the register before merge

The second prose pass audited: MY OWN scans confirm the mechanical
claims exactly — zero em/en dashes, zero prose semicolons, zero
question marks, zero curly quotes, 166 fence markers = 83 balanced
pairs; the single ellipsis outside fences is backticked code
notation in a table (a dict literal), the same syntax-exclusion
class as CLI --. The CMB roughness repair verified at code: loss()
composes c + penalty while chi2() reports plain c
(losses/cmb.py:279-281 vs :214) — the README now states exactly
that split. The unit-53 coordination passage is correct (the study
sentence updates in the same change as the eventual code landing).
The five-regression independent factual review and the preservation
evidence (math bytes, tables, links, image bytes, py_compile) are
register-witnessed under the campaign's clean track record. The
bounded emulator/README.md follow-up (pre-25M-37 import prose) is
accepted, routed to its package visit.

THE REDACTION (blocks the merge of this landing only): the register
section's scanned-pattern enumeration lists, in bullets 1-3, the
distinctive CATEGORY NAMES of the user's private editorial
checklist. The standing user rule caps repo artifacts at
"editorial pass against private standards". REQUIRED: collapse
bullets 1-3 into that sanctioned form (the zero-match RESULT
statement stays; the punctuation and list-format bullets 4-5 are
ordinary public style policy and stay verbatim). One small
follow-up commit on the branch, then this landing merges.

## D3 control-interaction ruling (Fable, 2026-07-13): the bar recalibrates BY MEASUREMENT inside the D3 landing — DIDACTICS-60's scalar-smoke half rides D3

THE BLOCKER (red-team filed, held without a commit — the correct
discipline): the honest disjoint fixtures (train 1234 / val 5678;
the authorized single-population split gives the IDENTICAL
0.0745958414) deterministically red the two-epoch prediction
control at 0.0746 against the existing 0.05 bar. Diagnosis: the
0.05 bar was calibrated on the OLD overlapping fixture — roughly a
thousand validation rows sat in the training set, so the control
was partly a memorization test. The honest fixture did not break
the control; it exposed that the bar's calibration was leak-
subsidized. The provisional D3 machinery itself is sound (overlap
census, row alignment, printed seeds/counts/overlap, the
1,000-overlap same-seed mutation refusing before training, and the
DIDACTICS-63 banner leg with staged-order verification).

RULING:

1. The bar RECALIBRATES BY MEASUREMENT, not by decree, INSIDE the
   D3 landing — this is DIDACTICS-60's scalar-smoke half
   (baseline recomputed AFTER D3, both bars, dead-network arm),
   which hereby TRANSFERS from D5 to the red team and rides D3:
   one landing, one fixture, one calibration.
2. The recipe is the standing dead-network doctrine: measure the
   MEAN-PREDICTOR (learned-nothing) error on the SAME disjoint
   fixture; the collapse bar sits strictly below that value (half
   of it is the default); the two-epoch accuracy bar is the
   measured honest error times a stated margin (1.5x recommended,
   ~0.112) and strictly below the collapse bar. All three measured
   numbers, both bars, and the margins are PRINTED by the check
   and recorded in the register — a bar with no visible derivation
   is the defect class this program exists to kill.
3. The dead-network mutation arm must red against the NEW bars,
   and the same-seed mutation stays. The two downstream
   prediction/readback reds that derive from the same 7.46% result
   recalibrate with it — one cause, one fix.
4. The TWO-SEED form (1234/5678, both RECORDED) is adopted as
   ruled; the single-population split was a proof, not the
   landing. Seeds 1/2 stay unadopted. No epoch count, model
   option, or production file changes — exactly as held.
5. D5's list DROPS DIDACTICS-60's scalar-smoke baseline item (its
   prose half was already the factual bundle's). The gauntlet is
   otherwise unchanged.

## Redaction delta audit (Fable, 2026-07-13): fcf68ec GO — the prose-hardening landing is merge-ready

The one-hunk redaction verified at the diff: the three private
editorial-category bullets are GONE, replaced by exactly the
sanctioned form ("a zero-match editorial pass against private
standards"); the ordinary public punctuation and list-format
evidence stays; 3 insertions / 9 deletions, register-only. My
whole-register rescan at the branch tip (8d103a9) finds ZERO hits
for the private-category vocabulary. The prose-hardening landing
(2d49984 + fcf68ec) is fully GO'd and merged into this branch —
the root README's public-prose constitution pass is COMPLETE, with
the emulator/README.md import-prose follow-up routed to its
package visit.

## 25M-15 sizing audit (Fable, 2026-07-13): 3031d02 content GO — HELD for the coordinated stage-ram fixture repair (Implementer)

The red team's sizing landing audited (codex/unit25m15-sizing;
production batching.py + six CPU tests + notes; latest main merged):

- compute_batch_byte_terms names every per-batch term
  (saved_activations, input, model_output, target, chi2_scratch);
  target width and float32 staging dtype are supplied by
  _build_loaders_one FROM THE BOUNDARY THAT CREATES THE TARGET
  (never-trust-defaults applied); the 84-byte witness is preserved
  (out_dim=7, target_dim=14, bs=3); the crafted boundary flips two
  batches -> one; the legacy max(1, ...) arm is a named-terms
  MemoryError refusal; the 0.8 planning allowance is KEPT so
  ordinary-target chunk boundaries are unchanged.
- Their own audit caught and REMOVED a scope expansion (charging
  resident encoded parameters — the separate resource-sizing
  unit's territory); the pre-amendment resident formula is retained
  exactly. Verified in the diff.
- MY RUNS: 6/6 CPU tests pass (cocoa interpreter); the stage-ram
  interaction REPRODUCED exactly — the repaired planner reds the
  foundation fixture's 200-byte disk budget with "required=944,
  available=160, resident=608, ..." and every term named.

COORDINATION RULING: the stage_ram.py fixture repair belongs to the
Implementer (foundation-gate owner; the red team correctly did not
edit it). Spec, adopting the filing's recommendation: expand the
unique unsorted selection to AT LEAST TEN rows so the raised budget
still takes the DISK path (assert the path taken); preserve the
seeded-order assertions; choose the allowance INSIDE
[resident + one complete batch, resident + the full encoded set]
and PRINT the chosen numbers. The sizing branch stays UNMERGED
until that repair lands on amazing-keller; the two then merge in
ONE landing so the board never reds on main. The packed-target
streaming integration remains the workstation/queue-5 exhibit.

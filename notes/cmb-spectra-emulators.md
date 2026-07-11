# CMB spectra emulators (spec)

**Date:** 2026-07-10. **Status:** EXECUTING (kickoff addendum at the
tail; SPE, unit 1 of the pass, is CLOSED at board 25/25). **Spec
code:** CME. **Home note** for the gates `cmb-identity` / `cmb-smoke`.
Companion unit to [[scalar-parameter-emulators]] (SPE); both ship in one
implementation pass (user away from the computer; sequencing below).

## The request (user design goal)

Bring CMB spectra emulation (TT / TE / EE / phiphi, the legacy emulcmb)
into this library: the training-set GENERATION (CAMB data vectors), the
training, and a cobaya theory block — replacing the legacy pattern
(per-spectrum file/extra/ord/extrapar lists + a manual eval mask + one
hand-written theory class). "The CMB emulator can have 4 outputs" = four
per-spectrum artifacts served by ONE generic theory block.

**What the legacy training script fixes as the physics conventions
(emultraincmb.py, read 2026-07-10):**

- loss = sqrt of a cosmic-variance-weighted chi2: covinv =
  diag(2/(2l+1) / Cl_fid^2), l = 2..ellmax — OUR existing sqrt loss over
  a DIAGONAL covariance built analytically from a fiducial Cl (no
  cosmolike, no data file);
- the primary amplitude scaling is DIVIDED OUT of the target:
  target' = Cl * exp(2*tau) / As — the network learns the shape, the
  As*exp(-2tau) law is imposed, not learned (the factored-IA philosophy);
- per-spectrum networks; TT/TE/EE are CNN-over-bins (our rescnn), phiphi
  was ResMLP + a PCA output projection.

## Design rules

### D-CM1 — the output geometry: diagonal, from a fiducial Cl

A constructor (new subclass or classmethod beside DiagonalGeometry) that
builds the standard DataVectorGeometry state from analytic pieces: ell
range 2..ellmax, kept = all, Cinv = the cosmic-variance diagonal from a
STORED fiducial Cl, center = the training-mean (of the amplitude-rescaled
target), whitening = the per-ell diagonal scale. Everything downstream
(CosmolikeChi2, the loop, save/rebuild via the cls marker, FTW) reuses
unchanged — the geometry is data, not new machinery. The artifact stores:
spectrum name ("tt"/"te"/"ee"/"pp"), ellmax, the fiducial Cl, the units
convention (raw Cl, muK^2 for T/E; dimensionless for pp), and the
amplitude law (D-CM2).

### D-CM2 — the imposed amplitude law

A small registry, persisted by name in the artifact (never a code
default): `as_exp2tau` (target' = Cl * e^{2 tau} / As; TT/TE/EE) and
`none` (phiphi V1, or any spectrum the user opts out of). The law reads
As/tau from NAMED input columns — reuse the AmplitudeFactorGeometry
pattern (the named columns ride raw, the law is closed-form) or a thin
chi2 wrapper mirroring TemplateFactoredChi2 with one template and a
closed-form coefficient; Implementer proposes the smaller diff, the
identity gate (exact target round-trip through the law) rules either way.
The decode path multiplies the law back, so get_Cl returns physical Cl.

### D-CM3 — training-set generation, in the library

New `compute_data_vectors/compute_cmb_dvs.py`: samples parameters (the
existing generator conventions — covmat + temperature sampling, the
README appendix pattern), evaluates Cl through cobaya's CAMB requirements
(the legacy dummy-likelihood trick, done properly: one cobaya model, Cl
to ellmax, loop the sample), and writes THE SAME dump format the whole
training stack already stages: params .txt (+ covmat header = the input
names) + dv .npy (rows = samples, columns = l=2..ellmax). One dump per
spectrum family run; TT/TE/EE/pp columns written as separate dv files
from one CAMB pass (CAMB gives all four at once — never re-run the
Boltzmann code per spectrum).

### D-CM4 — training: the existing drivers, a cmb data block

train_single (and the sweeps) gain a `data.cmb:` alternative to the
cosmolike keys: `{spectrum: tt, ellmax: 5000, fiducial_cl: <file>,
amplitude_law: as_exp2tau}` — mutually exclusive with cosmolike_data_dir/
dataset, loud both-present error. build_geometry branches to the D-CM1
constructor (no cosmolike import on this path). Architectures: resmlp /
rescnn / restrf as-is (rescnn IS the legacy CNNMLP's role); the phiphi
PCA output projection is OUT OF SCOPE V1 — train phiphi as a plain
design first; if quality demands compression, that is a recorded V2
(NPCE or a PCA head) with the evidence in hand.

### D-CM5 — the cobaya theory block: spectra derived from the files

`cobaya_theory/emul_cmb.py`, the emul_scalars pattern applied to Cl:

```yaml
theory:
  emul_cmb:
    python_path: ./external_modules/code/emulators_code_v2/cobaya_theory/
    extra_args:
      device: 'cuda'
      emulators:
        - projects/cmb/emulators/tt/emul_v2
        - projects/cmb/emulators/te/emul_v2
        - projects/cmb/emulators/ee/emul_v2
        - projects/cmb/emulators/pp/emul_v2
```

- WHICH spectra + each ellmax: read from the artifacts. The legacy
  `eval:` mask, `ord`, `file`, `extra`, `extrapar` all die.
- `get_can_support_params` / requirements = the union of stored input
  names; two artifacts with the same spectrum = loud error; a likelihood
  requesting a spectrum no artifact provides = loud error at
  must_provide, naming the loaded spectra.
- get_Cl assembles the cobaya Cl dict (ell array + per-spectrum arrays,
  zero-padded l=0,1; units per the artifact convention) from batch-1
  decodes, amplitude law multiplied back.

### D-CM6 — gates

- `cmb-identity` (CME-A; Mac+board, torch only): synthetic fiducial Cl ->
  geometry build + state round-trip byte-identical; the amplitude law
  exact both ways (encode(decode) == identity bitwise on the law's
  closed form); save -> rebuild -> predict bitwise (same-path); get_Cl
  assembly from two synthetic artifacts; the duplicate-spectrum and
  missing-spectrum errors.
- `cmb-smoke` (CME-B; board; needs camb+cobaya): END-TO-END — the D-CM3
  generator makes a TINY dump (e.g. 300 rows, ellmax 512) through real
  CAMB, train_single trains 2 epochs on it (tt, as_exp2tau), a cobaya
  evaluate run through emul_cmb returns Cl at a test point (the
  cobaya-adapter gate pattern). This smoke also gates the generator —
  the piece the user asked to "include in the library".

### D-CM7 — out of scope (recorded)

The phiphi PCA/TMAT output compression (V2, evidence-gated per D-CM4);
CosmoRec/recombination variants (the legacy comment "not trained with
CosmoRec" is a training-data property, not code); the legacy emulbaosn
(H(z) integrator) — its own future unit; TPE transfer over CMB emulators
(possible later — the geometry is standard — but not V1).

## Sequencing (both units, one implementation pass)

SPE first (smaller, establishes the artifact-derived provides pattern and
the EmulatorPredictor dispatch), then CME (reuses both). Each unit gets
its own commit and its own gates; the board grows by four (scalar-identity,
scalar-smoke, cmb-identity, cmb-smoke).

## Links

[[scalar-parameter-emulators]], [[finetune-warm-start]],
[[gates-harness-user-run]], [[py-module-style-conventions]],
[[docs-plain-language-define-or-drop]].

## Resume state (Implementer appends below)

### CME increment 1 (2026-07-10, Opus): geometry + amplitude-law registry landed + Mac-gated

**Base:** claude/amazing-keller-e798b6 @ e33c058 (CME kickoff; SPE closed at
7dd5062). Both new files uncommitted.

**Landed (D-CM1 + D-CM2):**
- `emulator/geometries_cmb.py` = `CmbDiagonalGeometry`: a diagonal
  cosmic-variance output geometry for one spectrum. `from_fiducial(device,
  spectrum, ell, fiducial_cl, center, units)` builds `cinv_l = 2/((2l+1)
  Cl_fid^2)` and the whitening scale `sigma_l = 1/sqrt(cinv_l) = Cl_fid *
  sqrt((2l+1)/2)`; encode = whiten(squeeze(dv) - center), decode = unwhiten +
  center; state/from_state carry the resolved set {spectrum, ell, center,
  sigma, fiducial_cl, units} (cls-marker persistence, zero results.py change,
  as SPE proved). dest_idx = arange(n_ell), total_size = n_ell (loop sizing,
  no CMB branch). A non-positive fiducial Cl is a loud error naming the
  multipole.
- `emulator/losses/cmb.py` = `AMPLITUDE_LAWS = {none:(), as_exp2tau:(as_name,
  tau_name)}` (registry, persisted by name), `CmbDiagonalChi2(CosmolikeChi2)`
  (the `none` law: chi2 = (r*r).sum(1), like ScalarChi2), `CmbFactoredChi2(
  CmbDiagonalChi2)` (the `as_exp2tau` law: needs_params=True; _factor decodes
  the whitened inputs through the param geometry and returns f = exp(2 tau)/As
  as a (B,1) column; encode = whiten(squeeze(dv)*f - center); decode =
  (unwhiten(pred)+center)/f; chi2 inherited plain; loss stashes+forwards),
  `make_cmb_chi2(geom, law, param_geometry, as_name, tau_name)`.

**Design decisions (Architect: audit / rule):**
1. **covinv FORMULA FLAG (highest priority, ruling requested).** The spec's
   verbatim `cinv = 2/((2l+1) Cl_fid^2)` (divide), the binding legacy source
   `emultraincmb.py:204` `covinv = 2/(2l+1)*cl_fid**2` (MULTIPLY), and the
   textbook cosmic-variance inverse `(2l+1)/(2 Cl^2)` are THREE different
   forms; none agree. I implemented the Architect's stated verbatim (divide)
   because "verbatim numerics are binding," on ONE clearly-commented line in
   from_fiducial, and flagged all three readings there. The legacy also
   loads `cl_fid` from `covmat/cv_fid_cls.npy` scaled by `2/exp(2*0.06)` (its
   contents decide whether cl_fid is the fiducial Cl or its variance) — the
   Architect has the physics + that file; please confirm which form is
   intended. NOTE: the increment-1 gates are independent of the constant
   (whitening = 1/sqrt(cinv), so the constant only rescales the per-l target
   units; round-trips, persistence, and amplitude-law exactness hold for any
   positive cinv), so this flag does not block the checkpoint — one line
   flips on the ruling.
2. **Diagonal storage (deviation from D-CM1's "reuses DataVectorGeometry
   state").** CmbDiagonalGeometry is STANDALONE (per-l vectors), not a
   DiagonalGeometry subclass, because a dense evecs + Cinv for ellmax~5000 is
   ~100MB/spectrum and the base chi2 einsum is O(n_ell^2) for a diagonal
   covariance. This mirrors ScalarGeometry (also standalone). The cls-marker
   persistence + the loop interface (encode/decode/squeeze/unsqueeze/
   dest_idx/total_size) are preserved, so the SPIRIT of "geometry is data,
   save/rebuild/loop reuse it" holds; only the matrix machinery is skipped.
   Recommend ENDORSE.
3. **Whitening = cosmic-variance sigma (not empirical std).** With sigma =
   1/sqrt(cinv), the plain sum-of-squares chi2 IS the cosmic-variance chi2
   (reconciles D-CM1's "whitening = per-ell diagonal scale" with "Cinv =
   cosmic-variance diagonal"). This DIFFERS from the legacy (whitens by
   empirical Y_std, weights the loss separately by covinv); our design folds
   both into one consistent whitening, so the network sees deviations in
   cosmic-variance units. Recommend ENDORSE.
4. **Amplitude law = Option B (chi2 wrapper), the smaller diff (D-CM2).**
   CmbFactoredChi2 mirrors RescaledChi2 (per-row scalar f in place of the
   per-element analytic R) and inherits the plain chi2 (like ResidualBaseChi2),
   reusing the whole needs_params threading (encode/decode/chi2/loss signatures
   verified against training.py:1428-1437 / 1730-1748 and inference.py's
   `chi2.decode(pred, x_enc)`). The AmplitudeFactorGeometry (input-side)
   pattern was NOT used: tau/logA must stay IN the whitened input (they shape
   Cl), whereas AmplitudeFactorGeometry removes the amplitude columns.
   Recommend ACCEPT.
5. **As-interpretation ruling (PROPOSED, the simpler one, like D-SP4).**
   `as_exp2tau` reads a RAW linear amplitude column (as_name) + tau (tau_name),
   f = exp(2 tau)/As. A run that samples logA materializes a raw As column in
   the generator (D-CM3), so the law's closed form stays dead simple and the
   logA<->As conversion lives in the generator/staging (by-name columns, the
   SPE .paramnames machinery). Recommend ACCEPT.
6. **ell endpoint DEFERRED to the generator (D-CM3).** from_fiducial takes
   explicit ell + fiducial_cl arrays, so 2..ellmax inclusive/exclusive is the
   generator's call; legacy used np.arange(2, ellmax) (EXCLUSIVE of ellmax,
   n = ellmax-2). Recorded for increment 3.
7. **Inference note (for increment 4).** EmulatorPredictor has NO rescaled
   branch; the CME amplitude-law decode needs a new _build_decoder branch that
   reuses CmbFactoredChi2.decode (exactly as ia/pce/transfer reuse their
   chi2.decode), and the predictor __init__ needs a CME early-return before
   the cosmolike section accounting (like the scalar branch). Recorded.

**Mac gate (increment 1).** py_compile OK (both files); numpy probe ALL PASS
(40+ checks): from_fiducial sigma == Cl*sqrt((2l+1)/2) (max rel 2.2e-16) and
strictly positive; none-law decode(encode(dv)) round-trip max|dv| = 0.00e+00;
as_exp2tau decode(encode(dv,p),p) == dv (max abs 1.4e-14 at dvmax 2.3e2);
factor f = exp(2 tau)/As per-row (B,1); chi2 = sum sq resid shape (B,), 0 at
pred==target; non-positive fiducial raises; state round-trip bitwise; plus
AST/source cross-checks (class bases, needs_params True on factored/absent on
diagonal, the verbatim cinv line, the FLAG comment, registry keys, no
cosmolike/getdist import, encode/decode/factor expressions verbatim). torch
training legs + the real save/rebuild/predict ride the cmb-identity board gate.

**Next:** the data.cmb driver branch + staging (D-CM4) — experiment.py
build_geometry scalar-style branch (center from the amplitude-rescaled
training targets, CmbDiagonalGeometry.from_fiducial + make_cmb_chi2), a cmb
source in data_staging, results.py surfacing info["amplitude_law"], the thin
driver; then the generator (D-CM3), emul_cmb (D-CM5, after the binding reads),
gates + configs + README (D-CM6).

## Unified ARCHITECT_HANDOFF (SPE + CME, 2026-07-10)

The block below is the handoff of record for BOTH units (relayed to the
Implementer session by the user); the SPE note carries a pointer here.

### ARCHITECT_HANDOFF: READY FOR EXECUTION (SPE + CME, one pass)

- **Units & sequencing:** SPE (scalar parameter emulators) FIRST, then
  CME (CMB spectra emulators) — SPE establishes the artifact-derived
  provides pattern and the EmulatorPredictor dispatch that CME reuses.
  One commit per unit on the branch. Implementer = Opus 4.8; base = main
  at the two-spec commit.
- **Target files, SPE:** emulator/geometries_scalar.py (ScalarGeometry:
  names/center/scale, state/from_state, cls marker);
  emulator/losses/scalar.py (standardized MSE exposing the loop
  interface); train_scalar_emulator.py (thin driver, no cosmolike);
  results.py save/rebuild generalization; the EmulatorPredictor scalar
  branch; cobaya_theory/emul_scalars.py; gates scalar-identity /
  scalar-smoke + configs + example YAML.
- **Target files, CME:** the from-fiducial diagonal output geometry
  (constructor beside DiagonalGeometry); the imposed-amplitude-law
  registry ({as_exp2tau, none}, persisted by name);
  compute_data_vectors/compute_cmb_dvs.py (CAMB via cobaya, ONE
  Boltzmann pass -> all four spectra, existing dump format);
  the data.cmb block in the drivers (exclusive with cosmolike keys);
  cobaya_theory/emul_cmb.py; gates cmb-identity / cmb-smoke + configs +
  example YAML.
- **Contracts & interfaces:** D-SP1..D-SP8 (scalar-parameter-emulators.md)
  and D-CM1..D-CM7 (this note), in full. Legacy references live in the
  user's Downloads/emulators_code-main/ (emultheta, emulrdrag, emulcmb,
  emultraining) — read them for the cobaya API surfaces
  (initialize / get_requirements / get_can_provide_params / get_param /
  get_Cl / must_provide) and the CME physics conventions; PORT NO CODE.
- **Verbatim numerics (CME, from the legacy trainer):**
  covinv = diag(2/(2l+1) / Cl_fid^2), l = 2..ellmax; the amplitude law
  target' = Cl * exp(2*tau) / As, multiplied back at decode.
  SPE has none (new machinery over existing conventions).
- **Constraints & edge cases:** provides / spectra / ellmax /
  requirements are ARTIFACT FACTS — a YAML provides: is a subset CHECK
  only, never a source; duplicate output names (SPE) and duplicate
  spectra (CME) are loud initialize errors; a likelihood requesting an
  unprovided spectrum is loud at must_provide naming the loaded set;
  data.cmb exclusive with cosmolike_data_dir/dataset (both-present
  loud); NO cosmolike import on either new path; the D-SP4 input-overlap
  question — propose the simpler ruling in the resume; standard schema
  v2 throughout (FTW composes; TPE out of scope for both, recorded);
  docs under define-or-drop; enumerate every config-key access on both
  new driver paths (the FTW forward-walk lesson).
- **Validation gate:** four new board gates — scalar-identity (SPE-A),
  scalar-smoke (SPE-B: deterministic derived target from existing dump
  columns + a cobaya evaluate leg), cmb-identity (CME-A: geometry +
  amplitude-law exact round-trips, get_Cl assembly, error paths),
  cmb-smoke (CME-B: END-TO-END — tiny generated dump through real CAMB,
  2-epoch train, cobaya evaluate through emul_cmb). Mac-limit discipline
  as established (compileall + AST + numpy probes; torch/h5py legs are
  workstation-confirmed). The user is away: each handback must be
  self-contained with the full force-rerun list.
- **Notes entries:** per-unit resume state appended to each spec note;
  mid-unit design questions go there as checkpoint requests (the TPE-2
  precedent) — the Architect rules from the notes when handbacks arrive.
- **Next milestone:** one IMPLEMENTER_HANDOFF per unit, SPE first, diffs
  on the branch, no git run by the Implementer.

## CME execution addendum (2026-07-10, Architect) — SPE closed, CME active

SPE is CLOSED (board 25/25, run 5; full record + delta ledger in
[[scalar-parameter-emulators]]). The unified handoff above stands, with
these updates and BINDING additions from SPE's five board runs:

**Corrections to the block above:**
- Board counts: the registry held 23 before SPE (the 24 was note
  arithmetic, corrected); CME lands the board at 25 -> 27. Count gates
  by enumerating the Gate() registry, never from notes.
- The user is present and relaying: checkpoint at every stop with a
  relayable CHECKPOINT handoff (the SPE cadence — eight checkpoints,
  audit each); the self-contained-handback clause still applies to any
  stretch the user steps away for.

**The reuse surface is now concrete (was anticipated):**
- cobaya_theory/emul_scalars.py is the TEMPLATE for emul_cmb.py: the
  _ALLOWED_EXTRA_ARGS whitelist, _pick_device, ROOTDIR-relative roots,
  the initialize unions + loud duplicate/overlap errors, and the
  D-SPE2-4 wrong-kind guard (emul_cmb must loudly reject a non-CMB
  artifact root; the reverse direction already works — emul_scalars
  rejects anything whose rebuild is not a ScalarGeometry).
- EmulatorPredictor dispatches on the rebuilt geometry; the CME
  spectra ride the dv branch (a DataVectorGeometry subclass or a
  from-fiducial constructor per D-CM1), so save/rebuild/predict come
  free via the cls marker exactly as SPE proved.

**The SPE lesson bank, binding on CME (each cost one board run):**
1. D-SPE2-7: every new gate cfg AND example YAML carries all six
   required train_args blocks (model / optimizer / lr / scheduler /
   trim / focus + loss / nepochs / bs); validate on the Mac with the
   build_run_specs required-subscript census BEFORE the board.
2. D-SPE2-6: the generator (compute_cmb_dvs.py) writes the standing
   dump format INCLUDING the .paramnames sidecar; any by-name column
   loading uses the chain-root-aware sidecar resolution already in
   data_staging (exact stem, then integer-suffix root).
3. D-SPE2-5 (+ the personal-memory rule): the cmb-smoke bars must fail
   a dead network — assert Cl at a NON-fiducial point and set the
   collapse bar below the mean-predictor baseline; compute that
   baseline explicitly when designing the gate.
4. D-SPE2-8: the evaluate YAML uses priors + the evaluate sampler's
   override, never value:-fixed params; theory-level stop_at_error;
   force: True.
5. D-SPE2-9: an evaluate run writes NO .paramnames; read products from
   the stdout blocks or the chain's own header row. For get_Cl the
   readback is different in kind (a theory product, not a derived
   param): the proven pattern is the cobaya-adapter leg — a real
   likelihood consuming the product + exit code; design the assertion
   around what the run provably emits, and SHIP THE DIAG (dir listing,
   stdout tail) in the first version, not after the first red.
6. Honest counts and honest margins: report what the probe/board says
   (the SPE count catch and the 4.65%-of-5% margin are the precedents).

**Unchanged and binding:** D-CM1..D-CM7 in full; the verbatim numerics
(covinv = diag(2/(2l+1)/Cl_fid^2), l = 2..ellmax; target' =
Cl * exp(2*tau) / As, multiplied back at decode); the BINDING reads
before writing (legacy emulcmb + emultraincmb.py for the physics
conventions, emul_cosmic_shear + emul_scalars for the adapter shape);
ONE CAMB pass -> all four spectra; data.cmb exclusive with cosmolike
keys, loud; phiphi PCA out of scope V1; no cosmolike import on the CME
path; whole-driver-path forward-walk with the config-access census.

### ARCHITECT_HANDOFF: CME EXECUTION START (relay to the Implementer)

- **You are the Implementer (Opus 4.8).** Unit: CME, the second of the
  SPE+CME pass. Base: main at the SPE close (branch
  claude/amazing-keller-e798b6 merged; verify with git log before
  starting).
- **Read first:** this note in full (D-CM1..7 + this addendum), the
  SPE close section of [[scalar-parameter-emulators]] (the lesson
  bank), and the binding sources before writing the adapter/trainer:
  Downloads/emulators_code-main/emulcmb + emultraining/emultraincmb.py,
  cobaya_theory/emul_cosmic_shear.py, cobaya_theory/emul_scalars.py.
- **Suggested increments (checkpoint at each):** (1) the from-fiducial
  diagonal geometry + the amplitude-law registry (D-CM1/2, the core);
  (2) the data.cmb driver branch + staging (D-CM4); (3) the generator
  compute_cmb_dvs.py (D-CM3); (4) emul_cmb.py (D-CM5, after the
  binding reads); (5) gates + configs + example YAML + README draft
  (D-CM6, with the lesson bank applied); then the full CME
  IMPLEMENTER_HANDOFF with the force-rerun list (cmb-identity,
  cmb-smoke; board 25 -> 27).
- **Mac discipline:** compileall + AST + numpy probes per increment;
  torch/CAMB legs ride the board.
- **Every stop emits a relayable CHECKPOINT handoff.** The Architect
  audits each against raw source, as in SPE.

## D-CM3-A — generator amendment (2026-07-10, user directive + ruling)

**The user's directive:** the repo has no CMB dump generator; build it
by SHARING code with the lensing generator — separate driver files,
duplication minimized. **Grounding (both files read):** the legacy
emultraining/dataset_generator_cmb.py and the repo's
compute_data_vectors/dataset_generator_lensing.py are the SAME ~1100-
line class — identical CLI, __setup_flags, checkpoint load/save,
sampling (emcee Gaussian-with-temperature / uniform + bounds), chain /
.paramnames / .ranges / .covmat outputs, and the MPI master-worker
farm — differing ONLY in _compute_dvs_from_sample, the dv container
shape (2D vs 3D), the probe whitelist (cs/ggl/gc vs
cmblensed/cmbunlensed), and the required train_args keys. ~85% shared.

**RULING — the three-file layout (supersedes D-CM3's single new file):**
- `compute_data_vectors/generator_core.py` (new): the shared machinery
  MOVED VERBATIM from dataset_generator_lensing.py —
  capture_native_output, the base class with setup/flags, the reorder
  helpers, checkpoint load/save/append, the sampling + output writers,
  the RAM-aware dv allocation, and the MPI farm. The subclass surface
  is small and explicit: (1) the valid probe names; (2) the required
  train_args keys; (3) the dv STORE (allocate / write-row / flush —
  abstracted so lensing keeps its single 2D array and CMB holds four);
  (4) _compute_dvs_from_sample.
- `compute_data_vectors/dataset_generator_lensing.py` (re-thinned): the
  CLI + the cosmolike compute + its probe whitelist, subclassing the
  core. NO behavior change: same CLI, same output names and formats.
  Evidence plan (a golden byte-rerun is impractical — emcee is
  unseeded): the shared spans move VERBATIM (the porting discipline),
  and a Mac AST census asserts the CLI arg set + output-filename
  formats unchanged; the board gates that consume its dumps remain the
  functional proof.
- `compute_data_vectors/dataset_generator_cmb.py` (new; named to mirror
  its sibling — supersedes the compute_cmb_dvs.py name): the CMB driver
  on the core, porting the legacy compute conventions verbatim where
  physics-bearing (the check_cache_and_compute walk over the component
  order, capture_native_output + the CAMB error keywords, the
  lensed/unlensed probe switch, the lrange slice), with THREE
  deliberate changes, each a recorded deviation from the legacy file:
  1. Four per-spectrum 2D dv files (tt / te / ee / pp) from ONE CAMB
     pass, replacing the legacy 3D (N, ell, 5) array — the training
     stack stages 2D dv files (D-CM3 unchanged on this point).
  2. phiphi is FILLED from get_Cl (the legacy file zeroed out[:,3] and
     never produced phiphi training data on this path); the model
     requirements must request lensed Cl INCLUDING pp.
  3. The legacy "EXTRA" derived-param column dies — derived scalars
     are SPE's job (train_scalar_emulator on the same params dump).
- Shared-by-construction: the .1.txt / .paramnames (with chi2*) /
  .ranges / .covmat outputs come from the CORE, so the sidecar
  conventions (the D-SPE2-6 lesson) hold for CMB dumps automatically.

**Sequencing update:** increment (3) becomes (3a) extract
generator_core + re-thin the lensing driver (verbatim moves + the CLI
census probe), then (3b) the CMB driver on the core. Everything else
in the kickoff block stands.

## D-CM8 — the smoothness (residual-roughness) loss term
(2026-07-10, user directive + ruling)

**The physics directive:** CMB spectra are smooth ell by ell; the loss
must be able to penalize high-frequency oscillations in the emulator
output — where "high frequency" means periods MUCH shorter than the
acoustic structure (the peak spacing ell_A = pi*D_A(z*)/r_s, roughly
200-300 in ell) — while never over-penalizing the physical peak
SMOOTHING that lensing imprints.

**The two structural rules that satisfy both cautions:**

1. **Residual, never prediction.** The penalty acts on the whitened
   residual r = pred - target, not on the prediction's own shape. A
   prediction-smoothness prior would bias the network toward
   over-smooth spectra — mimicking extra lensing-style peak smoothing
   (the A_L-shaped science risk) — while a residual term is
   identically ZERO when the prediction equals the lensed truth,
   however smooth or sharp the true peaks are. Lensing neutrality is
   therefore structural, not calibrated.
2. **Band-explicit.** The term's transfer function passes only
   oscillation periods well below the acoustic band: full weight at
   periods <= `period_cut` (default 50 in ell), negligible weight at
   periods >= ~200. Genuine parameter-induced misfits (a shifted
   theta*, a wrong lensing amplitude) produce residuals with
   acoustic-period structure — those belong to the PLAIN chi2, and the
   roughness term must not reweight them. The default separation
   (50 vs 200-300) is a factor >= 4.

**The composition rule (fits the existing machinery untouched):** the
roughness term is a per-sample scalar added to the per-sample chi2
BEFORE reduction:

    c_total = c_chi2 + lam * c_rough        (per sample)

so trim / focus / berhu / EMA / anchor all compose unchanged (they act
on one number per sample — the SPE inheritance argument again). The
whitened basis is the right home: the cosmic-variance sigma_ell and
the amplitude law are smooth in ell, so whitening introduces no
artificial high frequencies.

**YAML (paste-ready; absent block = OFF, byte-identical to the plain
loss — the standing off-identity rule):**

```yaml
  loss:
    mode: sqrt
    roughness:
      lam:        0.1   # weight of the residual-roughness term;
                        # block absent = the term does not exist
      period_cut: 50    # penalize residual oscillations with period
                        # below this many ells; the acoustic band
                        # (~200-300, incl. lensing peak smoothing)
                        # stays with the plain chi2
```

**Implementation freedom (Implementer proposes the smaller diff):** a
smooth high-pass of the residual (convolution kernel of width ~
period_cut, penalty = sum of squares of the remainder) or a DCT band
mask — either satisfies the transfer-function requirement; a bare
second-difference does NOT by itself (its (2pi/P)^4 law has no
explicit band edge), though second-difference-after-high-pass is
admissible. The gate rules, not the prose.

**Gate legs (cmb-identity additions):**
- band ratio: a synthetic residual wiggle of fixed amplitude at
  period 30 vs period 300 -> penalty ratio > 100;
- zero residual -> exactly zero penalty;
- OFF identity: no roughness block -> the loss path is byte-identical
  to the plain CosmolikeChi2 path (the ema-off-identity pattern);
- composition: trim/focus/berhu receive c_chi2 + lam*c_rough per
  sample (one reduction path, no second ladder);
- the lensing guard, made concrete: the penalty evaluated on a
  residual shaped like (lensed - unlensed) Cl — the exact signature
  of peak smoothing, acoustic-period by construction — must be
  negligible against the same vector's plain chi2 (< a few percent);
  synthetic acoustic-period proxy on the Mac, the real CAMB pair in
  cmb-smoke where truth is available.

**Recorded:** lam and period_cut are sweepable train_args leaves; the
default lam is 0-when-absent (never a silent nonzero); calibrating a
USEFUL lam is a science-thread experiment, not a gate claim — the
gates prove the band and the identities, not the benefit.

## D-CM9 — diagnostics PDF for the new families
(2026-07-10, user directive + ruling)

**The directive:** the cosmic-shear diagnostic PDF does not transfer
as-is; each emulator family needs working diagnostics (scalar, CMB,
and later BSN).

**The split, grounded in diagnostics.py's inventory:** the chi2-BASED
pages are family-generic BY CONSTRUCTION — every family's loss exposes
a per-sample chi2 (the SPE inheritance design), and history panels,
coverage-vs-kNN-distance, the local-linear floor, hard-direction
regression, and the shaded parameter triangle consume only
(params, per-sample chi2). They apply to ALL families unchanged. What
is cosmic-shear-specific is any theta-structure / dv-overlay page;
what is MISSING per family is the physical-units residual pages.

**RULING — land inside this unit (the driver/gates increments), and
the factoring must prove itself on TWO families at once:**
- diagnostics.py grows a family dispatch (the rebuilt geometry class /
  info flag, the predictor's own pattern): shared chi2 pages for
  everyone; per-family physical pages behind the dispatch; the
  cosmic-shear pages untouched on the cs path (byte-identical PDF —
  the off-identity rule; triangle-shading and production-diagnostic
  stay green unchanged).
- **CMB pages (this unit):** per-spectrum fractional residual vs ell —
  median and 68/95 percentile BANDS over the validation set, one panel
  per spectrum; a worst-cosmology overlay (pred vs truth Cl at the
  highest-chi2 val point); and the D-CM8 companion page — the
  high-pass residual content vs ell (the wiggle spectrum the roughness
  term targets), shown with the acoustic band marked so over-smoothing
  or ringing is visible at a glance.
- **Scalar pages (this unit — the factoring's second family, and the
  SPE follow-up the user asked for):** per-output truth-vs-predicted
  scatter with the identity line; per-output residual histograms in
  PHYSICAL units and in standardized units side by side; residual vs
  each input parameter (the bias hunt). train_scalar_emulator.py gains
  the --diagnostic flag it shipped without.
- Colorblind-safe throughout, never red+green ([[plots-no-red-green]]).
- Gates: each family smoke gains a cheap diagnostics leg — the PDF
  builds without exception and carries the expected page count; no new
  image-analysis gates (triangle-shading remains the only pixel-level
  check).
- BSN pages are specced in [[baosn-emulators]] (D-BSN8) and land with
  that unit on this factoring.

## D-CM10 — transfer/fine-tune scope (2026-07-10, user directive)

- **Fine-tuning (train_args.finetune) is IN SCOPE for this unit:**
  every family must support it. The CmbDiagonalGeometry is a standard
  dv geometry, so FTW should compose through the existing pin — the
  unit ASSERTS it rather than assumes it: cmb-identity gains a
  finetune leg (a warm start from a CMB artifact reproduces the
  source bitwise at epoch 0 on shared inputs; wrong-kind and
  geometry-mismatch sources loud). No new machinery expected; the leg
  is the proof.
- **Transfer (`transfer:` / `transfer.refine`) scope, ruled:** the two
  data-vector families ONLY — cosmolike and CMB. For CMB it stays
  deferred exactly as D-CM7 records (not V1; the geometry is standard
  so it composes when the science asks); for scalar and BSN the
  forbids are PERMANENT (the ruling of record is in
  [[scalar-parameter-emulators]]).

## Architect audit + rulings: CME increment-1 checkpoint
(2026-07-10, Fable; the covariance directive folded in)

### THE covinv RULING (supersedes every earlier form)

The authority is now Motloch & Hu 1709.03599 (user-supplied, eqs 1-7
read from the PDF). Eq 3 settles it:

    G^{XY,WZ}_{ll'} = delta_{ll'}/(2l+1) *
                      [C^XW_exp C^YZ_exp + C^XZ_exp C^YW_exp]
    with  C^XY_exp = C^XY + N^XY                          (eq 4)
    and   N^XY_l = Delta_XY^2 * exp(l(l+1)theta_FWHM^2/(8 ln2))  (eq 1)

So the per-spectrum Gaussian VARIANCE is 2/(2l+1)*(C_exp)^2 for
TT/EE/BB, and [C^TT_exp C^EE_exp + (C^TE)^2]/(2l+1) for TE — and the
chi2 metric is its INVERSE: covinv_l = (2l+1)/(2 (Cl_fid+Nl)^2) etc.
For the record, all three disputed forms are dispatched: the spec's
"verbatim" 2/((2l+1) Cl^2) was MY mis-transcription of the legacy
line (owned); the legacy emultraincmb.py line 2/(2l+1)*cl_fid^2 is
the VARIANCE misnamed covinv (whatever cv_fid_cls.npy baked in, it
no longer matters); the textbook inverse — WITH the eq-4 noise — is
the ruling. The Implementer's flagged one-line flip in from_fiducial
goes to the true inverse; the geometry's sigma_l = 1/sqrt(covinv_l).
Increment-1 gates are constant-independent as noted, so nothing
re-opens.

### The seven increment-1 decisions: ENDORSED

(2) standalone per-l vectors, not a DiagonalGeometry subclass —
ENDORSED (the memory argument is right; NOTE: the NG covariance,
below, lives in the LOSS as a dense contraction, never in the
geometry, so this stays true); (3) whitening = the Gaussian sigma_l,
plain sum-of-squares = the Gaussian chi2 — ENDORSED with the covinv
ruling's constants; (4) the RescaledChi2-shaped amplitude wrapper —
ENDORSED (the right split: tau/logA stay whitened inputs);
(5) as_exp2tau reads a raw linear As column, the generator
materializes it — ENDORSED (the simpler ruling wins, as
pre-authorized); (6) l endpoint deferred to the generator — ENDORSED;
(7) the CME _build_decoder branch in increment 4 — ENDORSED.

### D-CM11 — the CMB covariance script (user directive; ARCHITECT
implements this personally)

**The directive:** unlike lensing (covariance from cosmolike), the
CMB covariance must be COMPUTED, following Motloch & Hu eqs 1-7, by a
SEPARATE script. Gaussian terms first; the lensing-induced
non-Gaussian terms behind a flag, OFF by default. "Because this is a
more difficult task — you can implement it yourself": assigned to the
Architect, accepted; the Implementer proceeds with increments 2-5
against the FILE INTERFACE below, unblocked.

- **Script:** `compute_data_vectors/compute_cmb_covariance.py`.
- **Gaussian part (always):** eq 3 with eq-4 noise; user inputs =
  instrumental noise Delta_XY (muK-arcmin) and beam theta_FWHM
  (arcmin), eq-1 convention; fsky an explicit knob, default 1
  (recorded, never silent). All spectrum-pair blocks computed
  (TT,TE,EE incl. the TT-TE / TE-EE / TT-EE l-diagonal cross blocks);
  phiphi V1 = cosmic variance 2/(2L+1)(C^phiphi)^2 with a
  user-supplied N0 file hook recorded as the future knob
  (reconstruction noise is experiment-specific).
- **Non-Gaussian part (flag, default OFF):** eq 5 = N^(phi) (eq 6,
  lens-induced: sum over L of dC^XY_l/dC^phiphi_L * Cov^phiphi_LL *
  dC^WZ_l'/dC^phiphi_L) + N^(E) (eq 7, unlensed-EE sample variance
  into BB — V1 records it and may defer BB entirely since no BB
  emulator is planned). Derivatives by the 5-POINT STENCIL (the user
  upgrades the paper's 2-point central difference), with an explicit
  CONVERGENCE HARNESS: each derivative computed at >= 3 step sizes,
  the pairwise agreement reported, non-convergence LOUD — "getting
  the convergence of the 5-stencil rule with respect to step size is
  always tricky" (user), so the step study is a first-class output,
  not a hidden default.
- **CAMB via cobaya on HIGH settings — the user's verbatim theory
  block, the covariance's fixed configuration:**

```yaml
theory:
  camb:
    path: ./external_modules/code/CAMB
    extra_args:
      halofit_version: takahashi
      lmax: 7000
      kmax: 10
      k_per_logint: 130
      AccuracyBoost: 1.5
      lAccuracyBoost: 1.2
      lens_margin: 2050
      lens_k_eta_reference: 36000.0
      nonlinear: NonLinear_both
      recombination_model: CosmoRec
      Accuracy.AccurateBB: True
      min_l_logl_sampling: 6000
      DoLateRadTruncation: False
```

- **LCDM only:** the covariance is ALWAYS computed on a fiducial
  LCDM cosmology; the script validates its input cosmology block is
  plain LCDM (loud otherwise), and the TRAINING YAML (D-CM4 below)
  demands the fiducial-LCDM covariance file.
- **Output file (THE INTERFACE the Implementer builds against):** one
  .npz per experiment configuration: `ell` (l = 2..lmax); per
  spectrum s in {tt, te, ee, pp}: `sigma_<s>` (sqrt of the Gaussian
  diagonal, ALWAYS present — what CmbDiagonalGeometry consumes) and
  `cov_<s>` (the dense l x l block, present ONLY when the NG flag was
  on); `provenance` (a json string: fiducial parameters, Delta_XY,
  theta_FWHM, fsky, the NG flag, the stencil step study, the exact
  camb extra_args) — resolved values persisted, nothing re-derivable
  demanded of the consumer.
- **D-CM4 adjustment:** data.cmb gains `covariance: <file>` (the
  script's .npz; REQUIRED — it replaces the inline fiducial_cl ->
  cinv derivation on the training path; from_fiducial stays for the
  synthetic gate fixtures). The geometry takes sigma_<spectrum> from
  the file; the file's provenance rides into the artifact.
- **D-CM12 (recorded, after Gaussian tests pass):** training WITH the
  NG covariance = a dense-Cinv chi2 variant (the contraction lives in
  the loss over the unchanged diagonal whitening — decision (2)
  survives); "we first test with Gaussian terms" (user), so this is
  sequenced behind the Gaussian-trained CME close.
- **Gates:** the script gets its own check ride: a Gaussian-part leg
  verifiable against the closed-form eq 3 on a synthetic Cl (Mac,
  numpy); the NG path's convergence harness output asserted
  well-formed; cmb-smoke consumes a real script-produced .npz once
  the script lands (until then the smoke may use from_fiducial with
  the ruled constants — recorded as the interim).

### ARCHITECT_HANDOFF: INCREMENT-1 RULINGS DELIVERED — PROCEED TO INCREMENT 2

- **covinv RULED** (eq 3 + noise; the true inverse; flip the flagged
  line); the seven decisions ENDORSED as annotated.
- **Increment 2 (D-CM4) adjusted:** data.cmb consumes the covariance
  .npz per the interface above; sigma from the file; provenance into
  the artifact; the noise/beam/fsky/NG facts live in the FILE, the
  YAML only points at it.
- **Division of labor:** the Architect implements
  compute_cmb_covariance.py (D-CM11) in parallel; the Implementer is
  unblocked on increments 2-5 against the interface; the NG-trained
  variant (D-CM12) waits for Gaussian results.

## OVERNIGHT EXECUTION (2026-07-10/11, Architect implementing directly)

**Authorization:** the user, going to sleep ~8h: "all pending task you
can implement yourself in the background... implement them all". The
Architect implements; every diff stays uncommitted on the branch; this
section is the resume state for whoever continues (me or the
Implementer).

### Landed + Mac-gated this window

1. **The covinv flip (increment 1's flagged line):** from_fiducial now
   computes cinv_l = (2l+1)/(2 C_fid^2), sigma_l = C_fid*sqrt(2/(2l+1))
   — the ruled true inverse; docstrings updated; probe confirms sigma
   at l=2, Cl=250 equals 158.11 (= 250*sqrt(2/5)) and DECREASES with l
   (the physical error bar; the placeholder grew with l).
2. **D-CM11: compute_data_vectors/compute_cmb_covariance.py (new,
   Architect-written).** Gaussian part always (eq 3 all seven blocks
   incl. TT-TE/TT-EE/TE-EE crosses + pp cosmic variance; eq-1 noise
   from delta/beam in muK-arcmin/arcmin; fsky explicit). NG part
   behind cov_args.nongaussian.enabled (default false): eq 6 via
   band-perturbed re-lensing through provider.get_CAMBdata() +
   get_lensed_cls_with_spectrum (one Boltzmann solve ever), the
   5-point stencil at >= 2 step sizes, per-band convergence vs
   converge_rtol LOUD, the step study persisted. N^(E) (eq 7)
   recorded + skipped (no BB emulator). Output = the ruled .npz
   interface (ell, sigma_<s>, gauss cross blocks, cl_<s> fiducials,
   cov_<s> dense when NG, provenance json). LCDM-only validation
   (fixed values, name whitelist, omk/w/wa pinned). Mac probe 10/10:
   eq 1 + eq 3 closed forms exact, stencil exact on quartics +
   O(h^4) on sin (ratio 16.0), band cover, all three LCDM-guard legs.
   KNOWN FIRST-RUN RISKS (workstation): the
   get_lensed_cls_with_spectrum call signature (clpp array convention
   [L(L+1)]^2 C/2pi, CMB_unit/raw_cl kwargs) and the "CAMBdata"
   requirement name — the SPE evaluate-YAML precedent; first real run
   rules.
3. **example_yamls/cmb_covariance_lcdm.yaml (new):** the three-block
   config with the user's verbatim high-accuracy camb settings.
4. **CME increment 2 (D-CM4), COMPLETE:**
   - geometries_cmb.py: the geometry now PERSISTS the amplitude law
     (law / as_name / tau_name in __init__ + from_fiducial + state) —
     D-CM1's "the artifact stores the law" gets its home; probe:
     9-key state round-trip bitwise incl. the law strings.
   - data_staging.py: load_source cuts are opt-in (omegabh2_hi=None
     skips phys_cut_idx — the D-SPE2 pattern; cosmolike callers always
     pass a value, unchanged).
   - experiment.py (10 edits): DATA_KEYS + "cmb"; is_cmb detection +
     scalar/cmb mutual exclusion; validate_cmb (pure fn, probed 13/13:
     required keys, spectrum/law whitelists, law-column rules both
     ways, exclusivity, five required files, rescale/ia/pce loud,
     transfer deferred-loud naming D-CM7, finetune interim-loud naming
     D-CM10); the from_config cmb branch (plain designs; heads =
     LOUD D-CM13 interim: their basis-change buffers assume an
     eigenbasis geometry — "as-is" in D-CM4 was optimistic, corrected);
     __init__ _cmb/cmb defaults; both stage fns cuts-optional on cmb;
     the build_geometry cmb branch (covariance .npz consumed per the
     interface, key/width checks loud, per-row f = exp(2 tau)/As from
     RAW C columns by name, the training-mean of the amplitude-rescaled
     target streamed in float64 chunks — probed exact to 2e-17 vs a
     float64 reference; CmbDiagonalGeometry built via __init__ = the
     npz path; make_cmb_chi2 dispatched by law); the print_design cmb
     banner.
   - results.py: info gains "cmb" (isinstance dispatch) +
     "amplitude_law"/"as_name"/"tau_name" (getattr off the geometry;
     None on non-CMB artifacts).

### NOT built (the honest boundary; designed, next in line)

- Increment 3 (D-CM3-A): generator_core extraction + the re-thinned
  lensing driver + dataset_generator_cmb.py. Untouched — the 1100-line
  verbatim-move surgery deserves fresh context, not the tail of this
  window.
- Increment 4 (D-CM5): emul_cmb.py + the predictor's CME decoder
  branch (endorsed decision 7). The decode needs the law: rebuild now
  surfaces amplitude_law/as_name/tau_name, and CmbFactoredChi2.decode
  is the single-source (build a law chi2 from the rebuilt geom +
  pgeom, reuse its decode — the transfer-decoder precedent).
- Increment 5 (D-CM6/8/9/10): gates cmb-identity/cmb-smoke + configs +
  example training YAML + the README section draft + the D-CM8
  roughness term + the D-CM9 diagnostics + the D-CM10 finetune leg
  (and the finetune-dispatch integration the interim error guards).
- SPE-FT / BSN / GEO / POL: specced, queued, untouched (stacking
  further un-boarded units overnight = unreviewed bulk).

### For the board (when the user wakes)

Nothing new is board-runnable yet (the CME gates are increment 5); the
landed work is Mac-gated only. The landing sequence in the chat handoff
commits the overnight diffs; cmb-identity/cmb-smoke arrive with
increment 5.

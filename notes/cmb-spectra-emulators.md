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

(none yet)

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

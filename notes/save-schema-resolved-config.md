---
name: save-schema-resolved-config
description: "SPEC 2026-07-07 (Architect): save schema v2 — the .emul + .h5 pair must reconstruct the emulator PERFECTLY even if every code default drifts (user directive). Confirmed gap: the h5 saves config_yaml / train_args_yaml (search-collapsed, NOT default-materialized) — omitted knobs (norm, scheduler kwargs, n_gates, gate_init, compile_mode, freeze_trunk, berhu knots, ema, thresholds...) are resolved inside build_specs / run_emulator and never persisted; reconstruction silently trusts frozen defaults. FIX = persist the consumed view (the banner directive extended to artifacts): (1) build_specs assembles exp.resolved_model — a serializable rebuild recipe (class qualname, dims, every constructor kwarg with factories serialized by name: activation {type,n_gates}, norm, head pin) built BESIDE the specs by the same code; (2) run_emulator returns resolved_train (every training knob defaults-materialized incl. per-phase resolved blocks + derived eval_bs) — declared interface change; (3) save_emulator v2: keeps the raw configs (provenance of what was WRITTEN) + adds config_resolved_yaml + model_recipe/ group + attrs schema_version=2 / git commit best-effort; (4) NEW rebuild_emulator(path_root, device) in results.py — reconstructs module+geometries from the h5 ONLY (a missing recipe key = loud error, NEVER a code default), loads the .emul, returns the inference-ready triple; (5) THE acceptance property: save -> rebuild -> bitwise-equal outputs on a probe batch, AND the drift test — monkeypatch code defaults, rebuild unchanged. v1 files: rebuild_emulator refuses loudly (they predate the guarantee). Out of scope: training-resume checkpointing (optimizer state). Gates GSV-A..C. SEQUENCED AFTER the NPCE unit (same files). IMPLEMENTED 2026-07-07 (Opus, base 378dc95), uncommitted, GSV-A/B PASS, GSV-C rides the workstation queue."
metadata:
  node_type: memory
  type: project
---

# Save schema v2: persist the consumed view (spec)

User directive 2026-07-07: "emul + h5 needs allow us to reconstruct
the emulator perfectly even if ALL default values of parameters
drift in the code over time."

Confirmed gap (results.py read): the h5 persists the geometry states
(solid), the histories, and config_yaml / train_args_yaml — the
driver's config with search ranges collapsed but DEFAULTS NOT
MATERIALIZED. Everything the YAML omitted is resolved at build/run
time and lost. The fix is the banner directive's second half:
displays render the consumed view; ARTIFACTS PERSIST IT — by the
same code that consumes, never a parallel re-implementation.

## Scope

1. exp.resolved_model (build_specs, assembled BESIDE the spec dicts
   so it cannot diverge): a fully serializable rebuild recipe —
   model class qualname + (name, ia, arch), input_dim / output_dim,
   every constructor kwarg actually passed, with callables
   serialized by name: activation {type, n_gates}, norm (the
   make_norm name), the head activation pin {type, n_gates} | None,
   mlp width / n_blocks, the active head's knobs, compile_mode as
   resolved, n_amps / n_templates for factored designs, needs_geom.
2. run_emulator returns resolved_train (DECLARED interface change:
   one added return, experiment.train stores it): assembled right
   after the up-front validation/defaulting — bs, nepochs, seed,
   thresholds, use_amp, clip, rewind, trunk_epochs, freeze_trunk
   (resolved bool), loss (defaults-filled incl. berhu knots +
   anneal | None), ema (resolved | None), lr {lr_base, bs_base,
   warmup_epochs, the computed lr}, optimizer {cls name,
   weight_decay, extras}, scheduler {cls name + kwargs}, trim /
   focus (defaults-filled), the per-phase blocks AS RESOLVED PER
   PASS, the derived eval_bs, device string.
3. save_emulator schema v2: KEEP config_yaml + train_args_yaml (the
   provenance of what the user WROTE) and ADD:
   - config_resolved_yaml: resolved_train + the resolved data block
     (n_train, n_val, split_seed, ram_frac, param_cuts filled);
   - model_recipe/ group: exp.resolved_model;
   - root attrs: schema_version = 2, code git commit (best-effort:
     rev-parse if a repo, else "unknown"), torch version + created
     (already there).
   The pce group ([[npce-yaml-wiring]]) joins config_resolved
   coverage when both land.
4. NEW rebuild_emulator(path_root, device) in results.py — the
   loader that PROVES the guarantee and the missing in-package
   inference entry point: reads model_recipe + the geometry states
   (+ pce when present), re-makes factories from names
   (make_activation / make_norm), instantiates the class, loads the
   .emul state_dict (strict=True), returns (model, param_geometry,
   geometry). HARD RULE: it reads ONLY the h5 — a missing recipe
   key is a loud error naming it; it NEVER falls back to a code
   default (that is the whole point). v1 files (no schema_version):
   refuse with a clear message (they predate the guarantee).
5. Out of scope, stated: training-resume checkpointing (optimizer
   moments, schedulers mid-run) — a different feature; this unit is
   inference reconstruction + provenance.
6. Docs: the h5 layout docstring (results.py) rewritten; the
   train_single driver header --save paragraph; the code map
   results.py entry (+ rebuild_emulator line); README Run-it
   outputs sentence gains "the h5 carries the fully-resolved config
   (schema v2) — a saved emulator rebuilds bit-exactly even if code
   defaults later change".

## Gates

- GSV-A (Mac, pure/stub): a completeness CENSUS — every run_emulator
  signature knob and every model constructor kwarg appears in
  resolved_train / resolved_model (diff the signatures against the
  dicts programmatically, so a future knob that forgets to join the
  recipe FAILS this gate); YAML serialization round-trip of both
  dicts; rebuild_emulator (stubbed io) refuses on a missing key and
  on schema_version < 2.
- GSV-B (Mac, static): resolved_model assembled beside the specs
  (line-ordered, same values, no parallel resolution); the
  interface changes declared; save schema groups/attrs; docs;
  scans + py_compile.
- GSV-C (workstation, rides the queue): train small -> save ->
  rebuild_emulator -> outputs BITWISE-EQUAL to the live model on a
  probe batch; THE DRIFT TEST: monkeypatch a handful of code
  defaults (make_scheduler patience, make_activation n_gates,
  make_model width, the norm default) and rebuild again — still
  bitwise-equal (proof the h5, not the code, defines the emulator);
  a v1 file refused with the clear message.

## Handoff

### ARCHITECT_HANDOFF
Task: save schema v2 — persist the consumed view (spec:
notes/save-schema-resolved-config.md IN FULL, including the two
riders and the standing rule at the bottom; the HARD RULE is
binding: rebuild_emulator reads only the h5, never a code default).
Base: the NPCE commit; `git log -1` must show it — else STOP.
Scope, four parts:
1. The CORE (items 1-6): exp.resolved_model beside the specs;
   run_emulator returns resolved_train (declared interface change);
   save v2 = config_resolved_yaml + model_recipe/ + schema_version 2
   + best-effort git attr (raw configs kept); NEW
   rebuild_emulator(path_root, device) — h5-only, missing key =
   loud, v1 refused; docs. NPCE is now LANDED, so the pce block
   JOINS config_resolved and rebuild_emulator rebuilds the base
   from the pce group when present (the spec anticipated this).
2. RIDER 1: the emultrfv2 install-path sweep (fifteen doc-only
   sites, enumerated in the rider; AST identity on the drivers;
   grep emultrf/dev to zero, notes/ exempt).
3. RIDER 2: the .paramnames staging cross-check (pure helper + one
   staging call; sidecar-vs-covmat-header names, order included;
   loud mismatch naming both lists; absent sidecar = no check).
4. The STANDING RULE section is binding on every choice: write side
   = defaults materialized; read side = never a code-default
   fallback.
Interface changes (run_emulator return, save_emulator args, new
rebuild_emulator, the staging check) declared in the report. Gates
GSV-A/B on the Mac (GSV-A includes the signature-vs-recipe census +
the rider cases; GSV-B includes the rider greps); GSV-C embedded
for the workstation queue. Report: IMPLEMENTER_HANDOFF + resume
state appended here, raw gate outputs, every YAML change as a
paste-ready block, deviations declared. Do not commit: leave the
diff uncommitted and print the suggested commit command.
### END

## Status

IMPLEMENTED 2026-07-07 (Opus, base 378dc95), uncommitted, COMMIT-READY.
CORE + Riders 1/2/3/4/5 in the tree; D-SV1 (shared
DEFAULT_COMPILE_MODE) and D-SV2 (eval_bs stash) landed. GSV-A census
ALL PASS (26 run_emulator params mapped, 6 model __init__ covered,
recipes serializable, rebuild hard-rule static, RIDER-2 exec cases,
D-SV1/D-SV2 value cases); GSV-B unified five-rule math scanner CLEAN
on both READMEs + RIDER-1 grep 0; py_compile clean. GSV-C
(bitwise + drift) rides the workstation queue. Suggested commit
sentence: "Save schema v2: the h5 persists the fully-resolved
consumed config + a serializable model recipe (defaults materialized
at the source of truth); rebuild_emulator reconstructs from the file
alone — immune to code default drift; riders: emultrfv2 paths, the
.paramnames staging cross-check, the unified GitHub-math rules (gates
GSV-A/B Architect-verified)".

## STANDING RULE (user, 2026-07-07): artifacts never trust defaults

Verbatim: "The philosophy over the emul and h5 file has to be —
dont trust on default values — they can drift."

The two halves, binding on THIS unit and on every future
save/load surface (the pce group, any future checkpointing, every
new knob):

- WRITE side: everything the run consumed is written, defaults
  MATERIALIZED at save time — the file records values, never the
  fact that a value was defaulted. Enforced mechanically by the
  GSV-A census (signatures diffed against the recipe: a future
  knob that skips the recipe fails the gate).
- READ side: reconstruction reads ONLY the file — a missing key is
  a loud error naming it, NEVER a fallback to a code default.
  Enforced by rebuild_emulator's hard rule + the GSV-C drift test
  (monkeypatched defaults, bitwise-equal rebuild).

This is the consumed-view doctrine's second half
([[banner-prints-consumed-view]]): displays RENDER the consumed
view; artifacts PERSIST it; loaders TRUST ONLY it. Every future
spec that adds a knob or a save surface must state how it honors
both halves, and every audit checks them.

## Rider (2026-07-07): the emultrfv2 install-path sweep

User: the repo will be cloned into Cocoa as
external_modules/code/emulators/emultrfv2/ (no /dev segment; the
old documented path emulators/emultrf/dev is retired). FIFTEEN
doc-only sites (enumerated by grep): README.md:62 (the Run-it D=
line), emulator/README.md:67 (the code-map layout note), and twelve
driver-header comment lines across the five drivers (the
"python .../emultrf/dev/..." examples and the "sits beside the
emulator/ package (same .../emultrf/dev/ folder)" notes). Replace
every emultrf/dev with emultrfv2. All comment/doc lines — AST
code-identity must hold on the drivers. Gate (joins GSV-B): grep
"emultrf/dev" over the tree (notes/ exempt) returns ZERO;
grep emultrfv2 hits the same fifteen sites.

## Rider 2 (2026-07-07): the .paramnames cross-check (naming integrity)

User closed the naming loop: the dumps ship a getdist-style
<train_params>.paramnames whose FIRST column is the cobaya name of
each .txt column (trailing chi2* = the derived chi2 the staging
slice already drops) — so ParamGeometry.names are cobaya names by
construction, and get_requirements speaks the sampler's language
natively (the lambda bridge is only for true reparametrizations).

The integrity gap: TWO name sources exist — read_param_names reads
the COVMAT HEADER; the .paramnames sidecar independently declares
the .txt columns. They agree today by generator construction; a
divergence would silently pair wrong columns with wrong covmat rows
in the whitening. RIDER: at staging, WHEN a .paramnames file sits
beside train_params, cross-check its first column (minus the
starred derived entries) against the covmat-header names, order
included — mismatch = a loud error naming both lists; absent
sidecar = no check (back-compatible). One pure helper + one call in
load_source/experiment staging; a GSV-A case each (match, mismatch,
absent, starred-tail handling).

## Rider 3 (2026-07-07): the TATT display render fix + the
## line-start math policy

User screenshot: the TATT equation renders as raw text split into
bullets. ROOT CAUSE (a sibling of the double-subscript bug): the
display's continuation lines START with "+ ", which GitHub's
Markdown reads as a bullet-list marker — the block is shredded into
a paragraph + two bullets before MathJax runs. The scanner checked
underscores, not line starts.

FIX (one block, README ~601): move the operators to the END of the
preceding lines so no continuation line starts with a Markdown
token:

    $$\xi = K_0 + a_1 K_1 + a_2 K_2 + a_1 b_{TA} K_3 +
    a_1^2 K_4 + a_2^2 K_5 + (a_1 b_{TA})^2 K_6 +
    a_1 a_2 K_7 + a_1^2 b_{TA} K_8 + a_1 a_2 b_{TA} K_9$$

POLICY EXTENSION (binding, joins the GRO-G math rules): inside a
$$ block, no continuation line may start with a Markdown
list/quote token (+, -, *, >, or digit-dot). The math scanner
gains this check permanently (the Architect's sweep found exactly
the one offender; both READMEs otherwise clean). GSV-B runs the
extended scanner.

## Rider 4 (2026-07-07): two more loss-section render fixes + two
## permanent math-policy additions

User screenshots, both root-caused:

1. The berhu piecewise shows a literal "\[4pt]": the source uses
   \begin{cases} with the row separator \\[4pt] — GitHub's inline
   Markdown escape pass eats one backslash, MathJax receives
   \[4pt] as text. Row-break environments are FRAGILE on GitHub.
   FIX (README ~361): replace the cases display with the
   environment-free one-liner (the style of the capped sentence
   right below it, which renders fine):

       $$L_{\mathrm{berhu}}(c) = \sqrt{c} \ \ (c \le k), \qquad
       \dfrac{c+k}{2\sqrt{k}} \ \ (c > k)$$

   (one line, parenthesized conditions, no \begin, no row breaks;
   if it must wrap, trailing operators per Rider 3.)
2. The C^1 sentence renders raw: "$k = $" and "$K = $" have a SPACE
   before the closing $ — GFM refuses such spans as math, the stray
   dollars then garble the line. FIX (README ~366-367): move the
   equals outside the span — "$k$ = `berhu.knot` (default 0.2, the
   frac>0.2 goal), $K$ = `berhu.cap` (default 10)".

POLICY ADDITIONS (permanent, join the Rider-3 scanner; code fences
exempt): (a) no \begin{...} environments and no \\[npt] row
separators inside GitHub math — piecewise definitions use
parenthesized conditions or prose-anchored spans; (b) no inline
$...$ span with whitespace adjacent to either delimiter. The
Architect's sweep over both READMEs found exactly the three
reported sites and no others (the $ROOTDIR hit is a bash comment
inside a code fence — exempt, and the scanner must exempt fences).
GSV-B runs the extended scanner (Riders 3 + 4 rules together).

## Implementer resume state (2026-07-07, Opus, base 378dc95) — PARTIAL, RESUME HERE

DONE + verified in the tree (uncommitted): RIDER 1 (14 emultrf/dev ->
emultrfv2; grep=0; driver AST unchanged; the "fifteen" was an Architect
miscount — 15th hit was in notes/, exempt). RIDER 3 (TATT $$ trailing
operators, README:601). RIDER 4 (berhu one-liner README:361 =
$$L_{\mathrm{berhu}}(c) = \sqrt{c}\ \ (c\le k), \qquad \dfrac{c+k}{2\sqrt{k}}\ \
(c>k)$$; the $k$ = / $K$ = whitespace-span fix README:366-367). The combined
Riders 3+4 scanner (strip ``` fences; $$ blocks: no continuation line starts
+/-/*/>/digit-dot, no \begin, no \\[npt]; inline $...$: no whitespace adjacent
to a delimiter) is CLEAN over both READMEs — this scanner is GSV-B's math leg.

NOT YET DONE — the CORE + RIDER 2. Halted before the run_emulator-return
interface change to avoid shipping it half-wired. FULL DE-RISKED BLUEPRINT
(every anchor confirmed, no re-exploration needed):
- run_emulator (training.py) return @2499 `return model, train_losses,
  medians, means, fracs` -> append resolved_train; experiment.train @1771
  `out = run_emulator(...)` unpacks it -> exp.resolved_train (MUST change
  together). Values: out_dim=chi2fn.dest_idx.numel() @2206; in_dim=
  getattr(param_geometry,"encoded_dim",train_set["C"].shape[1]) @2218;
  lr=lr_base*(bs/bs_base)**0.5 @2209; compile_mode default "reduce-overhead"
  (make_model:33); make_model @2220 = cls(input_dim, output_dim, **extra),
  extra = model_opts minus cls/compile_mode.
- resolved_model in build_specs before `return specs` @1724 -> self
  .resolved_model: top {cls qualname, name=self.arch, ia=self.ia, input_dim,
  output_dim (same two formulas), compile_mode, needs_geom}; kwargs = every
  specs["model_opts"] key except cls/compile_mode/geom, with block_opts ->
  {act:{type:self.activation,n_gates}, norm:norm_name}, head_act ->
  {type,n_gates}|None for head_block models (None=no pin; head_pin local).
  geom = census allowlist (rebuild passes the saved geometry).
- resolved_train (run_emulator): bs, nepochs, seed, thresholds, use_amp, clip,
  rewind, trunk_epochs, freeze_trunk (bool = is not False), loss
  (validate_loss result), ema (validate_ema|None), lr {lr_base,bs_base,
  warmup_epochs,lr:learning_rate}, optimizer {cls qualname,weight_decay,
  extras}, scheduler {cls qualname,+kwargs}, trim, focus, trunk/head (the
  per-pass RESOLVED blocks), eval_bs (the ACTUAL derived value, per ruling),
  device str.
- save_emulator v2 (results.py): add resolved_train/resolved_model args; KEEP
  config_yaml+train_args_yaml; ADD config_resolved_yaml + model_recipe/ (plain
  dict -> json string or nested attrs, NOT write_state which is for tensors) +
  attrs schema_version=2 + git (subprocess rev-parse, else "unknown"); pce
  group already written joins config_resolved. train_single caller passes both.
- rebuild_emulator(path_root, device) NEW: schema_version!=2 -> loud refuse;
  read model_recipe (missing key = loud, never a default); ParamGeometry
  .from_state(device,state) + DataVectorGeometry.from_state; pce group ->
  PCEEmulator.from_state; make_activation/make_norm from names into block_opts;
  instantiate recipe cls(input_dim,output_dim,**kwargs,[geom]); torch.load
  .emul + load_state_dict(strict=True); return (model,pgeom,geom).
- RIDER 2: pure check_paramnames (data_staging): read .paramnames col-0, drop
  trailing starred/chi2* derived, compare to covmat_names order-included, loud
  mismatch naming both; absent sidecar = return None. One staging call guarded
  on the sidecar beside train_params.
- GSV-A census (totality, Architect rule): each run_emulator param ->
  resolved_train key OR allowlist {train_set,val_set,chi2fn,param_geometry,
  model_opts,opt_opts,lr_opts,sched_opts,trim_opts,focus_opts,trunk_opts,
  head_opts,gpu_mem_gb,silent}; each of the 6 model __init__ -> recipe kwargs/
  top{input_dim,output_dim}/allowlist{geom}; + YAML round-trip + rebuild
  refusal (missing key, schema<2) + RIDER 2 cases. GSV-B: static + RIDER-1 grep
  + Riders 3+4 scanner + docs. GSV-C workstation (bitwise + drift proof).
- Docs remaining: results.py h5-layout docstring + rebuild_emulator; train
  _single --save paragraph; code map results.py entry; README Run-it outputs
  sentence (schema v2 rebuilds bit-exactly under default drift).

## Rider 5 (2026-07-07): the UNIFIED math rule — no backslash +
## punctuation, ever

User found a literal "!" in the clip equation: the corpse of \! (a
negative thin space). ROOT CAUSE, now unified: GitHub's escape pass
eats BACKSLASH + ASCII PUNCTUATION (\_ -> _, \\ -> \, \! -> !,
\, -> ,) while backslash + letters (\sqrt, \frac, \qquad) survive.
Every math bug found so far — the ema double-subscript (\_), the
TATT bullet shred (line-start + after \\ handling), the cases
\\[4pt], the \! bang — is ONE class.

FIXES (twelve instances across nine lines, Architect-enumerated;
all pure typography, all deletions): the \! in the clip equation
(README ~325); the \, thin spaces in the param_cuts table rows
(~292-294, 3x), the berhu blend (~381, 2x), the lr rule (~410), the
weight-decay equation (~437, 2x), the ema equation (~540, 2x), the
NPCE ratio (~824). Replace \, with a plain space; delete \!.
(\ backslash-space passes GFM's escape list and may stay.)

THE PERMANENT RULE (subsumes Riders 3-4's specifics, which remain
as named instances): inside math, outside code fences, NO backslash
followed by ASCII punctuation, for any purpose. The scanner's
general check: flag every \<punct> in a math span. Spacing wants =
plain spaces or \quad / \qquad (letters survive). GSV-B runs the
unified scanner (Riders 3 + 4 + 5).

### 2026-07-07 — Architect: riders 1/3/4 AUDITED (accepted), GO for
### the CORE pass

The partial stop is COMMENDED, not tolerated: a half-wired
run_emulator-return interface breaks every run, and stopping at a
valid tree with an executable blueprint is exactly the right
failure mode. Riders 1/3/4 verified with the Architect's own
probes: emultrf/dev grep = 0 with all five drivers AST-identical;
the TATT display has no Markdown-token line starts; the cases
environment is gone; the k/K spans are fixed. The twelve Rider-5
backslash-punctuation sites remain (292-294, 325, 381 x2, 410,
437 x2, 540 x2, 824) — Rider 5 postdates the Implementer's pass.

GO DIRECTIVE (the fresh pass): execute the CORE + RIDER 2 + RIDER
5 from the blueprint. Base = THIS tree as it stands (riders 1/3/4
uncommitted in place; verify them present via the grep + scanner
before starting; if the tree does not match, STOP). Rider 5 =
delete the twelve \, / \! instances (plain spaces; pure
typography) + the scanner gains the UNIFIED rule (no backslash +
ASCII punctuation in math, fences exempt; riders 3/4 rules remain
as named instances). One combined commit suggestion at the end
covering riders + core (the unit as originally scoped). All prior
rulings stand (census totality; consumed-view recipe; the
non-config allowlists).

### 2026-07-07 — Architect re-audit of the CORE: ACCEPTED WITH TWO
### DELTAS (fix before commit)

Verified with own probes: footprint exact (4 code files + docs;
loss_functions/batching untouched); the resolved_train dict solid
(qualnames, computed lr, per-pass blocks, device string); the
recipe assembly exemplary where it matters most — constructor
defaults materialized via inspect.signature (the source of truth)
— and in_dim mirrors run_emulator's own derivation (consistent,
GSV-C backstops bitwise); my census: all 26 run_emulator params
mapped (wiring allowlist + config -> resolved keys), no silent
skips; check_paramnames match/order/length cases pass with the
absence guard correctly at the load_source call site (the
Architect's harness tested the wrong contract there — owned); the
unified five-rule math scanner CLEAN on both READMEs (Rider 5's
twelve deletions verified); rider-1 grep 0; py_compile clean.

THE TWO DELTAS (both latent drift channels inside the anti-drift
unit — the irony is the finding):

- D-SV1: the recipe's compile_mode falls back to a HARDCODED
  "reduce-overhead" — a duplicate of make_model's default. If
  make_model's default drifts, the recipe records the stale copy.
  FIX: one shared module constant (DEFAULT_COMPILE_MODE in
  training.py) consumed by BOTH make_model and the recipe assembly
  — one source, no duplicate.
- D-SV2: resolved_train["eval_bs"] reads data.get("eval_bs") but
  NOTHING writes it — the key is permanently None (the census
  passed on key-existence). FIX: training_loop_batched stashes
  data["eval_bs"] = eval_bs right after deriving it (the dict is
  shared with run_emulator); GSV-A gains a VALUE case (the stash
  statically verified / eval_bs non-null in the census) so
  key-exists-but-null cannot pass again.

Deviations RULED: (1) YAML-string datasets ACCEPTED (config text,
not tensors; round-trip identity is the requirement). (2) the data
block as-written ACCEPTED with the reasoning recorded — split_seed
/ ram_frac affect training provenance, not the reconstruction
guarantee (weights + recipe define the emulator), and duplicating
staging defaults in save_emulator would itself be a drift channel;
OBSERVATION of the same class: phase-block loss/ema sub-defaults
are not per-pass materialized — provenance, not reconstruction; a
future provenance-perfection item if rerun-exact provenance is
ever wanted. (3) the pce base rebuilt-but-local ACCEPTED (the
triple was the Architect's contract); FORWARD NOTE: the cobaya
predictor unit either extends rebuild's return (declared then) or
reads the pce group directly — decided there. The two remaining
prose docs (train_single --save paragraph; README outputs
sentence) CARRIED to the generator-import unit (GDG-B gains them).

After D-SV1/D-SV2 land: COMMIT-READY, one combined commit
(riders + core). Suggested sentence: "Save schema v2: the h5
persists the fully-resolved consumed config + a serializable model
recipe (defaults materialized at the source of truth);
rebuild_emulator reconstructs from the file alone — immune to code
default drift; riders: emultrfv2 paths, the .paramnames staging
cross-check, the unified GitHub-math rules (gates GSV-A/B
Architect-verified)".

### 2026-07-07 — D-SV1 + D-SV2 CLOSED (Architect-verified): the unit
### is COMMIT-READY

Own probes: DEFAULT_COMPILE_MODE defined once (training.py:98),
consumed by make_model (:134) AND the recipe (experiment.py:1746
via the :80 import) — no duplicated code fallback anywhere; the
eval_bs stash (:1594) follows the derivation and resolved_train
reads it (:2536), never a null literal; py_compile clean. The
explicit-path commit staging (excluding the compute_data_vectors
unit's artifacts) is the right call; MEMORY.md carrying that
unit's description-only index edit is accepted as flagged. GSV-C
(the bitwise + drift proof) rides the workstation board.

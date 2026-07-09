---
name: activation-families-norm-knob
description: "SPEC 2026-07-07 (Architect): relu + tanh activation families + the model.norm YAML knob. User confirmed the paper's prescription ('affine normalization of the form gx + b, with g b being numbers, between each layer before activation') IS the package's existing hardwired default (ResBlock: Linear -> Affine -> act, scalar gain/bias, init identity, weight decay off). New: make_activation gains relu / tanh (parameter-free factories); model.norm selects the ResBlock norm slot — affine (default, byte-identical, the paper) | per_feature (dim-sized gain/bias, a new 10-line Affine sibling; the tanh saturation guard) | none (nn.Identity, ablation). batchnorm DELIBERATELY NOT OFFERED — user deferred to the Architect ruling 2026-07-07 (batch coupling confounds the bs/EMA experiments; train/eval mode split + compiled-twin risk; BN buffers sit outside the theta_bar average; the paper prescribes the affine INSTEAD of batchnorm; per_feature is the escalation; a whitelist value is easy to add later, hard to remove). Threading mirrors activation (model-level key -> block_opts['norm']; describe_spec order gains norm; loud whitelist error; ordinary categorical sweep leaf). TRF's internal LayerNorm untouched; CNN head blocks have no norm slot. Gates GAN-A..C. Sequenced after the TATT micro-commit. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# relu + tanh families and the model.norm knob (spec)

User decisions 2026-07-07: (1) the paper quote — "To avoid
saturating the Tanh activation function, we employ an affine
normalization of the form gx + b, with g b being numbers, between
each layer before activation" — is confirmed as the target scheme,
and it IS the package's existing default (ResBlock: Linear ->
Affine -> act; Affine = scalar gain/bias, init 1/0, weight decay
off; verified in the tree). (2) batchnorm is NOT offered — the user deferred to the Architect
ruling ("if so I am ok not offering it"): batch coupling would
confound the bs / EMA experiments; BN splits train/eval into two
networks (running stats) with a mode-baking risk under the
compiled eval twin; BN buffers sit outside theta_bar (the
snapshot invariant gains an asterisk); the paper prescribes the
affine INSTEAD of batchnorm; per_feature is the escalation. The
README knob text documents this reasoning in two sentences (the
teaching value). Whitelist: affine | per_feature | none.

## The YAML

    model:
      activation:
        type: tanh         # families: H | power | multigate |
                           # gated_power | relu | tanh (new two:
                           # parameter-free)
      norm: affine         # affine (default = today, byte-identical;
                           # the paper's gx + b, one scalar pair per
                           # layer) | per_feature (dim-sized g, b;
                           # the tanh saturation guard) | none
                           # (batchnorm deliberately not offered:
                           # batch coupling + train/eval split + the
                           # EMA buffer asterisk; see the README)

## Scope

1. make_activation: "relu" -> lambda dim: nn.ReLU(); "tanh" ->
   lambda dim: nn.Tanh(). Docstring + error message list them. The
   bake-off default list stays the four learnable families
   (relu / tanh join via --activations) — veto-able.
2. A FeatureAffine module (emulator_designs_building_blocks.py):
   the 10-line per-feature sibling of Affine — gain/bias of shape
   (size,), init ones/zeros, weight decay off automatically (ndim
   1). Docstring mirrors Affine's.
3. model.norm: a model-level key like activation (NOT an mlp
   sub-key). from_config/build_specs validate against the three-value
   whitelist (loud error naming them) and set block_opts["norm"] to
   the factory: affine -> lambda s: Affine() (absent = this,
   byte-identical); per_feature -> FeatureAffine; none ->
   lambda s: nn.Identity(). describe_spec's
   order list gains "norm" (consumed-view banner). Sweepable as an
   ordinary categorical leaf (build_specs reads train_args).
4. Where it applies: the ResBlock layers only — the trunk of every
   architecture and the IA template trunks. The TRF block's internal
   LayerNorm is architectural and untouched; the CNN head blocks
   have no norm slot. Documented.
5. batchnorm NOT offered (the ruling above): the README knob text
   carries the two-sentence why (batch coupling vs the bs / EMA
   experiments; train/eval running-stats split + the theta_bar
   buffer asterisk; the paper's affine is the replacement). The
   whitelist error message does NOT suggest batchnorm.
6. Docs: the README model section gains a `### norm` SUBSECTION
   (user directive 2026-07-07) beside mlp / activation / cnn / trf,
   with its own small YAML block (the subsection-code-block style
   rule), the three values one line each, the two-sentence
   why-not-batchnorm, and a one-line DEFINITION of "feature" (one
   coordinate of the width-wide hidden vector inside the trunk —
   the ResBlock width, model.mlp.width; a per-feature pair g_i, b_i
   per column of the (B, width) tensor, shared across the batch;
   the same sense in which H's gamma / beta are per-feature).
   FeatureAffine's docstring carries the same definition. Plus:
   relu / tanh in the activation subsection's family list + the
   activation appendix, the code-map appendix lines
   (make_activation, FeatureAffine), train_single + tune YAML
   comments.

## Gates

- GAN-A (Mac, pure/exec): make_activation relu/tanh factories;
  norm whitelist validation (three accepted, absent = affine,
  unknown loud); FeatureAffine shapes (gain/bias (size,), identity
  init); block_opts["norm"] threading cases.
- GAN-B (Mac, static): absent norm = byte-identical (the default
  factory is textually today's); describe_spec order gains norm;
  ResBlock untouched (the slot already exists); TRF/CNN untouched;
  weight-decay grouping puts FeatureAffine's ndim-1 params in the
  undecayed group (verify make_optimizer's ndim rule covers it);
  the README `### norm` subsection present with its own YAML block
  (the style rule) + the feature definition + the
  why-not-batchnorm sentences; relu / tanh in the family lists;
  scans + py_compile.
- GAN-C (workstation, rides the queue): a tanh + per_feature run
  (smoke: banner shows norm; loss descends) + a tanh + affine run
  (the classic baseline); golden absent-key byte-identity run.

## Handoff

### ARCHITECT_HANDOFF
Task: relu + tanh activation families + the model.norm knob (spec:
notes/activation-families-norm-knob.md, read in full; scope
item 5's batchnorm-not-offered documentation is binding). Base: the
model-section micro-unit commit; `git log -1` must show it — else
STOP. Scope items 1-6 exactly; the bake-off default list stays the
four learnable families (flag if you disagree). Gates GAN-A/B on
the Mac; GAN-C embedded for the workstation queue. Report:
IMPLEMENTER_HANDOFF + resume state appended here, raw gate
outputs, every YAML change as a paste-ready block, deviations
declared. Do not commit: leave the diff uncommitted and print the
suggested commit command.
### END

## Status

SPEC DELIVERED 2026-07-07, NOT implemented. Sequenced after the
TATT/model-section micro-commit. Suggested commit sentence: "relu +
tanh activation families + the model.norm knob (affine default =
the paper's gx+b, per_feature, none; batchnorm deliberately not
offered, reasons documented; consumed-view banner; gates GAN-A/B
Architect-verified)".

## Addendum (2026-07-07, pre-build): the feature definition + the
## comparison table go in the README verbatim (user directive)

Item 6's "one-line definition" is upgraded: the `### norm`
subsection carries BOTH of the following, verbatim (they replace
the one-liner, not add to it):

The definition + diagram:

    A "feature" is one coordinate of the hidden vector flowing
    through the trunk — the ResBlock width (`model.mlp.width`).
    Inside every ResBlock the batch is a `(B, width)` tensor: B
    rows (cosmologies), width columns — the columns are the
    features:

                     features (columns, width = model.mlp.width) ─▶
              ┌─  x_1   x_2   x_3   ...   x_128 ─┐
      B rows  │  (one row = one cosmology's       │
      (batch) │   hidden representation)          │
              └───────────────────────────────────┘

      per layer   :  g·x + b       one (g, b) pair for the whole tensor
      per feature :  g_i·x_i + b_i  one pair per COLUMN, shared by every row

    Not a data-vector element, not a cosmological parameter, not a
    sample — the same sense in which H's gamma / beta are
    per-feature.

The table:

    | | `affine` (per layer — the paper's) | `per_feature` |
    |---|---|---|
    | parameters | one pair $(g, b)$ for the whole layer — 2 | vectors $g_i, b_i$ — 2·width (256 at width 128) |
    | action | $g \cdot x + b$ broadcast over all features | $g_i x_i + b_i$, each feature its own |
    | what it can fix | a global scale drift | each unit's individual operating point |

(GRO-G-safe: single-letter symbols and subscripts only.) GAN-B's
docs check now asserts both the diagram and the table in the
subsection. Budget: the subsection may run ~35 lines with these —
acceptable, it is the knob teaching the concept.

## Implementer resume state (2026-07-07, Opus, base 83a1e58)

IMPLEMENTED, uncommitted, awaiting Architect re-audit. All six scope
items + both gates green.

Code:
- `activations.py` — make_activation gains `relu` -> lambda dim:
  nn.ReLU() and `tanh` -> lambda dim: nn.Tanh() (parameter-free, ignore
  dim); docstring + the error message list all six families.
- `emulator_designs_building_blocks.py` — new `FeatureAffine(size)` (the
  per-feature Affine sibling: gain/bias (size,), init ones/zeros, ndim-1
  so undecayed; docstring carries the feature definition) + `make_norm`
  (affine -> `lambda s: Affine()` = the ResBlock default, byte-identical;
  per_feature -> FeatureAffine; none -> `lambda s: nn.Identity()`; loud
  whitelist error that does NOT suggest batchnorm).
- `experiment.py` — imports make_norm; build_specs reads model.norm in
  the model loop (default "affine"), adds norm to the unknown-key error,
  injects `block_opts["norm"] = make_norm(norm_name)` beside the act
  injection (reaches every trunk ResBlock incl. the IA templates).
- `emulator_designs.py` — describe_spec order gains "norm" after
  "activation" (consumed-view banner).
- `training.py` — make_optimizer doc lists FeatureAffine among the
  undecayed ndim-1 params (the ndim>=2 rule already covers it; no code
  change to the split).
- `bakeoff_activation_...py` — ACTS stays the four learnable families
  (the default), new BUILDABLE_ACTS = ACTS + [relu, tanh] validates
  --activations, so relu / tanh join via the flag (item 1's parenthetical
  needed this; the hard ACTS whitelist would otherwise reject them).

Docs: README `### norm` subsection (its own YAML block, the three values,
the two-sentence why-not-batchnorm, the feature definition + diagram +
table verbatim from the addendum); relu / tanh in the activation
subsection list + the activation-appendix table; the code map gains
FeatureAffine + make_norm and the make_activation line lists relu / tanh;
train_single + tune YAML model comments gain relu / tanh + a commented
norm block (behavior unchanged, comments only).

Gates:
- GAN-A (exec, stubbed torch): make_activation("relu"/"tanh")(dim) build
  nn.ReLU / nn.Tanh; "H" unchanged; unknown lists all six. FeatureAffine
  gain/bias (128,), init ones/zeros; Affine stays (1,). make_norm's three
  factories build Affine / FeatureAffine / Identity; unknown names the
  three, not batchnorm. All PASS.
- GAN-B (static): make_norm affine branch == ResBlock default
  `lambda s: Affine()` (byte-identical); describe_spec order has "norm";
  build_specs injects block_opts["norm"] and defaults affine; the
  unknown-key error lists norm; make_optimizer doc lists FeatureAffine;
  README norm subsection + diagram + table + why-not-batchnorm present;
  relu/tanh in both family lists; anchors resolve; GRO-G clean; no
  ` -- `, no emphasis caps in the norm subsection; new .py <= 90 cols;
  the YAML additions are comment-only. All PASS. py_compile clean
  (package + drivers).
- GAN-C: workstation queue (tanh+per_feature smoke, tanh+affine baseline,
  golden absent-key byte-identity).

Deviations / decisions:
1. The bake-off default staying the four learnable families is AGREED
   (not vetoed) — relu / tanh are parameter-free baselines, so a default
   bake-off compares the learnable families; they opt in via
   --activations. Realizing "join via --activations" required widening the
   accepted set (BUILDABLE_ACTS) while keeping the default (ACTS) at four
   — declared, since the driver was not in the file list.
2. norm is threaded / validated in build_specs (via make_norm), not
   from_config — there is no --norm flag, so it is a pure YAML knob; the
   loud error fires at build_specs (before training), mirroring how the
   activation FACTORY and n_gates are already consumed there.
3. The addendum diagram's "per COLUMN" de-capped to "per column" (the
   standing de-caps rule; same width, alignment unchanged).
4. model.norm is sweepable with no extra code (validate_sweep_paths
   passes model.*; build_specs reads the swept train_args leaf) —
   verified, not just asserted.

Untracked notes present in the tree that the Implementer did NOT author:
weight-decay-only-weight-matrices.md, readme-run-it-definitions.md
(left untouched).

Awaiting Architect re-audit.

### 2026-07-07 — Architect re-audit: ACCEPTED, no deltas

Own harness 11/11 (torch-stubbed exec of make_activation /
FeatureAffine / make_norm from the tree): relu/tanh factories exact,
H unchanged, unknown-family error lists all six; FeatureAffine
gain/bias (size,) identity-init vs Affine's scalar (1,); make_norm's
three values map exactly; batchnorm rejected WITHOUT being
suggested. Statics: AST footprint exact (five code files; training.py
+ IA docstring-only/untouched); build_specs threading clean (norm
intercepted in the model loop, factory injected beside act, unknown
key error gains norm); the byte-identity argument VERIFIED SOUND
(Affine init is deterministic ones/zeros — no RNG draw — so the
injected default factory constructs identical state in identical
order; the global RNG sequence is untouched); describe_spec order
gains norm; ACTS stays the four learnable families with
BUILDABLE_ACTS widening --activations; the README ### norm
subsection carries the YAML block + the feature definition/diagram
+ the comparison table + the why-not-batchnorm, de-capped; code map
updated; anchors + math scanner + py_compile clean.

Deviations RULED: (1) BUILDABLE_ACTS ACCEPTED — the spec gap was the
ARCHITECT'S ("join via --activations" without naming the driver's
hard ACTS whitelist); minimal, default untouched. (2) build_specs
threading ACCEPTED (matches the spec's wording; a pure YAML knob has
no flag to catch in from_config). (3) de-caps ACCEPTED (standing).
(4) sweepability verified, noted with credit.

COMMIT-READY (sentence in Status). GAN-C joins the workstation board
beside GFT-C / GHA-F. Next units: the README didactic pass
([[readme-run-it-definitions]]), then the weight-decay fix
([[weight-decay-only-weight-matrices]]).

### 2026-07-08 — board verdict (Architect): GAN-C relu-tanh-norm PASS
tanh + norm 'per_feature' (226,755 params) and tanh + norm 'affine'
(224,723) both train rc 0 with their model-spec banners; the norm knob
is visible in the 2,032-param difference (per-feature gains the
feature-wise gain/bias). Green runs 3-11.

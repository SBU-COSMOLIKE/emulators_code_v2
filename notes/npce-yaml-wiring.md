---
name: npce-yaml-wiring
description: "SPEC 2026-07-07 (Architect): first-class NPCE — the PCE closed-form base becomes YAML-drivable ('trunk = PCE, head = any SGD model'). User: 'Past failures on low T do not discourage me — make a plan.' Route: LOSS-SPACE (the recommendation; the folder's own design): a NEW TOP-LEVEL pce: block (sibling of data/train_args — deliberately NOT inside train_args: sweep_hyperparam builds geometry once and deep-copies train_args per point, so a pce knob there would sweep WITHOUT refitting = silent no-op; top-level = unsweepable by construction, one study per pce config like model.name). Schema mirrors from_training verbatim: {form: residual | ratio (required), p_max 4, r_max 2, q 0.5, k_max 40, loo_max 0.05, max_terms 30, max_fail 4}. Wiring: pure validate_pce (+ exclusivity: pce+rescale error, pce+model.ia error — both replace chi2fn); from_config stash; build_geometry fits PCEEmulator.from_training on the staged train set and wraps PCEResidualChi2/PCERatioChi2; banner line + quiet-gated fit report; PCEEmulator gains state()/from_state (buffers lo/hi/multi_index/C/Vk/Ybar) + a pce h5 group in save_emulator; diagnostics plain-only gate extends to PCE losses. The refiner = whatever model.name says (resmlp/rescnn/restrf; all current knobs incl. its own two-phase, activation pins, norm, berhu, EMA — the loss ladder already forwards kwargs and needs_params/batching plumbing already exist). PCEEmulator stays OUT of MODELS and DesignSpec (loss-owned, documented). Gates GPC-A..C. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# NPCE from the YAML: the closed-form trunk + any SGD head (spec)

User directive 2026-07-07: wire NPCE first-class, per the Architect
recommendation (loss-space; "past failures on low T do not
discourage me"). The audit that grounds this:
the PCE LOSS layer is already current (kwargs forwarded to the one
_reduce ladder -> berhu et al. work; needs_params declared;
batching.py carries the ratio form's target_dim plumbing); only the
CONFIG layer predates MODELS/DesignSpec/YAML. This unit adds the
config layer without moving the design.

## The YAML (new top-level block)

    pce:                   # presence = NPCE: fit the closed-form
      form: residual       # sparse-Legendre base on the staged
                           # training set, then train model.name as
                           # the refiner of it.
                           # residual = base(theta) + net(theta)
                           # ratio    = base(theta) * (1 + net(theta))
      # fit knobs (defaults = from_training's; all optional):
      # p_max:     4       # max total degree (the smoothness knob)
      # r_max:     2       # max interaction order (vars per term)
      # q:         0.5     # hyperbolic sparsity exponent in (0, 1]
      # k_max:     40      # max leading SVD modes to try
      # loo_max:   0.05    # keep a mode only if relative LOO < this
      # max_terms: 30      # per-mode active-set cap
      # max_fail:  4       # stop after this many consecutive misses

WHY top-level (not train_args): sweep_hyperparam stages + builds
geometry ONCE and deep-copies train_args per point — a pce knob
inside train_args would sweep without refitting the base (a silent
no-op, the exact trap validate_sweep_paths exists to kill).
Top-level = structurally unsweepable/unsearchable; one study / one
sweep per pce config (the model.name rule). Documented in the
precedence appendix.

## Scope

1. validate_pce (pure, experiment.py beside validate_param_cuts):
   key whitelist (the eight above); form in {residual, ratio}
   (required when the block is present); ints >= 1 for p_max /
   r_max / k_max / max_terms / max_fail; q in (0, 1]; loo_max > 0;
   returns the defaults-filled dict or None (absent). EXCLUSIVITY
   (loud, at config time): pce + rescale != "none" -> error;
   pce + model.ia -> error (each replaces chi2fn; one at a time).
2. from_config: validate + stash exp.pce_opts (None = absent =
   byte-identical everywhere; every wiring point guards on it).
3. build_geometry: after the geometry + before build_specs, with
   the staged train set in hand — materialize X_white (the
   pgeom-whitened params) and Y_white (geom.encode of the train
   dvs; the fit converts to float64 numpy internally), call
   PCEEmulator.from_training(device, X_white, Y_white,
   **pce_opts_minus_form, silent=self.quiet), then wrap
   chi2fn = PCEResidualChi2(geom, pce) | PCERatioChi2(geom, pce)
   per form. The fit report lines print quiet-gated beside the
   loading-sources lines. NOTE: sweep_ntrain / bakeoff rebuild
   geometry per N -> the base REFITS per point (correct, the
   learning curve includes the base's data-dependence);
   sweep_hyperparam keeps one base across points (by design).
4. Persistence: PCEEmulator gains state() / from_state mirroring
   the geometry classes (the six registered buffers lo / hi /
   multi_index / C / Vk / Ybar); save_emulator writes a "pce" h5
   group (+ a form attr) when the chi2fn carries a base, so
   inference rebuilds base + refiner with no refit and no
   cosmolike. (The refiner .emul is unchanged.)
5. Diagnostics: the local-linear floor's plain-chi2-only gate
   extends to skip the PCE losses (the rescale precedent).
6. Banner (consumed view): print_design gains, when pce_opts is
   set, one line "pce: form residual  p_max 4  r_max 2  q 0.5
   k_max 40  loo_max 0.05  max_terms 30 (base fit at staging; report
   below)". The kept-modes / terms summary is the runtime fit
   report (it does not exist at banner time).
7. The refiner is UNRESTRICTED except ia: model.name resmlp /
   rescnn / restrf with every current knob — its own two-phase
   schedule, freeze_trunk, per-head activation pins, model.norm,
   the loss ladder incl. berhu (the PCE losses forward kwargs),
   trim / focus / clip / rewind / EMA. "Trunk = PCE, head = MLP"
   is the resmlp case.
8. DesignSpec / MODELS: PCEEmulator deliberately stays OUT of both
   — it is loss-owned, not an SGD architecture; documented in the
   code map.
9. Docs: the README YAML chapter gains a `## pce:` section (a new
   top-level block -> its own numbered section after the two-phase
   one; renumber + anchors): what NPCE is in three sentences, the
   two forms' equations
   $$\mathrm{pred} = B(\theta) + f(\theta) \qquad
   \mathrm{pred} = B(\theta)\,(1 + f(\theta))$$
   (legend: B = the closed-form base, f = the refiner), the YAML
   block above, the two design rules from the class docstring
   (well-predicted modes only; low degree), the exclusivity +
   unsweepable facts (pointer to precedence). Precedence appendix:
   rows for pce+rescale / pce+ia errors + the not-sweepable rule.
   Code map: the PCE folder entry updated (wired, no longer
   manual-only) + state()/from_state lines. train_single YAML: the
   commented pce: block. Vocabulary box: unchanged.

## Gates

- GPC-A (Mac, pure): validate_pce — defaults fill, absent -> None,
  form required + whitelisted, each numeric bound, unknown key
  loud, the two exclusivity errors (messages name both sides).
- GPC-B (Mac, static + stub-exec): every wiring point guarded on
  pce_opts (absent = byte-identical, textually verified);
  build_geometry order (fit after geom, wrap before specs;
  quiet-gated report); state()/from_state round-trip on fabricated
  buffers (numpy stubs); save_emulator pce group guarded; the
  diagnostics gate; the banner line; docs presence (README section,
  precedence rows, code map, YAML block); scans + py_compile.
- GPC-C (workstation, rides the queue): a small residual-form run
  AND a ratio-form run (fit report prints kept modes; the refiner
  descends; save -> the h5 pce group -> from_state rebuild matches
  base(theta) on a probe batch); the exclusivity errors fire from
  real YAMLs; a 2-point sweep_ntrain smoke showing the base refit
  per point; golden absent-pce byte-identity run.

## Handoff

### ARCHITECT_HANDOFF
Task: NPCE YAML wiring (spec: notes/npce-yaml-wiring.md in full;
the loss-space route is binding — PCEEmulator stays out of MODELS
and DesignSpec). Base: the overnight combined commit; `git log -1`
must show it — else STOP. Scope items 1-9 exactly; the pce: block
is TOP-LEVEL (the why is in the note; do not move it into
train_args). Gates GPC-A/B on the Mac; GPC-C embedded for the
workstation queue. Report: IMPLEMENTER_HANDOFF + resume state
appended here, raw gate outputs, every YAML change as a paste-ready
block, deviations declared. Do not commit: leave the diff
uncommitted and print the suggested commit command.
### END

## Status

SPEC DELIVERED 2026-07-07, NOT implemented. Sequenced after the
overnight combined commit. Suggested commit sentence: "NPCE from
the YAML: top-level pce: block fits the sparse-Legendre base at
staging and wraps the residual/ratio refiner losses — any
architecture as the head, base persisted to the h5, exclusive of
rescale/ia, structurally unsweepable (gates GPC-A/B
Architect-verified)".

## Implementer resume state (2026-07-07, Opus, base 75c429e)

IMPLEMENTED, uncommitted, awaiting Architect re-audit. All nine scope items;
the loss-space route held (PCEEmulator stays out of MODELS + DesignSpec); the
pce: block is top-level.

Code:
- experiment.py — `PCE_KEYS` + `validate_pce(pce, rescale, ia)` (pure): the
  eight-key whitelist, form required + in {residual, ratio}, positive-int
  bounds, q in (0,1], loo_max > 0, defaults-fill, and BOTH exclusivity errors
  (pce+rescale, pce+ia — messages name both sides). from_config validates +
  stashes exp.pce_opts (None = off); __init__ defaults it None. build_geometry:
  after geom (and needs_bins), before the make_chi2 / factored branches, a
  pce_opts-guarded branch materializes X_white = pgeom.encode(params), Y_white
  = geom.encode(dvs) via the loaders' torch.from_numpy(...).float().to(device)
  path, calls PCEEmulator.from_training(device, X_white, Y_white,
  silent=self.quiet, **fit_opts_minus_form), and wraps PCEResidualChi2 |
  PCERatioChi2 per form (early return). print_design: a pce_opts-guarded banner
  line (the fit knobs; the kept-modes summary is the runtime report).
- PCE/emulator_designs.py — PCEEmulator.state() (the six buffers) + from_state
  (rebuild with from_training dtypes: f32 buffers, long multi_index).
- results.py — save_emulator gains pce / pce_form params; writes a "pce" h5
  group (write_state(pce.state()) + a form attr) when a base is passed.
  train_single passes exp.chi2fn.pce + exp.pce_opts["form"] on NPCE runs.
- Item 5 (diagnostics): NO code change needed — PCEResidualChi2 / PCERatioChi2
  declare needs_params = True, and the caller (train_single:372) already skips
  local_linear_floor for needs_params losses (the rescale precedent). Verified.
- Items 7/8: no code — the refiner is any model.name (the loss ladder already
  forwards kwargs; batching carries needs_params + target_dim); PCEEmulator was
  never in MODELS / DesignSpec (documented in the code map).

Docs (item 9): README gains `## 12. pce` (what NPCE is, the two-form equation
$$pred = B + f | B(1+f)$$ + B/f legend, the YAML block, the two design rules,
the exclusivity + unsweepable pointer); the appendices renumbered 12->13 ..
16->17 (headers + Contents + anchors + the vocab-box displayed numbers, via a
single-pass anchor remap); precedence rows (E: pce not-sweepable; F: pce
exclusive with rescale/ia); code map PCE entry updated (wired; state/from_state;
out of MODELS/DesignSpec); train_single YAML commented top-level pce: block.
Vocabulary box unchanged (as specced).

Gates:
- GPC-A (exec of the pure validate_pce): absent->None, defaults fill, override,
  form required + bad-form, each numeric bound (p_max/max_fail<1, q>1, q=0,
  loo_max=0), unknown key, both exclusivity errors name both sides. ALL PASS.
- GPC-B (code): state()/from_state round-trip (stubbed torch; six buffers,
  dtypes, values); every wiring point guarded on pce_opts (absent =
  byte-identical, textually verified); build_geometry order (fit after geom,
  before make_chi2); save_emulator pce group guarded; the diagnostics gate; the
  banner. (docs): the pce section + equation + YAML block + two rules;
  headers 1..17 in order; Contents + pce entry; precedence rows; code map;
  YAML block; GRO-G clean; anchors resolve (both files + cross-file); house
  style. ALL PASS. py_compile clean (package + PCE + IA + drivers).
- GPC-C: workstation queue (residual + ratio runs; the fit report; save ->
  h5 pce group -> from_state rebuild matches base(theta) on a probe; the
  exclusivity errors from real YAMLs; a 2-point sweep_ntrain refit-per-point
  smoke; golden absent-pce byte-identity).

Deviations declared:
1. validate_pce takes (pce, rescale, ia) rather than pce alone, so the two
   exclusivity checks stay pure + unit-testable (GPC-A tests them); from_config
   calls it after ia is resolved and passes kwargs' rescale. The pure schema +
   the exclusivity live in the one function (spec item 1 grouped them).
2. save_emulator gained pce / pce_form params (it did not receive the chi2fn
   before); the train_single caller passes the base + form. Byte-identical for
   non-NPCE runs (both default None; the pce group is guarded).
3. Item 5 required no code (the needs_params gate already covers PCE) —
   verified, not just asserted.

Untracked notes present that the Implementer did NOT author:
save-schema-resolved-config.md (left untouched).

Awaiting Architect re-audit.

### 2026-07-08 — board verdict (Architect): GPC-C npce-training PASS
Residual and ratio forms both stage the base at load ("PCE fit:
N 25000 n_dim 12 candidates 115 ... kept 1 (loo<0.05)") and train
rc 0. Both exclusivity guards fire verbatim (pce+ia, pce+--rescale,
rc 1). The 2-point multi-GPU sweep runs both points in parallel
(N 1000 f 1.0000 / N 2000 f 0.9990) with the parent staging banner —
the run-3 lesson encoded: the parent stream owns the banner, the GPU
workers own the per-fit lines. The rebuild-vs-base probe stays named
in the remainder (:117); save-rebuild-drift's NPCE leg covers the
round-trip. Green runs 3-11.

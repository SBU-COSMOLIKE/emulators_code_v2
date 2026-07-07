---
name: weight-decay-only-weight-matrices
description: "FINDING + queued fix 2026-07-07 (Architect-verified in the tree; user-raised): make_optimizer's decay rule `p.ndim >= 2` misclassifies TWO parameter families as weight matrices — (1) the multigate / gated_power activation shape parameters w / beta / mu, shape (K, dim) (activations.py 90-92, 198-200): decay would pull w->0 (gates die), mu->0 (centers collapse), beta->0 (gates flatten), unfairly handicapping the multi-gate families in any nonzero-decay run or bake-off; (2) BinLinear's biases, shape (G, out) — decayed despite the package's own 'decaying a bias has no principled meaning' rule. LATENT: every template ships weight_decay 0.0, so no run to date was damaged. Fix direction (user: 'only decay the model parameters'): replace the shape heuristic with a module-aware allowlist — decay exactly the .weight of nn.Linear / nn.Conv1d / BinLinear, everything else undecayed (safe failure direction: an unlisted future module defaults to undecayed; the tagging alternative fails dangerous and misses BinLinear biases). SEQUENCED: discuss + spec after the relu/tanh + norm-knob unit (GAN) returns. NOT specced in full, NOT implemented."
metadata:
  node_type: memory
  type: project
---

# weight_decay must touch only true weight matrices (finding, queued)

User 2026-07-07 (while the GAN unit is in flight): "we have
activation functions with more than two parameters [dimensions] —
so we need to fix this so weight_decay does not affect them — we
need to only decay the model parameters." Verified in the tree:

## The finding

make_optimizer groups by `p.ndim >= 2` -> decay (training.py ~30-34,
and its docstring CLAIMS activation params are never decayed).
True for H / power ((dim,) vectors). FALSE for:

1. GatedActivation + GatedPowerActivation: w / beta / mu are
   (K, dim) — ndim 2 -> DECAYED. Effect under nonzero decay: w
   pulled to 0 (gates die toward a0-only), mu to 0 (centers
   collapse), beta to 0 (gates flatten) — the activation is
   dragged toward degenerate forms; a bake-off with decay > 0
   would handicap exactly the multi-gate families.
2. BinLinear biases: (G, out) — ndim 2 -> DECAYED, violating the
   package's own bias rule (Affine docstring: "decaying a bias has
   no principled meaning").

LATENT, no damage yet: all three templates ship weight_decay 0.0
(with wd = 0 the group split has no effect). Goes live the moment
anyone sets nonzero decay or tunes over it.

## Fix direction (Architect recommendation, for the discussion)

Module-aware allowlist replacing the shape heuristic: walk
named_modules(); decay = exactly the `.weight` of nn.Linear,
nn.Conv1d, and BinLinear; every other parameter undecayed. Why over
parameter tagging (p.no_decay attributes, the capability-flag
style): the allowlist's failure direction is SAFE (a future module
left off the list defaults to undecayed — the mild error), tagging
fails dangerous (a forgotten tag = decayed shape params) and would
not catch the BinLinear biases without tagging those too. The
make_optimizer docstring's claim then becomes true by construction.
Gates to spec: a per-class parameter census (every MODELS entry:
decay set == exactly the Linear/Conv1d/BinLinear weights; multigate
w/beta/mu + BinLinear biases + all norms/activations in no_decay);
wd-0 golden byte-identity; the docstring updated.

## Status

QUEUED: discuss + spec after the relu/tanh + norm-knob unit
([[activation-families-norm-knob]]) returns and lands — same file
(training.py make_optimizer), sequential units.

## Addendum (2026-07-07): README documentation rides the fix unit

User directive: "also add in the readme documentation (be
didactical with code blocks and graphs) on how to implement weight
decay only on models (in the yaml) and not on activation functions
no matter how many parameters the af has." The docs land WITH the
fix (documentation describes real behavior, never ahead of it).
The optimizer part of the README's "optimizer, lr, scheduler"
section gains the weight-decay treatment:

- The one-line math (chapter style, GRO-G-safe):
  $$w \leftarrow w - \mathrm{lr}\,\lambda\,w$$
  with the legend "$\lambda$ = `weight_decay` (AdamW's decoupled
  decay, applied beside the gradient step)".
- The split diagram (verbatim):

        optimizer.weight_decay: lambda
                     │
         decayed ◀───┴───▶ never decayed
    ┌──────────────────────┐   ┌─────────────────────────────────┐
    │ weight MATRICES only │   │ everything else:                │
    │   Linear.weight      │   │   all biases                    │
    │   Conv1d.weight      │   │   Affine / FeatureAffine g, b   │
    │   BinLinear.weight   │   │   activation parameters         │
    └──────────────────────┘   │   (H's gamma/beta; multigate's  │
                               │   w/beta/mu — ANY shape)        │
                               └─────────────────────────────────┘

  (de-cap "ANY" at build per the house rule; kept here for spec
  emphasis only.)
- The why, two sentences: decay limits function complexity by
  preferring small MATRICES; pulling biases or activation shape
  parameters toward 0 has no such meaning — it drags the
  activation toward degenerate forms (gates dying, centers
  collapsing) — so membership is decided by MODULE ROLE, never by
  tensor shape, no matter how many parameters a family carries.
- The YAML block (paste-ready):

      optimizer:
        weight_decay: 1.0e-4   # L2 pull on the weight matrices only
                               # (Linear / Conv1d / BinLinear weights);
                               # biases, Affine / FeatureAffine gains,
                               # and every activation parameter are
                               # never decayed, whatever their shape.
                               # 0 = off (the template default).

Gate addition (rides the unit's gate set): the README treatment
present with the diagram + equation + block; the diagram's decay
list byte-matches the allowlist the code implements.

## Full spec (2026-07-07, finalized for the overnight queue)

The "discussion" was had in relay: the user directed the fix ("only
decay the model parameters") and did not object to the allowlist
recommendation. SPEC:

- make_optimizer (training.py): replace the `p.ndim >= 2` grouping
  with the module-aware allowlist — walk model.named_modules();
  decay = the `.weight` Parameter of every nn.Linear, nn.Conv1d,
  and BinLinear instance; EVERY other parameter (all biases,
  Affine / FeatureAffine gain+bias, every activation parameter of
  any shape, LayerNorm affines) -> no_decay. Deduplicate by id
  (shared modules appear once). Keep the two-group structure, the
  fused kwarg, and the signature UNCHANGED. Import BinLinear beside
  the existing Affine import (no cycle: training.py already imports
  from emulator_designs_building_blocks). Docstring rewritten to
  state the allowlist (its current ndim claim becomes true by
  construction) + one line naming the safe failure direction (an
  unlisted future module defaults to undecayed).
- The README section-6 weight-decay treatment: the addendum above,
  verbatim (equation + legend, the split diagram de-capped, the
  two-sentence why, the paste-ready optimizer.weight_decay block).
- Gates:
  - GWD-A (Mac, stubbed exec): a toy module tree carrying one of
    each (Linear, Conv1d, BinLinear, Affine, FeatureAffine, an
    activation with (K, dim) params, LayerNorm); the extracted
    grouping puts EXACTLY the three .weights in decay, everything
    else in no_decay; the old rule's two misclassifications
    (multigate (K, dim), BinLinear bias (G, out)) shown fixed.
  - GWD-B (Mac, static): signature + fused + two-group structure
    unchanged; wd = 0 semantics identical (both groups inert -> the
    regrouping cannot change any existing run: every template ships
    0.0); docstring truth; the README treatment present with the
    diagram byte-matching the implemented allowlist; scans +
    py_compile.
  - GWD-C (workstation, rides the queue): a gated_power + wd 1e-4
    smoke printing the param-group census (multigate w/beta/mu +
    BinLinear biases in the undecayed group; Linear/Conv1d/
    BinLinear weights in the decayed one); golden wd-0
    byte-identity run.

## The overnight handoff (two units, one session, one combined commit)

The user sleeps; no commits can happen mid-queue. Both units run in
ONE working tree, in order, with per-unit gate evidence, and ONE
combined commit suggestion in the morning (commit-unit purity
traded for overnight throughput, user-accepted by the sleep
directive). Unit 1 = the README didactic pass
([[readme-run-it-definitions]] in full). Unit 2 = this note's spec
(the fix + the README section-6 treatment). Disjoint files except
README (different regions).

### ARCHITECT_HANDOFF
Task: the OVERNIGHT queue — two units, one session, in order, one
combined diff. Base: the GAN commit (relu/tanh + model.norm);
`git log -1` must show it — else STOP and wait.
UNIT 1 — the README didactic pass: notes/readme-run-it-definitions.md
in full (the Run-it split with verbatim prose, the
workstation-assumption sweep over the six user-facing sites, the
section-2 vocabulary box, the ten local one-liners). Doc-only
except the four driver-header comment lines (AST identity holds).
Gates GRI-A + GRI-B.
UNIT 2 — the weight-decay fix + its README treatment:
notes/weight-decay-only-weight-matrices.md in full (the
module-aware allowlist in make_optimizer; the section-6 equation +
diagram + why + YAML block, diagram byte-matching the allowlist).
Gates GWD-A + GWD-B (GWD-C embedded for the workstation queue).
Execute 1 then 2 in the same tree (README regions are disjoint —
if they collide, STOP and report). Report ONE
IMPLEMENTER_HANDOFF: per-unit resume states appended to each spec
note, per-unit raw gate evidence, deviations declared per unit,
and ONE suggested combined commit command (sentence covering
both). Do not commit. If any gate fails or a spec contradicts the
tree, STOP at the failing unit and report what stands.
### END

## Implementer resume state (2026-07-07, Opus, UNIT 2 of the overnight queue, base 8ad25a1)

IMPLEMENTED, uncommitted (rides UNIT 1 into one combined commit).

- make_optimizer (training.py): the `p.ndim >= 2` heuristic replaced by the
  module-aware allowlist — walk model.modules(), collect the `.weight` (by id)
  of every nn.Linear / nn.Conv1d / BinLinear, then split named_parameters() on
  membership. Everything else (all biases, Affine / FeatureAffine gains, every
  activation parameter of any shape, LayerNorm affines, BinLinear's (G, out)
  bias) is undecayed. Signature / fused / two-group structure UNCHANGED; the
  set + id-membership dedupes (named_parameters yields each shared param once).
  BinLinear imported beside Affine. Docstring rewritten to the allowlist + the
  safe-failure-direction line. (Confirmed in the tree: BinLinear builds its
  internal nn.Linear layers as a LOCAL list it discards after stacking, so
  model.modules() never yields them — no risk of decaying unused weights.)
- README section 6: the optimizer bullet's weight_decay clause rewritten to
  the allowlist; a weight-decay treatment added after the section YAML block —
  the decoupled-decay equation ($$w <- w - lr*lambda*w$$) + legend, the split
  diagram (decayed = Linear/Conv1d/BinLinear .weight; everything else never
  decayed), the two-sentence why, and the paste-ready weight_decay: 1.0e-4
  block. The diagram's decay list byte-matches the code allowlist (GWD-B).

Gates GWD-A + GWD-B: ALL PASS. GWD-A (stubbed exec of the real make_optimizer
on a toy tree with one each of Linear/Conv1d/BinLinear/Affine/FeatureAffine/a
(K,dim) activation/LayerNorm): decay == exactly the three .weights; the two old
misclassifications (multigate (K,dim) w/beta/mu, BinLinear (G,out) bias) now
undecayed; wd=0 leaves both groups inert (so no template run changes). GWD-B
(static): signature/fused/two-group unchanged; ndim heuristic gone; docstring
truth; the README treatment present with the diagram byte-matching the
allowlist; py_compile clean. GWD-C: workstation queue (gated_power + wd 1e-4
census; golden wd-0 byte-identity).

Deviations declared:
1. The addendum diagram's emphasis caps MATRICES / ANY (and the why's MATRICES
   / MODULE ROLE) de-capped per the standing house rule (identical widths, so
   the ASCII alignment holds); the note itself flagged "de-cap ANY at build".
2. Section-6 overlap with UNIT 1: the Architect's "disjoint README regions"
   assumption is section-level optimistic — UNIT 1 fix #7 (the lr-rule why +
   the "fused on CUDA" gloss) and this unit (the weight_decay clause + the
   treatment) edit NON-overlapping spans of the same optimizer bullet, so both
   applied cleanly and coherently. No true content collision, so I did NOT
   invoke the collision-STOP; flagged for the re-audit.

Awaiting Architect re-audit.

### 2026-07-07 — Architect re-audit (morning pass): BOTH OVERNIGHT
### UNITS ACCEPTED, no deltas

Own probes on the confirmed base 8ad25a1:

- UNIT 2 (this note): my own census harness ran the REAL extracted
  make_optimizer on a toy tree carrying one of everything — 6/6:
  decay == exactly the three .weights (Linear / Conv1d / BinLinear);
  the multigate (3,128) w/beta/mu and the BinLinear (G,out) bias —
  the two old misclassifications — now UNDECAYED; all biases /
  Affine / FeatureAffine / LayerNorm undecayed; two groups with wd
  on group 0 only; parameter count conserved. AST: training.py the
  only code change; the diff is the spec verbatim (BinLinear import,
  decay_ids by id over modules(), docstring truth + the
  safe-failure sentence). The Implementer's BinLinear-locals finding
  (its temp nn.Linear layers are discarded, never in modules())
  matches the constructor I read — no phantom decay targets.
- UNIT 1 ([[readme-run-it-definitions]]): the 16 command lines
  byte-identical HEAD vs tree; workstation mentions in user-facing
  docs = ZERO; the vocabulary box carries all six entries ("Six
  terms" header); the four Run-it terms defined; the generalized
  Cocoa intro present; the metric spellings tied; anchors ALL
  RESOLVE; math scan clean; the four drivers AST-IDENTICAL;
  py_compile clean.

Deviations RULED, all ACCEPTED:
(a) The section-6 adjacency call was CORRECT — the STOP clause
    targeted content collision; non-overlapping spans of one bullet
    is adjacency, and my probe shows both edits present and
    coherent. Good judgment under an ambiguous instruction.
(b) The allowlist verified independently (above).
(c) Fix #9c's relocation to the code map: the spec-vs-tree mismatch
    was the ARCHITECT'S (my spec pointed at section 10 for a phrase
    the reorg had moved); applying the gloss where the phrase lives
    was right.
(d) "Five terms" -> "Six terms": the ARCHITECT'S arithmetic slip
    (six bullets under a five header); fix correct.
(e) Backticks + de-caps: the standing rules, correctly applied to
    my spec prose again.

COMBINED COMMIT-READY. Suggested sentence: "Overnight docs + fix:
the README didactic pass (Run-it split + definitions, the
vocabulary box, ten local glosses, the workstation-assumption
sweep) + weight decay by module role (decay exactly the
Linear/Conv1d/BinLinear weights — activation parameters and
BinLinear biases never decayed; README section-6 treatment; gates
GRI-A/B + GWD-A/B Architect-verified)". GWD-C joins the workstation
board.

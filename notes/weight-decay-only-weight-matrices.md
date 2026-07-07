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

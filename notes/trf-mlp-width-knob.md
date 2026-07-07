---
name: trf-mlp-width-knob
description: "SHELVED 2026-07-06 by user ruling — the pin is 'a fair rule'; the knob is NOT to be implemented, the explicit-pin documentation moved to audit-docs-2026-07-06 as D-DOC9; spec retained, do not re-propose unprompted. Original spec: model.trf.mlp_width — the interior width of the TRF per-token MLP stack. Today TRFBlock builds n_mlp_blocks layers each pinned dim->dim (dim = token width = padded bin length 26): the no-width-knob design pinned the TOKEN width to the physical bin length (correct, no adapters) and the MLP interior silently inherited it, so the textbook FFN expansion (dim -> 4 dim -> dim) is unreachable. New optional model.trf.mlp_width = w: ladder dim->w, w->w ..., w->dim with the LAST layer zero-init as today (identity-at-init survives, act(0)=0); absent = today exactly; requires n_mlp_blocks >= 2 when set (a single layer has no interior; loud error). Pure mlp_layer_widths helper carries the ladder + the conflict check (Mac-testable, no torch); TRFBlock consumes it; threading = MODEL_BLOCK_KEYS trf table + mlp_width=None on ResTRF/TemplateResTRF/TRFBlock. Attention untouched (n_heads divides the TOKEN width 26, not w); FiLM untouched; banner free via describe_spec; sweeps/tunes as an ordinary numeric leaf. Gates GTW-A..C + handoff embedded. Sequencing: doc-audit -> head-activation -> this (same files); one Implementer session, separate commit units. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# model.trf.mlp_width: the TRF per-token MLP interior width (spec)

User question: `n_mlp_blocks` sets the depth of the per-token MLP
inside the TRF block — where is the width? Answer: there is none.

## Today's structure (verified at 8bb5484)

TRFBlock (emulator_designs_building_blocks.py:517-529) builds the
MLP branch as n_mlp_blocks layers, EACH `dim -> dim`, each with its
own activation instance; the last layer is zero-initialized (with
wo) so the block is exactly the identity at init. `dim` is the token
width = max_bin = the padded bin length (26 on LSST-Y1).

The "tokens at their natural width, no width knob" design decision
(correct: no embedding in, no projection out, no adapter parameters)
pinned the TOKEN width to the physical bin length — and the MLP
interior silently inherited that pin. Consequence: the textbook
transformer FFN expansion (dim -> 4 dim -> dim) is unreachable; the
per-token MLPs can only be deep and narrow (26 wide), never wide.

## Design

New optional key, absent = today exactly:

```yaml
    trf:
      n_heads:      1       # must divide the bin length (26 -> 1 | 2 | 13)
      n_blocks:     1
      n_mlp_blocks: 2       # depth of each token's private MLP stack
      mlp_width:    104     # optional: the stack's INTERIOR width
                            # (the textbook FFN is 4 x the token
                            # width: 4 x 26 = 104). Ladder: 26 ->
                            # 104 -> 26, last layer zero-init as
                            # today. Needs n_mlp_blocks >= 2 (one
                            # layer has no interior). Absent = every
                            # layer at the token width (26 -> 26).
      shared_mlp:   false   # true = ONE MLP for every token
      gate_init:    1
```

Shape ladder with mlp_width = w and n_mlp_blocks = n:

    layer 1      dim -> w      act(w)
    layers 2..n-1  w -> w      act(w)
    layer n        w -> dim    act(dim), layer zero-init (identity)

- w = dim (or key absent) reproduces today's shapes exactly.
- n = 1 with mlp_width set: ERROR at construction, loud, naming both
  keys (a single dim -> dim layer has no interior; the branch output
  must stay dim for the residual add).
- Identity-at-init survives any w: the LAST layer is the zeroed one
  and every activation family maps 0 -> 0 (same argument as the head
  handoff continuity).
- Attention is UNAFFECTED: n_heads divides the TOKEN width (26), not
  w; the n_heads = 1 | 2 | 13 rule is unchanged. FiLM (per-token
  affine at dim, outside the MLP interior) unaffected. BinLinear and
  the shared nn.Linear both already take (in_features, out_features),
  so the per-token uniqueness and the shared_mlp ablation both carry
  w with no new machinery.
- Parameter cost (docs must say it): per token per block, today
  n (dim^2 + dim); with w, dim w + (n-2) w^2 + w dim + biases. At
  n = 2, w = 104: ~4x today's MLP parameters.

## Implementation (small; the pure-helper pattern)

1. emulator_designs_building_blocks.py: a PURE helper
   `mlp_layer_widths(dim, n_mlp_blocks, mlp_width)` returning the
   [(in, out), ...] ladder and raising the n_mlp_blocks >= 2 conflict
   (ValueError naming model.trf.mlp_width and n_mlp_blocks) — pure so
   the Mac gates exercise every case without torch. TRFBlock gains
   `mlp_width=None` and loops over the helper's ladder; the act after
   each layer is act(out_width of that layer); zero-init stays
   `mlp_lins[-1]`. Docstring: Arguments entry + the shape legend
   gains w (house style: every symbol in a legend).
2. emulator_designs.py ResTRF + IA/emulator_designs.py TemplateResTRF:
   `mlp_width=None` constructor kwarg, forwarded to every TRFBlock.
   Docstrings + the head-knob listings.
3. experiment.py MODEL_BLOCK_KEYS trf table: `"mlp_width":
   "mlp_width"` (a plain rename — unlike the activation key it needs
   no value translation).
4. Banner: FREE (describe_spec prints the active trf: dict verbatim);
   gate asserts.
5. Sweeps / tune: an ordinary numeric leaf (model.trf.mlp_width
   sweepable; [default, min, max, int] searchable) — no driver work.
   Doc note: when searching mlp_width, keep the n_mlp_blocks range
   >= 2 (a trial pairing mlp_width with n_mlp_blocks = 1 dies on the
   constructor error).

## Docs owed with the feature

train_single YAML trf: block (the commented mlp_width lines above,
paste-ready); tune YAML one range example; README section 6 trf knob
list + section 10 TRFBlock and ResTRF lines; train_single driver
header trf enumeration; experiment __init__ docstring trf entry.

## Rulings (veto points)

- (a) Name: mlp_width (pairs with n_mlp_blocks; model.trf.width would
  read as a token-width knob, which deliberately does not exist).
- (b) Absolute width, not a ratio (house style: explicit numbers;
  mirrors model.mlp.width).
- (c) The conflict (mlp_width + n_mlp_blocks = 1) is a loud error,
  not a silent ignore.

## Gates

- GTW-A (Mac, pure): mlp_layer_widths exhaustive — None/absent ->
  all (dim, dim); w = dim n >= 2 == the None ladder; n = 2/3/5 with
  w != dim exact ladders (first in = dim, last out = dim, interior
  all w); n = 1 + w set raises naming both keys; n = 1 + None fine;
  w <= 0 / non-int rejected.
- GTW-B (Mac, static): TRFBlock consumes the helper (no inline
  ladder); act widths follow layer outputs; zero-init still
  mlp_lins[-1]; mlp_width threaded ResTRF + TemplateResTRF -> every
  TRFBlock; MODEL_BLOCK_KEYS trf entry; describe_spec untouched;
  docstring legends carry w; house scans + py_compile.
- GTW-C (workstation, rides the queue): restrf + mlp_width 104 run —
  banner shows it, param count matches the formula, epoch-1 loss
  equals the trunk-only loss (identity at init with w != dim);
  golden no-key byte-identity run.

## Handoff

### ARCHITECT_HANDOFF
Task: model.trf.mlp_width (spec: notes/trf-mlp-width-knob.md; read
it + [[links]], especially [[banner-prints-consumed-view]]).
Base / sequencing: land AFTER the doc-audit fixes
([[audit-docs-2026-07-06]]) and the head-activation feature
([[head-activation-per-component]]) — all three touch the model.trf
schema docs and MODEL_BLOCK_KEYS; this one is its own commit unit.
Scope: the pure mlp_layer_widths helper + TRFBlock mlp_width=None
consuming it (zero-init stays mlp_lins[-1]; act(out) per layer);
ResTRF + TemplateResTRF threading; MODEL_BLOCK_KEYS trf entry
(plain rename); docs per the note's list (YAML block paste-ready in
the note). Rulings (a)/(b)/(c) binding unless vetoed in relay.
Gates: GTW-A/B on the Mac (recipes above); GTW-C embedded for the
workstation queue. Report: raw gate outputs, every YAML change as a
paste-ready block, deviations declared. Do not commit: leave the
diff uncommitted and print the suggested commit command.
### END

## Status

SHELVED 2026-07-06 by user ruling: "every MLP layer inside the TRF
block is pinned to the token width (26) — this is a fair rule — but
make sure it is explicit on documentation." The knob is NOT to be
implemented; the explicit-pin documentation duty moved into the
doc-audit handoff as D-DOC9 ([[audit-docs-2026-07-06]]). The spec
above is retained intact in case the ruling is ever revisited (the
design and gates would apply unchanged); do not re-propose it
unprompted. The handoff block above is VOID while shelved.
Sequencing of the remaining active work: doc-audit (now incl.
D-DOC9) -> head-activation.

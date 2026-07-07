---
name: readme-yaml-chapter
description: "SPEC 2026-07-07 (Architect): the README YAML chapter — user directive: a didactic, non-verbose block-by-block YAML reference ('quickly and easily remember the math, the options and what the YAML can do'), each YAML piece promoted to a FULL ## section (user's correction), with equations (LaTeX $$ per the activation-appendix precedent), small YAML examples, and ASCII diagrams. Order (user's): data first, training globals second, then loss / lr / trim / focus / ema / model (mlp-activation-cnn-trf as ### subsections) / two-phase + phase blocks. New sections 7..15; appendices renumber; STANDING RULE recorded: 'Appendix: every file's functions' is ALWAYS the last appendix — and it owes an update (the unit-B helpers have no lines). Dedup discipline: the chapter owns meaning + math, the precedence appendix owns collision tables — point, never restate; section 6's YAML tour shrinks to a pointer. Gates GYC-A..E. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# README: the YAML chapter, block by block (spec)

User directive 2026-07-07: a README section on the YAML file where
each piece is documented "didactical with graphs and small code
block examples but not overly verbose. Idea is we can quickly and
easy remember the math, the options and what can the YAML do."
Correction applied: each piece is a FULL numbered `##` section, not
a subsection. Order given by the user: data FIRST, the train_args
globals SECOND, then the blocks.

## Placement and numbering

Insert after `## 6. Run it`. New sections 7..15; the appendices
renumber to 16..20. STANDING RULE (user directive, record + honor in
every future README edit): `Appendix: every file's functions` is
ALWAYS the last appendix. Contents + anchors renumbered (the unit-C
slug checker applies; mind the em-dash double-hyphen slugs).

Section 6's YAML-tour paragraph (currently ~30 lines describing the
blocks inline) shrinks to two sentences + a pointer into the new
chapter ("The YAML has two blocks, data and train_args; sections
7-15 document every block; templates live in example_yamls/"). The
`sweep:` sub-section stays in Run it (driver-specific); section 7
points to it.

## Global style rules (bind every section)

- LaTeX math in `$$...$$` (the activation-appendix precedent).
- One small YAML example per section, values matching the
  example_yamls templates where they overlap (no divergent numbers).
- ASCII diagrams in the house shape-flow style, used where they earn
  their lines; the shared-schedule plot appears ONCE (in trim) and is
  cross-referenced by focus / ema / the berhu anneal.
- DEDUP: the chapter owns meaning + math; the precedence appendix
  owns collision rules; the activation appendix owns the activation
  math; the chi2 appendix owns Mahalanobis. Point, never restate a
  table that exists elsewhere.
- Budget: each section ~25-45 rendered lines; the whole chapter
  ~350-450. "Not overly verbose (this is a readme!)".
- Every equation copied VERBATIM from the code/docstrings it
  documents (gate GYC-A checks each against the tree).

## The sections

### 7. The YAML file (orientation)

Two top-level blocks (data / train_args); one compact full example
(~35 lines, production shape: restrf + nla, two-phase, berhu head);
the search-range convention in one line (any numeric leaf may be
`[default, min, max, kind]`; the train drivers use default, tune
searches); pointers: templates in example_yamls/, collision rules in
the precedence appendix, the sweep: block in Run it.

### 8. data

The five files (dv/params/covmat train + dv/params val, bare names
under --root/chains) + the cosmolike pair (under
external_modules/data); n_train / n_val (ABSOLUTE rows, enforced
AFTER param_cuts, pool-too-small raises); split_seed; ram_frac. A
mini staging diagram (dump -> seeded shuffle -> param_cuts ->
first-n_train -> RAM-or-memmap). Sub-heading `### param_cuts`: the
four windows as a table (key pair, quantity, formula, Planck
anchor):
omegabh2 = Omega_b (H0/100)^2 (0.0224), omegam2h2 = (Omega_m
H0/100)^2 (0.045), omegamh2 = Omega_m (H0/100)^2 (0.143),
omegamh2ns = omegamh2 * n_s (0.138, needs the ns column); omit a
key = no cut on that side; lo >= hi raises. YAML example = the
user's production block.

### 9. Training globals

nepochs; bs (+ the derived ~1024 eval batch, independent of bs, one
line); trunk_epochs + freeze_trunk (POINT to the precedence
appendix C2 schedule-modes table; one sentence each here); silent;
clip (per-step grad-norm ceiling, direction kept:
$$g \leftarrow g \cdot \min(1, \mathrm{clip}/\lVert g \rVert)$$);
rewind (reload best weights + optimizer on every plateau lr cut,
bounds an excursion to `patience` epochs). YAML example = the
user's globals block.

### 10. loss

The math section. Mode table — columns: mode | $L(c)$ | per-sample
gradient vote | use it when:

$$L_{\mathrm{chi2}} = c \qquad L_{\mathrm{sqrt}} = \sqrt{c} \qquad
L_{\mathrm{sqrt\_dchi2}} = \sqrt{1+2c} - 1$$

$$L_{\mathrm{berhu}}(c) = \begin{cases} \sqrt{c} & c \le k \\
\dfrac{c+k}{2\sqrt{k}} & c > k \end{cases} \qquad
L_{\mathrm{berhu\_capped}}\ \mathrm{adds} \quad
\dfrac{2\sqrt{Kc} + k - K}{2\sqrt{k}} \quad (c > K)$$

C1 at every knot; k = berhu.knot (default 0.2, the frac>0.2 goal),
K = berhu.cap (default 10). Vote intuition one line each: sqrt =
every sample an equal vote; chi2 = vote grows with c (tail-chasing);
berhu = equal votes in the bulk, rising ~x7 across (k, K),
berhu_capped plateaus above K (monster-robust). One line: this is
textbook BerHu in the whitened residual norm with
$\delta = \sqrt{k}$, knots in chi2 units. The berhu: sub-block
(knot / cap; family spelling or mode-named, both = error -> pointer
to precedence D) and anneal: (presence = on):
$$L_s = (1-s)\sqrt{c} + s\,L_{\mathrm{mode}}(c)$$
with s the shared schedule (pointer to trim's plot). YAML example =
the head berhu_capped block.

### 11. optimizer, lr, scheduler (one section: the optimization stack)

(Architect grouping, veto-able — the user listed lr standalone; these
three interact as one stack and are small.) optimizer: AdamW,
weight_decay decays weight matrices only (ndim >= 2), fused on CUDA.
lr:
$$\mathrm{lr} = \mathrm{lr\_base} \cdot
\sqrt{\mathrm{bs}/\mathrm{bs\_base}}$$
(the sqrt-noise rule; bs_base is the run-global anchor — never in a
phase block), + warmup_epochs of linear ramp. scheduler:
ReduceLROnPlateau {mode, patience, factor}, stepped on the RAW val
median every epoch (EMA never feeds it); per-phase scheduler = kwargs
replacement (pointer to precedence B). YAML example = the user's lr
+ scheduler blocks.

### 12. trim

Drop the worst trim(e) fraction of each batch before the mean (hard
reject; eval never trims). trim(e) from the shared schedule
anneal_value — THE schedule plot lives here:

    value
    start ────────╮
                  │╲   shape: cosine | linear | step | const
                  │ ╲
    end ──────────┴──╲─────────
          hold        anneal      (epochs)

hold start for hold_epochs, ramp to end over anneal_epochs, stay.
const = start forever (end / holds ignored). Advice: keep end > 0
(a floor keeps the worst fraction out forever). YAML example = the
user's trim block. Cross-refs: focus, ema.anneal, loss.berhu.anneal
reuse this schedule with their own start/end.

### 13. focus

Focal hardness weighting:
$$w_i = \left(\frac{c_i}{c_i + \kappa}\right)^{\gamma(e)}$$
(detached — a sample cannot shrink its own weight); gamma(e) runs
the shared schedule (0 = plain mean early, end = 2 typical); kappa
= the chi2 where hardness crosses 1/2 (fixed over the run). One
line on the berhu interplay (berhu carries the tail role; soften
focus on a berhu head). YAML example = the user's focus block.

### 14. ema

Polyak weight average, updated after every optimizer step:
$$\bar{\theta} \leftarrow \beta\,\bar{\theta} + (1-\beta)\,\theta
\qquad \beta = 1 - \frac{1}{\mathrm{horizon\_epochs}\cdot
\mathrm{steps\_per\_epoch}}$$
(bs-invariant: the horizon is in epochs). Selection + printed
metrics on $\bar{\theta}$; the scheduler stays on the raw median;
{theta, optimizer, theta_bar} snapshot/rewind as ONE unit. anneal:
= the shared schedule on the horizon, h(e) = horizon * s(e) (no
memory of the terrible early era). Per-phase ema: full replacement;
ema: null = per-phase off. YAML example: horizon_epochs 3 + anneal.

### 15. model

The class grid (2 axes pick 1 of 6):

| | plain | ia: nla | ia: tatt |
|---|---|---|---|
| resmlp | ResMLP | TemplateMLP | TemplateMLP |
| rescnn | ResCNN | TemplateResCNN | TemplateResCNN |
| restrf | ResTRF | TemplateResTRF | TemplateResTRF |

The shared trunk -> gated head correction diagram (once):

    params ─▶ ResMLP trunk ─▶ y (full-whitened)
                              │  basis change to theta order
                              ▼
                        head blocks (cnn conv | trf attention)
                              │
              y + gate · correction ─▶ whitened dv

IA factoring one equation (nla):
$$\xi = K_0 + A_1 K_1 + A_1^2 K_2$$
(amplitudes never enter the network; tatt = 10 templates, 3 amps).
Then `###` subsections: mlp (width, n_blocks — required, every
architecture is built on it); activation ({type, n_gates} or bare
string; families -> POINT to the activation appendix; the per-head
pin + license + head: alias in two sentences -> POINT to precedence
A); cnn (knob table: kernel_size, rescale_kernel, groups, separable,
film, n_blocks, gate_init, activation); trf (n_heads must divide the
token width 26 -> 1|2|13; n_blocks; n_mlp_blocks — depth only, every
MLP layer at the token width, no width knob; shared_mlp; film;
gate_init; activation). compile_mode one line (defaults ->
precedence F).

### 16. Two-phase schedule + the trunk: / head: blocks

The phase timeline diagram (trunk_epochs, restore-best, freeze or
joint, head pass); phase blocks are DIFFS against the top level;
the eight keys + per-key semantics -> POINT to precedence B (no
second table); head: activation: alias + trunk: error one sentence
-> precedence A; single-phase demotion two sentences ("what is in
the trunk is just the global"). YAML example: the trunk:/head:
production shape (lr + scheduler + loss per phase).

## Appendices after the chapter

17 AI-Usage, 18 chi2 (Mahalanobis), 19 activation functions, 20
precedence, 21 every file's functions — LAST (standing rule).
Appendix 21 UPDATE owed in the same unit: diff every module's
def/class list against its appendix bullets and add what is missing
(known: the four unit-B experiment.py helpers _head_activation_spec
/ _resolve_head_activation / _activation_flag_notice /
_pinned_head_warning have no lines; check the rest mechanically).

## Gates

- GYC-A (math-vs-tree): every equation checked against the code it
  documents — the berhu ladder coefficients vs CosmolikeChi2._reduce
  (exact, incl. (c+k)/(2 sqrt k) and the capped a sqrt(c) + b form),
  the lr sqrt rule vs build/run wiring, beta vs derive_ema_beta, the
  focal weight vs _reduce, the schedule shapes vs anneal_value, clip
  vs the loop's grad-norm rescale, the nla polynomial vs nla_coeffs.
- GYC-B (structure): sections 7..16 present in the user's order;
  appendices 17..21 with every-file's-functions LAST; Contents +
  anchors resolve (slug checker; em-dash slugs).
- GYC-C (dedup): the precedence tables appear ONCE in the README
  (appendix only — the chapter points); section 6's tour is the
  two-sentence pointer; example values byte-match example_yamls
  where they overlap.
- GYC-D (appendix completeness): per touched module, every public
  def/class has an appendix bullet; the four helpers present.
- GYC-E: documentation-only (git diff = README + notes; zero code /
  yaml); house scans (README one-line rows exempt from 90 cols; no
  double-hyphen em-dashes; caps = acronyms); py_compile regardless.

## Handoff

### ARCHITECT_HANDOFF
Task: the README YAML chapter (spec: notes/readme-yaml-chapter.md,
read in full + the [[links]] it cites). Base: the unit-C commit
(README precedence appendix); `git log -1` must show it — else STOP.
Scope: sections 7..16 exactly as specced (order: data, globals,
loss, optimization stack, trim, focus, ema, model with the four ###
subsections, two-phase + phase blocks); section 6 tour shrunk to
the pointer; appendices renumbered 17..21 with every-file's-
functions LAST (standing rule) and its completeness update; global
style rules binding (math verbatim from the tree, one example per
section matching example_yamls values, the shared-schedule plot
once, dedup by pointers, the length budget). DOCUMENTATION-ONLY:
zero .py / .yaml changes (a reason to touch code = STOP and
report). The grouping of optimizer+lr+scheduler into one section is
Architect-recommended and user-veto-able — flag it in your report.
Gates GYC-A..E. Report: IMPLEMENTER_HANDOFF + resume state appended
to this note, raw gate evidence (incl. the math-vs-tree check per
equation), deviations declared. Do not commit: print the suggested
commit command.
### END

## Status

SPEC DELIVERED 2026-07-07, NOT implemented. One doc-only commit
unit. Suggested commit sentence: "README YAML chapter: sections
7-16 document every YAML block (data, globals, loss math, the
optimization stack, trim/focus/ema schedules, the model grid,
two-phase + phase blocks) with equations + examples; appendices
renumbered, every-file's-functions kept last + completed (gate
GYC-A..E Architect-verified)".

## Architect re-audit 2026-07-07: ACCEPTED, no deltas

Own checks on the confirmed base ddb6c98; the chapter READ IN FULL
(sections 7-16), not just grepped:

- Doc-only proven (0 .py / .yaml diff lines). GYC-A: the equations
  re-verified against the code myself — the chi2/sqrt/sqrt_dchi2
  branches, the berhu ladder (c+k)/(2 sqrt k) and the capped region
  (2 sqrt(Kc)+k-K)/(2 sqrt k) verbatim at loss_functions 344-388,
  the blend (1-s) sqrt c + s v (both guarded sites), the focal
  weight + detach, derive_ema_beta 1 - 1/denom, the foreach_lerp_
  form (== beta theta_bar + (1-beta) theta), the lr rule
  (bs/bs_base)**0.5 at 2188, clip_grad_norm_ at 1699. I also
  re-derived the vote column: with vote = sqrt(c) dL/dc, sqrt = 1/2
  everywhere, berhu plateaus at sqrt(K/k)/2 (x7.07 at defaults ->
  the "~x7" is exact), sqrt_dchi2 -> 1/sqrt(2) asymptotically and
  -> 0 at the origin ("equal, softer near 0" is right).
- GYC-B: headers 7..16 in the user's order, appendices 17..21 with
  every-file's-functions last; my slug checker: 61 links, none
  missing. GYC-C: zero "trunk gets" in the chapter (no restated
  precedence table); section 6 carries the pointer; numeric example
  values match the template. GYC-D: the four unit-B helpers present
  in appendix 21. GYC-E: scans + py_compile clean.
- Deviations RULED: the restrf+nla categorical examples ACCEPTED
  (the spec itself mandated the production shape; numeric values
  match the template); block-style YAML is the house rule, not a
  deviation.
- OPEN for the USER at commit time: the optimizer+lr+scheduler
  one-section grouping (Architect-recommended, implemented; a split
  is a small follow-up if vetoed).

COMMIT-READY (sentence above). Next: the user's commit unblocks the
HOLD on [[readme-reorg-two-readmes]]; pending user decision there:
whether the reorg also deletes the dead emulator/parallel/ folder
(Architect-recommended; awaiting the user's word).

## Resume state (Implementer appends below)

### 2026-07-07 — Implementer (Opus 4.8): the YAML chapter executed

Base: the unit-C precedence-appendix commit ddb6c98; started there.
DOCUMENTATION-ONLY — git diff touches only README.md (+ notes); every
.py / .yaml byte-identical (GYC-E confirmed). One doc-only commit unit.

**Done:**

- New README sections 7–16, in the user's order: 7 The YAML file
  (orientation + one compact production example), 8 data (+ the staging
  diagram + the ### param_cuts window table), 9 Training globals
  (nepochs / bs + derived eval bs / trunk_epochs+freeze_trunk pointer /
  silent / clip eq / rewind), 10 loss (the mode table + the chi2 / sqrt /
  sqrt_dchi2 / berhu / berhu_capped equations + the anneal blend), 11
  optimizer+lr+scheduler (the optimization stack — see the flag below),
  12 trim (THE shared-schedule ASCII plot, cross-referenced by focus /
  ema / berhu.anneal), 13 focus (the focal-weight eq), 14 ema (the Polyak
  + beta eqs), 15 model (the 2x3 class grid + the trunk->head diagram +
  the nla polynomial + ### mlp / activation / cnn / trf), 16 two-phase +
  phase blocks (the phase timeline; per-key semantics point to precedence
  B, not a second table).
- Section 6's ~40-line YAML tour shrunk to a 10-line pointer (two blocks;
  sections 7–16; precedence appendix; templates; the sweep: block).
- Appendices renumbered 17 AI-Usage / 18 chi2 / 19 activation / 20
  precedence / 21 every-file's-functions (LAST, standing rule). Contents +
  anchors renumbered; the two body chi2 cross-refs updated to #18.
- Appendix 21 completeness: the four unit-B experiment.py helpers
  (_head_activation_spec / _resolve_head_activation /
  _activation_flag_notice / _pinned_head_warning) now have a bullet.
  Mechanically checked training.py (all public defs present, incl.
  build_run_specs / audit_devices / training_loop_batched — listed with
  their arg signatures) and experiment.py; the migration-message builders
  stay omitted (existing convention, no *_migration_message in any
  appendix).

**Architect grouping flag (per the handoff):** sections 11 groups
optimizer + lr + scheduler into ONE section (Architect-recommended; the
user listed lr standalone). Small, and the three interact as one stack.
USER-VETO-ABLE — split into three sections on request.

**GYC-A (math vs tree), per equation — each verified against the source:**

- clip $g \leftarrow g\cdot\min(1,\mathrm{clip}/\lVert g\rVert)$ ==
  nn.utils.clip_grad_norm_(max_norm=clip) (training.py:1699), docstring
  "rescaled by clip/norm, same direction".
- $L_{chi2}=c$, $L_{sqrt}=\sqrt c$, $L_{sqrt\_dchi2}=\sqrt{1+2c}-1$ ==
  loss_functions.py:344-349 (v = c / sqrt(c) / sqrt(1+2c)-1).
- berhu $\sqrt c$ ($c\le k$), $(c+k)/(2\sqrt k)$ ($c>k$) ==
  loss_functions.py:360-361; capped region 3
  $(2\sqrt{Kc}+k-K)/(2\sqrt k)$ == :382-383. C1 comments at :353-354,
  :368-369.
- anneal blend $L_s=(1-s)\sqrt c + s L_{mode}(c)$ ==
  loss_functions.py:365 / :387.
- focal weight $w_i=(c_i/(c_i+\kappa))^{\gamma(e)}$, detached ==
  loss_functions.py:402,406 (h=(c/(c+focus_scale)).detach(); w=keep*h**gamma;
  kappa == focus_scale).
- schedule shapes const / linear / cosine / step == anneal_value
  (loss_functions.py:52-78).
- lr $=\mathrm{lr\_base}\cdot\sqrt{bs/bs\_base}$ == training.py:2188 /
  :2371 (lr_base * (bs/bs_base)**0.5), docstring :1981.
- ema $\beta=1-1/(\mathrm{horizon\_epochs}\cdot\mathrm{steps\_per\_epoch})$
  == derive_ema_beta (training.py:1035); update
  $\bar\theta\leftarrow\beta\bar\theta+(1-\beta)\theta$ ==
  torch._foreach_lerp_(theta_bar, params, 1-beta) (training.py:1709).
- nla $\xi=K_0+A_1K_1+A_1^2K_2$ == nla_coeffs -> [1, A1, A1^2]
  (IA/loss_functions.py:37), docstring "xi = GG + A1_1*GI + A1_1^2*II".

**GYC-B/C/D/E (raw):**

- GYC-B: `## ` headers run 7..16 (chapter, user order) then 17..21
  (appendices, every-file's-functions last); the slug checker resolves
  ALL Contents anchors (header-slug + explicit <a name>), incl. the
  em-dash double-hyphen slugs (#16-two-phase-schedule--the-trunk--head-blocks,
  #20-appendix-precedence--who-wins-when-settings-collide).
- GYC-C: the chapter restates NO precedence table (0 "trunk gets" / 0 a
  real "override semantics" table — the lone hit is the pointer phrase
  "the per-key override semantics in precedence B"; 8 pointers to
  #20-appendix-precedence); section 6 is the two-sentence pointer; every
  example key/value the chapter uses is present in the train_single
  template (lr_base 0.0025, bs_base 64.0, warmup 10, patience 25, factor
  0.8, trim 0.1/0.025/50/300, focus 0/2.0/50/300/0.15, ema 3, mlp 128/4,
  n_train 25000, head lr_base 0.001 / patience 10, berhu 0.2/10).
- GYC-D: the four helpers present; training.py + experiment.py public
  defs all covered.
- GYC-E: git diff = README + notes only (0 .py / .yaml); 0 ` -- `; added
  caps are acronyms / code tokens only (YAML / RAM / MLP / CUDA / AI / IA /
  EMA / $ROOTDIR); README one-line rows exempt from 90-col; py_compile
  clean.

**Deviations declared:**

- The orientation (7) and model (15) YAML examples use `name: restrf` +
  `ia: nla` (the spec's "production shape"), where the train_single
  template's uncommented default is `name: resmlp`. This is a categorical
  choice (the template documents restrf / nla as options), NOT a divergent
  number — every numeric leaf matches the template.
- YAML examples are block style (house rule: never inline `{...}` flow),
  so a couple run a few lines longer than a flow form would.

Open: the Architect re-audit of GYC-A..E (+ a ruling on the
optimizer/lr/scheduler grouping). No workstation leg (docs only).

## Addendum 2026-07-07: the TATT polynomial (user directive)

The model section's IA paragraph shows only the nla polynomial and
waves at tatt ("10 templates, 3 amplitudes"). User: "needs to also
show the formula on TATT." Micro-unit, doc-only, one insertion in
the main README's model section (post-reorg numbering), replacing
the "(tatt = 10 templates, 3 amplitudes.)" parenthetical:

    For `tatt` (10 templates; amplitudes $a_1, a_2, b_{TA}$ — the
    IA field is $a_1 O_1 + a_2 O_2 + a_1 b_{TA} O_{1\delta}$, so
    $\xi$ is quadratic in it: 1 GG + 3 GI + 6 II terms):

    $$\xi = K_0 + a_1 K_1 + a_2 K_2 + a_1 b_{TA} K_3
    + a_1^2 K_4 + a_2^2 K_5 + (a_1 b_{TA})^2 K_6
    + a_1 a_2 K_7 + a_1^2 b_{TA} K_8 + a_1 a_2 b_{TA} K_9$$

Source of truth: tatt_coeffs (IA/loss_functions.py) — template
order [GG, GI1, GI2, GI1d, II11, II22, II1d1d, II12, II11d, II21d],
coefficients [1, a1, a2, a1 b, a1^2, a2^2, (a1 b)^2, a1 a2,
a1^2 b, a1 a2 b] with b = b_TA; the K indices above follow that
order exactly. GRO-G policy holds: $b_{TA}$ is a braced subscript
on a single letter, no code-name underscores enter the math.

Gate GTC-A: the equation's ten terms match tatt_coeffs order +
forms verbatim; the math scanner stays clean; doc-only diff.

### ARCHITECT_HANDOFF
Task: micro-unit — add the TATT polynomial to the main README's
model section (spec: this addendum in notes/readme-yaml-chapter.md;
the insertion text is verbatim above). Base: the reorg commit;
`git log -1` must show it — else STOP. DOC-ONLY: one insertion,
zero .py / .yaml. Gate GTC-A (verify each term against
tatt_coeffs, run the GRO-G math scanner, anchors unaffected,
py_compile regardless). Report: brief IMPLEMENTER_HANDOFF + one
line appended here; do not commit, print the suggested commit
command.
### END

## Addendum 2 (2026-07-07, same micro-unit): subsection code blocks

User: the model section's `### mlp` "does not have a code block.
CNN also does not have a code block." STYLE RULE (extends the
chapter's one-example-per-section rule, binding on future edits):
every ### subsection documenting YAML knobs carries its own small
YAML block. Fixes, values verbatim from the train_single template:

    ### mlp gains:
        mlp:
          width:    128
          n_blocks: 4

    ### activation gains (same rule, same gap):
        activation:
          type:    H
          n_gates: 3

    ### cnn gains:
        cnn:
          kernel_size:    11
          rescale_kernel: false
          groups:         1
          separable:      false
          film:           false
          n_blocks:       1
          gate_init:      0.1
          # activation:        # optional: the head's own family
          #   type: gated_power

    ### trf: already carried by the section-closing model example
    (no change; add a trf: block only if the closing example ever
    loses it).

Gate GTC-B: the three blocks present, one per subsection, values
byte-matching the template (the GYC-C rule); budget unchanged
(each block <= 10 lines).

The micro-unit handoff above now covers BOTH addenda (the TATT
polynomial + these blocks), gates GTC-A + GTC-B, still doc-only.

## Addendum 3 (2026-07-07, same micro-unit): the n_heads diagram

User: the trf table's n_heads row ("must divide the token width")
is worth a graph explaining attention heads. Insert in the ### trf
subsection, beneath the knob table (verbatim; the arithmetic is
TRFBlock's: assert dim % n_heads == 0, d_head = dim // n_heads,
q/k/v .view(B, G, H, d_head), att = softmax(q k / sqrt(d_head)),
concat + wo):

    one token = one bin, width 26            n_heads: 2 -> d_head = 13

        ┌──────────────┬───────────────┐
        │ head 1: 1-13 │ head 2: 14-26 │     the feature axis is SLICED
        └──────┬───────┴───────┬───────┘     into n_heads equal parts:
               │               │             26 = n_heads x d_head, so
        G x G attention  G x G attention     n_heads must divide 26
        over ALL bins,   over ALL bins,      (1 | 2 | 13)
        using slice 1    using slice 2
               │               │
               └─── concat ────┴──▶ width 26 again (wo mixes the heads)

    One line beneath: the four projections (wq / wk / wv / wo) are
    26 x 26 at ANY n_heads — same parameters, same FLOPs; more
    heads = more, narrower attention patterns per bin pair.

Gate GTC-C: the diagram present in ### trf; its arithmetic matches
TRFBlock (the divisibility assert, d_head = dim // n_heads, per-head
G x G maps, concat back); the same-params/same-FLOPs line verified
against the 26 x 26 projections (they never depend on n_heads).

The micro-unit handoff covers addenda 1 + 2 + 3; gates GTC-A/B/C;
still doc-only.

IMPLEMENTED 2026-07-07 (Opus, base 29b23dd): all three addenda inserted in
README section 10 — the TATT $$ polynomial (GTC-A: the 10 monomials match
tatt_coeffs K0..K9 exactly, parsed both sides and compared), the mlp /
activation / cnn blocks (GTC-B: verbatim template values, cnn exactly 10
lines), and the n_heads diagram (GTC-C: arithmetic cross-checked against
TRFBlock — dim % n_heads == 0, d_head = dim // n_heads, wq/wk/wv/wo all
dim x dim). GRO-G clean (14 $$ blocks, zero underscores), anchors unaffected,
doc-only diff (README.md). Deviation declared: the diagram's SLICED/ALL/ANY
de-capped to obey the standing de-caps rule (same character widths, so the
ASCII alignment holds). Also flagged: the relayed handoff block named only
GTC-A, while this note folds all three addenda into the micro-unit — executed
all three per the note. Uncommitted; awaiting Architect re-audit.

### 2026-07-07 — Architect re-audit of the micro-unit: ACCEPTED

All three addenda verified on the raw diff + corrected harnesses:
GTC-A all ten TATT monomials match tatt_coeffs's cat order exactly
(my first probe matched the NLA display — the file's first xi
equation — and my GTC-B slicer was broken; both MY harness bugs,
re-run correctly, the diff itself was unambiguous throughout);
GTC-B the three subsection blocks present verbatim with template
values; GTC-C the diagram + the 26x26-projections line, arithmetic
matching TRFBlock; GRO-G scanner clean; anchors resolve; doc-only.

Deviations RULED: (1) executing all three addenda off the note
while the relayed block named only the first — ACCEPTED, the note
is the spec of record (the D-DOC9 precedent, correctly applied);
(2) de-capping SLICED/ALL/ANY — ACCEPTED, and the caps were in MY
spec text; the house rule the audits enforce applies to my
insertion texts too; (3) backticked identifiers — ACCEPTED.

COMMIT-READY. Suggested sentence: "Model-section polish: the TATT
polynomial (ten terms verbatim from tatt_coeffs), YAML blocks for
the mlp / activation / cnn subsections, and the n_heads multi-head
attention diagram (gates GTC-A/B/C Architect-verified)".

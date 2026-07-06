---
name: param-cuts-nested-block
description: "Design spec (Architect, 2026-07-06): move the eight physical window-cut keys out of the flat data: block into a nested data.param_cuts: sub-block, renaming legacy omegabh2_cut -> omegabh2_hi (user-confirmed). Follows the nested-model-schema precedent: YAML nests, from_config validates (PARAM_CUTS_KEYS whitelist) and flattens to the existing explicit kwargs; phys_cut_idx numerics untouched except the cut -> omegabh2_hi parameter rename. Flat cut keys under data: raise a migration error that prints the paste-ready param_cuts block. Hard schema break, no aliases. Extends [[omegamh2-ns-product-cuts]]."
metadata:
  node_type: memory
  type: project
---

# data.param_cuts: nested cut block (design spec)

Restructures the YAML interface of the physical window cuts
([[omegamh2-ns-product-cuts]], [[omegam2h2-window-cut]]). User request
2026-07-06; rename decision (omegabh2_cut -> omegabh2_hi) user-confirmed
same day.

## Target schema (the interface, verbatim)

    data:
      # ... train_dv / train_params / val_* / cosmolike_* unchanged ...
      param_cuts:
        omegabh2_lo:    0.005    # omega_b h^2 window
        omegabh2_hi:    0.035    #   (renamed from the legacy omegabh2_cut)
        omegam2h2_lo:   0.015    # Gamma^2 window (Planck ~ 0.045)
        omegam2h2_hi:   0.08
        # omegamh2_lo:    0.10   # omegamh2 = Omega_m (H0/100)^2 (Planck ~ 0.143)
        # omegamh2_hi:    0.18
        # omegamh2ns_lo:  0.10   # omegamh2 * n_s (Planck ~ 0.138)
        # omegamh2ns_hi:  0.17

Requiredness preserves current semantics exactly: `param_cuts` is required
and `omegabh2_hi` is required inside it (it is today's mandatory
`omegabh2_cut`); the other seven keys are optional (absent = that side not
cut).

## Design (the nested-model-schema precedent)

The YAML nests; the code boundary translates. from_config validates
`data.param_cuts` against a PARAM_CUTS_KEYS whitelist and flattens the block
into the existing explicit keywords of load_source / phys_cut_idx. The
quantity table and mask numerics of [[omegamh2-ns-product-cuts]] — just
gate-verified — do not move.

One declared interface change beyond the nesting: the parameter `cut` of
phys_cut_idx and load_source is renamed `omegabh2_hi` (still required), so
the internal name matches the YAML key and the other seven kwargs. Both
call sites (load_source -> phys_cut_idx; experiment.pool_size) update; the
keyword-vs-signature scan is the mechanical guard. No other signature moves.

The validation/flatten step is a standalone pure function (no torch use
inside it), so it can be exec-extracted and tested on the Mac like
phys_cut_idx was.

## Migration is loud, not aliased (house precedent: old names error)

- Any of the eight cut keys — including the old `omegabh2_cut` — found flat
  under `data:` raises a migration error that prints the paste-ready
  `param_cuts:` block with the offending keys placed inside it (the error
  message itself follows the paste-ready-YAML rule).
- An unknown key inside `param_cuts` raises against PARAM_CUTS_KEYS (the
  omegamh2-vs-omegam2h2 typo guard carries over).
- `omegabh2_cut` written inside `param_cuts` raises naming the rename.
- Missing `param_cuts` or missing `omegabh2_hi` raises naming exactly what
  to add.
- DATA_KEYS shrinks accordingly (cut keys out, `param_cuts` in).

## Constraints

- No back-compat aliases, no deprecation shims (user: avoid bloat).
- phys_cut_idx mask numerics byte-identical for identical values.
- Banner and print_design output content unchanged.
- cocoa.py's data-block path resolution must be verified transparent to the
  nested sub-dict (it resolves filenames; it must not iterate or reject the
  dict-valued key).
- Docs per house rules: experiment.py data-block docstring enumerates the
  sub-block and every key; load_source / phys_cut_idx Arguments updated for
  the rename; README data-block table + change-X-edit-Y + any YAML snippets;
  all three example YAMLs restructured; scans stay clean.
- Sequencing: this builds on top of the (currently uncommitted) window-cuts
  feature; the user commits that first, this lands as its own commit unit.

## Validation gate

- GN-A semantics: a nested config with today's production values yields
  kept train + val idx hash-identical to the flat config on the current
  code, and an identical banner.
- GN-B the four loud errors demonstrated verbatim (flat-key migration error
  showing the paste-ready block; unknown param_cuts key; omegabh2_cut inside
  the block; missing omegabh2_hi).
- GN-C all three example YAMLs pass the validation helper.
- GN-D docs parity; the handoff report ends with the paste-ready YAML block
  (standing user rule).
- GN-E house scans + keyword-vs-signature (catches a missed cut -> 
  omegabh2_hi call site) + whole-tree py_compile.
- GN-F (workstation, foldable into the pending window-smoke): one load with
  the nested block shows the normal banner.

## Resume state (Implementer appends below)

(none yet)

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Target file(s):** emulator/experiment.py (DATA_KEYS / PARAM_CUTS_KEYS,
  validate+flatten helper, from_config, stage_train / stage_val / pool_size
  threading, data-block docstring), emulator/data_staging.py (cut ->
  omegabh2_hi rename in phys_cut_idx + load_source, docstrings/diagram),
  emulator/cocoa.py (verify nested-dict transparency only), README.md,
  example_yamls/*.yaml.
- **Contracts & interfaces:** the Target-schema block verbatim; param_cuts
  required, omegabh2_hi required, other seven optional; PARAM_CUTS_KEYS
  whitelist; declared rename cut -> omegabh2_hi on phys_cut_idx +
  load_source; validation helper is a pure standalone function.
- **Verbatim numerics:** none new; the existing window formulas must emerge
  byte-identical (GN-A).
- **Constraints & edge cases:** Constraints section; loud migration errors
  as specified (the flat-key error prints the paste-ready block); no
  aliases; user commits, never you — and the current window-cuts feature is
  committed by the user before this lands.
- **Validation gate:** GN-A through GN-F; paste raw outputs (idx hashes,
  the four error messages, the banner, scan counts) in the
  IMPLEMENTER_HANDOFF, and end it with the paste-ready YAML block.
- **Notes entry:** notes/param-cuts-nested-block.md (this file; append
  resume state here).
- **Next milestone:** IMPLEMENTER_HANDOFF with gate evidence; GN-F rides the
  pending workstation session (window smoke + item-27 A/B).

### 2026-07-06 — Implementer (Opus 4.8) execution

Built on amazing-keller, on top of the (still-uncommitted) window-cuts
feature. Mac dev machine (no torch/cosmolike): the validate helper and
phys_cut_idx are pure numpy/Python, so GN-A/B/C were run standalone by
exec-extracting them from source; GN-F rides the workstation session.

**Done:**

- data_staging.py: the `cut` parameter of phys_cut_idx and load_source is
  renamed `omegabh2_hi` (still required, still the always-applied obh2 upper
  bound); the quantity-table entry, both call sites, docstrings, and the
  module diagram updated. Mask numerics untouched (byte-identical, GN-A).
- experiment.py: DATA_KEYS shrunk (the 8 flat cut keys out, `param_cuts` in);
  new `PARAM_CUTS_KEYS` whitelist (8 keys); a standalone pure
  `validate_param_cuts(data)` (+ `_param_cuts_migration_message`) that raises
  on the flat-key migration (printing the paste-ready block), a missing
  param_cuts, omegabh2_cut inside param_cuts (the rename), an unknown
  param_cuts key, and a missing omegabh2_hi, returning the validated
  sub-block; from_config calls it before the generic whitelist. stage_train /
  stage_val / pool_size now read `pc = d["param_cuts"]` and pass
  `omegabh2_hi=pc["omegabh2_hi"]` + the seven optional `pc.get(...)`;
  print_design's cuts line reads param_cuts (content byte-identical); the
  data-block docstring enumerates the nested sub-block and every key.
  The validation errors are ValueError, not KeyError: KeyError reprs its
  message (escapes the newlines), which would mangle the multi-line
  paste-ready migration block; ValueError renders it verbatim.
- cocoa.py: verified transparent — _DATA_PATH_KEYS is a fixed 5-key tuple
  (train/val dv/params + covmat); the resolver never iterates or rejects the
  dict-valued param_cuts key. No change needed.
- README (data_staging appendix, phys_cut_idx signature, pipeline diagram,
  change-X-edit-Y, the config prose) + all three example YAMLs restructured
  to the nested `param_cuts:` block with omegabh2_cut -> omegabh2_hi.

**Beyond the handoff's target-file list, but forced by the hard rename
(flagged for the Architect):** two downstream consumers read the old flat
`omegabh2_cut` key and would have silently broken (a diagnostics plot that
shades the excluded region):
  - emulator/plotting.py: `_cut_exclusion` / `_shade_cuts` read the cuts
    mapping by key; renamed its "omegabh2_cut" key + docstrings to
    "omegabh2_hi" (the shading is dormant unless a caller threads `cuts=`).
  - train_single_emulator_cosmic_shear.py: its diagnostics call built that
    cuts dict from the flat keys; now passes `cfg["data"]["param_cuts"]`
    directly (the plot reads the four keys it needs, ignores the rest). Its
    header comment updated to the nested schema.
Under the "no aliases" directive a dangling reference to a removed key is a
loose end, so these were completed rather than left; the diff is a key-string
rename plus a one-line dict->sub-block swap, keyword-vs-signature clean.

**Gate evidence (raw):**

- GN-A semantics: a nested param_cuts config with production values
  {omegabh2_hi 0.035, omegabh2_lo 0.005, omegam2h2 (0.015, 0.08)} yields a
  kept idx byte-identical (sha256 a8f1adf571ca19fd, 21803/40000 rows) to the
  old flat layout mapped through the same phys_cut_idx, and an identical
  per-window report (the banner). GN-A: PASS.
- GN-B the four loud errors (ValueError, rendered with real newlines):
    flat key    -> "the physical cut keys moved into a nested
                    data.param_cuts sub-block, and omegabh2_cut was renamed
                    omegabh2_hi. Replace the flat keys under data: with:\n\n
                      param_cuts:\n    omegabh2_hi: 0.035\n
                      omegam2h2_lo: 0.015"  (paste-ready block)
    unknown key -> "unknown data.param_cuts key(s): ['omegamh2_low'];
                    allowed: [...]"
    rename      -> "data.param_cuts has 'omegabh2_cut'; it was renamed
                    'omegabh2_hi' (rename the key, keep the value)"
    missing hi  -> "data.param_cuts is missing the required 'omegabh2_hi'
                    (the upper omega_b h^2 bound, the former omegabh2_cut)"
  GN-B: PASS.
- GN-C: all three example YAMLs' param_cuts blocks (read straight from the
  files) pass validate_param_cuts: each {omegabh2_hi 0.035, omegabh2_lo
  0.014, omegam2h2_lo 0.015, omegam2h2_hi 0.08}. GN-C: PASS.
- GN-D: docs parity done (README + docstrings + YAMLs); the handoff ends with
  the paste-ready YAML block.
- GN-E: house scans clean on the touched tree (caps 0, ` -- ` 0, legend-less
  0, non-hot comprehensions 0, lines>90 0), keyword-vs-signature CLEAN (no
  missed cut -> omegabh2_hi call site), whole-tree py_compile clean. PASS.
- GN-F: NOT run (no torch/cosmolike on the Mac); one nested-block load should
  show the normal banner. Rides the pending workstation session.

Open: GN-F only.

### 2026-07-06 — Architect re-audit: ACCEPTED (GN-F deferred)

Verified independently: validate_param_cuts + the migration message
exec-extracted and exercised fresh (nested config returns the validated
block; all four ValueErrors raise with the specified content; the migration
message carries a real multi-line paste-ready param_cuts block); the renamed
phys_cut_idx reproduces an independent all-eight-bounds reference mask
exactly on a fresh synthetic dump; DATA_KEYS = 12 / PARAM_CUTS_KEYS = 8
exactly as specified; omegabh2_cut survives only in migration/doc text;
cocoa.py iterates a fixed _DATA_PATH_KEYS list (nested dict transparent, no
change needed — as claimed); all three example YAMLs nested; scans clean
(the 3 comment dashes are prior stock). The suspected example-value drift
(omegabh2_lo 0.014) was checked against HEAD and is NOT a drift — the
template carried 0.014 before this change; 0.005 is the user's production
value.

Both deviations accepted, and both were properly declared this time:
(1) plotting.py + train_single consumer renames, forced by the no-alias hard
break; (2) ValueError over KeyError so the migration block renders with real
newlines.

Open: GN-F only (one nested-block load banner), folded into the pending
workstation session (window smoke + item-27 A/B + triangle GT-C).

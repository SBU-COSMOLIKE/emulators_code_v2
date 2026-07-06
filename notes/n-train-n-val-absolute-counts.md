---
name: n-train-n-val-absolute-counts
description: "Design spec (Architect, 2026-07-06): replace data.train_divisor / val_divisor with required absolute counts data.n_train / data.n_val, enforced AFTER param_cuts. The divisor path computed keep = N_raw // divisor against the PRE-cut dump size, so the staged row count silently shrank-or-drifted with the windows; the user sizes runs in absolute rows (runs are named ntrain25000) and deliberately over-generates the dump so absolute targets stay satisfiable after cuts. load_source already takes rows from the post-cut pool and raises when it is too small — this removes the one pre-cut leak, makes n_keep the only sizing rule, and adds a loud no-alias migration like the param_cuts one."
metadata:
  node_type: memory
  type: project
---

# n_train / n_val: absolute counts, enforced after param_cuts

Follows [[param-cuts-nested-block]] + [[omegamh2-ns-product-cuts]]. User
request 2026-07-06 ("last step of the night"): replace `train_divisor` /
`val_divisor` with `n_train` / `n_val`, and enforce that the counts are
satisfied AFTER the param cuts — the dump is deliberately over-generated
to make that possible.

## Why (the pre-cut leak)

`load_source` (data_staging.py) already does the right mechanics: shuffle
the whole dump once (seeded), apply `phys_cut_idx`, take the first `keep`
rows of the CUT pool, and raise `physical pool too small` when the pool
cannot supply them. The leak is only how `keep` is computed on the divisor
path: `keep = N_raw // divisor` uses the PRE-cut dump size, so the staged
row count is a fixed fraction of the dump, not a guaranteed count — the
same YAML stages different effective sizes as the windows change, and a
tightened window can silently starve a run that was sized for the full
dump. The user thinks in absolute rows (t16_ntrain25000); divisors are a
translation step nobody wants.

## Design

1. **YAML schema.** `data.n_train` and `data.n_val`: required, positive
   ints, absolute row counts enforced after `param_cuts`.
   `train_divisor` / `val_divisor` are deleted from `DATA_KEYS` (still 12
   keys: two out, two in). No aliases, hard break.
2. **Pure validation + loud migration**, mirroring `validate_param_cuts`:
   a standalone pure `validate_sizes(data)` in experiment.py that
   - raises ValueError when `train_divisor` or `val_divisor` is present,
     with a paste-ready block naming both new keys. No automatic value
     conversion: the semantics changed (pre-cut fraction -> post-cut
     count), so the values are the user's choice — the message must say
     exactly that and show example values marked as placeholders;
   - raises ValueError when `n_train` or `n_val` is missing, non-integer
     (bools rejected too), or < 1, naming the offender and the rule
     ("absolute rows kept after param_cuts");
   - returns the validated `(n_train, n_val)` pair.
   `from_config` calls it right after `validate_param_cuts`, BEFORE the
   generic `DATA_KEYS` whitelist, so a legacy key gets the migration
   message and not a bare "unknown key".
3. **`load_source` loses `divisor`.** The kwarg is deleted (experiment.py
   is its only caller); `n_keep` becomes required (keyword, int >= 1; the
   exactly-one-of guard collapses to a required-arg check). The existing
   pool check stays the enforcement point, with a fuller message:

       physical pool too small after param_cuts: kept 18342 of 400000
       rows, requested n_keep = 25000 (<dv basename>); loosen the windows
       or enlarge the dump

   The verbose summary line grows the pool: `used {keep} of {len(phys)}
   cut rows`, so post-cut enforcement is visible on every run, not only
   on failure. Module-header shape-flow diagram + docstring updated
   (`n_keep` is a ROW count here — do not confuse it with the kept
   data-vector LENGTH called n_keep in geometries_output/emulator_designs;
   the docstring keeps saying "rows").
4. **`stage_train(n_train=None)` / `stage_val(n_val=None)`.** The None
   default now reads `d["n_train"]` / `d["n_val"]`; the explicit argument
   still overrides (the learning-curve machinery in sweep_ntrain /
   bakeoff passes explicit sizes capped by `pool_size()` — unchanged).
   Docstrings updated; `pool_size()`'s docstring gains one line naming it
   the ceiling for `n_train`.
5. **Config banner** (the `cuts:` log line block in experiment.py): one
   added line, e.g. `sizes: n_train 25000  n_val 5000 (enforced after
   param_cuts)`.
6. **sweep_hyperparam driver:** `n_est = dv.shape[0] // train_divisor`
   becomes `n_est = int(cfg["data"]["n_train"])` — now exact, a better
   VRAM estimate (the memmap stays open for `dv.shape[1]`); the
   `save_sweep_table` meta key `"train_divisor"` becomes `"n_train"`.
7. **Docs sweep:** train_single header comment (its data-block key list),
   README line ~142 (`idx = phys[:n_keep or N // divisor]` -> the
   post-cut `idx = phys[:n_train]`), the `__init__` data-block docstring,
   and the three example YAMLs:

       n_train: 25000     # absolute training rows, enforced AFTER
                          # param_cuts (raises if the cut pool is smaller)
       n_val:   5000      # absolute validation rows, same rule

   (replacing the two divisor lines; values are the user's usual scale,
   theirs to change).

## Validation gate

Mac has no torch/cosmolike: gates run by exec-extraction as before; the
shuffle may be stubbed (a numpy-backed fake `torch.Generator`/`randperm`
in the harness namespace only) since the shuffle is not under test.

- GS-A validation (pure, Mac): legacy `train_divisor`/`val_divisor` ->
  ValueError whose text contains a paste-ready block with both new keys
  and the semantics-changed warning; missing `n_train` or `n_val` ->
  loud; non-int (incl. bool) and < 1 rejected; a valid block returns the
  pair; the three example YAMLs pass. Paste raw outputs.
- GS-B sizing semantics (Mac, exec-extracted load_source + phys_cut_idx
  + stage_source, synthetic dump where the windows kill roughly half):
  (1) request <= pool -> exactly n rows staged and every staged row
  satisfies the cuts; (2) request > pool -> ValueError naming the pool,
  the request, and the file; (3) prefix nesting: the n=50 selection is a
  prefix of the n=100 selection under the same generator (the
  learning-curve invariant stage_train promises).
- GS-C style: house scans clean on touched files; whole-tree py_compile;
  keyword-vs-signature check on the load_source call sites.
- GS-D workstation (rides the pending queue's ONE train_single run,
  which must now use the new keys): banner shows the sizes line and
  `used N of P cut rows`, with N exactly the YAML n_train.

## Sequencing

Two accepted features (param_cuts nesting, triangle shading) sit
UNCOMMITTED on amazing-keller awaiting the user's split commit
([[session-status-2026-07-06]]). Preferably start this after that commit
lands; if stacked anyway, keep the diff separable (this feature touches
experiment.py, data_staging.py, sweep_hyperparam driver, train_single
header, README, 3 YAMLs — overlap with the nesting feature is
experiment.py + YAMLs, so a third commit bucket must be listed
explicitly in the handoff evidence).

Eventual commit (user runs it):

    git commit -m "Replace train/val divisors with absolute n_train/n_val enforced after param_cuts (loud migration, gates GS-A-C Architect-verified)"

then merge from the main checkout: `cd ../../..` and
`git merge claude/amazing-keller-e798b6`.

## Resume state (Implementer appends below)

Implemented 2026-07-06 (Opus 4.8, amazing-keller); execution log + raw
gate evidence in the last section of this file.

### ARCHITECT_HANDOFF: READY FOR EXECUTION

- **Target file(s):** emulator/experiment.py (DATA_KEYS, validate_sizes,
  from_config call, stage_train/stage_val defaults, banner line,
  docstrings), emulator/data_staging.py (load_source signature + pool
  message + verbose line + header diagram),
  sweep_hyperparam_emulator_cosmic_shear.py (n_est + meta key),
  train_single_emulator_cosmic_shear.py (header comment), README.md
  (staging diagram line), example_yamls/*.yaml (3).
- **Contracts & interfaces:** YAML keys `data.n_train` / `data.n_val`
  required positive ints; `load_source(..., n_keep=<int>)` required
  keyword, `divisor` deleted; `stage_train(n_train=None)` /
  `stage_val(n_val=None)` signatures unchanged (None = the YAML key);
  `validate_sizes(data)` pure, returns `(n_train, n_val)`. This is an
  interface change — declare any further deviation explicitly.
- **Verbatim numerics:** none (no formulas change; the cut pool and
  shuffle logic are untouched).
- **Constraints & edge cases:** enforcement is the EXISTING post-cut
  pool check — do not add a second sizing path; no aliases, no silent
  divisor fallback; migration message carries placeholder values and the
  semantics-changed warning (values are the user's choice); bools are
  not ints; the row-count n_keep vs dv-length n_keep naming collision is
  documented, not renamed.
- **Validation gate:** GS-A through GS-C on the Mac (paste raw outputs,
  including the migration message verbatim); GS-D rides the workstation
  run, which must switch its YAML to the new keys.
- **Notes entry:** notes/n-train-n-val-absolute-counts.md (this file).
- **Next milestone:** IMPLEMENTER_HANDOFF with GS-A/B/C evidence + the
  updated third-commit split if stacked on the uncommitted features.

### 2026-07-06 — Implementer (Opus 4.8) execution

Built on amazing-keller, on top of a CLEAN base: the two features the
handoff called "uncommitted" (param_cuts nesting, triangle shading) were
committed by the user as `dba7588` before this ran, and the two compaction
notes as `f67e9c3`. So this feature is its own single commit unit, not a
third bucket on stacked work (the Sequencing "keep it separable" clause is
moot). Mac dev box (no torch/cosmolike/matplotlib/getdist/yaml): gates ran
by exec-extraction, with a numpy-backed torch + psutil stub for the shuffle
and the RAM probe only.

**Done (exactly the handoff's target-file list, no beyond-scope edits):**

- experiment.py: `DATA_KEYS` swaps `train_divisor` / `val_divisor` for
  `n_train` / `n_val` (still 12 keys). New standalone pure
  `validate_sizes(data)` (+ two module constants `_SIZES_MIGRATION_MESSAGE`,
  `_SIZES_PLACEHOLDER`): raises on either legacy divisor key (the migration
  message, no auto-conversion, placeholder values + semantics-changed
  warning), on a missing `n_train` / `n_val`, on non-int (bool rejected via
  the `isinstance(v, bool)` guard, since bool is an int subclass) and on
  `< 1`; returns the `(n_train, n_val)` pair. `from_config` calls it right
  after `validate_param_cuts`, before the generic `DATA_KEYS` whitelist.
  `stage_train` / `stage_val` drop the `divisor=` kwarg and pass
  `n_keep=(explicit if not None else d["n_train"|"n_val"])`. New banner
  `sizes:` line next to `cuts:`. Docstrings updated: the `__init__`
  data-block block (n_train / n_val entry), `stage_train` / `stage_val`
  (the None default now reads `data["n_train"|"n_val"]`), `pool_size` (one
  line: it is the ceiling for n_train), `run` (default is the YAML
  n_train), and the `print_design` "Lines printed" list (the new sizes
  line). Errors are ValueError (KeyError would repr+escape the paste-ready
  block's newlines), same lesson as the param_cuts migration.
- data_staging.py: `load_source` drops `divisor`; `n_keep` becomes a
  required positional-or-keyword arg (5th, right after `omegabh2_hi`), so a
  missing size is a plain TypeError at the call site (the exactly-one-of
  guard is gone). `keep = int(n_keep)`. Enriched pool message: `physical
  pool too small after param_cuts: kept K of N rows, requested n_keep = M
  (<dv basename>); loosen the windows or enlarge the dump`. Verbose line now
  `used {keep} of {len(phys)} cut rows`, so post-cut enforcement shows on
  every run. Module-header diagram + legend + the `n_keep` Arguments entry
  updated (n_keep is a row count here; the collision with the kept
  data-vector length also called n_keep in geometries_output /
  emulator_designs is documented, not renamed).
- sweep_hyperparam driver: `n_est = int(cfg["data"]["n_train"])` (exact, was
  `dv.shape[0] // train_divisor`; the memmap stays open only for
  `dv.shape[1]`); `save_sweep_table` meta key `train_divisor` -> `n_train`.
- train_single header comment: the data-block key list now reads
  `absolute sizes (n_train, n_val; rows kept after param_cuts), split
  settings (split_seed, ram_frac)`.
- README staging diagram line: `idx = phys[:n_train]  (absolute, post-cut)`.
- All three example YAMLs: the two divisor lines -> `n_train: 25000` /
  `n_val: 5000` with the "kept after param_cuts (raises if the pool is
  smaller)" comment; the sweep YAML keeps its "sweep_ntrain owns the
  N_train axis" note.

**Deviations from blueprint:** none. The changed-file set is exactly the
eight target files (git diff --stat confirms: README, data_staging,
experiment, 3 YAMLs, sweep_hyperparam, train_single). No consumer outside
the list read `train_divisor` / `val_divisor` (grep clean tree-wide), so
unlike the param_cuts rename there was no forced downstream edit. The other
stage_train callers (bakeoff, sweep_ntrain pass explicit `n_train=int(N)`;
sweep_hyperparam / tune call bare and inherit the new YAML default) are
transparent to the change.

**Gate evidence (raw, exec-extracted on the Mac):**

- GS-A validate_sizes (pure): valid blocks return the pair; both legacy
  divisor keys raise the migration ValueError (message rendered with real
  newlines, contains both new keys + "no automatic conversion" +
  "semantics changed"); missing / float / str / bool-True / 0 / negative
  all raise naming the offender + rule; all three example YAMLs' size keys
  pass (n_train 25000, n_val 5000, no divisor). Migration message verbatim:

      train_divisor / val_divisor are gone: run sizes are now the absolute
      row counts data.n_train / data.n_val, enforced after param_cuts, not
      a fraction of the pre-cut dump. The semantics changed (a divisor kept
      a fraction of the whole dump; a count guarantees that many rows
      survive the cuts), so there is no automatic conversion; choose the
      counts you want. Replace the two flat keys under data: with (example
      values, set your own):

        n_train: 25000     # absolute training rows kept after param_cuts
        n_val:   5000      # absolute validation rows, same rule

  GS-A: ALL PASS.
- GS-B sizing semantics (real data_staging module exec'd with torch+psutil
  stubs; synthetic 4000-row dump, an omegamh2 (0.10, 0.20) window killing
  51% -> pool 1971): (1) n_keep 500 and 1000 each stage exactly that many
  rows, 0 violating the cuts; (2) n_keep = pool+1000 raises `physical pool
  too small after param_cuts: kept 1971 of 4000 rows, requested n_keep =
  2971 (dv.npy); loosen the windows or enlarge the dump`; (3) prefix
  nesting under the same seed: idx@50 == idx@100[:50] (memmap path,
  ram_frac 0, so idx stays global). GS-B: ALL PASS.
- GS-C style (diff-based vs HEAD where the rule is "introduce none"): 0 new
  all-caps emphasis tokens and 0 new ` -- ` on all eight touched files;
  every touched .py <= 90 cols; 0 new comprehensions in the two core
  modules; box-drawing arrowhead / legend counts unchanged vs HEAD
  (experiment 4/1, data_staging 7/2); keyword-vs-signature clean (both
  load_source call sites pass 16 kwargs, all in the new signature, no
  `divisor`); py_compile clean on all four .py (and whole-tree py_compile
  = ALL COMPILE). GS-C: ALL PASS.
- GS-D workstation: NOT run (no torch/cosmolike on the Mac). One short
  train_single with the new-keys YAML and a tight omegamh2 window: the
  banner must show the `sizes:` line and `used N of P cut rows` with N
  exactly the YAML n_train. Rides the pending workstation session (folds
  in with the window smoke / param_cuts load / triangle GT-C).

Open: GS-D only, plus the Architect re-audit of this feature.

### 2026-07-06 — Architect re-audit: ACCEPTED with ONE delta (D-1)

Verified independently (own harness, own seed, own synthetic dump —
3000 rows, omegamh2 window (0.11, 0.16), pool 1119): every Implementer
claim reproduced. GS-A: all 16 validation cases + migration-message
content (real newlines, both keys as YAML lines, no-auto-conversion +
semantics-changed wording) + from_config ordering (param_cuts -> sizes ->
whitelist) + all three example YAMLs; DATA_KEYS = 12 with no divisors.
GS-B: exact staging at 700 and at the pool boundary; 0 staged rows
violate the window; pool-too-small message names pool / total / request /
file / remedy; `used 700 of 1119 cut rows` printed; prefix nesting holds
(note: on the in-RAM path stage_source SORTS rows for sequential memmap
reads, so the invariant is set inclusion there — verified both as sets
and as the raw phys[:200] == phys[:500][:200] prefix; the Implementer's
idx@50 check used the memmap path where idx stays global, also valid).
GS-C: own scans clean, both call sites keyword-complete, whole-tree
py_compile. The "n_keep 5th positional" shape was declared and both call
sites pass it by keyword — fine. Deviations: none found beyond the one
below.

**D-1 (required, small): guard `keep < 1` in load_source.** The spec
said n_keep is "an int >= 1" and the new docstring promises it, but
nothing enforces it: `int(n_keep) = -5` silently evaluates
`phys[:-5]` — my probe staged 1114 of a 1119 pool (pool minus 5), a
silent wrong-rows path, and n_keep = 0 stages an empty source. Not
YAML-reachable (validate_sizes guards), but the explicit
stage_train(n_train=...) path (sweep_ntrain / bakeoff) reaches it. Add,
right after `keep = int(n_keep)`, a loud ValueError (e.g. "n_keep must
be >= 1 (absolute rows to stage), got {keep}"), and extend GS-B with the
-5 and 0 cases. Everything else is accepted as-is; GS-D unchanged
(workstation).

### 2026-07-06 — Implementer: D-1 closed

data_staging.py load_source now raises `n_keep must be >= 1 (absolute
rows to stage), got {keep}` right after `keep = int(n_keep)`, before the
pool check, with a comment noting the YAML path is already guarded by
validate_sizes and this catches the explicit stage_train(n_train=...)
path (sweep_ntrain / bakeoff). GS-B extended: n_keep = -5 and n_keep = 0
each raise the ValueError (message names ">= 1" and the offending value);
GS-B still ALL PASS end to end (500 / 1000 exact + cut-satisfaction,
pool+1000 too-small message, prefix nesting, plus the two D-1 cases).
whole-tree py_compile clean. Feature is now commit-ready (folds into the
n_train/n_val commit unit); GS-D and the workstation queue unchanged.

### 2026-07-06 — Architect: D-1 closed (verified)

Independently re-probed: n_keep = -5 and 0 each raise
`n_keep must be >= 1 (absolute rows to stage), got {keep}` before the
pool check; the 700-exact staging, the pool-too-small message, and the
prefix nesting all still pass. One correction to the closure line above:
the n_train/n_val feature was already committed AND merged to main as
906528c before D-1 landed, so D-1 does NOT fold in — it is its own
follow-up commit (data_staging.py + this note):

    git commit -m "Guard n_keep >= 1 in load_source (n_train/n_val audit delta D-1, Architect-verified)"

Feature status: ACCEPTED in full; only GS-D (workstation banner check)
remains open.

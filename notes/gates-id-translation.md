---
name: gates-id-translation
description: "TRANSLATION TABLE 2026-07-07 (Architect): the human-friendly gate names used in gates/ code, CLI, logs, and README <-> the legacy two-letter spec codes used across notes/ (user directive: 'in the code and in the readme it must be a human-friendly system'; the internal system lives only here and in the home notes). The human name IS the Gate.id (log filename, status key, --gate selector); the legacy code appears in code only as the Gate's spec_code field, printed once per log header. Tiers renamed too: standing->backlog, week->new-features, save-sample->save-and-sample."
metadata:
  node_type: memory
  type: reference
---

# Gate names: human-friendly (code) <-> legacy spec codes (notes)

The gates/ code, CLI, logs, and README use ONLY the human names.
The legacy codes remain the keys of the audit history in notes/
(home notes, audit verdicts) and appear in code solely as each
gate's `spec_code`, printed once in the log header so a log can be
traced to its spec.

| Human name (Gate.id)     | Legacy | Home note                            |
|--------------------------|--------|--------------------------------------|
| ema-off-identity         | GM-C   | weight-ema-snapshot-coupled          |
| ema-smoke                | GM-D   | weight-ema-snapshot-coupled          |
| production-diagnostic    | DIAG (G1, G-F, GN-F, GS-D, GT-C) | driver-audit-phase-sweep-guards |
| single-phase-demotion    | GP-D   | resolve-phase-args-single-phase      |
| head-scheduler-override  | GH-E   | phase-blocks-nested-lr-scheduler     |
| eval-batch-invariance    | GE-C   | eval-bs-decoupling                   |
| berhu-loss               | GB-C   | loss-mode-berhu                      |
| loss-schema-equivalence  | GL-D   | loss-block-nesting                   |
| berhu-anneal             | GBA-C  | berhu-anneal-schedule                |
| ema-anneal               | GME-C  | ema-anneal-schedule                  |
| param-window-cuts        | item-27| omegamh2-ns-product-cuts             |
| triangle-shading         | GT-B   | triangle-cut-shading-all-windows     |
| joint-training           | GFT-C  | freeze-trunk-joint-phase2            |
| head-activation-pin      | GHA-F  | head-activation-per-component        |
| relu-tanh-norm           | GAN-C  | activation-families-norm-knob        |
| weight-decay-census      | GWD-C  | weight-decay-only-weight-matrices    |
| npce-training            | GPC-C  | npce-yaml-wiring                     |
| save-rebuild-drift       | GSV-C  | save-schema-resolved-config          |
| cobaya-adapter           | GCT-C  | cobaya-theory-adapter                |

Tier names (the --tier selector and BOARD.md grouping):

| Human tier     | Legacy tier |
|----------------|-------------|
| backlog        | standing    |
| new-features   | week        |
| save-and-sample| save-sample |

Derived keys renamed with their gate (board_config.json):
gate_configs keys become <human-name>-<leg> (e.g. GM-D-emasmoke ->
ema-smoke-config, GHA-F-pin -> head-activation-pin-config,
GHA-F-license -> head-activation-pin-license, GPC-C-excl-ia ->
npce-training-excl-ia, ...); golden_bases keys become the human
gate names. The smoke YAML filenames under gates/configs/ rename to
match their key. The harness's behavior is otherwise UNCHANGED
(user directive: the run semantics need no modification).

Audit rule going forward: workstation logs arrive named by the
human ids; this table + each log header's spec_code line map them
back to the home notes for the per-gate verdicts.

## The rewrite, split into four ~20-minute pieces (train mode)

User constraint 2026-07-07: intermittent laptop closures; each
piece must complete inside ~20 min and leave a VALID tree. Order is
binding (piece 1 is the only behavior-touching one; 2-4 are prose).
Base: f2b66b0. The four deploy-path VALUES in board_config.json are
preserved verbatim throughout. Run semantics unchanged.

1. RENAME (mechanical, this table is exact): Gate.id + tier
   constants + spec_code/title fields + the log-header spec-code
   line in board.py/run_board.py; gate_configs + golden_bases keys
   in board.py AND board_config.json; git mv the gates/configs/
   YAMLs to their new names. Gates: py_compile; the GGH-A stub legs
   green; full --dry-run: 18 plans, zero UNSET, path values intact;
   grep: legacy codes in code only as spec_code values.
2. PROSE board.py: module-docstring glossary (board, gate, tier,
   golden run, smoke, banner, worktree, preflight, resume) + every
   gate docstring to the WHAT/WHY/HOW-decided template; citations
   keep + one-phrase gloss; protocol jargon (D-G*) out. Gates:
   py_compile; docstring-stripped AST identical to piece-1's
   board.py; grep D-G in code = nothing.
3. PROSE run_board.py + checks/*.py module docstrings, --help
   strings, [harness] messages: same template + de-jargon. Gates:
   py_compile; docstring-stripped AST identical except string
   literals declared (help/messages); --dry-run behavior unchanged.
4. gates/README.md (NEW, ~100-130 lines): what the board is + how
   implemented (runner / registry / checks / configs / logs), how
   to run, the 19-test table (human name + one-liner), how to read
   a log. Gates: no legacy codes anywhere in it; the five-rule
   math scanner (trivial here); length within bounds.

Each piece ends with a mini IMPLEMENTER report + resume state in
[[gates-harness-user-run]]; the Architect audits piece 1 before the
COMMIT, and 2-4 in one end pass. ONE commit carries all four.

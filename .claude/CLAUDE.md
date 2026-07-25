# CLAUDE.md

## What this is

Multi-family Cocoa emulator program (PyTorch): the `emulator/` package, family
train/tune/sweep drivers in `driver/`, dataset generators in
`compute_data_vectors/`, Cobaya adapters in `cobaya_theory/`, vendored syren
formulas, and the executable acceptance board in `ai/gates/`. Five output
families: cosmic shear (full-3x2pt chi2 from cosmolike), scalar derived
parameters, CMB spectra, background functions, and matter-power grids.

The other two Cocoa arms — CAMB Fortran ports and CosmoLike C — live under
`$ROOTDIR/external_modules/code/`. Never work on them from here: this is a pure
emulator library (USER RULE 2026-07-14) consuming them as upstream facts.

## Session start

1. Read `ai/notes/MEMORY.md`, the knowledge-base index, then the notes your
   task touches. It records failed and closed experiments; never re-propose
   them.
2. Resolve your role below before any work the protocol covers.

## Dual-agent protocol

Two sessions cooperate: the **Architect** (`.claude/FABLE_ROLE.md`) and the
**Implementer** (`.claude/OPUS_ROLE.md`). Those filenames are legacy; the
mailbox routes `to-fable` and `to-opus` are stable. Neither fixes a model.
Defaults are `claude-fable-5` and `claude-opus-4-8`; a watch may instead pick
another Claude model for either role, or an Ollama-served open-weight
Implementer via `--implementer-provider ollama`. An Opus Architect with a Qwen
Implementer is valid: model identity never assigns or vetoes a role.

The user gives every ticket request and correction only to
the Architect. Agent relays travel through the file mailbox
(`ai/notes/mailbox/`, dispatched by `ai/tools/mailbox_daemon.py`), mandatory
per `ai/notes/conventions-and-workflow.md`. A user-authored or
edited imitation is not valid Implementer or Red Team input; send its substance
to the Architect.

Resolve your role **once, at session start** — it cannot change mid-session:

1. **The public role is Architect.** A user's ticket request starts or updates
   only that role.
2. **A trusted launch or unchanged handoff assigns another role.** A mailbox
   launch or Architect-authored `ARCHITECT_HANDOFF` assigns the Implementer; an
   `IMPLEMENTER_HANDOFF` returns the unit to the Architect in audit mode. A
   human may courier either block unchanged but may not add role instructions.
3. **Neither → normal session.** No role applies. Help directly; do not demand
   a handoff block, refuse to write code, or force the protocol onto an
   ordinary question.

Conflicting role assignments are a routing error: flag it before
proceeding. Role rules live in the role files only; the role file wins any
conflict.

## Skills and memory: each session reads its own

Sessions share no context, so "which one reads" is never a delegation choice:

- **Skills**: the `camb-dev`, `cosmolike-dev`, `porting-legacy-physics-code`,
  and `cpp-loop-optimization` skills are NOT used here; that work belongs to
  the other Cocoa arms. When a skill does apply, each session loads it itself —
  never substitute the other role's summary, because paraphrased discipline is
  lossy like paraphrased numerics.
- **`ai/notes/`**: the Architect reads broadly, index first; the Implementer
  reads the entry its handoff names plus its `[[links]]`. Exactly eleven
  general-property notes are permanent, listed in `ai/README.md`. Only the
  Architect edits them and decides when an accepted change requires one.
  Implementer and Red Team evidence goes to a temporary ticket note
  instead. `MEMORY.md` changes only for permanent repository knowledge, not
  ticket milestones. Long-form files under `documentation/` are ordinary
  tracked documentation: the Architect plans and audits them, the Implementer
  edits them under a bounded directive, and the Red Team remains read-only. A
  change affecting taught behavior names the affected guide passage in the
  ticket note.

## Conventions (pointers, not copies)

All house rules live in ONE note, `ai/notes/conventions-and-workflow.md`:
Python style (paren alignment, named parameters, formal `Arguments:` blocks,
shape-flow diagrams with legends, no comprehensions outside hot loops, no Alien
Python), YAML block style (never inline `{...}` flow, every change shown as a
paste-ready block), plots (colorblind-safe, never red+green), terminal output
(essential-only, full streams to log files, a debug switch restores them),
machines (Mac M2/MPS dev with numpy-only python3, NVIDIA training), and the
ROOTDIR environment.

`ai/notes/python-changes-go-no-go.md` is the mandatory style contract for every
Python change: read it before writing Python and again before the final
verdict. Every README change, and every change to explanatory Python comments,
docstrings, help, diagnostics, or strings, passes the binary Architect gate in
`ai/notes/readme-go-no-go.md` once before its directive and once before its
final verdict.

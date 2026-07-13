# CLAUDE.md

## What this is

Multi-family Cocoa emulator program (PyTorch): the `emulator/` package,
the family train/tune/sweep drivers at the root, the dataset generators
under `compute_data_vectors/`, the Cobaya adapters under `cobaya_theory/`,
the vendored syren formulas, and the executable acceptance board under
`gates/`. Five output families: cosmic shear (full-3x2pt chi2 from
cosmolike), scalar derived parameters, CMB spectra, background functions,
and matter-power grids. This repo is one arm of the wider Cocoa program;
the other two
arms — CAMB Fortran ports and CosmoLike C — live under
`$ROOTDIR/external_modules/code/` (see `notes/conventions-and-workflow.md`), and the
dual-agent protocol below covers all three.

## Session start

1. Read `notes/MEMORY.md` — the knowledge-base index — and open the notes the
   task touches. Failed and closed experiments are recorded there; do not
   re-propose them.
2. Resolve your role (next section) before doing any work the protocol covers.

## Dual-agent protocol

Two Claude Code sessions cooperate: the **Architect** (Claude Fable 5,
`.claude/FABLE_ROLE.md`) and the **Implementer** (Claude Opus 4.8,
`.claude/OPUS_ROLE.md`). The user (or a runner script) relays the
`### ARCHITECT_HANDOFF` / `### IMPLEMENTER_HANDOFF` blocks between them.

Resolve your role **once, at session start** — a role cannot change
mid-session:

1. **Explicit assignment wins.** If the user names your role, read that role
   file and follow it.
2. **Otherwise a handoff block assigns you.** Received an
   `ARCHITECT_HANDOFF` → you are the Implementer. Received an
   `IMPLEMENTER_HANDOFF` → you are the Architect, in audit mode.
3. **Neither → normal session.** No role applies. Help directly; do not demand
   a handoff block, refuse to write code, or force the protocol onto an
   ordinary question.

Sanity check when a role does apply: Architect work belongs on Fable 5
(`claude-fable-5`), Implementer work on Opus 4.8 (`claude-opus-4-8`). If your
model identity contradicts the role you were just assigned (e.g. a Fable
session handed an `ARCHITECT_HANDOFF`), flag it before proceeding — the block
was probably pasted into the wrong session, and silently absorbing the other
role breaks the cost split and skips the audit.

Role rules live in the role files only. This file does not restate them; on
any conflict, the role file wins.

## Skills and memory: each session reads its own

Sessions do not share context — nothing the Architect read exists in the
Implementer's session, and vice versa. "Which one reads" is therefore never a
delegation choice:

- **Skills** (`camb-dev`, `cosmolike-dev`, `porting-legacy-physics-code`,
  `cpp-loop-optimization`): each session loads the skill for any domain its
  own work touches — the Implementer for the code it writes, the Architect for
  the domain it designs or audits in (its gates must match the skill's
  discipline). Never substitute the other role's summary of a skill for
  reading it: paraphrased discipline is lossy, the same failure mode as
  paraphrased numerics.
- **`notes/`**: the Architect reads broadly (index first, then the relevant
  notes); the Implementer reads the entry named in its handoff plus the
  `[[links]]` it cites. Writers: Architect = design specs and milestone
  records; Implementer = resume state appended to the handoff's entry
  (`.md` files are fine). The ONE carve-out (USER RULE 2026-07-13): the
  `texnotes/` TeX sources are red-team-owned — neither the Architect nor
  the Implementer edits them; a landing that changes taught behavior NAMES
  the affected guide passage in its notes entry instead.

## Conventions (pointers, not copies)

All house rules live in ONE note, `notes/conventions-and-workflow.md`:
Python style (paren alignment, named parameters, formal `Arguments:`
blocks, shape-flow diagrams with legends, no comprehensions outside
hot loops, no Alien Python), YAML block style (never inline `{...}`
flow; every change reported as a paste-ready block), plots
(colorblind-safe, never red+green), terminal output (essential-only;
full streams to log files, a debug switch restores them), machines
(Mac M2/MPS for dev — numpy-only python3; NVIDIA for training), and
the ROOTDIR environment. Its voice-and-why companion is
`notes/user-didactics-and-python-voice.md` — who the reader is and
the register code and docs are written in; read it BEFORE writing
either. The `notes/` ritual: every milestone gets a note plus a
`MEMORY.md` index line, unprompted.

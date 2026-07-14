# CLAUDE.md

## What this is

Multi-family Cocoa emulator program (PyTorch): the `emulator/` package,
the family train/tune/sweep drivers at the root, the dataset generators
under `compute_data_vectors/`, the Cobaya adapters under `cobaya_theory/`,
the vendored syren formulas, and the executable acceptance board under
`gates/`. Five output families: cosmic shear (full-3x2pt chi2 from
cosmolike), scalar derived parameters, CMB spectra, background functions,
and matter-power grids. This repo is one arm of the wider Cocoa program;
the other two arms — CAMB Fortran ports and CosmoLike C — live under
`$ROOTDIR/external_modules/code/` (see `notes/conventions-and-workflow.md`)
and are NOT worked on from here: this repo is a pure emulator library
(USER RULE 2026-07-14), consuming those arms as upstream facts only.

## Session start

1. Read `notes/MEMORY.md` — the knowledge-base index — and open the notes the
   task touches. Failed and closed experiments are recorded there; do not
   re-propose them.
2. Resolve your role (next section) before doing any work the protocol covers.

## Dual-agent protocol

Two Claude Code sessions cooperate: the **Architect**
(`.claude/FABLE_ROLE.md`, using Fable by default) and the **Implementer**
(`.claude/OPUS_ROLE.md`, using Opus by default). The role files keep their
legacy model-named paths, and mailbox messages keep the stable `to-fable` and
`to-opus` route names, but neither name fixes the model. A mailbox watch may
select another Claude model for either role with `--architect-model` and
`--implementer-model`; for example, Opus may be the Architect while Sonnet is
the Implementer. The user (or a runner script) relays the
`### ARCHITECT_HANDOFF` / `### IMPLEMENTER_HANDOFF` blocks between them.
Agent-emitted relays travel via the file mailbox (`notes/mailbox/`,
dispatched by `tools/mailbox_daemon.py`) — mandatory per
`notes/conventions-and-workflow.md`; a user-pasted block stays valid input.

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

Model identity never assigns or vetoes a role. The explicit assignment or
handoff block above does that, while the mailbox launch independently chooses
which Claude model performs it. The defaults remain `claude-fable-5` for the
Architect and `claude-opus-4-8` for the Implementer when no launch override is
given. A model/role pairing such as Opus Architect or Sonnet Implementer is
therefore valid; a conflict between two role assignments is still a routing
error and must be flagged before proceeding.

Role rules live in the role files only. This file does not restate them; on
any conflict, the role file wins.

## Skills and memory: each session reads its own

Sessions do not share context — nothing the Architect read exists in the
Implementer's session, and vice versa. "Which one reads" is therefore never a
delegation choice:

- **Skills**: this is a pure emulator library (USER RULE 2026-07-14) — no
  CAMB Fortran ports, no CosmoLike C edits, no legacy-physics migrations
  happen in this repo, so the `camb-dev`, `cosmolike-dev`,
  `porting-legacy-physics-code` and `cpp-loop-optimization` skills are NOT
  used here; work in those domains belongs in the other Cocoa arms. If a
  skill ever does apply, each session loads it for its own work — never
  substitute the other role's summary of a skill for reading it:
  paraphrased discipline is lossy, the same failure mode as paraphrased
  numerics.
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

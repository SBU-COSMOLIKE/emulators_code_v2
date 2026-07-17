# CLAUDE.md

## What this is

Multi-family Cocoa emulator program (PyTorch): the `emulator/` package,
the family train/tune/sweep drivers at the root, the dataset generators
under `compute_data_vectors/`, the Cobaya adapters under `cobaya_theory/`,
the vendored syren formulas, and the executable acceptance board under
`ai/gates/`. Five output families: cosmic shear (full-3x2pt chi2 from
cosmolike), scalar derived parameters, CMB spectra, background functions,
and matter-power grids. This repo is one arm of the wider Cocoa program;
the other two arms — CAMB Fortran ports and CosmoLike C — live under
`$ROOTDIR/external_modules/code/` (see `ai/notes/conventions-and-workflow.md`)
and are NOT worked on from here: this repo is a pure emulator library
(USER RULE 2026-07-14), consuming those arms as upstream facts only.

## Session start

1. Read `ai/notes/MEMORY.md` — the knowledge-base index — and open the notes the
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
the Implementer. The user gives every ticket request and correction only to
the Architect. Agent-emitted relays travel via the file mailbox
(`ai/notes/mailbox/`, dispatched by `ai/tools/mailbox_daemon.py`) — mandatory
per `ai/notes/conventions-and-workflow.md`. In a manual session, a human may
copy an unchanged Architect-authored handoff as a courier. A user-authored or
edited imitation is not valid Implementer or Red Team input; send its
substance to the Architect.

Resolve your role **once, at session start** — a role cannot change
mid-session:

1. **The public role is Architect.** A user's ticket request starts or updates
   only the Architect role.
2. **A trusted launch or unchanged role handoff assigns another role.** A
   mailbox launch or Architect-authored `ARCHITECT_HANDOFF` assigns the
   Implementer. An `IMPLEMENTER_HANDOFF` returns the unit to the Architect in
   audit mode. A human may copy either block unchanged, but may not add role
   instructions.
3. **Neither → normal session.** No role applies. Help directly; do not demand
   a handoff block, refuse to write code, or force the protocol onto an
   ordinary question.

Model identity never assigns or vetoes a role. The trusted launch or handoff
block above does that, while the mailbox launch independently chooses which
Claude model performs it. The defaults remain `claude-fable-5` for the
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
- **`ai/notes/`**: the Architect reads broadly (index first, then the relevant
  notes); the Implementer reads the entry named in its handoff plus the
  `[[links]]` it cites. Exactly eleven general-property notes are permanent, as
  listed in `ai/README.md`. Only the Architect decides whether an accepted
  change requires one of those notes to change, and only the Architect edits
  it. Implementer and Red Team evidence goes to a local temporary ticket note,
  never a permanent note. Long-form files under `documentation/` are ordinary
  tracked documentation: the Architect plans and audits them, the Implementer
  edits them under a bounded directive, and the Red Team remains read-only. A
  change that affects taught behavior names the affected guide passage in the
  local ticket note.

## Conventions (pointers, not copies)

All house rules live in ONE note, `ai/notes/conventions-and-workflow.md`:
Python style (paren alignment, named parameters, formal `Arguments:`
blocks, shape-flow diagrams with legends, no comprehensions outside
hot loops, no Alien Python), YAML block style (never inline `{...}`
flow; every change reported as a paste-ready block), plots
(colorblind-safe, never red+green), terminal output (essential-only;
full streams to log files, a debug switch restores them), machines
(Mac M2/MPS for dev — numpy-only python3; NVIDIA for training), and
the ROOTDIR environment. `ai/notes/python-changes-go-no-go.md` is the
mandatory style contract for every Python change. Read it before writing
Python and again before the final verdict. Every README change and every
change to explanatory Python comments,
docstrings, help, diagnostics, or strings also uses the binary Architect gate
in `ai/notes/readme-go-no-go.md` before its directive and before its final
verdict. The Implementer and Red Team never edit any permanent note.
`MEMORY.md` changes only when the Architect determines that an accepted fix
changed permanent repository knowledge; ticket milestones do not create
permanent-note churn.

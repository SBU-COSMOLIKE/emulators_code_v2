# Permanent AI knowledge index

This is the cold-start index for general properties of the emulator library.
It deliberately contains no ticket chronology, queue state, dated audit, or
incident narrative. Git history preserves retired material.

Start with this page, then open the one topic note that owns the behavior you
are changing. The operating loop itself is taught in [`ai/README.md`](../README.md).

## The permanent eleven

Exactly these eleven Markdown files under `ai/notes/` stay in Git:

1. **[`MEMORY.md`](MEMORY.md)** — this index and the permanent/local boundary.
2. **[`project-and-history.md`](project-and-history.md)** — project goal,
   development arc, family pattern, and program-level lessons.
3. **[`conventions-and-workflow.md`](conventions-and-workflow.md)** — Python,
   documentation, plotting, terminal, YAML, environment, and collaboration
   rules.
4. **[`user-didactics-and-python-voice.md`](user-didactics-and-python-voice.md)**
   — the reader, teaching style, and code voice.
5. **[`models-and-designs.md`](models-and-designs.md)** — model families,
   correction heads, initialization, conditioning, and design doctrine.
6. **[`training-stack.md`](training-stack.md)** — losses, phase schedules,
   snapshots, sizing, diagnostics, and training invariants.
7. **[`artifacts-inference-warmstart.md`](artifacts-inference-warmstart.md)**
   — artifact schemas, rebuild, inference adapters, fine-tuning, transfer,
   and geometry identity.
8. **[`data-generation-and-cuts.md`](data-generation-and-cuts.md)** — data
   generation, sampling, cuts, staging, and publication contracts.
9. **[`families-background-mps.md`](families-background-mps.md)** — background
   and matter-power family properties.
10. **[`families-scalar-cmb.md`](families-scalar-cmb.md)** — scalar and CMB
    family properties.
11. **[`readme-go-no-go.md`](readme-go-no-go.md)** — the Architect's required
    instruction-time and review-time gate for tracked READMEs and explanatory
    Python comments, docstrings, help, diagnostics, and strings.

The Architect alone decides whether an accepted fix changes a general
property in this set. Only the Architect edits a permanent note. The
Implementer and Red Team never edit one, regardless of ticket type.
`MEMORY.md` changes only when the permanent map itself needs clarification;
it is not a per-ticket index.

## Local working records

The backlog, gate board, state notes, dated audits, incident reports, and
handoff registers are local working records. They remain in the local checkout
but are ignored by Git. Implementers and the Red Team write their ticket
evidence there; mailbox and relay files remain transport copies.

The model does not determine the responsibility. The Architect and Red Team
are the thinking roles: they must resolve design choices and record complete,
ordered implementation or repair directives in the temporary ticket note.
The Implementer is the execution role and may be a less capable model; it
follows the Architect's validated directive and returns a blocker instead of
inventing missing architecture. A Red Team repair remains candidate input
until the Architect adjudicates it. The exact packet formats live in
`.claude/FABLE_ROLE.md` and `.codex/REDTEAM_ROLE.md`.

Mailbox routing retains the explicit binding TERMINAL/no-reply exception
defined in `conventions-and-workflow.md`. An ambiguous instruction follows the
ordinary outbound rule.

When unfinished work must move to another developer, package it instead of
committing these records:

```bash
python3 ai/tools/backlog_bundle.py pack
```

The recipient validates with `read` and stages a fresh local review copy with
`import`. The bundle records the exact Git base, so the permanent eleven come
from repository history rather than from emailed worktree bytes.

## Finding current execution state

Use the local `ai/notes/backlog.md` for countable unfinished work. Use
`python3 ai/gates/run_board.py --list` for the current gate inventory and
`python3 ai/tools/handoff_router.py --status` for a read-only loop summary.
Do not infer current work from an old permanent-note paragraph.

The source note wins over a routing summary. A ticket result becomes permanent
knowledge only after the Architect accepts it and determines that it changes a
general property documented above.

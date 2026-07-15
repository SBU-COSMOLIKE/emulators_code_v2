# Permanent AI knowledge index

This is the cold-start index for general properties of the emulator library.
It deliberately contains no ticket chronology, queue state, dated audit, or
incident narrative. Git history preserves retired material.

Start with this page, then open the one topic note that owns the behavior you
are changing. The operating loop itself is taught in [`ai/README.md`](../README.md).

## The permanent ten

Exactly these ten Markdown files under `ai/notes/` stay in Git:

1. **`MEMORY.md`** — this index and the permanent/local boundary.
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

The Architect alone decides whether an accepted fix changes a general
property in this set. Only the Architect edits a permanent note. `MEMORY.md`
changes only when the permanent map itself needs clarification; it is not a
per-ticket index.

## Local working records

The backlog, gate board, state notes, dated audits, incident reports, and
handoff registers are local working records. They remain in the local checkout
but are ignored by Git. Implementers and the Red Team write their ticket
evidence there; mailbox and relay files remain transport copies.

Mailbox routing retains the explicit binding TERMINAL/no-reply exception
defined in `conventions-and-workflow.md`. An ambiguous instruction follows the
ordinary outbound rule.

When unfinished work must move to another developer, package it instead of
committing these records:

```bash
python3 ai/tools/backlog_bundle.py pack
```

The recipient validates with `read` and stages a fresh local review copy with
`import`. The bundle records the exact Git base, so the permanent ten come
from repository history rather than from emailed worktree bytes.

## Finding current execution state

Use the local `ai/notes/backlog.md` for countable unfinished work. Use
`python3 ai/gates/run_board.py --list` for the current gate inventory and
`python3 ai/tools/handoff_router.py --status` for a read-only loop summary.
Do not infer current work from an old permanent-note paragraph.

The source note wins over a routing summary. A ticket result becomes permanent
knowledge only after the Architect accepts it and determines that it changes a
general property documented above.

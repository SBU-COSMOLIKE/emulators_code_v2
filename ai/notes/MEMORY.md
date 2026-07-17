# Permanent AI knowledge index and change contract

This file is the starting point for durable knowledge about the emulator
library. Read this file before opening a topic note. Then open only the note
that owns the behavior under study.

The permanent notes describe current rules, scientific assumptions, software
interfaces, failure boundaries, and evidence that remains useful after one
bounded work request, called a ticket, closes.
The permanent notes are not a development diary. Git, the repository's
version-control system, stores old
wording and superseded designs. The local backlog stores unfinished work.

The operating workflow is explained in [`ai/README.md`](../README.md).

## GO/NO-GO contract for changing a permanent note

This contract is mandatory. Only the Architect may edit a permanent note.
The Implementer and Red Team may identify missing or incorrect knowledge, but
must return the proposed correction to the Architect.

### GO before writing

A permanent-note change receives GO only when every statement below is true:

1. The change records a general property that will help a future user or
   development model understand, modify, test, or review the library.
2. The information belongs to one topic note in the map below. Existing text
   will be updated in place instead of receiving a chronological addendum.
3. The source of truth is available in current code, a current configuration,
   a scientific definition, a reproducible validation command, or an explicit
   user rule.
4. Temporary ticket state, its place in the current work list, role
   conversation, and review chronology remain in local working records.
5. The planned wording is neutral. The wording addresses **the user**, **the
   reader**, **the Architect**, **the Implementer**, or **the Red Team** only
   when the role matters. The wording never points to a named individual.
6. A historical milestone is included only when the milestone explains a
   current design choice or prevents a known failed design from being repeated.
   The milestone is named by capability, not by date, saved repository
   version, review wave, or overnight narrative.
7. The planned explanation serves both a future development model and a
   physics undergraduate. An unfamiliar repository term is defined where it
   appears, and a broad rule is followed by a real repository example when an
   example makes the boundary easier to understand.
8. A required behavior that the current code violates is not hidden inside a
   permanent note. The durable rule stays in its topic note, while the
   Architect creates or updates a local backlog ticket in the same turn. A
   deliberate unsupported capability is stated as a present boundary, not as
   promised future work.

Any failed statement gives NO-GO. Record the material in a local ticket note
instead.

### GO for the finished text

The final permanent-note change receives GO only when all of these checks pass:

- The first paragraph states the durable subject and why the subject matters.
- Every heading names a technical idea. Headings do not contain dates, ticket
  numbers, queue numbers, audit waves, role verdicts, or temporary status.
- Current rules use direct language such as **must**, **must not**, **accepts**,
  **refuses**, **records**, and **returns**.
- A requirement identifies the owning code path, saved field, configuration
  key, or gate when that reference helps a future change.
- A scientific requirement states the physical or mathematical reason. A test
  recipe states what failure the test must distinguish.
- Terms are defined at first use. Internal shorthand is removed or linked to a
  durable definition.
- Abstract terms are tied to a concrete file, setting, command, input, or
  observable result. For example, a saved-publication rule names the files
  that must appear together instead of relying on the word **publication**
  alone.
- Detail is preserved when the detail can prevent a future regression. Large
  notes are acceptable. Diary entries are not.
- The text contains no date or timestamp, no named person, no gendered
  pronoun, and no personal preference attributed to a particular user. The
  anti-AI contract may quote a first- or second-person phrase only to prohibit
  that phrase.
- The text contains no ordinal audit-wave labels, role-verdict labels,
  temporary review status, personal-awake narratives, numbered run history,
  or source-control archaeology terminology.
- Open work is not inferred from the permanent note. The local backlog and
  executable gates determine current work and current evidence.
- The plain-language, neutral-audience, and anti-AI requirements in
  [`readme-go-no-go.md`](readme-go-no-go.md) also govern permanent-note prose.
  This shared writing standard does not give the Implementer or Red Team
  permission to edit a permanent note.
- All links resolve, the permanent-note set remains exactly eleven files, and
  the integrity guard is updated only for the deliberate accepted change.

Any failed check gives NO-GO. Rewrite the note before accepting the change.

### Required shape for technical findings

When a former audit finding contains durable knowledge, rewrite the finding in
this order:

1. **Rule:** the behavior the library must provide.
2. **Reason:** the scientific, numerical, usability, or integrity failure that
   the rule prevents.
3. **Implementation boundary:** the code, configuration, saved model
   publication, or public interface that owns the rule.
4. **Acceptance evidence:** the smallest test that fails for the forbidden
   behavior and passes for the required behavior.

Do not preserve the identity of the discovering role, the order of discovery,
the ticket identifier, or the old review status. Those facts do not change the
technical rule.

### Integrity update for an intentional note change

SHA-256 is a fixed-length fingerprint calculated from exact file bytes. A
**tracked** file is a file Git saves in repository versions. A **commit** is
one saved repository version, and a **diff** is the line-by-line comparison
between two versions. A **pinned base** is the full commit identifier that the
Architect selects as the accepted starting point. The SHA-256 guard compares
the permanent files with that starting point and protects against accidental
edits. An intentional accepted change does not weaken or bypass the guard.

1. The Architect reviews the complete note diff and every renamed reference.
2. The Architect confirms that exactly eleven top-level Markdown notes remain
   tracked under `ai/notes/`.
3. The relevant note tests, link checks, and prose checks run before commit.
4. The accepted commit becomes the new pinned base for later work.
5. `permanent_note_guard.py` runs against that full accepted commit before the
   next Implementer or Red Team unit begins.

The guard script and protected path list change only when the permanent-note
map changes deliberately.

## The permanent eleven

Exactly these eleven Markdown files under `ai/notes/` stay in Git:

1. **[`MEMORY.md`](MEMORY.md)** — this index, the permanent/local boundary,
   and the mandatory permanent-note GO/NO-GO contract.
2. **[`project-and-history.md`](project-and-history.md)** — project purpose,
   capability milestones that explain the current design, the family pattern,
   and program-level lessons.
3. **[`conventions-and-workflow.md`](conventions-and-workflow.md)** — Python,
   documentation, plotting, terminal, YAML, environment, and collaboration
   rules, including the permanent bug-versus-feature classification, backlog
   priorities and checksum practice, advisory Red Team reopening and finding
   notes, discovery limit, separate role lanes, and the one-ticket cycle rule.
4. **[`python-changes-go-no-go.md`](python-changes-go-no-go.md)** — the
   mandatory style contract for every Python change.
5. **[`models-and-designs.md`](models-and-designs.md)** — model families,
   correction heads, initialization, conditioning, and design rules.
6. **[`training-stack.md`](training-stack.md)** — losses, phase schedules,
   snapshots, sizing, diagnostics, and training invariants.
7. **[`artifacts-inference-warmstart.md`](artifacts-inference-warmstart.md)**
   — artifact schemas, rebuild, inference adapters, fine-tuning, transfer, and
   geometry identity.
8. **[`data-generation-and-cuts.md`](data-generation-and-cuts.md)** — data
   generation, sampling, cuts, staging, and publication rules.
9. **[`families-background-mps.md`](families-background-mps.md)** —
   background and matter-power family properties.
10. **[`families-scalar-cmb.md`](families-scalar-cmb.md)** — scalar and CMB
    family properties.
11. **[`readme-go-no-go.md`](readme-go-no-go.md)** — the mandatory contract
    for tracked READMEs and explanatory Python comments, docstrings, command
    help, diagnostics, and strings.

`MEMORY.md` changes only when the permanent map or the permanent-note contract
needs clarification. The file is not a per-ticket index.

## Local working records

The backlog, gate board, state notes, audits, incident reports, and handoff
registers are local working records. These records remain in the local
checkout and stay outside Git. The records may contain dates and execution
detail because the records describe temporary work.

The Architect and Red Team provide the reasoning. A directive must resolve
design choices and give the Implementer complete, ordered steps. The
Implementer may use a simpler model and must return a blocker instead of
inventing missing architecture. Architect acceptance authorizes the parent
daemon to build and verify one exact local `main` landing without waiting for
Red Team. Fable records only the GO decision; Fable does not merge, commit,
update a Git reference, or push. A later Red Team finding remains advice: the
Architect records its `NEW TICKET` or `REOPEN` bookkeeping first and assesses
the detailed finding note only when priority brings the ticket forward. A
normal cycle still waits for one review of the exact daemon-created landing
commit before a finite watcher exits. One ticket always equals one cycle.
With `--skip-redteam`, the verified local landing completes that ticket's
cycle because that watcher has no Red Team pass. A remote push failure becomes
visible push debt and does not erase or endlessly repeat a valid local
landing. A positive cycle limit is one shared admission total for the watch.
When another slot remains,
the Implementer may code the next admitted ticket while the Architect audits
an earlier immutable candidate C and the Red Team reviews an earlier
daemon-recorded landing L. Only the Implementer lane edits source code.
Severity never selects roles. Every Architect implementation directive
assigns bounded subagent work. The Implementer attempts every named launch
before an Integrator-owned edit, runs independent non-overlapping work at the
same time, integrates every return, and personally runs the combined
validation. Only an actual pre-edit runtime rejection can begin the bound
same-cycle capability-exception path. High and Critical ratings require
explicit comparisons with the next lower severity, and an Architect NO-GO to
reopening permanently bars that ticket from another `REOPEN`. The required
fields in each role-to-role instruction are defined in
`.claude/FABLE_ROLE.md` and `.codex/REDTEAM_ROLE.md`.

Force pushes are never allowed. The protected target is currently `main`; the
daemon has no target-branch option. Any future supported user-selected target
must inherit this complete rule before that option ships. The protected branch
may advance only by fast-forward locally and remotely. The Architect must
reject force flags, leading-`+` refspecs, branch deletion and recreation,
backward reference moves, and any reset, rebase, amend, filter, or
hosting-service operation that replaces protected history. Divergence stops
the operation and remains visible for the user; preserving history is more
important than closing a ticket, finishing a cycle, recovering automation, or
clearing push debt. `conventions-and-workflow.md`, section **Protected branch
history is never rewritten**, owns the complete rule.

Permanent-note changes use a separate Architect-only administration turn;
they are never folded into an Implementer ticket or a Red Team review. From a
bound Architect turn, the Architect queues that work with
`handoff_router.py --architect-notes-admin` and a plain-language summary. The
turn runs only after ordinary ticket, candidate, landing, and closure work is
idle. If durable knowledge changes, the Architect creates one clean commit P
directly on the unchanged local-main commit B, changes only one or more of the
eleven permanent notes, and returns the exact B/P `architect-notes-go`
message. The parent daemon validates and fast-forwards B to P, records any
push debt, and advances only clean safe role baselines. This administration
uses no ticket cycle and receives no Sol review, but an unresolved admin turn
or P landing prevents a watcher from claiming a clean exit.

Roles exchange instructions by saving and moving Markdown files through the
mailbox folders. A received final message needs no artificial reply only when
the message explicitly says that it ends the exchange and no reply is owed.
This is the only no-reply exception; an ambiguous message still requires a
saved reply.
`conventions-and-workflow.md` owns the complete rule.

When unfinished work must move to another developer, package the local records
instead of committing the records:

```bash
python3 ai/tools/backlog_bundle.py pack
```

The recipient validates the package with the script's `read` action and
prepares a fresh local copy with its `import` action. The package records the
exact Git commit from which the package was created. Permanent notes come from
repository history rather than from emailed working files.

## Finding current execution state

Use the local `ai/notes/backlog.md` for countable unfinished work. Use
`python3 ai/gates/run_board.py --list` for the gate inventory. Use
`python3 ai/tools/handoff_router.py --status` for a read-only workflow summary.
Do not infer current work from a permanent-note paragraph.

The Architect's source note is authoritative for one ticket. A ticket result
becomes permanent knowledge only after the Architect accepts the result and
determines that the result changes a general property in the map above.

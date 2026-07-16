# Conventions, workflow, and environment

This note defines mandatory repository-wide conventions. It records current
rules, not the history of how those rules were discovered. A future change
receives **GO** only when the relevant rule and its acceptance evidence are
satisfied. A contradiction, missing proof, or undocumented exception receives
**NO-GO**.

`ai/notes/python-changes-go-no-go.md` is the binding GO/NO-GO contract for
every Python change. The Architect reads that contract before preparing an
implementation directive and again before accepting the result. The rules in
this note provide repository context; they do not weaken or replace that
mandatory review.

## Workflow words used throughout this note

A **watch** is one running mailbox command that repeatedly checks saved
messages and starts the enabled roles. The long-running watcher process is
called the **daemon**. To **re-execute** means to start the same command again
from the saved agent folder with the original options.

**Transport** means the mailbox files, locks, and logs that carry one role's
saved message to another role. **Bootstrap** is the first creation and
validation of the saved agent worktrees. A Git **worktree** is a separate
working folder attached to one branch. A **branch** is one named line of saved
Git versions. A **state record** is a small saved file containing the worktree
path and branch. A **ticket** is one bounded work request controlled by one
Architect source note. **Dispatch** sends a saved instruction to the next
role. **Landing** places an accepted ticket commit on Git's `main` branch.

A **detached** worktree has no branch selected. A **prunable** worktree is one
whose registered folder Git reports as missing and eligible for removal. A
**dirty** worktree has uncommitted changes. **Ahead** means its branch has
local commits not present on `main`; **diverged** means both branches have
different commits after their last shared version.

A **gate** is a named validation job whose required result is written before
the job starts. The validation board records each gate and the command that
runs it. A **fixture** is the fixed input setup used by a gate. A **control**
is a valid case that must pass. A
**mutation** deliberately restores forbidden behavior and must fail. **Catch
power** is the demonstrated ability of a gate to fail for that mutation. A
**compile lane** is the part of a gate that must run the same check through
`torch.compile`, the compilation interface in PyTorch, the tensor and
machine-learning package used by the library.

## Python house style

These rules apply to `emulator/`, public drivers, checks, and support scripts.

- Keep every Python line at or below 90 columns. Align continuation lines with
  the opening parenthesis when practical. Otherwise use one consistent
  two-space hanging indent and place one item on each line.
- Pass arguments by name whenever the callee permits it. Keep only genuinely
  positional interfaces positional, such as mathematical operands, plotting
  coordinates, `model(x)`, and `*args` forwarding. Add a short naming comment
  when a positional tensor is not obvious.
- Prefer explicit loops in non-performance-critical code. Keep vectorized
  NumPy or Torch operations and loops inside compiled, forward, or batch hot
  paths. Use an abstract syntax tree (AST), Python's parsed representation of
  code structure, to find comprehensions; text search is not enough.
- Prefer direct, C-readable control flow. Avoid nested comprehensions, a
  lambda where a named function reads better, walrus expressions, starred
  argument tricks, and stacked conditional expressions. A single conditional
  expression is acceptable when it remains easy to read.
- Do not read mutable module-global data silently from a function. Pass the
  value explicitly. A necessary exception carries
  `# WARNING: reads module global NAME` at the read site.
- Represent constructible components as `{"cls": class_object, ...kwargs}`
  dictionaries. A `make_*` helper injects computed values, device values, and
  runtime state. Those values do not belong in the reusable specification.
- Use ordinary sentence case. Do not use all capitals for emphasis. Acronyms,
  interface literals, and the `WARNING` marker keep their required case.
- Do not use a spaced double dash as prose punctuation. This rule also applies
  to command help, errors, logs, comments, and docstrings.

The teaching notebook is a read-only style reference with a narrower line
width. Notebook-specific formatting does not relax the production rules.

## Explanatory Python prose

Code must teach the current program rather than narrate a review history.

- A module docstring uses complete sentences with a subject and verb.
- Every public function and every nontrivial private function has an
  `Arguments:` block naming each argument and a `Returns:` block. Add a
  `Raises:` block for meaningful refusal conditions. For a dictionary
  argument, enumerate the accepted keys, shapes, units, and meanings.
- A short private callback or test double may use one sentence when a formal
  block would only repeat the signature.
- Define a technical term at first use or replace it with plain language. A
  short local glossary is appropriate when several necessary terms occur in
  one file.
- Explain a cross-module call with a short provenance comment when ownership
  is otherwise unclear: `# function_name (module.py): current purpose`.
- Write mathematical relationships as formulas with every symbol defined.
  Tensor pipelines need a shape-flow diagram and a legend defining every
  dimension.
- Derive constants from named symbols. A concrete Legacy Survey of Space and
  Time first-year (LSST-Y1) example may follow the general derivation, but the
  example cannot replace it.
- Never state a list length, key count, or family count without checking the
  source of truth. Schema changes require a complete census for stale counts
  and enumerations.
- A documentation-only Python change is proven by comparing ASTs after
  docstrings are removed. A prose claim is not evidence of no executable
  change.

Domain symbols must not collide with established cosmology notation. Reserve
`h` for the dimensionless Hubble parameter `H0 / 100`, where `H0` is the
Hubble constant in kilometers per second per megaparsec. Use `step_frac` in
Python and `s_step` in prose for covariance finite-difference control. This
rule applies to code, formulas, logs, comments, notes, and handoffs.

For covariance checks, a reasonable cosmology means the explicit
Planck-Lambda cold dark matter (Planck-LCDM) fiducial in
`example_yamls/cmb_covariance_lcdm.yaml`. This model uses parameters fitted to
Planck observations and includes a cosmological constant and cold dark
matter. A scientifically justified nearby cosmology is also reasonable. An
extreme synthetic case can prove that a validator catches bad input, but
cannot alone prove that a scientific result is wrong.

Runtime validation must not depend on `assert`. Public configuration, data,
shape, geometry, and numerical guards use explicit typed exceptions before
mutation or accelerator setup. An optimized-mode subprocess must reject the
same negative fixtures with the same messages as ordinary Python. An internal
invariant also uses an explicit exception when continuing could publish a
scientific result.

YAML is the human-readable configuration-file format used by the repository.
Internal tracking abbreviations and review codes belong only in temporary
working notes. Public README files, Python prose, errors, logs, YAML comments,
and check labels state the underlying fact. A permanent note may be cited by
path when the design record is useful. A repository-wide leak scan must check
both coded forms and bare abbreviations and must read the complete output.

## Scope of scientific review

This repository is scientific software used for emulator production and
Markov-chain Monte Carlo (MCMC) inference. Ordinary review focuses on
scientific correctness,
reproducibility, model and data identity, stale-test truth, numerical
stability, and publication integrity. Cybersecurity, hostile-user threat
models, secrets, network attacks, and exploit hardening are outside ordinary
scope unless explicitly requested or directly required to protect scientific
results.

## README and teaching contract

`ai/notes/readme-go-no-go.md` is the binding review contract for README text,
comments, docstrings, command help, errors, logs, and explanatory strings. The
Architect reads that contract before writing a directive and again before the
final GO/NO-GO decision.

The root README first teaches how to run and configure the library. Detailed
design explanations belong in clearly separated appendices or specialist
README files. A concept is defined before it is used. Every explained YAML
concept includes a short fenced example copied from the real schema. Point to
one authoritative explanation instead of restating it in several places.

README files describe the current library. They do not contain development
dates, review rounds, queue state, landing state, abandoned formulas, or
biographical commentary. A current limitation may remain only as:

1. the present scope;
2. the consequence for the user; and
3. the action the user should take.

Parentheses contain only a short local definition, symbol, unit, or acronym.
If removing a parenthetical changes an essential instruction, promote that
content to a sentence, table row, or diagram label. Review parentheticals over
twelve words or with more than one clause.

GitHub mathematics follows these rules:

- no backslash command immediately followed by ASCII punctuation inside math;
- no LaTeX environments in Markdown math;
- no line-initial Markdown token inside a display-math block;
- no whitespace-adjacent inline dollar delimiter; and
- no code-name underscore inside math unless it is valid mathematical syntax.

README acceptance includes a complete anchor census and a complete census of
backticked repository paths. Every link target must resolve and every named
path must exist.

## Plots, terminal output, and YAML

- Do not combine red and green as the distinguishing plot colors. Use the
  colorblind-safe palette `#0072B2`, `#E69F00`, `#CC79A7`, `#000000`, and
  `#56B4E9`; use `viridis` for continuous maps; vary line style for grayscale.
- Terminal output is a dashboard: a short header, current result, one-line
  detail, and product paths. Complete streams go to immutable per-run logs. A
  debug option may mirror the full stream.
- YAML uses block style, one key per line, and no inline mapping. Preserve
  established value-column alignment. Range leaves use
  `[default, minimum, maximum, kind]`.
- Every YAML change is reported as a paste-ready block with enough surrounding
  context to identify its location.

## User-facing role boundary

The user communicates only with the Architect. Public mailbox commands accept
only the `architect` destination. Requests for implementation, review,
severity, model choice, a widespread search, corrections, or changed scope
all go to the Architect.

The Architect decides which enabled role acts next and writes the complete
downstream instruction. The Implementer and Red Team do not accept direct
user substance. A direct request reaching either role is returned to the
Architect as a blocker. A human may copy a generated handoff unchanged; that
copy is transport, not a new user instruction to the receiving role.

The default topology contains Architect, Implementer, and Red Team. A watch
may intentionally omit Red Team with `--skip-redteam` or `--no-red-team`.
Omitting Red Team does not weaken Architect planning, evidence review, or
exclusive GO/NO-GO authority.

A Git worktree is a separate checked-out working folder tied to a branch, so
an agent can edit without changing the user's checkout. Model choice and role
choice are separate. Current command-line model options may assign different
models to the Architect and Implementer. Sol has a separate effort setting;
there is no independent Red Team model option. None of these choices changes
role authority, Git worktree ownership, mailbox route, or evidence
requirements.

The Architect's source note is the authority for role topology and discovery
severity. Manual router options only confirm that saved plan. A disagreement
between the note and a manual option refuses before any lock, clipboard,
archive, or mailbox write. A detailed Architect directive includes:

- exact worktree, branch, and base;
- one `path::symbol` edit target for every owned file or test;
- ordered edits and named interfaces;
- types, shapes, algorithms, and numerical invariants;
- failure behavior and forbidden alternatives;
- named tests with expected observations;
- exact validation commands;
- stop conditions; and
- non-overlapping ownership when work is divided.

The instruction must be complete enough for a simple Implementer to execute
without inventing design decisions. A design-sensitive gap is a blocker. The
Implementer reports the exact missing fact and waits for a revised Architect
directive.

When enabled, Red Team reviews the named change and directly affected
behavior. A repository-wide attack happens only when the Architect records an
explicit user request such as “instruct the Red Team to do a widespread
search for …”. A confirmed finding returns to the Architect with root cause,
exact symbols, ordered candidate edits, invariants, a regression witness,
commands, acceptance checks, exclusions, and stop conditions. Red Team never
sends repair instructions directly to the Implementer.

The Architect audits raw evidence rather than summaries. A harness is first
checked against a known-good case and then against a deliberate mutation.
Only the Architect writes the final GO/NO-GO record.

### Discovery severity

Discovery severity controls which newly found defects may become tickets. It
does not change the scope of a named-change review.

- **Critical** is not a user discovery setting and is not a Red Team rating.
  Only the Architect may give this final backlog classification, and only
  when evidence shows that a current defect broadly breaks a central library
  workflow or systematically makes the library's scientific results invalid.
- **High** covers a defect that severely damages core behavior, loses data,
  halts normal operation, or can make the scientific result wrong.
- **Medium** includes High defects and concrete defects that are reasonably
  likely during normal use. A merely theoretical or very improbable edge case
  does not qualify. Medium is the default.
- **Low** permits every concrete defect, including an improbable edge case.

Harm and likelihood are separate judgments. The Red Team reports High,
Medium, or Low severity, likelihood, impact, scope, and evidence. The
Architect accepts, upgrades, or downgrades that assessment with a reason and
alone decides whether the finding becomes a ticket.

The Critical bar is deliberately much higher than the High bar. A ticket is
not Critical merely because it is High, urgent, scientific, hard to fix,
limited to one important family or platform, or lacks a convenient workaround.
Before assigning Critical, the Architect records why High is insufficient and
the exact evidence for broad library breakage. The Architect never promotes a
ticket to Critical merely to cross an emergency threshold or obtain another
Implementer.

A High discovery setting does not authorize a repository-wide search.
Critical is not accepted by `--severity` or `MAILBOX-SEVERITY`. Fix-only mode,
an omitted Red Team, and the discovery-admission limit still take precedence.

The user's explicit phrase “do a widespread search” creates a special Low
discovery request. The saved mailbox severity is automatically Low. The
Architect does not send the search while any accepted Critical, High, or
Medium ticket remains open. Low tickets do not block the search. This stricter
rule exists because a broad search for optional new findings must not delay
known non-Low work.

### Ticket character limit

The `--max` option limits the complete committed change for one ticket. A
positive limit counts added characters plus deleted characters as Unicode code
points from the ticket's bound full base commit to a clean `HEAD`. Replacing
text counts both the removed text and the added text. The count covers every
tracked code, test, and documentation file in that ticket.

An exact-boundary result is accepted. `--max 0` removes the numeric ceiling
only; it does not weaken scientific correctness, completeness, tests,
documentation, or readability. If a complete readable fix cannot fit, the
Architect returns NO-GO and asks for a smaller ticket or a changed limit.

## Persisted agent worktrees

Ordinary agent work never occurs in the user's repository checkout. The
mailbox system owns two persisted worktrees. `<REPO_ROOT>` means the top folder
of the checked-out emulator repository:

| Resource | Required value |
| --- | --- |
| Claude coordination name | `mailbox-primary` |
| Claude worktree | `<REPO_ROOT>/.claude/worktrees/mailbox-primary` |
| Claude branch | `refs/heads/claude/mailbox-primary` |
| Claude state | `<REPO_ROOT>/.claude/worktrees/.mailbox-primary-worktree.json` |
| Sol worktree name | `mailbox-sol` |
| Sol worktree | `<REPO_ROOT>/.claude/worktrees/mailbox-sol` |
| Sol branch | `refs/heads/codex/mailbox-sol` |
| Sol state | `<REPO_ROOT>/.claude/worktrees/.mailbox-sol-worktree.json` |
| Bootstrap lock | `<REPO_ROOT>/.claude/worktrees/.mailbox-primary-worktree.lock` |

Architect and Implementer share the Claude coordination worktree because they
work one after another and both must see the same uncommitted code, note,
staging area, and mailbox. Sol uses the independent Sol worktree. Changing a
model option never selects a different worktree.

In normal advisory Red Team work, Sol does not edit tracked files or create a
commit. Sol edits code, tests, or tracked documentation in the saved Sol
worktree only for an explicit second-Implementer unit or another separately
authorized tracked unit; neither case permits edits to the permanent eleven.
The primary Claude worktree's `ai/notes/` directory is the one shared
coordination location for mailbox files, relay copies, the local backlog, and
temporary records. Sol receives explicit access to that directory and must
not create a second active mailbox or backlog under the Sol worktree.

Each state record stores the canonical Git common directory, stable name,
absolute path, and full branch reference. Every reuse is checked against
`git worktree list --porcelain`. Before touching the mailbox, the launcher
re-executes the saved primary worktree's current daemon with the original
arguments, interpreter, and working directory. The saved topology marker must
also prove that Sol has a dedicated worktree.

Command-line interface (CLI) validation happens before worktree provisioning.
The CLI is the set of options accepted by the terminal command. Help, preview
with no action, invalid combinations, and dry-run create no branch, worktree,
state, or lock. Live actions are `--watch`, `--once`, `--send architect`, and
`--ping architect`.

On a clean clone, establish the primary worktree with one valid live action
before writing an uncommitted source note. A new worktree starts from
committed local `main` and cannot see an uncommitted note in another checkout.

Legacy adoption is deliberately narrow. A current, attached, non-main
worktree under `.claude/worktrees/` may be adopted only when the first live
command starts from that same worktree and no conflicting active transport
exists elsewhere. Active, ambiguous, duplicated, or pre-migration transport
is never copied, merged, renumbered, or deleted. A unique main-checkout store
containing only completed messages and regular logs may be copied byte for
byte under both transport locks. Copies are bounded to 16 MiB per file and
64 MiB total. Partial identical copies are resumable; conflicting bytes
refuse.

An interrupted clean bootstrap may resume only when the exact default path,
branch, and Git registration validate. A uniquely registered `git worktree
move` may update the saved path after full validation. Detached branches,
wrong branches, deleted refs, manual directory moves, corrupt state, prunable
worktrees, or unregistered branches refuse without fallback.

The daemon preserves dirty, ahead, and diverged work. It never stashes,
cleans, resets, checks out, prunes, merges, fetches, pulls, pushes, or invents
a replacement worktree. Recovery starts by preserving the state and transport
paths and comparing them with Git's registered worktrees.

## Notes-first communication and mailbox transport

The substantive record for a ticket is a local temporary note under
`ai/notes/`. The note is written before a handoff. It contains scope,
scientific evidence, counterexample, design contract, exact file and symbol
targets, changed files, branch or commit identity, raw-test locations,
remaining obligations, and acceptance conditions.

### Red Team finding note GO / NO-GO

Red Team is always advisory, including when its model is more capable than the
Architect or Implementer model. It can find defects and propose evidence, but
it cannot decide a ticket, change the backlog, direct an Implementer, require a
GO, delay a commit, or veto a landing. Its influence must come from reading the
authorized code adversarially and explaining a real bug persuasively to the
Architect and a human reader.

Every `Backlog action: NEW TICKET` or `Backlog action: REOPEN` return has one
ignored temporary Markdown note at the stable repository-relative path
`ai/notes/<plain-ticket-slug>-red-team-finding.md`. The slug uses lowercase
words and hyphens. It contains no date, cycle number, model name, worktree
name, or severity. A later reopening of the same ticket updates the same note
instead of creating diary-like dated files. The relay cites this relative
path, never an absolute worktree path.

The note has these headings in this order:

1. **High-level summary** uses at least three short, complete sentences to
   explain expected behavior, observed failure, and the user or scientific
   consequence. It defines specialized terms before relying on them.
2. **Affected behavior and code path** gives a concrete input or action, the
   observable result, and the relevant repository paths and symbols. It walks
   through the execution path in reading order.
3. **Reproduction and evidence** gives numbered steps, exact commands or
   fixtures, expected and observed output, and raw-evidence locations. It
   labels reproduced facts separately from inferences.
4. **Impact and proposed severity** explains realistic harm, likelihood, the
   proposed High, Medium, or Low rating, and why the evidence meets that bar.
5. **Review scope and exclusions** names the bounded commit, change, behavior,
   paths, and symbols reviewed and states what was not checked. An authorized
   widespread search states its exact Architect-approved boundary.
6. **Proposed acceptance evidence** gives a regression witness, exact commands,
   and observable passing result. These are proposed checks for the Architect,
   not Red Team approval or a veto.
7. **Uncertainty and counterevidence** records missing facts, alternative
   explanations, successful cases, evidence against the finding, and what
   would disprove it. `None found` is acceptable only after the note explains
   how counterevidence was sought.
8. **Repair directive** contains the complete candidate repair packet required
   by `.codex/REDTEAM_ROLE.md`.

The note persuades through facts and explanation. A thin assertion such as
"broken" or "the test failed" is `NO-GO`. Rhetorical pressure, inflated
severity, diary/date/wave narration, model-centered history, hidden
uncertainty, and fabricated commands, files, outputs, or observations are
`NO-GO`. A finding does not omit counterevidence merely because it weakens the
argument.

This detail transfers the completed investigation and conserves Architect
tokens. The Architect already owns prioritization, design, decision-complete
Implementer directives, audits, and backlog maintenance. A strong finding note
lets the Architect later judge the issue and plan targeted independent checks
without reconstructing Red Team work. That economy does not lower evidence
standards and does not make the note authoritative.

Receipt and assessment happen at different times. On receipt, the Architect
does not reproduce or substantively analyze a `NEW TICKET` or `REOPEN`
finding. It performs bookkeeping only: create or restore the ticket, apply the
reopen-count and automatic-severity mechanics, preserve the note path,
acknowledge, and return to current work. The backlog technical record includes
this exact line:

```text
See further instructions at ai/notes/<plain-ticket-slug>-red-team-finding.md
```

Only when priority later brings that ticket forward does the Architect assess
the note, perform targeted independent verification, set the final severity,
and decide whether to plan a repair. A missing or weak section is recorded as
evidence the Red Team must improve then; it never holds admission bookkeeping
or an unrelated accepted commit open.

### Backlog ticket GO / NO-GO

`ai/notes/backlog.md` is the local list of unfinished and completed tickets.
It is written for a human reader first and retains a separate technical record
for development tools. The Architect owns its structure and is the only role
that admits a ticket, changes its status, or moves it between the open and
closed sections.

The file begins with **Open tickets** and **Closed tickets** entries in its
contents list. The full **Open tickets** section comes before **Closed
tickets**. Its index contains exactly one linked line for each unfinished
ticket. That line begins with the exact text `- OPEN` because the watcher
counts this marker. There is no second `- OPEN` line inside that ticket, and
every index link resolves to exactly one detailed open section.

The Architect classifies every admitted ticket as Critical, High, Medium, or
Low using the harm and likelihood rules above. The linked index shows that
classification and is grouped in priority order: Critical first, High second,
Medium third, and Low last. Work starts with the first dispatchable ticket in
the highest nonempty group. A blocked ticket remains in its severity group
with its blocker; work may move to the next ticket while required hardware,
data, or an external decision is unavailable. Every severity change records
the new evidence and the Architect's reason.

Every admitted ticket also records one type: **Bug fix** or **New
functionality**. Type says whether the ticket repairs behavior or adds a
capability. Priority says when the ticket should be worked.

- A Bug fix may be Critical, High, Medium, or Low.
- New functionality may be High, Medium, or Low, but never Critical.
- The user controls feature priority. An unstated feature priority defaults to
  Medium; the Architect does not invent urgency.
- Critical bugs preempt every feature.
- A user-designated High feature comes before High bugs.
- High bugs come before a Medium feature.
- A Low feature waits for Critical, High, and Medium bug fixes.
- “After the backlog is closed” means a Low feature whose prerequisites are
  every ticket that was already open when the feature was admitted. The
  feature's own open line is not one of those prerequisites.

Within one permitted group, preserve index order unless a recorded blocker or
prerequisite requires moving to the next ticket.

Every ticket also keeps an integer named **Red Team reopen count**. It starts
at `0` and never resets. This number records how many end-of-cycle Red Team
reviews said `REOPEN`. The Architect performs that state change as quick
bookkeeping: increment the count, restore the open ticket, acknowledge the
return, and leave the deeper evidence review for a later Architect turn. This
prevents an advisory finding from disappearing merely because the Architect
was busy when the return arrived.

The Architect still has the final word after that immediate bookkeeping. When
the count is greater than `1`, the Architect later compares the new evidence
with the ticket's earlier reopening reports and becomes stricter after each
additional attempt. The review asks whether the Red Team found a materially
new failure or is repeating an old objection without new evidence. The
Architect may close the ticket again or lower its priority with a recorded
reason. When the count becomes `6`, or is already greater than `5`, the
ticket's priority is automatically Low. No role may waive that automatic
change, even for a ticket that was previously Critical or High.

Red Team is always advisory. The ordinary acceptance path is: the Architect
assigns a ticket, the Implementer repairs it, the Architect audits the repair,
and an Architect `GO` closes and commits the ticket immediately. Red Team does
not supply a required `GO`, and the Architect never waits for Red Team before
committing an accepted fix.

At the end of each cycle, Red Team reviews the tickets closed during that
cycle. If the bug remains, its handoff says `Backlog action: REOPEN`. The
watcher does not hold the cycle open while this advisory message waits. On
receipt, the Architect immediately restores the ticket and increments its
reopen count. The Architect evaluates the evidence, final priority, and next
decision later.

#### Recreate the local backlog consistently

`backlog.md` is local and is not present in a clean Git clone. When it is
missing, the Architect creates it at the exact path `ai/notes/backlog.md`
before admitting work. The Architect does not copy an imported backlog
blindly and does not invent a shorter private format. A backlog received
through the supported bundle tool is input to review; the Architect validates
and normalizes it to this contract before dispatch. Every new local backlog
uses this exact opening and these headings in this order:

```markdown
# Execution backlog

This file is local to this clone and is not committed to GitHub. The Architect
recreates it from this contract and updates it whenever a ticket changes.

## Contents

- [Open tickets](#open-tickets)
- [Closed tickets](#closed-tickets)

## How to read this backlog

Each line beginning `- OPEN` represents one unfinished ticket. A Bug fix
repairs behavior that is wrong now. New functionality adds a capability.

Priority controls work order. Critical is reserved for a bug that broadly
breaks a central workflow or systematically makes the science wrong. High,
Medium, and Low use the harm and likelihood definitions in the permanent
workflow contract.

Every ticket has a Red Team reopen count that starts at zero. A ticket moved
to Closed does not wait for Red Team approval. At the end of a cycle, Red Team
may send REOPEN if the bug remains. The sixth REOPEN assessment automatically
makes that ticket Low.

New discovery stops when ten or more Critical, High, and Medium tickets are
open; Low tickets do not enter that count. Sol may act as a second Implementer
only in an emergency with more than one Critical Bug fix or more than ten High
Bug fix tickets. A malformed open line blocks both decisions until the
Architect repairs it.

# Open tickets

## Open ticket index

### Critical

No open Critical tickets.

### High

No open High tickets.

### Medium

No open Medium tickets.

### Low

No open Low tickets.

# Closed tickets

No closed tickets.
```

The introductory `How to read this backlog` section says, in ordinary
language, that priority controls work order, type distinguishes a repair from
a new capability, and each exact `- OPEN` index line represents one unfinished
ticket. It also states the discovery and emergency counts below. An empty
priority group remains visible. When its first ticket is added, the Architect
replaces the `No open PRIORITY tickets.` sentence with the index line; the
empty sentence and a ticket line never appear together. A clean clone with no
accepted work still receives the complete skeleton, including all four empty
priority groups and the `No closed tickets.` sentence.

Every open index line uses this exact form:

```text
- OPEN **PRIORITY** **TYPE** — [Plain human title](#unique-anchor)
```

`PRIORITY` is exactly `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`. `TYPE` is
exactly `BUG FIX` or `NEW FUNCTIONALITY`. `CRITICAL` with `NEW FUNCTIONALITY`
is invalid. The line appears under the matching priority subheading and is the
ticket's only text beginning `- OPEN`. The four groups remain in
Critical-High-Medium-Low order. Within High, user-designated High new
functionality appears before High bug fixes. Within any remaining group,
preserve admission order unless a recorded prerequisite or blocker explains
why the next ticket is being worked first.

`unique-anchor` contains only lowercase ASCII letters, digits, and hyphens;
for example, `cmb-progress-loses-multipole-labels`. It describes the problem,
is unique within the file, and is not merely an internal ticket number. The
link target and the `<a id="...">` value must match byte for byte. Each index
link resolves once, every detailed open ticket has one index link, and a
closed ticket has no `- OPEN` line.

Each detailed open ticket uses this exact heading order:

```markdown
<a id="unique-anchor"></a>
## Plain human title

### High-level summary

[Three or more short, complete sentences. Sentence 1 explains the normal
purpose with a concrete example. Sentence 2 explains the current failure.
Sentence 3 explains the user or scientific consequence.]

### Current status

[Use exactly one of these lines:]

**Ticket type: BUG FIX.**

**Ticket type: NEW FUNCTIONALITY.**

**Red Team reopen count: 0.**

**OPEN.** [Current stage, blocker, or prerequisite.]

[For a Bug fix, use:]

**Severity: PRIORITY.** [Concrete harm and likelihood. A Critical bug must
also explain why High is insufficient.]

[For New functionality, use:]

**Priority: PRIORITY.** [The user's feature priority and any prerequisite.]

### What is already fixed

[Completed work, stated without implying closure.]

### What is missing

[Every remaining action, check, review, landing, or prerequisite.]

<details>
<summary>Technical record for development tools</summary>

[Exact files, symbols, commits, commands, evidence, and failure boundaries.]

[For a Red Team `NEW TICKET` or `REOPEN`, include exactly:]

See further instructions at ai/notes/<plain-ticket-slug>-red-team-finding.md

</details>
```

The Architect writes a feature's user-supplied priority and any
“after the backlog is closed” prerequisite into `Current status`. For a
Critical bug, the priority reason must also explain why High is insufficient.
These rules apply to every user's local backlog; old local records are brought
into this shape when first touched rather than copied as an incompatible
private format.

This example illustrates the required level of explanation. It is an example,
not an admitted ticket:

```markdown
- OPEN **HIGH** **BUG FIX** — [Saved CMB progress can lose its multipole labels](#cmb-progress-loses-multipole-labels)

<a id="cmb-progress-loses-multipole-labels"></a>
## Saved CMB progress can lose its multipole labels

### High-level summary

A long CMB run should save both its spectra and the multipole values that label
those spectra; for example, the first saved row may represent multipole 2.
The current progress file can preserve the spectra while omitting those labels.
A resumed run can then attach a value to the wrong multipole and produce a
scientifically incorrect result without an obvious file-reading error.

### Current status

**Ticket type: BUG FIX.**

**Red Team reopen count: 0.**

**OPEN.** The format check is designed, but the resume-path test is missing.

**Severity: HIGH.** Normal checkpoint recovery can silently change the
scientific meaning of saved values.

### What is already fixed

The writer now stores the multipole array beside each new progress file.

### What is missing

Add a test that resumes from an old file without the array and confirms that
the program stops with a useful explanation instead of guessing the labels.

<details>
<summary>Technical record for development tools</summary>

Record the exact writer and reader symbols, the failing fixture, the expected
error text, and the validation commands here.

</details>
```

To close a ticket, the Architect removes its one index line, moves its complete
detailed section below `# Closed tickets`, changes `**OPEN.**` to
`**CLOSED.**`, and changes `What is missing` to the exact sentence `Nothing
for this ticket.` The title, anchor, type, final priority, reopen count, human
summary, completed work, and technical evidence remain. The Architect then
commits the accepted Implementer fix without waiting for Red Team. If any
required action remains, the ticket stays open or that action receives its own
linked open ticket.

At the end of each cycle, Red Team reviews the tickets closed during that
cycle. A no-finding result is advisory and changes nothing. If the bug remains,
the handoff says `Backlog action: REOPEN`. The Architect does not stop to audit
or reproduce the bug: immediately increment the reopen count, apply the
greater-than-five Low rule, move the full section back to the matching open
priority group, restore its index line, change `**CLOSED.**` to `**OPEN.**`,
replace `Nothing for this ticket.` with the concrete reopened work, and cite
the stable finding note with the exact `See further instructions at ...` line.
Record that evidence and priority will be assessed only when the ticket later
reaches the front of its priority group.

A new Red Team discovery uses the exact handoff label `Backlog action: NEW
TICKET`. On receipt, the Architect performs the same short bookkeeping step:
create the complete human-readable ticket, use the Red Team's High, Medium, or
Low assessment as a provisional priority, acknowledge the return, and record
that Architect analysis remains. It also cites the stable finding note with
the exact `See further instructions at ...` line. The Architect does not
reproduce the bug merely to add it. When priority later brings the ticket
forward, the Architect uses the detailed note and targeted independent
verification to accept, upgrade, downgrade, close, or reject it with evidence.
Only the Architect can assign Critical.

Every ticket section has these parts in this order:

1. **High-level summary** gives at least three complete sentences in ordinary
   language: normal purpose and one concrete example, current failure, and the
   user or scientific consequence. More sentences are allowed when a reader
   needs them. An internal unit number may follow a plain title, but it never
   replaces that title.
2. **Current status** says exactly `OPEN` or `CLOSED`, records `Bug fix` or
   `New functionality`, gives its priority reason, records the nonnegative
   Red Team reopen count, and names any blocker or prerequisite.
3. **What is already fixed** names completed work without implying that it
   closes the ticket.
4. **What is missing** names every action, machine run, review, or decision
   still required. A closed ticket says `Nothing for this ticket`; separate
   unfinished work must have its own linked open ticket.
5. **Technical record for development tools** retains exact files, symbols,
   commits, branches, evidence counts, failure cases, and source-note anchors.

The Architect applies this decision table whenever a ticket is added or
updated:

| Check | `GO` | `NO-GO` |
| --- | --- | --- |
| Human title | Names the problem in words a physics student can understand; an internal ID is secondary | Uses only `unit 8`, an acronym, a gate ID, or another internal label |
| Human summary | Gives normal purpose with a concrete example, the current failure, and its consequence in at least three complete sentences | Starts with commits, evidence counts, internal stages, or unexplained software language |
| Status | Appears in the correct Open or Closed section and agrees with the linked `- OPEN` index | Is missing, contradictory, or described as closed while required work remains hidden in prose |
| Partial work | Separates completed work from missing work | Treats a landed partial fix or local test result as ticket closure |
| Ticket type | Records Bug fix or New functionality and applies its ordering rule | Omits type, labels a feature Critical, or lets a feature bypass a higher-priority item |
| Severity | The Architect records Critical, High, Medium, or Low from concrete harm and likelihood, and the index places the ticket in that priority group | Severity is omitted, copied from Red Team without review, Critical is used without proof that High is insufficient, or a ticket is ordered below a lower-severity ticket without a recorded blocker |
| Reopen count | Uses one canonical nonnegative integer, starts at zero, never resets, and increments for every formal Red Team `REOPEN` assessment | Omits the count, uses prose instead of an integer, resets it, or loses a Red Team reopening return |
| Repeated reopening | Immediately restores every Red Team `REOPEN` return, then later compares new evidence with earlier attempts; a count above five forces Low | Delays the bookkeeping for a full audit, calls every repeated objection obnoxious without evidence, or keeps a priority above Low after the sixth attempt |
| Red Team authority | Red Team advice never blocks Architect acceptance, closure, commit, or landing | Requires a Red Team GO, delays an accepted commit for Red Team, or lets Red Team edit the backlog |
| New Red Team ticket | The handoff says `Backlog action: NEW TICKET`; the Architect records it promptly with provisional Red Team priority and analyzes it later | The finding waits outside the backlog while the Architect performs a full audit, or another role writes the backlog directly |
| Red Team source note | A `NEW TICKET` or `REOPEN` cites one stable repository-relative finding note with every persuasive-note heading and the backlog preserves the exact `See further instructions at ...` line | Uses an absolute or dated path, omits the note citation, gives a thin assertion, hides uncertainty or exclusions, inflates severity, or invents evidence |
| Technical detail | Preserves exact evidence in the technical record after the human explanation | Removes evidence or makes a human decode it before learning the problem |
| Closure | The Architect accepted the Implementer fix, every required action passed, and `What is missing` says nothing remains for this ticket | A required hardware run, scientific check, Architect decision, merge, or note update remains |
| Open-count check | The number of linked `- OPEN` index lines equals the number of detailed open ticket sections | The watcher count can omit, duplicate, or point to a missing ticket |

Malformed backlog state always fails closed. This includes an `- OPEN` line
that does not match the exact grammar, an unknown priority or type, a Critical
feature, a line under the wrong priority heading, a duplicate or missing
anchor, a link without one detailed section, an unlinked detailed open
section, contradictory `OPEN`/`CLOSED` text, a missing or malformed reopen
count, or priority groups in the wrong order. The Architect repairs the
structure before dispatching that ticket,
admitting discovery, or claiming that the backlog is complete. A malformed
line is never ignored, guessed, or rewritten as Low merely to make a count
smaller.

#### Protect the Architect-owned backlog

Only the Architect edits `ai/notes/backlog.md`. The ignored file
`ai/notes/.backlog-guard.json` stores the SHA-256 fingerprint of the exact
backlog bytes that the Architect last accepted. The fingerprint detects an
unexpected character change; it does not prove that the ticket description is
correct.

After creating and reading a new backlog, the Architect initializes the local
record:

```bash
python3 ai/tools/backlog_guard.py initialize --architect-ack
```

Before accepting another role's return or changing any backlog ticket, the
Architect runs:

```bash
python3 ai/tools/backlog_guard.py check
```

The Architect copies the printed 64-character `accepted SHA-256` before the
edit. A mismatch is `NO-GO`: stop, inspect the unexpected change, and do not
replace the fingerprint merely to silence the warning.

After one deliberate backlog edit, the Architect reads the changed ticket and
then records those exact bytes:

```bash
python3 ai/tools/backlog_guard.py seal \
  --previous-sha256 SHA256_FROM_THE_PRE_EDIT_CHECK \
  --architect-ack
python3 ai/tools/backlog_guard.py check
```

Mailbox Architect turns receive `MAILBOX_ROLE=architect`, so the write
commands recognize the role even when the manual acknowledgement option is
omitted. Manual terminal use keeps `--architect-ack`. Implementer and Red Team
turns receive non-Architect values. They may run `check`, but they never edit
the backlog, run `initialize` or `seal`, or edit `ai/tools/backlog_guard.py`,
the fingerprint record, or its `.backlog-guard.lock` write lock.

The backlog, fingerprint record, and lock stay outside Git. An incoming
backlog package is inspected in its separate import folder; it never replaces
the live backlog or its fingerprint automatically. This guard is intended to
catch accidental role edits and hallucinated replacements. A malicious
program able to rewrite both the backlog and guard is outside this limited
protection.

A workstation-only check stays open when it is required for acceptance. If a
large ticket is split, each follow-up either remains under the parent ticket's
missing-work list or becomes its own linked open ticket. A closed section may
mention a limitation outside its scope only by linking to the open ticket that
owns that work.

The Architect updates the ticket in the same turn as every state change,
including dispatch, returned evidence, GO or NO-GO, landing, a Red Team
`REOPEN`, a `NEW TICKET` return, and a new or cleared blocker. The ticket stays OPEN until
implementation, required evidence, Architect review, landing, and any required
permanent-note update are complete. Architect acceptance then closes and
commits it without waiting for Red Team; a later advisory review may reopen it.

The Architect note has one current `## Implementation directive`. A confirmed
Red Team return has one current `## Repair directive`. The appropriate
contract checker validates the packet structure before transport. Structural
validation does not replace scientific review.

A handoff is a compact routing summary that cites the source note. The source
note remains authoritative when a summary lags or differs. Files under
`ai/notes/relay/` are immutable transport copies for traceability. They are
not evidence and are not edited.

Mailbox files live under `ai/notes/mailbox/`. A numbered file is dispatched to
an internal role and then archived under `done/`. Public commands do not expose
those internal destinations. A `to-user` status file is not dispatched. A
terminal inbound that explicitly says no reply is owed does not require an
artificial receipt. This is the only
outbound exception; ambiguity requires an outbound response.

In two-role mode, Architect and Implementer communicate directly through the
mailbox and no Sol message is created. Existing Sol messages remain untouched
until a normal three-role watch handles them.

One completed **work period** means that five started role turns have finished,
or that 15 minutes have passed since the period began. At either boundary,
the watcher stops starting new work, lets every job that is already starting
or running finish, and opens the 20-second manual stop countdown. An ordinary
mailbox check with no such boundary does not complete a work period.

Cycle settings control planned stopping:

- with no `--cycle` option, the watcher continues watching;
- `--cycle N`, where `N` is positive, stops safely after `N` completed work
  periods even when recorded work remains; and
- `--cycle 0` exits only after enabled mailbox routes are idle and no local
  backlog index line begins with the exact marker `- OPEN`.

Cycle zero also requires a safe, stable backlog read. A missing, non-regular,
changing, unreadable, oversized, or non-UTF-8 backlog prevents exit and
reports that completion could not be verified.

Backlog prose never creates a mailbox request. The separate 20-second
interrupt countdown is an occasional manual safe-stop opportunity; it is not
the definition of a cycle. Fix-only mode permits work that closes an existing
ticket but refuses discovery and every request to create a new ticket.

### Discovery demand and a second Implementer

Two separate severity counts control two separate decisions:

1. **New-discovery admission.** Count open Critical, High, and Medium backlog
   tickets. Ten or more stops new discovery so accepted work can be closed.
   Open Low tickets and waiting mailbox files do not count. An unclassified
   open line fails closed until the Architect repairs its classification.
2. **Second-Implementer emergency.** Count Critical bugs and High bugs
   separately. Sol may become a second Implementer only when more than one
   open bug is Critical or more than ten open bugs are High. High features,
   Medium work, Low work, and waiting mailbox files do not contribute.

Queue depth remains useful status information, and every open ticket,
including Low, still prevents a `--cycle 0` run from claiming that all work is
finished. Neither threshold changes Sol's role automatically. During a proven
emergency, the Architect must still assign Sol for one named ticket in a
validated directive. Sol then follows the Implementer contract and cannot
audit the same ticket.

Only the Architect decides whether an accepted change alters a permanent
general property. Permanent notes are not edited by an Implementer or Red
Team. Routine milestones do not create permanent-note churn.

## Landing and branch discipline

Without an explicit grant, the user performs commit, merge, and push. A
daemon-dispatched Architect has the narrow standing grant to create and push
one squash commit for one audited GO result after checking for foreign
commits. No other role inherits that authority. Only the Architect uses the
main-landing lock. Only `main` is pushed; working branches remain local.

After a squash lands on `main`, the working branch merges `main` locally:

```bash
git merge main
```

`--ff-only` is incorrect because the squash and fine-grained branch histories
intentionally diverge even when their trees match.

## Environment assumptions

The lightweight development machine may provide only Python, NumPy, and the
standard library. Evidence there consists of compilation, AST censuses,
docstring-stripped AST comparison, and known-answer arithmetic probes against
the real function body where possible. Torch, CosmoLike, Hierarchical Data
Format version 5 (HDF5), YAML, SciPy, Matplotlib, and accelerator evidence run
in the configured Cocoa environment.

Apple Metal Performance Shaders (MPS) does not support device float64 and uses
float16 autocast. CUDA, NVIDIA's accelerator-computing platform, provides the
required compiled and
accelerator checks. Set `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES` before
process startup. The production system uses task-parallel processes, not
distributed data parallel (DDP), which replicates a model across workers, or
threads: spawn, not fork; one device selection per worker; no private copies
of the full random-access memory (RAM) payload in parallel paths;
longest-processing-time assignment; and retained Queue/Lock references until
every child joins.

The configured CoCoA environment uses NumPy 1.x. An isolated code,
documentation, or dependency change must not adopt NumPy 2 behavior. A NumPy
2 migration requires an explicit project-wide decision and validation across
the emulator families, data generators, inference adapters, tests, and gates.

`ROOTDIR` is defined by the Cocoa startup process. Repository paths anchor to
that value, and `cobaya-run` starts from `ROOTDIR`. Public installation
instructions point to Cocoa's official README instead of duplicating its
environment procedure.

## Recurring evidence rules

- Paste complete raw scan output into the working record. A summary is not a
  substitute.
- Read the recorded Git `HEAD`, the commit currently checked out, before
  interpreting a stored test failure.
- Build a fixture from the shipped YAML or source schema rather than retyping
  coupled keys from memory.
- Derive all coupled fixture widths from one named value.
- Resolve Cocoa-relative theory paths from `ROOTDIR`, including in-process
  model construction.
- Carve out a physical exception on the physical axis, not on an unrelated
  configuration label.
- When a hypothesis about a third-party mechanism fails on the real machine,
  switch to its documented application programming interface (API), the
  supported set of calls exposed to this repository, and add a tripwire
  capable of falsifying the replacement assumption.
- A search supporting “no match exists” must be untruncated. Count or inspect
  all matches, search the synonym set, and record the pattern and scope.

### Tests, gates, and the validation board

A **test** asks one narrow question. For example, a CMB progress-file test
shifts one saved multipole coordinate and checks that loading refuses the
mismatch before reading spectra.

A **gate** is a named final check for a larger requirement. It may run one
test, several tests, a scientific comparison, or a hardware-dependent job.
A passing narrow test does not replace a gate required by the Architect.

The **validation board** is the ordered registry of gates and the raw machine
evidence saved for each run. The Architect reads the board and the actual
output before deciding GO or NO-GO. Evidence that cannot run because required
hardware or data is absent is recorded as unavailable; it is never converted
into PASS. Command inventories and current gate membership belong in
`ai/tests/README.md` and `ai/gates/README.md`, not in this permanent note.

## Self-teaching generator entry files

Each production generator entry file contains:

1. a module docstring naming the product and physics engine;
2. a short flow diagram from sampled row to stored family payload;
3. a local description of the shared-core and family-specific ownership;
4. formal argument, return, raise, shape, unit, dtype, and ordering contracts
   for the physics callback;
5. an explanation of provider component ordering, dependency parameters,
   caching, and captured output;
6. a storage-hook contract stating allocation, mutation, append, load, and
   copy/view behavior; and
7. a runnable command or direct link to the exact family guide.

Acceptance requires module docstrings in all generator siblings, formal
contracts on every nontrivial override, a generated callback inventory with
no undocumented callback, and successful syntax compilation.

## Current-state API explanations

Loss decoding returns the kept-coordinate vector. It inverts the numerical
transform and does not restore masked positions. Full-vector reconstruction
is a separate `geometry.unsqueeze(kept)` step. Every loss subclass and caller
must preserve that distinction; equality of kept and full widths in a
diagonal family does not redefine the general contract.

Weight decay is selected by module role, not tensor rank. Only `.weight` from
`Linear`, `Conv1d`, and `BinLinear` is decay-eligible. All other parameters
remain undecayed unless the allowlist is deliberately expanded.

Geometry encode or whiten operations divide by scale or sigma. Decode or
unwhiten operations multiply. Errors and comments must name the correct
direction.

Automatic mixed precision (AMP) runs selected operations at a lower numeric
precision to reduce accelerator cost. AMP documentation distinguishes float16
on MPS from bfloat16 on CUDA or CPU.

## Teaching the experiment lifecycle

The experiment class boundary includes one lifecycle diagram:

1. resolve paths;
2. validate exactly one family;
3. choose a model class;
4. stage training and validation data;
5. construct parameter and output geometries;
6. construct the loss;
7. build model, optimizer, and scheduler specifications;
8. train; and
9. persist the result.

Every stage names its input, created instance attributes, eager or deferred
work, and state reused by sweeps. A family decision table records scalar,
cosmic microwave background (CMB), grid, grid2d, and CosmoLike differences.
Define `classmethod`, `cls(...)`,
`**kwargs`, capability flag, cached state, and alternative constructor before
using those terms. A long method that still owns several independent state
transitions is split into named cold-path helpers, with compile, binding,
leftover-pattern, and behavior checks.

Warm-start and transfer documentation includes a concrete named-column
example, exact encoded column order, input-weight shapes, copied and zeroed
columns, view/copy ownership, and the meaning of `torch.no_grad`. Packed
targets are shown with shapes. Parity is an executed epoch-zero equality check
with coordinate system, dtype, device, and tolerance stated. An unavailable
feature is described as unavailable and refused rather than promised through
unreachable code.

Gate files begin with the exact behavior they require, one real input and
visible result, their dependencies, and why a failure blocks acceptance. A
nontrivial check documents the system under test, fixture, independent
expected answer, and deliberate mutation. Terms such as
fixture, test double, fake, stub, monkeypatch, known answer, control, mutation,
and catch power are defined before use. A numerical reference cannot be
computed by the same helper as the value under test.

## Stable workflow evidence anchors

<a id="board-selftest-exit-truth"></a>
**The board runner reports what actually ran.** Unknown or conflicting
selectors, dependency skips, compile-lane skips, stale or edited logs,
unresolved anchors, duplicate assertion identifiers, and malformed evidence
all produce a non-green result. A stored pass is reusable only while its raw
log and digest remain intact.

<a id="cli-strict-strict-parse"></a>
**Every public executable rejects a misspelled flag.** Public entry points use
strict argument parsing. Representative drivers reject `--activaton` before
expensive work while a valid command reaches the intended boundary.

<a id="family-first-family-owned"></a>
**Every driver owns exactly one data family.** The cosmic-shear driver owns the
CosmoLike data-vector family and rejects scalar, CMB, grid, and grid2d YAML.
Family wrappers accept only their own family block. A source census verifies
the pinned family and strict check in every wrapper.

## Documentation ownership

The Architect decides what tracked documentation must change, writes a
detailed directive for that change, and reviews the rendered result. The
Implementer may edit a README or explanatory Python prose only when the
Architect's bounded directive names the exact section or symbol. The Red Team
may report a README or Python-prose defect but does not edit those artifacts.
TeX source under `documentation/` has separate Red Team ownership defined in
`CLAUDE.md`; that narrow ownership does not extend to READMEs, Python prose,
or permanent notes.

A behavior change that affects a “Current gap” paragraph names that paragraph
in the Architect source note. The directive requires the Implementer to
rewrite the paragraph to current behavior or narrow it to the remaining
limitation. A stale gap is a documentation defect. Permanent notes remain
Architect-only under [`MEMORY.md`](MEMORY.md), even when a documentation unit
is active.

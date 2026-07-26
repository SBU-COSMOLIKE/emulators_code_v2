# Conventions, workflow, and environment

Mandatory repository-wide conventions: current rules, not their history. A
change is **GO** only when the relevant rule and its acceptance evidence are
satisfied. A contradiction, missing proof, or undocumented exception is
**NO-GO**.

`ai/notes/python-changes-go-no-go.md` is the binding GO/NO-GO contract for
every Python change, read before the directive and again before accepting the
result. This note is context; it never weakens that review.

## Words with a local meaning

Ordinary Git, testing, and process vocabulary is assumed. These four are local:

**Watch**: one running mailbox command that repeatedly checks saved messages and
starts the enabled roles. **Ticket**: one bounded work request controlled by one
Architect source note. **Landing**: putting an accepted ticket commit on `main`.
**Catch power**: a gate's demonstrated ability to fail for the mutation that
deliberately restores the behavior the gate forbids.

## Python house style

Applies to `emulator/`, public drivers, checks, and support scripts.

- Lines at or below 90 columns. Align continuations with the opening
  parenthesis when practical; otherwise one consistent two-space hanging
  indent, one item per line.
- Pass arguments by name whenever the callee permits. Keep genuinely
  positional interfaces positional — mathematical operands, plotting
  coordinates, `model(x)`, `*args` forwarding — and add a naming comment when
  a positional tensor is not obvious.
- Explicit loops outside performance-critical code; vectorized NumPy or Torch
  operations stay inside compiled, forward, or batch hot paths. Find
  comprehensions by AST; text search is not enough.
- Direct, C-readable control flow: no nested comprehension, no lambda where a
  named function reads better, no walrus, starred-argument trick, or stacked
  conditional expression. One readable conditional expression is acceptable.
- Never read mutable module-global data silently inside a function; pass it
  explicitly. A necessary exception carries
  `# WARNING: reads module global NAME` at the read site.
- Constructible components are `{"cls": class_object, ...kwargs}` dictionaries.
  A `make_*` helper injects computed, device, and runtime values; those never
  belong in the reusable specification.
- Ordinary sentence case; no all-capital emphasis. Acronyms, interface
  literals, and the `WARNING` marker keep their required case.
- No spaced double dash as prose punctuation, including in command help,
  errors, logs, comments, and docstrings.

Teaching-notebook formatting, a read-only reference with narrower lines, never
relaxes these rules.

## Explanatory Python prose

Code teaches the current program; it never narrates a review history.

- Module docstrings use complete sentences with a subject and verb.
- Public function and nontrivial private function carries an `Arguments:` block
  naming each argument and a `Returns:` block, plus a `Raises:` block for
  meaningful refusals. For a dictionary argument, enumerate accepted keys,
  shapes, units, and meanings.
- Short private callback or test double: one sentence, when a formal block would
  only repeat the signature.
- Define a technical term at first use or replace it with plain language; a file
  with several may carry a short local glossary.
- Explain a cross-module call with a provenance comment when ownership is
  unclear: `# function_name (module.py): current purpose`.
- Write mathematical relationships as formulas with every symbol defined.
  Tensor pipelines need a shape-flow diagram and a legend defining every
  dimension.
- Derive constants from named symbols. An LSST-Y1 example may follow the
  general derivation but never replaces it.
- Never state a list length, key count, or family count without checking the
  source of truth. A schema change requires a complete census for stale counts
  and enumerations.
- Prove a documentation-only Python change by comparing ASTs with docstrings
  removed. Prose is not evidence of no executable change.

Domain symbols never collide with established cosmology notation: `h` is
reserved for the dimensionless `H0 / 100`. The covariance finite-difference
control is `step_frac` in Python and `s_step` in prose. This applies to code,
formulas, logs, comments, notes, and handoffs.

Covariance checks use the Planck-LCDM fiducial in
`example_yamls/cmb_covariance_lcdm.yaml` or a justified nearby cosmology. An
extreme synthetic case proves a validator catches bad input; it never alone
proves a scientific result wrong.

Runtime validation never depends on `assert`. Public configuration, data,
shape, geometry, and numerical guards use explicit typed exceptions before
mutation or accelerator setup. An optimized-mode subprocess must reject the
same negative fixtures with the same messages as ordinary Python. An internal
invariant also uses an explicit exception when continuing could publish a
scientific result.

Internal tracking abbreviations and review codes belong only in temporary
working notes; public README files, Python prose, errors, logs, YAML comments,
and check labels state the underlying fact. A permanent note may be cited by
path. A repository-wide leak scan checks both coded forms and bare
abbreviations and reads the complete output.

## Scope of scientific review

Review covers scientific correctness, reproducibility, model and data identity,
stale-test truth, numerical stability, and publication integrity.
Cybersecurity, hostile-user threat models, secrets, network attacks, and
exploit hardening are out of scope unless explicitly requested or directly
required to protect scientific results.

## README and teaching contract

`ai/notes/readme-go-no-go.md` is the binding review contract for README text,
comments, docstrings, command help, errors, logs, and explanatory strings, read
before the directive and again before the final decision.

The root README teaches how to run and configure the library first; detailed
design belongs in separated appendices or specialist README files. Define a
concept before using it. Every explained YAML concept carries a short fenced
example copied from the real schema. Point to one authoritative explanation
instead of restating it.

README files describe the current library — no development dates, review
rounds, queue state, landing state, abandoned formulas, or biographical
commentary. A current limitation may remain only as:

1. the present scope;
2. the consequence for the user; and
3. the action the user should take.

README and explanatory Python prose present one coherent current system. They
do not label a passage `hard user rule`, attribute policy to a user, or stack a
new correction beside the older rule. `readme-go-no-go.md` owns the complete
wording and subject-matter exceptions.

Parentheses hold only a short local definition, symbol, unit, or acronym. If
removing a parenthetical changes an essential instruction, promote it to a
sentence, table row, or diagram label. Review parentheticals over twelve words
or with more than one clause.

GitHub mathematics:

- no backslash command immediately followed by ASCII punctuation inside math;
- no LaTeX environments in Markdown math;
- no line-initial Markdown token inside a display-math block;
- no whitespace-adjacent inline dollar delimiter; and
- no code-name underscore inside math unless it is valid mathematical syntax.

README acceptance includes a complete anchor and backticked-path census: every
link target resolves, every named path exists.

## Plots, terminal output, and YAML

- Never combine red and green as the distinguishing plot colors. Use the
  colorblind-safe palette `#0072B2`, `#E69F00`, `#CC79A7`, `#000000`, and
  `#56B4E9`; `viridis` for continuous maps; vary line style for grayscale.
- Terminal output is a dashboard: short header, current result, one-line
  detail, product paths. Complete streams go to immutable per-run logs; a debug
  option may mirror the full stream.
- YAML uses block style, one key per line, no inline mapping. Preserve
  established value-column alignment. Range leaves use
  `[default, minimum, maximum, kind]`.
- Report every YAML change as a paste-ready block with enough surrounding
  context to locate it.

## User-facing role boundary

The user communicates only with the Architect; public mailbox commands accept
only the `architect` destination. Implementation, review, severity, model
choice, a widespread search, corrections, and changed scope all go there.

The Architect decides which enabled role acts next and writes the complete
downstream instruction. The Implementer and Red Team do not accept direct user
substance; a direct request reaching either is returned to the Architect as a
blocker. A human may copy a generated handoff unchanged — transport, not a new
user instruction.

The default topology is Architect, Implementer, and Red Team. A watch may omit
Red Team with `--skip-redteam` or `--no-red-team`, which never weakens
Architect planning, evidence review, or exclusive GO/NO-GO authority.

Model choice and role choice are separate. Sol has a separate effort setting;
there is no independent Red Team model option. Provider and model options never
change role authority, Git worktree ownership, mailbox route, or evidence
requirements.

Only the Implementer edits source code, tests, or ordinary tracked
documentation for a ticket. The Architect writes plans, maintains the tracked
backlog and permanent notes, audits named commits, and records GO or NO-GO. The
parent daemon performs the controlled landing after an Architect GO. The Red
Team writes findings and audit returns. Architect and Red Team audits read an
immutable commit by its full hash, never the Implementer's changing working
folder.

The roles have independent runtime lanes: Implementer in the implementation
worktree, Architect in the coordination worktree, Sol in the Red Team worktree.
With another unused ticket slot in the cycle limit, the Implementer may code
ticket B while the Architect audits ticket A's immutable candidate C and the
Red Team reviews an earlier daemon-recorded landing L. The overlap never
combines tickets: each keeps its own base, commit, messages, and one-cycle
count. The parent daemon uses the landing lock and never borrows the
Implementer's working folder. Fable never edits the user's checkout or runs the
merge, commit, reference-update, or push command for a ticket landing.

The Architect's source note is the authority for role topology and discovery
severity; manual router options only confirm it. A disagreement refuses before
any lock, clipboard, archive, or mailbox write. A detailed Architect directive
includes:

- exact worktree, branch, and base;
- one `path::symbol` edit target for every owned file or test;
- ordered edits and named interfaces;
- types, shapes, algorithms, and numerical invariants;
- failure behavior and forbidden alternatives;
- named tests with expected observations;
- exact validation commands;
- stop conditions;
- non-overlapping ownership when work is divided; and
- the Architect's decision to require bounded subagents or explain why a
  separate helper would add no independent value.

The instruction must be complete enough for a simple Implementer to execute
without inventing design decisions. A design-sensitive gap is a blocker: the
Implementer reports the exact missing fact and waits for a revised directive.

Only the Architect decides whether subagents add independent value. When a
ticket divides into an independent reproduction, implementation, test,
documentation, or audit job, the Architect requires those bounded helpers; the
Implementer integrates their work, reviews every changed file, and runs the
final validation. Otherwise the Architect records why a separate helper would
repeat the same indivisible work or evidence, and the Implementer repeats that
reason verbatim without creating or rewriting the waiver. Cost, convenience, or
"small ticket" alone is not sufficient.

`handoff_contract.py` rejects an informal sentence such as "use helpers where
useful." One of two visible forms is required. With independent work:

```markdown
#### Subagents required
- Launch: `required before implementation edits`
#### Subagent `failure-reproducer`
- Mode: `read-only`
- Ownership: `none (read-only)`
- Task: Run the named malformed-message test before any source edit.
- Return: Return the exact command, exit code, and failing assertion output.
- Acceptance: The output shows the expected pre-edit parser failure.
- Stop: Stop if the standard-library test cannot start.
#### Integrator
- Integration: Launch every helper before the Implementer begins its own owned edit. Let non-overlapping work run at the same time. Review every return before integrating helper work and before final validation.
- Final validation: Run `python3 -m unittest ai.tests.test_handoff_contract` and require exit zero.
```

When one edit and its assertion cannot be divided without duplicating the
same inspection:

```markdown
#### Subagents not required
- Reason: The complete edit and its assertion share one parser branch; a separate helper would repeat the same inspection without producing independent evidence.
```

An editing helper uses `Mode: edit` and owns exact, backticked
`repo/path::symbol` entries. One editing helper owns the whole file; two
helpers may not claim different symbols in the same file, because their edits
could still collide.

A capability exception is never guessed in advance. If the Implementer attempts
the named launch and the runtime rejects it before editing, it marks that
helper `blocked` in the same-cycle `IMPLEMENTER_HANDOFF`, recording the exact
`Capability checked`, `Attempted operation`, and `Raw failure` values from that
first rejected pre-edit launch as the final rows of its `Subagent work`
evidence. The relay records the full current cycle and SHA-256 digest of that
complete blocked handoff. The Architect copies those three digest-bound rows
character-for-character into the replacement plan, and both binding rows plus
the same failure evidence under this required sibling block:

```markdown
### Prior Implementer subagent launch failure

- Source cycle: `ticket-anchor@0123456789abcdef0123456789abcdef01234567`
- Source handoff SHA-256: `0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef`
- Source: `prior same-cycle IMPLEMENTER_HANDOFF checkpoint`
- Capability checked: `exact.launch.operation`
- Attempted operation: Launch the named subagent through the advertised operation before implementation edits.
- Raw failure: `the unchanged runtime error`
```

The relay verifies both binding rows and all three copied failure values
against the saved handoff before the revised plan can run. A missing,
paraphrased, normalized, or invented value refuses the exception, as does a
speculative or stale-cycle claim. The Architect revalidates and sends that
revised same-cycle directive. A truthful `blocked` return serves this
checkpoint only: unresolved blocked work never supports final `GO`, and every
helper in the final ordinary plan must return `pass`.

When enabled, Red Team reviews the named change and directly affected behavior.
A repository-wide attack happens only when the Architect records an explicit
user request such as "instruct the Red Team to do a widespread search for ...".
A confirmed finding returns to the Architect with root cause, exact symbols,
ordered candidate edits, invariants, a regression witness, commands, acceptance
checks, exclusions, and stop conditions. Red Team never sends repair
instructions directly to the Implementer.

The Architect audits raw evidence, not summaries. A harness is checked first
against a known-good case, then against a deliberate mutation. Only the
Architect writes the final GO/NO-GO record.

### Discovery severity

Discovery severity controls which newly found defects may become tickets. It
does not change the scope of a named-change review.

- **Critical**: not a user discovery setting, not a Red Team rating. Only the
  Architect assigns it, and only on evidence that a current defect broadly
  breaks a central library workflow or systematically invalidates the library's
  scientific results.
- **High**: severely damages core behavior, loses data, halts normal operation,
  or makes a primary scientific result wrong. Rules below.
- **Medium**: High defects plus concrete defects reasonably likely in normal
  use. A merely theoretical or very improbable edge case does not qualify.
  Medium is the default.
- **Low**: every concrete defect, including an improbable edge case.

**Low — Edge Case** is not a discovery severity. It is the parked remainder of
a bounded repair whose actionable failure is gone and whose complete coverage
would add disproportionate complexity. No command-line severity selects it;
only an explicit user request naming that exact parked ticket activates it as
ordinary Low work.

Harm and likelihood are separate judgments. Red Team reports severity,
likelihood, impact, scope, and evidence; the Architect accepts, upgrades, or
downgrades with a reason and alone decides whether the finding becomes a
ticket.

High is deliberately difficult to assign, bar below Critical. Red Team and
Architect must name
the damaged primary calculation, training data, served result, data-loss
boundary, or core operation, plus why Medium is not enough. "Wrong science"
alone does not satisfy that comparison. Harm ending in a plot, diagnostic,
ranking, or optional analysis product is Medium unless separate evidence shows
the same defect changes a primary result or stops a core workflow. Urgency, a missing test, unfinished cleanup, an expensive validation run, or a
desire to work sooner is not by itself High evidence. A missing comparison is NO-GO and defaults to
Medium until evidence supports an upgrade. Severity never selects a role or
changes the number of Implementers.

The Critical bar is far above High. High, urgent, scientific, hard to fix,
limited to one important family or platform, or lacking a convenient workaround
is not Critical. Before assigning Critical the Architect records why High is
insufficient and the exact evidence for broad library breakage. Never promote a
ticket to Critical to change the number or kind of active roles.

A High discovery setting does not authorize a repository-wide search. Critical
is not accepted by `--severity` or `MAILBOX-SEVERITY`. Fix-only mode, an
omitted Red Team, and the discovery-admission limit take precedence.

The user's explicit phrase "do a widespread search" creates a special Low
discovery request with saved mailbox severity Low. The Architect does not send
it while any accepted Critical, High, or Medium ticket is open; Low tickets do
not block it. A broad search for optional findings must not delay known non-Low
work.

### Ticket character limit

`--max` limits the complete committed change for one ticket. A positive limit
counts added plus deleted characters as Unicode code points from the ticket's
bound full base commit to a clean `HEAD`; replacing text counts both removed
and added text, across every tracked code, test, and documentation file in that
ticket.

An exact-boundary result is accepted. `--max 0` removes the numeric ceiling
only; it never weakens scientific correctness, completeness, tests,
documentation, or readability. If a complete readable fix cannot fit, the
Architect returns NO-GO and asks for a smaller ticket or a changed limit.

## Persisted agent worktrees

Ordinary agent work never occurs in the user's repository checkout. The mailbox
system owns three persisted worktrees. `<REPO_ROOT>` is the top folder of the
checked-out emulator repository:

| Resource | Required value |
| --- | --- |
| Architect coordination name | `mailbox-primary` |
| Architect worktree | `<REPO_ROOT>/.claude/worktrees/mailbox-primary` |
| Architect branch | `refs/heads/claude/mailbox-primary` |
| Architect state | `<REPO_ROOT>/.claude/worktrees/.mailbox-primary-worktree.json` |
| Implementer worktree name | `mailbox-implementer` |
| Implementer worktree | `<REPO_ROOT>/.claude/worktrees/mailbox-implementer` |
| Implementer branch | `refs/heads/claude/mailbox-implementer` |
| Implementer state | `<REPO_ROOT>/.claude/worktrees/.mailbox-implementer-worktree.json` |
| Sol worktree name | `mailbox-sol` |
| Sol worktree | `<REPO_ROOT>/.claude/worktrees/mailbox-sol` |
| Sol branch | `refs/heads/codex/mailbox-sol` |
| Sol state | `<REPO_ROOT>/.claude/worktrees/.mailbox-sol-worktree.json` |
| Bootstrap lock | `<REPO_ROOT>/.claude/worktrees/.mailbox-primary-worktree.lock` |

Claude-owned branches begin with `claude/`, Sol-owned with `codex/`. The older
`worktree-agent-*` form is reserved only so the explicit user command
`--clean-all` can recognize and remove it; ordinary recovery never invokes that
destructive command or discards a worktree.

Changing a model option never selects a different worktree. Only the
Implementer lane edits tracked source. The Architect audits a detached snapshot
of the exact candidate commit, Sol a detached snapshot of the exact
daemon-recorded landing L. Neither review follows the Implementer's moving
branch.

The primary Architect worktree's `ai/notes/` is the shared coordination
location for mailbox files, relay copies, the tracked backlog, and temporary
records. Other roles receive explicit access there and must not create another
active mailbox or backlog in their own worktrees.

Each state record stores the canonical Git common directory, stable name,
absolute path, and full branch reference, checked against
`git worktree list --porcelain` on every reuse. Before touching the mailbox the
launcher re-executes the saved primary worktree's current daemon with the
original arguments, interpreter, and working directory. The saved topology
marker must also prove Sol has a dedicated worktree.

CLI validation happens before worktree provisioning. Help, preview with no
action, invalid combinations, and dry-run create no branch, worktree, state, or
lock. Mailbox actions are `--watch`, `--once`, and
`--send architect`. The separate `--ping` makes one small direct request to
Claude and Sol without writing a mailbox file; `--ping --skip-redteam` checks
Claude alone.

On a clean clone, establish the primary worktree with one valid live action
before writing an uncommitted source note: a new worktree starts from committed
local `main` and cannot see an uncommitted note in another checkout.

Legacy adoption is deliberately narrow. A current, attached, non-main worktree
under `.claude/worktrees/` may be adopted only when the first live command
starts from that same worktree and no conflicting active transport exists
elsewhere. Active, ambiguous, duplicated, or pre-migration transport is never
copied, merged, renumbered, or deleted. A unique main-checkout store holding
only completed messages and regular logs may be copied byte for byte under both
transport locks, bounded to 16 MiB per file and 64 MiB total; partial identical
copies resume, conflicting bytes refuse.

An interrupted clean bootstrap resumes only when the exact default path,
branch, and Git registration validate. A uniquely registered `git worktree
move` may update the saved path after full validation. Detached branches, wrong
branches, deleted refs, manual directory moves, corrupt state, prunable
worktrees, and unregistered branches refuse without fallback.

Ordinary recovery never resets or prunes the user's checkout or a dirty,
unverified persistent role folder. It may reset only the verified clean
Implementer lane to the exact commit saved for that cycle, and may prune Git's
stale registration only after verifying and removing an unchanged disposable
audit snapshot. It does not stash, clean, fetch, pull, or invent a replacement
worktree. The one bounded exception is the parent daemon's post-GO landing
below: fast-forward a verified clean user `main` checkout to an already
prepared exact landing and attempt a non-force push. Recovery starts by
preserving the state and transport paths and comparing them with Git's
registered worktrees.

## Notes-first communication and mailbox transport

The substantive record for a ticket is a local temporary note under
`ai/notes/`, written before the handoff. It contains scope, scientific
evidence, counterexample, design contract, exact file and symbol targets,
changed files, branch or commit identity, raw-test locations, remaining
obligations, and acceptance conditions.

### Red Team finding note GO / NO-GO

Red Team is always advisory, including when its model is more capable than the
Architect's or Implementer's. It finds defects and proposes evidence; it cannot
decide a ticket, change the backlog, direct an Implementer, require a GO, delay
an accepted local landing, or veto that landing.

Every `Backlog action: NEW TICKET` or `Backlog action: REOPEN` return has one
ignored temporary Markdown note at the stable repository-relative path
`ai/notes/<plain-ticket-slug>-red-team-finding.md`. The slug is lowercase words
and hyphens, with no date, cycle number, model name, worktree name, or
severity. A later reopening updates the same note instead of creating dated
files. The relay cites this relative path, never an absolute worktree path.

Headings, in this order:

1. **High-level summary** — at least three short complete sentences: expected
   behavior, observed failure, user or scientific consequence. Defines
   specialized terms before relying on them.
2. **Affected behavior and code path** — a concrete input or action, the
   observable result, and the relevant repository paths and symbols, walked in
   reading order.
3. **Reproduction and evidence** — numbered steps, exact commands or fixtures,
   expected and observed output, raw-evidence locations. Reproduced facts are
   labeled separately from inferences.
4. **Impact and proposed severity** — realistic harm, likelihood, the proposed
   High, Medium, or Low rating, and why the evidence meets that bar.
5. **Review scope and exclusions** — the bounded commit, change, behavior,
   paths, and symbols reviewed, and what was not checked. An authorized
   widespread search states its exact Architect-approved boundary.
6. **Proposed acceptance evidence** — a regression witness, exact commands, and
   observable passing result. Proposed checks for the Architect, not Red Team
   approval or a veto.
7. **Uncertainty and counterevidence** — missing facts, alternative
   explanations, successful cases, evidence against the finding, and what would
   disprove it. `None found` is acceptable only after the note explains how
   counterevidence was sought.
8. **Repair directive** — the complete candidate repair packet required by
   `.codex/REDTEAM_ROLE.md`.

`NO-GO`: a thin assertion such as "broken" or "the test failed"; rhetorical
pressure; inflated severity; diary/date/wave narration; model-centered history;
hidden uncertainty; fabricated commands, files, outputs, or observations. A
finding never omits counterevidence because it weakens the argument.

This transfer never lowers evidence standards or makes the note authoritative.

Receipt and assessment happen at different times. On receipt the Architect does
not reproduce or substantively analyze a `NEW TICKET` or `REOPEN` finding —
bookkeeping only: create or restore the ticket, apply the reopen-count and
automatic-severity mechanics, preserve the note path, acknowledge, return to
current work. The backlog technical record includes this exact line:

```text
See further instructions at ai/notes/<plain-ticket-slug>-red-team-finding.md
```

Only when priority later brings that ticket forward does the Architect assess
the note, perform targeted independent verification, set the final severity, and
decide whether to plan a repair. A missing or weak section is recorded as
evidence the Red Team must improve then; it never holds admission bookkeeping
or an unrelated daemon-recorded landing open.

A finding's proposed repair receives the same skepticism as its claim: Red Team
reads for catch power, so its Repair directive can sketch more machinery than
the demonstrated failure needs. At assessment the Architect weighs it against
the proportional-repair rule in `python-changes-go-no-go.md`. When the
demonstrated harm supports only a narrow direct fix, plan that fix and discard
the surplus; when no demonstrated failure supports the construction at all,
close the ticket as not worth building and record the evidence. Catch power
never obligates construction; severity never justifies machinery the failure
does not need.

### Backlog ticket GO / NO-GO

`ai/notes/backlog.md` is the tracked list of unfinished and completed tickets,
written for a human reader first with a separate technical record for tools.
The Architect owns its structure and alone admits a ticket, changes its status,
or moves it between sections.

Contents list order: **Open tickets**, **Parked edge cases**, **Closed
tickets**. The open index contains
exactly one linked `- OPEN` line for each actionable unfinished ticket, because
the watcher counts that marker. Parked edge cases
use `- PARKED`, never enter that count, and are never selected automatically.

The Architect classifies every admitted ticket Critical, High, Medium, or Low
by the harm and likelihood rules above. The linked index shows that classification and is
grouped in priority order: Critical first, High second, Medium third, and Low
last. Work starts with the first
dispatchable ticket in the highest nonempty group. A blocked ticket stays in
its severity group with its blocker while work moves on. Every severity change
records the new evidence and the reason.

Every admitted ticket records one type, **Bug fix** or **New functionality**.
Type says whether it repairs behavior or adds a capability; priority says when
it is worked.

- A Bug fix may be Critical, High, Medium, or Low.
- New functionality may be High, Medium, or Low, never Critical.
- The user controls feature priority; an unstated feature priority defaults to
  Medium and the Architect does not invent urgency.
- Critical bugs preempt every feature.
- A user-designated High feature comes before High bugs.
- High bugs come before a Medium feature.
- A Low feature waits for Critical, High, and Medium bug fixes.
- "After the backlog is closed" means a Low feature whose prerequisites are
  every ticket already open when the feature was admitted. Its own open line is
  not one of those prerequisites.

A **Low — Edge Case** is always a Bug fix and sits below this work order,
created only to preserve the exact harmless exceptional remainder of a bounded
repair. It stays parked until the user explicitly asks for that ticket by its
human title; the Architect then moves it to the Low group and replaces the
parked line with an ordinary `- OPEN **LOW**` line.

Within one permitted group, preserve index order unless a recorded blocker or
prerequisite requires moving on.

Every ticket also keeps an integer named **Red Team reopen count**. It starts
at `0` and never resets. It records how many Red Team reviews in the final step of a normal cycle said
`REOPEN`. That return keeps the same cycle active until the Architect assesses
the evidence. Before that reasoning the trusted reopening checker prints the
exact ticket, landing, severity, count, and legal outcomes; after the backlog
is sealed it proves the counter changed once and the selected state is exact.
It never judges the finding or edits the backlog. The Architect records one
decision: GO restores the ticket to Open at the same severity, except for the
sixth-reopening Low rule; NO-GO keeps it Closed and bars that same objection.

The Architect has the final word before the cycle ends. Above a count of `1`,
compare the new evidence with the ticket's earlier reopening reports and grow
stricter with each attempt: did Red Team find a materially new failure, or
repeat an old objection without new evidence? The Architect may close the
ticket again or lower its priority with a recorded reason. When the count becomes `6`, or is already greater than `5`, priority is
automatically Low. No role may waive that, even
for a ticket that was Critical or High.

Every ticket has one exact reopening state, beginning `allowed`. Architect GO
on a reopening accepts the evidence and leaves the ticket open for repair;
Architect NO-GO closes it with a reason and changes the state permanently to
`barred by Architect NO-GO`. Red Team may not reopen a barred ticket; a
different defect must use `Backlog action: NEW TICKET`. A prohibited later
`REOPEN` changes nothing — not the ticket, its count, or its state.

Ordinary acceptance: the Architect assigns a ticket, the Implementer repairs
it, the Architect audits, and Architect `GO` authorizes the parent daemon to
create and verify one local landing immediately. Red Team does not supply a required
`GO`, and the Architect never waits for Red Team before authorizing an accepted
fix.

A cycle follows one ticket through Architect/Implementer exchanges, Architect
GO, one daemon-created landing, and one Red Team review of that exact landing.
If the bug remains the handoff says `Backlog action: REOPEN`. The Architect may
start the next ticket while that advisory review is pending only when the cycle
limit has another unused ticket slot; a finite watcher does not count or exit
that cycle until the correlated return exists. On receipt the Architect
immediately restores an allowed ticket and increments its reopen count, and
evaluates evidence, final priority, and GO/NO-GO later.

#### Maintain the tracked backlog consistently

`backlog.md` is tracked at `ai/notes/backlog.md`, so a clean clone already
holds the current Open and Closed tickets. If missing, restore the version from
`main`; never invent a shorter private format. A backlog received through the
bundle tool is still input to review, not an automatic replacement. Recreate it with
this exact opening and heading order:

```markdown
# Execution backlog

This operational record is tracked in Git so unfinished fixes survive a new
clone. Only the Architect updates it. The daemon includes the Architect-sealed
ticket update in the same landing commit as the accepted fix.

## Contents

- [Open tickets](#open-tickets)
- [Parked edge cases](#parked-edge-cases)
- [Closed tickets](#closed-tickets)

## How to read this backlog

Each line beginning `- OPEN` represents one unfinished ticket. A Bug fix
repairs behavior that is wrong now. New functionality adds a capability.

Priority controls work order. Critical is reserved for a bug that broadly
breaks a central workflow or systematically makes the science wrong. High,
Medium, and Low use the harm and likelihood definitions in the permanent
workflow contract.

Every ticket has a Red Team reopen count that starts at zero. A ticket moved
to Closed does not wait for Red Team approval. In the final step of a normal
cycle, Red Team may send REOPEN if the bug remains. The sixth REOPEN
assessment automatically makes that ticket Low.

Every ticket also says whether Red Team reopening is allowed. An Architect
NO-GO to reopening is permanent; that ticket is barred from another REOPEN.

New discovery stops when ten or more Critical, High, and Medium tickets are
open; Low tickets do not enter that count. Severity never selects a role. Sol
remains the advisory Red Team when enabled. A malformed open line blocks
discovery decisions until the Architect repairs it.

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

# Parked edge cases

No parked edge cases.

# Closed tickets

No closed tickets.
```

An empty priority group stays visible. When its first ticket is added the
Architect replaces the `No open PRIORITY tickets.` sentence with the index
line; the empty sentence and a ticket line never appear together. A clean clone
with no accepted work still receives the complete skeleton, including all four
empty priority groups and the `No closed tickets.` sentence.

Every parked edge case uses this exact form under `# Parked edge cases`:

```text
- PARKED **LOW — EDGE CASE** **BUG FIX** — [Plain human title](#unique-anchor)
```

It carries the same human summary and technical record as an open ticket, but
its status is `PARKED` and it has no `- OPEN` marker. Command-line severity
choices cannot create or activate it.

Every open index line uses this exact form:

```text
- OPEN **PRIORITY** **TYPE** — [Plain human title](#unique-anchor)
```

`PRIORITY` is exactly `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`. `TYPE` is
exactly `BUG FIX` or `NEW FUNCTIONALITY`. `CRITICAL` with `NEW FUNCTIONALITY`
is invalid. The line sits under the matching priority subheading and is the
ticket's only text beginning `- OPEN`. Groups stay in Critical-High-Medium-Low
order; within High, user-designated High new functionality precedes High bug
fixes; within any remaining group preserve admission order unless a recorded
prerequisite or blocker explains why the next ticket is worked first.

`unique-anchor` uses only lowercase ASCII letters, digits, and hyphens — for
example `cmb-progress-loses-multipole-labels`. It describes the problem, is
unique in the file, and is not merely an internal ticket number. The link
target and the `<a id="...">` value must match byte for byte. Each index link
resolves once, every detailed open ticket has one index link, and a closed
ticket has no `- OPEN` line.

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

**Red Team reopening: allowed.**

**OPEN.** [Current stage, blocker, or prerequisite.]

[For a Bug fix, use:]

**Severity: PRIORITY.** [Concrete harm and likelihood. A High bug must explain
why Medium is insufficient. A Critical bug must explain why High is
insufficient.]

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

The Architect writes a feature's user-supplied priority and any "after the
backlog is closed" prerequisite into `Current status`. A High bug's reason
explains why Medium is insufficient; a Critical bug's also explains why High is
insufficient. These rules apply to every tracked backlog; imported older records are brought
into this shape when first touched rather than copied as an incompatible
private format.

`ai/README.md` carries a fully worked ticket showing the required level of
explanation: `Saved CMB progress can lose its multipole labels`. It is an
example,
not an admitted ticket.

To close a ticket, the Architect removes its one index line, moves the complete
detailed section below `# Closed tickets`, changes `**OPEN.**` to
`**CLOSED.**`, and sets `What is missing` to the exact sentence `Nothing
for this ticket.` Title, anchor, type, final priority, reopen count, reopening
state, human summary, completed work, and technical evidence remain. The
Architect then emits the exact decision-only `architect-go` request for the
audited candidate C without waiting for Red Team. After that Architect process exits, the daemon creates and verifies distinct
landing L, fast-forwards a clean unchanged user `main`, and records any remote
push debt. The Architect does not merge,
commit, update a Git reference, target the user's checkout, or push. If any
required action remains, the ticket stays open or that action receives its own
linked open ticket.

As the final step of each normal cycle, Red Team reviews the one ticket and
daemon-recorded landing L from that cycle. A no-finding result is advisory. If
the bug remains and reopening is still allowed, the handoff says
`Backlog action: REOPEN`, the same cycle stays active while the Architect assesses that
evidence, and the Architect increments the reopen count and cites the stable
finding note with its exact `See further instructions at ...` line. GO restores
the complete ticket to Open at the same severity; NO-GO keeps it Closed,
records why, and permanently bars that objection. Only after this decision may
the cycle complete.

Every ticket section has these parts in this order:

1. **High-level summary** — at least three complete sentences in ordinary
   language: normal purpose and one concrete example, current failure, user or
   scientific consequence. An internal unit number may follow a plain title but
   never replaces it.
2. **Current status** — `OPEN`, `CLOSED`, or `PARKED`; `Bug fix` or `New
   functionality`; priority reason; the nonnegative reopen count and exact
   reopening state; any blocker or prerequisite.
3. **What is already fixed** — completed work, without implying closure.
4. **What is missing** — every action, machine run, review, or decision still
   required. A closed ticket says `Nothing for this ticket`; separate
   unfinished work gets its own linked open ticket.
5. **Technical record for development tools** — exact files, symbols, commits,
   branches, evidence counts, failure cases, and source-note anchors.

Decision table, applied whenever a ticket is added or updated:

| Check | `GO` | `NO-GO` |
| --- | --- | --- |
| Human title | Names the problem in words a physics student can understand; an internal ID is secondary | Uses only `unit 8`, an acronym, a gate ID, or another internal label |
| Human summary | Gives normal purpose with a concrete example, the current failure, and its consequence in at least three complete sentences | Starts with commits, evidence counts, internal stages, or unexplained software language |
| Status | Appears in the correct Open, Parked, or Closed section and agrees with its `- OPEN` or `- PARKED` index | Is missing, contradictory, or described as closed while required work remains hidden in prose |
| Partial work | Separates completed work from missing work | Treats a landed partial fix or local test result as ticket closure |
| Ticket type | Records Bug fix or New functionality and applies its ordering rule | Omits type, labels a feature Critical, or lets a feature bypass a higher-priority item |
| Severity | The Architect records Critical, High, Medium, or Low from concrete harm and likelihood, explains why Medium is insufficient for High and why High is insufficient for Critical, names the primary result or core workflow harmed by a High bug, and places the ticket in that priority group | Severity is omitted, copied from Red Team without review, says only “wrong science,” promotes a plot or diagnostic defect without evidence of primary harm, High or Critical lacks its required comparison, or a ticket is ordered below a lower-severity ticket without a recorded blocker |
| Parked edge case | Records the exact exceptional remainder of a bounded repair below Low, uses `- PARKED`, and waits for an explicit user request naming the ticket before activation | Uses `- OPEN`, appears as a command-line severity, is selected automatically, hides a probable or scientifically consequential failure, or is activated without the named user request |
| Reopen count | Uses one canonical nonnegative integer, starts at zero, never resets, and increments for every formal Red Team `REOPEN` assessment | Omits the count, uses prose instead of an integer, resets it, or loses a Red Team reopening return |
| Reopening state | Uses exactly `allowed` until an Architect NO-GO permanently changes it to `barred by Architect NO-GO`; a barred ticket cannot be reopened | Omits the state, removes a permanent bar, changes a barred ticket after another REOPEN, or treats a different defect as the same ticket |
| Repeated reopening | Immediately restores every Red Team `REOPEN` return, then later compares new evidence with earlier attempts; a count above five forces Low | Delays the bookkeeping for a full audit, calls every repeated objection obnoxious without evidence, or keeps a priority above Low after the sixth attempt |
| Red Team authority | Red Team advice never blocks Architect acceptance or the daemon's verified local landing | Requires a Red Team GO, delays an accepted local landing for Red Team, or lets Red Team edit the backlog |
| New Red Team ticket | The handoff says `Backlog action: NEW TICKET`; the Architect records it promptly with the Red Team assessment as a provisional priority and analyzes it later | The finding waits outside the backlog while the Architect performs a full audit, or another role writes the backlog directly |
| Red Team source note | A `NEW TICKET` or `REOPEN` cites one stable repository-relative finding note with every persuasive-note heading and the backlog preserves the exact `See further instructions at ...` line | Uses an absolute or dated path, omits the note citation, gives a thin assertion, hides uncertainty or exclusions, inflates severity, or invents evidence |
| Technical detail | Preserves exact evidence in the technical record after the human explanation | Removes evidence or makes a human decode it before learning the problem |
| Closure | The Architect accepted the Implementer fix, the daemon verified its exact local landing, every required ticket action passed, and `What is missing` says nothing remains for this ticket; separately recorded remote push debt does not reopen the ticket | A required hardware run, scientific check, Architect decision, daemon landing, or note update remains |
| Open-count check | The number of linked `- OPEN` index lines equals the number of detailed open ticket sections | The watcher count can omit, duplicate, or point to a missing ticket |

Malformed backlog state always fails closed: an `- OPEN` line not matching the
exact grammar, an unknown priority or type, a Critical feature, a line under
the wrong priority heading, a duplicate or missing anchor, a link without one
detailed section, an unlinked detailed open section, contradictory
`OPEN`/`CLOSED` text, a missing or malformed reopen count, or priority groups
out of order. Repair the structure before dispatching that ticket, admitting
discovery, or claiming the backlog is complete. A malformed
line is never ignored, guessed, or rewritten as Low to make a count smaller.

#### Protect the Architect-owned backlog

Only the Architect edits `ai/notes/backlog.md`. The ignored file
`ai/notes/.backlog-guard.json` stores the SHA-256 fingerprint of the exact
backlog bytes the Architect last accepted. It detects an unexpected character
change; it does not prove the ticket description correct.

On a clean clone the daemon initializes the local fingerprint. After a reviewed
intentional replacement the Architect may initialize explicitly:

```bash
python3 ai/tools/backlog_guard.py initialize --architect-ack
```

Before accepting another role's return or changing any ticket:

```bash
python3 ai/tools/backlog_guard.py check
```

Copy the printed 64-character `accepted SHA-256` before the edit. A mismatch is
`NO-GO`: stop, inspect the unexpected change, never replace the fingerprint to
silence the warning.

After one deliberate edit, read the changed ticket, then record those exact
bytes:

```bash
python3 ai/tools/backlog_guard.py seal \
  --previous-sha256 SHA256_FROM_THE_PRE_EDIT_CHECK \
  --architect-ack
python3 ai/tools/backlog_guard.py check
```

Mailbox Architect turns receive `MAILBOX_ROLE=architect`, so the write commands
recognize the role without the manual acknowledgement option; manual terminal
use keeps `--architect-ack`. Red Team turns receive `MAILBOX_ROLE=red-team`:
that role may read the backlog and run `check`, but never edits the backlog,
runs `initialize` or `seal`, or edits `ai/tools/backlog_guard.py`, the
fingerprint record, or its `.backlog-guard.lock` write lock.

The Implementer never opens the backlog at all, in any mode. Only the
Architect, who writes it, and the Red Team, who audits the Architect, read it.
That is deliberate: the backlog is a private planning ledger in Architect
shorthand holding ideas considered and dropped alongside scheduled work, so an
Implementer reading it would collect instructions nobody sent. The dispatched
directive is the Implementer's whole assignment and the notes it names are the
supporting material; if the directive is missing something, the Implementer
returns a blocker instead of going to look for it.

The backlog stays in Git; its fingerprint record and lock stay outside Git. An
incoming backlog package is inspected in its separate import folder and never
replaces the live backlog or fingerprint automatically. This guard catches
accidental role edits and hallucinated replacements; a malicious program able
to rewrite both backlog and guard is outside its scope.

A workstation-only check stays open when required for acceptance. If a large
ticket is split, each follow-up either remains under the parent's missing-work
list or becomes its own linked open ticket. A closed section may mention a
limitation outside its scope only by linking to the open ticket that owns it.

The Architect updates the ticket in the same turn as every state change:
dispatch, returned evidence, GO or NO-GO, landing, a Red Team `REOPEN`, a `NEW
TICKET` return, a new or cleared blocker. The ticket stays OPEN until
implementation, required evidence, Architect review, and any required
permanent-note update are complete. Architect acceptance closes and
seals the backlog ticket, then emits the exact GO request without waiting for
Red Team or landing L. The daemon then creates and records L; a later advisory review may reopen
the ticket.

The Architect note has one current `## Implementation directive`; a confirmed
Red Team return has one current `## Repair directive`. The appropriate contract
checker validates packet structure before transport. Structural validation does
not replace scientific review.

A handoff is a compact routing summary citing the source note, which stays
authoritative when a summary lags or differs. Files under `ai/notes/relay/` are
immutable transport copies for traceability: not evidence, never edited.

Mailbox files live under `ai/notes/mailbox/`. A numbered file is dispatched to
an internal role and archived under `done/`. Public commands do not expose
those internal destinations. A `to-user` status file is not dispatched. A
terminal inbound that explicitly says no reply is owed does not require an
artificial receipt. This is the only
outbound exception; ambiguity requires an outbound response.

In two-role mode, Architect and Implementer communicate directly through the
mailbox and no Sol message is created; existing Sol messages stay untouched
until a normal three-role watch handles them.

Sol is never reassigned as an Implementer. A normal watch uses Sol only for
advisory Red Team review and discovery; `--skip-redteam` turns that role off
rather than converting it into another source-code editor.

Five finished role turns or 15 elapsed minutes creates a manual safe-stop
opportunity: the watcher stops starting new work, lets every job already
starting or running finish, and opens the 20-second Ctrl-C countdown. This
boundary is not a cycle and never changes the `--cycle` count.

A **ticket cycle** always concerns exactly one indexed Open ticket. Its first
Implementer handoff starts with these saved lines:

```text
MAILBOX-FLOW: ticket
MAILBOX-CYCLE: ticket-anchor@full-starting-commit
MAILBOX-MODE: normal
```

Cycle and mode stay unchanged through every Architect/Implementer return. The
first handoff must go to the actual Implementer; an Architect message cannot
invent an unbound cycle. The anchor names exactly one Open backlog ticket and
the starting commit must exist. After the audit, Fable records this exact
decision-only request and performs no Git write:

```text
MAILBOX-RETURN: architect-go
MAILBOX-CYCLE: ticket-anchor@full-starting-commit
MAILBOX-CANDIDATE: full-implementer-candidate
MAILBOX-MODE: normal
MAILBOX-DECISION: GO
```

The parent daemon proves the request names the saved candidate, creates a
distinct one-parent landing commit, verifies the landing is the candidate's
exact clean squash onto current `main`, and fast-forwards only a clean,
still-matching user `main` checkout. In normal mode Sol receives one review of
that exact landing and returns `NO CHANGE` or `REOPEN`, which completes the
cycle count. The Architect may work the next ticket only when the finite limit
has another unused slot; the watcher waits for the correlated Red Team return
before counting or exiting that normal cycle.

In `two-role` mode the verified local landing completes the ticket and its
cycle, because that watcher has no Red Team pass.

Cycle settings control planned stopping:

- with no `--cycle` option, the watcher continues watching;
- `--cycle N`, where `N` is positive, stops safely after `N` completed ticket
  cycles even when recorded work remains; and
- `--cycle 0` exits only after enabled mailbox routes are idle and no local
  backlog index line begins with the exact marker `- OPEN`.

Cycle zero also requires a safe, stable backlog read. A missing, non-regular,
changing, unreadable, oversized, or non-UTF-8 backlog prevents exit and reports
that completion could not be verified.

Backlog prose never creates a mailbox request. Fix-only mode permits work that
closes an existing ticket but refuses discovery and every request to create a
new ticket. Positive cycle counts work with both role setups. Two-role ticket
flows use `MAILBOX-MODE: two-role`.

### Discovery demand

The open-ticket count controls one decision:

1. **New-discovery admission.** Count open Critical, High, and Medium backlog
   tickets. Ten or more stops new discovery so accepted work can be closed.
   Open Low tickets and waiting mailbox files do not count. An unclassified
   open line fails closed until the Architect repairs its classification.

Queue depth remains useful status, and every open ticket, including Low, still
prevents a `--cycle 0` run from claiming all work is finished. Severity never
changes Sol's role: advisory Red Team in a normal watch, absent from a
`--skip-redteam` watch.

Before any Implementer request leaves the mailbox root, the daemon reserves one
slot from the shared positive cycle limit. Active tickets, accepted tickets
waiting for a Red Team return, and completed returns saved for delivery all
consume that limit. With no slot left, the next request stays byte-for-byte at
the mailbox root; a restart restores the durable count before admitting more
work. This stops `--cycle 1` from starting ticket B while ticket A waits for
its Red Team return, and stops concurrent watch attempts from each spending the
full limit.

Only the Architect decides whether an accepted change alters a permanent
general property. Permanent notes are never edited by an Implementer or Red
Team. Routine milestones do not create permanent-note churn.

### One review before protected policy changes

Protected policy files: the eleven permanent notes,
`ai/notes/role-contract.yaml`, `.claude/FABLE_ROLE.md`, and
`.codex/REDTEAM_ROLE.md`. The YAML is the machine-readable source of truth for
stable role permissions, timing limits, and landing rules — not a twelfth
permanent Markdown note. Only the Architect may change these files, through
protected-policy administration; Implementer and Red Team access is read-only.

With Red Team enabled, the Architect prepares the exact draft and sends one
cycle-free `MAILBOX-TICKET: policy` request whose body holds the complete
proposed change and why it may be needed. Red Team checks necessity,
contradictions, and whether a smaller change would work. A change above 4,000
characters or touching several protected files receives a line-by-line review.

That is the only review round: Red Team returns one advisory GO or NO-GO, the
Architect weighs it and gives the final decision. Red Team does not review a
corrected draft, edit a protected file, veto the decision, or reopen this
administration work. With Red Team disabled the Architect records that the
independent review was unavailable and continues under the same narrow guards.

The request has one header followed by the draft:

```text
MAILBOX-TICKET: policy

COMPLETE DRAFT AND PURPOSE
```

Red Team returns one clear recommendation with reasons, then stops. The review
uses no ticket cycle and never creates an Implementer task.

When a protected rule really changes, the Architect queues one separate
administration turn from its bound primary worktree:

```bash
python3 "$MAILBOX_PRIMARY_WORKTREE/ai/tools/handoff_router.py" \
  --architect-notes-admin "PLAIN-LANGUAGE SUMMARY"
```

This publisher is Architect-only. It writes the exact
`MAILBOX-ADMIN: permanent-notes` self-route and refuses a second unresolved
note update. The admin turn begins only when ordinary ticket reservations,
candidate and landing recovery, role processes, and closure review are idle,
and is the sole role launch in that mailbox pass. It may make no change and
return silently. If a permanent note must change, it creates one clean commit
P whose single parent is the exact unchanged local-main commit B. P modifies
one or more protected policy files and no other tracked path.
It then writes exactly:

```text
MAILBOX-RETURN: architect-notes-go
MAILBOX-BASE: FULL-B-COMMIT
MAILBOX-NOTES-COMMIT: FULL-P-COMMIT
MAILBOX-DECISION: GO
```

The parent daemon, not the Architect subprocess, rechecks B, P, the protected
note set, every ordinary landing record, and the clean user checkout, then
fast-forwards B to P, records remote push debt, and fast-forwards clean safe
Architect, Implementer, and Red Team baselines. Dirty, active, or diverged role
work is preserved and refused rather than reset. The route consumes no ticket
cycle and creates no second Sol review. A queued, inflight, or failed
administration/P record is still visible work.
It cannot be abandoned merely because a positive cycle limit was reached or because
the ordinary backlog is empty.

## Landing and branch discipline

Without an explicit grant, the user performs a landing and push. During a live
watch with the saved standing grant, Fable still records only GO or NO-GO. The
parent daemon alone uses the main-landing lock and carries out the bounded Git
operation authorized by GO. No Implementer, Red Team, Fable subprocess, or
subagent inherits that Git authority. Only `main` is pushed; working branches
stay local.

The Implementer's candidate commit `C` and the daemon-created landing commit
`L` have different identities. The Architect audits an immutable snapshot of
`C`. After GO the parent daemon calculates the exact squash tree against the
then-current `main` parent, creates `L` with that one parent, and saves `L` on
a private crash-recovery reference before touching `main`. It refuses an empty
or conflicting squash.

### Protected branch history is never rewritten

The **protected target branch** is currently `main`; the present daemon has no
option that changes it. If a supported option later lets the user choose a
different target branch, that exact branch must receive every protection in
this section before the option may ship. A branch name mentioned only in prose
does not change the protected target.

Protecting the complete history of that branch is a paramount goal and a hard
Architect rule. No AI role, subagent, daemon, recovery path, suggested manual
command, or application programming interface (API) call may force-push or
replace its history. This prohibition includes:

- `git push --force`, `git push -f`, and `git push --force-with-lease`;
- a push refspec beginning with `+`;
- deleting and recreating the remote branch;
- moving the protected local branch backward with `reset` or `update-ref`;
- rebasing, amending, filtering, or otherwise replacing commits already in the
  protected branch's history; and
- using a hosting-service option or API field that permits a non-fast-forward
  update.

`--force-with-lease` is still a force push. Knowing the expected remote commit
does not make history replacement acceptable.

Choosing a target branch or granting landing or push authority never grants
authority to force-push or replace that branch's history.

The protected branch moves only by fast-forward: its new commit must contain
its exact previous commit in its history, and a normal push must meet the same
condition on the remote branch. If the local branch, remote branch, expected
parent, or verification state differs, the operation refuses and preserves the
commits for inspection. A remote refusal becomes visible push debt, never
repaired by rewriting history.

This rule outranks ticket closure, cycle completion, automation recovery,
conflict convenience, and clearing push debt. The Architect issues `NO-GO` to
any plan, candidate, recovery instruction, or manual command that could rewrite
the protected branch. The safe response: stop, show the divergence, and prepare
a new descendant commit only after the user chooses how the histories should be
reconciled without force.

### Commit messages explain the saved change

GitHub renders a commit subject and body as Markdown. Every commit message an
AI role authors for this repository follows this rule, in a mailbox watch or a
manual session; candidate, landing, and permanent-note commits are examples,
not the complete scope. A message is `GO` only when a reader understands the
saved change without opening the diff.

The subject names the concrete saved behavior in plain language — for example
`Keep each calculation result with its assigned dataset row`. A subject carries
no internal ticket number, date, wave name, role label, branch name, undefined
acronym, schema number, or project jargon. `Update files`, `Land unit 8`, and
`Fix issue` are `NO-GO`.

Every AI-authored commit message follows the subject with the exact four-part
Markdown body defined in `ai/notes/readme-go-no-go.md`:

1. **Why this change was needed** begins with behavior a user or maintainer
   could observe and states its consequence.
2. **What this commit changes** names the saved behavior and gives a concrete
   repository example before any broad rule.
3. **What remains unchanged** names behavior this commit does not change or
   support. It is not an empty ceremonial heading.
4. **Checks run** gives each exact command or check and its visible result. An
   important check that was not run is named together with the reason.

Short paragraphs or bullets per section; define an unfamiliar term at first
use; never paste a backlog ticket, an audit transcript, or one wall of text.
Recovery lines added by the mailbox program may follow the four human sections
but never replace or interrupt them.

Subject and all four sections describe the saved current behavior, not who
requested it, when a policy was added, or which ticket, audit wave, review
round, rollout phase, model, or earlier commit produced it. Scientific,
runtime, algorithmic, and compatibility subject matter follows the narrow
exception in `ai/notes/readme-go-no-go.md`.

Before accepting, landing, or pushing the commit, the Architect reviews the
exact full hash and records:

- the subject and a cold-reader paraphrase of the saved behavior;
- every unfamiliar term and the local definition or replacement used;
- the concrete example that introduces each broad idea;
- the important behavior the commit does not change or support;
- the four Markdown headings and their order; and
- the exact checks and visible results.

`NO-GO` when a physics undergraduate must open the diff, backlog, or an
internal note to understand the message; when evidence says only `tests pass`;
when a heading is empty; or when any applicable prose or anti-AI row in
`ai/notes/readme-go-no-go.md` fails.

The Architect reviews the exact candidate commit `C`, subject and body
included. Architect GO names the full hash of `C`, binding that reviewed
message before the landing commit exists. The daemon copies the human subject
and body from `C` into landing commit `L` without rewriting them, then appends
only the required mailbox recovery trailers. Creating or recovering `L` refuses
if its message differs from the approved candidate message plus those exact
trailers. Lines beginning `Mailbox-Cycle:` or `Mailbox-Candidate:` are reserved
for those trailers; a candidate message already using either label is refused
rather than copied into an ambiguous landing message. Letter-case changes and
spaces before the colon are still the same reserved labels.

Review evidence includes the visible result of:

```bash
git show -s --format=%B FULL_COMMIT
```

The deterministic landing test must also prove the message survives creation
and crash recovery unchanged. An internal ticket anchor or machine trailer
never replaces the human explanation.

If the user's checkout is not clean, is no longer attached to `main`, or no
longer names the prepared parent, the daemon stops, preserving `C`, `L`, the GO
request, and the user's files without resetting or overwriting anything. When
the checkout is clean and unchanged it performs only a fast-forward to the
already verified `L`, rechecks the result, and records the local landing
durably. The Red Team reviews an immutable snapshot of `L`.

The daemon then attempts a normal, non-force push of that exact `L`. Missing
credentials or a rejected push create a local push-debt record with the exact
manual command. Push debt is visible work for the user; it does not erase the
local landing, reopen the ticket, or make the same cycle run forever.

Never merge `main` back into the Implementer worktree. The daemon restores the
exact saved candidate when a repair is required and prepares later tickets from
their own recorded commits; mixing landing history into that lane would break
those identity checks.

## Environment assumptions

The lightweight development machine may provide only Python, NumPy, and the
standard library. Evidence there is compilation, AST censuses,
docstring-stripped AST comparison, and known-answer arithmetic probes against
the real function body where possible. Torch, CosmoLike, Hierarchical Data
Format version 5 (HDF5), YAML, SciPy, Matplotlib, and accelerator evidence run
in the configured Cocoa environment.

Apple Metal Performance Shaders (MPS) does not support device float64 and uses
float16 autocast. CUDA provides the required compiled and accelerator checks;
set `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES` before process startup. The
production system uses task-parallel processes, not distributed data parallel
(DDP) or threads: spawn, not fork; one device selection per worker; no private
copies of the full random-access memory (RAM) payload in parallel paths;
longest-processing-time assignment; and retained Queue/Lock references until
every child joins.

The configured CoCoA environment uses NumPy 1.x. An isolated code,
documentation, or dependency change must not adopt NumPy 2 behavior. A NumPy 2
migration requires an explicit project-wide decision and validation across the
emulator families, data generators, inference adapters, tests, and gates.

`ROOTDIR` is defined by the Cocoa startup process. Repository paths anchor to
that value and `cobaya-run` starts from `ROOTDIR`. Public installation
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
  switch to its documented application programming interface (API) and add a
  tripwire capable of falsifying the replacement assumption.
- A search supporting "no match exists" must be untruncated. Count or inspect
  all matches, search the synonym set, and record the pattern and scope.

### Tests, gates, and the validation board

A **test** asks one narrow question — for example, a CMB progress-file test
shifts one saved multipole coordinate and checks that loading refuses the
mismatch before reading spectra.

A **gate** is a named final check for a larger requirement, running one test,
several tests, a scientific comparison, or a hardware-dependent job. A passing
narrow test does not replace a gate the Architect required.

The **validation board** is the ordered registry of gates and the raw machine
evidence saved for each run. The Architect reads the board and the actual
output before deciding GO or NO-GO. Evidence that cannot run because required
hardware or data is absent is recorded as unavailable, never converted into
PASS. Command inventories and current gate membership belong in
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
contracts on every nontrivial override, a generated callback inventory with no
undocumented callback, and successful syntax compilation.

## Current-state API explanations

Loss decoding returns the kept-coordinate vector: it inverts the numerical
transform and does not restore masked positions. Full-vector reconstruction is
a separate `geometry.unsqueeze(kept)` step. Every loss subclass and caller
preserves that distinction; equality of kept and full widths in a diagonal
family does not redefine the general contract.

Weight decay is selected by module role, not tensor rank. Only `.weight` from
`Linear`, `Conv1d`, and `BinLinear` is decay-eligible; all other parameters
remain undecayed unless the allowlist is deliberately expanded.

Geometry encode or whiten operations divide by scale or sigma; decode or
unwhiten multiply. Errors and comments must name the correct direction.

Automatic mixed precision (AMP) runs selected operations at lower numeric
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

Warm-start and transfer documentation includes a concrete named-column example,
exact encoded column order, input-weight shapes, copied and zeroed columns,
view/copy ownership, and the meaning of `torch.no_grad`. Packed targets are
shown with shapes. Parity is an executed epoch-zero equality check with
coordinate system, dtype, device, and tolerance stated. An unavailable feature
is described as unavailable and refused rather than promised through
unreachable code.

Gate files begin with the exact behavior they require, one real input and
visible result, their dependencies, and why a failure blocks acceptance. A
nontrivial check documents the system under test, fixture, independent expected
answer, and deliberate mutation. Terms such as
fixture, test double, fake, stub, monkeypatch, known answer, control, mutation,
and catch power are defined before use. A numerical reference cannot be
computed by the same helper as the value under test.

## Stable workflow evidence anchors

<a id="board-selftest-exit-truth"></a>
**The board runner reports what actually ran.** Unknown or conflicting
selectors, dependency skips, compile-lane skips, stale or edited logs,
unresolved anchors, duplicate assertion identifiers, and malformed evidence all
produce a non-green result. A stored pass is reusable only while its raw log
and digest remain intact.

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

The Architect decides what tracked documentation must change, writes a detailed
directive, and reviews the rendered result. The Implementer may edit a README,
a long-form document under `documentation/`, or explanatory Python prose only
when that bounded directive names the exact section, document, or symbol. The
Red Team may report a documentation defect and review the rendered result, but
it never edits tracked documentation. Permanent notes remain Architect-only.

### Feature-specific long-form documentation

A request such as `write documentation about X` asks for one bounded guide to
an important feature, script, or mechanism whose complete explanation would
overload a README. It does not authorize another manual for the whole library.
The repository-wide example is `documentation/emulator_code_guide.tex`; the
focused-feature example is `documentation/candidate_to_landing.tex`.

Before planning a new file, the Architect searches `documentation/README.md`,
tracked files under `documentation/`, relevant README headings, and likely
source names, symbols, commands, and synonyms. The temporary source note
records what was searched and which possible owner sections were opened. If one
document already answers the same reader question, the plan updates that owner
or improves the link to it. A second guide for the same question is `NO-GO`.

A new guide is allowed only when both conditions hold:

1. the topic is important for understanding or maintaining the library; and
2. the full explanation is too long for the relevant README.

The README keeps a short introduction and links to the one long-form owner. The
Architect's directive names the reader's exact question, intended audience,
included and excluded scope, current source files and symbols,
existing-document census, README link, source and compiled deliverables, build
command, page-render command, and page-by-page visual checks. It also requires
comparison with current code so a polished explanation cannot preserve an
obsolete command or behavior.

Useful focused guides often include an executive summary, a small mental model,
separate definitions for easily confused objects, commands explained one at a
time, a complete worked example, important refusal behavior, alternatives and
why they are not used, safety properties, an implementation map, and a compact
translation table. This is a teaching pattern, not a fixed page template:
select only the parts that answer the named reader question.

Feature-specific documentation is a **Low new-functionality ticket** by
default. It becomes **High** only when the user explicitly requests High
priority because understanding that feature is urgent. Importance alone does
not promote it. Incorrect existing documentation that can damage normal use is
a bug and receives the ordinary evidence-based bug severity instead.

The Architect owns scope, duplicate prevention, the complete directive, factual
review, and final `GO` or `NO-GO`. The Implementer writes the tracked source
and compiled artifact. The Red Team remains an optional advisory reviewer.
Every changed PDF is compiled from its tracked source, rendered page by page,
and inspected for clipping, overlap, unreadable figures, broken references, and
stale terms before `GO`.

A behavior change affecting a "Current gap" paragraph names that paragraph in
the Architect source note, and the directive requires rewriting it to current
behavior or narrowing it to the remaining limitation. A stale gap is a
documentation defect. Permanent notes remain Architect-only under
[`MEMORY.md`](MEMORY.md), even when a documentation unit is active.

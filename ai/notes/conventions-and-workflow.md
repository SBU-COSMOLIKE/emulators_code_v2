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
Architect source note.

A **detached** worktree has no branch selected. A **prunable** worktree is one
whose registered folder Git reports as missing and eligible for removal. A
**dirty** worktree has uncommitted changes. **Ahead** means its branch has
local commits not present on `main`; **diverged** means both branches have
different commits after their last shared version.

A **gate** is a registered acceptance command. A **fixture** is the fixed
input setup used by a gate. A **control** is a valid case that must pass. A
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
choice are separate. Command-line model options may assign a different model
to Architect, Implementer, or Red Team without changing role authority, Git
worktree ownership, mailbox route, or evidence requirements.

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

### Backlog ticket GO / NO-GO

`ai/notes/backlog.md` is the local list of unfinished and completed tickets.
It is written for a human reader first and retains a separate technical record
for development tools. The Architect owns its structure and is the only role
that admits a ticket, changes its status, or moves it between the open and
closed sections.

The file begins with **Open tickets** and **Closed tickets** entries in its
contents list. The full **Open tickets** section comes before **Closed
tickets**. Under the open heading, one linked index line begins with the exact
text `- OPEN` for each unfinished ticket. This exact marker is required because
the watcher counts it. There is no second `- OPEN` line inside that ticket, and
every index link resolves to exactly one detailed open section.

Every ticket section has these parts in this order:

1. **High-level summary** gives two or three sentences in ordinary language.
   It states what goes wrong now, gives one concrete example when the problem
   is broad, and explains the user or scientific consequence. An internal unit
   number may follow a plain title, but it never replaces that title.
2. **Current status** says exactly `OPEN` or `CLOSED` and gives the reason.
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
| Human summary | Gives the current failure, a real example when needed, and its consequence in two or three sentences | Starts with commits, evidence counts, internal stages, or unexplained software language |
| Status | Appears in the correct Open or Closed section and agrees with the linked `- OPEN` index | Is missing, contradictory, or described as closed while required work remains hidden in prose |
| Partial work | Separates completed work from missing work | Treats a landed partial fix or local test result as ticket closure |
| Technical detail | Preserves exact evidence in the technical record after the human explanation | Removes evidence or makes a human decode it before learning the problem |
| Closure | Every required action passed and `What is missing` says nothing remains for this ticket | A required hardware run, scientific check, review, merge, or note update remains |
| Open-count check | The number of linked `- OPEN` index lines equals the number of detailed open ticket sections | The watcher count can omit, duplicate, or point to a missing ticket |

A workstation-only check stays open when it is required for acceptance. If a
large ticket is split, each follow-up either remains under the parent ticket's
missing-work list or becomes its own linked open ticket. A closed section may
mention a limitation outside its scope only by linking to the open ticket that
owns that work.

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
until a normal three-role watch handles them. A cycle limit controls safe
stopping; cycle zero means continue until admitted mailbox work and recorded
backlog work are finished. The watcher does not create a request merely from
backlog prose.

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

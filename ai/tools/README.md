# AI development tools

Most users need one program in this folder: `mailbox_daemon.py`. It waits for
a request to the Architect and starts the AI roles needed to work on that
request. The first example below previews the setup without changing anything.

Read [`ai/README.md`](../README.md) first if the Architect, Implementer, and
optional Red Team roles are new. This page concentrates on commands: how to
start them, what they change, and what result to expect.

## Contents

### Main guide

1. [Which tool do I use?](#which-tool-do-i-use)
2. [Where do I run these commands?](#where-do-i-run-these-commands)
3. [Which commands only inspect, and which commands change files?](#which-commands-only-inspect-and-which-commands-change-files)
4. [Remove every AI work folder](#remove-every-ai-work-folder)
5. [Send or check a request](#send-or-check-a-request)
6. [Choose the minimum discovery severity](#choose-the-minimum-discovery-severity)
7. [Fix-only maintenance](#fix-only-watches)
8. [Limit the size of one ticket](#limit-the-size-of-one-ticket)
9. [Runtime controls](#runtime-controls)
10. [Exact command reference](#exact-command-reference)

### Common questions raised by developers

**[Appendices for AI roles and maintainers](#appendices-for-ai-roles-and-maintainers)**

- [How does a role check a handoff directive?](#check-a-handoff-directive)
- [What happens if a candidate changes an unplanned file?](#ticket-file-scope)
- [How does the Architect check protected project notes?](#check-protected-project-notes)
- [How is a closed ticket reviewed?](#review-a-closed-ticket)
- [How does the Architect protect the tracked backlog?](#protect-the-tracked-backlog)

**[Appendices about ticket size](#appendices-about-ticket-size)**

- [FAQ A1. Exactly what does `--max` count?](#faq-a1-max-count)
- [FAQ A2. Why can the size check refuse a ticket?](#faq-a2-max-refusal)

**[Appendices about stopping the watcher](#appendices-about-stopping-the-watcher)**

- [FAQ B1. When can I interrupt the watcher?](#appendix-b--when-is-it-safe-to-stop-the-watcher)
- [FAQ B2. What does `--cycle` count?](#faq-b2-cycle-count)

**[Appendices about setup, problems, and recovery](#appendices-about-setup-and-recovery)**

- [FAQ E1. What should I check first?](#appendix-e--how-do-i-troubleshoot-a-run)
- [FAQ E2. What should I do if the tool rejects a saved AI folder?](#faq-e2-primary-recovery)
- [FAQ G. How do I set this up on another computer?](#appendix-g--how-do-i-install-this-on-another-machine)

**[Appendices about sharing unfinished work](#appendices-about-occasional-transfer)**

- [FAQ H1. How can I send unfinished work to another person?](#appendix-h--how-can-i-send-unfinished-work-to-someone-else)
- [FAQ H2. How can the other person open that package safely?](#faq-h2-inspect-unfinished-work)

## Which tool do I use?

Most users can ignore six of the seven programs until a command or error
message points to one of them. Begin with `mailbox_daemon.py`; it previews or
runs the ordinary workflow and sends every user request to the Architect.

The table starts with the task. Git names, saved-record names, and helper
rules are introduced later, next to the command that needs them.

### Preview the ordinary workflow

The **watcher** is the program that waits for requests and starts the matching
AI role. Run this and the following examples from the repository's top folder,
which contains both `README.md` and `ai/`. Preview the setup first:

```bash
python3 ai/tools/mailbox_daemon.py --dry-run
```

A dry run prints the folders and waiting requests it would use. It does not
create a folder, write a request, or start an AI role.

### Start the watcher

Start the ordinary three-role workflow in one terminal:

```bash
python3 ai/tools/mailbox_daemon.py --watch
```

Leave that terminal open. The watcher checks every 20 seconds. Later sections
show how to choose models, omit the Red Team, or stop after a chosen number of
tickets.

### Send one request to the Architect

The sample uses `version-flag.md`, the source note from the
[first-ticket tutorial](../README.md#complete-one-small-ticket). A source note
is a Markdown file that explains one requested change and the checks that will
show whether it is finished.

In another terminal, preview the request before saving it:

```bash
python3 ai/tools/mailbox_daemon.py --dry-run --send architect \
  --unit "Please coordinate the ticket in ai/notes/version-flag.md."
```

Remove `--dry-run` when the preview is correct:

```bash
python3 ai/tools/mailbox_daemon.py --send architect \
  --unit "Please coordinate the ticket in ai/notes/version-flag.md."
```

Success prints `queued PATH`. The command saves one small Markdown request
file; the watcher starts the Architect when it reaches that file. The user
sends every work request to the Architect, never directly to the Implementer
or Red Team.

### Read the current status

```bash
python3 ai/tools/handoff_router.py --status
```

This command only reads. It summarizes saved work, reviews that have finished,
reviews that are still waiting, and the next suggested action.

### Stop only during the safe message

Press Ctrl-C only while the latest status line literally says
`safe to Ctrl-C`. For example:

```text
every enabled role is idle; safe to Ctrl-C for 19s more; 3 messages waiting.
```

Do not interrupt while a role is running or the watcher is preparing to start
one. [FAQ B1](#appendix-b--when-is-it-safe-to-stop-the-watcher) explains each
stop-related message.

### Restart after an accidental Ctrl-C

If Ctrl-C stopped the Implementer, preserve the Architect's plan and discard
only the partial implementation:

```bash
python3 ai/tools/mailbox_daemon.py --restart-implementer
```

If Ctrl-C stopped the Red Team, preserve its exact review request and discard
only the interrupted review work:

```bash
python3 ai/tools/mailbox_daemon.py --restart-redteam
```

Each command discards partial work only for the named role, requeues the saved
handoff, and tells the user when `--watch` may start again. It refuses when a
completed result already exists because that result belongs to the Architect.

### Choose another program only when needed

| What you want to do | Program | First command | Effect |
| --- | --- | --- | --- |
| Preview, start, or stop the mailbox workflow; send a request to the Architect; choose role models | `mailbox_daemon.py` | `python3 ai/tools/mailbox_daemon.py --dry-run` | A dry run only prints. Commands that write files may create AI work folders, save or move mailbox files, write logs, and start AI roles. |
| Check that an Architect or Red Team instruction contains every required part | `handoff_contract.py` | `python3 ai/tools/handoff_contract.py --help` | Reads one Markdown note. It does not run its tests or judge the scientific plan. |
| Read status or run the manual clipboard workflow | `handoff_router.py` | `python3 ai/tools/handoff_router.py --status` | Status only reads. A run with `--note` changes the clipboard, waits for copied replies, runs local commands, and writes relay records. |
| Check that eleven protected project notes and the machine role contract still match the Architect's starting commit | `permanent_note_guard.py` | `python3 ai/tools/permanent_note_guard.py --help` | Reads Git and the protected files. It changes nothing and does not issue `GO` or `NO-GO`. |
| Detect an accidental change to the tracked backlog | `backlog_guard.py` | `python3 ai/tools/backlog_guard.py check` | `check` only reads. Architect-only `initialize` and `seal` commands write the ignored fingerprint record. |
| Count the text added and removed by one proposed ticket | `ticket_change_guard.py` | `python3 ai/tools/ticket_change_guard.py --help` | Reads two saved versions of one ticket and reports the character count. |
| Package unfinished backlog work and its local supporting files for another person | `backlog_bundle.py` | `python3 ai/tools/backlog_bundle.py pack --dry-run` | A dry run lists files. `pack` writes one `.tar.xz` archive; `unpack` writes a new review folder that Git does not include in commits. |

## Where do I run these commands?

Run the examples from the repository's top folder: the folder containing both
`README.md` and `ai/`. From any project folder that Git recognizes, this
read-only shell command moves the terminal there and confirms that the tools
exist:

```bash
cd "$(git rev-parse --show-toplevel)"
test -d ai/tools && printf '%s\n' 'AI tools found'
```

Expected result:

```text
AI tools found
```

Changing the terminal's current folder does not change a project file.
`mailbox_daemon.py` can resolve the saved AI work folders from another
project folder, but using the top folder keeps paths in examples predictable.

## Which commands only inspect, and which commands change files?

For a first run, remember this shorter rule:

- `--help`, `mailbox_daemon.py --dry-run`, and
  `handoff_router.py --status` only print information.
- `--send` saves a request file.
- `--ping` spends one very small model request per checked service but does
  not write a mailbox file or start a ticket.
- `--watch` and `--once` may create AI work folders, start roles, move request
  files, and write workflow records.
- `--restart-implementer` and `--restart-redteam` discard the named role's
  partial work and requeue its exact saved handoff.
- `--clean-all` permanently discards the extra local work folders and local
  branches created for AI roles.

<details><summary>Show the complete command safety table</summary>

| Command form | Changes project files or saved AI records? | What it does |
| --- | --- | --- |
| Any `--help` command | No | Prints options and exits. |
| `mailbox_daemon.py --dry-run` | No | Prints work folders and messages that a command which writes files would use. |
| `handoff_contract.py architect NOTE` or `handoff_contract.py redteam NOTE` | No | Reads and checks one directive. |
| `handoff_router.py --status` | No | Reads branches and local records, then suggests a next action. It does not run that action. |
| `permanent_note_guard.py --base FULL_COMMIT` | No | Compares the protected files in saved Git versions, the files selected for the next commit, and the files currently visible. |
| `backlog_guard.py check` | No | Compares the current backlog with the SHA-256 fingerprint last accepted by the Architect. |
| `ticket_change_guard.py --base FULL_COMMIT --max NUMBER` | No | Implementer check: counts characters added and removed between the named starting commit and a clean current `HEAD`, Git's name for the current saved commit. |
| `ticket_change_guard.py --base FULL_COMMIT --architect-audit --candidate FULL_COMMIT --max NUMBER` | No | Architect check: counts the same ticket against the exact saved candidate named after `--candidate`. Later commits and unsaved work do not become part of this measurement. |
| `backlog_bundle.py pack --dry-run` | No | Lists the proposed package without writing it. |
| `backlog_bundle.py inspect ARCHIVE` | No | Validates and lists an incoming package without unpacking it. |
| `mailbox_daemon.py --send architect` | Yes | Saves one numbered request for the Architect. The command may create or reuse the AI work folders first. |
| `mailbox_daemon.py --ping` | No repository change; uses AI credits | Makes one small live request to Claude and one to Sol, then prints whether each service answered. Add `--skip-redteam` to check Claude without contacting Sol. |
| `mailbox_daemon.py --once` or `--watch` | Yes | May create or reuse AI work folders, start roles, move mailbox files, and write relay or saved workflow records. |
| `mailbox_daemon.py --clean-all` | Yes—destructive | Removes every extra local CoCoA-Flow worktree and every matching local AI branch, even when that AI folder contains unfinished or unmerged work. It does not alter remote branches, tags, or stashes. |
| `handoff_router.py --architect-notes-admin "SUMMARY"` | Yes | Architect-only internal operation. From an already bound Architect process, queues one later permanent-note admin self-route. It refuses from a normal user, Implementer, or Red Team process and cannot be combined with another router operation. |
| `backlog_guard.py initialize --architect-ack` or `backlog_guard.py seal --previous-sha256 SHA256 --architect-ack` | Yes | Writes the ignored backlog fingerprint record. These manual forms are Architect-only. |
| `handoff_router.py --note NOTE` | Yes | Changes the clipboard, writes local relay records, and runs the selected shell commands. It does not launch a web session for you. |
| `backlog_bundle.py pack` | Yes | Writes a new ignored `.tar.xz` file and never replaces an existing file. |
| `backlog_bundle.py unpack ARCHIVE` | Yes | Writes into a fresh ignored folder under `ai/backlog-imports/`; it does not replace live notes. |

</details>

In this table, `NOTE` means the path to a ticket note, `FULL_COMMIT` means the
full Git name of the starting saved version, `SHA256` means a 64-character
fingerprint printed by the guard, and `ARCHIVE` means the path to a received
`.tar.xz` file.

A first mailbox command that writes files may create three extra Git-managed
project folders, one for each role.
The [role guide](../README.md#appendix-f--what-is-the-worktree-topology)
explains what each folder is for.

## Remove every AI work folder

Run this command from the user's main repository folder only when no old AI
work needs to be kept:

```bash
python3 ai/tools/mailbox_daemon.py --clean-all
```

The command refuses while a mailbox watcher is running. It permanently
removes:

- every extra AI worktree, including its uncommitted files and unmerged
  commits; and
- local `claude/*`, `codex/*`, and old `worktree-agent-*` branches.

There is no backup or second question. The user's current folder, non-AI
branches and worktrees, remote branch records, tags, and stashes remain.

Cleanup never runs automatically. After it finishes, `--once` creates fresh
role folders:

```bash
python3 ai/tools/mailbox_daemon.py --once
```

New Claude worktrees use `claude/*` branches. New Sol worktrees use `codex/*`.

## Send or check a request

The opening workflow already gives the three commands used most often:

- [Read the current AI work status](#read-the-current-status).
- [Preview and send one request to the Architect](#send-one-request-to-the-architect).
- [Stop during a printed safe interval](#stop-only-during-the-safe-message).

<a id="read-the-current-ai-work-status"></a>
<a id="preview-one-request-to-the-architect"></a>
<a id="send-a-ticket-request-to-the-architect"></a>

The remaining examples in this section are optional user actions. Checks run
by the AI roles themselves are collected under
[Appendices for AI roles and maintainers](#appendices-for-ai-roles-and-maintainers).

### Run a two-role manual relay

First ask the Architect for a two-role plan. The validated source note must
contain these exact rows under `### Role plan`:

```markdown
- Roles: `Architect + Implementer`
- Discovery severity: `not-used`
- Review scope: `not-used`
- Ticket class: `ordinary`
```

Then carry out the plan:

```bash
python3 ai/tools/handoff_router.py \
  --note ai/notes/version-flag.md \
  --skip-redteam
```

Unlike `--status`, this manual relay changes the system
clipboard, waits for copied result sections, creates local relay records, and
runs the selected validation commands. Read every `--gate-cmd` string before
allowing the shell to run it.

The Architect wrote the decisions in the source note. The relay tool checks
those decisions and builds each clipboard block by adding the paths and
record locations needed for this run. `--skip-redteam` only confirms the
two-role plan; it cannot change another plan into a two-role plan.

Copy each generated block unchanged. Do not edit it, add a request for
another role, or answer on that role's behalf. Give new information to the
Architect. The Architect must update and revalidate the source note before
another relay.

This command controls only that clipboard relay. It does not change the roles
used by an already running mailbox watcher. Its source must be an ordinary
`.md` file rather than a linked shortcut in this project folder's `ai/notes/`
directory. The tool refuses a relay log, a mailbox file, a path outside
`ai/notes/`, or a path containing `../` because none of those is the original
source note.

In a path, `../` asks to move above the current folder.

### Ask the Architect for a Red Team search

```bash
python3 ai/tools/mailbox_daemon.py --send architect \
  --severity medium \
  --unit "Please instruct the Red Team to review the version-flag change in ai/notes/version-flag.md. Keep the review within that change and use medium as the minimum discovery severity."
```

Success prints `queued PATH` and writes one numbered request to the Architect.
The Architect records the scope and severity, checks whether new discovery is
allowed, and writes the internal Red Team handoff. If accepted, that later
internal `to-sol` request contains `MAILBOX-TICKET: discovery` and
`MAILBOX-SEVERITY: medium`.

A **discovery** asks the Red Team to look for a new problem in the named
change. It is refused when fix-only mode is on or when ten or more open
Critical, High, and Medium tickets are recorded. Low tickets and waiting
mailbox files do not count. The role guide explains this
[limit on new searches](../README.md#appendix-d--what-is-the-demand-guard).

The user states a ticket-specific `high`, `medium`, or `low` choice to the
Architect. A watcher or one-time run may also set the default. The Architect
saves the selected value in the internal discovery request, so a later
watcher's default cannot change it.

### Ask the Architect to finish a recorded Red Team review

```bash
python3 ai/tools/mailbox_daemon.py --send architect \
  --unit "Please instruct the Red Team to finish the existing review described in ai/notes/backlog.md."
```

A **closure** asks the Red Team to finish or recheck a problem that is already
recorded. Success here writes the user's request to the Architect. The
Architect later writes the internal `to-sol` request beginning with
`MAILBOX-TICKET: closure`.

### Check whether the selected AI services can answer

```bash
python3 ai/tools/mailbox_daemon.py --ping
```

This makes one small, no-tool request to the configured Claude Architect and
one read-only request to Sol. When the Implementer uses Ollama, it also checks
that Ollama model independently. The command checks installed programs,
current login, selected models, and service responses. It does not write a
mailbox message or start a ticket.

A successful run ends with:

```text
Claude Architect: online and answered the connection test.
Sol: online and answered the connection test.
connection check passed: Claude and Sol responded.
```

When a run will not use Red Team, omit Sol:

```bash
python3 ai/tools/mailbox_daemon.py --ping --skip-redteam
```

That form never starts Sol. To check a Claude Architect and Ollama
Implementer, add the same provider and model selected for the watch:

```bash
python3 ai/tools/mailbox_daemon.py --ping --skip-redteam \
  --implementer-provider ollama \
  --implementer-model glm-5.2:cloud
```

A failed check returns a nonzero status and names the affected service. The
check has a two-minute limit for each service and continues checking the other
selected services after one fails.

## Choose the minimum discovery severity

Here, a **discovery** is a request for the Red Team to inspect one named
change for a new bug that could become a separate piece of work.

**Severity** means how much harm a bug can cause. The value is the user's
minimum for opening new work from a Red Team discovery. The default is
`medium`.

From any project folder that Git recognizes, tell the Architect when one
request needs a particular value:

```bash
python3 ai/tools/mailbox_daemon.py --send architect \
  --severity high \
  --unit "Please instruct the Red Team to review the named change in ai/notes/version-flag.md."
```

Success prints `queued PATH` and writes one numbered request to the Architect.

<details><summary>Show the fields saved with this request</summary>

Its first two lines save the severity and the review boundary:

```text
MAILBOX-SEVERITY: high
MAILBOX-SCOPE: bounded
```

The user's exact request follows after one blank line. The Architect decides
whether discovery is allowed and writes the internal Red Team request. That
internal request begins with three lines:

```text
MAILBOX-TICKET: discovery
MAILBOX-SEVERITY: high
MAILBOX-SCOPE: bounded
```

The user does not create or send this `to-sol` request directly.

</details>

The three values mean:

| Value | Which findings may become new tickets? |
| --- | --- |
| `high` | Only a bug that severely impacts core functionality, causes data loss, halts system operations, or makes the science wrong. The evidence must show that harm and explain why Medium is insufficient. |
| `medium` | High-severity bugs, plus a probable bug that can affect normal operation. Merely theoretical or improbable edge cases do not qualify. |
| `low` | Any concrete discovered bug, including an improbable edge case. A guess without a code path and evidence does not qualify. |

High must remain unusual. A difficult repair, missing optional feature,
inconvenient cleanup, or desire for another Implementer does not meet this
bar. The Architect records why the evidence is too severe for Medium before
classifying a ticket High.

Critical is not a fourth command value and is not a Red Team rating. Only the
Architect may use it as a final backlog classification after evidence shows
that a current bug broadly breaks a central library workflow or systematically
invalidates scientific results. High does not automatically become Critical;
the Architect records why High is insufficient.

A watch or one-time run can set the default for discovery requests created by
its roles. Run either command from any project folder that Git recognizes:

```bash
python3 ai/tools/mailbox_daemon.py --watch --severity high
python3 ai/tools/mailbox_daemon.py --once --severity low
```

On first live use, the daemon creates or reuses the saved Claude and Sol work
folders. `--watch` keeps checking for requests; `--once` checks the current
waiting requests and then exits. Both print `discovery severity default:`
followed by the selected value. If they handle work, they can start AI roles
and move completed request files into `ai/notes/mailbox/done/`.

Each new discovery still saves its value in the request file. A saved value
does not change when a later watch uses another default. A stored discovery
that lacks either its exact severity line or its exact scope line is refused
instead of receiving a guessed value.

<details><summary>Use the same severity in a manual clipboard relay</summary>

For a manual clipboard relay, first ask the Architect to write these rows in
the source note:

```markdown
- Roles: `Architect + Implementer + Red Team`
- Discovery severity: `high`
- Review scope: `bounded`
- Ticket class: `ordinary`
```

`bounded` means review only the named change and the behavior it directly
affects. Only a user request that begins with the explicit command “do a
widespread search” changes this row to `widespread`; such a search is Low and
waits for every open Critical, High, and Medium ticket. Quoting, negating, or
mentioning that phrase later does not widen the request. Plans without a Red
Team use `not-used`.

Open the exact work folder whose path appears in the source note's `Execution
checkout`, then confirm the saved severity:

```bash
python3 ai/tools/handoff_router.py \
  --note ai/notes/version-flag.md \
  --severity high
```

Success copies the Implementer prompt, prints numbered progress lines, waits
for the Implementer return, runs the named checks, and copies those records to
the Architect for the audit. It writes supporting copies under
`ai/notes/relay/`. It does not put Red Team between the Implementer and the
Architect, start a mailbox watcher, or create a mailbox request.

If the saved plan includes Red Team, the Architect first audits the returned
work. An Architect `GO` ends the audit; it does not merge, commit, update a
reference, push, or touch the user's checkout. In a watched run, the parent
daemon creates the distinct landing only after the Architect process exits,
then queues the separate Red Team handoff for that landing. The manual relay
itself does not perform that daemon-owned landing. The later review is optional
advice and cannot delay the audit decision or local landing.

Here `--severity high` confirms the Architect's saved value. It cannot change
the value or add a Red Team to another role plan. The router refuses a
mismatch before it changes the clipboard.

Copy those generated prompts and returned blocks unchanged. The user is a
courier in this manual mode, not the author of an Implementer or Red Team
instruction. Send any new request or correction to the Architect first.

</details>

### Record a Red Team finding

The Red Team records the user's minimum severity, its own rating, the bug's
likelihood, and the evidence. A finding that should become separate work
begins with `Backlog action: NEW TICKET`.

The Architect immediately adds that finding to the backlog so it is not lost.
This first step is bookkeeping, not a new investigation. When the ticket later
reaches the correct priority, the Architect reads the cited Red Team note and
keeps, raises, or lowers the rating with a reason. Only the Architect makes
the final `GO` or `NO-GO` decision. Red Team never edits the backlog.

### Keep a normal review narrow

`--severity` changes the minimum harm for a new ticket. It does not ask for a
broad search. Red Team still reviews only the named change unless the user
asks the Architect for a widespread search.

A request beginning with “do a widespread search” is automatically Low and
waits until no Critical, High, or Medium ticket remains open. Low tickets do
not block it.

Fix-only mode and `--skip-redteam` still take priority. A new discovery is
also refused while ten or more Critical, High, and Medium tickets are Open;
finish accepted non-Low work first.

<a id="fix-only-watches"></a>
## Fix-only maintenance

Fix-only maintenance uses two commands. First, save one general request:

```bash
python3 ai/tools/mailbox_daemon.py --send architect --fix-only true
```

This command starts no AI role and accepts neither `--unit` nor `--severity`.
Fix-only copies a valid sealed backlog into the Architect worktree. Without one,
it creates no request. Restarting after a failure retries the exact request and
preserves a duplicate.

Start that watcher separately:

```bash
python3 ai/tools/mailbox_daemon.py --watch --fix-only true \
  --severity high --cycle 5 --max 10000
```

The watcher owns the run settings:

- `--severity high` allows Critical and High bug fixes;
- `--severity medium` also allows Medium bug fixes;
- `--severity low` also allows Low bug fixes;
- `--cycle 5` permits at most five completed tickets; and
- `--max 10000` limits each ticket's added plus deleted characters.

Only `BUG FIX` entries qualify. New functionality, new Red Team discovery,
and parked `LOW — EDGE CASE` work do not. The Architect must select the first
eligible Open bug in backlog order.

After the Architect sends an Implementer plan, the daemon saves the same
general request again. It waits while that ticket is being implemented,
audited, or reviewed, then asks the Architect for the next eligible bug.

A positive `--cycle` value is a cap. When the cap is reached, the next general
request remains queued for a later fix-only watch. Use a positive value when
Open feature tickets remain: `--cycle 0` waits for every Open backlog item,
including features that fix-only mode does not select.

Both a watch without `--fix-only true` and `--once` leave the request
untouched.
Add `--skip-redteam` to use only Architect and Implementer.

## Limit the size of one ticket

Use `--max` when you want the watcher to tell every role the maintenance
limit and require the Architect to reject a ticket that changes too much
text:

```bash
python3 ai/tools/mailbox_daemon.py --watch --max 1200
```

This gives each ticket handled by that run a limit of 1,200 changed
characters. The number is the characters added plus the characters removed
since the full starting commit in the Architect's directive. Code, tests,
comments, command help, and documentation all count. Replacing one character
with another counts as two: one removed and one added.

The Implementer first checks the complete proposed result at a clean current
`HEAD`. Its folder must have no changes to files Git already knows about and
no new files that Git would include in a commit but that are not yet saved:

```bash
python3 ai/tools/ticket_change_guard.py \
  --repo "$PWD" \
  --base FULL_COMMIT \
  --max 1200
```

Here, `FULL_COMMIT` is the ticket's 40-character starting commit. This default
form is deliberately strict about the Implementer's current folder: the
commit at `HEAD` and the visible files must describe the same candidate.

The Architect audits a different boundary. The Implementer may already be
working on ticket B while the Architect audits ticket A. The Architect must
therefore name ticket A's exact saved candidate instead of relying on a
moving `HEAD`:

```bash
python3 ai/tools/ticket_change_guard.py \
  --repo "$PWD" \
  --base FULL_BASE_COMMIT \
  --architect-audit \
  --candidate FULL_CANDIDATE_COMMIT \
  --max 1200
```

Both commit names must contain all 40 hexadecimal characters. The base is the
version before ticket A began. The candidate is ticket A's proposed saved
version. `--architect-audit` and `--candidate` must appear together; an
abbreviated name or `HEAD` is refused.

For example, suppose ticket A adds 700 characters and ticket B later adds 400
more. Comparing A's base with the newer implementation `HEAD` would count
1,100 characters and mix two tickets. Naming A's 700-character candidate
keeps the A audit at 700 even after B exists. A commit name identifies saved
content and does not move when another commit is created.

The watcher supplies an isolated Architect audit worktree. The Architect runs
A's tests there while the Implementer edits and tests B in the implementation
worktree. The separate folders prevent one role's temporary files or branch
movement from changing the other role's work. The character guard still names
A's full candidate explicitly; the separate folder is not a substitute for
`--architect-audit --candidate FULL_CANDIDATE_COMMIT`.

A result within the limit begins with:

```text
ticket change guard: within limit
```

The command then prints the starting commit, proposed commit, limit, and the
added, deleted, and total character counts. It exits with code 0 when the
proposed commit is within the limit, code 1 when it is over the limit, and
code 2 when the repository state or changed file cannot be measured. Only the
Architect turns that evidence into `GO` or `NO-GO`.

The exact boundary is allowed: a count of 1,200 satisfies `--max 1200`, while
a count of 1,201 does not.

`--max 0`, the default, sets no size limit. It does not weaken the readability
review. The Architect and Red Team must never save characters by shortening
clear names, packing statements together, removing explanations or tests, or
leaving a partial fix. The Python must remain readable to a C programmer and a
physics undergraduate learning Python. If a complete readable ticket cannot
fit, the Architect records `NO-GO` and asks the user for a smaller ticket or a
larger number.

For `mailbox_daemon.py`, `--max` applies to `--watch` and `--once`. A
`--send` command only saves a request; the later watcher or one-time run
supplies the policy. A manual clipboard run uses the same limit this way:

```bash
python3 ai/tools/handoff_router.py \
  --note ai/notes/version-flag.md \
  --max 1200
```

The number must match the directive before the router changes the clipboard.
`--max 0` and `--cycle 0` are different: the first removes the character
limit, while the second waits for all enabled work to finish before exiting.
Repeat the same `--max` when restarting work on a ticket. To change the
number, ask the Architect to revise and recheck the directive before more
implementation work begins.

The [ticket-size appendices](#appendices-about-ticket-size) explain file moves,
text formats, unsaved files, and other counting details.

## Runtime controls

| Concern | Options | Default |
| --- | --- | --- |
| Architect and Implementer models | `--architect-model`, `--implementer-model` | `claude-fable-5`, `claude-opus-4-8` |
| Implementer service | `--implementer-provider` | `claude`; choose `ollama` for an Ollama-served open-weight model |
| Claude effort | `--fable-effort`, `--opus-effort` | `xhigh`, `max`; Implementer effort is left to Ollama when that provider is selected |
| Sol effort | `--sol-effort` | `xhigh` |
| Roles used | `--skip-redteam`, `--no-red-team` | Architect + Implementer + advisory Sol Red Team |
| Implementer complexity review | automatic | pause after 90 minutes |
| Implementer context replacement | automatic | save an exact handoff before automatic compaction |
| AI job emergency timeout | `--dispatch-timeout` | 120 minutes |
| Compaction point inside one long role turn | `--claude-context`, `--sol-context` | 500000 tokens each |
| Watch lifetime | `--cycle` | omitted: indefinite; `N>0`: stop after N completed ticket cycles; `0`: finish all recorded work and then stop |
| Text changed by one ticket | `--max` | `0`: no character limit |
| Minimum severity for new discovery tickets | `--severity` | `medium` |
| Discovery policy | `--fix-only` | off |

The 90-minute review and the 120-minute timeout solve different problems. The
review waits for a tool action to finish, asks the Implementer for a concise
progress handoff, and keeps the same ticket open. The timeout is a later
fallback for an AI program that no longer responds; it stops that process and
moves the request to `failed/` when the move can be confirmed.

### If an AI account runs out of tokens

When Claude or Sol reports an account limit, running jobs finish. The watcher
exits status 1: `Error: <role> is out of tokens`. The request remains in
`failed/`, or `inflight/` if that move cannot be verified; logs and worktree
edits remain. Add credits, inspect edits, then requeue. It never resets or
commits them.

`ai/notes/role-contract.yaml` records these stable timing facts together with
role permissions and landing authority. The tools check that machine contract
instead of relying on a README value. Only the Architect may edit it, through
protected-policy administration; Implementer and Red Team access is read-only.

### Change the machine contract safely

The YAML file is the source for settings that the Architect may configure.
The watcher reads one copy when it starts. For example, adding `.ai-secrets/`
to `candidate_forbidden_prefixes` does not require a matching Python constant.
As soon as a protected update replaces that copy, the watcher finishes no
other message and asks the user to restart it.

Some identities are not ordinary settings. Python fixes the contract's own
location, the trusted tool and role files, the backlog location, and the saved
worktree layout. It also fixes the maximum bytes read before parsing, the
schema versions understood, and Git rules such as “never force-push.” The
reader refuses a YAML edit that removes or redirects one of these boundaries.
Changing one requires a separate code migration with tests for the saved
state or protocol it affects.

A future change to the YAML's shape uses `schema_version` in three steps:

1. Stop the watcher and release a reader that accepts both the old and new
   schema.
2. Update the YAML through protected-policy administration, then restart the
   watcher.
3. Remove old-schema support in a later maintenance change.

The contract does not list compatible daemon versions. A new YAML file cannot
prove that an older Python reader understands a field that did not exist when
that reader was written.

Model selection, provider selection, and role authority are independent.
Choosing Sonnet does not silently lower a Claude Implementer's effort. Ollama
models use their provider's own reasoning behavior, so `--opus-effort` is not
passed to an Ollama Implementer.

Each model option accepts one nonempty value containing no spaces, tabs, line
breaks, or hidden zero character, such as `sonnet` or `glm-5.2:cloud`. This checks
only the name's format. The selected provider confirms whether the model
exists when a live check or role starts.

The mailbox addresses `fable` and `opus` continue to mean Architect and
Implementer even when the Implementer comes from Ollama. There is no
`--sol-model` option. The program currently pins Sol to `gpt-5.6-sol`. The
watch can include the advisory Red Team or omit it with `--skip-redteam`, but
Sol's role itself does not change.

There are two role setups. The default uses Architect, Implementer, and the
advisory Red Team. `--skip-redteam` uses only Architect and Implementer. Sol
never becomes an Implementer.

### Does an old ticket consume the next ticket's context?

No. Every mailbox dispatch starts a new provider conversation. The daemon
marks Claude turns as non-persistent and Sol turns as ephemeral. No later turn
can resume one of these conversations.

For example, after the Implementer returns a candidate commit, the
Architect's audit starts with an empty provider conversation and reads the
saved handoff and repository evidence. After that ticket closes, the next
ticket also starts empty.

The daemon does not send a separate `compact` request at ticket closure. Such
a request would open another empty provider conversation and spend a role turn
without helping the next ticket.

The context options still protect one unusually long role turn. When that
single turn reaches the chosen token count, its provider summarizes older text
and continues. Claude receives `CLAUDE_CODE_AUTO_COMPACT_WINDOW`; Sol receives
`model_auto_compact_token_limit`. Raising the limit or splitting an oversized
ticket is the remedy when one turn compacts before that ticket finishes.

For an Implementer turn, the
[Claude Code `PreCompact` hook](https://code.claude.com/docs/en/hooks#precompact)
first asks the Implementer to save this small record:

```text
### IMPLEMENTER_HANDOFF: CONTEXT HANDOFF

- Ticket and cycle
- Base commit
- Current worktree HEAD
- Candidate created: yes or no

Completed
Known failures
Rejected approaches
Uncommitted changes
Next exact action
Do not revisit
```

Every named section contains a concrete bullet or `none`. The watcher checks
the cycle, base commit, current Implementer commit, and whether the declared
uncommitted work agrees with Git. It then sends the exact record to the
Architect through the existing checkpoint route. This record is not candidate
C and does not complete a cycle.

If the Architect permits continuation, the replacement Implementer receives
the path of the saved record. It reads that file and the repository instead of
receiving a summary written by the watcher. A listed rejected approach stays
off limits unless the Architect explicitly reopens it. If the provider stops
before writing the record, the ordinary out-of-token recovery above preserves
the files without inventing missing history.

On every pass, the watcher checks when `mailbox_daemon.py` was last modified.
If the file changed, the running watcher exits. Start it again to load the new
code.

## Exact command reference

The program's live help contains its current options:

```bash
python3 ai/tools/mailbox_daemon.py --help
```

The current transcript is kept here for offline reading and regression checks.

<details>
<summary>Current <code>mailbox_daemon.py --help</code> transcript</summary>

```
usage: mailbox_daemon.py [-h] [--dry-run] [--once] [--clean-all]
                         [--restart-implementer] [--restart-redteam] [--watch]
                         [--cycle count] [--max characters] [--skip-redteam]
                         [--fix-only value] [--send {architect}] [--ping]
                         [--unit UNIT] [--severity {high,medium,low}]
                         [--architect-model MODEL] [--implementer-model MODEL]
                         [--fable-effort {low,medium,high,xhigh,max}]
                         [--opus-effort {low,medium,high,xhigh,max}]
                         [--sol-effort {none,low,medium,high,xhigh}]
                         [--dispatch-timeout MINUTES]
                         [--claude-context TOKENS] [--sol-context TOKENS]

save mailbox requests and start the assigned role for each request

options:
  -h, --help            show this help message and exit
  --dry-run             show the message files and work this command would
                        handle, but do not start a role or write a message
                        file
  --once                start every request that is waiting now, then exit
  --clean-all           permanently discard every local AI worktree and local
                        claude/*, codex/*, or legacy worktree-agent-* branch;
                        dirty files and unmerged commits in those worktrees
                        are lost
  --restart-implementer
                        after an interrupted Implementer turn, discard its
                        partial work and requeue the exact Architect handoff
  --restart-redteam     after an interrupted Red Team turn, discard its
                        partial work and requeue the exact Architect-to-Red-
                        Team handoff
  --watch               check the mailbox every 20 seconds and start waiting
                        requests
  --cycle count         with --watch, stop after this many completed ticket
                        cycles; one cycle is always one ticket; with Red Team
                        it ends when the matching review returns for daemon-
                        recorded local landing L, and without Red Team it ends
                        when the daemon records local landing L; 0 instead
                        waits until no enabled role has a waiting message and
                        ai/notes/backlog.md has no open item; omit this option
                        to keep watching
  --max characters      with --watch or --once, limit each ticket to this many
                        added and removed characters, counted from the
                        starting saved Git version named in the Architect's
                        instructions; use only digits 0 through 9; 0 means no
                        limit (default: 0)
  --skip-redteam, --no-red-team
                        with --watch, start Architect and Implementer jobs but
                        no Red Team job for ordinary tickets; protected
                        control-plane tickets become BLOCKED_RED_TEAM_REQUIRED
                        before Implementer dispatch and resume on a later
                        watch without this option; with --ping, check the
                        Architect and Implementer providers but not Sol
  --fix-only value      with --send architect, save a backlog-repair request;
                        with --watch, run existing bug fixes at the watcher's
                        severity; the value accepts 1, true, or yes in any
                        capitalization
  --send {architect}    save the user's ticket request for the Architect and
                        exit
  --ping                make one small live request to every provider selected
                        for this run and exit; add --skip-redteam to omit Sol
  --unit UNIT           the user's request text for --send architect; include
                        the path to its source note in ai/notes/
  --severity {high,medium,low}
                        minimum severity for new discovery tickets: high keeps
                        only bugs that severely impact core functionality,
                        cause data loss, halt system operations, or make the
                        science wrong; medium also keeps probable normal-
                        operation bugs but not improbable edge cases; low
                        keeps every concrete discovered bug; with --send
                        architect, save the choice for that request (default:
                        medium)
  --architect-model MODEL
                        Claude model alias or full name used for the
                        Architect; mailbox filenames for this role still
                        contain fable (default: claude-fable-5)
  --implementer-model MODEL
                        model name used for the Implementer; select its
                        service with --implementer-provider; mailbox filenames
                        still contain opus (default: claude-opus-4-8)
  --implementer-provider {claude,ollama}
                        service used for the Implementer: claude or ollama;
                        the Architect remains on Claude (default: claude)
  --fable-effort {low,medium,high,xhigh,max}
                        claude CLI reasoning effort for the Architect
                        (default: xhigh)
  --opus-effort {low,medium,high,xhigh,max}
                        claude CLI reasoning effort for the Implementer
                        (default: max)
  --sol-effort {none,low,medium,high,xhigh}
                        codex CLI reasoning effort for Sol as Red Team
                        (default: xhigh)
  --dispatch-timeout MINUTES
                        stop a running role after this many minutes and try to
                        move its request file to failed/; if the result or
                        move cannot be verified, the file may remain in
                        inflight/ for inspection (default: 120)
  --claude-context TOKENS
                        inside one Architect or Implementer turn, ask the
                        coding runtime to replace older context with a shorter
                        summary at this many tokens (default: 500000)
  --sol-context TOKENS  inside one Red Team turn, ask Codex to replace older
                        conversation text with a shorter summary at this many
                        tokens (default: 500000)
```

</details>

### Action rules

- Choose only one of `--once`, `--watch`, `--clean-all`,
  `--restart-implementer`, `--restart-redteam`, `--send`, and `--ping` in one
  command.
- `--clean-all` cannot be combined with `--dry-run`. The explicit flag is the
  user's instruction to discard all local AI work, including dirty or
  unmerged work.
- A restart command cannot be combined with `--dry-run`. It is the user's
  explicit instruction to discard partial work by the named role.
- `--cycle` accepts an integer from 0 through 1,000,000 and is valid only with
  `--watch`.
- Omitting `--cycle` watches indefinitely. `--cycle 0` instead waits until the
  enabled roles have no waiting messages and no backlog line begins `- OPEN`.
- `--max` accepts digits from 0 through 9 and is valid with `--watch` or
  `--once`. Omitting it or writing `--max 0` sets no character limit.
- `--skip-redteam` and `--no-red-team` are two names for the same setting.
  With `--watch`, they disable Red Team for ordinary tickets; a protected
  control-plane ticket is saved as `BLOCKED_RED_TEAM_REQUIRED`. With `--ping`,
  they omit the Sol connection check.
- Positive cycle limits work with both role setups. In a two-role watch, the
  daemon's recorded local landing completes that one-ticket cycle.
- A two-role watch preserves waiting internal Sol files and refuses new
  role-to-role Sol files until that watcher stops and releases its saved
  two-role rule.
- `--unit` is required for an ordinary Architect send. The exact
  `--send architect --fix-only true` shorthand forbids `--unit`.
- The only public send target is `architect`. The connection check has no role
  target: write `--ping`, not `--ping architect` or `--ping sol`.
- `--severity` accepts `high`, `medium`, or `low`. An ordinary Architect send
  may save it for a discovery request. The fix-only shorthand forbids it;
  the later watcher supplies the maintenance threshold.
- `--dispatch-timeout`, `--claude-context`, and `--sol-context` accept integers
  from 1 through 1,000,000.
- For actions that exit on their own, `--dry-run` prints the proposed action
  without writing workflow files.
- Malformed model values, invalid effort or context values, invalid timeouts,
  and invalid action combinations fail before the tool writes a mailbox file.

The manual relay has separate live help:

```bash
python3 ai/tools/handoff_router.py --help
```

Its `--mode`, `--skip-redteam`, and `--severity` options confirm the matching
values in the Architect's validated role plan. They cannot change that plan.
These options apply to one clipboard relay, not a running watcher.

Every tool has live help. These commands only print and exit:

```bash
python3 ai/tools/mailbox_daemon.py --help
python3 ai/tools/handoff_router.py --help
python3 ai/tools/handoff_contract.py --help
python3 ai/tools/permanent_note_guard.py --help
python3 ai/tools/backlog_guard.py --help
python3 ai/tools/ticket_change_guard.py --help
python3 ai/tools/backlog_bundle.py --help
python3 ai/tools/backlog_bundle.py pack --help
python3 ai/tools/backlog_bundle.py inspect --help
python3 ai/tools/backlog_bundle.py unpack --help
```

# Common questions raised by developers

The ordinary workflow does not require these sections. Read only the question
that matches the command or problem currently on screen.

## Appendices for AI roles and maintainers <a id="appendices-for-ai-roles-and-maintainers"></a>

These sections explain checks normally run by the Architect, Red Team,
or the watcher. A user sending an ordinary ticket does not need to perform
them.

### Check a handoff directive

The Architect or Red Team uses `handoff_contract.py` to check that a detailed
instruction note has all required parts before another role receives it. The
program checks structure, not scientific correctness. Only the Architect
decides `GO` or `NO-GO`.

Replace `<ticket>` with the real source-note filename:

```bash
python3 ai/tools/handoff_contract.py architect ai/notes/<ticket>.md
python3 ai/tools/handoff_contract.py redteam ai/notes/<ticket>.md \
  --severity medium
```

The result is `VALID` when required sections appear in order. The note must
name exact files and tests, number its work steps, show the commands to run,
and include acceptance checkboxes. `INVALID` explains the missing or
misordered part.

Every Architect directive also has four exact `Role plan` rows. This ordinary
example uses all three roles:

```markdown
- Roles: `Architect + Implementer + Red Team`
- Discovery severity: `medium`
- Review scope: `bounded`
- Ticket class: `ordinary`
```

Only the Architect may replace the final value with
`protected-control-plane`. The row is validated data, not a label the
Implementer may add after editing begins.

For a Red Team directive, replace `medium` with the setting chosen for the
run. A mailbox job supplies the same setting through
`MAILBOX_DISCOVERY_SEVERITY`.

#### Show exactly where work belongs

An Architect directive names the Implementer's extra Git-managed project
folder, its branch, and the full 40-character name of the saved version where
the ticket starts. Every requested edit and test also names a real location:

```markdown
- `ai/tools/mailbox_daemon.py::agent_preamble`: Keep `agent="user"` invalid.
  Require a `ValueError` containing `unknown mailbox agent`, without changing
  the accepted `fable`, `opus`, or `sol` cases.
- `ai/tests/test_role_directive_contract.py::RoleDirectiveContractTests`:
  Test the invalid `user` input, the error text, and the unchanged valid roles.
```

Here, `path::name` means the file followed by the function, class, section, or
test to inspect. The real note uses locations for its own ticket; it does not
copy this example for unrelated work.

<a id="ticket-file-scope"></a>
#### What happens if a candidate changes an unplanned file?

Before the Implementer starts, the watcher saves the exact files named under
`Files and symbols` and `Tests to write`. This becomes that ticket's file
list. The watcher later compares the complete proposed change with the saved
list.

For example, suppose a plan names `emulator/training.py` and
`ai/tests/test_training.py`, but the proposed change also edits
`emulator/model.py`. The watcher preserves the proposal and reports:

```text
SCOPE_EXCEEDED
paths: 'emulator/model.py'
```

The Architect must then accept that exact expansion or send the ticket back
for repair. The Implementer cannot silently enlarge the plan. A change to a
globally protected control file is stricter. An `ordinary` ticket reports
`PROTECTED_PATH_VIOLATION` and refuses the proposal, even if the directive
accidentally named that file. The watcher never turns an ordinary ticket into
a protected ticket after seeing the Implementer's edits. A proposal
containing only planned files is reported as `IN_SCOPE`.

<a id="protected-control-plane-tickets"></a>
#### How does a protected control-plane ticket work?

A **protected control-plane ticket** changes the code or policy that enforces
the workflow itself. The protected set comes from
`ai/notes/role-contract.yaml` and its trusted path groups. It includes the
mailbox daemon, candidate-admission and C-to-L landing code, role and handoff
validators, durable recovery logic, permanent-note guard code, and trusted
workflow test harnesses. The eleven permanent notes, role instructions,
machine authority contract, and protected failure-mode catalog remain on the
separate Architect-only protected policy route. An Implementer candidate may
never edit those files.

The validated Architect directive adds one exact row under `### Role plan`:

```markdown
- Ticket class: `protected-control-plane`
```

An ordinary directive uses the same row with the value `ordinary`. Only the
Architect chooses the value. The Implementer copies it unchanged and cannot
promote an ordinary ticket after discovering that a protected file would be
convenient to edit.

##### Why the guide calls the controllers D0 and D1

- **D0** is the daemon and validators from the trusted `main` commit that
  started the watch.
- **D1** is the proposed replacement code inside immutable candidate C.

D1 is untrusted until the protected workflow finishes. It cannot approve C,
write the live landing journal, create L, update `main`, or replace the tests
that decide whether it is safe. D0 independently checks the changed paths,
decision identities, test result, landing parent, and final delta.

##### The two pre-landing decisions

The Architect first audits immutable C and sends `GO(C)`. D0 then asks Red
Team to inspect that same candidate before any landing exists. A Red Team
acceptance has these exact structured fields:

```text
MAILBOX-RETURN: redteam-control-plane
MAILBOX-CYCLE: TICKET-ANCHOR@FULL-STARTING-COMMIT
MAILBOX-CANDIDATE: FULL-40-CHARACTER-C
MAILBOX-RESULT: ACCEPT-CONTROL-PLANE
```

Use `REJECT-CONTROL-PLANE` in the final row when the evidence requires repair.
Another candidate hash, another cycle, a shortened hash, a missing decision,
or Architect `NO-GO` cannot satisfy the gate. Red Team cannot land the
candidate; it supplies only the second required key for this ticket class.
For ordinary tickets, Red Team remains advisory and reviews L after landing.

After both exact decisions exist, D0 runs D1 in an isolated temporary Git and
mailbox environment. The trusted test driver comes from D0, so candidate C
cannot weaken the test that judges C. D0 checks identity mismatches, scope,
restart behavior, skipped-Red-Team behavior, safe landing, and isolation from
the live journal. A failure preserves C and the evidence and creates no L.

When the shadow test passes, D0 creates L with the existing safe landing
mechanism and advances `main` automatically. The user is not asked to approve
the landing. If `main` advanced meanwhile, the candidate and both decisions
remain saved and the watcher reports
`STALE — REQUIRES INTEGRATION REVALIDATION`; it never overwrites newer work.
D0 sends the Architect a same-cycle check of C on the newer main. After a
fresh GO for that combined result, D0 retires only the private stale L,
rebuilds L on the approved main version, reruns the trusted shadow checks,
and lands automatically.

##### What happens with `--skip-redteam`?

Ordinary tickets keep their existing two-role behavior. A protected ticket
cannot start without Red Team. Before dispatching the Implementer, D0 records:

```text
BLOCKED_RED_TEAM_REQUIRED
```

The source request and the exact Architect handoff remain saved. No
Implementer process starts, and no candidate, landing, trusted ref, or
landing-journal entry is created for the blocked attempt. The watcher explains
the setting and continues with compatible ordinary work. Restart the watcher
without `--skip-redteam`; D0 then recovers the saved protected handoff instead
of requiring the user to rewrite it.

##### What if the newly landed controller fails its health check?

After L reaches `main`, D0 performs a bounded, non-destructive check: the role
contract loads, enforcement agrees with it, durable state parses, the daemon
recognizes both ticket classes, and ownership still prevents two live writers.
A failure records:

```text
CONTROL_PLANE_HEALTH_FAILED
```

This state preserves L and every log, refuses new state-changing work, and
leaves the trusted recovery path available. It does not reset `main`, rewrite
history, repeat the landing, or crash. Read the exact health log named by the
watcher, correct the reported condition through the recovery command it
prints, and restart. A restart must find the saved landing and health state;
missing Red Team evidence is never reinterpreted as acceptance.

#### Record whether helpers add value

A **subagent** is a short-lived helper given one small task.
It is not another mailbox role. One helper might reproduce a parser error
while another checks one named function. The plan says what each helper
returns and when it must stop. An Integrator step tells the Implementer how to
combine and check those returns.

Only the Architect decides whether helpers add independent value. A required
plan names each bounded job. A no-helper plan contains one concrete reason why
a separate session would repeat the same work without independent evidence.
The Implementer cannot invent or rewrite that reason.

The manual relay compares a required plan's helper names with the Implementer's
`Subagent work` evidence. Missing, renamed, reordered, or extra helper results
are refused. For a no-helper plan, it requires the exact saved reason instead.
The Architect still reads that evidence and makes the decision.

If the first required helper cannot start before editing begins, the evidence ends with
the fields `Capability checked`, `Attempted operation`, and `Raw failure` in
these exact rows:

```text
- Capability checked: `the exact launch capability`
- Attempted operation: The concrete first subagent launch attempted before editing.
- Raw failure: `the unchanged first runtime failure`
```

A **fingerprint** here is a SHA-256 identifier calculated from the exact text.
The router fingerprints the blocked `IMPLEMENTER_HANDOFF`, including its
`Subagent work` evidence. A later Architect exception copies these three rows
and cites that identifier; it cannot invent replacement evidence.

#### Check the ticket-size setting

When a watcher or manual relay uses a positive `--max`, pass the same number
to the contract check:

```bash
python3 ai/tools/handoff_contract.py architect ai/notes/<ticket>.md \
  --max 1200
```

A different number is `INVALID`. The Implementer must not guess which limit
applies.

### Check protected project notes

Only the Architect uses this result in a `GO` or `NO-GO` decision. The command
below checks that the eleven permanent notes, `ai/notes/role-contract.yaml`,
and the protected failure-mode catalog still match the saved Git version where
the work began:

```bash
BASE="$(git rev-parse HEAD)"
python3 ai/tools/permanent_note_guard.py \
  --repo "$PWD" \
  --base "$BASE"
```

`HEAD` is Git's short name for the saved version currently open in this
folder. A real ticket uses the full starting version recorded before work
began, which may be older. Success ends with:

```text
PERMANENT-NOTE-GUARD PASS base=... notes=11
```

A passing result proves only that the protected files match. It does not
approve the ticket.

#### Land an Architect-only protected-rule update

Only the Architect may edit and commit the eleven permanent notes,
`ai/notes/role-contract.yaml`, and the three role files, and
only through protected-policy administration. The YAML holds configurable
machine settings inside the fixed safety boundaries described above. This
work is separate from an Implementer ticket and consumes no cycle slot.

When Red Team is enabled, it reviews the exact draft once before the
Architect's final decision. It returns one advisory `GO` or `NO-GO`
recommendation and may recommend a smaller change, but it cannot edit the
files or veto the Architect. There is no second review round, including after
a correction. With `--skip-redteam`, the Architect records that the review
was unavailable.

The internal check names two saved versions:

- B is the local `main` commit before the note edit.
- P is the clean Architect coordination `HEAD` after the note update is
  saved. P has one parent: B.

Inside an already bound Architect process, the Architect may queue the update
with:

```bash
python3 "$MAILBOX_PRIMARY_WORKTREE/ai/tools/handoff_router.py" \
  --architect-notes-admin "PLAIN-LANGUAGE SUMMARY"
```

This is not a public user command. It requires `MAILBOX_ROLE=architect`, the
saved Architect path, and the saved shared-notes path. Its request begins:

```text
MAILBOX-ADMIN: permanent-notes

PLAIN-LANGUAGE UPDATE
```

The watcher supplies exact B through `MAILBOX_NOTES_BASE`. If the Architect
creates P, the job returns exactly these four lines:

```text
MAILBOX-RETURN: architect-notes-go
MAILBOX-BASE: FULL-B-FROM-MAILBOX_NOTES_BASE
MAILBOX-NOTES-COMMIT: FULL-P
MAILBOX-DECISION: GO
```

If no protected rule needs to change, the Architect leaves `HEAD` at B and
creates neither a daemon request nor an Implementer request.

After the Architect exits, the parent watcher checks B, P, and every changed
path. It refuses while ordinary ticket work is active. It also checks the
three role folders before updating local `main` and their saved starting
versions.

Active ticket work includes **candidate-to-landing recovery**: the saved
interval after the Implementer proposes a version but before the watcher
records the accepted version on local `main`. It also includes a saved
Architect GO whose accepted local version has not yet been recorded.

The next ticket begins from saved version P. The code calls the watcher the
**daemon**; this update ensures that the authoritative daemon and role files
do not stay behind the accepted note update. A role folder with unfinished or
different work is preserved and causes refusal rather than a reset.

In plain terms, before the next ticket starts, the watcher updates every safe,
unused AI folder to P so no role starts from older code.

The watcher tries once to send P without overwriting newer remote work. Git
calls this a non-force push. If the result fails or is uncertain, the watcher
saves a **push-debt record** containing exact P and the command still owed. It
does not rerun the Architect or create a ticket.

### Review a closed ticket

Red Team reviews an accepted ticket after the Architect's decision. It is
advisory: it cannot block or undo the accepted local change.

The sequence is:

1. Implementer saves the proposed change.
2. Architect checks it and records `GO`.
3. Watcher records the accepted version on local `main`.
4. Red Team reviews that exact accepted version.
5. Red Team returns `NO CHANGE` or `REOPEN`.

The AI guide calls the proposed saved version **candidate C** and the accepted
`main` version **landing L**. [FAQ G1](../README.md#faq-g1-candidate-and-landing)
explains why they are different.

A `NO CHANGE` return means no remaining bug was found. It is a status report,
not approval. A `REOPEN` return names a concrete failing input, command, or
scientific result.

#### Preserve the Red Team investigation

Before `REOPEN` or `NEW TICKET`, Red Team writes a detailed local note such as
`ai/notes/cmb-axis-red-team-finding.md`. The note explains:

- expected and observed behavior;
- affected code and a reproduction command;
- realistic impact and uncertainty; and
- the check that would prove a repair.

The Architect links that note from the backlog. This saves Architect tokens
because the later audit does not have to reconstruct the investigation.

#### Count repeated reopen requests

Every ticket starts with `Red Team reopen count: 0`. Each formal `REOPEN`
raises the count by one. The Architect first records the request, then audits
it when the ticket reaches the front of the allowed priority order.

- Architect `GO` accepts the evidence and leaves the ticket Open for repair.
- Architect `NO-GO` closes it and records `Red Team reopening: barred by
  Architect NO-GO`.
- Red Team cannot reopen that same ticket again after the bar. A different
  defect requires `NEW TICKET`.
- At count `6`, the ticket automatically becomes Low.

For example, a loader ticket with count `0` is closed. Red Team supplies a
one-row input that still crashes, so the Architect records count `1`. A second
request must add evidence beyond that same row before the Architect considers
more work.

#### Internal landing and recovery rules

<details><summary>Show the Git recovery rules</summary>

The Architect's decision names candidate C by its full saved Git identifier.
That identifier does not change, which is what **immutable C** means here.
After the Architect exits, the **parent daemon**—the watcher that started the
Architect—creates the different accepted version, landing L.

The watcher updates a clean, unchanged user `main` to L without rewriting
saved history. Before another role starts, it also updates every safe, clean,
idle role folder to L. A dirty, different, or active folder is preserved and
causes refusal rather than a reset.

The authoritative daemon and role files do not stay behind the accepted code.
After L, the watcher makes one non-force push attempt: one try to send L
without overwriting newer remote work. An uncertain or failed attempt creates
`pending-main-push-<L>.txt`. This **push-debt record** names the exact version
and command still owed; it does not reopen the ticket or repeat the landing.

In plain terms, before the next ticket starts, the watcher updates every safe,
unused AI folder to L so no role starts from older code.

If the process stops after C but before L, recovery uses that ticket's saved
candidate and landing records. It does not compare the separate Architect
coordination branch with `main` or create work from a changed-line threshold.

The Red Team return completes a three-role cycle. It never blocks the local
landing, but a finite watcher waits for that return before it exits for the
cycle count. Without Red Team, the cycle ends when the watcher records L.

</details>

### Protect the tracked backlog

Git saves completed versions of `ai/notes/backlog.md`. While a ticket is
active, `backlog_guard.py` stores a local SHA-256 fingerprint for the exact
version accepted by the Architect. The fingerprint exposes an accidental
change before the daemon includes the sealed backlog in the ticket landing.

The Architect performs three steps:

1. After creating the backlog, read every entry and initialize the guard:

   ```bash
   python3 ai/tools/backlog_guard.py initialize --architect-ack
   ```

2. Before reading another role's result or editing the backlog, check it:

   ```bash
   python3 ai/tools/backlog_guard.py check
   ```

3. After one deliberate edit, seal the new version with the previous
   64-character value printed by `check`:

   ```bash
   python3 ai/tools/backlog_guard.py seal \
     --previous-sha256 SHA256 \
     --architect-ack
   ```

Initialization success includes:

```text
Backlog guard initialize passed.
accepted SHA-256: SHA256
BACKLOG-GUARD-INITIALIZE PASS sha256=SHA256
```

If `check` reports a mismatch, stop and inspect the backlog. Do not approve
unknown text by replacing the saved value. `seal` also refuses a different
previous value, a linked file, or a file that changes while it is read.

Implementer and Red Team may run `check`, but they never edit the backlog and
never run `initialize` or `seal`. This protects against accidental edits, not
a deliberately malicious program that rewrites both local files.

## Appendices about ticket size <a id="appendices-about-ticket-size"></a>

### FAQ A1. Exactly what does `--max` count? <a id="faq-a1-max-count"></a>

`--max` counts characters added plus characters removed. For example, adding
100 characters and deleting 20 gives a total of 120. Spaces and line breaks
count too.

#### Compare two saved versions

The tool compares the saved project version where the ticket started with the
saved version proposed as the finished ticket. Temporary edits removed before
the proposed version do not count.

The Implementer checks the latest saved version in a clean work folder. The
Architect checks the exact saved version named by `--candidate`, so later work
cannot change the measurement.

#### Count the changed text

Every added and deleted text character counts, including spaces and line
breaks. The guard counts Unicode characters rather than storage bytes, so
replacing `a` with `π` counts as two: one deleted and one added. Code, tests,
comments, command help, and documentation follow the same rule.

Moving an unchanged file and changing only its file permission both count
zero. When a moved file also changes, run the tool instead of estimating the
result from the filenames.

<details><summary>How does the tool calculate the exact count?</summary>

Git calls the saved version currently open in a worktree `HEAD`. A
**worktree** is another project folder managed by the same Git repository.
The Implementer's default proposal is a clean current `HEAD`; the Architect's
proposal is the full commit named by `--candidate`.

For a changed file, the guard keeps the longest ordered sequence of
characters shared by the starting and final text. The remaining starting
characters are deletions, and the remaining final characters are additions.
This gives the smallest exact insertion-and-deletion count even when a line
contains repeated characters.

Git may treat a moved and edited file as one deleted file plus one new file.
The tool applies the same character comparison to the exact saved versions.

</details>

### FAQ A2. Why can the size check refuse a ticket? <a id="faq-a2-max-refusal"></a>

The tool refuses when it cannot measure one complete saved ticket without
guessing. The usual causes are unsaved work, unrelated edits in the same
folder, or a changed file that is not text.

#### First check the saved ticket

With a positive limit, save the complete ticket in one commit and run the
check from a clean Implementer folder. A clean folder has no unsaved edits and
no new files waiting to be included. A newly saved file counts as entirely
added text.

#### Check whether every changed file is text

The tool cannot count a changed non-text file, such as an image or compiled
model. It also refuses text that is not valid UTF-8, the text format used by
this repository, and a changed Git submodule, which points to another
repository. These cases return code 2, and the Architect records `NO-GO` for
a positive limit.

#### Use the candidate named by the Architect

The Architect's `--architect-audit` command reads the exact commit named by
`--candidate`. That commit must belong to the same repository and follow the
ticket's starting commit. The watcher supplies the correct folder and full
commit name; use them as written.

<details><summary>Which resource limits can also cause refusal?</summary>

The guard has fixed memory and time limits. A **blob** is Git's saved contents
for one version of one file.

| Work being checked | The guard refuses above this boundary |
| --- | --- |
| One blob that must be read | 4,194,304 bytes |
| All different blobs that must be read | 16,777,216 bytes |
| Unmatched starting plus final text when both sides have unmatched characters | 200,000 characters |
| Unmatched starting characters multiplied by unmatched final characters for one file | 4,000,000 pairs |
| Those character pairs added across the ticket | 8,000,000 pairs |
| One Git command used by the guard | 30 seconds |

These boundaries prevent a size check from consuming excessive memory or
processor time. Exceeding one returns code 2. Raising `--max` does not bypass
these fixed boundaries.

</details>

When a complete change is too large to measure, split it only at a boundary
where each ticket remains readable, testable, and useful by itself. `--max 0`
turns off the numerical size limit, but it does not remove the readability or
evidence review.

## Appendices about stopping the watcher <a id="appendices-about-stopping-the-watcher"></a>

### FAQ B1. When can I interrupt the watcher? <a id="appendix-b--when-is-it-safe-to-stop-the-watcher"></a>

Press Ctrl-C only while the watcher literally prints `safe to Ctrl-C`. If the
line says that a role is running, starting, or timing out, wait.

#### Match the latest message

Read the watcher's latest status line:

| Printed status | Plain meaning | Safe to press Ctrl-C? |
| --- | --- | --- |
| A periodic progress message, `turn in flight`, or `turns in flight` | An AI role is running | No |
| `dispatch preparation admitted; not safe to stop` | The watcher is starting a role | No |
| `safe interval ended; not safe to stop` | An earlier safe period has ended | No |
| `safe to Ctrl-C` | No AI role is running or starting during the printed countdown | Yes |
| `watcher stopped` | The watcher has already stopped | Yes; no action is needed |
| A timeout message | The watcher is stopping one long-running role and saving its result | No; wait for a later safe or stopped message |

#### After a timeout

Wait for `safe to Ctrl-C` or `watcher stopped`. Then inspect `failed/`, the
folder for requests that did not complete, and `inflight/`, the folder for
requests whose outcome still needs inspection. Do not send the request again
until the saved request and its log show what happened.

### FAQ B2. What does `--cycle` count? <a id="faq-b2-cycle-count"></a>

One cycle always means one ticket. It is not a timer and it does not count
individual AI messages.

#### Choose how long the watcher should run

| Command | What it does |
| --- | --- |
| `--watch` | Keep watching until you stop it during a printed safe countdown |
| `--watch --cycle 1` | Finish one ticket; with Red Team enabled, wait for that ticket's review before exiting |
| `--watch --cycle 2` | Exit safely after two completed cycles, even if more work is waiting |
| `--watch --cycle 3` | Permit three tickets to overlap, but never start a fourth |
| `--watch --cycle 0` | Finish saved requests and their required reviews; exit after no backlog item remains Open |
| `--watch --skip-redteam --cycle 2` | Finish two Architect-and-Implementer tickets; leave Red Team messages for a later run |

The 20-second `safe to Ctrl-C` countdown is separate. It gives a person a
manual chance to stop while no AI role is running or starting. Reaching that
countdown never starts or completes a cycle.

#### Follow one ordinary ticket to completion

1. Architect and Implementer work back and forth on one ticket.
2. Architect accepts the Implementer's saved change.
3. The watcher records that accepted change on local `main`.
4. When Red Team is enabled, it reviews the accepted change and returns
   `NO CHANGE` or `REOPEN`.

Red Team is advisory: its review does not delay or undo step 3. The review is
still the final completion step for that cycle, so `--cycle 1` neither starts
ticket B nor exits while ticket A's review is waiting. In a run without Red
Team, step 3 completes the cycle.

A protected control-plane ticket uses the earlier
[two-key path](#protected-control-plane-tickets): its Red Team decision occurs
before landing, and the cycle finishes only after automatic landing and the
bounded health check succeed. It cannot start in a `--skip-redteam` watch.

An Architect-only update to the eleven permanent notes or the protected
machine role contract is not a ticket. It does not use a cycle and receives no
Red Team closure review.

#### Let different tickets overlap only when the limit has room

With `--cycle 3`, the Implementer may change ticket B while the Architect
checks ticket A and Red Team reviews an earlier accepted ticket. The roles use
different folders or read-only saved versions. A fourth ticket cannot start,
even when one role becomes idle first.

#### Understand `--cycle 0`

`--cycle 0` does not read a backlog description and invent an AI request from
it. Someone must still send the appropriate mailbox message. The roles that
handle that request must also update its backlog entry when the work is
genuinely finished. Zero means “finish every saved request for a role included
in this run.” It does not skip Red Team unless the user selected a run without
Red Team.

Fix-only maintenance has one explicit exception: after the user sends its
general maintenance request, the daemon repeats that same request after each
Implementer handoff. It does not invent the ticket text; the Architect still
chooses the next eligible bug and writes its plan.

Just before a zero-cycle exit, the watcher briefly prevents `--send` from
saving another message. It checks that the backlog is an ordinary readable
text file, that the file did not change during the check, and that no enabled
message is waiting. If it cannot perform those checks, it keeps running
instead of guessing that the work is finished.

#### Read the final status

This line marks a safe countdown:

```text
every enabled role is idle; safe to Ctrl-C for 19s more; 3 messages waiting.
```

The first part means that no enabled AI role is running or starting. The last
number says how many request files still wait, including work for a role that
may be disabled in this run.

While a role is running, a periodic progress message looks like this:

```text
  ... 0046-to-opus.md still running (3 min elapsed, log 12.4 kB; tail -f .../ai/notes/relay/20260714-031840-dispatch-opus.log)
```

It names the request, elapsed time, and current log-file size. The final path
is the log that an experienced user may inspect while the job runs.

```text
cycle limit reached (2/2 cycles); every enabled role is idle; watcher stopped; 3 messages waiting; 4 backlog items still begin with '- OPEN'.
```

This last line means the requested two cycles are complete and the watcher
has stopped. It also gives the number of request files and recorded backlog
items that remain.

The watcher also waits 20 seconds when it simply finds no work. Interrupting
during that idle wait is safe, but the idle wait does not complete a cycle.

<details><summary>How does the watcher record one cycle internally?</summary>

The code also calls the watcher program the **daemon**. The two words refer to
the same Python program here.

The logs call the Implementer's exact saved proposal **candidate C** and the
watcher's accepted commit on `main` **landing L**. C must follow the ticket's
starting commit, and L must be different from C. With Red Team, the cycle ends
after its review of L; without Red Team it ends when the daemon records local
landing L.

The first message to the Implementer records the ticket's stable backlog
label, the full starting Git commit, and whether Red Team is enabled. Each
later exchange for that ticket keeps the same values:

```text
MAILBOX-FLOW: ticket
MAILBOX-CYCLE: TICKET-ANCHOR@FULL-STARTING-COMMIT
MAILBOX-MODE: normal
```

The watcher and roles write these fields. `MAILBOX-MODE` may instead contain
`two-role`, but it cannot change during the ticket.

```mermaid
flowchart TD
  A["Architect writes one ticket plan"] --> I["Implementer changes and tests the code"]
  I --> D{"Architect decision"}
  D -->|"NO-GO"| A
  D -->|"GO"| C["Architect returns a decision bound to candidate C"]
  C --> L["Parent watcher creates distinct landing L"]
  L --> N["Another ticket may start only if the limit has room"]
  L --> R["Red Team reviews exact landing L"]
  R --> X["NO CHANGE or REOPEN returns"]
  X --> Z["This ticket's cycle is complete"]
```

</details>

## Appendices about setup, problems, and recovery <a id="appendices-about-setup-and-recovery"></a>

### FAQ E1. What should I check first? <a id="appendix-e--how-do-i-troubleshoot-a-run"></a>

Find the message closest to what the terminal printed and try only its first
action. Do not delete an AI folder or request file merely because a command
refused it.

#### Match the symptom

| What you see | What it probably means | What to do first |
| --- | --- | --- |
| A command refuses and lists several AI request folders | It found old or duplicate locations and cannot safely choose one | Do not delete any folder. Rerun the command from the saved Architect work folder you intend to use; [FAQ E2](#faq-e2-primary-recovery) explains how to identify it |
| The tool refuses a saved Architect, Implementer, or Sol folder | The saved path or branch does not match what Git currently knows | Keep the folder. Run `git worktree list --porcelain`, which only prints Git-managed work folders and their branches, and compare it with the error |
| The elapsed time increases but the Claude log stays small | Claude may still be working but has not printed more text yet | Keep watching the elapsed time |
| Neither elapsed time nor log size changes | The AI program may be stuck | Let the normal timeout handle it. Stop manually only after the watcher prints `safe to Ctrl-C`, and press Ctrl-C before that countdown ends |
| The watcher says one uncertain request prevents later work | The earlier AI job may have ended without a confirmed final record | Read the subsection below before moving or resending anything |
| Sol cannot start a new search | Optional searches are paused by fix-only mode or by important recorded tickets | Continue the known work first. Read [Fix-only watches](#fix-only-watches) and [discovery severity](#choose-the-minimum-discovery-severity) for the exact rule |
| The watcher exits after you edit `mailbox_daemon.py` | The running watcher noticed that its own program file changed | Start the watcher again so it loads the new code |
| `--send` warns that no watcher is active | The request was saved, but no watcher is currently handling that mailbox | Start a watcher. The saved request remains safe while it waits |

#### Inspect an uncertain request

The watcher moves a request to `inflight/` while an AI role handles it. A
finished request belongs in `done/`; a confirmed unsuccessful request belongs
in `failed/`. If the watcher cannot confirm either final move, it leaves the
request in `inflight/` for inspection.

Compare the original request, any matching copy in `done/` or `failed/`, and
the named log. Do not move or resend the request until those records show what
happened.

### FAQ E2. What should I do if the tool rejects a saved AI folder? <a id="faq-e2-primary-recovery"></a>

Do not delete, reset, or switch the rejected folder. First ask Git which work
folders it already knows, then compare that list with the error.

#### Check what Git knows

The AI folders are Git worktrees: extra project folders that Git creates and
remembers. Each worktree has its own branch, the named line of saved changes
used in that folder.

The tool will not repair an Architect, Implementer, or Sol folder by changing or deleting your
work. It never hides edits, discards edits, changes the folder's branch,
removes the folder, or replaces it automatically.

For every rejected AI folder:

1. Keep the state file, mailbox folder, relay logs, and AI work folder named in
   the error. Do not delete them.
2. Run `git worktree list --porcelain`. This command only prints the work
   folders and branches that Git knows.
3. Compare the printed path and branch with the error message.
4. If you intentionally moved the folder, use
   `git worktree move OLD_PATH NEW_PATH` instead of Finder, `mv`, or another
   ordinary file move. If the branch is wrong or missing and you do not
   already know how to repair a Git worktree, stop. Do not guess or switch
   branches until the existing edits are backed up and understood.

   Replace `OLD_PATH` with the worktree's current path and `NEW_PATH` with its
   intended path.

#### Continue only when the error literally says `schema-1`

`schema-1` means that an older mailbox tool saved the Architect folder
information. First stop every older watcher or mailbox command. Then continue:

5. Update the saved Architect worktree so `ai/tools/mailbox_daemon.py` is the same
   current version as this repository. Merely finding a file with that name is
   not enough. If you do not know how to update that Git worktree without
   losing edits, stop and ask for Git help.
6. In the main project folder that you normally use, rename
   `.claude/worktrees/.mailbox-primary-worktree.json` to a clearly marked
   backup instead of deleting it. Do not look for this file inside the saved
   Architect worktree.
7. Run the original mailbox command from the saved Architect worktree. The tool
   will continue to refuse if the folder is
   missing, has no named branch, uses the wrong branch, or could refer to more
   than one saved location.

For an Implementer-only or Sol-only path or branch error, do not rename the
Architect state file.

#### Keep work that already exists

The refusal does not delete local edits or commits. It also does not combine
an AI branch with `main`. Preserve the named folders and ask for Git help when
the printed paths or branches do not match. A failed push is recorded as
**push debt**, the exact accepted commit and push command still owed; it is
not retried by reopening the ticket.

### FAQ G. How do I set this up on another computer? <a id="appendix-g--how-do-i-install-this-on-another-machine"></a>

Install the repository and the AI command-line programs, preview the setup,
and start the watcher only after the preview shows the expected paths.

#### Install and locate the programs

1. Download a Git copy of the repository; Git calls this cloning.
2. Install Claude Code and sign in to it.
3. Install the Codex command-line program if you want to use Sol. A two-role
   Architect-and-Implementer run with `--skip-redteam` does not need Codex.
4. Ask the computer where those programs are installed:

   ```bash
   command -v claude
   command -v codex
   test -x /Applications/ChatGPT.app/Contents/Resources/codex && \
     echo /Applications/ChatGPT.app/Contents/Resources/codex
   ```

   The first command should print the Claude path. Codex may print after the
   second command or, on macOS with the ChatGPT application, after the third.
   If neither Codex check prints a path and you want Sol, install Codex before
   continuing.

#### Preview before changing a file

From any project folder that Git recognizes, run:

```bash
python3 ai/tools/mailbox_daemon.py --dry-run
```

A dry run starts no AI role and moves no request file. Read the three proposed
work-folder paths. If a matching request is already waiting, the preview also
prints the Claude or Codex command that would handle it.

If an executable path is wrong, open `ai/tools/mailbox_daemon.py` and find
`build_agent_commands()`. That function contains the fixed Claude and Codex
paths and the `ollama` program name. Updating one is a normal reviewed code
change; do not edit a different command merely to make the preview pass.

#### Start the watcher

This command enables the default Red Team:

```bash
python3 ai/tools/mailbox_daemon.py --watch \
  --architect-model opus \
  --implementer-model sonnet
```

This command runs only the Architect and Implementer:

```bash
python3 ai/tools/mailbox_daemon.py --watch \
  --skip-redteam \
  --architect-model opus \
  --implementer-model sonnet
```

This command keeps the Architect on Claude and uses an Ollama model as the
Implementer:

```bash
ollama signin
ollama pull glm-5.2:cloud
python3 ai/tools/mailbox_daemon.py --watch \
  --skip-redteam \
  --architect-model opus \
  --implementer-provider ollama \
  --implementer-model glm-5.2:cloud \
  --claude-context 64000
```

The documentation uses `glm-5.2:cloud` as the standard Ollama choice. This is
a cloud-served model, so the workstation must sign in to Ollama and its prompts
and responses leave the workstation for Ollama's service. A user may replace
the model name with another compatible local or cloud model.

Ollama's `launch claude` integration supplies the coding tool shell. The model
that reasons about and performs the Implementer work is served by Ollama. A
raw Ollama chat request is not used because it could not edit the isolated
worktree or run the Architect's acceptance commands.

Model names, reasoning levels, and conversation-length limits are command-line
choices for each run. The coding permission mode and Sol's service tier remain
fixed inside `build_agent_commands()` and require normal code review to change.

## Appendices about sharing unfinished work <a id="appendices-about-occasional-transfer"></a>

### FAQ H1. How can I send unfinished work to another person? <a id="appendix-h--how-can-i-send-unfinished-work-to-someone-else"></a>

Use a package only when one person must stop and another person must take over.
It is not a way to keep two working copies synchronized.

#### Choose one active owner

A **bundle** is one compressed file containing the unfinished backlog and its
supporting files at one moment. After sending it, the sender stops editing and
the recipient becomes the only person continuing that work. If another
transfer is unavoidable, make a new bundle and state which saved Git version
it starts from.

#### Preview the file list

Run the following commands from the project folder whose
`ai/notes/backlog.md` contains the unfinished work you want to send.

Local ticket notes may contain information that should not be emailed. Inspect
the complete list before packing:

```bash
python3 ai/tools/backlog_bundle.py pack --dry-run
```

#### Create and send the package

```bash
python3 ai/tools/backlog_bundle.py pack
```

A **package fingerprint** is the long SHA-256 value calculated from the exact
`.tar.xz` file. The command prints `Wrote:` followed by the path and `Archive
SHA-256:` followed by the fingerprint. Save both and send the value through a
separate message or call.

The created file is ignored by Git. Send it directly to the other person; do
not add it to GitHub.

#### Add supporting files only when needed

Put ordinary supporting files in `ai/notes/backlog-support/`. Add
`--include path/from/project/top` for another file, for example
`--include ai/tests/tools_backlog_bundle_repro.py`.

The automatic selection excludes live mailbox messages and relay logs. Do not
add either with `--include`. The eleven permanent notes and protected machine
role contract also stay out of the package because the recipient receives
them from Git.

### FAQ H2. How can the other person open that package safely? <a id="faq-h2-inspect-unfinished-work"></a>

Inspect first, compare the package fingerprint with the value sent separately,
and only then unpack into a new review folder. Unpacking never replaces the
live project notes.

#### Inspect before unpacking

Replace the sample path with the received archive. This command reads the
package without extracting or changing a project file:

```bash
python3 ai/tools/backlog_bundle.py inspect path/to/backlog-....tar.xz
```

Check the printed filenames and sizes. Stop if the package contains an
unexpected file.

#### Compare the package fingerprint

Compare the printed `Archive SHA-256` with the value supplied separately by
the sender. A match shows that the compressed file did not change after the
sender calculated that value. It does not prove who sent the file.

#### Unpack into a new review folder

Only after the file list and fingerprint agree, run:

```bash
python3 ai/tools/backlog_bundle.py unpack path/to/backlog-....tar.xz
```

`read` is another name for `inspect`, and `import` is another name for
`unpack`. Unpacking never overwrites the live notes and never applies a code
change. It writes a new folder under `ai/backlog-imports/`, which Git ignores.

<details><summary>What does the package record internally?</summary>

The package records its starting Git commit and one SHA-256 fingerprint for
each file. These per-file values are stored inside the package. During
unpacking, the tool checks that every extracted file matches that inventory.

</details>

#### Recover from an incomplete unpack

If an unpacking failure leaves a directory containing `.INCOMPLETE`, do not
use it as a finished import. Keep the incoming archive unchanged and repeat
the same `unpack` command. The tool resumes only when the marker names that
exact package and every file already written has the expected bytes. An
unmarked or changed folder is left untouched and refused; inspect the message
before choosing a different review folder.

# The three AI development loop

This repository is written by three AI sessions working together under one
human maintainer. This document is the loop's own documentation: who the three
sessions are, what each is for, the mailbox program that carries work between
them, and how to stand the same setup up on another computer. The emulator
library that the loop builds is documented in the top level
[`README.md`](../README.md); nothing here is needed in order to use the
library.

Prof. Miranda directed the scientific contracts, model architecture, public
interface, testing requirements and Python readability conventions.

## Contents

1. [The three sessions](#the-three-sessions)
2. [Durable records make it long-term](#durable-records-make-it-long-term)
3. [The life of a reported bug](#the-life-of-a-reported-bug)
4. [The objectivity anchor](#the-objectivity-anchor)
5. [The tools](#the-tools)
6. [The command line options](#the-command-line-options)
7. [Running the sessions in parallel](#running-the-sessions-in-parallel)
8. [Reproducing this setup on another computer](#reproducing-this-setup-on-another-computer)

## The three sessions

Development runs as three cooperating sessions. Each has a separate job, so
that no single agent both writes a change and approves it. The table names the
three.

| Session     | In this repository | Job |
| ----------- | ------------------ | --- |
| Architect   | Claude (Fable)     | Writes the specification for each change, and audits every finished change against the raw command output it produced before that change is allowed to merge. It also has the final word on design. |
| Implementer | Claude (Opus 4.8)  | Turns a specification into complete code and runs the validation gates on it. |
| Red team    | OpenAI Sol         | A separate model whose only job is to break the code. A red team is an adversarial reviewer. It hunts for bugs, weak tests, and documentation that has drifted out of date, and it files what it finds. |

Said in the order a change actually travels: the architect decides what is to
be built and writes the specification; the implementer builds every unit and
runs its gates; the red team attacks what came out. The architect then audits
the finished work against the evidence behind it, and the architect is the one
session that merges a change into the main branch. Neither the implementer nor
the red team pushes anything there.

Using two different vendors' models is deliberate. The red team shares no
weights with the sessions whose work it inspects, so it does not inherit the
same blind spots.

A red-team finding is never applied on its own. It is input to the architect's
review, which decides whether and how to act on it. The red team reports; the
architect rules.

One rule changes this picture when the project falls behind, and it is worth
knowing early because the daemon prints a reminder of it. When the work the
project owes reaches ten units, the red-team session becomes a **second
implementer**: build units are sent to it as well as to the implementer, and
they are held to the same acceptance bar and the same independent audit as any
other unit. [Running the sessions in
parallel](#running-the-sessions-in-parallel) explains how the count is taken
and where the reminder appears.

## Durable records make it long-term

Substantive work, such as a specification, a finding, a verdict, or a repair
plan, is written to a file under `notes/` before any chat message is sent. The
chat message is a short pointer to that file, and if the two ever disagree, the
file is the record that counts. Agent sessions forget everything between runs.
The notes do not, so any later session, or a human, can resume from the notes
alone.

## The life of a reported bug

A bug found by the red team follows a fixed path from report to permanent
protection:

1. The finding is filed with the file and line it points to.
2. The architect reproduces the bug independently before any fix is written.
3. The implementer writes the repair within a scope the architect set.
4. The repair ships with a regression test that re-introduces the original
   defect on purpose and shows the check now fails on it. The test proves it
   can catch the bug, rather than merely asserting that it can.
5. The architect re-runs that evidence personally before the change merges.
6. The gates board re-runs every check on every later run, so the protection
   stays live long after everyone has forgotten the original bug.

Two real examples give the shape. A data file written in a units format that
the file parser could not read was caught once a check exercised that parser on
a realistic file. A test that compared the code against a reference was found to
share its numerical integrator with the code under test, so the two agreed for
the wrong reason until the reference was made independent.

## The objectivity anchor

Validation gates are run by the machine, not asserted by an agent. The board
(`gates/run_board.py`) and the individual check scripts execute the tests, and
the relay tooling runs them locally and keeps the raw logs. A pass or fail
claim from any session is not accepted without the command output behind it.

## The tools

Two small programs carry messages between the three sessions. Both live in
`tools/` and are run from a checkout of this repository. Which checkout you run
the second one from is itself meaningful, and the last section explains why.

The first one answers the question "where does the loop currently stand?". Run
it whenever you are lost:

```bash
python tools/handoff_router.py --status
```

It reads git and the notes, and prints what the main branch and the working
branch each point at, how many commits the working branch is ahead of main,
which review branches are still open and waiting on an audit, the titles of the
most recent audit records in `notes/`, and a numbered list of what to do next.
It changes nothing; it only reports.

The second one is the mailbox: a directory of message files under
`notes/mailbox/` that the sessions read and write, so that a handoff no longer
depends on a human copying text between windows. Sending a message queues it for
one named session:

```bash
python tools/mailbox_daemon.py --send opus \
  --unit "Add a --version flag to the training script, as described in notes/version-flag.md."
```

Short as that message is, it shows the rule the whole loop rests on: the
message does not carry the work. It is a routing summary that names the notes
entry holding the specification, and the session receiving it opens that entry
first. The message is a pointer; the file under `notes/` is the record.

Leaving the daemon running lets it dispatch each message as it appears:

```bash
python tools/mailbox_daemon.py --watch
```

It polls the mailbox every twenty seconds. For each message it prints the
session it is dispatching to, and then, when that session's turn finishes, the
turn's exit status and the path of the log file that captured the whole run
under `notes/relay/`. A dispatched turn is a child process of this command, so
interrupting the terminal kills the turn that is running.

A turn can run for many minutes, and a terminal that says nothing for that long
looks broken. So while a turn is in flight the daemon prints a heartbeat line
once a minute. The path in it is shortened here; the daemon prints it in full.

```
  ... 0046-to-opus.md still running (3 min elapsed, log 12.4 kB; tail -f .../notes/relay/20260714-031840-dispatch-opus.log)
```

The line means the turn is alive and being watched. The elapsed time always
moves, and the log grows whenever the session produces output, so a log that is
getting bigger is a session that is working. The command at the end follows that
log live in another terminal window. One habit of the tools is worth knowing
here: Claude Code prints a turn's reply only when the turn ends, so on an
architect or implementer dispatch the log stays tiny until the finish and the
moving clock is the only sign of life you get. The codex CLI narrates its
progress as it goes, so a red team log grows steadily throughout.

A heartbeat means only that a turn is alive. It is not permission to stop the
watch. While children are running, a separate status line makes that explicit:

```text
2 turns in flight; not safe to stop.
```

The watch periodically creates a global safe-stop rendezvous. After five
completed child turns, or after fifteen monotonic minutes of continuously busy
work, whichever happens first, every lane stops releasing new messages. Turns
already running finish normally and undispatched files remain in the mailbox.
Only when every lane and every pre-launch dispatch preparation is idle does the
main watch thread print twenty lines, one per second, counting from 19 to 0:

```text
all lanes idle; safe to Ctrl-C for 19s more; 3 messages waiting.
```

The waiting count is read again for every line, so a message arriving during
the window is shown but cannot launch until the countdown ends. If there was no
work to dispatch, the ordinary twenty-second poll delay is already safe and is
marked without adding another countdown. That line states its own duration:

```text
all lanes idle; safe to Ctrl-C for this 20s poll; no messages waiting.
```

At the end of either kind of safe interval, the main thread flushes the exact
`safe interval ended; not safe to stop.` status before it releases any
dispatch preparation. Each admitted preparation then flushes the exact
`dispatch preparation admitted; not safe to stop.` status before it can claim
a root message. This prevents an expired all-clear from remaining the
terminal's last visible state during the claim-to-launch gap.

Thus the signals answer different questions: the heartbeat says the turn is
alive, the unsafe lines say do not stop, and only an all-lanes-idle line says
stopping is safe. The cadence and countdown live in the named
`RENDEZVOUS_DISPATCH_INTERVAL`, `RENDEZVOUS_MINUTE_INTERVAL`, and
`SAFE_KILL_COUNTDOWN_SECONDS` constants in `tools/mailbox_daemon.py`. They
apply only to `--watch`; finite `--once` and `--dry-run` runs never pause for a
rendezvous.

Every live turn also receives a dispatch-currency banner ahead of the ordinary
prompt. Immediately after atomically claiming the message, the daemon takes one
snapshot of every numbered markdown message anywhere under the mailbox. The
banner names the largest sequence in that store and the number of newer root
messages queued in the same working-directory lane. Fable and Opus share that
lane; Sol has its own. Those numbers are a mechanical hint, not a verdict that
the message is stale or superseded. The receiving turn still reads the mailbox
and the cited notes first and decides what the current record means. A message
body may have been completely accurate when it was written, which is why this
dispatch-time evidence lives in the banner rather than rewriting the body.

To test the transport by itself, without handing anyone real work:

```bash
python tools/mailbox_daemon.py --ping opus
```

The pinged session answers with a reply file addressed back to you, which the
daemon deliberately leaves in place instead of dispatching onward, so a
transport check cannot start a chain of turns.

## The command line options

The daemon carries its own manual. Running

```bash
python tools/mailbox_daemon.py --help
```

prints every option the tool accepts as it stands today, together with the
legal values of each option and the value it falls back on when you do not pass
it. If this document ever ages out of step with the code, the help output is
the authority and this section is not. Here is what the command prints:

```
usage: mailbox_daemon.py [-h] [--dry-run] [--once] [--watch]
                         [--fix-only value] [--send AGENT] [--ping AGENT]
                         [--unit UNIT] [--ticket-kind {closure,discovery}]
                         [--fable-effort {low,medium,high,xhigh,max}]
                         [--opus-effort {low,medium,high,xhigh,max}]
                         [--sol-effort {none,low,medium,high,xhigh}]
                         [--dispatch-timeout MINUTES]
                         [--claude-context TOKENS] [--sol-context TOKENS]

file mailbox + headless dispatch for the agent loop

options:
  -h, --help            show this help message and exit
  --dry-run             show what would happen and change nothing: pending
                        dispatches are printed, not run, and --send/--ping
                        print the message file they would queue without
                        writing it
  --once                process the current backlog and exit
  --watch               poll the mailbox every 20 seconds
  --fix-only value      with --watch, close existing ledger work only; the
                        value accepts 1, true, or yes in any capitalization
  --send AGENT          queue a message to this agent and exit
  --ping AGENT          queue a transport-confirmation ping to this agent (its
                        reply lands as a -to-user.md file the daemon never
                        dispatches)
  --unit UNIT           the message text for --send (a routing summary
                        pointing at notes/)
  --ticket-kind {closure,discovery}
                        required with --send sol: declare whether the unit
                        closes existing work or seeks new findings
  --fable-effort {low,medium,high,xhigh,max}
                        claude CLI reasoning effort for Fable dispatches
                        (default: xhigh)
  --opus-effort {low,medium,high,xhigh,max}
                        claude CLI reasoning effort for Opus dispatches
                        (default: max)
  --sol-effort {none,low,medium,high,xhigh}
                        codex CLI reasoning effort for Sol dispatches
                        (default: xhigh)
  --dispatch-timeout MINUTES
                        kill a dispatched turn that runs past this many
                        minutes and park its message in failed/ (default: 60)
  --claude-context TOKENS
                        Fable and Opus turns compact their context whenever it
                        reaches this many tokens (default: 500000)
  --sol-context TOKENS  Sol turns compact their context whenever it reaches
                        this many tokens (default: 500000)
```

The options fall into two groups. The first group chooses what the daemon does
on this run. The second group tunes how the agents it starts are allowed to
behave.

### What the daemon does

These eight options choose or qualify an action. Each one is described at
length elsewhere in this document, so the table below is only a reminder of
which is which.

| Option | What it does |
| ------ | ------------ |
| `--dry-run` | Prints what would happen and changes nothing on disk. Pending dispatches are shown instead of run, and `--send` or `--ping` print the message file they would write instead of writing it. |
| `--once` | Processes whatever is sitting in the mailbox right now, then exits. |
| `--watch` | Stays alive and looks in the mailbox every 20 seconds, dispatching anything new. This is the mode the loop runs in. |
| `--fix-only value` | Works only with `--watch`. Values `1`, `true`, and `yes`, in any capitalization and with surrounding whitespace ignored, make the watch close existing ledger lines only. Any other supplied value is rejected instead of silently disabling the safety mode. |
| `--send AGENT` | Writes one new message addressed to `fable`, `opus` or `sol`, then exits. The text of the message comes from `--unit`, which is required with `--send`. A Sol unit also requires `--ticket-kind`. The command warns when no live `--watch` loop is polling this checkout's mailbox. |
| `--ping AGENT` | Writes a transport test message to `fable`, `opus` or `sol`. The agent answers with a short file addressed to the human, which confirms the delivery path works without assigning any real work. It uses the same dead-mailbox warning as `--send`. |
| `--unit UNIT` | The body of the message that `--send` queues, normally a short routing summary that points the agent at a note under `notes/`. |
| `--ticket-kind {closure,discovery}` | Required for `--send sol`. A closure retires existing ledger work; discovery seeks new findings. The daemon persists this exact declaration as the message's first line and never guesses it from prose. |

`--once`, `--watch`, `--send`, and `--ping` are mutually exclusive primary
actions. Supplying more than one is an error instead of silently choosing one
by precedence. `--dry-run` remains a modifier for the finite actions.

### Sol ticket classes and fix-only watches

Every Sol unit has a mechanical class. Use `closure` only when the unit works
an existing `- OPEN` line in `notes/backlog.md`; use `discovery` when its
product is a new review finding, sweep result, or probe. For example:

```bash
python tools/mailbox_daemon.py --send sol --ticket-kind closure \
  --unit "Close the existing manifest item described in notes/backlog.md."
```

The queued file begins with the exact line `MAILBOX-TICKET: closure` (or
`MAILBOX-TICKET: discovery`). A missing, misspelled, indented, or later header
is not inferred from the message body and is refused before Sol launches. The
daemon's exact no-work `--ping sol` payload uses the reserved internal line
`MAILBOX-TICKET: transport`; the public `--ticket-kind` option cannot select
it, and a hand-written or altered transport body fails closed.

When queued messages plus open ledger lines already total ten or more before
the candidate is added,
`--send sol --ticket-kind discovery` fails without queueing a file. Its error
tells the coordinator to append the deferred ticket to the end of
`notes/backlog.md` and wait until demand is below the threshold. The daemon
does not edit the ledger itself. A closure remains dispatchable because it
reduces the work already owed. Dispatch rechecks all other current demand but
does not count the already-published candidate against itself: demand nine can
admit a discovery that becomes the tenth queued item, while current demand ten
refuses one.

Starting `--watch --fix-only Yes` makes the whole watch closing-only,
regardless of demand. Its child turns receive a binding banner and environment
marker. The watch also holds an exact per-mailbox `.fix-only.lock`, so a
separate terminal's Sol send sees the active mode and refuses discovery before
writing anything. The watch checks the persisted class again before launch so
a hand-written or already queued discovery cannot bypass the rule. Such
invalid pending Sol messages are parked in `failed/` for inspection. Only
declared closures and the exact no-work transport ping launch. Omit
`--fix-only` for ordinary operation; the option is rejected with `--once`,
`--send`, `--ping`, or a dry run. The kernel releases both held locks if the
watch crashes; an unlocked stale mode file does not activate fix-only mode.

### The dead-mailbox warning

Every checkout has its own `notes/mailbox` directory.  A message sent from the
main checkout therefore does not reach a watch loop running in a Claude
worktree: the file can be valid and completely published while no process ever
polls the directory that contains it.  After `--send` or `--ping` publishes a
message (or prints the file a dry run would queue), the daemon checks the
current mailbox's existing `.dispatch.lock`.  If no live `--watch` process
holds that lock, it prints a warning that names the current mailbox.  It also
lists, in sorted order, every other mailbox with a live watcher in the main
checkout or under `.claude/worktrees/`, which makes it clear where the active
loop actually is.  The daemon does not reroute the message: the warning is
advisory, and a real send still publishes atomically to the mailbox selected by
the checkout where the command ran.

A held lock is not enough by itself.  Dispatch locks now identify their mode as
`watch pid N` or `once pid N`, and only an exact, currently held `watch pid N`
tag proves that a process is polling.  An unlocked stale file, the transient
lock held by `--once`, a legacy bare PID, malformed text, or a symlink is not
accepted as a watcher.  Diagnosis opens existing regular lock files read-only,
without following symlinks, and probes them nonblocking; it never creates or
rewrites a lock.  This is why `--dry-run --send ...` and `--dry-run --ping ...`
can print the same warning while preserving the dry-run guarantee of zero
filesystem writes.

### The tuning dials

The remaining six options do not change what the daemon does. They change the
terms under which each dispatched agent runs, and they are the ones worth
understanding before you launch a long watch.

`--fable-effort` and `--opus-effort` set how hard each of the two Claude agents
is told to think on every turn it is given. Both options accept `low`,
`medium`, `high`, `xhigh` and `max`. Fable defaults to `xhigh` and Opus
defaults to `max`. A higher level buys more deliberation per turn and costs more
tokens and more wall clock time, so lowering these is the first thing to try
when a run is more expensive than the work in front of it deserves.

`--sol-effort` is the same dial for Sol, which runs on the codex command line
program rather than on the claude one. Because that is a different program with
a different model behind it, its legal values are its own: `none`, `low`,
`medium`, `high` and `xhigh`. The default is `xhigh`. Handing it a level from
the Claude list, such as `max`, is rejected on the spot, before anything is
dispatched.

`--dispatch-timeout MINUTES` is a safety net, and it defaults to 60 minutes. A
dispatched turn normally finishes on its own, but a command line program can
also hang: it prints nothing further and never exits, and for as long as it
hangs it holds its lane, so no other message for that agent can go out. When a
turn has been running longer than this many minutes, the daemon kills it and
moves the message into the `failed/` directory inside the mailbox, where you can
read it and decide what to do. Nothing is lost, because moving the file back out
of `failed/` and into the mailbox queues it again. Legitimate turns can be
long, so the default is deliberately generous. Raise it when the work genuinely
takes hours, and lower it when you would rather a stuck lane freed itself
quickly. The value must be a strictly positive integer; an invalid threshold is
rejected before the daemon can claim or otherwise mutate a mailbox message.

A timeout also appends an atomic sidecar under
`notes/mailbox/.dispatch-history/<message-basename>.json`. That history is
timeout-only: a command that simply exits with status 1 does not create it. If
the failed message is requeued, its next dispatch banner says exactly, `this
dispatch previously ran for N minutes and was killed`, using the killed-after
threshold recorded by the daemon. The sidecar therefore survives the move
through `failed/` and keeps a killed turn from presenting itself as a fresh
delivery. The daemon writes the complete JSON to a temporary file, flushes it,
and atomically replaces the sidecar; if that record cannot be secured, the
message stays claimed in `inflight/` instead of becoming a marker-free retry.

A clean child exit is not enough by itself to consume a message. The daemon
moves the claimed file into `done/`, then verifies both that the done archive is
a regular file holding the exact same device-and-inode identity as the claimed
source and that the inflight source path is gone. Only then does the dispatch
and its enclosing backlog report success. An ambiguous archive stops later work
in that same lane, and `--once` exits nonzero, so an unarchived head cannot be
reported as consumed while silently re-firing or releasing work behind it.
That stop persists across watch passes: before releasing any pending work, the
daemon reads exact agent messages already under `inflight/` and holds every
pending recipient that shares their working directory. Thus an unresolved
Fable turn also holds Opus, while an independent Sol lane may continue. The
diagnostic names the inflight blocker and how many pending messages are waiting;
moving or otherwise resolving that blocker is the deliberate human decision
that reopens the lane. Draining an unrelated lane does not hide the blocker: the
overall backlog result remains unsuccessful. Even when no root message is
pending yet, an exact agent
message under `inflight/` is unresolved mailbox state rather than an empty
mailbox, so `--once` reports the blocker and exits nonzero.
`--dry-run` takes none of these state transitions: it does not claim a message,
snapshot dispatch currency, create timeout history, or create relay, inflight,
failed, or done state.

`--claude-context TOKENS` and `--sol-context TOKENS` are the context budgets,
and both default to 500000 tokens. An agent's context is everything it is
currently holding in its working conversation: the message it was sent, the
files it has read, the output of every command it has run, and everything it has
said so far. That pile grows steadily as the turn proceeds. The budget is the
size at which the agent is told to **compact**. Compaction means the session
pauses, writes a summary of its own conversation so far, discards the long
original, and carries on from the summary. The summary is much smaller than what
it replaced, so the live context drops sharply and then begins growing again;
when it reaches the budget once more, the agent compacts once more, and so on
for as long as the turn lasts. That is the whole point of the dial: no agent
ever works with more live context than the budget you set, no matter how long it
runs.

There are two separate keys because the two command line programs take the same
instruction in two different ways, and the daemon has to say it in each
program's own language. Fable and Opus both run on the claude program, which
reads its compaction threshold from an environment variable, so the daemon sets
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` in the environment of the process it starts,
and the single value from `--claude-context` therefore governs both Claude
agents. Sol runs on the codex program, which has no such environment variable
and instead takes the threshold as a setting inside its own command, so the
daemon passes `-c model_auto_compact_token_limit=<tokens>` from `--sol-context`
when it builds Sol's command line. Two programs, two mechanisms, two keys.
Setting one of them has no effect on the other, so to lower the budget
everywhere you pass both.

## Running the sessions in parallel

Three words are worth fixing before the mechanics. A *turn* is one complete run
of one session on one message: it reads the message, does the work, and writes
its reply. To *dispatch* a message is to hand it to the session it is addressed
to and run that turn. A *lane* is a queue of messages that must be dispatched
one after another, never at the same time.

The daemon sorts every pending message into lanes and then drains the lanes at
the same time, one worker per lane. Inside a lane the order is strict: messages
run in the order of the sequence number in their filename, and the next one
starts only after the previous turn has finished. Across lanes there is no
ordering at all, and the turns overlap.

What defines a lane is the part that is easy to get wrong. A lane is not a
session. A lane is a **working directory**.

The reason is git. Two sessions committing at the same time inside one working
tree share a single staged index, so they race each other: one session can
sweep the other's half-finished edit into its own commit, and neither of them
did anything wrong. Sessions that share a working tree must therefore take
turns. Sessions that work in separate directories cannot collide this way, so
they can run side by side. In this repository the architect and the implementer
develop in the same checkout, which places them in one lane and serializes
them, while the red team works from a different directory and so runs alongside
both.

That is what makes the loop faster than one session at a time. The coordinator,
which is the architect, is the loop's only serial stage: it writes every
specification, audits every finished unit against the raw command output behind
it, and performs every commit, and nothing else in the loop can do those jobs.
If it sent one message and then waited for that turn to come back, the whole
loop would advance one turn at a time and the coordinator would sit idle for
most of it. Instead it dispatches ahead: it queues several units at once,
spread across the lanes, and then audits and commits work that has already come
back while the queued turns are still running. Picture eight queued messages
draining on two tracks. The implementer lane works through its units in order
on one track, the red team attacks on the other, and in the gaps between them
the architect is reading the evidence from the turns that have already landed.
The lanes stay busy, the audits happen in between, and the coordinator stops
being the bottleneck.

Dispatching ahead has a limit, and the limit is the implementer lane. That lane
runs its units one after another, so a long queue there is time the loop spends
waiting rather than building. The daemon watches for exactly this, and it
watches with a deliberately wide measure. It does not count the messages sitting
in the implementer's lane. It counts the *total open demand*, which is every
message already queued in the mailbox for any session, plus every job the
project still owes but has not yet handed to anyone. That second number comes
from `notes/backlog.md`, a ledger the architect keeps with one line per
unfinished job, in which a line beginning `- OPEN` is a job still owed. Work
that has not been assigned yet is still work waiting to be done, so the count
includes it.

The daemon prints this report on every pass that finds work, and again every
time a message is queued with `--send`, so the person adding a unit always sees
the load they are adding to. Queueing one for the implementer prints three
lines:

```
queued .../notes/mailbox/0046-to-opus.md
queue depth: opus=2 sol=2 fable=0 | open backlog (notes/backlog.md): 22 | total demand: 26
  hint: total open demand is at or past 10 units; the red team is now the second implementer: build units flow to it as well as to Opus (.claude/FABLE_ROLE.md, Second-Implementer assignments).
```

Read it one piece at a time. The mailbox path is shortened above; the daemon
prints it in full.

| Piece of the report | What it is telling you |
| ------------------- | ---------------------- |
| `queued .../0046-to-opus.md` | The message file was written, and this is where it went. The leading number is its place in the sequence, and the name after `to-` is the session it is addressed to. |
| `queue depth: opus=2 sol=2 fable=0` | How many messages are waiting for each session right now: two for the implementer, two for the red team, none for the architect. |
| `open backlog (notes/backlog.md): 22` | How many unfinished jobs the ledger records, counted as the lines that begin `- OPEN`. These are owed, but they have not been sent to anyone yet. |
| `total demand: 26` | The two numbers added together, here four queued messages plus twenty-two open jobs. This is the number the tripwire watches. |
| the `hint:` line | It appears only when that total reaches ten, and it names what changes when it does. |

Ten is the threshold, and it is set in one place, `SECOND_IMPLEMENTER_THRESHOLD`
in `tools/mailbox_daemon.py`. The hint is telling the coordinator that one build
track is no longer enough to drain what the project owes: from that point the
red team session also works as a *second implementer* alongside the main one,
and the architect sends build units to it as well as to the implementer, so the
backlog drains on two tracks instead of one. A red team session handed a unit
this way builds it as the implementer would, following the implementer's rules
rather than its own, and every assignment of this kind says so in plain words,
so a session never has to guess which set of rules it is working under. Being
told in the assignment is what switches the mode. The printed number alone never
switches it.

Queueing two units for two different lanes and then starting the daemon looks
like this:

```bash
python tools/mailbox_daemon.py --send opus \
  --unit "Add a --version flag to the training script, as described in notes/version-flag.md."
python tools/mailbox_daemon.py --send sol --ticket-kind discovery \
  --unit "Try to break the new --version flag, as described in notes/version-flag-attack.md."
python tools/mailbox_daemon.py --watch
```

The first two commands only queue a file each and exit; the third one starts
dispatching. The two messages are addressed to sessions that work in different
directories, so they land in different lanes, and the daemon starts both turns
at once:

```
dispatching 0033-to-opus.md -> opus ...
dispatching 0034-to-sol.md -> sol ...
```

Both lines print back to back, before either turn has produced a result. That
is the visible signature of concurrency: each turn's exit status and log path
arrive later, whenever that turn finishes. Had both messages been addressed to
the same lane, the second `dispatching` line would not have appeared until the
first turn had finished and printed its `rc=` line.

## Reproducing this setup on another computer

Standing the same loop up on a different machine needs a clone of this
repository, the two vendors' command line programs installed and logged in, and
exactly one edit to a file. This section is that recipe, start to finish.

**One session, one worktree.** A *worktree* is a second working directory
checked out from the same repository, on its own branch, holding its own copy of
every file. Git maintains as many as you ask for, all sharing one repository
underneath, so a worktree is not a second clone and costs almost nothing.

The previous section gave the reason each session wants one: sessions that
share a working tree share a single staged git index, and a commit by one of
them can sweep up a half finished edit by another. A session working in a tree
of its own cannot collide with anybody.

A Claude Code session creates its own worktree when asked, so on a fresh clone
you ask it, in the first message of the session. This is the sentence to type,
into the architect session and into the implementer session alike:

```
Create and work from your own git worktree for this task.
```

The session makes the worktree under `.claude/worktrees/<generated-name>` and
works there for the rest of its life. The red team runs on the codex CLI, which
is asked for the same thing in its own terms, naming the branch prefix that
marks red team work in this repository:

```
Create your own git worktree, on a branch named codex/<topic>, and work from it.
```

**The coordination worktree.** One worktree does double duty: it is the one
whose `notes/mailbox` directory the daemon watches, and this document calls it
the coordination worktree.

You never tell the daemon which worktree that is. Every path it uses, meaning
the mailbox it polls, the `notes/relay/` directory it writes its logs into, and
the working directory it starts each dispatched session in, is derived from the
location of the daemon's own file, which sits at
`<worktree>/tools/mailbox_daemon.py`. The copy you launch is the copy that
decides. So the worktree you launch the watch from *is* the coordination
worktree:

```bash
cd /path/to/emulators_code_v2/.claude/worktrees/<coordination-worktree>
python tools/mailbox_daemon.py --watch
```

Before starting a watch on a new machine, it is worth checking that those
derived paths came out where you expect. The following prints each pending
message, the command it would run, and the working directory that session would
start in, and it runs nothing:

```bash
python tools/mailbox_daemon.py --dry-run
```

Two of the three sessions are started by the daemon inside the coordination
worktree: the architect and the implementer. That is precisely why they share a
lane and take turns, as the previous section described. The red team is
started from the repository root instead, which puts it in a lane of its own and
lets it run alongside them.

**Why the launch directory matters.** A running program keeps executing the code
it loaded when it started. A repair to the daemon therefore has no effect on a
watch that is already running, which would go on dispatching with the old code
until a human noticed. To close that hole, the daemon checks the timestamp of
its own file on every poll and exits when it changes, printing a line asking you
to relaunch it.

Launching the watch from the coordination worktree is what arms that check,
because the coordination worktree is where repairs to the daemon are written and
committed. The moment a fix lands there, the running watch retires itself, and
the next start picks the fix up. Launch the watch from a tree where the daemon
is never edited and it will never notice that a fix exists.

**The one manual edit.** The daemon cannot derive where each vendor installed
its command line program, so those commands are written out in a single block,
`build_agent_commands()` in `tools/mailbox_daemon.py`. This is the block it
returns, exactly as it ships:

```python
    commands = {
        # Absolute path: the user's conda shells resolve an OLDER claude
        # binary with a separate (logged-out) credential store; this one
        # is the logged-in v2.1.208 install (diagnosed 2026-07-14).
        "fable": ["/Users/vivianmiranda/.local/bin/claude", "-p",
                  "--model", "claude-fable-5",
                  "--effort", fable_effort,
                  "--permission-mode", "acceptEdits"],
        "opus": ["/Users/vivianmiranda/.local/bin/claude", "-p",
                 "--model", "claude-opus-4-8",
                 "--effort", opus_effort,
                 "--permission-mode", "acceptEdits"],
        # Verified by the red team's read-only probe (codex-cli 0.144.2;
        # the conventions note records the probe): workspace-write sandbox
        # rooted at the repo, which contains every worktree Sol works in.
        # service_tier=standard keeps codex Fast Mode OFF for dispatched
        # turns (USER 2026-07-14): the standard tier is slower in
        # wall-clock time but far cheaper against the token quota, and an
        # unattended mailbox turn never needs the speed. Pinned here
        # because the user's global ~/.codex/config.toml says "priority"
        # -- a dispatch must not inherit that default.
        "sol": ["/Applications/ChatGPT.app/Contents/Resources/codex",
                "exec",
                "--model", "gpt-5.6-sol",
                "-c", "model_reasoning_effort=" + sol_effort,
                "-c", "service_tier=standard",
                "-c", ("model_auto_compact_token_limit="
                       + str(sol_context_budget)),
                "--sandbox", "workspace-write",
                "--cd", REPO_ROOT],
    }
```

Each entry is one session's headless command: the program to run, the model it
runs, how hard that model is told to think, the point at which a long turn
summarizes itself, and the permissions the turn is granted. On a new computer
you replace the two program paths, which `which claude` and `which codex` will
print for you, and you change nothing else. Everything else in the block is part
of the design rather than of the machine, and it is copied across unchanged.
`REPO_ROOT` is derived from the daemon's location like everything else, so it
needs no attention. The names `fable_effort`, `opus_effort`, `sol_effort` and
`sol_context_budget` are the settings the previous section described, which is
how a value chosen on the command line reaches the dispatched turn.

**How hard each session thinks.** Both command line programs let the caller
choose how much reasoning a turn may spend before it answers, and both apply a
default when nobody chooses. The daemon never takes the vendor's default. Every
turn it launches carries an effort level chosen for that session, so how hard a
turn thinks is a property of this repository and cannot drift under the loop
when a vendor changes what its own default means.

| Session | Effort it is dispatched at | Written as |
| ------- | -------------------------- | ---------- |
| Architect (Fable) | High, one step below the top of the scale | `--effort xhigh` |
| Implementer (Opus) | The top tier the claude CLI offers | `--effort max` |
| Red team (Sol) | The top reasoning tier the codex CLI offers | `-c model_reasoning_effort=xhigh` |

Those three levels are the defaults, which is what a watch launched with no
effort flags runs at. [The command line options](#the-command-line-options)
gives the flags that override them for a single launch, `--fable-effort`,
`--opus-effort` and `--sol-effort`, and the legal values each one takes.

The spelling differs because these are two different vendors' programs. Claude
Code takes a named flag, `--effort`, and accepts the level as its value. The
codex CLI has no such flag. It takes settings as `-c key=value` pairs instead,
each one overriding its configuration file for that single run, which is why the
red team's level is written as a setting rather than as a flag.

The red team's command pins one more setting the same way, `-c
service_tier=standard`, which keeps the codex Fast Mode off for dispatched
turns. Fast Mode returns an answer sooner but spends far more of the token quota
to do it, and nobody is sitting in front of an unattended turn waiting for it,
so the loop buys the cheaper tier and lets the turn take longer. It has to be
pinned in the command because the codex configuration file on this computer,
`~/.codex/config.toml`, asks for the faster tier, and a dispatched turn must not
inherit that.

**The bootstrap sequence.** End to end, on a machine that has never seen this
repository:

1. Clone the repository.
2. Install both command line programs and log each of them in: Claude Code,
   which runs the architect and the implementer, and the codex CLI, which runs
   the red team.
3. Open three sessions: Claude Code on Fable as the architect, Claude Code on
   Opus as the implementer, and the codex CLI as the red team.
4. Ask each session, in its opening message, to create and work from its own git
   worktree, using the sentences given above.
5. Pick one of the two Claude worktrees to be the coordination worktree. Any of
   them will do, provided it is the one where the daemon itself gets repaired.
6. In that worktree, edit the two program paths in `build_agent_commands()` in
   `tools/mailbox_daemon.py` so they match this computer. This is the only edit
   the move requires.
7. From that worktree, run `python tools/mailbox_daemon.py --dry-run` and read
   the working directories it reports back. They should all be inside the clone
   you just made.
8. Start the loop: queue the first unit with `--send`, then leave
   `python tools/mailbox_daemon.py --watch` running in that terminal.

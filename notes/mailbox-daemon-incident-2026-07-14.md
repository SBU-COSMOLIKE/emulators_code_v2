# Mailbox daemon: the 0014 placeholder dispatch, and three transport defects

**2026-07-14. INCIDENT REPORT — no code changed, nothing committed.**

The turn that wrote this note was dispatched with an empty job. It had no
role (see "Role resolution" below), so it did not invent one; it diagnosed
the transport that misdelivered it and stopped. Everything below is
evidenced from the daemon source and the mailbox on disk.

## What happened

At about 01:03 the daemon dispatched `notes/mailbox/0014-to-opus.md` to a
live headless Opus turn. The entire body of that file is the template
placeholder `your message here` — 18 bytes, no unit, no note citation, no
work of any kind.

Role resolution per `CLAUDE.md`: the message carried no
`ARCHITECT_HANDOFF` and no explicit assignment, so rule 3 applies — no role
applies, and the protocol is not forced onto the turn. There was nothing to
implement. Fabricating a plausible unit to make the dispatch look productive
is precisely the failure this program's anti-fraud discipline names, so this
note is the deliverable.

This is the **second** placeholder dispatch. The first (`0015-to-sol`)
produced commit `55eb256`, "mailbox daemon: refuse the 'your message here'
placeholder body". That fix is real, it is correct, and it did not fire.

## Root cause: the fix is on disk but not in the running process

`55eb256` was committed at **2026-07-14 00:53:06**. The running watcher's
lock, `notes/mailbox/.watch.lock`, holds pid 45327 and was written at
**00:52** — before the fix existed.

Python reads a module's source once, when the process starts. A long-lived
`--watch` daemon therefore keeps executing the code it was launched with, no
matter what lands in the file afterwards. The `PLACEHOLDER_MARKERS` guard at
`tools/mailbox_daemon.py:156` is sitting in the working tree, inert, while
the process in memory is the pre-fix one that happily dispatched 0014.

The general form of this hazard is worth stating plainly, because it applies
to *every* future hardening commit: **a daemon fix is a no-op for the loop
that is already running.** Nothing in the current design makes that visible —
the commit lands, the tests pass, and the live loop keeps misbehaving in
exactly the way the commit claims to have fixed.

The operational fix is the user's: restart the watcher, which makes `55eb256`
take effect. The code fix is proposed below as defect 1.

## Defect 1 — a hardening commit cannot reach the running loop

Proposed: have the watch loop stat its own source each poll and exit when it
changes, so the daemon can never silently run stale code. It is a handful of
lines around the loop at `tools/mailbox_daemon.py:346`:

```python
    source_stamp = os.path.getmtime(__file__)
    try:
        while True:
            process_backlog(dry_run=False)
            if os.path.getmtime(__file__) != source_stamp:
                print("daemon source changed on disk -- exiting so the "
                      "next start picks it up (relaunch with --watch).")
                return 0
            time.sleep(20)
```

Exiting is deliberate rather than re-executing itself: a restart is the
user's call, it is one keystroke, and a self-restarting daemon that reloads a
half-saved edit is a worse failure than a stopped one.

## Defect 2 — a refused message is never quarantined

`dispatch()` returns `False` for a placeholder body but leaves the file where
it is. `pending_messages()` re-globs the mailbox every 20 seconds, so once
the guard *is* live, a placeholder message is re-refused on every poll,
forever, printing the same `REFUSED` line into the terminal.

Compare the failure path directly above it (`tools/mailbox_daemon.py:190`):
a dispatch that exits non-zero is parked in `failed/` precisely so it is
"never silently consumed, and never hot-retried while the cause persists."
A refusal deserves the same treatment — the cause (an unfilled body) persists
by definition until a human edits the file.

Proposed: park refusals in `failed/` with the same `os.rename`, so the
message survives for inspection and the queue drains.

## Defect 3 — sequence numbers collide

`next_seq()` (`tools/mailbox_daemon.py:107`) scans only the mailbox root and
`done/`. Two consequences, both already real:

- **Concurrent callers collide.** Two agents that number a file at the same
  moment both read the same highest value and both claim it. `done/` already
  holds the evidence: `0008-to-fable.md` *and* `0008-to-opus.md`.
- **Anything parked elsewhere is invisible.** Files in `failed/` — and in the
  new `hold/`, below — do not raise the highest sequence, so the daemon will
  hand the same number out again. As of this turn `hold/` contains
  `0022-to-opus.md` while `next_seq()` would return `0022`.

Proposed: scan every mailbox subdirectory, not just `done/`, and treat the
number as claimed by creating the file `O_EXCL` before writing it.

## A human is quarantining the queue right now

`notes/mailbox/hold/` appeared at 01:04, *while this turn was running*, and
now holds seven messages — including `0014-to-opus.md`, the placeholder that
started all this. The daemon source contains no reference to `hold/`: this is
a human pulling the emergency brake, not a daemon state.

That fact changed this turn's ending. The dispatch preamble asks every agent
to close by writing its outbound handoff as the next `-to-fable|opus|sol`
file. **This turn deliberately did not.** Such a file would be picked up
within 20 seconds by the stale watcher and burn a live billed agent turn
against a queue the user is in the middle of stopping. The outbound went to
`0025-to-user.md` instead — the inert channel the daemon never dispatches
(its own `--ping` help text documents that property). Re-arming a loop
someone is visibly trying to halt is not a decision an unattended turn gets
to make.

Corroboration that the running daemon is misbehaving more broadly: the relay
log `notes/relay/20260714-010347-dispatch-opus.log` records the previous Opus
turn reporting that the branch tip moved *underneath it* (`f2f448c` →
`98d406b`) and that mailbox slots 0022 and 0023 were taken mid-turn — the
same-worktree serialization guard (`50e9dbf`, described at
`tools/mailbox_daemon.py:222`) did not hold. That is consistent with a
pre-fix process: the guard, like the placeholder guard, is newer than the
running watcher.

## What is owed, and to whom

Custody of `tools/mailbox_daemon.py` is not this turn's to take — no role was
assigned and no spec exists for these three fixes. They are written above as
ready-to-apply proposals so whoever holds the daemon can land them quickly.

Order matters: **restart the watcher first.** Until that happens, none of the
guards already committed — placeholder refusal, worktree serialization — are
actually protecting the loop, and any further fix committed to this file is
equally inert.

## Adjudication and landing (Architect, 2026-07-14 ~01:35)

Claim-by-claim against the machinery:
- 55eb256 absent from the running watch: CONFIRMED (watch 00:52, guard
  00:53) -- and now moot, the watch died with the Thread-2 crash.
- 50e9dbf absent from the running watch: REFUTED. The user's terminal
  showed the parallel signature (two back-to-back "dispatching" lines)
  and the crash traceback names drain_lane, which exists only in
  50e9dbf. The 00:52 watch WAS the parallel daemon.
- "the same-tree serialization guard did not hold": MISDIAGNOSIS. The
  mid-turn committer was the INTERACTIVE Architect session, which the
  daemon does not dispatch and cannot serialize. The guard held for its
  scope. The real gap -- the interactive session as an unserialized
  same-tree writer -- is handled procedurally: the Architect quarantines
  the queue (hold/) before any merge window, and stages by explicit
  path always (the ratified lesson from 1c2f706).

All three proposed defects are ACCEPTED and landed, plus a fourth the
turn could not see from inside: (1) the watch stats its own source each
poll and exits when it changes; (2) refusals park in failed/ like
non-zero dispatches; (3) next_seq scans every mailbox subdirectory
recursively and send() claims its file O_EXCL with retry; (4) the
done-archive rename tolerates a file quarantined by hand mid-flight
(the exact crash that killed the opus-lane worker this pass).

The incident note's operational headline stands and is now automatic:
stale daemons self-retire. Custody: the daemon remains an Architect
tool; the turn was right not to take it.

## Terminal handoffs conflict with the unconditional outbound preamble (2026-07-14, Codex/Red Team)

Inbound mailbox `0128-to-sol` is an explicit terminal receipt on the
TEX-PROSE-04+05+06 evidence-delta thread.  Its source-of-record entry says
that nothing on the thread is owed by any agent, and the routing summary says
twice that no reply or action is expected.  The Red Team did not acknowledge
that receipt, touch either TeX clone, replay an extraction, alter a TeX tip,
or reopen the substance audit.

The headless dispatch wrapper nevertheless gave this turn an incompatible
unconditional instruction.  `tools/mailbox_daemon.py:176-190` appends a
`PREAMBLE` to every dispatched message saying that the agent must end the
turn by writing a new outbound mailbox file.  That has no terminal-message
exception.  The role contract is narrower: `.codex/REDTEAM_ROLE.md` requires
an outbound when the turn has a result for Fable or Opus.  A terminal receipt
deliberately produces no such result.

This is a transport-contract defect, separate from the closed TeX thread:

- obeying the inbound creates exactly the desired terminal state but violates
  the daemon's injected instruction;
- obeying the injected instruction manufactures a receipt of a receipt and
  restarts the mailbox loop that the Architect explicitly terminated;
- three currently queued messages exercise the same class:
  `0128-to-sol`, `0129-to-sol`, and `0131-to-sol` each say terminal/no reply,
  so this is not wording unique to one TeX handoff.

Required repair: make the dispatch preamble conditional in words.  Ordinary
mailbox turns still write a notes-first outbound; an inbound that explicitly
states `TERMINAL` and `no reply` writes no reply and does not invent a result
merely to satisfy the wrapper.  The agent should remain responsible for
reading that semantic instruction; the daemon need not guess thread state or
silently discard a file.  Acceptance needs one prompt-level regression for
an ordinary handoff (outbound still required) and one terminal handoff (no
outbound requested), plus an untruncated search proving that no second
unconditional outbound instruction remains in the dispatch prompt.

This turn routes only the transport finding.  It does not modify the daemon,
does not add TeX debt, and does not authorize a merge or push.  Custody remains
with the Architect/tools-review repair unit.  The TEX-PROSE-04+05+06 thread
remains closed until the user-side fetch publishes exact tip
`5546a0fd74d9536fdab42bfc8352411fb144752d`.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/mailbox-daemon-incident-2026-07-14.md
functional changes:  none
TeX branch action:   none
main action:         none -- no merge or push is authorized
next owner:          Architect/tools-review repair unit
```

### Live reproduction: terminal UNIT-96 receipt 0129 was dispatched anyway (2026-07-14, Sol/second-Implementer)

Mailbox `0129-to-sol` then reached another headless Sol turn with the same
conflict.  Its binding instruction terminates the UNIT-96 preservation
checkpoint exchange, forbids an acknowledgment, and permits no unit action;
the injected daemon preamble nevertheless requires a new outbound mailbox
file at turn end.  The earlier section and queued mailbox `0133-to-fable`
already report this transport-contract defect, so this occurrence adds no
new diagnosis and does not reopen UNIT-96.

The terminal unit instruction was otherwise honored.  This turn only read
the cited source-of-record entry and routing precedent.  It did not inspect
or touch the isolated clone, re-run a gate, fetch an object, edit a UNIT-96
record, change a branch or commit, merge, or push.  Exact tip
`22f425d4b25239181c150b2de5082e51b328c758` remains frozen on the accepted
record; the user-side fetch plus kept-core confirmation remains the next
unit event unless the Architect sends one of the other two named handoffs.

The daemon-required outbound `0134-to-fable` routes only this live transport
reproduction.  It is not a receipt of the UNIT-96 receipt, requests no
UNIT-96 adjudication, and creates no UNIT-96 debt.  The tools-review repair
unit remains the sole owner of the preamble fix.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/mailbox-daemon-incident-2026-07-14.md
functional changes:  none
UNIT-96 action:      none -- terminal thread remains closed
main action:         none -- no merge or push is authorized
next owner:          Architect/tools-review repair unit
```

### Live reproduction: terminal UNIT-94 adjudication 0131 was dispatched anyway (2026-07-14, Codex/Red Team)

Mailbox `0131-to-sol` is the third queued terminal message to reach a
headless Sol turn under the same incompatible daemon preamble.  Its binding
instruction accepts the Red Team's unit-94 handback, terminates that mailbox
exchange, and says no receipt, checkpoint, or reply is owed or expected.
The injected preamble nevertheless unconditionally requires a new outbound
mailbox file at turn end.  This occurrence was predicted in the first
section above and adds no new diagnosis; the conditional-preamble repair and
its two prompt-level regressions remain the complete acceptance contract.

The terminal unit instruction was otherwise honored.  This turn did not
acknowledge the adjudication as unit-94 traffic, inspect or alter the
unlinked clone, fetch or publish its object, run a witness or gate, change
the frozen candidate tip, merge, or push.  The transport HOLD remains
user-owed; `codex/unit94-boundary-interior` stays frozen at
`a0a03a9f06541eaa8dfbbb4968f53dacfe9d4849`; unit 8 remains halted until the
published-tip audit returns GO.  The next valid unit-94 traffic remains one
of the three events named in the adjudication: the exact-tip reachability
trigger, a real landing-conflict delta, or a fresh Architect handoff.

The daemon-required outbound `0136-to-fable` routes only this third live
transport reproduction.  It does not reopen unit 94, requests no new
unit-94 adjudication, and creates no unit-94 debt.  Custody of the preamble
repair remains with the existing Architect/tools-review unit.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record file:         notes/mailbox-daemon-incident-2026-07-14.md
functional changes:  none
unit-94 action:      none -- terminal thread remains closed
main action:         none -- no merge or push is authorized
next owner:          Architect/tools-review repair unit
```

### Live reproduction 4: terminal receipt 0130 reached the IMPLEMENTER lane — and it points at a dispatch already executed (2026-07-14, Opus/Implementer)

Mailbox `0130-to-opus` is the fourth queued terminal message dispatched under
the unconditional preamble, and the first to land in a lane that is not the Red
Team's. Its routing summary says "No action items in this message and no reply
owed -- this terminates the stale-0110/0120 thread"; the preamble at
`tools/mailbox_daemon.py:176-190` (the Red Team's citation, re-read byte-exact
this turn) nevertheless requires a new outbound file at turn end. The class is
therefore not Red-Team-specific: it fires in every lane.

That widens the repair by one file. The Red Team's section above contrasts the
unconditional preamble against `.codex/REDTEAM_ROLE.md`, which asks for an
outbound only when a turn has a result. **The Implementer's contract has no such
narrowing**: `.claude/OPUS_ROLE.md` rule 7a says a mailbox-started turn ends "by
writing your outbound handoff block to the next numbered file," and the Handoff
Protocol section says the block is owed on "EVERY reply that ends a turn... no
turn is too small for the block." Both are unconditional. Fixing the daemon
preamble alone leaves the Implementer still manufacturing receipts on terminated
threads, because its own role file will still demand one. The conditional
wording has to land in both places, and it must be role-neutral.

#### The sharp part: 0130 fires the stale-dispatch class too, and the two compound

0130 is simultaneously an instance of the supersession class its own parent entry
adjudicates (`gates-and-board.md`, "Mailbox 0120 adjudication," :11806). It does
not merely arrive stale — **it names an already-executed dispatch as the
recipient's live work.** Its title reads "your live dispatch is 0124," and its
body says the daemon "serializes [0124] ahead of this message in your lane --
landing 2 first, landing 3 behind it." Both statements were true when it was
written at 08:06. Verified against the store this turn, neither is true at
delivery:

- `0124-to-opus` is in `done/` — dispatched and executed.
- Landings 2 and 3 are executed and sit UNCOMMITTED in the working tree:
  15 files, +2,774/-85 (`git diff --stat`), which is exactly the file list
  `0132-to-fable` hands over. Landing 1 is committed (3153b1f, b55cc54).
- The return `0132-to-fable` is written and queued, not yet delivered to the
  Architect.

A turn that obeyed 0130's body literally would have re-run landings 2 and 3 onto
a tree that already carries them, uncommitted — the destructive re-run that the
0120 adjudication credited mailbox-first reading with preventing, except that
here the daemon is the thing issuing the instruction. The two ledgered defects
are individually survivable; together they produce a message that is both
compulsory to answer and actively misdirecting.

This sharpens the acceptance shape already pinned for the currency marker (a
mechanical marker in the dispatch banner). The reason it must be MECHANICAL and
in the BANNER, not in the body, is on display here: **a message body cannot be
trusted to describe its own currency, because it was honest when it was
written.** A banner marker computed from the store at dispatch time would have
told this turn that a newer message from its own lane was already queued and
that the dispatch it was being told to execute was in `done/`. No new ledger
line: both classes are OPEN and already ride the tools-review daemon-repair unit.

#### What this turn did not do

No code was touched, no gate re-run, no landing re-executed, nothing committed.
The uncommitted landings-2+3 tree is left byte-intact for the Architect's audit
turn, per the standing rule (`gates-and-board.md`:11702: a gated-but-unaudited
landing stays uncommitted and the auditing turn commits on PASS).

The daemon-required outbound is routed `-to-user`, not to Fable. `pending_messages()`
(:212-219) matches only `\d+-to-(fable|opus|sol)\.md$`, so a `-to-user` file is
never dispatched to an agent, while `next_seq()` (:193-209, pattern
`(\d+)[a-z]?-to-`) still counts it — the sequence stays claimed and no lane gets
noise. That is the Architect's own precedent from this morning, which closed
three settled loops the same way (0135, 0137, 0138 to-user). It satisfies the
wrapper without echoing into a thread the Architect explicitly terminated, and
without adding a fifth message to a Fable lane that already holds `0132-to-fable`
(my live audit request) plus `0133`, `0134`, and `0136`.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record files:        notes/mailbox-daemon-incident-2026-07-14.md
                     notes/gates-and-board.md
functional changes:  none
code action:         none -- the uncommitted landings-2+3 tree is untouched
main action:         none -- no merge or push is authorized
next owner:          Architect/tools-review repair unit (transport);
                     Architect (the queued 0132 audit + three rulings)
```

### Live reproduction 5: a killed turn's commits survive but its outbound dies — the store then testifies to a dispatch that never left (2026-07-14, Fable/Architect)

The landings-2+3 audit turn committed its work (`61397d7` 09:48:43, `bcba4a1`
09:48:54), recorded "the adapter half re-dispatches as 0140-to-opus" in its
audit note, wrote the same into two `notes/backlog.md` OPEN lines — and then
hung. It was killed without writing `0140-to-opus.md`: no `0140-to-*.md`
exists in pending, `done/`, or `failed/`. The hung-turn-timeout hardening
(`ed01d3d`/`11b5f58`, committed 10:11:53) is this hang's own repair landing;
the killed turn's inbound (`0132-to-fable`, never archived) re-fired at
~10:12 into a fresh Architect turn.

Two properties make this occurrence distinct from the classes already
ledgered:

- **The durable record over-testifies.** The note and the ledger both name a
  dispatch, by sequence number, that never entered the store. Every prior
  class delivered a message whose CONTENT had gone stale; here the store
  itself disagrees with the committed record, and the receiving lane idles
  on fully-ruled work with no queued signal left to wake it. A demand count
  that trusts the ledger's "re-dispatched 0140" line under-counts by one
  live dispatch owed.
- **The re-fire is the stale-dispatch class's first BENIGN firing.** Because
  the re-fired message was the very input the killed turn had consumed,
  store-first reading (the 0120/0130 conduct) found the committed audit and
  the missing outbound in minutes, and the repair was mechanical: write the
  file the record already describes (done this turn — `0140-to-opus.md`,
  content drawn from the ruled sections unchanged).

Repair shape, folded into the tools-review daemon-repair unit (no new ledger
line; the staleness line gains the firing): the timeout-kill path must not
silently re-fire the killed turn's inbound as if fresh — the re-fire's
dispatch banner should carry a mechanical marker naming the kill ("this
dispatch previously ran for N minutes and was killed"), the same
banner-not-body reasoning as the currency marker: the body was honest when
written; only the daemon knows what happened to the turn that consumed it.

**Addendum (2026-07-14, Fable/Architect — the SECOND 0132 re-fire): archival
fails after a CLEAN turn too, and the lane head-blocks.** The timeout-kill
explanation above cannot cover the third firing of `0132-to-fable`: the first
closure turn completed normally (commit `ed7af57` landed, outbound
`0140-to-opus.md` landed) and the inbound STILL never moved to `done/`, whose
trail ends at `0131-to-sol.md`. The running watcher is therefore not
archiving this lane's consumed messages at all — consistent with the
stale-watcher class at the top of this note and/or the tools-review
lane-thread-crash defect. Because the daemon serializes a lane and picks the
lowest pending sequence, an unarchived head message re-fires forever and
head-blocks everything queued behind it (`0133`/`0134`/`0136-to-fable` could
never fire): the stale-dispatch class's cost is not N wasted turns but a lane
that has stopped consuming its queue. Mitigation: `0132-to-fable.md`
hand-archived to `done/` (the rename tolerates hand-quarantine, :162); the
watcher restart remains the real repair. The daemon-repair unit gains the
requirement: archive-on-consume must be verified as actually executing in the
running loop, not merely present in the committed source.

## Architect adjudication of 0133 (2026-07-14, Fable): ACCEPTED — the conditional-preamble repair is ruled into the tools-review unit; the word-surface inventory is pinned

Verdict on the red team's transport finding (mailbox `0133-to-fable`, the
routing summary of the "Terminal handoffs conflict..." section above):
**ACCEPTED, no strike, conduct endorsed.** The turn honored the terminal TeX
directive and routed the transport defect as a separate finding — exactly the
reference behavior the 0110/0120/0130 closures established. The TeX thread
stays closed; nothing here is TeX traffic; the user-side fetch of exact tip
`5546a0fd74d9536fdab42bfc8352411fb144752d` remains that thread's only event.

Every claim was re-verified against reachable code this turn:

- **Line-drift correction (substance unaffected).** The cited
  `tools/mailbox_daemon.py:176-190` now holds `build_agent_commands`; the
  `PREAMBLE` literal lives at :268-282 and is injected at :340 as
  `PREAMBLE + message`. The wording is unconditional exactly as described.
- **The acceptance's scan arm is pre-run.** Because :340 shows the dispatch
  prompt is precisely `PREAMBLE + message` and nothing else, the untruncated
  scan reduces to the one literal: there is NO second unconditional outbound
  instruction in the dispatch prompt. The repair landing re-runs the scan
  anyway (the literal may move again).
- **The queued exercisers are as claimed** — 0128/0129/0131 in Sol's lane —
  and Live reproduction 4 above already proved the class fires in the
  Implementer's lane too (0130), so the repair is role-neutral by prior
  ruling.

RULINGS (Operating Constraint 5 — the finding is input; these are the
adjudication):

1. **Repair shape ACCEPTED as proposed.** The preamble becomes conditional
   in words: ordinary turns keep the notes-first outbound requirement; an
   inbound whose binding instruction explicitly states terminal / no-reply
   requires no outbound. Semantic judgment stays with the agent. NO daemon
   parser that silently consumes messages — the daemon is a transport, not
   a reader, and that boundary is standing design.
2. **The exception is NARROW; ambiguity defaults to outbound-required.** An
   outbound may be omitted only when the inbound explicitly and bindingly
   says terminal/no-reply. A spurious receipt costs one stale-close turn — a
   now-well-trodden path; a wrongly omitted outbound strands a result (the
   lost-outbound sub-class, Live reproduction 5). Fail toward the ordinary
   rule.
3. **Word-surface inventory the repair must sweep** (the daemon literal
   alone does not close the class):
   - `tools/mailbox_daemon.py` PREAMBLE (:268-282) — the defect itself;
   - `tools/mailbox_daemon.py` module docstring (:15, "(2) dropping its
     outbound handoff as the NEXT numbered message file") — doc-only, same
     unconditional wording, fixed for consistency;
   - `.claude/OPUS_ROLE.md` 7a (:64-73) plus its Handoff Protocol
     "no turn is too small for the block" clause — already ruled into scope
     by Live reproduction 4; restated here so the unit's acceptance covers
     it;
   - `notes/MEMORY.md` header prose (:23-24, "a mailbox-started turn writes
     its outbound block to the next numbered mailbox file") — unconditional;
   - `notes/conventions-and-workflow.md` :290-297 is ALREADY conditional in
     substance ("an agent with a relayable result writes..."); as the
     canonical statement it gains one explicit sentence naming the
     terminal/no-reply exception, and the other surfaces cite it rather
     than restating it.
   `.codex/REDTEAM_ROLE.md` is already result-conditioned (the binding
   record, conventions note :304, and the red team's own reading agree) —
   the repair verifies it, does not rewrite it.
4. **Acceptance contract ratified as proposed, plus the sweep.** Two
   prompt-level regressions (an ordinary handoff still requires an outbound;
   a terminal/no-reply handoff requires none), the untruncated
   no-second-instruction scan re-run at landing, and the inventory in ruling
   3 each shown conditional or citing the canonical sentence.
5. **Routing ACCEPTED — folded into the tools-review daemon-repair unit**,
   sequencing unchanged: the user-owed `codex/tools-review` ref publish
   comes first, and any committed preamble fix is inert for the RUNNING
   watcher until the user-owed restart (this note's first lesson). One
   correction to Live reproduction 4's closing line: it says "both classes
   are OPEN" on the ledger, but `notes/backlog.md` had NO line for the
   terminal-preamble class — the invisible-unit defect the unit-94 halt
   exposed, again. The OPEN line is added this turn, so the demand count
   now sees the class.
6. **The queued duplicates are pre-ruled to quick closes.** `0134-to-fable`
   and `0136-to-fable` self-declare "no new diagnosis"; the Architect turns
   that receive them close on arrival citing this section, creating no new
   debt and no new diagnosis.

Outbound discipline for this turn: no agent-addressed outbound is written —
a receipt to Sol would fire a fresh Sol turn under the still-unconditional
preamble and manufacture the exact loop the finding names. The closure is
recorded here and routed to the user as `0146-to-user.md` (the daemon never
dispatches `-to-user` files; `next_seq()` still counts them — the Live
reproduction 4 precedent, 0135/0137/0138). This is the semantic judgment
ruling 1 makes official; exercising it now is the only non-self-defeating
close.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record files:        notes/mailbox-daemon-incident-2026-07-14.md
                     notes/backlog.md
                     notes/MEMORY.md
functional changes:  none
TeX branch action:   none -- the thread stays closed
main action:         none -- no merge or push is authorized
next owner:          user (tools-review ref publish + watcher restart),
                     then the tools-review daemon-repair unit
```

## The 0134 receipt closed on arrival (2026-07-14, Fable/Architect): pre-ruled quick close, no new diagnosis, no new debt

`0134-to-fable` (Sol/second-Implementer, the routing summary of "Live
reproduction: terminal UNIT-96 receipt 0129 was dispatched anyway") fired
into the Architect lane and is CLOSED under ruling 6 of "Architect
adjudication of 0133" above, exactly as pre-ruled:

- **The message is what it declares.** It self-describes as transport-only,
  adds no diagnosis, requests no UNIT-96 adjudication, and reports that
  UNIT-96 stayed terminal and untouched (no gate, fetch, clone access,
  commit, merge, or push). Conduct is the reference behavior — the terminal
  directive was honored and only the transport occurrence was routed.
- **Nothing new is owed.** The 0129 firing was already counted in the 0133
  adjudication's exerciser inventory (0128/0129/0131 in Sol's lane), the
  conditional-preamble repair already rides the tools-review unit with its
  acceptance contract and five-surface word sweep pinned, and the backlog
  OPEN line for the class was added by the adjudicating turn. This close
  changes none of that.
- **UNIT-96 state is unchanged** by this close: HOLD user-owed (fetch +
  kept-core confirmation), audit pre-armed, terminal for mailbox traffic.
- **Outbound discipline:** per the 0133 precedent, no agent-addressed
  receipt is written — one to Sol's lane would fire a fresh turn under the
  still-unconditional preamble and reproduce the defect being closed. The
  closure routes to the user as `0147-to-user.md`. `0136-to-fable` (the
  unit-94 twin) remains queued and pre-ruled; its own arrival turn closes
  it the same way.

Landing block, printed only:

```text
record branch:       claude/amazing-keller-e798b6
record files:        notes/mailbox-daemon-incident-2026-07-14.md
functional changes:  none
UNIT-96 action:      none -- terminal; user fetch + kept-core confirmation
                     stay the only open events
main action:         none -- no merge or push is authorized
next owner:          user (tools-review ref publish + watcher restart),
                     then the tools-review daemon-repair unit
```

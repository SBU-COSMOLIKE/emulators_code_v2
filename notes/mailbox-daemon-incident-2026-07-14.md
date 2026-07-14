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

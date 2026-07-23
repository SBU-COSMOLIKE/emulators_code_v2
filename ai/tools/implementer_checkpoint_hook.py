#!/usr/bin/env python3
"""Ask a long-running Implementer to pause for an Architect review.

A hook is a small program that Claude Code (the agent runtime) launches
at fixed moments in a session; this file is such a hook. Claude Code
calls it after each batch of tool calls, when the session tries to
finish, and just before compaction, the step that condenses a long
conversation and loses working detail. Once the deadline has passed —
measured on the monotonic clock, the timer that only moves forward and
ignores wall-clock adjustments — the first call tells the Implementer
to write a short progress handoff. A small state file makes that
instruction one-shot: even if Claude Code invokes another hook
immediately, only one call delivers it.
"""

import fcntl
import io
import json
import math
import os
from pathlib import Path
import sys
import time

try:
    from ai.tools.role_contract import ROLE_CONTRACT
except ImportError:  # Direct execution from ai/tools/.
    from role_contract import ROLE_CONTRACT

DEADLINE_ENVIRONMENT = "MAILBOX_IMPLEMENTER_CHECKPOINT_DEADLINE"
STATE_ENVIRONMENT = "MAILBOX_IMPLEMENTER_CHECKPOINT_STATE"
SUPPORTED_EVENTS = {"PostToolBatch", "Stop", "PreCompact"}
TRIGGERED_MARKER = b"triggered\n"
CHECKPOINT_MINUTES = ROLE_CONTRACT["runtime"]["implementer_review_minutes"]
CHECKPOINT_INSTRUCTION = (
    f"The Implementer has worked for {CHECKPOINT_MINUTES} minutes, may be "
    "stuck, and must "
    "pause now. "
    "Make no further implementation edit. Let launched helpers finish, make "
    "one clean checkpoint commit, update the ticket note, and write `### "
    "IMPLEMENTER_HANDOFF: CHECKPOINT`. Begin Current state with "
    f"`{CHECKPOINT_MINUTES} minutes reached; work is paused and may be "
    "stuck.` Then briefly "
    "report changed "
    "production files, added plus deleted characters, completed checks, "
    "unfinished work, why the work took this long, and whether the design "
    "has become too complicated. Wait for the Architect's decision before "
    "doing more implementation."
)

CONTEXT_HANDOFF_INSTRUCTION = """\
The Implementer is about to lose detailed working context. Stop making source
edits. Send one same-cycle handoff to the Architect with this exact shape:

### IMPLEMENTER_HANDOFF: CONTEXT HANDOFF

- **Ticket and cycle:** `THE-CURRENT-MAILBOX-CYCLE`
- **Base commit:** `THE-DIRECTIVE-BASE-COMMIT`
- **Current worktree HEAD:** `THE-FULL-CURRENT-COMMIT`
- **Candidate created:** `yes` or `no`

#### Completed
- concrete result, or none

#### Known failures
- concrete failure, or none

#### Rejected approaches
- concrete rejected approach, or none

#### Uncommitted changes
- each path from git status --short, or none

#### Next exact action
- one concrete next action

#### Do not revisit
- rejected approach a replacement must not repeat, or none

This is a checkpoint, not candidate evidence or a completed ticket. End the
turn after sending it. The replacement will read this exact record and the
repository; the daemon will not invent a summary.
"""


def _checkpoint_payload(event):
    """Return the JSON instruction for one supported hook event.

    Claude Code reads one JSON object from a hook's standard output. An
    object of the form ``{"decision": "block", "reason": ...}`` stops
    the current action and shows the reason text to the model, so the
    finish and compaction events use it to force a handoff. The
    after-tools event cannot block anything; it attaches the checkpoint
    instruction as additional context the model reads on its next step.

    Arguments:
      event = one of ``SUPPORTED_EVENTS``: ``"PostToolBatch"`` (after a
              batch of tool calls), ``"Stop"`` (the session tries to
              finish), or ``"PreCompact"`` (compaction is imminent).

    Returns:
      The JSON text, newline-terminated, ready to print.
    """
    if event == "PreCompact":
        output = {"decision": "block",
                  "reason": CONTEXT_HANDOFF_INSTRUCTION}
    elif event == "PostToolBatch":
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolBatch",
                "additionalContext": CHECKPOINT_INSTRUCTION,
            }
        }
    else:
        output = {"decision": "block", "reason": CHECKPOINT_INSTRUCTION}
    return json.dumps(output, sort_keys=True) + "\n"


def _open_checkpoint_state(path):
    """Open the one-shot state file used as a crash-released lock.

    The file is opened read-write, created if missing, with owner-only
    permission bits. The caller then locks it with ``flock``, an
    advisory lock the operating system releases automatically when the
    process exits, so a crashed hook can never leave the checkpoint
    permanently locked.

    Arguments:
      path = filesystem path of the state file.

    Returns:
      An open file descriptor — the small integer the operating system
      uses to name an open file. The caller must close it.
    """
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return os.open(path, flags, 0o600)


def checkpoint_result(*, event, now, deadline, state_path,
                      output_stream=None):
    """Return ``(exit code, stdout, stderr)`` for one hook event.

    Compaction always receives the context-handoff instruction. The
    other two events do nothing before the deadline; after it, the
    first call wins a race through the state file: it takes the
    exclusive lock, reads the file, and only when the triggered marker
    is absent prints the checkpoint instruction and writes the marker.
    Every later call finds the marker and stays silent, which is what
    makes the instruction one-shot. If the marker cannot be recorded
    after the instruction text was staged, the partial marker is
    cleared and the call reports exit code 2, so the hook fails loudly
    instead of staying half-armed.

    Arguments:
      event         = one of ``SUPPORTED_EVENTS``; any other value
                      produces exit code 2 with an error message.
      now           = current monotonic-clock reading in seconds.
      deadline      = monotonic-clock reading after which the
                      checkpoint fires.
      state_path    = path of the one-shot state file.
      output_stream = stream that receives the JSON instruction, or
                      ``None`` to return the text in the tuple instead.

    Returns:
      ``(exit_code, stdout_text, stderr_text)``. Exit code 0 with empty
      text means there was nothing to do on this call; exit code 2
      carries the failure description in the stderr slot.
    """
    if event not in SUPPORTED_EVENTS:
        return 2, "", "unsupported checkpoint hook event: " + str(event) + "\n"
    if event == "PreCompact":
        output = _checkpoint_payload(event=event)
        if output_stream is None:
            return 0, output, ""
        output_stream.write(output)
        output_stream.flush()
        return 0, "", ""
    if now < deadline:
        return 0, "", ""
    descriptor = -1
    marker_started = False
    capture = io.StringIO() if output_stream is None else output_stream
    try:
        descriptor = _open_checkpoint_state(state_path)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, 32) == TRIGGERED_MARKER:
            return 0, "", ""
        capture.write(_checkpoint_payload(event=event))
        capture.flush()
        marker_started = True
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        written = os.write(descriptor, TRIGGERED_MARKER)
        if written != len(TRIGGERED_MARKER):
            raise OSError("short checkpoint marker write")
        os.fsync(descriptor)
    except OSError as error:
        if descriptor >= 0 and marker_started:
            try:
                os.ftruncate(descriptor, 0)
            except OSError:
                pass
        message = ("cannot deliver or record the " + str(CHECKPOINT_MINUTES)
                   + "-minute checkpoint: " + str(error))
        return 2, "", message + "\n"
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    output = capture.getvalue() if output_stream is None else ""
    return 0, output, ""


def main():
    """Read Claude Code's hook request and print the expected response.

    The request arrives as one JSON object on standard input. A request
    carrying an ``agent_id`` comes from a subagent — a helper session
    the Implementer launched — not from the Implementer itself, so it
    is ignored: helpers must keep working while the main session
    pauses. The deadline and state-file path arrive through the two
    environment variables named at the top of this file, which the
    mailbox daemon sets when it launches the Implementer.

    Returns:
      The process exit code: 0 for delivered or nothing-to-do, 2 for a
      malformed request or an undeliverable checkpoint.
    """
    try:
        request = json.load(sys.stdin)
        if request.get("agent_id"):
            return 0
        event = request.get("hook_event_name")
        deadline = float(os.environ[DEADLINE_ENVIRONMENT])
        state_text = os.environ[STATE_ENVIRONMENT]
        if not math.isfinite(deadline):
            raise ValueError("the deadline is not finite")
        state_path = Path(state_text)
        if not state_text or not state_path.is_absolute():
            raise ValueError("the checkpoint state path is not absolute")
    except (AttributeError, json.JSONDecodeError, KeyError, TypeError,
            ValueError) as error:
        print("invalid " + str(CHECKPOINT_MINUTES)
              + "-minute checkpoint hook input: " + str(error),
              file=sys.stderr)
        return 2

    code, output, error = checkpoint_result(
        event=event, now=time.monotonic(), deadline=deadline,
        state_path=state_path, output_stream=sys.stdout)
    sys.stdout.write(output)
    sys.stderr.write(error)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

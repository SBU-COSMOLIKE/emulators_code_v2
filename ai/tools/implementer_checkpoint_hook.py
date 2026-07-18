#!/usr/bin/env python3
"""Ask a long-running Implementer to pause for an Architect review.

Claude Code calls this file after a tool batch and again when it tries to
finish.  Once the monotonic deadline has passed, the first call tells the
Implementer to write a short progress handoff.  A small state file makes that
instruction one-shot even if Claude Code invokes another hook immediately.
"""

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
SUPPORTED_EVENTS = {"PostToolBatch", "Stop"}
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


def _claim_once(path):
    """Create the state file atomically and return whether this call won."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return False
    try:
        os.write(descriptor, b"triggered\n")
    finally:
        os.close(descriptor)
    return True


def checkpoint_result(*, event, now, deadline, state_path):
    """Return ``(exit code, stdout, stderr)`` for one hook event."""
    if event not in SUPPORTED_EVENTS:
        return 2, "", "unsupported checkpoint hook event: " + str(event) + "\n"
    if now < deadline:
        return 0, "", ""
    try:
        first = _claim_once(state_path)
    except OSError as error:
        message = ("cannot record the " + str(CHECKPOINT_MINUTES)
                   + "-minute checkpoint: " + str(error))
        return 2, "", message + "\n"
    if not first:
        return 0, "", ""
    if event == "PostToolBatch":
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolBatch",
                "additionalContext": CHECKPOINT_INSTRUCTION,
            }
        }
    else:
        output = {"decision": "block", "reason": CHECKPOINT_INSTRUCTION}
    return 0, json.dumps(output, sort_keys=True) + "\n", ""


def main():
    """Read Claude's hook event and print the response Claude expects."""
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
        state_path=state_path)
    sys.stdout.write(output)
    sys.stderr.write(error)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

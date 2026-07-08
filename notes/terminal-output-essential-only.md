---
name: terminal-output-essential-only
description: "STANDING RULE 2026-07-08 (user): a CLI's terminal shows essential progress only — gate headers, verdicts, one-line details, artifact paths; the full streams (driver stdout, config dumps) go to per-run log files. A debug switch restores the full mirror. Log files are NEVER thinned. First applied in gates/run_board.py (a debug key in board_config.json + a --debug flag)."
metadata:
  node_type: memory
  type: feedback
---

# Terminal output: essential-only

Standing rule the user set 2026-07-08, after the board's run 3.

## The rule

A CLI's **terminal** carries only what a watcher needs to follow the run:
progress markers (a one-line header per unit of work), pass/fail
verdicts, the one-line detail behind each verdict, and the paths of any
artifacts written. The **full streams** — a subprocess's entire stdout,
a dumped effective-config, anything verbose — go to a **per-run log
file**, not the terminal. A `debug` switch (a config key and/or a
`--debug` flag) restores the full mirror for when you are actually
debugging. **Log files are never thinned**: they always get everything,
so the record stays complete; only the terminal is quieted.

## Why

Run 3 of the gates board buried its four FAIL verdicts under 19 gates'
worth of per-epoch training scroll — the signal (which gate failed, and
why) was lost in the noise. The terminal is a dashboard, not an archive;
the log file is the archive.

## First applied

`gates/run_board.py`: `RunContext._emit` gained a `log_only` route (the
tee'd command output and the config-dump header go to the log alone), a
required `debug` key in `gates/board_config.json` (committed false;
missing = a loud preflight failure), and a `--debug` flag that forces the
full mirror. The terminal shows the one-line gate header
(`GATE <id> [<tier>] started <hh:mm:ss>`), the `CHECK` / `GATE`
verdicts, preflight, the dry-run plan, and the end-of-run summary; the
gate log keeps the full multi-line header, the config dump, and every
streamed line — byte-compatible with the pre-rule format.

Part of the user-run gates harness ([[gates-harness-user-run]]); the
board and its order live in [[workstation-board-2026-07]].

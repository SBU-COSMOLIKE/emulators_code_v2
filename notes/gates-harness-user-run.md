---
name: gates-harness-user-run
description: "SPEC 2026-07-07 (Architect): the user-run gates harness — gates/run_board.py executes the whole workstation board ([[workstation-board-2026-07]]) with NO Claude session on the workstation (user decision 2026-07-07: 'There is no Claude code session on the workstation. I have to run them'). Framework: stdlib-only CLI (--check preflight, --list, --gate/--tier/--from selectors, --dry-run, resume-by-default via gates/logs/board_status.json), one raw log per gate under gates/logs/<GATE>.log (tee stdout+stderr; the Architect audits RAW LOGS, never summaries), a final BOARD.md table. GM-C's pinned pre-EMA leg runs in a TEMPORARY git worktree at 46ec5e1 (never checkout-in-place). Dependency skip: GSV-C failure auto-skips GCT-C (its artifact feeds the parity probe). Each gate's acceptance is encoded from its HOME note (the note is the spec of record); the harness never edits notes/ — logs are evidence, note verdicts land in the Architect's audit pass. User flow: git pull -> --check -> run -> commit gates/logs -> relay. NOT implemented."
metadata:
  node_type: memory
  type: project
---

# The gates harness: the user runs the board, the harness drives

User decision 2026-07-07: no Claude Code on the workstation — the
user executes the board personally. So the board's execution mode
changes from "workstation Implementer session" to "a committed,
self-driving harness the user launches": `python gates/run_board.py`.
This supersedes the handoff block inside
[[workstation-board-2026-07]] (the board's CONTENT — gates, order,
homes — is unchanged and stays the checklist of record).

## Design rules (binding)

1. **Zero babysitting.** After `run_board.py` starts, nothing asks
   for input. Long GPU runs stream progress to the log; the user
   can walk away.
2. **Raw logs are the evidence.** Every gate writes
   `gates/logs/<GATE_ID>.log` — the FULL stdout+stderr of every
   command it ran (tee'd, so the terminal shows it live), plus the
   harness's own PASS/FAIL verdict line and the acceptance values
   it computed. The Architect audits raw logs; a gate whose log
   holds only a summary FAILS the audit by construction.
   `gates/logs/BOARD.md` = the final pass/fail table +
   `board_status.json` = machine state for resume.
3. **Resume by default.** A gate already PASS in board_status.json
   is skipped on rerun (`--force-rerun <GATE>` overrides). A crash
   mid-board loses only the in-flight gate.
4. **Preflight before GPU time** (`--check`, and always run first):
   (a) the BASE-NOTES commit hash (the commit carrying the board +
   this spec, recorded as a constant at build time) is an ANCESTOR
   of HEAD (`git merge-base --is-ancestor`) — never an exact-tip
   equality, since committing logs moves the tip and a commit
   cannot name its own hash; (b) the working tree is clean in
   emulator/ + gates/ + the drivers (`git status --porcelain` —
   a dirty tree is not a reproducible run); (c) cocoa env
   importable (torch, cuda visible, cosmolike, cobaya); (d) the
   data paths the gate configs name exist. Any failure prints the
   remedy and exits nonzero.
5. **GM-C's pinned leg never touches the user's tree**: the golden
   pre-EMA leg runs in a TEMPORARY `git worktree add <tmp> 46ec5e1`
   (created + removed by the harness); the current-code leg runs in
   place. No checkout-in-place, ever.
6. **Dependency skip, not abort.** Gates are independent unless
   declared: GSV-C's saved artifact feeds GCT-C — if GSV-C fails,
   GCT-C records SKIPPED(dependency) and the board continues.
   Any other gate's failure never stops the rest.
7. **The home note is each gate's spec of record.** The Implementer
   encodes every gate's commands + acceptance check from its home
   note (listed in [[workstation-board-2026-07]]), not from memory.
   Numeric acceptances (bitwise equality, rtol 1e-6 parity, the
   drift proof's monkeypatched equality, census counts) are
   computed and asserted by the harness itself; run-shaped gates
   (the --diagnostic production run, training legs) assert on the
   banner/log patterns their home notes name.
8. **The harness never mutates notes/**. Logs only. Per-gate
   verdicts are appended to home notes in the Architect's audit
   pass, after the logs come back.
9. **Selectors**: `--list` (the board with status),
   `--gate GSV-C [GCT-C ...]`, `--tier standing|week|save-sample`,
   `--from <GATE>`, `--dry-run` (print every command a selection
   would run, no execution). Default = the whole board in the
   board note's order.
10. **Dependencies**: stdlib-only harness (argparse, subprocess,
    json, pathlib); the gates themselves may import the emulator
    package / torch as needed. House style
    ([[py-module-style-conventions]]) binds all new code.

## Layout

    gates/
      run_board.py          the CLI + runner (order, resume, logs)
      board.py              the gate table: id, tier, home note,
                            commands, acceptance fn, deps
      checks/               one module per non-trivial acceptance
                            (gsv_bitwise_drift.py, gct_parity.py,
                            gm_golden_ema.py, gwd_census.py, ...)
      logs/                 (gitignored EMPTY, committed after runs)
        <GATE_ID>.log
        BOARD.md
        board_status.json

## The user's run instructions (final form, also printed by --help)

    # on the workstation
    cd <repo>          # the cocoa clone of emultrfv2
    git pull           # tip must be the harness commit
    <activate the cocoa env>
    python gates/run_board.py --check       # preflight only
    python gates/run_board.py --dry-run     # optional: see the plan
    python gates/run_board.py               # the whole board, in order
    git add gates/logs
    git commit -m "workstation board run: logs"
    git push           # then tell the Architect the logs are in

The Architect audits the committed raw logs, appends per-gate
verdicts to the home notes, and either declares the board green or
issues targeted rerun/fix instructions (`--force-rerun <GATE>`).

## Gates (content unchanged — see [[workstation-board-2026-07]])

Tier `standing`: GM-C (worktree-pinned golden leg FIRST), GM-D,
DIAG (the one production --diagnostic run closing
G-F/GN-F/GS-D/GT-C/G1), GP-D, GH-E, GE-C, GB-C, GL-D, GBA-C,
GME-C, item-27, GT-B (registered but marked optional; skipped
unless --gate names it).
Tier `week`: GFT-C, GHA-F, GAN-C, GWD-C, GPC-C.
Tier `save-sample` (LAST, ordered): GSV-C (bitwise + drift proof;
one plain + one factored + one NPCE save so the geometry-class
marker and the pce group round-trip) -> GCT-C (parity probe rtol
1e-6, the factored save->rebuild->predict round-trip, the example
evaluate run vs the lsst_y1 likelihood with the printed datavector
compared, an MCMC evaluate + short-chain smoke). The MPS-float64
rider stays a dev-Mac item, NOT in this harness.

## Gates on the harness itself (Mac, before it ships)

- GGH-A (framework, stub-runnable on the Mac): --list/--dry-run
  print the full board in order; resume skips a PASS gate;
  --force-rerun overrides; the dependency skip fires (a stub-failed
  GSV-C marks GCT-C SKIPPED); preflight fails loudly on a wrong
  git tip and on a missing import; the GM-C worktree is created at
  46ec5e1 and removed even on failure (try/finally); every command
  a gate runs is tee'd into its log.
- GGH-B (static): each of the 19 gates' acceptance encodes checks
  traceable to its home note (the Implementer lists, per gate, the
  home-note line(s) each assertion implements — the Architect
  audits that mapping); house style; py_compile; the five-rule
  scanner if any doc changes ride along.
- The gates' REAL acceptance happens on the workstation — the
  harness run IS the board run; no GPU gate is closable on the Mac.

## Handoff (relay to the Implementer)

### ARCHITECT_HANDOFF
Task: the user-run gates harness (spec:
notes/gates-harness-user-run.md in full — the ten design rules are
binding, esp. raw-log evidence, resume, the GM-C temporary
worktree, the GSV-C->GCT-C dependency skip, stdlib-only framework,
and encode every gate from its HOME note, never from memory; the
board content/order is notes/workstation-board-2026-07.md). Base:
the commit carrying this spec + the board note (git log -1 shows
it — else STOP); record THAT hash as the harness's base-notes
constant (preflight rule 4a: ancestor-of-HEAD, never exact-tip). Scope: gates/ exactly (framework + the 19 gates + GT-B
optional), plus a one-line pointer in the board note's handoff
section marking it superseded by this harness. Budget wall: STOP
at a valid tree with the framework + as many gates as encoded;
resume state here names the remainder. Gates GGH-A/B. Report:
IMPLEMENTER_HANDOFF + resume state appended here, raw gate
outputs, deviations declared. Do not commit: print the suggested
commit command (explicit paths; gates/logs stays empty).
### END

## Status

SPEC DELIVERED 2026-07-07, NOT implemented. Sequenced after the
board note lands on main (the harness's expected-tip constant
needs that commit hash). After the harness commit: the user runs
the board on the workstation per the instructions above and
commits gates/logs; the Architect audits the raw logs.

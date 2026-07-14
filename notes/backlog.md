# Execution backlog — the countable ledger

One line per OPEN unit of execution work, maintained by the Architect.
The mailbox daemon counts the "- OPEN" lines here PLUS the queued
mailbox messages; that total is the queue depth the second-Implementer
threshold (SECOND_IMPLEMENTER_THRESHOLD = 10) compares against — at or
past it, the red team is the second Implementer and build units flow
to both lanes. A unit leaves this
list when it lands and its audit records GO — not when it is
dispatched. Details live in the notes each line names; this file is
deliberately just countable lines.

- OPEN unit 74 (artifact chain, CRITICAL): immutable per-attempt logs + atomic status/board publication — notes/red-team-audit-and-didactics-2026-07-13.md
- OPEN unit 77 (artifact chain, CRITICAL): unknown/mixed selector handling — same register
- OPEN unit 80 (artifact chain, CRITICAL): see register
- OPEN unit 76 (artifact chain): see register
- OPEN unit 78 (artifact chain): see register
- OPEN unit 84 (artifact chain, fixed-facts adapter half): re-dispatched 0140 under rulings 3+4+5 + riders (F1 amendment leg, F2 vertical arm, bsn missing-quantity leg, cs refusal arms) — notes/gates-and-board.md landings-2+3 audit 2026-07-14
- OPEN unit 85 (artifact chain, fixed-facts adapter half): re-dispatched 0140 with unit 84 — same audit entry
- OPEN unit 96: add-or-toggle vs declared unmasked artifact — ruled contract in notes/families-background-mps.md:1217 + red-team register :2897 (the training-stack.md section arrives with the codex/unit-96 landing; transport HOLD adjudicated in the register, 2026-07-14)
- OPEN unit 94 (boundary-interior half; blocks unit 8): interval-coordinate helper + pre-sampling refusal in generator_core.py's uniform branch — red-team-owned (f46166c), dispatched 0117 — notes/state-2026-07-11-and-next.md adjudication 2026-07-14; candidate FROZEN at unlinked-clone tip a0a03a9 — user fetch owed, audit pre-armed (register 0121 adjudication, 2026-07-14)
- OPEN unit 8: rebased on unit 94's seam — BLOCKED until unit 94 lands and its audit records GO (halt adjudicated 2026-07-14) — notes/state-2026-07-11-and-next.md
- OPEN unit 24: see state note
- OPEN unit 56: resume machinery — kept, see state note
- OPEN unit 62 (D5 remainder): see didactics register
- OPEN units 64/70 (D5 remainder): see didactics register
- OPEN staging reopen (unit 32 successor): needs Architect ruling
- OPEN unit-93 hold resolution (83e4507)
- OPEN daemon: --dry-run mutates (placeholder check precedes the dry_run branch, mailbox_daemon.py 236-247) — repair rides the tools-review daemon-repair unit; adjudicated in the README-DELTA audit, notes/gates-and-board.md 2026-07-14
- OPEN daemon: user-facing prints violate the register (` -- ` + all-caps emphasis) — RULED in the README-DELTA audit (notes/gates-and-board.md 2026-07-14): fix the prints in the tools-review repair unit and refresh the README's quoted lines in the same series; the verbatim quotes stand until then
- OPEN unit 41-REPAIR, awaiting Architect audit/publication: persisted amp_dtype/scaler_policy + one immutable sweep-product record are implemented; the positive witness is 16/16 GREEN and all touched Python compiles (register: gates-and-board.md, "Unit 41-REPAIR implementation return"). Git publication is blocked only by this turn's read-only linked-worktree metadata, so no SHA exists yet.
- OPEN unit 53-REPAIR: canonical study manifest + digest + name resolver (witness: gates/checks/redteam_unit53_manifest_witness.py) — UNBLOCKED 2026-07-14 (fixed-facts landing 1 audited GO)
- OPEN gate defect: generator_ranges retired-header mutation arm is hollow on GetDist 1.6.2 (comment lines skipped; red on HEAD, proven pre-existing) — notes/gates-and-board.md landing-1 audit 2026-07-14
- OPEN gate defect: transfer-identity.cross-family-base-refusal is hollow (fixture config lacks n_train, the leg dies before the rule it names; red on HEAD on any torch machine) — notes/gates-and-board.md landing-1 audit 2026-07-14
- OPEN staging defect: the .paramnames cross-check is silently skipped on every real <paramsf>.1.txt chain (stem yields <paramsf>.1.paramnames; absence treated as skip) — repair template is read_facts_sidecar's stem resolver — notes/gates-and-board.md landing-1 audit 2026-07-14
- OPEN unit 90 rebase: content GO, merge HELD for the batch-5 conflict rebase + Architect delta re-audit — notes/gates-and-board.md UNIT 90 verdict
- OPEN unit-13 covariance package (25M-08/11/12 + 45M-01): reassigned to the red team (f46166c), never dispatched — notes/state-2026-07-11-and-next.md REASSIGNMENT EXTENDED
- OPEN daemon: the dispatch PREAMBLE unconditionally orders an outbound (mailbox_daemon.py 268-282), contradicting terminal/no-reply inbounds (fired four times 2026-07-14: 0128/0129/0131 in Sol's lane, 0130 in the Implementer's) — repair = conditional wording + the five-surface word sweep + two prompt-level regressions + the untruncated no-second-instruction scan, riding the tools-review daemon-repair unit; adjudicated ACCEPTED in notes/mailbox-daemon-incident-2026-07-14.md, "Architect adjudication of 0133" (2026-07-14)
- OPEN daemon: no staleness/supersession check — a dispatch can deliver a pointer whose work already returned (fired three times 2026-07-14: 0110 toward the Implementer, 0120 toward the Architect, 0132 re-fired after a timeout kill — benign, and it exposed the lost-outbound sub-class: a killed turn's commits survive but its outbound dies, incident note Live reproduction 5) — repair rides the tools-review daemon-repair unit; acceptance shape in notes/gates-and-board.md mailbox-0120 adjudication 2026-07-14
- OPEN reader defect: check_names_match runs only at save (results.py:382); rebuild never re-proves the record's names against the rebuilt geometry (residual exposure = a coordinated edit of blocks + sidecar text) — notes/gates-and-board.md landings-2+3 audit 2026-07-14
- OPEN tools-review daemon-repair unit: publish the ref, then the daemon/README repair series the three daemon lines above ride — red-team register adjudication 2026-07-14
- OPEN daemon: --send sol discovery-ticket deferral guard + `--fix-only` watch flag — (a) when total demand >= SECOND_IMPLEMENTER_THRESHOLD, a Sol-bound message that is discovery/attack work (product = new findings, not a closed ledger line) is refused with instructions to append it to the END of this ledger instead; (b) NEW USER DIRECTIVE 2026-07-14 (second): a `--fix-only` option on --watch — when set truthy (accept 1/true/yes in ANY capitalization: parse with .strip().lower(), never an exact-string compare, "the user can make mistakes in capitalization"), the loop is closing-only: Fable sends Sol NO adversarial discovery tickets and no new tickets are created, regardless of demand; document it in --help and the notes/ README options section; code edit deferred to an idle watch (source change retires the running watch) — rides the tools-review daemon-repair series; the procedural rule is live now in .claude/FABLE_ROLE.md

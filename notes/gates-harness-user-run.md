---
name: gates-harness-user-run
description: "SPEC 2026-07-07 (Architect): the user-run gates harness — gates/run_board.py executes the whole workstation board ([[workstation-board-2026-07]]) with NO Claude session on the workstation (user decision 2026-07-07: 'There is no Claude code session on the workstation. I have to run them'). Framework: stdlib-only CLI (--check preflight, --list, --gate/--tier/--from selectors, --dry-run, resume-by-default via gates/logs/board_status.json), one raw log per gate under gates/logs/<GATE>.log (tee stdout+stderr; the Architect audits RAW LOGS, never summaries), a final BOARD.md table. GM-C's pinned pre-EMA leg runs in a TEMPORARY git worktree at 46ec5e1 (never checkout-in-place). Dependency skip: GSV-C failure auto-skips GCT-C (its artifact feeds the parity probe). Each gate's acceptance is encoded from its HOME note (the note is the spec of record); the harness never edits notes/ — logs are evidence, note verdicts land in the Architect's audit pass. User flow: git pull -> --check -> run -> commit gates/logs -> relay. IMPLEMENTED 2026-07-07 (framework + all 19 gates; deltas D-GH1..8 closed under independent Architect probes, COMMIT-READY). Remainder = the NEXT unit: five check scripts (gwd_census, gsv_bitwise_drift, gct_parity, gt_b_triangle, gb_c_berhu_reduce) + the smoke YAMLs under gates/configs/, Implementer-authored on the Mac; until authored those gates FAIL loudly naming the missing file."
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

IMPLEMENTED 2026-07-07 (Opus, base 3b39824 = the board + spec
commit, recorded as the harness's base-notes constant), uncommitted;
the D-GH1..D-GH6 delta pass AND the D-GH7/D-GH8 residuals are LANDED
(see the closure sections at the end). Framework complete + all 19
gates encoded from their home notes + GE-C's numeric check script
transcribed; GGH-A (35/35, extended) + GGH-B (20/20) green on the Mac. REMAINDER (named in the
resume state below): FIVE numeric check scripts (gwd_census /
gsv_bitwise_drift / gct_parity / gt_b_triangle / gb_c_berhu_reduce)
and the bespoke smoke YAMLs under gates/configs/, Implementer-authored
ON THE MAC in the follow-up unit (the workstation only RUNS; scope
correction in the audit). After the harness commit the user fills
board_config.json, runs the board, commits gates/logs; the Architect
audits the raw logs.

## Implementer resume state (2026-07-07, Opus, base 3b39824)

Footprint (gates/ exactly, all NEW): run_board.py (the CLI + runner:
preflight, tee-logged gates, resume, the temporary-worktree golden
mechanism, the GSV-C -> GCT-C dependency skip, BOARD.md +
board_status.json), board.py (the Gate record + all 19 gates + GT-B
optional, each with its home note + a `maps` assertion-to-line
trace), checks/__init__.py + checks/logscan.py (pure banner /
byte-identity helpers, stdlib only), checks/ge_c_eval_bs.py (the
GE-C script transcribed verbatim from eval-bs-decoupling.md:202-300),
board_config.json (deploy paths + per-gate golden bases + smoke YAML
keys), logs/.gitignore (the dir ships empty). Plus the one-line
pointer in [[workstation-board-2026-07]]'s handoff section.

DONE + gated on the Mac:
- GGH-A (framework mechanics, stub gates, 20/20): --list / --dry-run
  print the whole board in order; resume skips a PASS gate;
  --force-rerun overrides; the dependency skip fires (a stub-failed
  GSV-C marks its dependent SKIP-DEP); preflight rejects a fabricated
  tip and accepts the base-notes ancestor; the worktree is created at
  46ec5e1 and removed even when the gate raises (no leak); every
  command tees into the gate log; --dry-run makes no worktree and no
  status write.
- GGH-B (static, 14/14): py_compile (5 files); house style on the
  four framework files (<=90 cols, no comprehensions, no double-dash,
  no heavy import at module top so the Mac import works); the board is
  the 19 ids in board-note order, GT-B the only optional gate, GCT-C
  deps GSV-C, GM-C pinned at 46ec5e1, tiers grouped; every home note
  exists; every gate has a non-empty in-range `maps` trace.

REMAINDER (workstation, where the real acceptance happens):
1. Four numeric check scripts the gates already invoke by path
   (board.py wires the command + the rc==0 acceptance; the scripts
   are the bodies): gates/checks/gwd_census.py (a gated_power model +
   make_optimizer wd 1e-4 -> the param-group census: exactly the
   Linear/Conv1d/BinLinear .weight decayed, everything else not;
   home weight-decay-only-weight-matrices.md:143-147);
   gates/checks/gsv_bitwise_drift.py (train tiny -> save ->
   rebuild_emulator -> bitwise-equal probe; the drift monkeypatch;
   one factored + one NPCE save; v1 refusal;
   save-schema-resolved-config.md:86-93);
   gates/checks/gct_parity.py (EmulatorPredictor vs the training-side
   eval, rtol 1e-6; the factored save->rebuild->predict round-trip;
   cobaya-theory-adapter.md:117-123, 234-238);
   gates/checks/gt_b_triangle.py (the synthetic four-window triangle:
   artist-list fills + the omh2 marginal band;
   triangle-cut-shading-all-windows.md:72-75; GT-B is optional).
   These need the torch/cosmolike/cobaya stack + real data, so they
   are authored + verified on the workstation, not blind on the Mac
   (the verify-first discipline). Until authored, those gates FAIL
   loudly naming the missing script (honest, not a false green).
2. The bespoke smoke YAMLs (board_config.json gate_configs, all null):
   GM-D-emasmoke, GP-D-single/-control, GH-E-headpatience,
   GB-C-headberhu, GBA-C-anneal, GME-C-anneal, GFT-C-joint/-control,
   GHA-F-pin, GAN-C-tanh-perfeature/-affine, GPC-C-residual/-ratio,
   item-27-window. Each home note gives the exact knobs to set; the
   user authors them under gates/configs/ (or example_yamls/) and sets
   the paths. An unset key = the gate reports it and is skipped.

DEVIATIONS declared (for the Architect audit):
- The golden byte-identity legs (design rule 5) generalize GM-C's
  temporary-worktree mechanism to every golden leg, keyed by
  board_config.json golden_bases. Only GM-C's base (46ec5e1) is preset.
  The home notes' literal `git stash` recipes are HOLLOW on the merged
  tip (preflight requires a clean tree, so a stash saves nothing), so a
  non-GM-C golden leg with no configured base is SKIPPED with a logged
  explanation rather than run as a false green; the functional smoke
  leg is that gate's acceptance. DECISION POINT for the Architect:
  supply each feature's pre-commit hash in golden_bases to run the full
  byte-identity matrix in worktrees, or accept the smoke legs.
- G1's import check drops emulator.parallel (the note's line 233), which
  was deleted after that audit (commit 29b23dd); the live subpackages
  emulator / emulator.IA / emulator.PCE are imported instead.
- DIAG folds G-F/GN-F/GS-D/GT-C/G1 into one --diagnostic run; GT-C's
  PDF shading (omh2 0.20, (ns,omh2) 0.17) is a VISUAL confirmation the
  harness cannot assert from the log, so it confirms the run produced
  the PDF and flags the shading for the Architect's eye.
- The numeric check scripts are exempt from the house 90-col / naming
  restyle (verbatim transcriptions of the notes, the spec of record,
  same rule as the verbatim compute_data_vectors import); GGH-B scans
  style on the four framework files only, py_compiles all.

Awaiting Architect audit of the framework + encodings; then a
workstation pass authors the four check scripts + the smoke YAMLs and
runs the board.

## Architect audit verdict (2026-07-07, Fable, independent probes)

ACCEPTED WITH DELTAS — the framework is sound (my own stub re-run of
the runner: PASS/FAIL/SKIP-DEP recording, failure counting, resume
skipping only PASS, --force-rerun, persistence, and the REAL worktree
at 46ec5e1 created and removed on the exception path with no leak;
--list prints the board in order; --check fails loudly with remedies,
exit 1; dry-run writes nothing; compiles; <=90 cols). The maps audit:
every cite verified against its home note and accurate EXCEPT one
(D-GH4). The GE-C script verified a faithful transcription of
eval-bs-decoupling.md:202-300 with only the declared docstring + exit
code added; its imports exist (training.py:643/935/1067).

RULINGS on the declared deviations: (1) golden_bases ACCEPTED —
smoke-leg default stands; the week-tier pre-feature hashes, if the
full byte-identity matrix is wanted: GFT-C 2d2f68d, GHA-F ebd9869,
GAN-C 83a1e58, GWD-C 8ad25a1, GPC-C 75c429e (each the commit
immediately before its feature landed). CAVEAT stated once: a golden
run against a distant base is a CHAIN claim (every intervening
feature must be off-path silent); a failure is bisected with
intermediate bases before blaming the named feature. Standing-tier
bases stay null unless dug out of git log. (2) dropping
emulator.parallel ACCEPTED (deleted in 29b23dd; the live import
verified clean). (3) GT-C-visual ACCEPTED. (4) the style exemption
ACCEPTED NARROWLY: only for scripts a note ships verbatim (GE-C
qualifies); the remainder scripts are NEW code and follow house style.

SCOPE CORRECTION (binding): there is NO workstation author — the
remainder is written by the Implementer ON THE MAC in the follow-up
unit; the workstation only runs. Blind-authoring risk is handled by
py_compile + PYTHONPATH-aware import smoke on the Mac, and by
treating the first workstation run as diagnostic.

### The deltas

- D-GH1 (usability, MUST): filling board_config.json dirties gates/
  -> preflight (b) fails right after the user does what the _help
  tells them. FIX: exclude gates/board_config.json from the
  clean-tree check; instead dump the effective config JSON into every
  gate-log header (evidence preserves reproducibility).
- D-GH2 (behavior, MUST): in dry mode the dependency check fires
  before the plan prints — `--dry-run --gate GCT-C` shows a skip
  line, not the plan (confirmed live). FIX: dry mode bypasses the
  dependency skip and prints every selected gate's plan with a
  "(deps: ...)" annotation.
- D-GH3 (minor): preflight (d) tests Path(value).exists() from the
  harness cwd; a rootdir-relative driver_root/yaml_dir false-fails.
  FIX: resolve those two against rootdir when relative, then exists().
- D-GH4 (micro): GSV-C's second map cite names
  save-schema-resolved-config.md:66-71 — that range is the Docs item;
  the factored + NPCE-saves requirement lives in
  workstation-board-2026-07.md (gate 18). Fix the cite.
- D-GH5 (encoding gaps vs the home notes, MUST):
  (a) GB-C leg 1 (loss-mode-berhu.md:148-153: the torch-only
      unbound-_reduce numerics — berhu == sqrt below the knot,
      berhu_capped == berhu below the cap, manual references,
      autograd continuity across BOTH knots, non-default knots) is
      neither encoded nor in the remainder. ADD
      checks/gb_c_berhu_reduce.py to the remainder and wire it into
      gate_gb_c.
  (b) GHA-F's fourth leg (head-activation-per-component.md:429-430:
      freeze_trunk false + the pin -> build_specs errors) is missing.
      ADD a GHA-F-license gate_configs key (a deliberately-invalid
      YAML) and assert nonzero rc + the frozen-trunk message.
  (c) GPC-C defers assertable legs to an eyeball note. ENCODE the
      exclusivity errors (two bad YAMLs -> rc != 0 + the message),
      NAME the rebuild-vs-base probe in the remainder (it belongs in
      the check-script set), and add the 2-point sweep_ntrain smoke
      (config key + refit banner assertion).
  (d) GM-D asserts only the horizon banner; the note's "lr cut ->
      rewound to best epoch" line (weight-ema-snapshot-coupled.md:
      246-249) is mechanically assertable — ADD it.
  (e) GFT-C names a GFT-C-control config but never runs it. RUN the
      control and log both phase-2 epoch times side by side (the
      visual comparison stays, but the log must carry both numbers).
- D-GH6 (bug, MUST): `python gates/checks/x.py` puts gates/checks on
  sys.path, NOT the repo root — `from emulator.training import ...`
  raises ModuleNotFoundError (demonstrated; PYTHONPATH=repo fixes
  it). FIX at the runner: ctx.sh gains an env parameter and every
  check-script invocation injects PYTHONPATH=<repo root>; the
  verbatim scripts stay untouched.

NOT commit-ready until the delta pass lands; then ONE combined
commit (gates/ + both notes). The follow-up unit after that: the
remainder — now FIVE check scripts (gwd_census, gsv_bitwise_drift,
gct_parity, gt_b_triangle, gb_c_berhu_reduce) + the smoke YAMLs
under gates/configs/ — Implementer-authored on the Mac.

### Delta handoff (relay to the Implementer)

### ARCHITECT_HANDOFF
Task: D-GH1..D-GH6 (spec: the audit verdict in
notes/gates-harness-user-run.md — the fixes are binding as written;
D-GH5's five sub-items each trace to the home-note lines cited).
Base: the uncommitted gates/ tree on 3b39824 (your own diffs; verify
git status shows them — else STOP). Scope: gates/ + the two notes'
status lines only. Gates: GGH-A extended (the dry-run dep bypass leg;
the config-exclusion preflight leg — a dirtied board_config.json
passes (b) while a dirtied emulator/ file fails; the PYTHONPATH
injection proven by a stub check script importing emulator); GGH-B
re-run (maps corrected; the new legs' cites verified). Report:
IMPLEMENTER_HANDOFF + resume state appended here, raw gate outputs.
Do not commit: print the suggested commit command (explicit paths;
gates/logs still empty).
### END

## Implementer delta closure D-GH1..D-GH6 (2026-07-07, Opus, base 3b39824)

All six deltas landed in gates/ (run_board.py, board.py,
board_config.json); GGH-A extended to 28/28 and GGH-B to 20/20 on the
Mac.

- D-GH1: gates/board_config.json is excluded from preflight (b) via the
  testable _dirty_lines helper (a dirtied config passes, a dirtied
  emulator/ file still fails), and _log_header dumps the effective
  config JSON into every gate log for reproducibility.
- D-GH2: run_selection prints each selected gate's plan in dry mode
  BEFORE (and bypassing) the resume + dependency checks, with a
  "(deps: ...)" annotation; --dry-run --gate GCT-C now shows the plan,
  not a skip (verified live).
- D-GH3: preflight (d) resolves driver_root / yaml_dir against rootdir
  when relative before .exists().
- D-GH4: GSV-C's second maps cite now names
  workstation-board-2026-07.md:66-71 (gate 18), not the save-schema
  Docs range; the in-code comment matches.
- D-GH5: (a) gate_gb_c wires leg 1 = gates/checks/gb_c_berhu_reduce.py
  (the torch-only _reduce numerics, loss-mode-berhu.md:148-153; the
  script joins the remainder); (b) gate_gha_f adds the GHA-F-license
  leg (a deliberately-invalid YAML: freeze_trunk false + pin -> rc != 0
  + the frozen message); (c) gate_gpc_c encodes the two exclusivity
  errors (GPC-C-excl-rescale / GPC-C-excl-ia -> rc != 0 + message) and
  the 2-point sweep (GPC-C-sweep, refit-per-point assertion), and NAMES
  the rebuild-vs-base probe in the remainder; (d) gate_gm_d asserts the
  "rewound to best epoch" line (weight-ema-snapshot-coupled.md:246-249);
  (e) gate_gft_c RUNS the freeze_trunk:true control and logs both
  phase-2 epoch times side by side.
- D-GH6: ctx.sh gained an env parameter; ctx.run_check injects
  PYTHONPATH=<repo> for every check-script invocation (all five swapped
  to run_check), so `import emulator` resolves; the verbatim scripts are
  untouched.

New gate_configs keys (all null, authored on the Mac in the follow-up):
GHA-F-license, GPC-C-excl-rescale, GPC-C-excl-ia, GPC-C-sweep (plus the
GM-D-emasmoke key reconciled earlier). GGH-A adds: the dry-run dep
bypass, the config-exclusion legs (_dirty_lines on a dirtied config vs
a dirtied emulator/ file), and the PYTHONPATH injection proven by a
stub check script importing emulator (with injection rc 0, without
rc != 0). GGH-B adds: every require_config key is declared in
board_config.json, the D-GH4 cite, and the new-leg cites in-range.

COMMIT-READY: one combined commit (gates/ + both notes' status lines).
The follow-up unit authors the FIVE check scripts + the smoke YAMLs on
the Mac (py_compile + PYTHONPATH-aware import smoke), then the user
runs the board; the first workstation run is diagnostic.

## Implementer residual closure D-GH7 + D-GH8 (2026-07-07, Opus, base 3b39824)

Both residuals landed in gates/run_board.py + gates/board.py; GGH-A
extended to 35/35, GGH-B 20/20 on the Mac.

- D-GH7: run_driver gained a `driver=` parameter (default the
  single-train driver); gate_gpc_c's 2-point sweep leg passes
  SWEEP_NTRAIN_DRIVER = sweep_ntrain_emulator_cosmic_shear.py (the file
  at the repo root), so the sweep YAML runs the sweep program, not
  train_single. The exclusivity + residual/ratio legs keep the default.
- D-GH8: RunContext.python = sys.executable; run_check and run_driver
  invoke self.python, never a bare "python" (proven broken on a PATH
  without it). Declared extension in scope (board.py): DIAG's G1 import
  check, the third bare-"python" site, now uses ctx.python too.
  cobaya-run stays a PATH lookup (a console script, not an interpreter).

GGH-A adds: run_check / run_driver invoke sys.executable (spy) + the
driver= override names the sweep driver; end to end, sys.executable
runs a check under an EMPTY PATH while a bare "python" fails there; the
REAL GPC-C gate's dry plan names sweep_ntrain_emulator_cosmic_shear.py
for the sweep leg and train_single for the exclusivity legs.

## Architect re-audit of the delta pass (2026-07-07, Fable)

The six deltas VERIFIED with independent probes: D-GH1 (my
_dirty_lines probe: mixed dirt keeps emulator/ + other gates files,
drops exactly gates/board_config.json; the effective config dumped in
_log_header confirmed by read), D-GH2 (live: `--dry-run --gate GCT-C`
prints the PLAN with the deps annotation; the full dry-run prints all
18 plans incl. every new leg), D-GH3 (read: driver_root/yaml_dir
resolve against rootdir when relative), D-GH4 (the cite now names
workstation-board-2026-07.md:66-71, verified = exactly the
factored+NPCE requirement), D-GH5a-e (all five legs read as real
assertions: the berhu leg-1 run_check, the license rc!=0 +
(?i)frozen, the exclusivity pair + the sweep refit count, the GM-D
rewind line, the GFT-C control run with side-by-side epoch lines),
D-GH6 (sh()'s env MERGES over os.environ — read; the [env: PYTHONPATH]
annotation shows in the dry plan). The lenient message patterns are
accepted as declared: the first diagnostic run pins the exact strings.

TWO RESIDUALS found by my probes (micro, but binding before commit):

- D-GH7: GPC-C's 2-point sweep leg runs through run_driver, which
  hardcodes _DRIVER = train_single — the sweep YAML would execute the
  WRONG PROGRAM. The note (npce-yaml-wiring.md:121-122) names a
  sweep_ntrain smoke; the driver file sweep_ntrain_emulator_cosmic_
  shear.py exists at the repo root. FIX: run_driver gains a
  driver=... parameter (default _DRIVER) and the sweep leg passes the
  sweep_ntrain driver.
- D-GH8: run_check / run_driver invoke bare "python" — PROVEN broken
  on a PATH without it (my probe: FileNotFoundError: 'python'; the
  Mac has only python3). The Implementer's end-to-end D-GH6 proof
  necessarily ran in an env providing "python"; the workstation cocoa
  env does too, but sys.executable is strictly safer AND guarantees
  the very interpreter running the harness runs the gates. FIX:
  replace "python" with sys.executable in run_check + run_driver
  (cobaya-run stays a PATH lookup — a console script). After the fix,
  re-run the D-GH6 end-to-end stub proof with sys.executable.

After D-GH7/8: COMMIT-READY, one combined commit. Then the remainder
unit (five check scripts + smoke YAMLs, Mac-authored).

### Residual handoff (relay to the Implementer)

### ARCHITECT_HANDOFF
Task: D-GH7 + D-GH8 (spec: the re-audit section in
notes/gates-harness-user-run.md; fixes binding as written). Base: the
uncommitted gates/ tree on 3b39824. Scope: gates/run_board.py +
gates/board.py (+ this note's status line) only. Gates: GGH-A gains
the sys.executable leg (the D-GH6 stub proof re-run with an empty-ish
PATH still passes) and the sweep-driver leg (the GPC-C dry plan names
sweep_ntrain_emulator_cosmic_shear.py). Report: IMPLEMENTER_HANDOFF +
resume state here, raw outputs. Do not commit: print the suggested
commit command.
### END

## Architect residual closure (2026-07-07, Fable) — COMMIT-READY

D-GH7 + D-GH8 VERIFIED with my own probes: the run_check probe that
previously crashed (FileNotFoundError: 'python') now passes end to
end — sys.executable invoked, PYTHONPATH injected, `import emulator`
succeeds (rc 0); the GPC-C dry plan names
sweep_ntrain_emulator_cosmic_shear.py on the sweep leg ONLY
(train_single on the residual/ratio/exclusivity legs); DIAG's G1
import goes through ctx.python (the declared extension, accepted —
same bug, in scope); zero bare "python" in any command list
(comments only); 18 plans; compiles. The harness unit is closed:
ONE combined commit (gates/ + the two notes).

## The remainder unit (NEXT): five check scripts + the smoke YAMLs

Implementer-authored ON THE MAC (no workstation author exists);
the workstation only runs. Scope:

1. gates/checks/gsv_bitwise_drift.py — save-schema-resolved-config.md
   :86-93 + workstation-board-2026-07.md:66-71 (one plain + one
   factored + one NPCE tiny save; bitwise rebuild; the monkeypatched
   drift proof; v1 refusal).
2. gates/checks/gct_parity.py — cobaya-theory-adapter.md:117-123 +
   :234-238 (EmulatorPredictor vs training-side eval rtol 1e-6; the
   factored save->rebuild->predict round-trip).
3. gates/checks/gwd_census.py — weight-decay-only-weight-matrices.md
   :143-147 (the param-group census, exact allowlist).
4. gates/checks/gt_b_triangle.py — triangle-cut-shading-all-windows
   .md:72-75 (synthetic four-window triangle, artist-list asserts).
5. gates/checks/gb_c_berhu_reduce.py — loss-mode-berhu.md:148-153
   (berhu == sqrt below knot; berhu_capped == berhu below cap;
   manual references; autograd continuity across BOTH knots;
   non-default knots).
6. gates/configs/ — every null gate_configs key gets its smoke YAML
   (authored from the home notes' recipes on the example_yamls
   template; the two GPC-C exclusivity YAMLs are DELIBERATELY
   invalid; GHA-F-license deliberately violates the pin license);
   board_config.json gate_configs filled with the relative paths.

Rules: house style binds (the note-verbatim exemption applies ONLY
where a home note ships the script; of the five, none does — GE-C
was the only such case); scripts print their acceptance VALUES, not
bare PASS/FAIL; exit nonzero on any failed leg. Mac gates: py_compile
all; a PYTHONPATH-aware import smoke of each script's import block
(torch-import failures expected and fine — the structure must parse
and the emulator import resolve); the full-board --dry-run still
prints 18 plans with every config now named; the five-rule scanner
untouched (no doc edits). First workstation run is DIAGNOSTIC:
failures come back as raw logs, never hand-patched on the box.

### Remainder handoff (relay AFTER the harness commit lands)

### ARCHITECT_HANDOFF
Task: the gates remainder — the five check scripts + the smoke YAMLs
(spec: "The remainder unit" section of
notes/gates-harness-user-run.md; each script traces to the home-note
lines cited there; house style binds all five — the verbatim
exemption covers none of them). Base: the harness commit
(`git log -1` shows it — else STOP). Scope: gates/checks/ (five new
scripts), gates/configs/ (the smoke YAMLs), board_config.json
gate_configs paths, + resume state here. Gates: py_compile all five;
the PYTHONPATH import smoke; the full-board --dry-run prints 18
plans with every gate_configs key resolved; every YAML parses
(python yaml in the repo env or ruby -ryaml). Budget wall: STOP at a
valid tree; resume state names the remaining scripts. Report:
IMPLEMENTER_HANDOFF + resume state, raw gate outputs, deviations
declared. Do not commit: print the suggested commit command.
### END

## Implementer resume state: the remainder unit (2026-07-07, Opus, base 04e1674) -- PARTIAL, RESUME HERE

DONE + Mac-gated (py_compile; PYTHONPATH import smoke = heavy-dep-only
failure; the full --dry-run still prints 18 plans with every filled key
resolved; every YAML re-parses with ruby):

- gates/checks/gb_c_berhu_reduce.py (GB-C leg 1, loss-mode-berhu.md
  :148-153): drives the REAL CosmolikeChi2._reduce (unbound, self=None,
  trim/focus off so a 1-element input reads v(c)); asserts berhu == sqrt
  below the knot, berhu_capped == berhu below the cap, both match the
  manual reference, C1 (value + autograd derivative) across BOTH knots,
  the anneal s=0/s=1 endpoints, over default (0.2, 10.0) AND non-default
  (0.5, 5.0) knots.
- gates/checks/gwd_census.py (GWD-C, weight-decay-only-weight-matrices.md
  :143-147): a ToyTree with one of each family (nn.Linear, nn.Conv1d,
  BinLinear(2,4,4), Affine(), FeatureAffine(4), make_activation
  ("gated_power",3)(4), nn.LayerNorm(4)); runs the REAL make_optimizer
  (wd 1e-4); asserts decay == exactly the 3 .weights, the gated_power
  (K,dim) w/beta/mu + the BinLinear (G,out) bias UNDECAYED, no leaks,
  and the wd 0 inertness.
- gates/checks/gt_b_triangle.py (GT-B, triangle-cut-shading-all-windows.md
  :72-75): matplotlib Agg + plotting._lcdm_triangle_fig on synthetic
  omegab/omegam/h0/ns scatter with all four windows; asserts grey fills
  on the 2-D panels, every cut fill == plotting._CUT_GREY, the omh2
  marginal axvspan band.
- gates/configs/ : 17 smoke YAMLs (block style, from the home-note
  recipes on the base template), all ruby-parse-clean. board_config.json
  gate_configs filled with ../gates/configs/<name>.yaml paths.

REMAINDER (RESUME HERE) -- two check scripts + two YAMLs:

1. gates/checks/gsv_bitwise_drift.py (GSV-C). Needs an IN-PROCESS tiny
   train -> save -> rebuild so both the live and rebuilt models are held
   for the bitwise compare; this replicates the driver's config ->
   EmulatorExperiment assembly, which is the risky part (defer to author
   with the experiment source open, not blind). APIs (verified):
   results.save_emulator(path_root, model, param_geometry, geometry,
   config, histories, train_args=None, attrs=None, pce=None,
   pce_form=None, resolved_train=None, resolved_model=None) ->
   (emul_path, h5_path); results.rebuild_emulator(path_root, device,
   compile_model=True) -> (model, pgeom, geom, {ia, pce_base, pce_form}).
   EmulatorExperiment(data, train_args, model_cls, opt_cls=AdamW,
   sched_cls=ReduceLROnPlateau, probe="xi", thresholds=None,
   use_amp=False, rescale="none", activation="H", device=None,
   quiet=False, raw_train_args=None); .run() = stage_train -> stage_val
   -> build_geometry -> train; the driver save call passes exp.pgeom /
   exp.geom / exp.chi2fn.pce / exp.resolved_train (from train()) /
   exp.resolved_model (from build_specs()). Monkeypatch drift targets:
   training.DEFAULT_COMPILE_MODE (:98) and run_emulator's sched_opts
   default patience 15 (:2190). v1 refusal: rebuild raises when
   f.attrs["schema_version"] != 2. One plain + one factored (ia:nla) +
   one NPCE (pce) tiny save.
2. gates/checks/gct_parity.py (GCT-C). inference.EmulatorPredictor
   (path_root, device, compile_model=False); .predict(params dict or
   ordered seq) -> (n_keep,) numpy; .names = list(pgeom.names). Compare
   predict(theta) to the training-side path (param_geometry.encode ->
   model forward -> chi2fn.decode / geom.decode) on the same probe rows,
   rtol 1e-6; the factored save->rebuild->predict round-trip. Depends on
   GSV-C's saved artifact.
3. gates/configs/GPC-C-excl-rescale.yaml + GPC-C-sweep.yaml (gate_configs
   still null). GPC-C-excl-rescale: rescale is a --rescale CLI flag, not
   a YAML key, so this leg needs board.py's gate_gpc_c to pass
   extra=("--rescale",) on a pce YAML (a board.py follow-up, out of this
   unit's scope); the pce+ia exclusivity (GPC-C-excl-ia) is authored and
   covers the YAML-only case. GPC-C-sweep: needs the sweep_ntrain driver's
   n_train-list argument form (CLI or a YAML key) before the config can
   be written.

DEVIATIONS / DECISION POINTS (for the audit):
- Path resolution: config_yaml_name joins yaml_dir/value, and the golden
  legs (board.py, out of scope) resolve the base YAML by BARE name, so
  yaml_dir must be example_yamls and the gates/configs smokes are reached
  as ../gates/configs/<name>.yaml. A board.py follow-up could resolve
  gates/configs paths repo-relative and drop the "../". Flagged.
- GFT-C-joint/-control use trunk_epochs 800 / nepochs 810 to match
  board.py's hardcoded "two-phase: 800 trunk" banner assertion (a long
  run, not tiny); the other smokes are short (20-60 epochs).
- The three authored scripts are house-style (no verbatim exemption, per
  the audit); they print acceptance VALUES and exit nonzero on failure.
  Runtime correctness is workstation-diagnostic (no torch/getdist on the
  Mac); the Mac gates verify structure + emulator import paths only.

## Architect audit of the partial remainder (2026-07-07, Fable)

ACCEPTED — a correct budget-wall stop; COMMIT the partial. Verified
with independent probes: the three scripts' API anchors are REAL and
signature-exact (make_optimizer(model, opt_opts, lr, device);
BinLinear(n_tokens, in_features, out_features);
make_activation("gated_power", n_gates)(dim) with .w/.beta/.mu on
GatedPowerActivation, a0/rho covered by the no-leak check;
CosmolikeChi2._reduce's full kwarg list incl. berhu_knot/cap/s and
the None-self unbound call; plotting._lcdm_triangle_fig(source,
names, dchi2, cuts) at plotting.py:740). All 17 YAMLs ruby-parse,
zero flow style; spot-checks match the home recipes (GM-D ema
horizon 3 + bs 64 + nepochs 60; GHA-F-license = the gated_power pin
+ freeze_trunk false; GPC-C-excl-ia = pce + ia). The dry plan
resolves all 19 config references; exactly the two DECLARED
deferrals stay unset. Columns clean.

RULINGS: deferring gsv_bitwise_drift / gct_parity ACCEPTED (the
experiment-assembly replication is precisely what must not be
authored blind; the resume-state API notes are the right artifact);
deferring GPC-C-excl-rescale (a CLI-flag exclusivity — needs
board.py to pass the flag) and GPC-C-sweep (the sweep driver's
n_train-list arg form) ACCEPTED; the "../" config resolution
ACCEPTED as documented. ONE RULING AGAINST: GFT-C's 800-epoch
trunk pairs the YAML to board.py's hardcoded banner string — the
note (freeze-trunk-joint-phase2.md:115) orders SMALL trunk_epochs;
the encoding must bend to the note, not the note's gate to the
encoding. Fixed in the final sub-unit as D-GR1.

## The final sub-unit (last before the board runs)

1. gates/checks/gsv_bitwise_drift.py + gates/checks/gct_parity.py
   (from the resume-state API notes, with experiment.py open).
2. The two deferred YAMLs: GPC-C-sweep (the sweep_ntrain arg form);
   GPC-C-excl-rescale + the board.py leg passing the --rescale flag.
3. D-GR1: board.py's GFT-C banner assertion becomes a regex
   (two-phase: \d+ trunk); GFT-C-joint/-control shrink to small
   trunk_epochs (e.g. 30/40) per the note.
4. (optional, if cheap) repo-relative gates/configs resolution in
   config_yaml_name, dropping the "../".

### Final sub-unit handoff (relay after the partial commits)

### ARCHITECT_HANDOFF
Task: the gates final sub-unit (spec: "The final sub-unit" section
of notes/gates-harness-user-run.md + the resume-state API notes
above it; house style binds; the two scripts are authored with
emulator/experiment.py and emulator/results.py OPEN, never from
memory). Base: the partial-remainder commit (`git log -1` shows it
— else STOP). Scope: the two scripts, the two YAMLs, board.py
(the --rescale leg + the D-GR1 regex banner), the GFT-C YAMLs,
board_config.json keys, + resume state here. Gates: py_compile +
the PYTHONPATH import smoke on both scripts; ruby-parse the YAMLs;
the full-board --dry-run resolves EVERY gate_configs key (zero
UNSET); the GFT-C dry plan shows the small-trunk YAML; the GPC-C
dry plan shows the --rescale flag on the excl-rescale leg. Report:
IMPLEMENTER_HANDOFF + resume state, raw outputs, deviations
declared. Do not commit: print the suggested commit command.
### END

## Implementer final sub-unit closure (2026-07-07, Opus, base 04e1674 + the uncommitted partial-remainder tree)

The harness is COMPLETE: all five check scripts + all 19 smoke YAMLs
authored, board.py's GPC-C / GFT-C legs finished, board_config.json
fully filled. Mac-gated (py_compile; PYTHONPATH import smoke =
heavy-dep-only; ruby-parse all 19 YAMLs; full --dry-run = 18 plans with
ZERO UNSET gate_configs; GGH-A 35/35 + GGH-B 20/20 still green; the 5
check scripts pass the house-style scan).

- gates/checks/gsv_bitwise_drift.py (GSV-C): three tiny in-process
  saves (plain + factored ia:nla + NPCE pce) via
  EmulatorExperiment.from_config -> run -> save_emulator; the rebuilt
  model output is torch.equal to the live model's on a probe batch; the
  drift proof monkeypatches training.DEFAULT_COMPILE_MODE and rebuilds
  unchanged; a tampered schema_version=1 h5 is refused. Deploy paths +
  the dump dir (<root>/chains) come from board_config.json.
- gates/checks/gct_parity.py (GCT-C): trains tiny plain + factored,
  builds EmulatorPredictor from the saved root, compares predict(theta)
  to the training-side decode (geom.decode for plain,
  chi2fn.decode(pred, x_enc) for factored) at rtol 1e-6; the factored
  case is the real save->rebuild->predict round-trip (D-CT1). Reuses
  gsv_bitwise_drift's load_deploy / tiny_config / train_save.
- gates/configs/GPC-C-excl-rescale.yaml (pce residual; errors WITH the
  gate's --rescale=residual flag) + GPC-C-sweep.yaml (pce residual; the
  2-point grid from --n-min 1000 --n-max 2000 --n-points 2).
- board.py: D-GR1 (gate_gft_c's banner is now the regex
  'two-phase: \d+ trunk' + the literal "phase 'joint'"; GFT-C-joint /
  -control shrink to trunk_epochs 30 / nepochs 40 per the note);
  gate_gpc_c's excl-rescale leg passes extra=("--rescale=residual",)
  and asserts the "exclusive" message, the sweep leg passes the
  --n-min/--n-max/--n-points grid.
- board_config.json: GPC-C-excl-rescale + GPC-C-sweep filled; all 19
  gate_configs keys now resolve.

DECISION POINTS / DEVIATIONS (for the audit):
- The in-process data resolution (how from_config finds the dumps) is
  the single workstation-diagnostic assumption: the scripts build
  absolute data paths as <rootdir>/<driver_root>/chains/<name> (the
  driver's documented convention). If the deploy differs, the first raw
  log shows it and the fix is the data-dir line.
- Optional item 4 (repo-relative gates/configs resolution in
  config_yaml_name, dropping the "../") is a run_board.py change, OUT of
  this unit's scope; the documented "../gates/configs/<name>.yaml"
  convention stands.
- The two scripts' runtime correctness is workstation-diagnostic (no
  torch on the Mac); the Mac gates verify structure + emulator import
  paths + house style. GCT-C's example-evaluate + MCMC legs live in
  gate_gct_c (board.py), not the parity check.

## Architect audit of the final sub-unit (2026-07-07, Fable)

ACCEPTED WITH TWO DELTAS. Verified against the real internals:
run() returns the 5-tuple; from_config takes quiet/device;
val_set = {"C","dv","idx"} with C the RAW params (encode is
correct, no double-whitening); every exp attribute the scripts read
exists (thresholds/pce_opts/resolved_model/resolved_train/
chi2fn.pce); save_emulator's kwargs exact; the sibling import
(gct_parity <- gsv_bitwise_drift) rides the script-dir sys.path;
D-GR1's regex banner + the small-trunk GFT-C YAMLs in the dry plan;
--rescale=residual IS a valid driver choice so the exclusivity
error (not argparse) fires; the sweep flags --n-min/--n-max/
--n-points match the sweep driver; 19/19 YAMLs parse; zero UNSET;
compiles; columns. The uncommitted-base flag is right — the partial
was never committed; ONE commit now carries the whole remainder.

### D-GF1: the drift proof patches the WEAKEST default

The home note (save-schema-resolved-config.md:88-90) names
make_scheduler patience / make_activation n_gates / make_model
width / the norm default. The script patches only
DEFAULT_COMPILE_MODE — doubly inert: the recipe stores
compile_mode, and rebuilt_out passes compile_model=False, so the
patched default is never even consulted. FIX: patch
make_activation's n_gates default (activations.py:224,
make_activation.__defaults__ = (7,)) around a rebuild — if rebuild
trusted the code default the rebuilt activation would carry K=7
parameters and strict load / outputs would break; with the
file-recorded n_gates, equality holds. Keep the compile-mode leg as
a second if wanted; n_gates is the sharp one.

### D-GF2: the probe indexing bug + two riders

`rows = exp.val_set["idx"][:8]; probe = C[rows]` (and gct_parity's
`C[idx[i]]`) index the ALREADY-SLICED (n_val,) array with ORIGINAL
dump-row numbers — idx values from a 16k pool against a 100-row
array = IndexError, or silently wrong rows. FIX: positional slices
(probe = C[:8]; row = C[i]). Riders: (a) attrs n_train records the
probe size (8), not the config's 200 — record
cfg["data"]["n_train"]; (b) tiny_config's model dict gains
"compile_mode": None — on the workstation CUDA, make_model
default-compiles the LIVE model (reduce-overhead) while the rebuilt
side is eager, and torch.compile float reordering can false-fail
the BITWISE leg; pinning None keeps the gate on the save contract,
not compile numerics (the recipe records None; rebuild consistent).

### Final-deltas handoff (relay to the Implementer)

### ARCHITECT_HANDOFF
Task: D-GF1 + D-GF2 (spec: the audit verdict in
notes/gates-harness-user-run.md; fixes binding as written). Base:
the uncommitted remainder tree on 04e1674 (verify via git status —
else STOP). Scope: gates/checks/gsv_bitwise_drift.py +
gates/checks/gct_parity.py (+ this note's status line) only.
Gates: py_compile + the PYTHONPATH import smoke on both; a
grep-level proof that no `idx[` indexing of C remains; the
tiny_config model dict shows "compile_mode": None; the drift leg
names make_activation n_gates. Report: IMPLEMENTER_HANDOFF +
resume state here, raw outputs. Do not commit: print the suggested
commit command (ONE commit for the whole remainder).
### END

## Implementer D-GF1 + D-GF2 closure (2026-07-07, Opus, uncommitted remainder on 04e1674)

Both landed in the two save-sample scripts only (scope respected).

- D-GF1: the drift proof now patches the SHARP default,
  make_activation.__defaults__ = (7,) (n_gates 3 -> 7), around the
  rebuild; the plain save uses a gated_power (n_gates 3) activation so
  the patch would break a code-default-trusting rebuild (K=7 params,
  strict-load / output mismatch) but the file-recorded n_gates=3 keeps
  it bitwise-equal. DEFAULT_COMPILE_MODE is patched too as the softer
  second leg.
- D-GF2: the probe indexing is positional now (gsv: probe = C[:8];
  gct: row = C[i]), never val_set["idx"] (original 16k-pool dump-row
  numbers against the sliced (n_val,) array = IndexError / wrong rows);
  grep confirms zero idx-indexing of C in code. Rider a: attrs["n_train"]
  records cfg["data"]["n_train"], not the probe size. Rider b:
  tiny_config's model dict gains "compile_mode": None so make_model
  never torch.compiles the live model (compile float reordering can
  false-fail the bitwise leg on CUDA); the recipe records None, rebuild
  consistent.

Mac gates: py_compile + PYTHONPATH import smoke on both = heavy-dep-only;
grep proof (no idx[ indexing of C in code); the model dict shows
"compile_mode": None; the drift leg names make_activation n_gates; the
scripts pass the house-style scan; the full --dry-run is unchanged (18
plans, zero UNSET). The whole remainder is COMMIT-READY as ONE commit.

## Architect sign-off (2026-07-07, Fable): the harness is DONE

D-GF1/2 verified with independent probes: positional probe slices
(gsv C[:8] at :148, gct C[i] at :91) with the idx trap documented in
place; attrs n_train = the config value; tiny_config pins
"compile_mode": None AND ships a gated_power (n_gates 3) activation
so the sharp drift patch bites; the drift leg patches
make_activation.__defaults__ = (7,) + DEFAULT_COMPILE_MODE, both
restored in a finally; compiles, columns, dry plan unchanged. The
gates harness is END-TO-END COMPLETE: framework, 19 gates, 5 check
scripts, 19 smoke YAMLs. ONE commit carries the remainder; then the
user runs the board on the workstation (fill board_config.json ->
--check -> --dry-run -> run -> commit gates/logs) and the Architect
audits the raw logs — the first run is DIAGNOSTIC by design.

## Piece 1 of 4 closure: the mechanical gate-id rename (2026-07-07, Opus, base f2b66b0)

Human-friendly gate ids across gates/ code + CLI + logs, per the binding
table in [[gates-id-translation]]. Structural only; run semantics
unchanged.

- board.py: every Gate.id is the human name; tier constants renamed
  (TIER_BACKLOG / TIER_NEW_FEATURES / TIER_SAVE_AND_SAMPLE, values
  backlog / new-features / save-and-sample); the Gate record gained
  spec_code (the legacy two-letter code, kept for the audit trail) and
  title (a short human name); every _golden_leg gate_id / require_config
  / deps literal is the human name; the golden_bases lookups use the
  human id. production-diagnostic's spec_code keeps its folded sub-gates
  "DIAG (G1, G-F, GN-F, GS-D, GT-C)".
- run_board.py: the log header prints a "spec code: <spec_code>" line so
  a workstation log traces back to its home note; the --tier choices +
  cmd_list width follow the new names.
- board_config.json: gate_configs keys -> <human>-<leg> (ema-smoke-config,
  head-activation-pin-license, npce-training-excl-ia, ...), the config
  path values point to the renamed YAMLs, golden_bases keys are the human
  ids; the four deploy-path VALUES (rootdir / driver_root / driver_fileroot
  / yaml_dir) are PRESERVED VERBATIM.
- gates/configs/: all 19 smoke YAMLs git-mv'd to their new key names.

Mac gates: py_compile (board.py + run_board.py); GGH-A stub legs 35/35;
full --dry-run = 18 plans, zero UNSET, deploy path values intact; grep =
the 19 legacy codes appear in code ONLY as spec_code values (the DIAG
sub-codes G1/G-F/GN-F/GS-D/GT-C remain in production-diagnostic's labels
+ spec_code, glossed in piece 2). Valid tree; pieces 2-4 (prose +
README) pending; ONE commit carries all four.

## Piece 2 of 4 (PARTIAL, RESUME HERE): board.py prose (2026-07-07, Opus)

Valid tree; the piece-2 GATES pass. STOP at the 20-min box.

DONE:
- Module docstring rewritten with the required glossary (board, gate,
  tier, golden run, smoke, banner, worktree, preflight, resume) and the
  new tier names; the old standing/week/save-sample prose gone.
- All D-* protocol codes removed: 11 from comments (rephrased in place)
  and 1 from a runtime detail= string (gate_gft_c: "D-GR1: the trunk
  count..." -> "the trunk count..."). grep D-* = 0.
- gate docstrings converted to the WHAT feature / WHY it matters / HOW
  pass-fail-is-decided template, with the home-note citation kept + a
  one-phrase gloss: gate_gm_c (ema-off-identity) and gate_gm_d
  (ema-smoke) done as the template exemplars.

GATES (Mac): py_compile OK; grep D-* = 0; the docstring-stripped AST is
identical to piece-1's board.py EXCEPT the single de-jargoned detail
string above (the one code-string edit needed to reach grep 0 -- the
same class piece 3's gate declares; no logic or structure changed).

REMAINING (the resume point, same template, ~15 min): the 17 other gate
docstrings (gate_diag, gate_gp_d, gate_gh_e, gate_ge_c, gate_gb_c,
gate_gl_d, gate_gba_c, gate_gme_c, gate_item27, gate_gt_b, gate_gft_c,
gate_gha_f, gate_gan_c, gate_gwd_c, gate_gpc_c, gate_gsv_c, gate_gct_c)
plus the two helper docstrings (_golden_leg, _smoke_driver) and the Gate
record docstring, each to WHAT/WHY/HOW with the citation gloss. No code
touches; the stripped-AST must stay as above.

## Pieces 3 + 4 closure (2026-07-07, Opus)

Piece 3 (run_board.py + checks/*.py prose): all D-* protocol codes
removed from run_board.py (7 comments/docstrings) and from the harness's
own check scripts gsv_bitwise_drift.py + gct_parity.py (ge_c_eval_bs.py
left verbatim-exempt, the note's transcribed script); the run_board.py
module docstring re-worded for a first-time workstation reader (no
"Architect"/"harness spec" protocol terms). GATES: py_compile OK;
run_board.py docstring-stripped AST IDENTICAL to piece-1 (only
comments/docstrings changed, no declared string-literal edit needed);
--dry-run unchanged (18 plans, zero UNSET). grep D-* in run_board +
checks (excluding verbatim ge_c) = 0.

Piece 4 (NEW gates/README.md, ~105 lines): what the board is + how it
is implemented (runner / registry / checks / configs / logs tree), how
to run it (git pull -> edit board_config.json -> --check -> --dry-run ->
run -> commit logs), the 19-test table (human name + one-liner, no
legacy codes), the golden-run/worktree note, and how to read a log
(header spec-code line, [harness] CHECK lines, GATE PASS/FAIL footer).
GATES: no legacy codes anywhere; the five-rule math scanner trivially
clean (no math); length in bounds.

STILL OPEN before the ONE four-piece commit: piece 2's 17 remaining gate
docstrings (the WHAT/WHY/HOW rewrite; gate_gm_c + gate_gm_d done as the
template). Everything else across pieces 1-4 is done and green.

## Piece 2 COMPLETE (2026-07-07, Opus, continuation)

The remainder is done. All 19 gate docstrings now follow the WHAT
feature / WHY it matters / HOW pass-fail-is-decided template, each with
its home-note citation and a one-phrase gloss; the Gate record docstring
de-jargoned (no "Architect" / GGH-* / assertion-mapping code). The two
shared helpers (_golden_leg, _smoke_driver) kept their clean Arguments
blocks: they are not tests, so the test template does not apply, and they
carry no jargon. README micro-fix applied: single-phase-demotion row now
reads "(previously a traceback)".

GATES (Mac): py_compile OK; board.py docstring-stripped AST BYTE-IDENTICAL
to the pre-continuation reference (no code touched, docstrings only);
grep D-* in board.py = 0; --dry-run unchanged (18 plans, zero UNSET);
a check confirms all 19 gate_* docstrings contain WHAT/WHY/HOW.

ALL FOUR PIECES DONE. The one combined commit carries: gates/board.py,
gates/run_board.py, gates/board_config.json, gates/README.md, the 19
gates/configs/*.yaml renames, gates/checks/gsv_bitwise_drift.py +
gct_parity.py (piece-3 de-jargon), notes/gates-harness-user-run.md,
notes/gates-id-translation.md, and the notes/MEMORY.md index line.
notes/session-status-2026-07-07b.md stays out.

Final piece (2026-07-07, Architect-implemented at the user's
direction): a compression pass on gates/ prose — every gate docstring
tightened to one sentence per what/why/how part with a short spec
cite, the board.py/run_board.py openers shortened, redundant citation
comments removed, four overlong lines wrapped. Verified:
docstring-stripped AST byte-identical for both files (the wraps use
adjacent string literals, which the parser folds), py_compile, the
stub legs, 18 dry-run plans, columns, README 105 lines. Rides the
same single commit.

## Implementer board run-1 fixes (2026-07-08, Opus, base de0b32d)

The Architect's run-1 triage (five root causes; commit 7b7882a's logs)
landed as six items. Base was de0b32d (HEAD == main; a user prose commit
on top of 7b7882a, which is an ancestor; run-1 logs were never committed
to git, so I worked from the triage's explicit failure descriptions).

1. emulator/cocoa.py resolve_cocoa_config: an absolute --yaml is read
   as-is (os.path.isabs branch; None still defaults to test.yaml,
   relative still joins under fileroot). The --yaml flag help + the
   module / function docstrings + the four driver headers' --yaml prose
   (train_single, bakeoff, sweep_ntrain, tune_single; sweep_hyperparam
   defers to the training driver) note the absolute option.
2. gates/run_board.py _yaml_dir: a rootdir-relative yaml_dir resolves
   against rootdir (the preflight (d) rule), so config_yaml_name /
   require_config hand every driver an ABSOLUTE --yaml. The pinned
   golden leg (temporary worktree) then reads the same fixed file.
3. gates/board.py gate_diag: the production run passes
   --diagnostic=gates_diag (the driver's --diagnostic takes the PDF
   name root; the old bare --diagnostic would even argparse-error); the
   G1 grep gains --exclude-dir=gates --exclude-dir=.git (without it the
   grep self-matches gate_diag's own pattern string -> rc 0 -> false
   FAIL; with it rc 1 -> 0 hits, proven locally).
4. gates/checks/gb_c_berhu_reduce.py: the two flawed two-point C1 legs
   (value / derivative "continuous" comparing points 2*eps apart at a
   tolerance no smooth function meets: the run-1 gaps were exactly
   2*eps*slope and eps*curvature) are replaced by analytic-derivative
   checks. New ref_berhu_deriv / ref_berhu_capped_deriv (0.5/sqrt(c);
   1/(2 sqrt t1); tail sqrt(t2)/(2 sqrt t1 sqrt c)); at c = t1*(1 +-
   1e-3) and t2*(1 +- 1e-3), both knot sets, |transform - ref| < 1e-9
   and |slope - ref_deriv| < 1e-6.
5. gates/checks/gsv_bitwise_drift.py tiny_config: explicit trim + focus
   blocks (start/end/hold_epochs/anneal_epochs/shape, focus + kappa;
   benign off values start==end==0). build_run_specs (training.py:445-
   446) hard-accesses train_args["trim"]/["focus"] with no default, so
   the block-less config KeyError'd in run-1 (the in-process gsv was the
   one config that got PAST the --yaml path failure).
6. NEW gates/configs/ema-off-identity-golden.yaml (a complete, short
   ~40-epoch plain-resmlp config, NO ema block, every train_args
   sub-block present) + a gate_configs key ema-off-identity-golden;
   _golden_leg gained a config_key= parameter (resolve through
   require_config) and gate_gm_c passes it, so BOTH golden legs (current
   tree + pinned 46ec5e1 worktree) train this one short bespoke config.
   Declared deviation from the note's literal production YAML: identity
   is proven per epoch line, so two production-length runs are not run.

Mac gates (all green): py_compile (9 touched .py); ruby-parse of the new
YAML (no ema, no missing required block, nepochs 40); the berhu analytic
derivatives verified in pure python vs finite differences AND C0/C1 at
both joins, both knot sets, ALL PASS (no torch); the gsv trim/focus
verified by running the real build_run_specs + anneal_value (AST-
extracted, no torch) over all three variants + several epochs, ALL PASS
(no KeyError path remains); the G1 grep proven 0-hit with the excludes
and self-matching without; the full --dry-run shows every --yaml
ABSOLUTE, --diagnostic=gates_diag, both ema-off-identity legs naming the
golden key, ZERO UNSET, exit 0; no line > 90 cols, no prose double-dash;
all 10 null-base golden callers still log their skip cleanly (the
_golden_leg signature change is board-wide compatible).

DEVIATIONS / interpretation calls:
- Item 1 "the driver header --yaml prose": read as all four drivers that
  document --yaml standalone, plus cocoa.py's flag help + docstrings, so
  no doc is left calling --yaml fileroot-only.
- Item 6 golden trim/focus: the config carries the production template's
  trim/focus shape; with nepochs 40 < hold_epochs 50 both hold their
  start value all run (no anneal), deterministic and identical on both
  commits. ASSUMPTION: 46ec5e1 reads the same trim/focus keys (they
  predate EMA); if 46ec5e1's schema differed the golden leg's first raw
  log shows it.

### BLOCKING FINDING beyond the six items (Architect decision needed)

Tracing item 5 surfaced a board-wide latent failure the run-1 triage
could not see. build_run_specs (training.py:441-446) hard-accesses
train_args["optimizer"], ["lr"], ["scheduler"], ["trim"], ["focus"]
with NO default; from_config does no base-template merge (experiment.py
:1095 default_train_args only collapses ranges; __init__:985-986 just
assigns). ALL 19 smoke YAMLs under gates/configs are missing at least
`optimizer` (most also lack lr/scheduler/trim/focus) -> build_run_specs
raises KeyError('optimizer') for every driver-run smoke gate. In run-1
these gates failed EARLIER at --yaml path resolution (items 1+2), so the
missing-blocks failure was masked; once items 1+2 land, run-2 will
KeyError ~15 smoke gates at build_run_specs (proven on the Mac: ema-
smoke, berhu-loss, npce-residual, head-activation-pin-license all raise
KeyError 'optimizer'; my complete golden YAML passes). The error-path
gates (npce excl-*, head-activation-pin-license) would also fail their
message assertions, since the KeyError fires before the intended
validation error. This is a SIXTH root cause, outside the enumerated six
items and a genuine design fork, so I did NOT act on it:
  (a) complete every smoke YAML with the required blocks (aligns with
      the never-trust-defaults philosophy: explicit config, no code
      default) -- ~19 YAMLs, mechanical, benign off values; OR
  (b) make build_run_specs tolerate absent optional blocks via
      run_emulator's defaults (smaller, but leans on code defaults, and
      run_emulator lacks defaults for every one of the five blocks, so
      this is not a one-liner and touches production emulator code).
RECOMMEND (a). I can execute either in the immediate follow-up on the
Architect's ruling. Until then run-2 is still a diagnostic pass for the
smoke tier.

## Implementer smoke-YAML completion (2026-07-08, Opus, base d5694a5)

Architect ruled fork option (a) (option (b) rejected: build_run_specs's
hard requirement is the never-trust-defaults rule, production gains no
fallbacks). Base d5694a5 (a user commit removing a duplicate commands/
dir; the six-fix tree still uncommitted on top, verified intact). Every
gates/configs/*.yaml now carries the full required train_args set
(optimizer / lr / scheduler / trim / focus), explicit and template-
shaped (ema-off-identity-golden.yaml is the 5/5 example), inserted before
each model: block with a one-line "baseline blocks build_run_specs
requires" comment. Warmup sized by nepochs: 20 -> 3, 40 -> 5, 60 -> 8.
Only the MISSING blocks were added per file: the two ema smokes kept
their lr + ema (added optimizer/scheduler/trim/focus); head-scheduler-
override kept its run-default scheduler patience 25 + the head: patience
10 override (added optimizer/lr/trim/focus); everything else got all
five. Every gate-specific knob and banner input is preserved: the ema
horizons/anneals, the berhu_capped head, freeze_trunk true/false on the
joint pair, and the three DELIBERATELY INVALID configs stay invalid for
their one declared reason (head-activation-pin-license: freeze_trunk
false + a trf pin; npce-training-excl-ia: pce + model.ia; npce-training-
excl-rescale: pce run with the gate's --rescale=residual) -- those errors
fire in from_config / build_specs, before build_run_specs, so completing
the blocks does not mask them.

Mac gates (all green): ruby-parse all 20 configs; the AST-extracted REAL
build_run_specs + anneal_value run cleanly on EVERY config's train_args
(20/20, six spec dicts, trim/focus/kappa read, no KeyError anywhere, not
just the three variants); full --dry-run unchanged (exit 0, 24 absolute
--yaml, zero UNSET, --diagnostic=gates_diag); zero inline flow style; no
NEW line > 90 cols (the four >90 lines are the pre-existing spec-code
header banners on line 1 of four files, untouched). This resolves the
blocking finding above: run-2's smoke tier no longer KeyErrors at
build_run_specs. ONE commit now carries the six run-1 fixes + this
completion.

## Implementer board run-2 fixes (2026-07-08, Opus, base 17f08ce)

Six items from the Architect's run-2 log triage. Base 17f08ce (the run-1
fixes + smoke completion, user-committed; tree clean).

1. emulator/results.py: rebuild_emulator gained its own lazy `import
   h5py` (beside `import importlib`, same rationale comment as
   save_emulator's :213-215). Run-2 NameError at results.py:440 (h5py
   used at :401/:403/:440, never imported). Grep confirms save_emulator
   (:215) is the only other h5py user and it already imports; no other
   function reaches h5py without a local import.
2. gates/board.py _golden_leg + a new RunContext.staged_golden
   (run_board.py): the pinned worktree (46ec5e1) predates the absolute-
   path passthrough, so an absolute --yaml is re-prefixed there and dies.
   staged_golden copies the resolved golden config into
   <rootdir>/<driver_root>/<driver_fileroot>/gates-golden-<gate>.yaml
   (mkdir -p, removed in a finally) and _golden_leg passes the BARE name
   to BOTH legs, so the fileroot convention resolves it on every commit,
   old or new. ASSUMES the deploy rootdir == $ROOTDIR (declared).
3. gates/configs/ema-smoke-config.yaml: scheduler patience 25 -> 6 (the
   lr never cut in 60 epochs, so no rewind fired; a short patience
   provokes a plateau cut so 'rewound to best epoch' can appear).
4. gates/configs/berhu-anneal-config.yaml: single-phase -> TWO-PHASE
   (trunk loss sqrt; trunk_epochs 5; head loss berhu_capped knot 0.2
   cap 10 WITH the hold-5 ramp-10 cosine anneal; model resmlp ->
   rescnn/nla, the head-capable twin of berhu-loss-config). The anneal
   banner 'anneal: hold 5 + 10 cosine' is a HEAD-PHASE line (a single-
   phase run prints only the loss dict); the gate's banner assertion is
   unchanged. Head phase = 15 epochs (nepochs 20 - trunk 5), so the
   anneal completes (s=1 by head epoch 15).
5. gates/configs/param-window-cuts-config.yaml: n_train 25000 -> 5000,
   n_val 5000 -> 1000 (the 0.14-0.15 omegamh2 window keeps ~6.7k of
   100k, and the pool-too-small guard correctly refused n_train 25k).
6. gates/board.py npce sweep assertion: new acceptance = rc == 0 AND
   >=2 lines matching N_train\s+\d+\s+f\(>0\.2\) AND >=1 "PCE fit:"
   line. DECLARED DEVIATION: the per-point fit reports print in the GPU
   workers' stdout, so the parent stream carries only >=1 PCE-fit line;
   the two parent N_train f(>0.2) lines prove both points ran, and the
   per-point refit is structural to the top-level pce design (one base
   per point). rc failing on ntrain_sweep.txt writing into the
   placeholder fileroot is resolved by the user's real driver_fileroot.

Mac gates (all green): py_compile (results.py, board.py, run_board.py);
ruby-parse the 3 changed YAMLs; the AST-extracted real build_run_specs +
anneal_value still pass on ALL 20 configs (the restructured two-phase
berhu-anneal included); full --dry-run exit 0 with the golden STAGING
plan (stage -> current-tree run -> worktree run -> remove) and the BARE
gates-golden-ema-off-identity.yaml on BOTH golden legs, zero UNSET;
staged_golden unit-tested in isolation (stages into the fileroot, removed
after the block AND on an exception); grep confirms rebuild_emulator now
imports h5py; no >90-col line and no prose double-dash in the touched
files. ONE commit carries all six run-2 fixes.

### 2026-07-08 — Architect: board run-3 audit (HEAD 38f4d68)
15 PASS / 4 FAIL / triangle-shading not run (optional eyeball). Newly
proven: save-rebuild-drift end-to-end on GPU (plain + factored + npce
bitwise 0.0, drift proof, v1 refusal); cobaya-adapter parity rtol 1e-6
(plain 3.2e-7, factored 7.9e-7); berhu-anneal two-phase banner; ema-anneal
banner; param-window-cuts 0.14-0.15 window (kept 7563/100000, used 5000 of
6774); npce residual + ratio + both exclusivity errors.
The four failures, root-caused (no physics defect among them):
1. ema-off-identity — all 40 epoch lines character-identical between
   46ec5e1 and tip EXCEPT the trailing wall-clock column (1.9s vs 2.3s);
   identity is proven by this run's own data; the comparator must strip
   the timing field (fix item 2).
2. ema-smoke — the lr cut fired twice (epochs 36, 51) but rewind is an
   opt-in train_args key (default false) the smoke config never set; the
   feature under test was off (item 3).
3. npce-training — the sweep parent carries the pce banner and both
   N_train result lines but zero "PCE fit:" reports (workers own them);
   the assertion read the wrong stream (item 4).
4. cobaya-adapter — parity PASS; the evaluate leg died inside cobaya-run
   on the shipped example YAML's stale theory.path (retired emultrfv2
   deploy) and literal <run> placeholder. The board now owns a bespoke
   evaluate YAML reading the tiny emulator save-rebuild-drift persists
   (item 5). The classy NOTICE in the log is euclidemu2 import-time
   noise, not the failure.
Science margin note (recurring): the EMA average's val minimum lands
early again (ema-anneal best epoch 7 of 20) with val rising after — the
bs+EMA thread's first data point stands.
New standing rule: terminal output essential-only
([[terminal-output-essential-only]]); run_board.py quiet by default with
a debug key (item 1).

## Implementer board run-3 fixes (2026-07-08, Opus, base 38f4d68)

Six items from the Architect's run-3 triage (15/19 PASS; four failures,
no physics defect) plus the new terminal-output-essential-only rule. Base
38f4d68 (run-2 fixes, user-committed; tree clean; worktree at that tip).

1. run_board.py quiet mode + the new rule. RunContext._emit gains a
   log_only route (self.debug or not log_only -> terminal; the log always
   gets everything); sh's command echo + streamed output and the
   _log_header block + config dump are log_only; the runner prints a
   one-line terminal header "GATE <id> [<tier>] started <hh:mm:ss>" and
   the footer verdict via ctx._emit (terminal). debug threads from a new
   required board_config.json "debug": false key (preflight (e) fails
   loudly if it is missing or non-bool) or a --debug flag forcing true.
   Log bytes are unchanged (the same header/dump/stream still reach the
   log fh). board_config.json gained "debug": false and its _help entry.
2. logscan.byte_identity gains strip=None: after line selection,
   re.sub(strip, "", line) on both sides before comparison; the
   divergence detail shows the stripped lines. _golden_leg passes
   strip=r"[ \t]+\d+(?:\.\d+)?s$" for EVERY golden leg (the trailing
   wall-clock column is the one machine-noise field). gate_gm_c docstring
   notes the wall-clock strip. matching_lines unchanged.
3. ema-smoke-config.yaml: rewind: true joins ema (opt-in, default false);
   with scheduler patience 6 (run 2) the lr cut fires and the rewind line
   the gate asserts appears.
4. gate_gpc_c sweep assertion: rc_s == 0 AND >=2 "N_train N f(>0.2)"
   parent lines AND the parent staging banner logscan.search(r"^pce:
   form"); the "PCE fit:" requirement is dropped (run 3 proved the parent
   stream carries zero such lines; the GPU workers own the per-point
   reports). Declared deviation kept.
5. cobaya-adapter evaluate leg board-owned. (a) gsv_bitwise_drift.py: the
   plain case gets a SECOND persistent save_emulator to
   <driver_root>/chains/gates_emul_evaluate (same bytes as the tmp
   round-trip, which is untouched; via a save_kwargs dict + a
   persist_root param on train_save/run_variant) and prints the path.
   (b) NEW gates/configs/cobaya-adapter-evaluate.yaml: a copy of
   EXAMPLE_EMUL_EVALUATE.yaml with a gate header, theory.path ->
   emulators_code_v2, extra_args.emulators -> gates_emul_evaluate, output
   -> gates_cobaya_adapter_evaluate; board_config.json evaluate_yaml now
   defaults to it. (c) gate_gct_c expects
   <rootdir>/<driver_root>/chains/gates_emul_evaluate.h5 before cobaya-run
   (label "evaluate emulator present (saved by save-rebuild-drift)"; a
   failed expect raises and skips the run), and the log message is
   corrected (the evaluate leg proves the run completes; parity is
   gct_parity's job). The cobaya-adapter Gate already carried
   deps=("save-rebuild-drift",). (d) EXAMPLE_EMUL_EVALUATE.yaml: both
   emulators/emultrfv2 occurrences -> emulators_code_v2 (the <run>
   placeholder kept).
6. NEW notes/terminal-output-essential-only.md + its MEMORY.md index line
   + a CLAUDE.md Conventions pointer; this audit block appended above.

Mac gates (all green): py_compile run_board.py / board.py / logscan.py /
gsv_bitwise_drift.py; yaml.safe_load (ruby) the changed + new configs and
json.load board_config.json; a byte_identity unit probe (two lines
differing only in a trailing wall-clock -> equal with the strip, differing
in a val digit -> unequal, detail shows the stripped lines); a quiet-mode
probe (a stubbed sh command lands in the log fh, NOT stdout, when debug
false; --debug/debug true mirrors it; log_only header suppressed on the
terminal); staged_golden persistent-save path checked; --dry-run plans 18
gates and --dry-run --gate ema-off-identity shows the staging lines +
bare gates-golden --yaml. Do not commit; the command is printed.

### 2026-07-08 — Architect: board runs 4+5 audit (HEAD 9ca0cda)
Run 4 (resume, quiet terminal working as designed): ema-off-identity PASS
(byte-identity with the wall-clock strip — GM-C closed), ema-smoke PASS
(rewind fired at BOTH plateau cuts, epochs 36 and 51, metrics jumping
with the raw ones — GM-D closed), npce-training PASS (banner evidence).
cobaya-adapter failed its own new presence check: resume had skipped
save-rebuild-drift (PASS from run 3, before the persist existed), so
gates_emul_evaluate was never written — the dependency edge guarantees
PASS status, not artifact existence. One-time divergence; remedy was
--force-rerun save-rebuild-drift.
Run 5: gsv persisted gates_emul_evaluate, presence check PASS, and
cobaya-run resolved a theory class for the first time — the LEGACY v1
adapter bundled in cocoa's cobaya fork, because both evaluate YAMLs used
`path:` where cobaya's external-class key is `python_path`
([[cobaya-theory-adapter]] 2026-07-08 addendum). Architect applied the
two-key YAML fix + this record directly (declared role deviation: no
Python touched; a relay round-trip for a fully-determined two-line edit
was not worth the cost). Board stands 18 PASS / 1 FAIL (cobaya-adapter,
evaluate leg only — its parity probe passes at rtol 1e-6) /
triangle-shading optional.

### 2026-07-08 — Architect: board run-6 audit (HEAD 485a7c6)
The python_path fix advanced the evaluate leg past class lookup into
cobaya's output bookkeeping, which found the info files run 5's failed
attempt left under chains/gates_cobaya_adapter_evaluate and refused to
resume onto a changed theory block ("Old and new run information not
compatible"). A smoke leg reruns with a fixed output prefix by design,
so the board's evaluate YAML now sets force: True (cobaya overwrites its
own products). Architect applied the one-key edit directly (same
declared deviation as run 5's fix). Board unchanged otherwise: 18 PASS /
cobaya-adapter evaluate pending / triangle-shading optional.

### 2026-07-08 — Architect: board run-7 audit (HEAD 487632b)
force: True worked (cobaya deleted run 5/6's stale products) and
python_path held: our schema-v2 adapter class loaded, initialized, and
declared its requirements for the first time. The evaluate leg then
died one layer deeper, in Model dependency resolution: "Requirement
As_1e9 of emul_cosmic_shear is not provided by any component, nor
sampled directly". Root cause: the params block both evaluate YAMLs
inherited from the LEGACY lcdm example marks As_1e9 / omegab / omegam
`drop: true` and bridges them to logA / omegabh2 / omegach2 via lambdas
— but the v2 dumps' covmat header (verified on the Mac dev copy,
~/data/pytorch/dvs/w0wa_takahashi_params_train_cs_16.covmat) stores the
sampler-side names directly: As_1e9 ns H0 omegab omegam LSST_DZ_S1..S5
LSST_A1_1 LSST_A1_2. A dropped param never reaches a theory, so the
adapter (whose requirements are the h5's stored names, by design) was
starved of exactly the names the YAML was walking. The likelihood needs
none of the bridged names with use_emulator: 1 (probe xi requires only
the cosmic_shear product; _cosmolike_prototype_base.get_requirements),
and the DZ/A1 names come from the likelihood's params_source.yaml
defaults, as they did for the legacy v1 adapter. Fix (Architect-applied,
same declared deviation): remove the three drop: true keys and the dead
bridge (mnu + the three lambdas) from both
gates/configs/cobaya-adapter-evaluate.yaml and
cobaya_theory/EXAMPLE_EMUL_EVALUATE.yaml; the example's comment now
states the general rule (bridge lambdas ONLY when the h5's stored names
are a reparametrization of what the sampler walks). Cobaya requirement
names untested past As_1e9 (resolution fails on the first missing name,
and As_1e9 is column 1) — but the full list is now pinned by the covmat
header, not inferred. Board unchanged otherwise: 18 PASS /
cobaya-adapter evaluate pending / triangle-shading optional. Layers
peeled so far on this leg: stale example paths (run 3/4) -> artifact
existence (run 4) -> class shadowing (run 5) -> stale output info
(run 6) -> legacy params bridge (run 7).

### 2026-07-08 — Architect: board run-8 audit (HEAD 0372d7d)
The params fix landed: dependency resolution cleared, the evaluate
sampler assembled its reference point (our 12 stored names + the
likelihood's LSST_M1..M5 defaults), and OUR ADAPTER EXECUTED FOR THE
FIRST TIME — "[emul_cosmic_shear] Average evaluation time: 0.109418 s
(1 evaluations)": EmulatorPredictor loaded the persisted tiny emulator,
encoded, forwarded, decoded, and delivered a data vector inside cobaya.
The failure moved into the likelihood's consumption of that vector:
"ValueError: Incompatible Sizes (Emulator Cosmic Shear)"
(_cosmolike_prototype_base.internal_get_datavector_emulator:479). Root
cause is the dv-shape decision point declared at design time
([[cobaya-theory-adapter]]): predict() returns the KEPT entries
(n_keep, the unmasked positions), while the likelihood demands
len(get_cosmic_shear()) == sizes[0] — the FULL cosmic-shear section as
cosmolike sizes it (the legacy v1 adapter emitted exactly that;
OUTPUT_DIM 780 in the legacy lsst_y1 example). The declaration "kept
entries == legacy res[0]" is now falsified by a real run: with the
M1_GGL0.05 mask, n_keep != sizes[0]. Resolution (USER's design,
overriding the Architect's first proposal; dated addendum in
[[cobaya-theory-adapter]]): the likelihood stays UNTOUCHED — it is the
gluer of per-probe section-sized products, and separate cs/ggl/wtheta
emulators are the intended future. predict() gains a shape flag
dv_return: 'section' | '3x2pt', DEFAULT 'section' (a cosmic-shear
emulator returns exactly the xi± block, full[0:sizes[0]], via decode ->
geometry.unsqueeze scatter -> block slice); the section boundaries
become part of the artifact (state()/from_state gain section_sizes +
probe, both ALREADY resolved from cosmolike inside from_cosmolike at
staging and today dropped by state() — persist-resolved-values applied;
old v2 files load them as None and section mode errors loudly naming
the re-save). The training-side data vector stays full-3x2pt-length,
unchanged. Emulator-side unit handed to the Implementer (geometry state
+ inference flag + adapter whitelist + parity shape assertions); no
cocoa-side change. Board: 18 PASS / cobaya-adapter evaluate pending /
triangle-shading optional. Layer 6 peeled: params bridge (run 7) ->
dv-shape contract (run 8) — the first failure inside our own
parity-proven code, and it is a contract mismatch, not physics.

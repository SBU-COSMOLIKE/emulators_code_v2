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

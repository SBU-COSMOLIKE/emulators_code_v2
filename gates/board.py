"""The workstation board: the list of tests and what each one is.

This file holds BOARD, a plain Python list of the 19 tests in the order
they run, and a small class, Gate, that says what one test is: its name,
its tier, its home note, and the function that runs it. Each Gate also
carries a ``maps`` string pointing at the home-note lines its checks
come from, so a reviewer can see the test follows the note and not a
memory of it. The tiers group the list (the EMA off-mode identity test
first, then the rest of the backlog, then this cycle's new-feature
tests, then the save-and-sample chain). run_board.py imports this list
and runs the tests; nothing in this file runs on its own.

Every test's run function has one shape. It issues its shell commands
through ``ctx.sh`` / ``ctx.run_driver`` (which stream output to the
test's raw log), returns early on ``ctx.dry`` (so --dry-run prints the
plan and stops before any check runs), then judges pass/fail with
``ctx.expect`` and the pure helpers in ``checks.logscan``. Numeric
acceptances (the weight-decay census, the eval-batch invariance, the
bitwise save/rebuild equality, the training-to-inference parity) live
in the executable ``checks/`` scripts a test launches, so the harness
itself computes them and the raw log holds every value.

Golden runs build the same config in a throwaway worktree, not a checkout in
place: the pinned pre-feature build runs in a throwaway ``git worktree``
the runner removes even on failure. Only the EMA identity test has a
preset base (the pre-EMA commit); the other golden runs read their base
from board_config.json and, when it is unset, skip that leg with a
logged note and keep the functional smoke leg as the acceptance (a
merged feature's off path is already exercised by the standard runs).

Glossary:
  board     = the ordered list of tests the harness drives.
  gate      = one test: a home note, the commands it runs, and how its
              pass/fail is decided (the Gate class; "test" in prose).
  tier      = the board's coarse grouping and the --tier selector value
              (backlog / new-features / save-and-sample).
  golden run= a byte-identity run: the same config built on the current
              tree and on a pinned pre-feature commit, whose selected
              log lines must match to the character.
  smoke     = a short training run judged on the banner lines it prints
              rather than on a numeric tolerance.
  banner    = a driver's human-readable startup or per-epoch status line
              that a test asserts on.
  worktree  = a throwaway git checkout of another commit that never
              disturbs the user's working tree.
  preflight = run_board.py's pre-GPU checks (git tip, clean tree, cocoa
              imports, data paths) that must pass before any test runs.
  resume    = re-running the board skips tests already marked PASS, so
              a crash mid-board loses only the in-flight test.
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple

from checks import logscan


class GateFailure(Exception):
  """Raised inside a gate when an acceptance check fails.

  The runner catches it, writes the gate FAIL with the message in the
  raw log and board_status.json, and moves on to the next gate (a
  single gate's failure never stops the board). Gate functions raise
  it directly for an unrecoverable precondition (a missing config), and
  ``ctx.expect`` raises it for a failed acceptance value.
  """


# The three tiers, in board order. The strings are the tier selector
# values --tier accepts and the labels the BOARD.md table prints.
TIER_BACKLOG = "backlog"
TIER_NEW_FEATURES = "new-features"
TIER_SAVE_AND_SAMPLE = "save-and-sample"

# The sweep-over-n_train driver npce-training's 2-point smoke runs; the
# default single-train driver would execute the wrong program on a sweep
# YAML. It sits beside the emulator package at the repo root.
SWEEP_NTRAIN_DRIVER = "sweep_ntrain_emulator_cosmic_shear.py"


@dataclass(frozen=True)
class Gate:
  """One row of the board: one test, what it runs, and how it is judged.

  Arguments:
    id      = the test name (e.g. "save-rebuild-drift"); the log filename
              stem, the selector --gate accepts, and the resume key.
    tier    = one of TIER_BACKLOG / TIER_NEW_FEATURES / TIER_SAVE_AND_SAMPLE.
    home    = the home note filename stem (the note that defines it); printed
              in the log header so a log traces back to its spec.
    maps    = the home-note line(s) each check implements (assertion ->
              note line), printed in the header so a review confirms the
              test encodes the note and not a memory of it.
    run     = the test body, run(ctx) -> None; issues commands and judges
              pass/fail, raising GateFailure on any failure.
    deps    = the tests whose PASS this test needs; an unmet dependency
              marks SKIPPED(dependency) rather than running.
    optional= when True the test is skipped unless --gate names it
              (triangle-shading, registered but off the default sweep).
    needs   = the environment capabilities the test requires ("torch",
              "cosmolike", "cobaya", "gpu"); documentation for the header
              and a clearer skip message.
    worktree_commit = a commit the test pins a temporary worktree at (the
              EMA identity test's pre-EMA build); None for tests that
              never leave the current tree.
    spec_code = the internal short code that keys this test's audit
              history in its home note; printed once in the log header.
    title   = a one-line human name for the test (for the README table).
  """
  id: str
  tier: str
  home: str
  maps: str
  run: Callable
  deps: Tuple[str, ...] = ()
  optional: bool = False
  needs: Tuple[str, ...] = ()
  worktree_commit: str = None
  spec_code: str = ""
  title: str = ""


# --------------------------------------------------------------------------
# Shared gate bodies. The golden byte-identity leg and the plain driver
# smoke are common enough to factor; the per-gate functions below call
# them with the gate's own config keys and acceptance substrings.
# --------------------------------------------------------------------------

def _golden_leg(ctx, gate_id, yaml_name, grep_pattern):
  """Run the byte-identity golden leg for a gate, or skip it loudly.

  Builds the same config on the current tree and on the gate's pinned
  pre-feature commit (a temporary worktree), then requires the
  grep-selected lines identical. When no base commit is configured the
  leg is a no-op on the committed tip, so it is skipped with a logged
  explanation rather than run hollow.

  Arguments:
    ctx          = the per-test helper (its sh / worktree / expect / dry).
    gate_id      = the gate whose golden_bases entry names the base.
    yaml_name    = the shared config both builds train on.
    grep_pattern = the grep-style regex selecting the lines to compare
                   (e.g. "^(phase|epoch|best)").

  Returns:
    None. Raises GateFailure (via ctx.expect) if the lines diverge.
  """
  base = ctx.golden_base(gate_id)
  if base is None:
    ctx.log("golden byte-identity leg: no base commit configured in "
            "board_config.json golden_bases['" + gate_id + "']. On the "
            "merged tip a git-stash diff is a no-op, so this dev-time "
            "pre-commit leg is skipped; the functional smoke leg is the "
            "acceptance (harness handoff decision point).")
    return

  yaml_path = ctx.config_yaml_name(yaml_name)
  ctx.log("golden byte-identity: current tree vs pinned " + base)
  _, cur = ctx.run_driver(yaml_path=yaml_path)
  with ctx.worktree(commit=base) as wt:
    _, pre = ctx.run_driver(yaml_path=yaml_path, cwd=wt)

  if ctx.dry:
    return

  equal, detail = logscan.byte_identity(text_a=pre,
                                        text_b=cur,
                                        pattern=grep_pattern)
  ctx.expect(label=gate_id + " golden byte-identity (" + base + " vs tip)",
             ok=equal,
             detail=detail)


def _smoke_driver(ctx, config_key, required_banners, *, extra=()):
  """Run one training smoke from a gate config and assert its banners.

  The common workstation shape: run the driver on a gate-specific YAML,
  require the run to finish, and require every home-note banner
  substring present in the output.

  Arguments:
    ctx              = the per-test helper.
    config_key       = the board_config.json gate_configs key naming the
                       smoke YAML; a missing/unset config raises
                       GateFailure (the gate cannot run without it).
    required_banners = the literal banner substrings that must all
                       appear (quoted verbatim from the home note).
    extra            = extra driver flags (e.g. ("--activation=power",)).

  Returns:
    the captured run output (so a caller can make further assertions).
  """
  yaml_path = ctx.require_config(config_key)
  rc, out = ctx.run_driver(yaml_path=yaml_path, extra=extra, allow_fail=True)

  if ctx.dry:
    return out

  ctx.expect(label=config_key + " run completes (rc 0)",
             ok=(rc == 0),
             detail="driver exit code " + str(rc))
  ok, missing = logscan.contains_all(text=out, needles=required_banners)
  ctx.expect(label=config_key + " banners present",
             ok=ok,
             detail="missing: " + repr(missing))
  return out


# --------------------------------------------------------------------------
# Standing tier.
# --------------------------------------------------------------------------

def gate_gm_c(ctx):
  """ema-off-identity: the EMA feature must leave existing runs untouched.

  WHAT: weight EMA (the train_args.ema block), run here with that block
  ABSENT so the feature is off. WHY: a feature that is off must not
  perturb a single existing run, else every earlier result is in doubt;
  byte identity is the strongest form of that guarantee. HOW: build the
  same config on the current tree and on the pinned pre-EMA commit (in a
  throwaway worktree, never a checkout in place) and require the epoch
  and best-epoch lines identical to the character (home note
  weight-ema-snapshot-coupled.md:98-101, the byte-identity gate; 229-238,
  the epoch-line diff recipe).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: weight-ema-snapshot-coupled.md:98-101 (byte-identity gate),
  # :229-238 (the diff <(grep '^(epoch|best epoch)') recipe).
  _golden_leg(ctx=ctx,
              gate_id="ema-off-identity",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(epoch|best epoch)")


def gate_gm_d(ctx):
  """ema-smoke: the EMA feature, switched on, behaves as designed.

  WHAT: a short bs=64 run with ema.horizon_epochs=3. WHY: byte identity
  (the ema-off-identity test) proves EMA off is harmless, not that EMA
  on works; this run exercises the live averaging path. HOW: the startup
  banner must name the horizon ("ema: horizon 3 epochs") and a plateau
  lr cut must print a "rewound to best epoch" line, since the average
  follows the rewind (home note weight-ema-snapshot-coupled.md:240-251,
  the on-mode recipe; 246-249, the rewind line).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: weight-ema-snapshot-coupled.md:104-107, :240-251 (ema-smoke recipe:
  # the "ema: horizon 3 epochs" banner; metrics track then smooth).
  out = _smoke_driver(ctx=ctx,
                      config_key="ema-smoke-config",
                      required_banners=["ema: horizon 3 epochs"])
  if ctx.dry:
    return
  # the note's rewind line is mechanically assertable
  # (weight-ema-snapshot-coupled.md:246-249: a plateau lr cut rewinds
  # to the best epoch and the ema metrics jump WITH the raw ones).
  ctx.expect(label="ema-smoke rewind line ('lr cut -> rewound to best epoch')",
             ok=logscan.search(text=out, pattern=r"rewound to best epoch"),
             detail="weight-ema-snapshot-coupled.md:246-249: a rewind fires")


def gate_diag(ctx):
  """production-diagnostic: one --diagnostic run that closes five checks.

  WHAT: a single production training run with the diagnostics PDF, which
  exercises the dead-class census, a tight density-window cut, a nested
  param_cuts block, the absolute row counts, and the shaded diagnostics
  triangle all at once. WHY: these five otherwise-separate checks all
  ride one ordinary run, so one run proves the whole diagnostic path
  instead of five. HOW: the package imports with no dead classes, the
  run finishes, the sizes line reports "used N of P cut rows" with N the
  configured n_train, and the regenerated PDF shades every hard sample
  edge (a visual confirmation from the committed PDF). Home notes: the
  five home notes listed in this test's maps.
  """
  ctx.require_caps("cosmolike")
  # G1 home: audit-package-style-2026-07-05.md:232-234.
  ctx.log("G1: dead-class census (NLATemplateMLP / NLAInputGeometry) "
          "+ clean package import.")
  rc_grep, out_grep = ctx.sh(
    cmd=["grep", "-rn", "NLATemplateMLP\\|NLAInputGeometry",
         "--include=*.py", ".", "README.md"],
    allow_fail=True)
  # DEVIATION from the note's line 233 import: emulator.parallel was
  # deleted after the audit (commit 29b23dd), so it is dropped here; the
  # live subpackages are emulator / emulator.IA / emulator.PCE.
  # the harness's own interpreter, never a bare "python" on PATH.
  rc_imp, out_imp = ctx.sh(
    cmd=[ctx.python, "-c", "import emulator, emulator.IA, emulator.PCE"],
    allow_fail=True)

  diag_yaml = ctx.require_config("production-diagnostic-config")
  ctx.log("production-diagnostic production run: tight omegamh2 window + nested "
          "param_cuts + absolute n_train/n_val, with the diagnostics PDF.")
  rc_run, out_run = ctx.run_driver(yaml_path=diag_yaml,
                                   extra=("--diagnostic",),
                                   allow_fail=True)

  if ctx.dry:
    return

  # G1 (grep 0 hits: grep exits 1 when it finds nothing).
  ctx.expect(label="G1 dead-class census -> 0 hits",
             ok=(rc_grep == 1 and out_grep.strip() == ""),
             detail="grep rc " + str(rc_grep) + ", output: "
                    + repr(out_grep.strip()[:200]))
  ctx.expect(label="G1 clean package import",
             ok=(rc_imp == 0),
             detail="import rc " + str(rc_imp))
  # G-F / GN-F / GS-D banners (home lines named in `maps`).
  ctx.expect(label="production-diagnostic production run completes",
             ok=(rc_run == 0),
             detail="driver exit code " + str(rc_run))
  ctx.expect(label="GS-D sizes line ('used N of P cut rows')",
             ok=logscan.search(text=out_run,
                               pattern=r"used\s+\d+\s+of\s+\d+\s+cut rows"),
             detail="the sizes line must report used N of P cut rows")
  ctx.log("GT-C: the diagnostics PDF is a VISUAL check (the omh2 marginal "
          "at 0.20 and the (ns, omh2) diagonal corner at 0.17 must show "
          "adjoining grey); the harness confirms the run produced it, the "
          "Architect confirms the shading from the committed PDF/log.")


def gate_gp_d(ctx):
  """single-phase-demotion: a single-phase model accepts two-phase keys.

  WHAT: the phase-argument resolver that lets a single-phase model
  (resmlp) accept a config written with two-phase (trunk/head) keys.
  WHY: that exact config used to crash the run; the resolver must demote
  it cleanly, yet stay a no-op for a model that genuinely has two phases
  (or it would corrupt those). HOW: the resmlp config now trains with no
  traceback and the banner prints the demotion notice, while the same
  config on a two-phase model (rescnn + nla) reproduces today's behavior
  unchanged (home note resolve-phase-args-single-phase.md:110-113, the
  demotion + no-op control).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: resolve-phase-args-single-phase.md:110-113.
  single_yaml = ctx.require_config("single-phase-demotion-single")
  control_yaml = ctx.require_config("single-phase-demotion-control")
  rc_s, out_s = ctx.run_driver(yaml_path=single_yaml, allow_fail=True)
  rc_c, out_c = ctx.run_driver(yaml_path=control_yaml, allow_fail=True)

  if ctx.dry:
    return

  ctx.expect(label="single-phase-demotion single-phase resmlp trains (was a traceback)",
             ok=(rc_s == 0),
             detail="resmlp run exit code " + str(rc_s))
  ctx.expect(label="single-phase-demotion demotion notice in the banner",
             ok=logscan.search(text=out_s,
                               pattern=r"(single-phase|demot|resolve)"),
             detail="EXACT notice string to confirm against "
                    "resolve-phase-args-single-phase.md:111")
  ctx.expect(label="single-phase-demotion control rescnn+nla reproduces today (no-op)",
             ok=(rc_c == 0),
             detail="control run exit code " + str(rc_c))


def gate_gh_e(ctx):
  """head-scheduler-override: the head phase cuts the lr on its own patience.

  WHAT: a per-phase scheduler override, so phase 2 (the frozen-trunk
  head) can cut its learning rate on a patience of its own. WHY: a head
  phase often needs a shorter patience than the run default, and the
  override must take effect for that phase only. HOW: the banner shows
  "[head overrides: scheduler]" and the head phase's first lr cut lands
  on the patience-10 cadence (vs the run's 25); plus a golden
  no-phase-blocks run for byte identity (home note
  phase-blocks-nested-lr-scheduler.md:262-267, the override + cadence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: phase-blocks-nested-lr-scheduler.md:262-267 (override), :269-279
  # (golden diff <(grep '^(phase|epoch|best)')).
  _golden_leg(ctx=ctx,
              gate_id="head-scheduler-override",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  out = _smoke_driver(ctx=ctx,
                      config_key="head-scheduler-override-config",
                      required_banners=["[head overrides: scheduler]"])
  if ctx.dry:
    return
  ctx.log("head-scheduler-override cadence: the head phase's first lr cut should land on the "
          "patience-10 cadence (vs 25); confirm from the lr-cut epoch "
          "spacing in the log (phase-blocks-nested-lr-scheduler.md:265).")


def gate_ge_c(ctx):
  """eval-batch-invariance: the validation metrics do not depend on chunking.

  WHAT: the decoupled validation batch, which lets the eval pass use a
  large batch no matter the training batch. WHY: the reported validation
  numbers must not shift with how the eval set is chunked, or a
  batch-size change would silently move every metric. HOW: a torch-only
  check script confirms the per-row chi2 from eval_val agrees across eval
  batch sizes to rtol 1e-6 (it prints "Part 1: PASS") and that the
  derived eval batch cuts the eval time on CUDA (home note
  eval-bs-decoupling.md:102-108, the invariance + timing).
  """
  ctx.require_caps("torch", "gpu")
  # home: eval-bs-decoupling.md:102-108 (acceptance), :202-300 (the
  # ready-to-paste script this check mirrors).
  rc, out = ctx.run_check("gates/checks/ge_c_eval_bs.py")
  if ctx.dry:
    return
  ctx.expect(label="eval-batch-invariance Part 1 partition invariance (rtol 1e-6)",
             ok=logscan.contains(text=out, needle="Part 1: PASS"),
             detail="the script must print 'Part 1: PASS'")
  ctx.expect(label="eval-batch-invariance check script exit 0",
             ok=(rc == 0),
             detail="check exit code " + str(rc))


def gate_gb_c(ctx):
  """berhu-loss: the berHu head loss trains beside a plain-sqrt trunk.

  WHAT: the berHu loss family (a robust sqrt below a knot, capped above
  a cap) as a per-phase head loss under the nested loss schema. WHY: the
  head often wants a monster-robust loss while the trunk stays plain
  sqrt, so the two loss blocks must resolve independently per phase. HOW:
  the trunk banner reads "loss_mode sqrt" and the head banner
  "loss_mode berhu_capped (knot 0.2, cap 10)", the loss decreases, and a
  torch-only check confirms the berHu reduction numerics; plus a golden
  non-berhu run (home note loss-mode-berhu.md:148-153 numerics, 290-314
  the run).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # leg 1: the torch-only unbound _reduce numerics
  # (loss-mode-berhu.md:148-153): berhu == sqrt below the knot,
  # berhu_capped == berhu below the cap, manual references, autograd
  # continuity across BOTH knots, non-default knots. The check script
  # is part of the check-script remainder.
  rc, out = ctx.run_check("gates/checks/gb_c_berhu_reduce.py")
  if not ctx.dry:
    ctx.expect(
      label="berhu-loss leg 1 berhu/_reduce numerics + autograd continuity",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/gb_c_berhu_reduce.py)")
  # leg 2: the golden non-berhu byte-identity + the head-berhu run
  # (loss-mode-berhu.md:290-314).
  _golden_leg(ctx=ctx,
              gate_id="berhu-loss",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="berhu-loss-config",
                required_banners=["loss_mode sqrt",
                                  "loss_mode berhu_capped (knot 0.2, cap 10)"])


def gate_gl_d(ctx):
  """loss-schema-equivalence: the new loss schema changes config, not physics.

  WHAT: the nested loss: block that replaced the old flat loss keys.
  WHY: it is a config-layer rename only, so the same physics expressed
  in the new schema must reproduce the old run exactly, or the migration
  silently changed a number. HOW: a golden equivalence run: the same
  physical config in the new schema reproduces the pre-change run's epoch
  lines to the character (numerics untouched), reusing the head-berhu
  config as the production shape (home note loss-block-nesting.md:237-244,
  the schema equivalence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: loss-block-nesting.md:237-244.
  _golden_leg(ctx=ctx,
              gate_id="loss-schema-equivalence",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="berhu-loss-config",
                required_banners=["loss_mode berhu_capped (knot 0.2, cap 10)"])


def gate_gba_c(ctx):
  """berhu-anneal: the berHu shape ramps in smoothly and late.

  WHAT: the berHu anneal, which starts as plain sqrt and blends into the
  berHu shape over a hold-then-ramp schedule. WHY: the escalated tail
  votes should arrive late, after the trim schedule has absorbed the
  worst outliers, and the ramp must not jolt the loss at its start. HOW:
  with anneal on, the banner reads "(knot 0.2, cap 10; anneal: hold 5 +
  10 cosine)", the printed train loss is continuous at the hold boundary,
  and the shape is full berHu by epoch 15; plus the golden no-anneal run
  (home note berhu-anneal-schedule.md:199-221).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: berhu-anneal-schedule.md:199-221.
  _golden_leg(ctx=ctx,
              gate_id="berhu-anneal",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  out = _smoke_driver(ctx=ctx,
                      config_key="berhu-anneal-config",
                      required_banners=["anneal: hold 5 + 10 cosine"])
  if ctx.dry:
    return
  ctx.log("berhu-anneal schedule: confirm the train loss is continuous at the "
          "hold boundary (epoch 5->6) and s=1 (full berhu) by epoch 15; "
          "the first ~5 epochs match a plain sqrt run "
          "(berhu-anneal-schedule.md:213-221).")


def gate_gme_c(ctx):
  """ema-anneal: the EMA average wakes up only after the bad early era.

  WHAT: the EMA anneal, which grows the averaging window from zero over
  a hold-then-ramp schedule. WHY: averaging through the high-loss early
  epochs would poison the shipped weights, so the average must stay
  dormant until the model has settled. HOW: with anneal on, the banner
  names the horizon and the "anneal: hold 5 + 10 cosine" schedule, and
  the average's metrics first appear at the live point (epoch 6+, after
  the hold plus warmup); plus the golden no-anneal run (home note
  ema-anneal-schedule.md:180-197).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: ema-anneal-schedule.md:180-197.
  _golden_leg(ctx=ctx,
              gate_id="ema-anneal",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|ema)")
  _smoke_driver(ctx=ctx,
                config_key="ema-anneal-config",
                required_banners=["ema: horizon 3 epochs",
                                  "anneal: hold 5 + 10 cosine"])


def gate_item27(ctx):
  """param-window-cuts: a tight density window drops exactly the rows it says.

  WHAT: the physical density-window cut (here a tight omegamh2 window),
  plus the paired inspection of a duplicate cosmolike init_probes call.
  WHY: rarefied density corners fail training, so the cut must remove
  precisely the rows the banner reports, and the duplicate init_probes
  call needs workstation evidence to keep or drop. HOW: a short training
  with the tight window runs end to end and the pool shrinkage matches
  the "used N of P cut rows" banner (a nested param_cuts block shows the
  normal banner); the init_probes call is the paired A/B check (home
  notes omegamh2-ns-product-cuts.md:125-126, param-cuts-nested-block.md:94-95).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: omegamh2-ns-product-cuts.md:125-126, param-cuts-nested-block.md:94-95.
  window_yaml = ctx.require_config("param-window-cuts-config")
  rc, out = ctx.run_driver(yaml_path=window_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="param-window-cuts tight-window run completes",
             ok=(rc == 0),
             detail="run exit code " + str(rc))
  ctx.expect(label="param-window-cuts pool shrinkage banner ('used N of P cut rows')",
             ok=logscan.search(text=out,
                               pattern=r"used\s+\d+\s+of\s+\d+\s+cut rows"),
             detail="the banner cut count must match the pool shrinkage")
  ctx.log("param-window-cuts ci.init_probes A/B: the duplicate init_probes call in "
          "geometries_output.py is inspected with this run's evidence "
          "(omegamh2-ns-product-cuts.md:248); a manual A/B, not an "
          "automatable assertion.")


def gate_gt_b(ctx):
  """triangle-shading: the diagnostics triangle greys the right panels.

  WHAT: the cut-window shading on the diagnostics corner plot with all
  four density windows active. WHY: each window must grey exactly the
  panels the coverage table names, in one shared colour, or the plot
  misleads a reader about which region was cut. HOW: a synthetic-sample
  triangle fills exactly the coverage-table panels (asserted on each
  axis's artist list) in a single rgba, plus the omh2 marginal band; off
  the default sweep (runs only when --gate names it), matplotlib +
  getdist, no cosmolike (home note
  triangle-cut-shading-all-windows.md:72-75).
  """
  ctx.require_caps("torch")
  # home: triangle-cut-shading-all-windows.md:72-75.
  rc, out = ctx.run_check("gates/checks/gt_b_triangle.py")
  if ctx.dry:
    return
  ctx.expect(label="triangle-shading four-window triangle shading check exit 0",
             ok=(rc == 0),
             detail="check exit code " + str(rc)
                    + " (gates/checks/gt_b_triangle.py)")


# --------------------------------------------------------------------------
# This week's legs.
# --------------------------------------------------------------------------

def gate_gft_c(ctx):
  """joint-training: freeze_trunk false really trains the trunk in phase 2.

  WHAT: the freeze_trunk-false option, which fine-tunes trunk and head
  together in phase 2 instead of freezing the trunk. WHY: a joint
  fine-tune warm-started by phase 1 is a distinct training mode, and it
  must actually run the trunk backward, not silently stay frozen. HOW: a
  restrf + nla run with a small trunk_epochs and freeze_trunk false
  announces "two-phase: N trunk + M joint" and "phase 'joint'", the loss
  is continuous at the handoff, and its phase-2 epoch time sits visibly
  above a freeze_trunk-true control (the trunk backward returned); plus
  the golden absent-key run (home note freeze-trunk-joint-phase2.md:115-120).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: freeze-trunk-joint-phase2.md:115-120, :211-228.
  _golden_leg(ctx=ctx,
              gate_id="joint-training",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|run:)")
  # the joint run (the trunk-count banner is matched by regex, so
  # the YAML uses a small trunk_epochs per the note, not a pinned 800).
  joint_yaml = ctx.require_config("joint-training-config")
  rc_j, out = ctx.run_driver(yaml_path=joint_yaml, allow_fail=True)
  # RUN the freeze_trunk:true control (not just name it) and log
  # both phase-2 epoch times side by side; the visual comparison stays,
  # but the log must carry both numbers.
  control_yaml = ctx.require_config("joint-training-control")
  rc_c, out_c = ctx.run_driver(yaml_path=control_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="joint-training joint run completes",
             ok=(rc_j == 0),
             detail="joint exit code " + str(rc_j))
  ctx.expect(label="joint-training two-phase banner (regex 'two-phase: \\d+ trunk')",
             ok=logscan.search(text=out, pattern=r"two-phase: \d+ trunk"),
             detail="the trunk count is matched by regex, not pinned")
  ctx.expect(label="joint-training phase 'joint' banner",
             ok=logscan.contains(text=out, needle="phase 'joint'"),
             detail="phase 2 must announce the joint pass")
  ctx.expect(label="joint-training freeze_trunk:true control run completes",
             ok=(rc_c == 0),
             detail="control exit code " + str(rc_c))
  joint_epochs = logscan.matching_lines(text=out, pattern=r"^epoch")
  control_epochs = logscan.matching_lines(text=out_c, pattern=r"^epoch")
  joint_last = joint_epochs[-1] if len(joint_epochs) > 0 else "(no epoch line)"
  control_last = (control_epochs[-1] if len(control_epochs) > 0
                  else "(no epoch line)")
  ctx.log("joint-training phase-2 epoch time, side by side (the sanity signal is the "
          "joint time ABOVE the control, the trunk backward returned):")
  ctx.log("  joint (freeze_trunk:false):  " + joint_last)
  ctx.log("  control (freeze_trunk:true): " + control_last)
  ctx.log("loss continuous at the handoff (freeze-trunk-joint-phase2.md:"
          "226-228); the two numbers above are the Architect's visual check.")


def gate_gha_f(ctx):
  """head-activation-pin: the phase-2 head can pin its own activation.

  WHAT: a per-head activation pin, so the frozen-trunk head can use its
  own activation family (gated_power) regardless of the trunk's. WHY:
  the head trains in phase 2, so its family should be set there; and the
  pin is only legal on a frozen-trunk head phase, so an illegal pin must
  error rather than silently misbuild. HOW: the "model spec:" banner
  shows the trf activation dict and the head param count rises vs the
  shared default; passing --activation power prints the flag-vs-pin
  warning; freeze_trunk false with the pin errors in build_specs; plus
  the golden no-pin run (home note
  head-activation-per-component.md:239-242, 429-430).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: head-activation-per-component.md:239-242, :405-430.
  _golden_leg(ctx=ctx,
              gate_id="head-activation-pin",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|model spec)")
  out = _smoke_driver(ctx=ctx,
                      config_key="head-activation-pin-config",
                      required_banners=["gated_power"])
  # the flag-vs-pin warning leg: pass --activation power on the pinned YAML.
  pin_yaml = ctx.require_config("head-activation-pin-config")
  rc_w, out_w = ctx.run_driver(yaml_path=pin_yaml,
                               extra=("--activation=power",),
                               allow_fail=True)
  # the license-error leg (freeze_trunk false, or trunk_epochs 0)
  # WITH the head pin makes build_specs error with the frozen-trunk-head
  # message (head-activation-per-component.md:429-430). head-activation-pin-license is a
  # deliberately-invalid YAML.
  license_yaml = ctx.require_config("head-activation-pin-license")
  rc_l, out_l = ctx.run_driver(yaml_path=license_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(
    label="head-activation-pin flag-vs-pin warning",
    ok=logscan.contains(
      text=out_w,
      needle="the head keeps its model.trf.activation pin (gated_power)"),
    detail="the startup warning must state the pin wins over --activation")
  ctx.expect(
    label="head-activation-pin license error (freeze_trunk false + pin -> build_specs errors)",
    ok=(rc_l != 0 and logscan.search(text=out_l, pattern=r"(?i)frozen")),
    detail="build_specs must exit nonzero with the frozen-trunk-head message "
           "(rc " + str(rc_l) + ")")


def gate_gan_c(ctx):
  """relu-tanh-norm: the plain activations work, with the norm knob.

  WHAT: the parameter-free relu and tanh activation families plus the
  ResBlock normalization knob (per_feature / affine). WHY: tanh needs a
  saturation guard (per_feature norm) to be usable, the classic affine
  baseline must still work, and the banner must report which norm is
  live. HOW: a tanh + per_feature run and a tanh + affine run each name
  their norm in the banner and descend in loss; plus the golden
  absent-key run (home note activation-families-norm-knob.md:99-101).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: activation-families-norm-knob.md:99-101.
  _golden_leg(ctx=ctx,
              gate_id="relu-tanh-norm",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="relu-tanh-norm-per-feature",
                required_banners=["per_feature"])
  _smoke_driver(ctx=ctx,
                config_key="relu-tanh-norm-affine",
                required_banners=["affine"])


def gate_gwd_c(ctx):
  """weight-decay-census: weight decay touches only true weight matrices.

  WHAT: the module-aware weight-decay rule, which decays a parameter by
  its module's role rather than by its tensor shape. WHY: decaying an
  activation's shape parameters or a bias drags the model toward
  degenerate forms, so only real weight matrices should be decayed. HOW:
  with weight_decay 1e-4 on a gated_power model, the parameter-group
  census puts exactly the Linear / Conv1d / BinLinear weights in the
  decayed group and everything else (the multigate w/beta/mu, the
  BinLinear biases, all norms and biases) undecayed; plus the golden
  wd-0 run (home note weight-decay-only-weight-matrices.md:143-147).
  """
  ctx.require_caps("torch", "gpu")
  # home: weight-decay-only-weight-matrices.md:143-147 (weight-decay-census census +
  # golden wd-0 byte-identity).
  rc, out = ctx.run_check("gates/checks/gwd_census.py")
  if not ctx.dry:
    ctx.expect(label="weight-decay-census param-group census (allowlist exact)",
               ok=(rc == 0),
               detail="census check exit code " + str(rc))
  _golden_leg(ctx=ctx,
              gate_id="weight-decay-census",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")


def gate_gpc_c(ctx):
  """npce-training: the closed-form base plus its refiner train and save.

  WHAT: NPCE, a closed-form sparse-Legendre base under an SGD refiner,
  in both residual and ratio forms. WHY: the base must fit and the
  refiner correct it, both forms must work, NPCE must be exclusive with
  the features it replaces, and the base must refit for each training-set
  size. HOW: a residual run and a ratio run print the kept modes and
  descend, and save then rebuild of the h5 pce group matches base(theta)
  on a probe; the pce+ia and pce+rescale configs error; a 2-point
  n_train sweep refits per point; plus the golden absent-pce run (home
  note npce-yaml-wiring.md:117-122).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: npce-yaml-wiring.md:117-122, :201-204.
  _golden_leg(ctx=ctx,
              gate_id="npce-training",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="npce-training-residual",
                required_banners=["pce"])
  _smoke_driver(ctx=ctx,
                config_key="npce-training-ratio",
                required_banners=["pce"])
  # the exclusivity errors, each must error loudly
  # (npce-yaml-wiring.md:117-122). pce + ia is invalid via the YAML
  # alone; pce + rescale is a CLI exclusivity, so the excl-rescale leg
  # passes --rescale=residual (--rescale is a flag, never a YAML key).
  ia_yaml = ctx.require_config("npce-training-excl-ia")
  rc_ia, out_ia = ctx.run_driver(yaml_path=ia_yaml, allow_fail=True)
  rs_yaml = ctx.require_config("npce-training-excl-rescale")
  rc_rs, out_rs = ctx.run_driver(yaml_path=rs_yaml,
                                 extra=("--rescale=residual",),
                                 allow_fail=True)
  # the 2-point sweep_ntrain smoke: the base refits per point:
  # the sweep-over-n_train driver, with a 2-point geometric grid
  # (--n-min / --n-max / --n-points), not the single-train driver.
  sweep_yaml = ctx.require_config("npce-training-sweep")
  rc_s, out_s = ctx.run_driver(
    yaml_path=sweep_yaml,
    driver=SWEEP_NTRAIN_DRIVER,
    extra=("--n-min=1000", "--n-max=2000", "--n-points=2"),
    allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(
    label="npce-training exclusivity error (pce + ia)",
    ok=(rc_ia != 0 and logscan.search(text=out_ia, pattern=r"(?i)exclusive")),
    detail="pce + model.ia must exit nonzero with the exclusive message "
           "(rc " + str(rc_ia) + ")")
  ctx.expect(
    label="npce-training exclusivity error (pce + --rescale)",
    ok=(rc_rs != 0 and logscan.search(text=out_rs, pattern=r"(?i)exclusive")),
    detail="pce + --rescale=residual must exit nonzero with the exclusive "
           "message (rc " + str(rc_rs) + ")")
  refits = logscan.matching_lines(text=out_s, pattern=r"(?i)pce|refit|kept")
  ctx.expect(
    label="npce-training 2-point sweep_ntrain refits the base per point",
    ok=(rc_s == 0 and len(refits) >= 2),
    detail="the fit report should print once per sweep point (>=2 fit-report "
           "lines); saw " + str(len(refits)))
  ctx.log("npce-training rebuild-vs-base probe: save -> the h5 pce group -> from_state "
          "rebuild == base(theta) on a probe batch belongs in the check-script "
          "set (save-rebuild-drift's NPCE save round-trips the pce group; a standalone npce-training "
          "probe if wanted). Named in the remainder (npce-yaml-wiring.md:117).")


# --------------------------------------------------------------------------
# The save-to-sample acceptance chain (save-rebuild-drift's artifact feeds cobaya-adapter).
# --------------------------------------------------------------------------

def gate_gsv_c(ctx):
  """save-rebuild-drift: a saved emulator reloads exactly, forever.

  WHAT: the schema-v2 save/rebuild contract, which reconstructs an
  emulator from the h5 file alone. WHY: a saved emulator must reproduce
  exactly even if the code's default values drift later, or a reload
  silently ships a different model than was trained. HOW: train small,
  save, rebuild, and require the output bitwise-equal to the live model
  on a probe; the drift proof monkeypatches a code default and rebuilds
  unchanged; one factored and one NPCE save round-trip the geometry-class
  marker and the pce group; a v1 file is refused with a clear message
  (home note save-schema-resolved-config.md:86-93).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: save-schema-resolved-config.md:86-93 (save-rebuild-drift: bitwise + the
  # drift test + v1 refusal); the one-factored + one-NPCE-save
  # requirement is workstation-board-2026-07.md:66-71 (gate 18).
  rc, out = ctx.run_check("gates/checks/gsv_bitwise_drift.py")
  if ctx.dry:
    return
  ctx.expect(label="save-rebuild-drift save->rebuild bitwise + drift + v1-refusal",
             ok=(rc == 0),
             detail="check exit code " + str(rc)
                    + " (gates/checks/gsv_bitwise_drift.py)")


def gate_gct_c(ctx):
  """cobaya-adapter: inference reproduces training, so an MCMC is faithful.

  WHAT: the in-package inference predictor the cobaya theory block uses
  at MCMC time. WHY: the predictor must reproduce the training stack's
  physics exactly, or an MCMC would sample a different model than was
  trained. HOW: the predictor matches the training-side prediction on the
  same probe points to rtol 1e-6, including the factored save ->
  rebuild -> predict round-trip; the example evaluate run against the
  lsst_y1 likelihood matches the training-side datavector, and a short
  MCMC smoke confirms the theory drives a chain; this test depends on
  save-rebuild-drift, whose saved artifact feeds the probe (home note
  cobaya-theory-adapter.md:117-123).
  """
  ctx.require_caps("torch", "cosmolike", "cobaya", "gpu")
  # home: cobaya-theory-adapter.md:117-123 (cobaya-adapter), :234-238 (the real
  # factored round-trip added for the cobaya adapter).
  rc, out = ctx.run_check("gates/checks/gct_parity.py")
  if not ctx.dry:
    ctx.expect(label="cobaya-adapter parity probe (rtol 1e-6) + factored round-trip",
               ok=(rc == 0),
               detail="check exit code " + str(rc)
                      + " (gates/checks/gct_parity.py)")

  evaluate_yaml = ctx.evaluate_yaml()
  ctx.log("cobaya-adapter evaluate: cobaya-run the example evaluate YAML against the "
          "lsst_y1 likelihood (use_emulator 1); the printed datavector is "
          "compared to the training-side prediction.")
  rc_ev, out_ev = ctx.sh(
    cmd=["cobaya-run", str(evaluate_yaml)],
    cwd=ctx.rootdir(),
    allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="cobaya-adapter example evaluate run completes",
             ok=(rc_ev == 0),
             detail="cobaya-run exit code " + str(rc_ev))
  ctx.log("cobaya-adapter MCMC smoke: a short-chain sampler run confirms the theory "
          "block drives an MCMC (cobaya-theory-adapter.md:123); run it with "
          "an mcmc sampler override once the evaluate leg is green.")


# --------------------------------------------------------------------------
# The board, in execution order (workstation-board-2026-07.md).
# --------------------------------------------------------------------------

BOARD = [
  Gate(id="ema-off-identity",
       spec_code="GM-C",
       title="EMA off-mode byte-identity",
       tier=TIER_BACKLOG,
       home="weight-ema-snapshot-coupled",
       maps="98-101 (byte-identity gate); 229-238 (the epoch-line diff recipe)",
       run=gate_gm_c,
       needs=("torch", "cosmolike", "gpu"),
       worktree_commit="46ec5e1"),
  Gate(id="ema-smoke",
       spec_code="GM-D",
       title="EMA on-mode smoke",
       tier=TIER_BACKLOG,
       home="weight-ema-snapshot-coupled",
       maps="104-107, 240-251 (on-mode smoke: horizon banner + metrics); "
            "246-249 (the lr-cut rewind line)",
       run=gate_gm_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="production-diagnostic",
       spec_code="DIAG (G1, G-F, GN-F, GS-D, GT-C)",
       title="Production diagnostic run",
       tier=TIER_BACKLOG,
       home="driver-audit-phase-sweep-guards",
       maps="G1 audit-package-style-2026-07-05.md:232-234; G-F "
            "omegamh2-ns-product-cuts.md:125-126; GN-F "
            "param-cuts-nested-block.md:94-95; GS-D "
            "n-train-n-val-absolute-counts.md:110-112; GT-C "
            "triangle-cut-shading-all-windows.md:76-79",
       run=gate_diag,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="single-phase-demotion",
       spec_code="GP-D",
       title="Single-phase phase-arg demotion",
       tier=TIER_BACKLOG,
       home="resolve-phase-args-single-phase",
       maps="110-113 (single-phase demotion trains; two-phase control no-op)",
       run=gate_gp_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="head-scheduler-override",
       spec_code="GH-E",
       title="Head scheduler override",
       tier=TIER_BACKLOG,
       home="phase-blocks-nested-lr-scheduler",
       maps="262-267 (head override banner + cadence); 269-279 (golden diff)",
       run=gate_gh_e,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="eval-batch-invariance",
       spec_code="GE-C",
       title="Eval-batch partition invariance",
       tier=TIER_BACKLOG,
       home="eval-bs-decoupling",
       maps="102-108 (partition invariance rtol 1e-6 + timing); 202-300 script",
       run=gate_ge_c,
       needs=("torch", "gpu")),
  Gate(id="berhu-loss",
       spec_code="GB-C",
       title="berHu head loss",
       tier=TIER_BACKLOG,
       home="loss-mode-berhu",
       maps="148-153 (leg 1: berhu/_reduce numerics + autograd continuity); "
            "290-314 (leg 2: golden + head-berhu banners)",
       run=gate_gb_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="loss-schema-equivalence",
       spec_code="GL-D",
       title="Nested loss-schema equivalence",
       tier=TIER_BACKLOG,
       home="loss-block-nesting",
       maps="237-244 (new-schema reproduces pre-change epoch lines)",
       run=gate_gl_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="berhu-anneal",
       spec_code="GBA-C",
       title="berHu anneal schedule",
       tier=TIER_BACKLOG,
       home="berhu-anneal-schedule",
       maps="199-221 (golden no-anneal + anneal banner + continuity + s=1)",
       run=gate_gba_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="ema-anneal",
       spec_code="GME-C",
       title="EMA anneal schedule",
       tier=TIER_BACKLOG,
       home="ema-anneal-schedule",
       maps="180-197 (golden no-anneal + anneal banner + live-point metrics)",
       run=gate_gme_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="param-window-cuts",
       spec_code="item-27",
       title="Parameter-window cuts",
       tier=TIER_BACKLOG,
       home="omegamh2-ns-product-cuts",
       maps="125-126 (tight window: pool shrinkage matches count); "
            "param-cuts-nested-block.md:94-95 (nested block normal banner); "
            "248 (ci.init_probes A/B inspection)",
       run=gate_item27,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="triangle-shading",
       spec_code="GT-B",
       title="Triangle cut shading",
       tier=TIER_BACKLOG,
       home="triangle-cut-shading-all-windows",
       maps="72-75 (synthetic four-window triangle: artist-list fills + band)",
       run=gate_gt_b,
       optional=True,
       needs=("torch",)),

  Gate(id="joint-training",
       spec_code="GFT-C",
       title="freeze_trunk-false joint training",
       tier=TIER_NEW_FEATURES,
       home="freeze-trunk-joint-phase2",
       maps="115-120, 211-228 (joint banners + continuity + epoch-time signal)",
       run=gate_gft_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="head-activation-pin",
       spec_code="GHA-F",
       title="Pinned head activation",
       tier=TIER_NEW_FEATURES,
       home="head-activation-per-component",
       maps="239-242, 405-430 (model-spec banner + param count + warning); "
            "429-430 (leg 4: freeze_trunk false + pin -> build_specs errors)",
       run=gate_gha_f,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="relu-tanh-norm",
       spec_code="GAN-C",
       title="relu/tanh with the norm knob",
       tier=TIER_NEW_FEATURES,
       home="activation-families-norm-knob",
       maps="99-101 (tanh+per_feature + tanh+affine + golden absent-key)",
       run=gate_gan_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="weight-decay-census",
       spec_code="GWD-C",
       title="Weight-decay param-group census",
       tier=TIER_NEW_FEATURES,
       home="weight-decay-only-weight-matrices",
       maps="143-147 (gated_power wd 1e-4 census + golden wd-0 byte-identity)",
       run=gate_gwd_c,
       needs=("torch", "gpu")),
  Gate(id="npce-training",
       spec_code="GPC-C",
       title="NPCE training",
       tier=TIER_NEW_FEATURES,
       home="npce-yaml-wiring",
       maps="117-122, 201-204 (residual + ratio + rebuild + exclusivity + sweep)",
       run=gate_gpc_c,
       needs=("torch", "cosmolike", "gpu")),

  Gate(id="save-rebuild-drift",
       spec_code="GSV-C",
       title="Save/rebuild bitwise + drift",
       tier=TIER_SAVE_AND_SAMPLE,
       home="save-schema-resolved-config",
       maps="86-93 (bitwise + drift + v1 refusal); "
            "workstation-board-2026-07.md:66-71 (one factored + one NPCE save)",
       run=gate_gsv_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="cobaya-adapter",
       spec_code="GCT-C",
       title="Cobaya adapter parity",
       tier=TIER_SAVE_AND_SAMPLE,
       home="cobaya-theory-adapter",
       maps="117-123 (parity rtol 1e-6 + evaluate + MCMC); 234-238 (round-trip)",
       run=gate_gct_c,
       deps=("save-rebuild-drift",),
       needs=("torch", "cosmolike", "cobaya", "gpu")),
]

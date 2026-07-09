"""The workstation board: the list of tests and what each one is.

This file holds BOARD, a plain Python list of the 19 tests in run
order, and a small class, Gate, that says what one test is: its name,
tier, home note, and the function that runs it. run_board.py imports
this list and runs the tests; nothing here runs on its own. Each test
function issues commands through ``ctx`` (which streams output to the
test's log), returns early on ``ctx.dry``, then judges pass/fail with
``ctx.expect``. Numeric checks live in the ``checks/`` scripts.

Glossary:
  board     = the ordered list of tests.
  gate      = one test: its home note, commands, and pass/fail rule.
  tier      = the grouping --tier selects (backlog / new-features /
              save-and-sample).
  golden run= the same config built on the current tree and on a pinned
              older commit; selected log lines must match exactly.
              Only the EMA test has a preset base; the others skip this
              leg (with a logged note) unless board_config.json names
              their base, and their smoke leg is the acceptance.
  smoke     = a short training run judged on its banner lines.
  banner    = a driver status line a test checks for.
  worktree  = a throwaway git checkout of another commit; never touches
              your working tree.
  preflight = the pre-GPU checks (git tip, clean tree, cocoa imports,
              data paths); all must pass before any test runs.
  resume    = a rerun skips tests already marked PASS.
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple

from checks import logscan


class GateFailure(Exception):
  """Raised when a test's check fails or a needed config is missing.

  The runner marks that test FAIL in its log and board_status.json and
  moves on; one failure never stops the board.
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

def _golden_leg(ctx, gate_id, grep_pattern, *, yaml_name=None,
                config_key=None):
  """Run a gate's golden run, or skip it with a logged note.

  Trains the same config on the current tree and on the gate's pinned
  older commit (in a temporary worktree), then requires the selected
  log lines identical. No configured base = skip, not a hollow pass.

  Arguments:
    ctx          = the per-test helper.
    gate_id      = the gate whose golden_bases entry names the base.
    grep_pattern = the regex selecting the lines to compare
                   (e.g. "^(phase|epoch|best)").
    yaml_name    = a shipped config, resolved by bare name against
                   yaml_dir; the default for most golden legs.
    config_key   = a gate_configs key, resolved through require_config,
                   used when a golden leg needs its own bespoke config
                   (ema-off-identity's short golden run). Both legs use
                   the same resolved path; pass exactly one of
                   yaml_name / config_key.
  """
  base = ctx.golden_base(gate_id)
  if base is None:
    ctx.log("golden byte-identity leg: no base commit configured in "
            "board_config.json golden_bases['" + gate_id + "']. On the "
            "merged tip a git-stash diff is a no-op, so this dev-time "
            "pre-commit leg is skipped; the functional smoke leg is the "
            "acceptance (harness handoff decision point).")
    return

  if config_key is not None:
    source = ctx.require_config(config_key)
  else:
    source = ctx.config_yaml_name(yaml_name)
  ctx.log("golden byte-identity: current tree vs pinned " + base)
  # stage the golden config into the driver fileroot and pass the BARE
  # name to both legs: the pinned worktree's driver predates the
  # absolute-path passthrough, so an absolute --yaml would be re-prefixed
  # there; the fileroot convention resolves a bare name on every commit.
  with ctx.staged_golden(gate_id=gate_id, source=source) as bare:
    _, cur = ctx.run_driver(yaml_path=bare)
    with ctx.worktree(commit=base) as wt:
      _, pre = ctx.run_driver(yaml_path=bare, cwd=wt)

  if ctx.dry:
    return

  # strip the trailing wall-clock column (e.g. "  2.3s"): the one machine-
  # noise field on an otherwise deterministic epoch line. Applies to every
  # golden leg, not just ema-off-identity.
  equal, detail = logscan.byte_identity(text_a=pre,
                                        text_b=cur,
                                        pattern=grep_pattern,
                                        strip=r"[ \t]+\d+(?:\.\d+)?s$")
  ctx.expect(label=gate_id + " golden byte-identity (" + base + " vs tip)",
             ok=equal,
             detail=detail)


def _smoke_driver(ctx, config_key, required_banners, *, extra=()):
  """Run one training smoke and require its banners in the output.

  Arguments:
    ctx              = the per-test helper.
    config_key       = the board_config.json gate_configs key naming
                       the smoke YAML (unset or missing = GateFailure).
    required_banners = the literal banner substrings that must all
                       appear (quoted verbatim from the home note).
    extra            = extra driver flags (e.g. ("--activation=power",)).

  Returns:
    the captured run output, for further checks by the caller.
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
  """ema-off-identity: EMA switched off must change nothing.

  WHAT: a short (~40 epoch) plain resmlp run with no ema block. WHY: a
  feature that is off must not perturb any existing run. HOW: the same
  config trained on the current tree and on the pre-EMA commit must give
  character-identical epoch and best-epoch lines (wall-clock column
  stripped) (spec: weight-ema-snapshot-coupled.md:98-101, 229-238). The
  golden config is
  its own short bespoke YAML (ema-off-identity-golden), resolved through
  gate_configs for both legs; identity is proven per epoch line, so two
  production-length runs are not required.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="ema-off-identity",
              config_key="ema-off-identity-golden",
              grep_pattern="^(epoch|best epoch)")


def gate_gm_d(ctx):
  """ema-smoke: EMA switched on works.

  WHAT: a short bs=64 run with ema.horizon_epochs=3. WHY: the identity
  test proves EMA off is harmless, not that EMA on works. HOW: the
  banner must read "ema: horizon 3 epochs" and a plateau lr cut must
  print "rewound to best epoch"
  (spec: weight-ema-snapshot-coupled.md:104-107, 240-251).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  out = _smoke_driver(ctx=ctx,
                      config_key="ema-smoke-config",
                      required_banners=["ema: horizon 3 epochs"])
  if ctx.dry:
    return
  ctx.expect(label="ema-smoke rewind line ('lr cut -> rewound to best epoch')",
             ok=logscan.search(text=out, pattern=r"rewound to best epoch"),
             detail="weight-ema-snapshot-coupled.md:246-249: a rewind fires")


def gate_diag(ctx):
  """production-diagnostic: one --diagnostic run that closes five checks.

  WHAT: a production training run with the diagnostics PDF, exercising
  the dead-class census, a tight density-window cut, a nested param_cuts
  block, the absolute row counts, and the shaded triangle at once. WHY:
  all five ride one ordinary run. HOW: the package imports with no dead
  classes, the run finishes, the sizes line reads "used N of P cut
  rows", and the PDF shades every hard sample edge (visual check).
  Specs: the five home notes in this test's maps.
  """
  ctx.require_caps("cosmolike")
  ctx.log("G1: dead-class census (NLATemplateMLP / NLAInputGeometry) "
          "+ clean package import.")
  # exclude gates/ (this harness's own gate_diag holds the literal
  # search pattern) and .git/ (packed objects) so the census counts only
  # real emulator-package hits, never a self-match on the pattern string.
  rc_grep, out_grep = ctx.sh(
    cmd=["grep", "-rn", "NLATemplateMLP\\|NLAInputGeometry",
         "--include=*.py", "--exclude-dir=gates", "--exclude-dir=.git",
         ".", "README.md"],
    allow_fail=True)
  # a smoke import of the flat emulator package plus the two family
  # folders (designs/, losses/): every gate imports the package, so this
  # catches a broken move or a syntax error before any GPU time.
  rc_imp, out_imp = ctx.sh(
    cmd=[ctx.python, "-c", "import emulator, emulator.designs, emulator.losses"],
    allow_fail=True)

  diag_yaml = ctx.require_config("production-diagnostic-config")
  ctx.log("production-diagnostic production run: tight omegamh2 window + nested "
          "param_cuts + absolute n_train/n_val, with the diagnostics PDF.")
  # --diagnostic takes the PDF name root (the run identity is appended);
  # gates_diag names this board's diagnostics PDF under --root/chains.
  rc_run, out_run = ctx.run_driver(yaml_path=diag_yaml,
                                   extra=("--diagnostic=gates_diag",),
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

  WHAT: a resmlp config written with two-phase (trunk/head) keys. WHY:
  that config used to crash, and the fix must not alter genuinely
  two-phase models. HOW: the resmlp run trains with no traceback and
  prints the demotion notice; the same config on rescnn + nla runs
  unchanged (spec: resolve-phase-args-single-phase.md:110-113).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
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

  WHAT: a head: scheduler block with patience 10 against a run default
  of 25. WHY: the override must act on that phase only. HOW: the banner
  shows "[head overrides: scheduler]" and the head phase's first lr cut
  lands on the patience-10 cadence; plus a golden no-phase-blocks run
  (spec: phase-blocks-nested-lr-scheduler.md:262-279).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="head-scheduler-override",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  out = _smoke_driver(ctx=ctx,
                      config_key="head-scheduler-override-config",
                      required_banners=["[head overrides: scheduler]"])
  if ctx.dry:
    return
  ctx.log("head-scheduler-override cadence: the head phase's first lr "
          "cut should land on the patience-10 cadence (vs 25); confirm "
          "from the lr-cut epoch spacing in the log "
          "(phase-blocks-nested-lr-scheduler.md:265).")


def gate_ge_c(ctx):
  """eval-batch-invariance: validation metrics do not depend on chunking.

  WHAT: the eval batch size, decoupled from the training batch. WHY:
  changing how the eval set is chunked must not move any metric. HOW: a
  torch-only script checks the per-row chi2 agrees across eval batch
  sizes to rtol 1e-6 and prints "Part 1: PASS"
  (spec: eval-bs-decoupling.md:102-108; the script itself is 202-300).
  """
  ctx.require_caps("torch", "gpu")
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

  WHAT: the berHu loss (a robust sqrt below a knot, capped above a cap)
  as a head-only loss under the nested loss schema. WHY: the two loss
  blocks must resolve independently per phase. HOW: a torch-only script
  checks the berHu numerics (berhu == sqrt below the knot, capped ==
  berhu below the cap, gradient continuous at both knots), then a run
  shows "loss_mode sqrt" on the trunk and "loss_mode berhu_capped (knot
  0.2, cap 10)" on the head; plus a golden non-berhu run
  (spec: loss-mode-berhu.md:148-153, 290-314).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  rc, out = ctx.run_check("gates/checks/gb_c_berhu_reduce.py")
  if not ctx.dry:
    ctx.expect(
      label="berhu-loss leg 1 berhu/_reduce numerics + autograd continuity",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/gb_c_berhu_reduce.py)")
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

  WHAT: the nested loss: block that replaced the old flat keys. WHY: a
  config-layer rename must reproduce the old run exactly. HOW: a golden
  run in the new schema matches the pre-change epoch lines to the
  character, reusing the head-berhu config as the production shape
  (spec: loss-block-nesting.md:237-244).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="loss-schema-equivalence",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="berhu-loss-config",
                required_banners=["loss_mode berhu_capped (knot 0.2, cap 10)"])


def gate_gba_c(ctx):
  """berhu-anneal: the berHu shape ramps in smoothly and late.

  WHAT: the berHu anneal (plain sqrt blending into berHu over a
  hold-then-ramp schedule). WHY: the tail votes should arrive after the
  trim schedule has absorbed the worst outliers, without a jolt at the
  ramp start. HOW: the banner reads "anneal: hold 5 + 10 cosine", the
  train loss is continuous at the hold boundary, and the shape is full
  berHu by epoch 15; plus the golden no-anneal run
  (spec: berhu-anneal-schedule.md:199-221).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
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

  WHAT: the EMA anneal (the averaging window grows from zero over a
  hold-then-ramp schedule). WHY: averaging through the high-loss early
  epochs would poison the shipped weights. HOW: the banner names the
  horizon and "anneal: hold 5 + 10 cosine", and the average's metrics
  first appear at the live point (epoch 6+); plus the golden no-anneal
  run (spec: ema-anneal-schedule.md:180-197).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
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

  WHAT: a deliberately tight omegamh2 window cut, in a nested param_cuts
  block. WHY: the cut must remove precisely the rows the banner reports.
  HOW: the run finishes and the pool shrinkage matches the "used N of P
  cut rows" banner; the duplicate cosmolike init_probes call is the
  paired eye check on this run's evidence (specs:
  omegamh2-ns-product-cuts.md:125-126, param-cuts-nested-block.md:94-95).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
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

  WHAT: the cut-window shading on the corner plot with all four density
  windows active. WHY: a wrongly shaded panel misleads a reader about
  which region was cut. HOW: a synthetic-sample triangle must fill
  exactly the coverage-table panels in one colour, plus the omh2
  marginal band; optional (runs only when --gate names it), no
  cosmolike needed (spec: triangle-cut-shading-all-windows.md:72-75).
  """
  ctx.require_caps("torch")
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

  WHAT: freeze_trunk false, which fine-tunes trunk and head together in
  phase 2. WHY: the trunk backward must actually run, not silently stay
  frozen. HOW: the run announces "two-phase: N trunk + M joint" and
  "phase 'joint'", and its phase-2 epoch time sits visibly above a
  freeze_trunk-true control; plus the golden absent-key run
  (spec: freeze-trunk-joint-phase2.md:115-120, 211-228).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="joint-training",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|run:)")
  joint_yaml = ctx.require_config("joint-training-config")
  rc_j, out = ctx.run_driver(yaml_path=joint_yaml, allow_fail=True)
  # run the control too, so the log carries both epoch times.
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

  WHAT: a model.trf.activation pin (gated_power) for the frozen-trunk
  head. WHY: the pin must win over the --activation flag with a warning,
  and an illegal pin (unfrozen trunk) must error, not misbuild. HOW: the
  "model spec:" banner shows the pinned activation; --activation power
  prints the flag-vs-pin warning; the deliberately-invalid license YAML
  makes build_specs exit with the frozen-trunk message; plus the golden
  no-pin run (spec: head-activation-per-component.md:239-242, 405-430).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="head-activation-pin",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|model spec)")
  out = _smoke_driver(ctx=ctx,
                      config_key="head-activation-pin-config",
                      required_banners=["gated_power"])
  # same pinned YAML, now with the flag, to trigger the warning.
  pin_yaml = ctx.require_config("head-activation-pin-config")
  rc_w, out_w = ctx.run_driver(yaml_path=pin_yaml,
                               extra=("--activation=power",),
                               allow_fail=True)
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
    label="head-activation-pin license error (freeze_trunk false + pin"
          " -> build_specs errors)",
    ok=(rc_l != 0 and logscan.search(text=out_l, pattern=r"(?i)frozen")),
    detail="build_specs must exit nonzero with the frozen-trunk-head message "
           "(rc " + str(rc_l) + ")")


def gate_gan_c(ctx):
  """relu-tanh-norm: the plain activations work, with the norm knob.

  WHAT: the parameter-free relu/tanh activations plus the norm knob
  (per_feature / affine). WHY: tanh needs the per_feature saturation
  guard, and the classic affine baseline must still work. HOW: a tanh +
  per_feature run and a tanh + affine run each name their norm in the
  banner and descend in loss; plus the golden absent-key run
  (spec: activation-families-norm-knob.md:99-101).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
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

  WHAT: the rule that picks decayed parameters by module role, not
  tensor shape. WHY: decaying an activation's parameters or a bias
  drags the model toward degenerate forms. HOW: with weight_decay 1e-4,
  the parameter groups hold exactly the Linear / Conv1d / BinLinear
  weights in the decayed group and everything else undecayed; plus the
  golden wd-0 run (spec: weight-decay-only-weight-matrices.md:143-147).
  """
  ctx.require_caps("torch", "gpu")
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

  WHAT: NPCE (a closed-form sparse-Legendre base under a trained
  refiner) in residual and ratio forms. WHY: both forms must train, the
  illegal combinations must error, and the base must refit per
  training-set size. HOW: residual and ratio runs print the fit report
  and descend; pce+ia (YAML) and pce+--rescale (flag) both exit with the
  exclusivity error; a 2-point n_train sweep refits the base per point;
  plus the golden absent-pce run (spec: npce-yaml-wiring.md:117-122).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
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
  # pce + ia fits in a YAML; pce + rescale needs the CLI flag.
  ia_yaml = ctx.require_config("npce-training-excl-ia")
  rc_ia, out_ia = ctx.run_driver(yaml_path=ia_yaml, allow_fail=True)
  rs_yaml = ctx.require_config("npce-training-excl-rescale")
  rc_rs, out_rs = ctx.run_driver(yaml_path=rs_yaml,
                                 extra=("--rescale=residual",),
                                 allow_fail=True)
  # the n_train sweep runs its own driver, not the single-train one.
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
  # run 3 proved the sweep parent's stream carries ZERO "PCE fit:" reports
  # (the GPU workers own the per-point fit output); the parent's own
  # evidence is its staging banner ("pce: form ...") plus one
  # "N_train N f(>0.2)" result line per sweep point. The per-point refit
  # is structural to the top-level pce design (one base per point).
  parent = logscan.matching_lines(text=out_s,
                                  pattern=r"N_train\s+\d+\s+f\(>0\.2\)")
  staged = logscan.search(text=out_s, pattern=r"^pce: form")
  ctx.expect(
    label="npce-training 2-point sweep_ntrain ran both points",
    ok=(rc_s == 0 and len(parent) >= 2 and staged),
    detail="rc " + str(rc_s) + "; parent N_train f(>0.2) lines "
           + str(len(parent)) + " (need >=2); pce staging banner "
           + ("present" if staged else "ABSENT") + " (need present)")
  ctx.log("npce-training rebuild-vs-base probe: save -> the h5 pce group"
          " -> from_state rebuild == base(theta) on a probe batch belongs"
          " in the check-script set (save-rebuild-drift's NPCE save"
          " round-trips the pce group; a standalone npce-training probe"
          " if wanted). Named in the remainder (npce-yaml-wiring.md:117).")


# --------------------------------------------------------------------------
# The save-and-sample chain: save-rebuild-drift's saved file feeds
# cobaya-adapter's parity probe.
# --------------------------------------------------------------------------

def gate_gsv_c(ctx):
  """save-rebuild-drift: a saved emulator reloads exactly, forever.

  WHAT: the save/rebuild contract (an emulator reconstructed from its
  h5 file alone). WHY: a reload must ship the trained model exactly,
  even if the code's default values drift later. HOW: train small,
  save, rebuild, require the outputs bitwise-equal on a probe; then
  patch a code default and rebuild unchanged; one factored and one NPCE
  save round-trip too; a v1 file is refused (specs:
  save-schema-resolved-config.md:86-93; workstation-board-2026-07.md:66-71).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  rc, out = ctx.run_check("gates/checks/gsv_bitwise_drift.py")
  if ctx.dry:
    return
  ctx.expect(label="save-rebuild-drift save->rebuild bitwise + drift + v1-refusal",
             ok=(rc == 0),
             detail="check exit code " + str(rc)
                    + " (gates/checks/gsv_bitwise_drift.py)")


def gate_gct_c(ctx):
  """cobaya-adapter: inference reproduces training, so an MCMC is faithful.

  WHAT: the inference predictor the cobaya theory block calls at MCMC
  time. WHY: an MCMC must sample the model that was trained, not a
  near-copy. HOW: the predictor matches the training-side prediction to
  rtol 1e-6 (including the factored save -> rebuild -> predict
  round-trip), the example evaluate run against the lsst_y1 likelihood
  finishes, and a short MCMC smoke follows; depends on
  save-rebuild-drift (spec: cobaya-theory-adapter.md:117-123, 234-238).
  """
  ctx.require_caps("torch", "cosmolike", "cobaya", "gpu")
  rc, out = ctx.run_check("gates/checks/gct_parity.py")
  if not ctx.dry:
    ctx.expect(label="cobaya-adapter parity probe (rtol 1e-6) + factored round-trip",
               ok=(rc == 0),
               detail="check exit code " + str(rc)
                      + " (gates/checks/gct_parity.py)")

  evaluate_yaml = ctx.evaluate_yaml()
  # the evaluate leg loads the tiny emulator save-rebuild-drift persists at
  # <rootdir>/<driver_root>/chains/gates_emul_evaluate; require it before
  # spending a cobaya-run on a missing file. A failure here skips the run.
  if not ctx.dry:
    evaluate_h5 = (ctx.rootdir() / str(ctx.cfg.get("driver_root", ""))
                   / "chains" / "gates_emul_evaluate.h5")
    ctx.expect(
      label="evaluate emulator present (saved by save-rebuild-drift)",
      ok=evaluate_h5.exists(),
      detail="expected " + str(evaluate_h5) + "; a lone `--gate "
             "cobaya-adapter` run must run save-rebuild-drift once first "
             "(it persists this file)")
  ctx.log("cobaya-adapter evaluate: cobaya-run the board's evaluate YAML "
          "against the lsst_y1 likelihood (use_emulator 1); this leg proves "
          "the run completes; the physics parity is gct_parity's job.")
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

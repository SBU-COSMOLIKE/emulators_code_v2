"""The workstation board: the list of tests and what each one is.

This file holds BOARD, a plain Python list of the tests in run order
(count them by enumerating this list, never from a doc's number), and
a small class, Gate, that says what one test is: its name,
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
SWEEP_NTRAIN_DRIVER = "cosmic_shear_sweep_ntrain_emulator.py"


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
    spec_code = the key of this test's audit-history entry inside its home
              note's ledger tables. Registry data for the notes only: it is
              never printed or documented outside notes/ (user ruling
              2026-07-12 — internal tracking codes stay in notes).
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
  stripped) (spec: training-stack.md:98-101, 229-238). The
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
  (spec: training-stack.md:104-107, 240-251).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  out = _smoke_driver(ctx=ctx,
                      config_key="ema-smoke-config",
                      required_banners=["ema: horizon 3 epochs"])
  if ctx.dry:
    return
  ctx.expect(label="ema-smoke rewind line ('lr cut -> rewound to best epoch')",
             ok=logscan.search(text=out, pattern=r"rewound to best epoch"),
             detail="training-stack.md:246-249: a rewind fires")


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
  ctx.log("dead-class census (NLATemplateMLP / NLAInputGeometry) "
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

  # the dead-class census (grep 0 hits: grep exits 1 when it finds nothing).
  ctx.expect(label="dead-class census -> 0 hits",
             ok=(rc_grep == 1 and out_grep.strip() == ""),
             detail="grep rc " + str(rc_grep) + ", output: "
                    + repr(out_grep.strip()[:200]))
  ctx.expect(label="clean package import",
             ok=(rc_imp == 0),
             detail="import rc " + str(rc_imp))
  # the window-cut / row-count / sizes banners (home lines named in `maps`).
  ctx.expect(label="production-diagnostic production run completes",
             ok=(rc_run == 0),
             detail="driver exit code " + str(rc_run))
  ctx.expect(label="sizes line ('used N of P cut rows')",
             ok=logscan.search(text=out_run,
                               pattern=r"used\s+\d+\s+of\s+\d+\s+cut rows"),
             detail="the sizes line must report used N of P cut rows")
  ctx.log("shaded triangle: the diagnostics PDF is a VISUAL check (the omh2 marginal "
          "at 0.20 and the (ns, omh2) diagonal corner at 0.17 must show "
          "adjoining grey); the harness confirms the run produced it, the "
          "Architect confirms the shading from the committed PDF/log.")


def gate_gp_d(ctx):
  """single-phase-demotion: a single-phase model accepts two-phase keys.

  WHAT: a resmlp config written with two-phase (trunk/head) keys. WHY:
  that config used to crash, and the fix must not alter genuinely
  two-phase models. HOW: the resmlp run trains with no traceback and
  prints the demotion notice; the same config on rescnn + nla runs
  unchanged (spec: training-stack.md:110-113).
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
                    "training-stack.md:111")
  ctx.expect(label="single-phase-demotion control rescnn+nla reproduces today (no-op)",
             ok=(rc_c == 0),
             detail="control run exit code " + str(rc_c))


def gate_gh_e(ctx):
  """head-scheduler-override: the head phase cuts the lr on its own patience.

  WHAT: a head: scheduler block with patience 10 against a run default
  of 25. WHY: the override must act on that phase only. HOW: the banner
  shows "[head overrides: scheduler]" and the head phase's first lr cut
  lands on the patience-10 cadence; plus a golden no-phase-blocks run
  (spec: training-stack.md:262-279).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="head-scheduler-override",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)")
  out = _smoke_driver(ctx=ctx,
                      config_key="head-scheduler-override-config",
                      required_banners=["[head overrides: scheduler]"])
  if ctx.dry:
    return
  ctx.log("head-scheduler-override cadence: the head phase's first lr "
          "cut should land on the patience-10 cadence (vs 25); confirm "
          "from the lr-cut epoch spacing in the log "
          "(training-stack.md:265).")


def gate_ge_c(ctx):
  """eval-batch-invariance: validation metrics do not depend on chunking.

  WHAT: the eval batch size, decoupled from the training batch. WHY:
  changing how the eval set is chunked must not move any metric. HOW: a
  torch-only script checks the per-row chi2 agrees across eval batch
  sizes to rtol 1e-6 and prints "Part 1: PASS"
  (spec: training-stack.md:102-108; the script itself is 202-300).
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
  (spec: training-stack.md:148-153, 290-314).
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
              yaml_name="cosmic_shear_train_emulator.yaml",
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
  (spec: training-stack.md:237-244).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="loss-schema-equivalence",
              yaml_name="cosmic_shear_train_emulator.yaml",
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
  (spec: training-stack.md:199-221).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="berhu-anneal",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)")
  out = _smoke_driver(ctx=ctx,
                      config_key="berhu-anneal-config",
                      required_banners=["anneal: hold 5 + 10 cosine"])
  if ctx.dry:
    return
  ctx.log("berhu-anneal schedule: confirm the train loss is continuous at the "
          "hold boundary (epoch 5->6) and s=1 (full berhu) by epoch 15; "
          "the first ~5 epochs match a plain sqrt run "
          "(training-stack.md:213-221).")


def gate_gme_c(ctx):
  """ema-anneal: the EMA average wakes up only after the bad early era.

  WHAT: the EMA anneal (the averaging window grows from zero over a
  hold-then-ramp schedule). WHY: averaging through the high-loss early
  epochs would poison the shipped weights. HOW: the banner names the
  horizon and "anneal: hold 5 + 10 cosine", and the average's metrics
  first appear at the live point (epoch 6+); plus the golden no-anneal
  run (spec: training-stack.md:180-197).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="ema-anneal",
              yaml_name="cosmic_shear_train_emulator.yaml",
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
  data-generation-and-cuts.md:125-126, data-generation-and-cuts.md:94-95).
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
          "geometries.output.py is inspected with this run's evidence "
          "(data-generation-and-cuts.md:248); a manual A/B, not an "
          "automatable assertion.")


def gate_gt_b(ctx):
  """triangle-shading: the diagnostics triangle greys the right panels.

  WHAT: the cut-window shading on the corner plot with all four density
  windows active. WHY: a wrongly shaded panel misleads a reader about
  which region was cut. HOW: a synthetic-sample triangle must fill
  exactly the coverage-table panels in one colour, plus the omh2
  marginal band; optional (runs only when --gate names it), no
  cosmolike needed (spec: data-generation-and-cuts.md:72-75).
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
  (spec: training-stack.md:115-120, 211-228).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="joint-training",
              yaml_name="cosmic_shear_train_emulator.yaml",
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
  ctx.log("loss continuous at the handoff (training-stack.md:"
          "226-228); the two numbers above are the Architect's visual check.")


def gate_gha_f(ctx):
  """head-activation-pin: the phase-2 head can pin its own activation.

  WHAT: a model.trf.activation pin (gated_power) for the frozen-trunk
  head. WHY: the pin must win over the --activation flag with a warning,
  and an illegal pin (unfrozen trunk) must error, not misbuild. HOW: the
  "model spec:" banner shows the pinned activation; --activation power
  prints the flag-vs-pin warning; the deliberately-invalid license YAML
  makes build_specs exit with the frozen-trunk message; plus the golden
  no-pin run (spec: models-and-designs.md:239-242, 405-430).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="head-activation-pin",
              yaml_name="cosmic_shear_train_emulator.yaml",
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
  (spec: models-and-designs.md:99-101).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="relu-tanh-norm",
              yaml_name="cosmic_shear_train_emulator.yaml",
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
  golden wd-0 run (spec: training-stack.md:143-147).
  """
  ctx.require_caps("torch", "gpu")
  rc, out = ctx.run_check("gates/checks/gwd_census.py")
  if not ctx.dry:
    ctx.expect(label="weight-decay-census param-group census (allowlist exact)",
               ok=(rc == 0),
               detail="census check exit code " + str(rc))
  _golden_leg(ctx=ctx,
              gate_id="weight-decay-census",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)")


def gate_gpc_c(ctx):
  """npce-training: the closed-form base plus its refiner train and save.

  WHAT: NPCE (a closed-form sparse-Legendre base under a trained
  refiner) in residual and ratio forms. WHY: both forms must train, the
  illegal combinations must error, and the base must refit per
  training-set size. HOW: residual and ratio runs print the fit report
  and descend; pce+ia (YAML) and pce+--rescale (flag) both exit with the
  exclusivity error; a 2-point n_train sweep refits the base per point;
  plus the golden absent-pce run (spec: models-and-designs.md:117-122).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="npce-training",
              yaml_name="cosmic_shear_train_emulator.yaml",
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
          " if wanted). Named in the remainder (models-and-designs.md:117).")


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
  artifacts-inference-warmstart.md:86-93; gates-and-board.md:66-71).
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
  save-rebuild-drift (spec: artifacts-inference-warmstart.md:117-123, 234-238).
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
          "block drives an MCMC (artifacts-inference-warmstart.md:123); run it with "
          "an mcmc sampler override once the evaluate leg is green.")


def gate_ftw_a(ctx):
  """finetune-identity: a warm-started emulator starts as the source, exactly.

  This gate proves the promise fine-tuning rests on: at epoch 0 a warm-started
  emulator computes the source emulator's own function, bit for bit, whatever
  the new parameters are. The check script builds a tiny synthetic source (a
  small ResMLP with hand-built geometries, no cosmolike), saves it, then runs
  the warm-start path with two extra parameters and asserts the shared-
  parameter encoding, the weight transfer, the pre-train parity, the no-extras
  degenerate case, and the loud errors (spec: artifacts-inference-warmstart.md, the
  finetune-identity validation gate). torch only, no cosmolike.
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/finetune_identity.py")
  if not ctx.dry:
    ctx.expect(
      label="finetune-identity encode + transfer + parity + degenerate + errors",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/finetune_identity.py)")


def gate_ftw_b(ctx):
  """finetune-smoke: a real fine-tune run continues the board's own emulator.

  A names-equal fine-tune (no extra parameters) that warm-starts from the tiny
  emulator save-rebuild-drift persists under the board fileroot, at a lower
  learning rate for two epochs. It confirms the pre-train parity verdict
  prints, the startup banner names the source, and the run completes. The
  saved-artifact provenance (the finetuned_from root attr) and the save ->
  rebuild -> predict round-trip are the save-and-sample follow-up the Architect
  confirms from the workstation artifact (spec: artifacts-inference-warmstart.md, the
  finetune-smoke validation gate). Depends on save-rebuild-drift, which
  persists the source emulator this run continues.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  smoke_yaml = ctx.require_config("finetune-smoke-config")
  rc, out = ctx.run_driver(yaml_path=smoke_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="finetune-smoke run completes",
             ok=(rc == 0),
             detail="finetune driver exit code " + str(rc))
  ctx.expect(
    label="finetune-smoke pre-train parity verdict printed",
    ok=logscan.search(text=out, pattern=r"finetune parity: max\|dv\|"),
    detail="the [ok] parity line proves epoch 0 reproduces the source")
  ctx.expect(
    label="finetune-smoke banner names the warm-start source",
    ok=logscan.search(text=out, pattern=r"finetune: from "),
    detail="print_design must announce the source artifact")
  ctx.log("finetune-smoke save/rebuild + finetuned_from: the saved h5 carries "
          "the finetuned_from root attr, and a rebuild_emulator round-trip "
          "predicts identically to the in-memory model (the save-rebuild-drift "
          "pattern); the Architect confirms these from the saved artifact.")


def gate_tpe_a(ctx):
  """transfer-identity: a frozen base plus a zero correction is the base, exactly.

  This gate proves the transfer identity: at epoch 0 the composed prediction
  (a frozen base under a parallel correction whose output is zeroed) is bitwise
  the frozen base's own decode, in every form (gain / sum) x space (physical /
  whitened) combination, for a plain base and a factored (three-template) base.
  The check builds two tiny synthetic bases (no cosmolike), saves and reloads
  them, and asserts the base-encoding slice, the epoch-0 identity, the base
  caching, the zero-init surgery, and the config error paths. Added
  2026-07-12 (the family symmetry ruling), check_diagonal: TransferDiagChi2
  on a GridGeometry — epoch-0 identity bitwise through the log law for both
  forms, the packed-target discipline, the whitened-only rejections, the
  family validators' acceptance matrix, a grid transfer artifact predicting
  the composition bitwise, and the cross-family-base loud error (spec:
  artifacts-inference-warmstart.md, the transfer-identity validation gate). torch
  only, no cosmolike.
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/transfer_identity.py")
  if not ctx.dry:
    ctx.expect(
      label="transfer-identity slice + identity + packing + surgery + errors",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/transfer_identity.py)")


def gate_tpe_b(ctx):
  """transfer-smoke: a real transfer run composes the board's own base.

  A names-equal gain-form transfer (no extra parameters) that warm-starts a
  parallel correction over the plain base save-rebuild-drift persists under the
  board fileroot, for two epochs. It confirms the epoch-0 parity verdict prints,
  the banner names the base and form, and the run completes. The saved-artifact
  provenance (the transfer_from root attr, the embedded transfer_base group) and
  the save -> rebuild -> predict round-trip are the save-and-sample follow-up the
  Architect confirms from the workstation artifact (spec:
  artifacts-inference-warmstart.md, the transfer-smoke validation gate). Depends on
  save-rebuild-drift, which persists the base this run composes.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  smoke_yaml = ctx.require_config("transfer-smoke-config")
  rc, out = ctx.run_driver(yaml_path=smoke_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="transfer-smoke run completes",
             ok=(rc == 0),
             detail="transfer driver exit code " + str(rc))
  ctx.expect(
    label="transfer-smoke epoch-0 parity verdict printed",
    ok=logscan.search(text=out, pattern=r"transfer parity: epoch 0 == frozen base"),
    detail="the [ok] parity line proves epoch 0 is the frozen base")
  ctx.expect(
    label="transfer-smoke banner names the base and form",
    ok=logscan.search(text=out, pattern=r"transfer: from "),
    detail="print_design must announce the base + form/space")
  # The transfer artifact lifecycle: the run SAVES a self-contained
  # transfer artifact (the correction net + the embedded frozen base). Assert
  # the save completed and its two output paths printed.
  ctx.expect(
    label="transfer-smoke saved the transfer artifact",
    ok=logscan.search(text=out, pattern=r"saved emulator ->")
       and logscan.search(text=out, pattern=r"saved run record ->"),
    detail="the composed run persists a reloadable artifact "
           "(the saved file embeds its frozen base)")
  ctx.log("transfer-smoke artifact provenance + round-trip: the saved .h5 "
          "carries the transfer_from root attr + the embedded transfer_base "
          "group, and rebuild_emulator -> composed predict reproduces the "
          "in-memory composition bitwise (the transfer-identity lifecycle leg "
          "proves the mechanism; the Architect confirms it on this artifact "
          "from the workstation).")


# --------------------------------------------------------------------------
# The board, in execution order (gates-and-board.md).
# --------------------------------------------------------------------------

def gate_spe_a(ctx):
  """scalar-identity: a scalar emulator saves, rebuilds, and predicts exactly.

  WHAT: a tiny synthetic scalar emulator (a ParamGeometry over a written
  covmat + a ScalarGeometry over synthetic targets + a small ResMLP), saved
  and rebuilt, reproduces predict bitwise; its ScalarGeometry state round-trips
  byte-identical; and every scalar-path loud error fires (the constant-column
  guard both directions, the duplicate-sidecar-name guard, the trunk-only
  guard, plus the emul_scalars provides / duplicate / overlap / subset /
  wrong-kind legs, the adapter loaded torch-only
  through a cobaya.theory stub). Added 2026-07-12: the NPCE check_npce
  leg (residual algebra bitwise, base + net {name: value} prediction
  exact). torch only, no cosmolike (spec:
  families-scalar-cmb.md, the scalar-identity gate).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/scalar_identity.py")
  if not ctx.dry:
    ctx.expect(
      label="scalar-identity round-trip + state + provides + all error legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/scalar_identity.py)")


def gate_spe_b(ctx):
  """scalar-smoke: a real scalar emulator learns an exactly-derivable target.

  WHAT: a fixture parameter chain whose only output is
  omegamh2 = omegam*(H0/100)^2, trained two epochs; the validation error
  collapses below the mean-predictor baseline (so a network that learned
  nothing fails), predict reproduces
  the analytic value at an off-center point within 5%, and a cobaya evaluate
  through emul_scalars returns the same derived value. torch + cobaya, no
  cosmolike (the scalar path is cosmolike-free); the check writes its own
  fixture, so it needs no other gate (spec: families-scalar-cmb.md, the
  scalar-smoke gate).
  """
  ctx.require_caps("torch", "cobaya")
  rc, out = ctx.run_check("gates/checks/scalar_smoke.py")
  if not ctx.dry:
    ctx.expect(
      label="scalar-smoke fixture train + off-center predict + cobaya evaluate",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/scalar_smoke.py)")


def gate_cme_a(ctx):
  """cmb-identity: the CMB emulator identity + law + roughness + finetune.

  WHAT: tiny synthetic CMB emulators (a ParamGeometry over a written covmat
  + a CmbDiagonalGeometry over a synthetic fiducial C_ell + a small ResMLP)
  prove: the RULED cosmic-variance constants (sigma_l = C_fid*sqrt(2/(2l+1)),
  the covinv ruling); the geometry state round-trip byte-identical (nine
  keys incl. the law strings); the as_exp2tau law exact both ways (_factor
  bitwise, encode(decode) to float32 round-off) + its loud errors; save ->
  rebuild -> predict bitwise on BOTH laws (the predictor's CMB branch); the
  emul_cmb adapter's Cl assembly + every loud error (duplicate spectrum,
  wrong-kind, unknown-spectrum / beyond-lmax must_provide, both get_Cl
  convention guards; cobaya.theory stubbed, torch-only); the
  roughness legs (band ratio > 100, zero -> exactly 0, OFF identity
  bitwise, one-reduction composition, the lensing guard < 3%); and the
  finetune legs (epoch-0 parity from a CMB source, the cosmolike
  pin's wrong-kind refusal, validate_cmb accepting finetune). Added
  2026-07-11/12: the correction-head leg (ResTRF + n_tokens: attach,
  identity basis, epoch-0 identity, the two-phase discipline, save ->
  rebuild -> predict bitwise) and the NPCE check_npce leg (residual
  algebra bitwise, roughness composition, base + net prediction
  bitwise, the pce x amplitude-law exclusivity). Also the eq-6
  lens-induced covariance legs (check_covariance_oracle): an affine fake
  CAMBdata makes the 5-point stencil exact, so compute_cmb_covariance's
  non-Gaussian contraction is checked against an independent known
  answer for eq 6, built directly from the sensitivity matrix and the
  lensing-potential variance. Five legs: the exact contraction (the
  pipeline equals the direct eq 6); the old-weight miss (the earlier
  band-summed-variance weights are wrong by orders of magnitude); the
  raw-vs-scaled fixture integrity (the fake serves the scaled
  [L(L+1)]^2 C potential and refuses the raw getter, the pipeline reads
  raw for the weight); the width-3 band projection (a constant-response
  band reproduces the per-multipole eq 6); and the exact zero-band
  weight (a zeroed band's persisted weight is exactly 0). torch only, no
  CAMB (spec: notes/families-scalar-cmb.md).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/cmb_identity.py")
  if not ctx.dry:
    ctx.expect(
      label="cmb-identity constants + law + round-trip + roughness + "
            "finetune + adapter + covariance known-answer legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/cmb_identity.py)")


def gate_cme_b(ctx):
  """cmb-smoke: the CMB emulator end to end on real CAMB.

  WHAT: dataset_generator_cmb.py writes two tiny dumps (200 rows each,
  l = 2..350, cmblensed, As sampled linearly) — four per-spectrum dv files
  + sidecars, phiphi actually filled; compute_cmb_covariance.py
  writes the Gaussian .npz on the fixture LCDM (its first real run);
  a data.cmb / as_exp2tau training run collapses the val median below
  0.5x the staged mean predictor (the bar a dead network cannot pass);
  the saved artifact serves Cl through the real cobaya
  lifecycle (get_model + add_requirements + provider.get_Cl equals the
  predictor's own output); and the family diagnostics pages build. Added
  2026-07-12, leg 2b (check_cov_nondiagonal): the Motloch & Hu eq-6
  NON-DIAGONAL covariance runs end to end at smoke scale (16
  re-lensings) — all six dense blocks (3 per-spectrum + 3 cross),
  symmetric + PSD + off-diagonals alive, the stencil step study and the
  fractional-amplitude contraction keys in the provenance (the
  normalization fix, notes/families-scalar-cmb.md). torch + cobaya + a
  compiled CAMB under $ROOTDIR; budget
  several minutes (~400 serial low-accuracy CAMB calls) (spec:
  notes/families-scalar-cmb.md).
  """
  ctx.require_caps("torch", "cobaya")
  rc, out = ctx.run_check("gates/checks/cmb_smoke.py")
  if not ctx.dry:
    ctx.expect(
      label="cmb-smoke generator + covariance + train + cobaya + diagnostics",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/cmb_smoke.py)")


def gate_bsn_a(ctx):
  """bsn-identity: the grid emulator + the imposed-physics pipeline.

  WHAT: the cumulative Simpson (even doubled-grid points exact on cubics,
  the odd node the correct one-interval integral, exact on quadratics —
  45M-12, superseding the old half-chunk form; the old form is the gate's
  mutation control) and the H(z)->distances pipeline against a closed-form
  flat LCDM at 1e-6; the GridGeometry log_offset law both ways + the
  state round-trip byte-identical + the un-standardizable /
  log-positivity / unknown-law guards; save -> rebuild -> predict
  bitwise on BOTH laws (the predictor's grid branch); the emul_baosn
  adapter legs (pair validation, window layout, the DESERT loud at
  must_provide AND the getters, the piecewise chi vs the pipeline / the
  D_M artifact, get_Hubble units + window guards, D_A_2); and the
  finetune legs (epoch-0 parity from a grid source; the
  metadata-mismatch and cross-quantity from_config errors). Added
  2026-07-12: the NPCE check_npce leg (residual algebra bitwise
  through the log law, base + net prediction bitwise). torch +
  scipy, no CAMB (spec: notes/families-background-mps.md).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/bsn_identity.py")
  if not ctx.dry:
    ctx.expect(
      label="bsn-identity pipeline + law + round-trip + adapter + "
            "finetune legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/bsn_identity.py)")


def gate_bsn_b(ctx):
  """bsn-smoke: the BAOSN emulators end to end, checked against CAMB.

  WHAT: dataset_generator_background.py writes two tiny dumps (200
  rows, one background-only CAMB evaluation per sample — fast) carrying
  BOTH quantities + their _z.npy grid sidecars (one CAMB pass fills
  both — the one-pass rule); two
  data.grid training runs (Hubble/log_offset + D_M/none) each collapse
  below 0.5x the staged mean predictor (the dead-network-relative bar);
  the real cobaya lifecycle through emul_baosn serves H / D_A (SN
  window) and D_M (recombination window) within 2% of CAMB's OWN
  background at an off-center point — truth is available here, the
  strongest smoke of the program; the desert stays loud through the
  real lifecycle; the grid-family diagnostics pages build. torch + cobaya +
  a compiled CAMB under $ROOTDIR (spec: notes/families-background-mps.md).
  """
  ctx.require_caps("torch", "cobaya")
  rc, out = ctx.run_check("gates/checks/bsn_smoke.py")
  if not ctx.dry:
    ctx.expect(
      label="bsn-smoke generator + two trainings + cobaya-vs-CAMB "
            "+ diagnostics",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/bsn_smoke.py)")


def gate_mps_a(ctx):
  """mps-identity: the grid2d emulator + the syren-law assembly math.

  WHAT: the Grid2DGeometry standardize/state round-trips + its width /
  unknown-law guards (the partial-constant raise leg died when the
  constant-column pin was made law-agnostic — the run-10 catch); the STAGING
  law transform through the REAL load_source (law rows = log(raw/base)
  with the base dump aligned by dump_rows through a real shuffled
  staging; k_stride keeps the top edge; positivity loud); the BOUNDED
  staging on the production 122 x 2,000 grid (guarded memmap reads prove
  every raw + base read is row-chunked and column-thinned, an
  independent known-answer match, a disk-backed low-RAM result, and the
  guard trips on the old whole-selection access) and the STABLE streamed
  moments (a 50,000-row 1e8/1-ULP column keeps its true std through the
  Chan/Welford accumulator, never a false constant pin); save ->
  rebuild -> predict bitwise on both laws (the predictor's grid2d
  branch returns the reshaped (nz, nk) surface); the emul_mps assembly
  EXACT against synthetic base stubs (P_lin = exp(net)*base, the low-k
  blend pins boost -> 1 below k_t, P_nl = B*P_lin, the boost base fed
  the EMULATED P_lin — the legacy flow), its pair/grid/wrong-kind
  guards, the legacy state keys + interpolator node round-trip, and
  the reject-on-bad-spectrum semantics; validate_grid2d's pairing /
  base-file / k_stride legs (transfer ACCEPTED since the 2026-07-12
  symmetry ruling); the finetune parity + metadata-mismatch
  legs. Added 2026-07-11/12: the correction-head leg (ResCNN on z-slice
  channels, the two-phase discipline, the n_tokens-on-real-bins
  rejection, the bitwise round-trip), the constant-pin legs,
  and the NPCE check_npce leg (residual algebra + base + net
  prediction bitwise, the diagonal ratio rejection). torch + scipy,
  no CAMB, no symbolic_pofk (the real syren formulas ride the EMUL2
  acceptance) (spec: notes/families-background-mps.md).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/mps_identity.py")
  if not ctx.dry:
    ctx.expect(
      label="mps-identity geometry + staging law + assembly + finetune "
            "legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/mps_identity.py)")


def gate_mps_b(ctx):
  """mps-smoke: the MPS emulators end to end on real CAMB (law none).

  WHAT: dataset_generator_mps.py writes two tiny dumps (200 rows,
  16 z x 40 k) through the real Pk_interpolator requirement (incl. the
  verbatim wants-Cl quirk): pklin + boost + the grid sidecars; two
  data.grid2d trainings (law none) each collapse below 0.5x the staged
  mean predictor (the boost training also proves the grid2d diagnostics
  pages: the two (z, k) figures build and the plot_diagnostics PDF
  lands); the real cobaya lifecycle through emul_mps serves
  P_lin and P_nl (grid + interpolator) within 5% of CAMB's OWN
  P(k, z) at an off-center point; the interpolator range guard. The
  syren-law path is exactly gated by mps-identity's stubbed legs (the
  formulas themselves are vendored in syren/), and the full syren +
  EMUL2 hybrid run is the unit's recorded acceptance experiment
  (cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml, user-run on the
  workstation). torch + cobaya + a compiled CAMB under $ROOTDIR
  (spec: notes/families-background-mps.md).
  """
  ctx.require_caps("torch", "cobaya")
  rc, out = ctx.run_check("gates/checks/mps_smoke.py")
  if not ctx.dry:
    ctx.expect(
      label="mps-smoke generator + two trainings + cobaya-vs-CAMB",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/mps_smoke.py)")


def gate_geo_a(ctx):
  """geo-paths: the geometry folder is the ONLY geometry home.

  WHAT: a fresh save writes the folder cls paths automatically
  (emulator.geometries.<name>.<Class> via type().__module__, the
  resolved-values rule) and rebuilds + predicts through those stored
  strings; the six legacy flat modules are DEAD — gone from disk and
  raising ModuleNotFoundError on import (a pre-retirement artifact
  fails loudly with the module path in the error, never a silent
  partial load); and the tree-wide census proves nothing references
  the old flat names. The shims that originally rode the move were
  retired by user ruling 2026-07-11 (no science artifact predates the
  move). Acceptance beyond this gate = the full board green
  (every gate touches geometries) with ema-off-identity pinning
  byte-identity — the same acceptance pattern the family-folders move
  used (spec: notes/artifacts-inference-warmstart.md).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/geo_paths.py")
  if not ctx.dry:
    ctx.expect(
      label="geo-paths new-save markers + dead legacy paths + census",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/geo_paths.py)")


def gate_finite_contract(ctx):
  """finite-contract: a NaN / Inf never ranks, selects, or ships a model.

  WHAT: the finite training/evaluation contract, on the REAL functions. A
  non-finite per-sample chi2 counts as below every threshold, so a diverged
  model would report a perfect fraction of 0.0 and be snapshotted best, and
  the pre-train parity gates would print "[ok] ... max|dv| = nan" as if the
  warm start held. Every site must abort loudly instead. WHY: the board's
  primary selection metric and the warm-start parity verdict must be
  impossible to satisfy with a non-finite value; the dead-network collapse
  bars themselves fail open otherwise. HOW: a CPU torch-only check drives
  eval_val (a finite control reproduces the reference metrics; one NaN;
  +/-Inf), the real training_loop_batched step (a NaN loss raises before
  backward, a non-finite gradient before optimizer.step, the weights bitwise
  unchanged), eval_source_chi2 (side diagnostic), both warm-start parity gates
  (no-extra both-arms NaN, one-arm NaN, Inf, extras-present NaN, and a
  non-finite transfer surface, each with the finite-contract message, never a
  misleading "extras leaked" / "frozen base" / tolerance verdict), and the
  45M-24 safe-sqrt producer (an exact-fit chi2 == 0 has a finite, zero
  gradient in every sqrt mode instead of the 0/0 = NaN it used to produce;
  positives agree with sqrt; a negative / NaN chi2 is refused; eager and
  torch.compile agree), and the 45M-47 epoch reduction (a finite per-batch
  loss near the float32 max yields a finite epoch mean via host float64
  accumulation, where the old device float32 loss*bs product overflowed to
  Inf), and the 45M-53 / 45M-60 chi2-domain boundary (eval_val and
  eval_source_chi2 raise on a finite negative chi2 that training folds; the
  scale-aware band scales with the per-row kept WIDTH, not w^2, so a
  production-width leg refuses a chi2 = -2 the retired w^2 rule crowned as
  perfect, a mutation arm restoring w^2 is caught, and an ill-conditioned SPD
  control shows genuine roundoff near zero falls inside the band); the valid
  controls keep their metrics and their [ok] parity lines. torch only, no
  cosmolike, no GPU (spec: training-stack.md, the "NaN scores as a perfect
  emulator" section and its pre-training parity + 45M-24 + 45M-47 + 45M-53 +
  45M-60 clauses).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/finite_contract.py")
  if not ctx.dry:
    ctx.expect(
      label="finite-contract eval/train/diagnostic/parity/safe-sqrt legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/finite_contract.py)")


BOARD = [
  Gate(id="ema-off-identity",
       spec_code="GM-C",
       title="EMA off-mode byte-identity",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="98-101 (byte-identity gate); 229-238 (the epoch-line diff recipe)",
       run=gate_gm_c,
       needs=("torch", "cosmolike", "gpu"),
       worktree_commit="46ec5e1"),
  Gate(id="ema-smoke",
       spec_code="GM-D",
       title="EMA on-mode smoke",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="104-107, 240-251 (on-mode smoke: horizon banner + metrics); "
            "246-249 (the lr-cut rewind line)",
       run=gate_gm_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="production-diagnostic",
       spec_code="DIAG (G1, G-F, GN-F, GS-D, GT-C)",
       title="Production diagnostic run",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="dead-class census conventions-and-workflow.md:232-234; "
            "density-window cut data-generation-and-cuts.md:125-126; "
            "nested cuts + absolute row counts "
            "data-generation-and-cuts.md:94-95; sizes line "
            "training-stack.md:110-112; shaded triangle "
            "data-generation-and-cuts.md:76-79",
       run=gate_diag,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="single-phase-demotion",
       spec_code="GP-D",
       title="Single-phase phase-arg demotion",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="110-113 (single-phase demotion trains; two-phase control no-op)",
       run=gate_gp_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="head-scheduler-override",
       spec_code="GH-E",
       title="Head scheduler override",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="262-267 (head override banner + cadence); 269-279 (golden diff)",
       run=gate_gh_e,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="eval-batch-invariance",
       spec_code="GE-C",
       title="Eval-batch partition invariance",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="102-108 (partition invariance rtol 1e-6 + timing); 202-300 script",
       run=gate_ge_c,
       needs=("torch", "gpu")),
  Gate(id="finite-contract",
       spec_code="FIN-A",
       title="Finite training/evaluation contract",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the training-stack finite contract: the 'NaN scores as a "
            "perfect emulator' section (the eval_val / train-step / "
            "eval_source_chi2 guards), its pre-training parity clause "
            "(build_warm_start + build_transfer_start), the 45M-24 "
            "safe-sqrt producer clause (exact-fit finite gradients per "
            "mode, positives analytic, negative/NaN chi2 refused, eager + "
            "compiled), the 45M-47 epoch-reduction clause (host float64 "
            "accumulation; a finite epoch mean where the old float32 "
            "loss*bs product overflowed), the 45M-53 chi2-domain "
            "clause (eval_val / eval_source_chi2 raise on a finite "
            "negative chi2 that training folds; the scale-aware band; the "
            "finite-only false-crowning mutation; the capability-gated "
            "compile arm), and the 45M-60 width-band clause (the band "
            "scales with the kept WIDTH, not w^2: a production-width leg "
            "refuses a chi2 = -2 the retired w^2 rule crowned perfect, a "
            "w^2-restoring mutation arm, a scalar-width leg, a subclass "
            "census, and an ill-conditioned SPD roundoff control); the red "
            "legs plus the finite controls",
       run=gate_finite_contract,
       needs=("torch",)),
  Gate(id="berhu-loss",
       spec_code="GB-C",
       title="berHu head loss",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="148-153 (leg 1: berhu/_reduce numerics + autograd continuity); "
            "290-314 (leg 2: golden + head-berhu banners)",
       run=gate_gb_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="loss-schema-equivalence",
       spec_code="GL-D",
       title="Nested loss-schema equivalence",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="237-244 (new-schema reproduces pre-change epoch lines)",
       run=gate_gl_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="berhu-anneal",
       spec_code="GBA-C",
       title="berHu anneal schedule",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="199-221 (golden no-anneal + anneal banner + continuity + s=1)",
       run=gate_gba_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="ema-anneal",
       spec_code="GME-C",
       title="EMA anneal schedule",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="180-197 (golden no-anneal + anneal banner + live-point metrics)",
       run=gate_gme_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="param-window-cuts",
       spec_code="item-27",
       title="Parameter-window cuts",
       tier=TIER_BACKLOG,
       home="data-generation-and-cuts",
       maps="125-126 (tight window: pool shrinkage matches count); "
            "data-generation-and-cuts.md:94-95 (nested block normal banner); "
            "248 (ci.init_probes A/B inspection)",
       run=gate_item27,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="triangle-shading",
       spec_code="GT-B",
       title="Triangle cut shading",
       tier=TIER_BACKLOG,
       home="data-generation-and-cuts",
       maps="72-75 (synthetic four-window triangle: artist-list fills + band)",
       run=gate_gt_b,
       optional=True,
       needs=("torch",)),

  Gate(id="joint-training",
       spec_code="GFT-C",
       title="freeze_trunk-false joint training",
       tier=TIER_NEW_FEATURES,
       home="training-stack",
       maps="115-120, 211-228 (joint banners + continuity + epoch-time signal)",
       run=gate_gft_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="head-activation-pin",
       spec_code="GHA-F",
       title="Pinned head activation",
       tier=TIER_NEW_FEATURES,
       home="models-and-designs",
       maps="239-242, 405-430 (model-spec banner + param count + warning); "
            "429-430 (leg 4: freeze_trunk false + pin -> build_specs errors)",
       run=gate_gha_f,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="relu-tanh-norm",
       spec_code="GAN-C",
       title="relu/tanh with the norm knob",
       tier=TIER_NEW_FEATURES,
       home="models-and-designs",
       maps="99-101 (tanh+per_feature + tanh+affine + golden absent-key)",
       run=gate_gan_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="weight-decay-census",
       spec_code="GWD-C",
       title="Weight-decay param-group census",
       tier=TIER_NEW_FEATURES,
       home="training-stack",
       maps="143-147 (gated_power wd 1e-4 census + golden wd-0 byte-identity)",
       run=gate_gwd_c,
       needs=("torch", "gpu")),
  Gate(id="npce-training",
       spec_code="GPC-C",
       title="NPCE training",
       tier=TIER_NEW_FEATURES,
       home="models-and-designs",
       maps="117-122, 201-204 (residual + ratio + rebuild + exclusivity + sweep)",
       run=gate_gpc_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="finetune-identity",
       spec_code="FTW-A",
       title="Fine-tune warm-start identity",
       tier=TIER_NEW_FEATURES,
       home="artifacts-inference-warmstart",
       maps="256-273 (encode + transfer + parity + degenerate + error paths)",
       run=gate_ftw_a,
       needs=("torch",)),
  Gate(id="transfer-identity",
       spec_code="TPE-A",
       title="Transfer frozen-base identity",
       tier=TIER_NEW_FEATURES,
       home="artifacts-inference-warmstart",
       maps="204-219 (slice + 4x form/space identity + packing + surgery + errors)",
       run=gate_tpe_a,
       needs=("torch",)),
  Gate(id="scalar-identity",
       spec_code="SPE-A",
       title="Scalar emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-scalar-cmb",
       maps="123-127 (round-trip + state + auto-provides + subset/dup + "
            "the constant-column / duplicate-name / trunk-only / "
            "wrong-kind error legs)",
       run=gate_spe_a,
       needs=("torch",)),
  Gate(id="cmb-identity",
       spec_code="CME-A",
       title="CMB emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-scalar-cmb",
       maps="110-117 (identity legs) + the 45M-21 amplitude-metric legs "
            "(the factored chi2 divides f out: physical-chi2 invariance "
            "under (A_s, tau), the uncorrected f^2 catch-power, the "
            "factor-corrected roughness residual, params-required); "
            "517-530 (roughness gate legs); 582-591 (finetune legs); "
            "141-203 (the eq-6 covariance known-answer legs: the exact "
            "contraction, the old-weight miss, the raw-vs-scaled fixture "
            "integrity, the width-3 band projection, and the exact "
            "zero-band weight)",
       run=gate_cme_a,
       needs=("torch",)),
  Gate(id="bsn-identity",
       spec_code="BSN-A",
       title="BAOSN grid emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-background-mps",
       maps="118-127 (identity legs); 138-176 (the "
            "two-regime + desert legs); 217-231 (finetune legs)",
       run=gate_bsn_a,
       needs=("torch",)),
  Gate(id="mps-identity",
       spec_code="MPS-A",
       title="MPS grid2d emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-background-mps",
       maps="the note's matter-power sections: geometry + laws; the base "
            "placement + staging transform; the bounded-staging + "
            "stable-moments legs and the staging-lifecycle legs "
            "(experiment-owned temp files, supersede-on-restage, "
            "sweep-lane release, failure unlink; Grid2d staging defeats "
            "its own memory ladder, data-generation-and-cuts.md); "
            "identity legs; finetune",
       run=gate_mps_a,
       needs=("torch",)),
  Gate(id="geo-paths",
       spec_code="GEO-A",
       title="Geometry folder is the only geometry home",
       tier=TIER_NEW_FEATURES,
       home="artifacts-inference-warmstart",
       maps="the note's geometry-folder section: import rewrite census; "
            "new-save markers + full-board acceptance; shims retired "
            "(legacy flat paths dead, loudly)",
       run=gate_geo_a,
       needs=("torch",)),

  Gate(id="save-rebuild-drift",
       spec_code="GSV-C",
       title="Save/rebuild bitwise + drift",
       tier=TIER_SAVE_AND_SAMPLE,
       home="artifacts-inference-warmstart",
       maps="86-93 (bitwise + drift + v1 refusal); "
            "gates-and-board.md:66-71 (one factored + one NPCE save)",
       run=gate_gsv_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="cobaya-adapter",
       spec_code="GCT-C",
       title="Cobaya adapter parity",
       tier=TIER_SAVE_AND_SAMPLE,
       home="artifacts-inference-warmstart",
       maps="117-123 (parity rtol 1e-6 + evaluate + MCMC); 234-238 (round-trip)",
       run=gate_gct_c,
       deps=("save-rebuild-drift",),
       needs=("torch", "cosmolike", "cobaya", "gpu")),
  Gate(id="finetune-smoke",
       spec_code="FTW-B",
       title="Fine-tune warm-start smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="artifacts-inference-warmstart",
       maps="277-284 (names-equal fine-tune: parity line + banner + completes; "
            "finetuned_from + save-rebuild round-trip are the workstation leg)",
       run=gate_ftw_b,
       deps=("save-rebuild-drift",),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="transfer-smoke",
       spec_code="TPE-B",
       title="Transfer frozen-base smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="artifacts-inference-warmstart",
       maps="221-228 (names-equal gain transfer: parity + banner + completes; "
            "transfer_from + embedded base + round-trip are the workstation leg)",
       run=gate_tpe_b,
       deps=("save-rebuild-drift",),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="scalar-smoke",
       spec_code="SPE-B",
       title="Scalar emulator smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="families-scalar-cmb",
       maps="128-134 (fixture train + collapse + off-center predict + "
            "cobaya evaluate through emul_scalars)",
       run=gate_spe_b,
       needs=("torch", "cobaya")),
  Gate(id="cmb-smoke",
       spec_code="CME-B",
       title="CMB emulator smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="families-scalar-cmb",
       maps="118-124 (end-to-end: generator + covariance + train + "
            "cobaya lifecycle); 575-578 (the family diagnostics leg); "
            "141-203 (leg 2b: the eq-6 non-diagonal blocks + the "
            "fractional-amplitude weight-key provenance)",
       run=gate_cme_b,
       needs=("torch", "cobaya")),
  Gate(id="bsn-smoke",
       spec_code="BSN-B",
       title="BAOSN emulator smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="families-background-mps",
       maps="128-136 (end-to-end vs CAMB's own background); "
            "178-194 (the grid diagnostics leg)",
       run=gate_bsn_b,
       needs=("torch", "cobaya")),
  Gate(id="mps-smoke",
       spec_code="MPS-B",
       title="MPS emulator smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="families-background-mps",
       maps="the note's matter-power sections: the generator incl. the "
            "wants-Cl quirk; the emul_mps lifecycle vs CAMB's own P(k, z)",
       run=gate_mps_b,
       needs=("torch", "cobaya")),
]

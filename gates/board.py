"""The workstation board: the ordered gate table the harness drives.

This module is the gate registry. It defines the Gate record, the
three tier names, and BOARD, the single ordered list every gate lives
in (the order the home notes and the board note fix: GM-C's pinned
golden leg first, then the standing gates, then this week's legs, then
the save to sample acceptance chain). Each gate carries its home note
and a ``maps`` string naming the home-note line(s) its assertions
implement, so the Architect can audit that the harness encodes the
note and not a memory of it. run_board.py imports Gate, GateFailure,
and BOARD; it constructs the RunContext each gate's run function
receives.

Every gate's run function follows one shape. It issues its shell
commands through ``ctx.sh`` / ``ctx.run_driver`` (which tee output to
the gate's raw log), then returns early on ``ctx.dry`` (so --dry-run
prints the plan and stops before touching any acceptance), then
asserts its acceptance with ``ctx.expect`` and the pure helpers in
``checks.logscan``. Numeric acceptances (the parameter-group census,
the partition invariance, the bitwise save/rebuild equality, the
training-to-inference parity) live in the executable ``checks/``
scripts a gate launches, so the harness itself computes and asserts
them and the raw log records every value.

Golden byte-identity legs use the temporary-worktree mechanism, never
a checkout or stash in place: the pinned pre-feature build runs in a
throwaway ``git worktree`` the runner removes even on failure. GM-C's
base is preset (the pre-EMA commit); the other golden legs read their
base from board_config.json and, when it is unset, log that the leg
was a dev-time pre-commit diff and run only the functional smoke leg
(a committed feature's off path is already exercised by the standard
runs). This is the one declared deviation from the home notes' literal
``git stash`` recipes, which are hollow on the merged tip.

PS: gate = one verification with a home note, commands, and an
acceptance; tier = the board's coarse grouping (standing / week /
save-sample); golden byte-identity = two builds' selected log lines
equal to the character; smoke = a short run asserted on its banners
rather than on a numeric tolerance; worktree = a throwaway checkout of
another commit that never disturbs the user's tree.
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple

from checks import logscan


class GateFailure(Exception):
  """Raised inside a gate when an acceptance check fails.

  The runner catches it, records the gate FAIL with the message in the
  raw log and board_status.json, and moves on to the next gate (a
  single gate's failure never stops the board). Gate functions raise
  it directly for an unrecoverable precondition (a missing config), and
  ``ctx.expect`` raises it for a failed acceptance value.
  """


# The three tiers, in board order. The strings are the tier selector
# values --tier accepts and the labels the BOARD.md table prints.
TIER_STANDING = "standing"
TIER_WEEK = "week"
TIER_SAVE_SAMPLE = "save-sample"

# The sweep-over-n_train driver GPC-C's 2-point smoke runs (D-GH7); the
# default single-train driver would execute the wrong program on a sweep
# YAML. It sits beside the emulator package at the repo root.
SWEEP_NTRAIN_DRIVER = "sweep_ntrain_emulator_cosmic_shear.py"


@dataclass(frozen=True)
class Gate:
  """One row of the board: what to run and how it is judged.

  Arguments:
    id      = the gate identifier (e.g. "GSV-C"); the log filename
              stem, the selector --gate accepts, and the resume key.
    tier    = one of TIER_STANDING / TIER_WEEK / TIER_SAVE_SAMPLE.
    home    = the home note filename stem (the spec of record); printed
              in the log header for the Architect's audit trail.
    maps    = the home-note line reference(s) each assertion implements
              (the GGH-B mapping: assertion -> note line), printed in
              the header so the audit checks encoding, not memory.
    run     = the gate body, run(ctx) -> None; issues commands and
              asserts acceptance, raising GateFailure on any failure.
    deps    = gate ids whose PASS this gate needs; an unmet dependency
              records SKIPPED(dependency) rather than running.
    optional= when True the gate is skipped unless --gate names it
              explicitly (GT-B, registered but off the default sweep).
    needs   = the environment capabilities the gate requires
              ("torch", "cosmolike", "cobaya", "gpu"); documentation
              for the header and a clearer skip message.
    worktree_commit = a commit the gate pins a temporary worktree at
              (GM-C's pre-EMA build); None for gates that never leave
              the current tree.
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
    ctx          = the run context (sh / worktree / expect / dry).
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
    ctx              = the run context.
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
  """GM-C: the golden pre-EMA off-mode byte-identity run.

  The EMA feature's acceptance: with the ema block absent the loop is
  byte-identical to the pre-EMA build. The pre-EMA build runs in a
  temporary worktree at the pinned commit (never a checkout in place),
  and the epoch / best-epoch lines must match to the character.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: weight-ema-snapshot-coupled.md:98-101 (byte-identity gate),
  # :229-238 (the diff <(grep '^(epoch|best epoch)') recipe).
  _golden_leg(ctx=ctx,
              gate_id="GM-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(epoch|best epoch)")


def gate_gm_d(ctx):
  """GM-D: the on-mode EMA smoke (banner + tracked metrics).

  A short bs=64 run with ema.horizon_epochs=3: the banner names the
  horizon, and the per-epoch metrics are the average's from the
  warmup-end epoch onward.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: weight-ema-snapshot-coupled.md:104-107, :240-251 (GM-D recipe:
  # the "ema: horizon 3 epochs" banner; metrics track then smooth).
  out = _smoke_driver(ctx=ctx,
                      config_key="GM-D-emasmoke",
                      required_banners=["ema: horizon 3 epochs"])
  if ctx.dry:
    return
  # D-GH5d: the note's rewind line is mechanically assertable
  # (weight-ema-snapshot-coupled.md:246-249: a plateau lr cut rewinds
  # to the best epoch and the ema metrics jump WITH the raw ones).
  ctx.expect(label="GM-D rewind line ('lr cut -> rewound to best epoch')",
             ok=logscan.search(text=out, pattern=r"rewound to best epoch"),
             detail="weight-ema-snapshot-coupled.md:246-249: a rewind fires")


def gate_diag(ctx):
  """DIAG: the one production --diagnostic run closing five gates.

  Closes G1 (dead-class census + clean package import), G-F (a tight
  omegamh2 window: pool shrinkage matches the banner count), GN-F (a
  nested param_cuts block shows the normal banner), GS-D (the sizes
  line reports "used N of P cut rows" with N the YAML n_train), and
  GT-C (the regenerated diagnostics PDF shades every hard sample edge).
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
  # D-GH8: the harness's own interpreter, never a bare "python" on PATH.
  rc_imp, out_imp = ctx.sh(
    cmd=[ctx.python, "-c", "import emulator, emulator.IA, emulator.PCE"],
    allow_fail=True)

  diag_yaml = ctx.require_config("DIAG")
  ctx.log("DIAG production run: tight omegamh2 window + nested "
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
  ctx.expect(label="DIAG production run completes",
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
  """GP-D: single-phase demotion trains; the two-phase control is a no-op.

  The exact resmlp YAML that used to traceback now trains and the
  banner shows the demotion notice; the same YAML on rescnn+nla
  reproduces today's behavior (resolution is a no-op for capable
  models).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: resolve-phase-args-single-phase.md:110-113.
  single_yaml = ctx.require_config("GP-D-single")
  control_yaml = ctx.require_config("GP-D-control")
  rc_s, out_s = ctx.run_driver(yaml_path=single_yaml, allow_fail=True)
  rc_c, out_c = ctx.run_driver(yaml_path=control_yaml, allow_fail=True)

  if ctx.dry:
    return

  ctx.expect(label="GP-D single-phase resmlp trains (was a traceback)",
             ok=(rc_s == 0),
             detail="resmlp run exit code " + str(rc_s))
  ctx.expect(label="GP-D demotion notice in the banner",
             ok=logscan.search(text=out_s,
                               pattern=r"(single-phase|demot|resolve)"),
             detail="EXACT notice string to confirm against "
                    "resolve-phase-args-single-phase.md:111")
  ctx.expect(label="GP-D control rescnn+nla reproduces today (no-op)",
             ok=(rc_c == 0),
             detail="control run exit code " + str(rc_c))


def gate_gh_e(ctx):
  """GH-E: a head scheduler override cuts on its own cadence.

  A two-phase run with a head scheduler patience 10: the banner shows
  [head overrides: scheduler] and the head phase's first lr cut arrives
  on the patience-10 cadence (vs the run's 25). Plus the golden
  no-phase-blocks byte-identity leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: phase-blocks-nested-lr-scheduler.md:262-267 (override), :269-279
  # (golden diff <(grep '^(phase|epoch|best)')).
  _golden_leg(ctx=ctx,
              gate_id="GH-E",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  out = _smoke_driver(ctx=ctx,
                      config_key="GH-E-headpatience",
                      required_banners=["[head overrides: scheduler]"])
  if ctx.dry:
    return
  ctx.log("GH-E cadence: the head phase's first lr cut should land on the "
          "patience-10 cadence (vs 25); confirm from the lr-cut epoch "
          "spacing in the log (phase-blocks-nested-lr-scheduler.md:265).")


def gate_ge_c(ctx):
  """GE-C: eval_val partition invariance + the eval-batch timing.

  Runs the torch-only check script: per-row chi2 from eval_val agrees
  across eval batch sizes to rtol 1e-6 (Part 1: PASS) and the derived
  eval batch cuts the eval time on CUDA (Part 2).
  """
  ctx.require_caps("torch", "gpu")
  # home: eval-bs-decoupling.md:102-108 (acceptance), :202-300 (the
  # ready-to-paste script this check mirrors).
  rc, out = ctx.run_check("gates/checks/ge_c_eval_bs.py")
  if ctx.dry:
    return
  ctx.expect(label="GE-C Part 1 partition invariance (rtol 1e-6)",
             ok=logscan.contains(text=out, needle="Part 1: PASS"),
             detail="the script must print 'Part 1: PASS'")
  ctx.expect(label="GE-C check script exit 0",
             ok=(rc == 0),
             detail="check exit code " + str(rc))


def gate_gb_c(ctx):
  """GB-C: a head-berhu run under the nested loss schema.

  The trunk banner reads plain "loss_mode sqrt" and the head banner
  "loss_mode berhu_capped (knot 0.2, cap 10)"; loss decreases. Plus the
  golden non-berhu byte-identity leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # leg 1 (D-GH5a): the torch-only unbound _reduce numerics
  # (loss-mode-berhu.md:148-153): berhu == sqrt below the knot,
  # berhu_capped == berhu below the cap, manual references, autograd
  # continuity across BOTH knots, non-default knots. The check script
  # is part of the check-script remainder.
  rc, out = ctx.run_check("gates/checks/gb_c_berhu_reduce.py")
  if not ctx.dry:
    ctx.expect(
      label="GB-C leg 1 berhu/_reduce numerics + autograd continuity",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/gb_c_berhu_reduce.py)")
  # leg 2: the golden non-berhu byte-identity + the head-berhu run
  # (loss-mode-berhu.md:290-314).
  _golden_leg(ctx=ctx,
              gate_id="GB-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="GB-C-headberhu",
                required_banners=["loss_mode sqrt",
                                  "loss_mode berhu_capped (knot 0.2, cap 10)"])


def gate_gl_d(ctx):
  """GL-D: the nested loss schema reproduces the pre-change epoch lines.

  Golden equivalence: the same physical config expressed in the new
  loss schema reproduces the pre-change run's epoch lines (a
  config-layer change, numerics untouched). Reuses the head-berhu
  config as the production shape.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: loss-block-nesting.md:237-244.
  _golden_leg(ctx=ctx,
              gate_id="GL-D",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="GB-C-headberhu",
                required_banners=["loss_mode berhu_capped (knot 0.2, cap 10)"])


def gate_gba_c(ctx):
  """GBA-C: the berhu anneal smoke (continuous at the hold boundary).

  With berhu.anneal on, the banner reads
  "...(knot 0.2, cap 10; anneal: hold 5 + 10 cosine)"; the printed
  train loss is continuous at the hold boundary and s reaches 1 (full
  berhu) by epoch 15. Plus the golden no-anneal byte-identity leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: berhu-anneal-schedule.md:199-221.
  _golden_leg(ctx=ctx,
              gate_id="GBA-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  out = _smoke_driver(ctx=ctx,
                      config_key="GBA-C-anneal",
                      required_banners=["anneal: hold 5 + 10 cosine"])
  if ctx.dry:
    return
  ctx.log("GBA-C schedule: confirm the train loss is continuous at the "
          "hold boundary (epoch 5->6) and s=1 (full berhu) by epoch 15; "
          "the first ~5 epochs match a plain sqrt run "
          "(berhu-anneal-schedule.md:213-221).")


def gate_gme_c(ctx):
  """GME-C: the EMA anneal smoke (average appears at the live point).

  With ema.anneal on, the banner reads
  "ema: horizon 3 epochs (beta -> 0.99...; anneal: hold 5 + 10 cosine;
  ...)"; the average's metrics first appear at the live point
  (epoch 6+, after the hold + warmup). Plus the golden no-anneal leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: ema-anneal-schedule.md:180-197.
  _golden_leg(ctx=ctx,
              gate_id="GME-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|ema)")
  _smoke_driver(ctx=ctx,
                config_key="GME-C-anneal",
                required_banners=["ema: horizon 3 epochs",
                                  "anneal: hold 5 + 10 cosine"])


def gate_item27(ctx):
  """item-27: a tight omegamh2 window run + the ci.init_probes A/B.

  One short training with a deliberately tight omegamh2 window runs end
  to end and the pool shrinkage matches the banner count (a nested
  param_cuts block shows the normal banner). The unexplained duplicate
  ci.init_probes call is the paired A/B inspection, resolved with
  workstation evidence.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: omegamh2-ns-product-cuts.md:125-126, param-cuts-nested-block.md:94-95.
  window_yaml = ctx.require_config("item-27-window")
  rc, out = ctx.run_driver(yaml_path=window_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="item-27 tight-window run completes",
             ok=(rc == 0),
             detail="run exit code " + str(rc))
  ctx.expect(label="item-27 pool shrinkage banner ('used N of P cut rows')",
             ok=logscan.search(text=out,
                               pattern=r"used\s+\d+\s+of\s+\d+\s+cut rows"),
             detail="the banner cut count must match the pool shrinkage")
  ctx.log("item-27 ci.init_probes A/B: the duplicate init_probes call in "
          "geometries_output.py is inspected with this run's evidence "
          "(omegamh2-ns-product-cuts.md:248); a manual A/B, not an "
          "automatable assertion.")


def gate_gt_b(ctx):
  """GT-B (optional): the synthetic four-window triangle shading.

  A synthetic-sample triangle with all four windows active fills
  exactly the coverage-table panels (asserted on the axes' artist
  lists), same rgba everywhere, plus the omh2-marginal band. Off the
  default sweep; runs only when --gate names it. Matplotlib + getdist,
  no cosmolike.
  """
  ctx.require_caps("torch")
  # home: triangle-cut-shading-all-windows.md:72-75.
  rc, out = ctx.run_check("gates/checks/gt_b_triangle.py")
  if ctx.dry:
    return
  ctx.expect(label="GT-B four-window triangle shading check exit 0",
             ok=(rc == 0),
             detail="check exit code " + str(rc)
                    + " (gates/checks/gt_b_triangle.py)")


# --------------------------------------------------------------------------
# This week's legs.
# --------------------------------------------------------------------------

def gate_gft_c(ctx):
  """GFT-C: freeze_trunk false joint phase 2 trains both stacks.

  A restrf + nla run with small trunk_epochs and freeze_trunk false:
  the startup says "(two-phase: N trunk + M joint)", phase 2 says
  "phase 'joint': ...", loss is continuous at the handoff, and the
  phase-2 epoch time sits visibly above a freeze_trunk true control
  (the trunk backward returned). Plus the golden absent-key leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: freeze-trunk-joint-phase2.md:115-120, :211-228.
  _golden_leg(ctx=ctx,
              gate_id="GFT-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|run:)")
  # the joint run (D-GR1: the trunk-count banner is matched by regex, so
  # the YAML uses a small trunk_epochs per the note, not a pinned 800).
  joint_yaml = ctx.require_config("GFT-C-joint")
  rc_j, out = ctx.run_driver(yaml_path=joint_yaml, allow_fail=True)
  # D-GH5e: RUN the freeze_trunk:true control (not just name it) and log
  # both phase-2 epoch times side by side; the visual comparison stays,
  # but the log must carry both numbers.
  control_yaml = ctx.require_config("GFT-C-control")
  rc_c, out_c = ctx.run_driver(yaml_path=control_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="GFT-C joint run completes",
             ok=(rc_j == 0),
             detail="joint exit code " + str(rc_j))
  ctx.expect(label="GFT-C two-phase banner (regex 'two-phase: \\d+ trunk')",
             ok=logscan.search(text=out, pattern=r"two-phase: \d+ trunk"),
             detail="D-GR1: the trunk count is matched by regex, not pinned")
  ctx.expect(label="GFT-C phase 'joint' banner",
             ok=logscan.contains(text=out, needle="phase 'joint'"),
             detail="phase 2 must announce the joint pass")
  ctx.expect(label="GFT-C freeze_trunk:true control run completes",
             ok=(rc_c == 0),
             detail="control exit code " + str(rc_c))
  joint_epochs = logscan.matching_lines(text=out, pattern=r"^epoch")
  control_epochs = logscan.matching_lines(text=out_c, pattern=r"^epoch")
  joint_last = joint_epochs[-1] if len(joint_epochs) > 0 else "(no epoch line)"
  control_last = (control_epochs[-1] if len(control_epochs) > 0
                  else "(no epoch line)")
  ctx.log("GFT-C phase-2 epoch time, side by side (the sanity signal is the "
          "joint time ABOVE the control, the trunk backward returned):")
  ctx.log("  joint (freeze_trunk:false):  " + joint_last)
  ctx.log("  control (freeze_trunk:true): " + control_last)
  ctx.log("loss continuous at the handoff (freeze-trunk-joint-phase2.md:"
          "226-228); the two numbers above are the Architect's visual check.")


def gate_gha_f(ctx):
  """GHA-F: a pinned-head gated_power activation run.

  restrf + trf.activation gated_power: the "model spec:" banner shows
  the trf activation dict, the head param count rises vs the shared H,
  the handoff stays loss-continuous; the flag-vs-pin warning prints
  when --activation power is passed, and freeze_trunk false with the
  pin errors in build_specs. Plus the golden no-pin leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: head-activation-per-component.md:239-242, :405-430.
  _golden_leg(ctx=ctx,
              gate_id="GHA-F",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best|model spec)")
  out = _smoke_driver(ctx=ctx,
                      config_key="GHA-F-pin",
                      required_banners=["gated_power"])
  # the flag-vs-pin warning leg: pass --activation power on the pinned YAML.
  pin_yaml = ctx.require_config("GHA-F-pin")
  rc_w, out_w = ctx.run_driver(yaml_path=pin_yaml,
                               extra=("--activation=power",),
                               allow_fail=True)
  # D-GH5b: the license-error leg (freeze_trunk false, or trunk_epochs 0)
  # WITH the head pin makes build_specs error with the frozen-trunk-head
  # message (head-activation-per-component.md:429-430). GHA-F-license is a
  # deliberately-invalid YAML.
  license_yaml = ctx.require_config("GHA-F-license")
  rc_l, out_l = ctx.run_driver(yaml_path=license_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(
    label="GHA-F flag-vs-pin warning",
    ok=logscan.contains(
      text=out_w,
      needle="the head keeps its model.trf.activation pin (gated_power)"),
    detail="the startup warning must state the pin wins over --activation")
  ctx.expect(
    label="GHA-F license error (freeze_trunk false + pin -> build_specs errors)",
    ok=(rc_l != 0 and logscan.search(text=out_l, pattern=r"(?i)frozen")),
    detail="build_specs must exit nonzero with the frozen-trunk-head message "
           "(rc " + str(rc_l) + ")")


def gate_gan_c(ctx):
  """GAN-C: relu/tanh with the norm knob (per_feature / affine).

  A tanh + per_feature run (banner shows norm; loss descends) and a
  tanh + affine run (the classic baseline). Plus the golden absent-key
  byte-identity leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: activation-families-norm-knob.md:99-101.
  _golden_leg(ctx=ctx,
              gate_id="GAN-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="GAN-C-tanh-perfeature",
                required_banners=["per_feature"])
  _smoke_driver(ctx=ctx,
                config_key="GAN-C-tanh-affine",
                required_banners=["affine"])


def gate_gwd_c(ctx):
  """GWD-C: the weight-decay census on a live optimizer.

  A gated_power model built with weight_decay 1e-4: the parameter-group
  census puts exactly the Linear / Conv1d / BinLinear weights in the
  decayed group and everything else (multigate w/beta/mu, BinLinear
  biases, all norms and biases) undecayed. Plus the golden wd-0
  byte-identity leg.
  """
  ctx.require_caps("torch", "gpu")
  # home: weight-decay-only-weight-matrices.md:143-147 (GWD-C census +
  # golden wd-0 byte-identity).
  rc, out = ctx.run_check("gates/checks/gwd_census.py")
  if not ctx.dry:
    ctx.expect(label="GWD-C param-group census (allowlist exact)",
               ok=(rc == 0),
               detail="census check exit code " + str(rc))
  _golden_leg(ctx=ctx,
              gate_id="GWD-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")


def gate_gpc_c(ctx):
  """GPC-C: NPCE residual + ratio + refit smoke.

  A small residual-form run and a ratio-form run: the fit report prints
  the kept modes, the refiner descends, and save -> the h5 pce group ->
  from_state rebuild matches base(theta) on a probe batch. The
  exclusivity errors fire from real YAMLs; a 2-point sweep_ntrain smoke
  refits the base per point. Plus the golden absent-pce leg.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: npce-yaml-wiring.md:117-122, :201-204.
  _golden_leg(ctx=ctx,
              gate_id="GPC-C",
              yaml_name="train_single_emulator_cosmic_shear.yaml",
              grep_pattern="^(phase|epoch|best)")
  _smoke_driver(ctx=ctx,
                config_key="GPC-C-residual",
                required_banners=["pce"])
  _smoke_driver(ctx=ctx,
                config_key="GPC-C-ratio",
                required_banners=["pce"])
  # D-GH5c: the exclusivity errors, each must error loudly
  # (npce-yaml-wiring.md:117-122). pce + ia is invalid via the YAML
  # alone; pce + rescale is a CLI exclusivity, so the excl-rescale leg
  # passes --rescale=residual (--rescale is a flag, never a YAML key).
  ia_yaml = ctx.require_config("GPC-C-excl-ia")
  rc_ia, out_ia = ctx.run_driver(yaml_path=ia_yaml, allow_fail=True)
  rs_yaml = ctx.require_config("GPC-C-excl-rescale")
  rc_rs, out_rs = ctx.run_driver(yaml_path=rs_yaml,
                                 extra=("--rescale=residual",),
                                 allow_fail=True)
  # the 2-point sweep_ntrain smoke: the base refits per point. D-GH7:
  # the sweep-over-n_train driver, with a 2-point geometric grid
  # (--n-min / --n-max / --n-points), not the single-train driver.
  sweep_yaml = ctx.require_config("GPC-C-sweep")
  rc_s, out_s = ctx.run_driver(
    yaml_path=sweep_yaml,
    driver=SWEEP_NTRAIN_DRIVER,
    extra=("--n-min=1000", "--n-max=2000", "--n-points=2"),
    allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(
    label="GPC-C exclusivity error (pce + ia)",
    ok=(rc_ia != 0 and logscan.search(text=out_ia, pattern=r"(?i)exclusive")),
    detail="pce + model.ia must exit nonzero with the exclusive message "
           "(rc " + str(rc_ia) + ")")
  ctx.expect(
    label="GPC-C exclusivity error (pce + --rescale)",
    ok=(rc_rs != 0 and logscan.search(text=out_rs, pattern=r"(?i)exclusive")),
    detail="pce + --rescale=residual must exit nonzero with the exclusive "
           "message (rc " + str(rc_rs) + ")")
  refits = logscan.matching_lines(text=out_s, pattern=r"(?i)pce|refit|kept")
  ctx.expect(
    label="GPC-C 2-point sweep_ntrain refits the base per point",
    ok=(rc_s == 0 and len(refits) >= 2),
    detail="the fit report should print once per sweep point (>=2 fit-report "
           "lines); saw " + str(len(refits)))
  ctx.log("GPC-C rebuild-vs-base probe: save -> the h5 pce group -> from_state "
          "rebuild == base(theta) on a probe batch belongs in the check-script "
          "set (GSV-C's NPCE save round-trips the pce group; a standalone GPC-C "
          "probe if wanted). Named in the remainder (npce-yaml-wiring.md:117).")


# --------------------------------------------------------------------------
# The save-to-sample acceptance chain (GSV-C's artifact feeds GCT-C).
# --------------------------------------------------------------------------

def gate_gsv_c(ctx):
  """GSV-C: the save-schema-v2 acceptance (bitwise + drift proof).

  Train small -> save -> rebuild_emulator -> outputs bitwise-equal to
  the live model on a probe batch; then the drift proof (monkeypatch a
  handful of code defaults, rebuild again, still bitwise-equal); one
  factored and one NPCE save so the geometry-class marker and the pce
  group both round-trip; a v1 file refused with the clear message.
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  # home: save-schema-resolved-config.md:86-93 (GSV-C: bitwise + the
  # drift test + v1 refusal); the one-factored + one-NPCE-save
  # requirement is workstation-board-2026-07.md:66-71 (gate 18) (D-GH4).
  rc, out = ctx.run_check("gates/checks/gsv_bitwise_drift.py")
  if ctx.dry:
    return
  ctx.expect(label="GSV-C save->rebuild bitwise + drift + v1-refusal",
             ok=(rc == 0),
             detail="check exit code " + str(rc)
                    + " (gates/checks/gsv_bitwise_drift.py)")


def gate_gct_c(ctx):
  """GCT-C: the cobaya adapter acceptance (parity + evaluate + MCMC).

  The parity probe (EmulatorPredictor vs the training-side eval on the
  same probe points, rtol 1e-6); the factored save -> rebuild ->
  predict round-trip; the example evaluate run against the lsst_y1
  likelihood (use_emulator 1) with the printed datavector compared to
  the training-side prediction; an MCMC evaluate + short-chain smoke.
  Depends on GSV-C (its saved artifact feeds the parity probe).
  """
  ctx.require_caps("torch", "cosmolike", "cobaya", "gpu")
  # home: cobaya-theory-adapter.md:117-123 (GCT-C), :234-238 (the real
  # factored round-trip added by D-CT1).
  rc, out = ctx.run_check("gates/checks/gct_parity.py")
  if not ctx.dry:
    ctx.expect(label="GCT-C parity probe (rtol 1e-6) + factored round-trip",
               ok=(rc == 0),
               detail="check exit code " + str(rc)
                      + " (gates/checks/gct_parity.py)")

  evaluate_yaml = ctx.evaluate_yaml()
  ctx.log("GCT-C evaluate: cobaya-run the example evaluate YAML against the "
          "lsst_y1 likelihood (use_emulator 1); the printed datavector is "
          "compared to the training-side prediction.")
  rc_ev, out_ev = ctx.sh(
    cmd=["cobaya-run", str(evaluate_yaml)],
    cwd=ctx.rootdir(),
    allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(label="GCT-C example evaluate run completes",
             ok=(rc_ev == 0),
             detail="cobaya-run exit code " + str(rc_ev))
  ctx.log("GCT-C MCMC smoke: a short-chain sampler run confirms the theory "
          "block drives an MCMC (cobaya-theory-adapter.md:123); run it with "
          "an mcmc sampler override once the evaluate leg is green.")


# --------------------------------------------------------------------------
# The board, in execution order (workstation-board-2026-07.md).
# --------------------------------------------------------------------------

BOARD = [
  Gate(id="GM-C",
       tier=TIER_STANDING,
       home="weight-ema-snapshot-coupled",
       maps="98-101 (byte-identity gate); 229-238 (the epoch-line diff recipe)",
       run=gate_gm_c,
       needs=("torch", "cosmolike", "gpu"),
       worktree_commit="46ec5e1"),
  Gate(id="GM-D",
       tier=TIER_STANDING,
       home="weight-ema-snapshot-coupled",
       maps="104-107, 240-251 (on-mode smoke: horizon banner + metrics); "
            "246-249 (the lr-cut rewind line)",
       run=gate_gm_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="DIAG",
       tier=TIER_STANDING,
       home="driver-audit-phase-sweep-guards",
       maps="G1 audit-package-style-2026-07-05.md:232-234; G-F "
            "omegamh2-ns-product-cuts.md:125-126; GN-F "
            "param-cuts-nested-block.md:94-95; GS-D "
            "n-train-n-val-absolute-counts.md:110-112; GT-C "
            "triangle-cut-shading-all-windows.md:76-79",
       run=gate_diag,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GP-D",
       tier=TIER_STANDING,
       home="resolve-phase-args-single-phase",
       maps="110-113 (single-phase demotion trains; two-phase control no-op)",
       run=gate_gp_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GH-E",
       tier=TIER_STANDING,
       home="phase-blocks-nested-lr-scheduler",
       maps="262-267 (head override banner + cadence); 269-279 (golden diff)",
       run=gate_gh_e,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GE-C",
       tier=TIER_STANDING,
       home="eval-bs-decoupling",
       maps="102-108 (partition invariance rtol 1e-6 + timing); 202-300 script",
       run=gate_ge_c,
       needs=("torch", "gpu")),
  Gate(id="GB-C",
       tier=TIER_STANDING,
       home="loss-mode-berhu",
       maps="148-153 (leg 1: berhu/_reduce numerics + autograd continuity); "
            "290-314 (leg 2: golden + head-berhu banners)",
       run=gate_gb_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GL-D",
       tier=TIER_STANDING,
       home="loss-block-nesting",
       maps="237-244 (new-schema reproduces pre-change epoch lines)",
       run=gate_gl_d,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GBA-C",
       tier=TIER_STANDING,
       home="berhu-anneal-schedule",
       maps="199-221 (golden no-anneal + anneal banner + continuity + s=1)",
       run=gate_gba_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GME-C",
       tier=TIER_STANDING,
       home="ema-anneal-schedule",
       maps="180-197 (golden no-anneal + anneal banner + live-point metrics)",
       run=gate_gme_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="item-27",
       tier=TIER_STANDING,
       home="omegamh2-ns-product-cuts",
       maps="125-126 (tight window: pool shrinkage matches count); "
            "param-cuts-nested-block.md:94-95 (nested block normal banner); "
            "248 (ci.init_probes A/B inspection)",
       run=gate_item27,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GT-B",
       tier=TIER_STANDING,
       home="triangle-cut-shading-all-windows",
       maps="72-75 (synthetic four-window triangle: artist-list fills + band)",
       run=gate_gt_b,
       optional=True,
       needs=("torch",)),

  Gate(id="GFT-C",
       tier=TIER_WEEK,
       home="freeze-trunk-joint-phase2",
       maps="115-120, 211-228 (joint banners + continuity + epoch-time signal)",
       run=gate_gft_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GHA-F",
       tier=TIER_WEEK,
       home="head-activation-per-component",
       maps="239-242, 405-430 (model-spec banner + param count + warning); "
            "429-430 (leg 4: freeze_trunk false + pin -> build_specs errors)",
       run=gate_gha_f,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GAN-C",
       tier=TIER_WEEK,
       home="activation-families-norm-knob",
       maps="99-101 (tanh+per_feature + tanh+affine + golden absent-key)",
       run=gate_gan_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GWD-C",
       tier=TIER_WEEK,
       home="weight-decay-only-weight-matrices",
       maps="143-147 (gated_power wd 1e-4 census + golden wd-0 byte-identity)",
       run=gate_gwd_c,
       needs=("torch", "gpu")),
  Gate(id="GPC-C",
       tier=TIER_WEEK,
       home="npce-yaml-wiring",
       maps="117-122, 201-204 (residual + ratio + rebuild + exclusivity + sweep)",
       run=gate_gpc_c,
       needs=("torch", "cosmolike", "gpu")),

  Gate(id="GSV-C",
       tier=TIER_SAVE_SAMPLE,
       home="save-schema-resolved-config",
       maps="86-93 (bitwise + drift + v1 refusal); "
            "workstation-board-2026-07.md:66-71 (one factored + one NPCE save)",
       run=gate_gsv_c,
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="GCT-C",
       tier=TIER_SAVE_SAMPLE,
       home="cobaya-theory-adapter",
       maps="117-123 (parity rtol 1e-6 + evaluate + MCMC); 234-238 (round-trip)",
       run=gate_gct_c,
       deps=("GSV-C",),
       needs=("torch", "cosmolike", "cobaya", "gpu")),
]

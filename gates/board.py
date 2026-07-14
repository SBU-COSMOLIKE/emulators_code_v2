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
  assertion = one acceptance leg named by a stable id and paired with a
              note anchor (the Assertion class); the id names the leg, the
              anchor points at the note passage that proves it.
  evidence  = a gate's structured evidence map (Gate.evidence): the tuple
              of assertions the gate is built to prove. The runner checks,
              before any test runs, that every anchor resolves in notes/
              and no two assertions share an id.

How a gate teaches its evidence.
  A PASS is only worth trusting if a reviewer can re-derive it from what
  the run recorded, without rerunning it. Four records make a gate's
  verdict legible, and each answers a different question:

    what the gate claims to prove -- the maps= line (prose) and the
      structured evidence map (assertion id -> note anchor). The runner
      validates the anchors, so the claim cannot drift from the note.
    what the run actually observed -- every ctx.expect writes a CHECK line
      carrying not a bare PASS/FAIL but the acceptance value behind it (a
      number, a count, a byte-identity result), so the log shows the
      measurement, not a memory of it.
    what code produced it -- the immutable per-attempt log names the home
      note, the base-notes commit, and HEAD at run; the resume record
      stores the executable-surface digest and the input digest, so a
      stored PASS is trusted only while both are unchanged.
    whether it can be believed now -- --list reports each gate as current
      PASS, stale-code, stale-input, or interrupted, so a PASS that no
      longer matches the tree reads as stale rather than green.

  Read together, these let a reviewer confirm the gate encodes its note
  (not a memory of it), see the values it measured, and know the verdict
  still describes the current tree.
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


# The cosmic-shear driver gates all rebuild an artifact through the model-recipe
# (results.py / warmstart.py dynamic imports), so each declares the training
# driver plus the design and loss trees; npce-training additionally declares the
# sweep driver. Shared here so the 1b manifest closure is one reviewed tuple.
_CS_TRAIN_CODE = ("cosmic_shear_train_emulator.py",
                  "emulator/designs", "emulator/losses")

# The deploy-data fixture keys every cosmic-shear driver gate consumes, resolved
# against board_config.json's deploy_data block. Shared so gates that share a
# fixture share the key (no deploy path appears twice); a gate's manifest inputs
# are its own gate_configs key(s) + these.
_CS_DEPLOY_DATA = ("deploy_data.w0wa_takahashi_dv_train_cs16",
                   "deploy_data.w0wa_takahashi_params_train_cs16",
                   "deploy_data.w0wa_takahashi_covmat_cs16",
                   "deploy_data.w0wa_takahashi_dv_val_cs8",
                   "deploy_data.w0wa_takahashi_params_val_cs8",
                   "deploy_data.lsst_y1_ggl_dataset")


@dataclass(frozen=True)
class Assertion:
  """One acceptance leg's stable id and the note anchor that proves it.

  The board's trust problem: a gate's free-form maps= prose claims to
  encode a home-note passage, but nothing checks the claim, so a
  reworded or deleted note silently orphans the pointer (only a
  handful of the board's tests currently name a note passage the note
  actually still carries). An Assertion makes the pointer machine
  checkable. Before any gate runs, the runner verifies that every
  anchor resolves to a real, explicitly-declared marker in notes/, and
  that no two assertions anywhere on the board share an id -- so a
  review can trust that the gate encodes the note, not a memory of it.

  Arguments:
    aid    = the assertion id: a stable, board-unique name for one
             acceptance leg, "<gate-id>.<plain-leg-name>" where <gate-id>
             is the gate's board id (the --gate selector), e.g.
             "board-selftest.exit-truth". Chosen once and never reworded,
             so a log line or a review can cite the leg by a name that
             does not move when the prose around it does -- and a red aid
             line names both the gate to rerun and the leg that failed.
    anchor = the home-note anchor the leg encodes, in the form
             "<note>.md#<marker>", where the marker is the aid with
             "." -> "-" (e.g. "gates-and-board.md#board-selftest-exit-truth"
             for aid "board-selftest.exit-truth"). validate_evidence
             enforces that transform, so the aid <-> anchor map is one
             mechanical string rule. The marker is an explicit
             <a id="..."></a> element in the note (chosen over a heading
             slug because it survives a heading rewording); the runner
             fails loudly, before running, if it does not resolve.
  """
  aid: str
  anchor: str


@dataclass(frozen=True)
class Manifest:
  """A gate's declared executable / input dependency roots (queue 1b).

  A resume digest is only evidence if its membership is inspectable. A gate
  declares the ROOTS of its dependency graph; the runner derives the transitive
  repo-local closure, hashes it, and persists the resolved members. The
  declaration states intent (and stays readable across refactors); the
  persisted closure is the whole truth. This is phase 1: the field and its
  static validation (validate_manifests in run_board.py). The digest rewrite
  that consumes it and the per-gate population are later phases; a gate with no
  Manifest keeps the conservative dual-digest fallback until then.

  Arguments:
    code   = repo-relative production modules this gate's checks depend on
             BEYOND the always-hashed shared harness (run_board.py, board.py)
             and the gates/checks/*.py scripts the gate body names (both added
             automatically). A subprocess-invoked driver named in the gate body
             is declared here too, since it appears in no check script's import
             graph; declaring it lets the literal-path census cover it and its
             imports join the closure. Each path lives inside the executable
             surface. Only the ROOTS are declared: the deriver walks the rest.
    inputs = board_config keys (dotted) whose resolved values are the specific
             external files this gate consumes (its smoke YAML, data, covmat,
             axis, artifact inputs), resolved against board_config at run time
             so a raw deploy path is never baked into the registry.
  """
  code:   Tuple[str, ...] = ()
  inputs: Tuple[str, ...] = ()


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
    evidence= the structured evidence map: the acceptance legs this test
              is built to prove, each an Assertion pairing a stable,
              board-unique id with the home-note anchor it encodes. The
              runner validates it statically before any gate runs (every
              anchor must resolve to a declared marker in notes/, every id
              must be unique), so the maps= prose can never quietly drift
              from the note it cites. Empty on tests not yet migrated to
              the structured map -- their free-form maps= line still
              documents them; the migration is rolling, not a flag day.
    title   = a one-line human name for the test (for the README table).
    manifest= the executable / input dependency roots this test declares (a
              Manifest, or None until it is populated). Validated statically by
              validate_manifests before any gate runs; None keeps the
              conservative dual-digest fallback (queue 1b, rolling migration).
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
  evidence: Tuple[Assertion, ...] = ()
  title: str = ""
  manifest: Manifest = None


# --------------------------------------------------------------------------
# Shared gate bodies. The golden byte-identity leg and the plain driver
# smoke are common enough to factor; the per-gate functions below call
# them with the gate's own config keys and acceptance substrings.
# --------------------------------------------------------------------------

def _golden_leg(ctx, gate_id, grep_pattern, *, yaml_name=None,
                config_key=None, aid=None):
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
    aid          = the structured-evidence assertion id this golden leg
                   proves (queue 2), or None for a gate not yet migrated
                   to the map. When given, a configured base emits the
                   byte-identity expect under this aid; a NULL base emits
                   an explicit UNAVAILABLE under it (not a silent drop),
                   so the leg the gate declared is reconciled every run.
  """
  base = ctx.golden_base(gate_id)
  if base is None:
    if aid is not None:
      # the gate DECLARED this leg: a null base is the honest UNAVAILABLE
      # terminal (fork D1-ii), not a silent skip that reds reconciliation.
      ctx.unavailable(aid=aid,
                      label=gate_id + " golden byte-identity",
                      reason="no base commit in board_config.json "
                             "golden_bases['" + gate_id + "']; the golden "
                             "byte-identity leg needs a configured historical "
                             "base (schema equivalence is unproven here)")
      return
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
    cur_rc, cur = ctx.run_driver(yaml_path=bare)
    with ctx.worktree(commit=base) as wt:
      pre_rc, pre = ctx.run_driver(yaml_path=bare, cwd=wt)

  if ctx.dry:
    return

  # strip the trailing wall-clock column (e.g. "  2.3s"): the one machine-
  # noise field on an otherwise deterministic epoch line. Applies to every
  # golden leg, not just ema-off-identity.
  equal, detail = logscan.byte_identity(text_a=pre,
                                        text_b=cur,
                                        pattern=grep_pattern,
                                        strip=r"[ \t]+\d+(?:\.\d+)?s$")
  # A golden proof is evidence only when BOTH children COMPLETED (rc 0) AND the
  # compared selection is NON-EMPTY. The pre-46 leg discarded both child return
  # codes (``_, cur`` / ``_, pre``) and compared whatever the pattern selected,
  # so a child that crashed after its last matching line -- or a pattern that
  # matched nothing on both sides -- passed byte-identity vacuously. Require
  # clean rcs and a non-empty selection, and report both rcs + both selected
  # counts beside the equality verdict.
  n_pre = len(logscan.matching_lines(text=pre, pattern=grep_pattern))
  n_cur = len(logscan.matching_lines(text=cur, pattern=grep_pattern))
  reasons = []
  if pre_rc != 0 or cur_rc != 0:
    reasons.append("a child exited nonzero (a golden run must complete)")
  if n_pre == 0 or n_cur == 0:
    reasons.append("empty selection (the pattern matched no lines to compare)")
  if not equal:
    reasons.append(detail)
  ok = (len(reasons) == 0)
  status = ("rc pre=" + str(pre_rc) + " cur=" + str(cur_rc)
            + "; selected pre=" + str(n_pre) + " cur=" + str(n_cur))
  ctx.expect(aid=aid,
             label=gate_id + " golden byte-identity (" + base + " vs tip)",
             ok=ok,
             detail=status + ("" if ok else "; " + "; ".join(reasons)))


def _smoke_driver(ctx, config_key, required_banners, *, extra=(),
                  exit_aid=None, banner_aid=None):
  """Run one training smoke and require its banners in the output.

  Arguments:
    ctx              = the per-test helper.
    config_key       = the board_config.json gate_configs key naming
                       the smoke YAML (unset or missing = GateFailure).
    required_banners = the literal banner substrings that must all
                       appear (quoted verbatim from the home note).
    extra            = extra driver flags (e.g. ("--activation=power",)).
    exit_aid         = the structured-evidence assertion id for the
                       exit-zero leg (queue 2), or None for a gate not
                       yet migrated to the map.
    banner_aid       = the structured-evidence assertion id for the
                       banner-presence leg (queue 2), or None. This
                       helper emits TWO legs, so a migrated caller names
                       both aids; an unmigrated caller leaves both None.

  Returns:
    the captured run output, for further checks by the caller.
  """
  yaml_path = ctx.require_config(config_key)
  rc, out = ctx.run_driver(yaml_path=yaml_path, extra=extra, allow_fail=True)

  if ctx.dry:
    return out

  ctx.expect(aid=exit_aid,
             label=config_key + " run completes (rc 0)",
             ok=(rc == 0),
             detail="driver exit code " + str(rc))
  ok, missing = logscan.contains_all(text=out, needles=required_banners)
  ctx.expect(aid=banner_aid,
             label=config_key + " banners present",
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
              grep_pattern="^(epoch|best epoch)",
              aid="ema-off-identity.golden-selected-text-equality")


def gate_gm_d(ctx):
  """ema-smoke: EMA switched on works.

  WHAT: a short bs=64 run with ema.horizon_epochs=3. WHY: the identity
  test proves EMA off is harmless, not that EMA on works. HOW: the smoke
  exits zero (driver-exit-zero), its banner reads "ema: horizon 3 epochs"
  (horizon-banner-present), and a plateau lr cut prints "rewound to best
  epoch" (rewind-line-present)
  (spec: training-stack.md#ema-smoke-evidence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  out = _smoke_driver(ctx=ctx,
                      config_key="ema-smoke-config",
                      required_banners=["ema: horizon 3 epochs"],
                      exit_aid="ema-smoke.driver-exit-zero",
                      banner_aid="ema-smoke.horizon-banner-present")
  if ctx.dry:
    return
  ctx.expect(aid="ema-smoke.rewind-line-present",
             label="ema-smoke rewind line ('lr cut -> rewound to best epoch')",
             ok=logscan.search(text=out, pattern=r"rewound to best epoch"),
             detail="training-stack.md:246-249: a rewind fires")


def gate_diag(ctx):
  """production-diagnostic: one --diagnostic run that closes five checks.

  WHAT: a production training run with the diagnostics PDF, exercising
  the dead-class census, a tight density-window cut, a nested param_cuts
  block, the absolute row counts, and the shaded triangle at once. WHY:
  all five ride one ordinary run. HOW: seven declared legs.
  retired-class-name-census: the literal NLATemplateMLP|NLAInputGeometry
  search over repository *.py files (gates/, .git/ excluded) returns zero
  hits. package-import: importing emulator, emulator.designs, and
  emulator.losses exits zero. driver-exit-zero: the --diagnostic training
  subprocess exits zero. sizes-banner: the stream carries a line matching
  "used N of P cut rows" (shape, not integer truth). The final three legs
  are UNAVAILABLE on this wrapper: cut-row-selection (the gate computes no
  independent expected-row count and does not assert its YAML is the
  intended tight-window fixture), diagnostics-pdf (the run requests the PDF
  but the gate neither confirms the file exists nor reads it back), and
  triangle-shading (the gate prints a visual-inspection instruction but
  compares no plotted artists). Spec:
  training-stack.md#production-diagnostic-evidence.
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
  ctx.expect(aid="production-diagnostic.retired-class-name-census",
             label="dead-class census -> 0 hits",
             ok=(rc_grep == 1 and out_grep.strip() == ""),
             detail="grep rc " + str(rc_grep) + ", output: "
                    + repr(out_grep.strip()[:200]))
  ctx.expect(aid="production-diagnostic.package-import",
             label="clean package import",
             ok=(rc_imp == 0),
             detail="import rc " + str(rc_imp))
  # the window-cut / row-count / sizes banners (home lines named in `maps`).
  ctx.expect(aid="production-diagnostic.driver-exit-zero",
             label="production-diagnostic production run completes",
             ok=(rc_run == 0),
             detail="driver exit code " + str(rc_run))
  ctx.expect(aid="production-diagnostic.sizes-banner",
             label="sizes line ('used N of P cut rows')",
             ok=logscan.search(text=out_run,
                               pattern=r"used\s+\d+\s+of\s+\d+\s+cut rows"),
             detail="the sizes line must report used N of P cut rows")
  # the last three declared legs are honestly UNAVAILABLE on this wrapper:
  # the run exercises the machinery, but the gate reads back none of the
  # diagnostic content, so none may go green (binding ruling 6 / fork D1-ii).
  ctx.unavailable(aid="production-diagnostic.cut-row-selection",
                  label="cut-row selection (independent expected-row count)",
                  reason="the gate computes no independent expected retained-row "
                         "count and does not assert its configured YAML is the "
                         "intended tight-window fixture, so the sizes banner's "
                         "integers are unchecked for truth")
  ctx.unavailable(aid="production-diagnostic.diagnostics-pdf",
                  label="diagnostics PDF exists and reads back",
                  reason="the run requests --diagnostic=gates_diag but the gate "
                         "neither confirms the PDF file exists nor reads it back")
  ctx.log("shaded triangle: the diagnostics PDF is a VISUAL check (the omh2 marginal "
          "at 0.20 and the (ns, omh2) diagonal corner at 0.17 must show "
          "adjoining grey); the harness confirms the run produced it, the "
          "Architect confirms the shading from the committed PDF/log.")
  ctx.unavailable(aid="production-diagnostic.triangle-shading",
                  label="shaded-triangle artist comparison",
                  reason="the gate prints a visual-inspection instruction for the "
                         "omh2 marginal and the (ns, omh2) corner shading but "
                         "compares no plotted artists programmatically")


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

  ctx.expect(aid="single-phase-demotion.single-phase-exit-zero",
             label="single-phase-demotion single-phase resmlp trains (was a traceback)",
             ok=(rc_s == 0),
             detail="resmlp run exit code " + str(rc_s))
  ctx.expect(aid="single-phase-demotion.demotion-text-present",
             label="single-phase-demotion demotion notice in the banner",
             ok=logscan.search(text=out_s,
                               pattern=r"(single-phase|demot|resolve)"),
             detail="stream matches single-phase|demot|resolve "
                    "(broad presence, not an exact notice string)")
  ctx.expect(aid="single-phase-demotion.two-phase-control-exit-zero",
             label="single-phase-demotion control rescnn+nla reproduces today (no-op)",
             ok=(rc_c == 0),
             detail="control run exit code " + str(rc_c))


def gate_gh_e(ctx):
  """head-scheduler-override: the head phase cuts the lr on its own patience.

  WHAT: a head: scheduler block with patience 10 against a run default
  of 25. WHY: the override must act on that phase only. HOW: the override
  smoke exits zero and prints the "[head overrides: scheduler]" banner; the
  golden no-phase-blocks selected-text equality is UNAVAILABLE while
  golden_bases has no configured base, and the patience-10 lr-cut cadence is
  UNAVAILABLE (logged instruction only, no cadence comparison)
  (spec: training-stack.md:262-279).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="head-scheduler-override",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)",
              aid="head-scheduler-override.golden-selected-text-equality")
  out = _smoke_driver(ctx=ctx,
                      config_key="head-scheduler-override-config",
                      required_banners=["[head overrides: scheduler]"],
                      exit_aid="head-scheduler-override.driver-exit-zero",
                      banner_aid="head-scheduler-override.override-banner-present")
  if ctx.dry:
    return
  ctx.unavailable(aid="head-scheduler-override.lr-cut-cadence",
                  label="head-scheduler-override lr-cut cadence",
                  reason="the head phase's first lr cut should land on the "
                         "patience-10 cadence (vs 25), but the gate only prints "
                         "an instruction to inspect the lr-cut epoch spacing "
                         "(training-stack.md:265) and runs no cadence comparison")


def gate_ge_c(ctx):
  """eval-batch-invariance: validation metrics do not depend on chunking.

  WHAT: the eval batch size, decoupled from the training batch. WHY:
  changing how the eval set is chunked must not move any metric. HOW: a
  torch-only script emits four per-leg ##AID terminals (partition-invariance:
  real eval_val's aggregate median/mean/fractions and a per-row loop agree
  across eval batch sizes 32/517/1000/2048 to rtol 1e-6; ordinary-median: the
  helper and real eval_val return the arithmetic midpoint for an even sample,
  stay batch-invariant, and catch the lower-middle Tensor.median mutation;
  cuda-timing and production-timing-claim: informational, so UNAVAILABLE on
  this run -- CUDA durations carry no acceptance bound and the production
  speedup is a documented sentence, not a measurement)
  (spec: training-stack.md#eval-batch-invariance-evidence).

  The check-script rc expect stays aid-less: the child exit is the aggregate
  verdict of the two logical legs, and its four ##AID lines carry the declared
  evidence (the geo-paths check-script template).
  """
  ctx.require_caps("torch", "gpu")
  rc, out = ctx.run_check("gates/checks/ge_c_eval_bs.py")
  if ctx.dry:
    return
  ctx.expect(label="eval-batch-invariance check script exit 0",
             ok=(rc == 0),
             detail="check exit code " + str(rc)
                    + " (gates/checks/ge_c_eval_bs.py)")


def gate_gb_c(ctx):
  """berhu-loss: the berHu head loss trains beside a plain-sqrt trunk.

  WHAT: the berHu loss (a robust sqrt below a knot, capped above a cap)
  as a head-only loss under the nested loss schema. WHY: the two loss
  blocks must resolve independently per phase. HOW: a torch-only script
  checks the berHu numerics and emits three per-leg ##AID terminals
  (reference-values: values match the piecewise analytic reference;
  join-derivatives: the autograd slopes match the analytic derivatives at
  the t1 and t2 joins; anneal-endpoints: the blend is plain sqrt at s = 0
  and full berHu at s = 1). Then a run shows "loss_mode sqrt" on the trunk
  and "loss_mode berhu_capped (knot 0.2, cap 10)" on the head (smoke-exit-
  zero + loss-banners), plus a golden non-berhu run (golden-selected-text-
  equality, UNAVAILABLE while its base is null)
  (spec: training-stack.md#berhu-loss-evidence).

  The check-script rc expect stays aid-less: the child exit is the aggregate
  verdict, and its three ##AID lines carry the numerics legs (the geo-paths
  check-script template).
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
              grep_pattern="^(phase|epoch|best)",
              aid="berhu-loss.golden-selected-text-equality")
  _smoke_driver(ctx=ctx,
                config_key="berhu-loss-config",
                required_banners=["loss_mode sqrt",
                                  "loss_mode berhu_capped (knot 0.2, cap 10)"],
                exit_aid="berhu-loss.smoke-exit-zero",
                banner_aid="berhu-loss.loss-banners")


def gate_gl_d(ctx):
  """loss-schema-equivalence: the new loss schema changes config, not physics.

  WHAT: the nested loss: block that replaced the old flat keys. WHY: a
  config-layer rename should reproduce the old run exactly. HOW: the
  nested-loss smoke exits zero and prints the berHu-capped banner
  (reusing the head-berhu config as the production shape); the golden
  byte-identity leg would match the pre-change epoch lines to the
  character, but it is UNAVAILABLE while golden_bases has no historical
  base configured, so schema equivalence is not yet proven here
  (spec: training-stack.md:237-244).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="loss-schema-equivalence",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)",
              aid="loss-schema-equivalence.golden-selected-text-equality")
  _smoke_driver(ctx=ctx,
                config_key="berhu-loss-config",
                required_banners=["loss_mode berhu_capped (knot 0.2, cap 10)"],
                exit_aid="loss-schema-equivalence.smoke-exit-zero",
                banner_aid="loss-schema-equivalence.berhu-banner")


def gate_gba_c(ctx):
  """berhu-anneal: the berHu shape ramps in smoothly and late.

  WHAT: the berHu anneal (plain sqrt blending into berHu over a
  hold-then-ramp schedule). WHY: the tail votes should arrive after the
  trim schedule has absorbed the worst outliers, without a jolt at the
  ramp start. HOW: a run shows the banner "anneal: hold 5 + 10 cosine"
  (smoke-exit-zero + anneal-banner), plus a golden no-anneal run
  (golden-selected-text-equality, UNAVAILABLE while its base is null).
  The hold-boundary continuity and full-berHu-by-epoch-15 shape are named
  by the note but not measured here, so schedule-behavior is UNAVAILABLE
  (logged-only, no comparison runs)
  (spec: training-stack.md#berhu-anneal-evidence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="berhu-anneal",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)",
              aid="berhu-anneal.golden-selected-text-equality")
  _smoke_driver(ctx=ctx,
                config_key="berhu-anneal-config",
                required_banners=["anneal: hold 5 + 10 cosine"],
                exit_aid="berhu-anneal.smoke-exit-zero",
                banner_aid="berhu-anneal.anneal-banner")
  ctx.unavailable(
    aid="berhu-anneal.schedule-behavior",
    label="berhu-anneal schedule continuity + full-shape epoch",
    reason="the note names hold-boundary continuity (epoch 5->6) and "
           "s=1 (full berHu) by epoch 15, but this gate runs no such "
           "comparison -- the schedule inspection is logged-only, not "
           "measured (training-stack.md#berhu-anneal-schedule-behavior)")


def gate_gme_c(ctx):
  """ema-anneal: the EMA average wakes up only after the bad early era.

  WHAT: the EMA anneal (the averaging window grows from zero over a
  hold-then-ramp schedule). WHY: averaging through the high-loss early
  epochs would poison the shipped weights. HOW: the smoke exits zero
  (smoke-exit-zero) and its banners name the horizon and the schedule --
  both "ema: horizon 3 epochs" and "anneal: hold 5 + 10 cosine"
  (ema-anneal-banners); the golden no-anneal selected-text equality
  (golden-selected-text-equality) is UNAVAILABLE while golden_bases has
  no configured base; and the average's first-live-point metrics
  (live-point-metrics) are UNAVAILABLE -- described in the note (epoch
  6+), but this gate parses and asserts no metric-appearance comparison
  (spec: training-stack.md#ema-anneal-evidence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="ema-anneal",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best|ema)",
              aid="ema-anneal.golden-selected-text-equality")
  _smoke_driver(ctx=ctx,
                config_key="ema-anneal-config",
                required_banners=["ema: horizon 3 epochs",
                                  "anneal: hold 5 + 10 cosine"],
                exit_aid="ema-anneal.smoke-exit-zero",
                banner_aid="ema-anneal.ema-anneal-banners")
  ctx.unavailable(aid="ema-anneal.live-point-metrics",
                  label="ema-anneal averaged-metric first-live epoch",
                  reason="the note describes the averaged metrics first "
                         "appearing at the live point (epoch 6+), but this "
                         "gate parses no metrics and runs no metric-appearance "
                         "comparison (training-stack.md#ema-anneal-live-point-metrics)")


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
  ctx.expect(aid="param-window-cuts.driver-exit-zero",
             label="param-window-cuts tight-window run completes",
             ok=(rc == 0),
             detail="run exit code " + str(rc))
  ctx.expect(aid="param-window-cuts.cut-count-banner-present",
             label="param-window-cuts pool shrinkage banner ('used N of P cut rows')",
             ok=logscan.search(text=out,
                               pattern=r"used\s+\d+\s+of\s+\d+\s+cut rows"),
             detail="stream carries one 'used N of P cut rows' line "
                    "(broad presence, not a compared-count claim)")
  ctx.unavailable(aid="param-window-cuts.init-probes-inspection",
                  label="param-window-cuts ci.init_probes A/B inspection",
                  reason="the duplicate init_probes call in the geometries "
                         "output module is a manual A/B eye check "
                         "(data-generation-and-cuts.md:248); the gate prints "
                         "the inspection instruction and runs no executable "
                         "comparison")


def gate_gt_b(ctx):
  """triangle-shading: each physical cut shades its exact parameter panel.

  WHAT: the cut-window shading on the corner plot with all four density
  windows active. WHY: a wrongly shaded panel misleads a reader about
  which region was cut. HOW: a synthetic triangle must match an independent
  panel-and-window reference, use one color for every cut artist and place
  both excluded intervals on the omegamh2 diagonal. The child also moves one
  real artist to a wrong panel and requires the exact-owner check to reject
  that mutation. The gate is optional, runs only when --gate names it and
  needs no CosmoLike data (spec: data-generation-and-cuts.md).
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
  frozen. HOW: seven legs. The joint freeze_trunk:false run exits zero
  (joint-exit-zero), announces "two-phase: N trunk" (two-phase-banner, N
  matched by regex not pinned) and "phase 'joint'" (joint-phase-banner);
  the freeze_trunk:true control run exits zero (control-exit-zero). The
  golden absent-key selected-text equality (golden-selected-text-equality)
  is UNAVAILABLE while golden_bases has no configured base. The phase-2
  epoch-time ordering (epoch-time-order) and the handoff loss continuity
  (handoff-loss-continuity) are UNAVAILABLE: both are printed for
  inspection only, with no ordering or continuity comparison run
  (spec: training-stack.md#joint-training-evidence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="joint-training",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best|run:)",
              aid="joint-training.golden-selected-text-equality")
  joint_yaml = ctx.require_config("joint-training-config")
  rc_j, out = ctx.run_driver(yaml_path=joint_yaml, allow_fail=True)
  # run the control too, so the log carries both epoch times.
  control_yaml = ctx.require_config("joint-training-control")
  rc_c, out_c = ctx.run_driver(yaml_path=control_yaml, allow_fail=True)
  if ctx.dry:
    return
  ctx.expect(aid="joint-training.joint-exit-zero",
             label="joint-training joint run completes",
             ok=(rc_j == 0),
             detail="joint exit code " + str(rc_j))
  ctx.expect(aid="joint-training.two-phase-banner",
             label="joint-training two-phase banner (regex 'two-phase: \\d+ trunk')",
             ok=logscan.search(text=out, pattern=r"two-phase: \d+ trunk"),
             detail="the trunk count is matched by regex, not pinned")
  ctx.expect(aid="joint-training.joint-phase-banner",
             label="joint-training phase 'joint' banner",
             ok=logscan.contains(text=out, needle="phase 'joint'"),
             detail="phase 2 must announce the joint pass")
  ctx.expect(aid="joint-training.control-exit-zero",
             label="joint-training freeze_trunk:true control run completes",
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
  ctx.unavailable(aid="joint-training.epoch-time-order",
                  label="joint-training phase-2 epoch-time order (joint > control)",
                  reason="the joint phase-2 epoch time should sit visibly above "
                         "the freeze_trunk:true control (the trunk backward ran), "
                         "but the gate only prints the two last epoch lines "
                         "(training-stack.md:211-228) and runs no ordering "
                         "comparison")
  ctx.unavailable(aid="joint-training.handoff-loss-continuity",
                  label="joint-training handoff loss continuity",
                  reason="loss continuity across the phase-1->phase-2 handoff "
                         "(training-stack.md:226-228) is an inspection "
                         "instruction, not a numerical assertion in this gate")


def gate_gha_f(ctx):
  """head-activation-pin: the phase-2 head can pin its own activation.

  WHAT: a model.trf.activation pin (gated_power) for the frozen-trunk
  head. WHY: the pin must win over the --activation flag with a warning,
  and an illegal pin (unfrozen trunk) must error, not misbuild. HOW: the
  pinned-head smoke exits zero and prints the gated_power text; --activation
  power prints the flag-vs-pin warning; the deliberately-invalid license YAML
  makes build_specs exit with the frozen-trunk message; plus the golden
  no-pin run, UNAVAILABLE while golden_bases has no configured base
  (spec: models-and-designs.md#head-activation-pin-evidence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="head-activation-pin",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best|model spec)",
              aid="head-activation-pin.golden-selected-text-equality")
  out = _smoke_driver(ctx=ctx,
                      config_key="head-activation-pin-config",
                      required_banners=["gated_power"],
                      exit_aid="head-activation-pin.pinned-config-exit-zero",
                      banner_aid="head-activation-pin.gated-power-text-present")
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
    aid="head-activation-pin.flag-vs-pin-warning",
    label="head-activation-pin flag-vs-pin warning",
    ok=(rc_w == 0 and logscan.contains(
      text=out_w,
      needle="the head keeps its model.trf.activation pin (gated_power)")),
    detail="the flag-vs-pin run must succeed (rc " + str(rc_w) + ") AND print "
           "the startup warning that the pin wins over --activation; a warning "
           "printed on a failed run does not count")
  ctx.expect(
    aid="head-activation-pin.unfrozen-pin-refusal",
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
  per_feature run and a tanh + affine run each exit zero and name their
  norm in the banner; plus the golden absent-key run. The runs' epoch
  histories are printed but no assertion compares their loss values, so
  loss descent is logged-only, not asserted evidence
  (spec: models-and-designs.md#relu-tanh-norm-evidence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="relu-tanh-norm",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)",
              aid="relu-tanh-norm.golden-selected-text-equality")
  _smoke_driver(ctx=ctx,
                config_key="relu-tanh-norm-per-feature",
                required_banners=["per_feature"],
                exit_aid="relu-tanh-norm.per-feature-config-exit-zero",
                banner_aid="relu-tanh-norm.per-feature-text-present")
  _smoke_driver(ctx=ctx,
                config_key="relu-tanh-norm-affine",
                required_banners=["affine"],
                exit_aid="relu-tanh-norm.affine-config-exit-zero",
                banner_aid="relu-tanh-norm.affine-text-present")


def gate_gwd_c(ctx):
  """weight-decay-census: weight decay touches only true weight matrices.

  WHAT: the rule that picks decayed parameters by module role, not
  tensor shape. WHY: decaying an activation's parameters or a bias
  drags the model toward degenerate forms. HOW: a torch-only child runs
  the real make_optimizer at weight_decay 1e-4 over a toy module tree and
  emits four per-leg ##AID terminals (allowed-weight-set: the decayed group
  is exactly the Linear / Conv1d / BinLinear weights; undecayed-role-
  exclusions: the gated_power shape parameters, the BinLinear bias, and
  every other non-allowlisted parameter stay undecayed; parameter-group-
  partition: the two groups are disjoint and their union is every parameter
  exactly once; zero-decay-inert: at weight_decay 0 every group carries decay
  0). Then a golden wd-0 run compares selected log lines (golden-selected-
  text-equality, UNAVAILABLE while its base is null)
  (spec: training-stack.md#weight-decay-census-evidence).

  The check-script rc expect stays aid-less: the child exit is the aggregate
  verdict, and its four ##AID lines carry the census legs (the geo-paths
  check-script template).
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
              grep_pattern="^(phase|epoch|best)",
              aid="weight-decay-census.golden-selected-text-equality")


def gate_gpc_c(ctx):
  """npce-training: the closed-form base plus its refiner train and save.

  WHAT: NPCE (a closed-form sparse-Legendre base under a trained
  refiner) in residual and ratio forms. WHY: both forms must train, the
  illegal combinations must error, and the base must refit per
  training-set size. HOW: the residual and ratio runs each exit zero and
  name their pce form in the banner (residual-config-exit-zero /
  residual-pce-text-present, ratio-config-exit-zero /
  ratio-pce-text-present); pce+ia (YAML) and pce+--rescale (flag) both
  exit nonzero with the exclusivity error (pce-ia-refusal /
  pce-rescale-refusal); a 2-point n_train sweep exits zero, prints both
  result lines, and names its staging banner (sweep-result-lines-and-pce-
  banner). The golden selected-text equality is UNAVAILABLE with no
  configured base, and the save/rebuild/base(theta) round-trip
  (rebuild-vs-base) is UNAVAILABLE and owed to the check-script set: this
  wrapper logs that it belongs there but runs no comparison. No assertion
  compares loss values, so loss descent is logged-only, not asserted
  evidence (spec: models-and-designs.md#npce-training-evidence).
  """
  ctx.require_caps("torch", "cosmolike", "gpu")
  _golden_leg(ctx=ctx,
              gate_id="npce-training",
              yaml_name="cosmic_shear_train_emulator.yaml",
              grep_pattern="^(phase|epoch|best)",
              aid="npce-training.golden-selected-text-equality")
  _smoke_driver(ctx=ctx,
                config_key="npce-training-residual",
                required_banners=["pce"],
                exit_aid="npce-training.residual-config-exit-zero",
                banner_aid="npce-training.residual-pce-text-present")
  _smoke_driver(ctx=ctx,
                config_key="npce-training-ratio",
                required_banners=["pce"],
                exit_aid="npce-training.ratio-config-exit-zero",
                banner_aid="npce-training.ratio-pce-text-present")
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
    aid="npce-training.pce-ia-refusal",
    label="npce-training exclusivity error (pce + ia)",
    ok=(rc_ia != 0 and logscan.search(text=out_ia, pattern=r"(?i)exclusive")),
    detail="pce + model.ia must exit nonzero with the exclusive message "
           "(rc " + str(rc_ia) + ")")
  ctx.expect(
    aid="npce-training.pce-rescale-refusal",
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
    aid="npce-training.sweep-result-lines-and-pce-banner",
    label="npce-training 2-point sweep_ntrain ran both points",
    ok=(rc_s == 0 and len(parent) >= 2 and staged),
    detail="rc " + str(rc_s) + "; parent N_train f(>0.2) lines "
           + str(len(parent)) + " (need >=2); pce staging banner "
           + ("present" if staged else "ABSENT") + " (need present)")
  ctx.unavailable(
    aid="npce-training.rebuild-vs-base",
    label="npce-training rebuild == base(theta) round-trip",
    reason="this wrapper only logs that a save -> h5 pce group -> "
           "from_state rebuild == base(theta) probe belongs in the "
           "check-script set (save-rebuild-drift already round-trips the "
           "NPCE save); it runs no such comparison here, so the leg is "
           "owed to a standalone npce-training probe")


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

  WHAT: the child check exercises five acceptance legs on synthetic scalar
  artifacts, torch only, no cosmolike. artifact-round-trip: a saved-then-
  rebuilt emulator predicts bitwise and its ScalarGeometry state round-trips
  byte-identical. geometry-and-schema-guards: the constant-column,
  duplicate-sidecar-name, and trunk-only errors fire, while a genuinely
  varying tiny-magnitude column still builds. scalar-adapter-contract: the
  emul_scalars adapter (loaded torch-only through a cobaya.theory stub)
  derives its provides/requirements from the artifacts and raises on the
  duplicate / overlap / subset-superset / wrong-kind legs. npce-composition:
  the residual base + refiner algebra is bitwise and a saved base + net
  {name: value} prediction is exact. finetune-parity: the epoch-0 warm-start
  parity holds, the anchor mask zeros exactly the appended input column, and
  the outputs-mismatch / wrong-kind-source refusals fire. The board wrapper
  reduces the child's five legs to its exit code (spec:
  families-scalar-cmb.md#scalar-identity-evidence).
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

  WHAT: the child check exercises seven acceptance legs on synthetic CMB
  artifacts (a ParamGeometry over a written covmat + a CmbDiagonalGeometry
  over a synthetic fiducial C_ell + a small ResMLP), torch only, no cosmolike
  and no CAMB. geometry-and-reference-schema: the ruled Gaussian
  per-multipole scale, byte-identical persistence of the fiducial amplitude
  references, the geometry state round-trip, the endpoint scale comparison,
  and the nonpositive / typed reference-value refusals. amplitude-law-and-
  score: the order-one as_exp2tau_ref law, its transform round-trip, the
  parameter-aware physical score and factor-corrected roughness residual, the
  stale-parameter isolation, and the missing / invalid-law + raw-factor
  mutation refusals. artifact-and-adapter-round-trip: save -> rebuild ->
  predict bitwise on BOTH laws and the stubbed emul_cmb adapter's Cl assembly
  (shared axis, low-l padding, requirements, both get_Cl convention guards,
  spectrum uniqueness, request-range refusals). roughness-contract: band
  ratio > 100, exact zero on a zero residual, OFF identity bitwise, the
  one-reduction score composition, and the bounded lensing-period leg.
  model-variant-composition: the correction-head leg (attach, identity basis,
  epoch-0 identity, two-phase discipline, save -> rebuild -> predict bitwise)
  and the NPCE residual algebra, roughness composition, saved prediction, and
  the pce x amplitude-law exclusivity guard. finetune-parity: the epoch-0
  parity from a CMB source, the CMB fine-tune config shape, and the
  cosmolike-only pin's wrong-kind refusal. covariance-known-answer: an affine
  fake CAMBdata makes the 5-point stencil exact, so compute_cmb_covariance's
  non-Gaussian eq-6 contraction is checked against a direct sensitivity-matrix
  known answer across all six blocks, the retired weights miss by orders of
  magnitude, the raw-vs-scaled lensing-potential fixtures stay distinct, a
  width-3 constant-response band reproduces the per-multipole eq 6, and a
  zeroed band's weight is exactly 0. The board wrapper reduces the child's
  seven legs to its exit code (spec:
  families-scalar-cmb.md#cmb-identity-evidence).
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

  WHAT: the CMB dataset generator writes two tiny dumps (200 rows each,
  l = 2..350, cmblensed, As sampled linearly) — four per-spectrum dv files
  + sidecars, phiphi actually filled; the CMB covariance builder
  writes the Gaussian .npz on the fixture LCDM (its first real run);
  a data.cmb / as_exp2tau_ref training run collapses the val median below
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
  several minutes (~400 serial low-accuracy CAMB calls).

  HOW: the child (gates/checks/cmb_smoke.py) folds its many human-readable
  sub-checks into SIX declared board legs, emitting one '##AID' manifest
  line per leg — generated-spectrum-dumps (both generator subprocesses exit
  zero, four dv files + sidecars land, the dump shape + filled phiphi),
  gaussian-covariance (the Gaussian .npz ell axis 2..LMAX + positive TT
  sigma), nondiagonal-covariance-structure (the eq-6 six dense blocks,
  symmetry, diagonal growth, live off-diagonals, PSD, the stencil + weight
  provenance keys), training-collapse (the best val median below half the
  staged mean predictor — the dead-network bar), cobaya-serving (the real
  in-process get_model + get_Cl equals the predictor's own C_ell at rtol
  1e-6), and diagnostics-output (two CMB pages + the non-trivial PDF). The
  later legs depend on the earlier fixtures, so a failed generator /
  covariance leg leaves the training, cobaya, and diagnostics aids
  unemitted. The wrapper's rc-check stays the single aggregate verdict; the
  six ##AID lines carry the per-leg map
  (spec: notes/families-scalar-cmb.md#cmb-smoke-evidence).
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
, superseding the old half-chunk form; the old form is the gate's
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
  scipy, no CAMB.

  HOW: the child (gates/checks/bsn_identity.py) folds its many
  human-readable sub-checks into SIX declared board legs, emitting one
  '##AID' manifest line per leg — simpson-polynomial-nodes (the
  cumulative-Simpson node-by-node integrals + the mutation control +
  the guard), distance-pipeline-consistency (the pipeline vs the dense
  same-integrator reference at 1e-6), geometry-and-artifact-round-trip
  (the log-offset law both ways + grid-state + the law/domain guards +
  the save/rebuild/predict bitwise legs on BOTH laws), npce-composition
  (the residual encode/decode algebra + base-plus-net save/rebuild),
  adapter-piecewise-contract (the emul_baosn two-window layout, the
  piecewise getters, the units + desert + pair-validation refusals),
  and finetune-parity (the epoch-0 warm-start parity + the from_config
  refusals). The wrapper's rc-check stays the single aggregate verdict;
  the six ##AID lines carry the per-leg map
  (spec: notes/families-background-mps.md#bsn-identity-evidence).
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

  WHAT: the check gates the BAOSN background family through four evidence
  legs, and it folds one '##AID <leg> <PASS|FAIL>' terminal per leg into
  this gate's executed set (each leg aggregates its group of the child's
  report() probes; the wrapper's rc-check below stays the single aggregate
  verdict, not a leg):

    - generated-background-dumps: the background dataset generator writes
      two tiny dumps (200 rows, one background-only CAMB evaluation per
      sample — fast) carrying BOTH quantities + their _z.npy grid sidecars
      (one CAMB pass fills both — the one-pass rule), and every Hubble
      column varies across cosmologies (the stale-cache tripwire, relative
      spread > 1e-5);
    - training-collapse: two data.grid training runs (Hubble/log_offset +
      D_M/none) each collapse below 0.5x the staged mean predictor (the
      dead-network-relative bar — a net that learned NOTHING fails it);
    - cobaya-vs-camb: the real cobaya lifecycle through emul_baosn serves
      H / D_A (SN window) and D_M (recombination window) within 2% of
      CAMB's OWN background at an off-center point — truth is available
      here, the strongest smoke of the program — and the desert stays loud
      through the real lifecycle;
    - diagnostics-output: the grid-family diagnostics build two pages and
      the PDF lands.

  torch + cobaya + a compiled CAMB under $ROOTDIR
  (spec: notes/families-background-mps.md#bsn-smoke-evidence).
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

  WHAT: the check gates the grid2d matter-power family through seven
  evidence legs, and it folds one '##AID <leg> <PASS|FAIL>' terminal per
  leg into this gate's executed set (each leg aggregates its group of the
  child's report() probes; the wrapper's rc-check below stays the single
  aggregate verdict, not a leg):

    - geometry-laws-and-pins: the Grid2DGeometry standardize / state
      round-trips, the width / unknown-law guards, the exact
      constant-column pins under both laws, and the wholly-constant
      refusal;
    - bounded-staging-values: the STAGING law transform through the REAL
      load_source (law rows = log(raw/base) base-aligned by dump_rows;
      k_stride keeps the top edge; positivity loud), and the BOUNDED
      staging on the production 122 x 2,000 grid (guarded memmap reads
      prove every raw + base read is row-chunked and column-thinned, an
      independent known-answer + mean match, a disk-backed low-RAM
      result, and the whole-selection + mean-before-cast mutations both
      disagree);
    - stable-streamed-moments: a 50,000-row 1e8/1-ULP column keeps its
      true std through the Chan/Welford accumulator over uneven
      chunkings (never a false pin), the relative constant-pin boundary,
      and the from_stats encode == the materialized standardization;
    - staging-file-lifecycle: the experiment-owned temp files —
      supersede-on-restage, sweep-lane release bounded to one live file,
      failure unlink, and the resident-RAM control that makes no temp;
    - saved-model-variants: save -> rebuild -> predict bitwise on the
      syren_linear and none laws, the correction-head leg (ResCNN on
      z-slice channels, the two-phase discipline, the n_tokens-on-real-
      bins rejection, the bitwise round-trip), and the NPCE leg
      (residual algebra + base + net prediction bitwise, diagonal ratio
      rejection);
    - adapter-assembly-and-defaults: the emul_mps assembly EXACT against
      synthetic base stubs (P_lin = exp(net)*base, the low-k blend pins
      boost -> 1 below k_t, P_nl = B*P_lin, the boost base fed the
      EMULATED P_lin), the getters serving Cobaya's public nonlinear
      default (an omitted argument returns the nonlinear grid /
      interpolator, != the explicit linear branch, pinned against the
      installed BoltzmannBase signature), the interpolator node
      round-trip, the pair / grid / wrong-kind guards, and the
      reject-on-bad-spectrum semantics;
    - config-and-finetune: validate_grid2d's pairing / base-file /
      k_stride / transfer legs (transfer ACCEPTED since the 2026-07-12
      symmetry ruling) and the finetune parity + metadata-mismatch legs.

  torch + scipy, no CAMB, no symbolic_pofk (the real syren formulas ride
  the EMUL2 acceptance)
  (spec: notes/families-background-mps.md#mps-identity-evidence).
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

  WHAT: the MPS emulators run law-none end to end against CAMB's own
  P(k, z): the generator writes tiny linear-power and boost dumps, two
  grid2d networks train and collapse below their staged mean predictors,
  the grid2d diagnostics path builds its pages and PDF, and the real
  Cobaya provider agrees with CAMB inside 5%. The syren-law path is
  exactly gated by mps-identity's stubbed legs (the formulas themselves
  are vendored in syren/), and the full syren + EMUL2 hybrid run is the
  unit's recorded acceptance experiment
  (cobaya_theory/EXAMPLE_EMUL2_EVALUATE.yaml, user-run on the
  workstation).

  HOW: gates/checks/mps_smoke.py runs four board-declared evidence legs,
  and it folds one '##AID <leg> <PASS|FAIL>' terminal per leg into this
  gate's executed set:
    - generated-power-dumps: both MPS generator subprocesses exit zero
      and write the linear-power and boost arrays plus both axis
      sidecars (incl. the verbatim wants-Cl quirk), and the linear dump
      has the expected shape with every nonfailed row positive.
    - training-collapse: the linear-power and boost grid2d trainings
      (law none) each reduce their best validation median below half
      their own staged mean-predictor median (the dead-network bar).
    - diagnostics-output: the boost training's grid2d diagnostic builds
      the two (z, k) pages and lands a nonempty plot_diagnostics PDF
      through the grid2d dispatch.
    - cobaya-vs-camb: the real Cobaya lifecycle through emul_mps serves
      P_lin and P_nl (grid + interpolator) within 5% of CAMB's OWN
      P(k, z) at an off-center point, and an out-of-range interpolator
      query raises.

  torch + cobaya + a compiled CAMB under $ROOTDIR
  (spec: notes/families-background-mps.md#mps-smoke-evidence).
  """
  ctx.require_caps("torch", "cobaya")
  rc, out = ctx.run_check("gates/checks/mps_smoke.py")
  if not ctx.dry:
    # AID-LESS rc guard: the four per-leg verdicts are the '##AID' lines the
    # child prints (folded into the executed set by run_check); this line only
    # asserts the child's aggregate exit status, so it carries no aid.
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
 safe-sqrt producer (an exact-fit chi2 == 0 has a finite, zero
  gradient in every sqrt mode instead of the 0/0 = NaN it used to produce;
  positives agree with sqrt; a negative / NaN chi2 is refused; eager and
  torch.compile agree), and the epoch reduction (a finite per-batch
  loss near the float32 max yields a finite epoch mean via host float64
  accumulation, where the old device float32 loss*bs product overflowed to
  Inf), and the chi2-domain boundary (eval_val and
  eval_source_chi2 raise on a finite negative chi2 that training folds; the
  scale-aware band scales with the per-row kept WIDTH, not w^2, so a
  production-width leg refuses a chi2 = -2 the retired w^2 rule crowned as
  perfect, a mutation arm restoring w^2 is caught, and an ill-conditioned SPD
  control shows genuine roundoff near zero falls inside the band); the valid
  controls keep their metrics and their [ok] parity lines. torch only, no
  cosmolike, no GPU (spec: training-stack.md, the "NaN scores as a perfect
  emulator" section and its pre-training parity + +
 clauses).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/finite_contract.py")
  if not ctx.dry:
    # exit code 0 = every leg ran and passed; 2 = a mandatory lane (the
    # torch.compile backward) could not run on this box, a non-green result
    # rather than a silent PASS; any other nonzero = a tested assertion
    # failed. Both nonzero codes make the gate non-PASS.
    if rc == 2:
      reason = ("a mandatory lane could not run (torch.compile backward "
                "unavailable); run this gate on a compile-capable box")
    else:
      reason = "check exit code " + str(rc)
    ctx.expect(
      label="finite-contract eval/train/diagnostic/parity/safe-sqrt legs",
      ok=(rc == 0),
      detail=reason + " (gates/checks/finite_contract.py)")


def gate_board_selftest(ctx):
  """board-selftest: the runner reports the truth about what actually ran.

  WHAT: pure-Python self-tests of run_board's own control flow (no torch, no
  cosmolike), over a small set of fake gates. WHY: several board-truth defects
  let a run report success (or reuse a stale PASS) without testing what it
  claimed. A dependency-skipped selected gate ran no test code but was counted
  green; an unknown --gate / --from / --force-rerun id printed a warning and
  then exited 0 on a smaller (or empty) surface; the finite-contract check
  printed a compile-lane skip inside a process that still returned 0; a stored
  PASS was trusted on status alone, so a configuration change or a mutated
  referenced YAML reused it and an interrupted forced rerun preserved the prior
  PASS while its cited log had already been truncated. HOW: the check drives the
  real run_board.main / select_gates over fake gates -- a dependency-skipped
  gate exits nonzero and runs no body; an unknown selector id is a usage error
  with a suggestion; the selectors are mutually exclusive; the finite-contract
  compile-lane code maps to a non-PASS; a stored PASS is trusted only when both
  the executable-surface digest and the input digest are current (a config
  change, a mutated YAML, and a mismatched digest all rerun); and a RUNNING
  record is persisted before any gate code, so an interruption leaves an
  interrupted attempt with its own immutable log (never the prior PASS), while
  a successful run publishes a fresh log whose stored digest matches its bytes.
  No torch, no cosmolike, no GPU.
  """
  rc, out = ctx.run_check("gates/checks/board_selftest.py")
  if not ctx.dry:
    ctx.expect(
      aid="board-selftest.exit-truth",
      label="board-selftest exit-truth / selector / lane-code legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/board_selftest.py)")


def gate_artifact_readback(ctx):
  """artifact-readback: saved attributes are parsed by type, not truthiness.

  WHAT: a CPU check of the typed attribute reader (_read_native_bool) plus a
  census that no artifact boolean attribute is still read with a bool()
  truthiness coercion. WHY: HDF5 attributes are weakly typed, so a marker read
  back with Python truthiness can flip a feature bit -- the string "False" is
  truthy, so a transfer artifact whose transfer_refined marker literally reads
  "False" would load its drifted prediction weights. HOW: the reader accepts a
  native Python / numpy boolean, returns the default for an absent key, and
  refuses every string ("False", "true", "0", ...) and integer, naming the
  file and the required native-boolean schema; the source census confirms the
  read boundary routes through it. The live save/forge/rebuild proof needs a
  real HDF5 artifact and is owned by the workstation artifact-integrity gate.
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/artifact_readback.py")
  if not ctx.dry:
    ctx.expect(
      aid="artifact-readback.typed-bool",
      label="artifact-readback typed-attribute legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/artifact_readback.py)")


def gate_generator_seed(ctx):
  """generator-seed: the dataset generator samples from an owned, recorded RNG.

  WHAT: a CPU census of the generator's sampling surface plus the numpy
  Generator's replay guarantee. WHY: the generator had no seed and drew every
  sample from the process-global np.random, so two runs with identical YAML,
  command line, and code produced different parameter tables and nothing
  recorded the seed -- the dataset could not be replayed from its inputs. HOW:
  a required integer seed owns a numpy Generator threaded through the uniform
  sampling, the emcee walker init and the sampler's own moves, and the thinning
  subselection; the seed and RNG are written into the chain header. The check
  confirms no process-global np.random draw remains, the owned Generator is
  used, the seed is required / type-checked / recorded, and same-seed draws
  reproduce. The generator imports MPI / cobaya / CAMB, so the live end-to-end
  replay rides the workstation smoke gates; no torch here.
  """
  rc, out = ctx.run_check("gates/checks/generator_seed.py")
  if not ctx.dry:
    ctx.expect(
      aid="generator-seed.owned-rng",
      label="generator-seed sampling-RNG legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/generator_seed.py)")


def gate_cli_strict(ctx):
  """cli-strict: a misspelled flag is a usage error, not a silent ignore.

  WHAT: a census that all eight public entry points parse with strict
  parse_args, plus a live test of two driver mains with the expensive boundary
  monkeypatched. WHY: the drivers and data producers used parse_known_args and
  discarded the unknown tokens, so a misspelled flag (--activaton, --quieet,
  --diagnostc, --sav) was silently ignored and the run proceeded at the YAML or
  default value -- most dangerously publishing to the default --save root. HOW:
  a valid command line reaches the boundary (parsing succeeded); a misspelled
  flag exits nonzero before the boundary, so no data is read, no artifact
  loaded, no CAMB started, no worker spawned, and no output root chosen.
  Importing the drivers needs torch; the parse itself is pure Python.
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/cli_strict.py")
  if not ctx.dry:
    ctx.expect(
      aid="cli-strict.strict-parse",
      label="cli-strict flag-parsing legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/cli_strict.py)")


def gate_family_first(ctx):
  """family-first: every driver owns exactly one data-block family.

  WHAT: a CPU check of require_family_block plus a census of the four
  cosmic_shear drivers. WHY: the direct cosmic_shear drivers passed family=None,
  which skipped the family check, so a CMB / grid / grid2d / scalar YAML
  launched through cosmic_shear_train_emulator.py trained under the wrong public
  identity (a scalar YAML died later at run_tag on a missing train_dv key). HOW:
  a direct cosmic-shear run now owns the "cosmolike" data-vector family and
  rejects any other family's block naming its driver, while a clean cosmic-shear
  YAML trains; the per-family wrappers accept their own block; and the census
  confirms the four cosmic_shear drivers default family=cosmolike, always call
  the check, and drop the misleading dispatcher prose. Importing the driver
  needs torch; the check is pure Python.
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/family_first.py")
  if not ctx.dry:
    ctx.expect(
      aid="family-first.family-owned",
      label="family-first driver-identity legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/family_first.py)")


def gate_stage_ram(ctx):
  """stage-ram: the host-RAM staging decision counts every array, keeps the
  seeded row order, and prints honest arithmetic.

  WHAT: a CPU check of stage_source's resident-vs-disk branch decision with a
  mocked available-memory value, plus the row order the two branches hand the
  training loop. WHY: two ways the branch could differ silently. First, the
  resident branch materializes BOTH the compact parameter table C[idx] and the
  compact target dv[idx], but the budget once counted only the dv bytes, so a
  narrow-output dump (many input columns, one output column) chose the resident
  branch even when the two copies together exceeded the allowance -- an
  avoidable out-of-memory. Second, both branches must present the selected rows
  in the run's one seeded selection order; if the resident branch renumbered to
  a plain arange over the sorted compact copy, the same seed would train a
  different cosmology at each step than the disk branch, so host-memory
  availability would change training. HOW: pinned memory between "dv alone fits"
  and "dv plus C fits" (the corrected code keeps disk there), resident/disk
  controls, an unequal dtype/width case, a duplicate selection refused loudly,
  the exact-fit boundary (need below, equal, above budget), a banner that names
  params + dv + idx and adds up, and the seeded-order proof -- the real
  per-source loader built in both storage
  regimes, one shared epoch permutation, identical executed params, targets,
  and minibatch order, with a mutation arm restoring arange that must break the
  match. Importing the module needs torch; every array is tiny.
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/stage_ram.py")
  if not ctx.dry:
    ctx.expect(
      aid="stage-ram.both-copies",
      label="stage-ram host-RAM accounting legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/stage_ram.py)")


def gate_diagnostics_domain(ctx):
  """diagnostics-domain: a corrupted chi2 never crowns a diagnostic.

  WHAT: the shared score-domain boundary at every chi2 CONSUMER, not just the
  training reduction and the two evaluation boundaries (increment (e)).
  local_linear_floor computed its interpolation-floor score by calling
  chi2fn.chi2 DIRECTLY and interpreting the unchecked value (f_floor via
  dchi2_floor > 0.2, median_floor via np.median), and the CMB / grid / grid2d
  residual functions upcast each chunk with .double() before any check. WHY: a
  geometry whose Cinv is not positive-definite (a same-shaped h5 edit strict
  weight loading accepts) makes the floor go negative, and the > 0.2 test read
  a -1 floor as a PERFECT 0 -- an impossible "data-only floor" reported ideal.
  HOW: a CPU torch-only check drives screen_chi2 (the one shared helper in
  the losses core module: a valid positive score passes byte-identical, a within-band
  roundoff negative normalizes to exact 0, and a materially negative / NaN /
  +-Inf score raises naming the boundary, the rows, the minimum, and the band;
  a loss without _chi2_n_terms falls back to the 1e-6 band floor; the term
  count widens the band with the kept width) and the REAL producers: the
  local_linear_floor refuses a reachable negative floor BEFORE it computes
  f_floor (the floor guard fires ahead of the model arm), refuses a NaN floor,
  and returns finite scores on a valid run, with a mutation arm (the guard
  bypassed) recreating the false f_floor = 0; the cmb_residual_diagnostic
  refuses a corrupt per-sample score and keeps its bands on a valid run; and a
  source census proves the grid / grid2d residual functions route through the
  same shared boundary (_screen_diag_chi2 -> screen_chi2) with no raw .double()
  score path left. torch only, no cosmolike, no CAMB, no GPU (spec:
  training-stack.md, the increment (h) diagnostic-score-boundary section).
  """
  ctx.require_caps("torch")
  rc, out = ctx.run_check("gates/checks/diagnostics_domain.py")
  if not ctx.dry:
    ctx.expect(
      aid="diagnostics-domain.score-boundary",
      label="diagnostics-domain floor/residual score-boundary legs",
      ok=(rc == 0),
      detail="check exit code " + str(rc)
             + " (gates/checks/diagnostics_domain.py)")


BOARD = [
  Gate(id="ema-off-identity",
       spec_code="GM-C",
       title="EMA off-mode byte-identity",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the current and pre-EMA drivers produce matching epoch and "
            "best-epoch log lines once the trailing wall-clock field is "
            "stripped, run against the configured historical base",
       evidence=(Assertion("ema-off-identity.golden-selected-text-equality",
                           "training-stack.md#ema-off-identity-golden-selected-text-equality"),),
       run=gate_gm_c,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.ema-off-identity-golden",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu"),
       worktree_commit="46ec5e1"),
  Gate(id="ema-smoke",
       spec_code="GM-D",
       title="EMA on-mode smoke",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the EMA-on training smoke exits zero, prints its resolved "
            "\"ema: horizon 3 epochs\" banner, and reaches the logged "
            "\"rewound to best epoch\" rewind line",
       evidence=(Assertion("ema-smoke.driver-exit-zero",
                           "training-stack.md#ema-smoke-driver-exit-zero"),
                 Assertion("ema-smoke.horizon-banner-present",
                           "training-stack.md#ema-smoke-horizon-banner-present"),
                 Assertion("ema-smoke.rewind-line-present",
                           "training-stack.md#ema-smoke-rewind-line-present")),
       run=gate_gm_d,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.ema-smoke-config",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="production-diagnostic",
       spec_code="DIAG (G1, G-F, GN-F, GS-D, GT-C)",
       title="Production diagnostic run",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="one --diagnostic run asserts the retired-class census returns no "
            "hits, the package imports cleanly, the driver exits zero, and the "
            "sizes banner has the right shape, while the cut-row selection, the "
            "diagnostics PDF, and the triangle shading stay unread evidence",
       evidence=(Assertion("production-diagnostic.retired-class-name-census",
                           "training-stack.md#production-diagnostic-retired-class-name-census"),
                 Assertion("production-diagnostic.package-import",
                           "training-stack.md#production-diagnostic-package-import"),
                 Assertion("production-diagnostic.driver-exit-zero",
                           "training-stack.md#production-diagnostic-driver-exit-zero"),
                 Assertion("production-diagnostic.sizes-banner",
                           "training-stack.md#production-diagnostic-sizes-banner"),
                 Assertion("production-diagnostic.cut-row-selection",
                           "training-stack.md#production-diagnostic-cut-row-selection"),
                 Assertion("production-diagnostic.diagnostics-pdf",
                           "training-stack.md#production-diagnostic-diagnostics-pdf"),
                 Assertion("production-diagnostic.triangle-shading",
                           "training-stack.md#production-diagnostic-triangle-shading")),
       run=gate_diag,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.production-diagnostic-config",)
                                + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="single-phase-demotion",
       spec_code="GP-D",
       title="Single-phase phase-arg demotion",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the single-phase and two-phase-control drivers both exit zero, and "
            "the single-phase stream carries demotion-related text",
       evidence=(Assertion("single-phase-demotion.single-phase-exit-zero",
                           "training-stack.md#single-phase-demotion-single-phase-exit-zero"),
                 Assertion("single-phase-demotion.demotion-text-present",
                           "training-stack.md#single-phase-demotion-demotion-text-present"),
                 Assertion("single-phase-demotion.two-phase-control-exit-zero",
                           "training-stack.md#single-phase-demotion-two-phase-control-exit-zero")),
       run=gate_gp_d,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.single-phase-demotion-single",
                                 "gate_configs.single-phase-demotion-control") + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="head-scheduler-override",
       spec_code="GH-E",
       title="Head scheduler override",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the head-scheduler override driver exits zero and prints its "
            "override banner, while the golden selected-text equality and the "
            "lr-cut cadence remain conditional evidence",
       evidence=(Assertion("head-scheduler-override.golden-selected-text-equality",
                           "training-stack.md#head-scheduler-override-golden-selected-text-equality"),
                 Assertion("head-scheduler-override.driver-exit-zero",
                           "training-stack.md#head-scheduler-override-driver-exit-zero"),
                 Assertion("head-scheduler-override.override-banner-present",
                           "training-stack.md#head-scheduler-override-override-banner-present"),
                 Assertion("head-scheduler-override.lr-cut-cadence",
                           "training-stack.md#head-scheduler-override-lr-cut-cadence")),
       run=gate_gh_e,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.head-scheduler-override-config",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="eval-batch-invariance",
       spec_code="GE-C",
       title="Eval-batch partition invariance",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="a torch-only script proves the validation scores and the ordinary "
            "median are independent of how the same rows are chunked into eval "
            "batches, while the two timing observations remain informational",
       evidence=(Assertion("eval-batch-invariance.partition-invariance",
                           "training-stack.md#eval-batch-invariance-partition-invariance"),
                 Assertion("eval-batch-invariance.ordinary-median",
                           "training-stack.md#eval-batch-invariance-ordinary-median"),
                 Assertion("eval-batch-invariance.cuda-timing",
                           "training-stack.md#eval-batch-invariance-cuda-timing"),
                 Assertion("eval-batch-invariance.production-timing-claim",
                           "training-stack.md#eval-batch-invariance-production-timing-claim")),
       run=gate_ge_c,
       manifest=Manifest(code=(), inputs=()),
       needs=("torch", "gpu")),
  Gate(id="finite-contract",
       spec_code="FIN-A",
       title="Finite training/evaluation contract",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the training-stack finite contract: the 'NaN scores as a "
            "perfect emulator' section (the eval_val / train-step / "
            "eval_source_chi2 guards), its pre-training parity clause "
            "(build_warm_start + build_transfer_start), the "
            "safe-sqrt producer clause (exact-fit finite gradients per "
            "mode, positives analytic, negative/NaN chi2 refused, eager + "
            "compiled), the epoch-reduction clause (host float64 "
            "accumulation; a finite epoch mean where the old float32 "
            "loss*bs product overflowed), the chi2-domain "
            "clause (eval_val / eval_source_chi2 raise on a finite "
            "negative chi2 that training folds; the scale-aware band; the "
            "finite-only false-crowning mutation; the capability-gated "
            "compile arm), and the width-band clause (the band "
            "scales with the kept WIDTH, not w^2: a production-width leg "
            "refuses a chi2 = -2 the retired w^2 rule crowned perfect, a "
            "w^2-restoring mutation arm, a scalar-width leg, a subclass "
            "census, and an ill-conditioned SPD roundoff control); the red "
            "legs plus the finite controls",
       run=gate_finite_contract,
       manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="board-selftest",
       spec_code="BRD-A",
       title="Board runner reports the truth about what ran",
       tier=TIER_BACKLOG,
       home="gates-and-board",
       maps="the board-truth campaign: a dependency-skipped selected gate "
            "exits nonzero and runs no body; an unknown --gate / --from / "
            "--force-rerun id is a usage error with a suggestion and a "
            "nonzero exit; the run selectors are mutually exclusive; the "
            "finite-contract compile-lane skip is a distinct non-green exit "
            "code the board wrapper maps to a non-PASS; a stored PASS whose "
            "cited raw log is deleted / truncated / edited / undigested reads "
            "stale-log through the same resume decision and reruns; and the "
            "structured evidence map validates (the shipped board resolves, "
            "and a bad anchor / missing note / duplicate id / malformed anchor "
            "are each rejected) (the red legs plus the valid controls)",
       evidence=(Assertion("board-selftest.exit-truth",
                           "gates-and-board.md#board-selftest-exit-truth"),),
       run=gate_board_selftest,
       manifest=Manifest(code=(), inputs=()),
       needs=()),
  Gate(id="generator-seed",
       spec_code="GEN-A",
       title="Dataset generator samples from an owned, recorded RNG",
       tier=TIER_BACKLOG,
       home="data-generation-and-cuts",
       maps="the generator sampling-seed contract: a required integer seed "
            "owns a numpy Generator threaded through the uniform sampling, the "
            "emcee walker init + the sampler's own moves, and the thinning "
            "subselection (no process-global np.random draw remains); the seed "
            "is type-checked and written to the chain header; same-seed draws "
            "reproduce. The append-replay and worker-invariance legs ride the "
            "workstation smoke gates",
       evidence=(Assertion("generator-seed.owned-rng",
                           "data-generation-and-cuts.md#generator-seed-owned-rng"),),
       run=gate_generator_seed,
       manifest=Manifest(code=(), inputs=()),
       needs=()),
  Gate(id="cli-strict",
       spec_code="CLI-A",
       title="Every public executable rejects a misspelled flag",
       tier=TIER_BACKLOG,
       home="conventions-and-workflow",
       maps="the strict-CLI contract: all eight public entry points parse with "
            "parse_args (no parse_known_args), and two representative driver "
            "mains reject a misspelled flag (--activaton) with a nonzero exit "
            "before the expensive boundary, while a valid command line reaches "
            "it",
       evidence=(Assertion("cli-strict.strict-parse",
                           "conventions-and-workflow.md#cli-strict-strict-parse"),),
       run=gate_cli_strict,
       manifest=Manifest(
           code=("cosmic_shear_train_emulator.py",
                 "cosmic_shear_sweep_ntrain_emulator.py",
                 "cosmic_shear_sweep_hyperparam_emulator.py",
                 "cosmic_shear_bakeoff_activation_emulator.py",
                 "cosmic_shear_tune_emulator.py",
                 "scalar_train_emulator.py",
                 "compute_data_vectors/generator_core.py",
                 "compute_data_vectors/compute_cmb_covariance.py",
                 "emulator/designs",
                 "emulator/losses"),
           inputs=()),
       needs=("torch",)),
  Gate(id="family-first",
       spec_code="FAM-A",
       title="Every driver owns exactly one data-block family",
       tier=TIER_BACKLOG,
       home="conventions-and-workflow",
       maps="the family-first driver contract: a direct cosmic_shear run owns "
            "the cosmolike data-vector family and rejects a CMB / grid / "
            "grid2d / scalar YAML naming its driver, a clean cosmic-shear YAML "
            "trains, the per-family wrappers accept their own block; the "
            "census confirms the four cosmic_shear drivers default "
            "family=cosmolike, always check, and drop the dispatcher prose",
       evidence=(Assertion("family-first.family-owned",
                           "conventions-and-workflow.md#family-first-family-owned"),),
       run=gate_family_first,
       manifest=Manifest(code=("cosmic_shear_train_emulator.py",
                               "emulator/designs", "emulator/losses"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="stage-ram",
       spec_code="SRM-A",
       title="Host-RAM staging counts every materialized array",
       tier=TIER_BACKLOG,
       home="data-generation-and-cuts",
       maps="the host-RAM staging accounting and the seeded row order: "
            "stage_source counts BOTH the parameter and target compact copies "
            "(each at its own dtype and width) plus the reindex array, so a "
            "narrow-output dump keeps the disk-backed branch when the two "
            "copies together exceed the budget; and the resident branch returns "
            "local coordinates that walk the compact copy in the seeded "
            "selection order, so the real loader trains the same cosmology at "
            "the same step in either storage regime; resident / disk controls, "
            "an unequal-dtype case, a duplicate selection refused loudly, the "
            "exact-fit boundary, the honest three-term banner, the loader-driven "
            "order proof, and mutation arms for the dv-only estimate and the "
            "retired arange reindex",
       evidence=(Assertion("stage-ram.both-copies",
                           "data-generation-and-cuts.md#stage-ram-both-copies"),),
       run=gate_stage_ram,
       manifest=Manifest(code=(), inputs=()),
       needs=("torch",)),
  Gate(id="artifact-readback",
       spec_code="ARB-A",
       title="Saved attributes parsed by type, not truthiness",
       tier=TIER_SAVE_AND_SAMPLE,
       home="artifacts-inference-warmstart",
       maps="the artifact-readback type contract: the shared typed reader "
            "accepts a native boolean, returns the default for an absent key, "
            "and refuses every string / integer (the truthy 'False' that "
            "would load drifted transfer weights) naming the file + schema; a "
            "source census confirms no artifact boolean is truthiness-coerced. "
            "The live save/forge/rebuild proof is workstation-owed",
       evidence=(Assertion("artifact-readback.typed-bool",
                           "artifacts-inference-warmstart.md#artifact-readback-typed-bool"),),
       run=gate_artifact_readback,
       manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="diagnostics-domain",
       spec_code="DIAG-A",
       title="Diagnostic score-domain boundary",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the training-stack increment (h) diagnostic-score-boundary "
            "section: the shared screen_chi2 helper (valid "
            "byte-identical, within-band roundoff to exact 0, materially "
            "negative / NaN / +-Inf refused naming the boundary + rows + "
            "band, the fallback-1 floor, the width-scaled band), the REAL "
            "local_linear_floor (a reachable negative floor refused before "
            "f_floor, a NaN floor refused, a valid control, the "
            "guard-bypassed mutation recreating the false f_floor = 0), the "
            "REAL cmb_residual_diagnostic (corrupt-score refusal + valid "
            "control), and the grid / grid2d producer census through the one "
            "shared boundary",
       evidence=(Assertion("diagnostics-domain.score-boundary",
                           "training-stack.md#diagnostics-domain-score-boundary"),),
       run=gate_diagnostics_domain,
       manifest=Manifest(code=(), inputs=()),
       needs=("torch",)),
  Gate(id="berhu-loss",
       spec_code="GB-C",
       title="berHu head loss",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the shipped berHu transform matches its analytic values and "
            "join derivatives and its anneal endpoints, and the berHu training "
            "smoke exits zero and prints the trunk-sqrt and head-berHu banners",
       evidence=(Assertion("berhu-loss.reference-values",
                           "training-stack.md#berhu-loss-reference-values"),
                 Assertion("berhu-loss.join-derivatives",
                           "training-stack.md#berhu-loss-join-derivatives"),
                 Assertion("berhu-loss.anneal-endpoints",
                           "training-stack.md#berhu-loss-anneal-endpoints"),
                 Assertion("berhu-loss.golden-selected-text-equality",
                           "training-stack.md#berhu-loss-golden-selected-text-equality"),
                 Assertion("berhu-loss.smoke-exit-zero",
                           "training-stack.md#berhu-loss-smoke-exit-zero"),
                 Assertion("berhu-loss.loss-banners",
                           "training-stack.md#berhu-loss-loss-banners")),
       run=gate_gb_c,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.berhu-loss-config",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="loss-schema-equivalence",
       spec_code="GL-D",
       title="Nested loss-schema equivalence",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the nested-loss smoke exits zero and prints the berHu-capped "
            "banner, while the golden byte-identity equivalence to the earlier "
            "schema remains conditional on a configured historical base",
       evidence=(Assertion("loss-schema-equivalence.golden-selected-text-equality",
                           "training-stack.md#loss-schema-equivalence-golden-selected-text-equality"),
                 Assertion("loss-schema-equivalence.smoke-exit-zero",
                           "training-stack.md#loss-schema-equivalence-smoke-exit-zero"),
                 Assertion("loss-schema-equivalence.berhu-banner",
                           "training-stack.md#loss-schema-equivalence-berhu-banner")),
       run=gate_gl_d,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.berhu-loss-config",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="berhu-anneal",
       spec_code="GBA-C",
       title="berHu anneal schedule",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the configured berHu-anneal run exits clean and names its "
            "anneal schedule banner; the schedule's continuity and "
            "full-shape epoch are not measured here.",
       evidence=(Assertion("berhu-anneal.golden-selected-text-equality",
                           "training-stack.md#berhu-anneal-golden-selected-text-equality"),
                 Assertion("berhu-anneal.smoke-exit-zero",
                           "training-stack.md#berhu-anneal-smoke-exit-zero"),
                 Assertion("berhu-anneal.anneal-banner",
                           "training-stack.md#berhu-anneal-anneal-banner"),
                 Assertion("berhu-anneal.schedule-behavior",
                           "training-stack.md#berhu-anneal-schedule-behavior")),
       run=gate_gba_c,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.berhu-anneal-config",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="ema-anneal",
       spec_code="GME-C",
       title="EMA anneal schedule",
       tier=TIER_BACKLOG,
       home="training-stack",
       maps="the EMA-anneal training smoke exits zero and prints both its "
            "\"ema: horizon 3 epochs\" horizon banner and its \"anneal: hold "
            "5 + 10 cosine\" schedule banner",
       evidence=(Assertion("ema-anneal.golden-selected-text-equality",
                           "training-stack.md#ema-anneal-golden-selected-text-equality"),
                 Assertion("ema-anneal.smoke-exit-zero",
                           "training-stack.md#ema-anneal-smoke-exit-zero"),
                 Assertion("ema-anneal.ema-anneal-banners",
                           "training-stack.md#ema-anneal-ema-anneal-banners"),
                 Assertion("ema-anneal.live-point-metrics",
                           "training-stack.md#ema-anneal-live-point-metrics")),
       run=gate_gme_c,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.ema-anneal-config",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="param-window-cuts",
       spec_code="item-27",
       title="Parameter-window cuts",
       tier=TIER_BACKLOG,
       home="data-generation-and-cuts",
       maps="the configured training driver runs and reports that a "
            "parameter-window cut was applied",
       evidence=(Assertion("param-window-cuts.driver-exit-zero",
                           "data-generation-and-cuts.md#param-window-cuts-driver-exit-zero"),
                 Assertion("param-window-cuts.cut-count-banner-present",
                           "data-generation-and-cuts.md#param-window-cuts-cut-count-banner-present"),
                 Assertion("param-window-cuts.init-probes-inspection",
                           "data-generation-and-cuts.md#param-window-cuts-init-probes-inspection")),
       run=gate_item27,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.param-window-cuts-config",) + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="triangle-shading",
       spec_code="GT-B",
       title="Triangle cut shading",
       tier=TIER_BACKLOG,
       home="data-generation-and-cuts",
       maps="a synthetic corner plot matches an independent panel/window "
            "reference, every cut artist uses the shared gray, a moved-artist "
            "mutation is rejected and the omegamh2 diagonal owns both bands",
       evidence=(Assertion("triangle-shading.figure-produced",
                           "data-generation-and-cuts.md#triangle-shading-figure-produced"),
                 Assertion("triangle-shading.panel-window-set-exact",
                           "data-generation-and-cuts.md#triangle-shading-panel-window-set-exact"),
                 Assertion("triangle-shading.all-cut-artists-use-shared-gray",
                           "data-generation-and-cuts.md#triangle-shading-all-cut-artists-use-shared-gray"),
                 Assertion("triangle-shading.omegamh2-marginal-bands-exact",
                           "data-generation-and-cuts.md#triangle-shading-omegamh2-marginal-bands-exact")),
       run=gate_gt_b,
       optional=True,
       manifest=Manifest(code=(), inputs=()),
       needs=("torch",)),

  Gate(id="joint-training",
       spec_code="GFT-C",
       title="freeze_trunk-false joint training",
       tier=TIER_NEW_FEATURES,
       home="training-stack",
       maps="the joint freeze_trunk:false run and the freeze_trunk:true "
            "control run each exit zero, the joint run prints its two-phase "
            "and phase-'joint' banners, while the golden selected-text "
            "equality, the phase-2 epoch-time ordering, and the handoff loss "
            "continuity remain conditional or inspection-only evidence",
       evidence=(Assertion("joint-training.golden-selected-text-equality",
                           "training-stack.md#joint-training-golden-selected-text-equality"),
                 Assertion("joint-training.joint-exit-zero",
                           "training-stack.md#joint-training-joint-exit-zero"),
                 Assertion("joint-training.two-phase-banner",
                           "training-stack.md#joint-training-two-phase-banner"),
                 Assertion("joint-training.joint-phase-banner",
                           "training-stack.md#joint-training-joint-phase-banner"),
                 Assertion("joint-training.control-exit-zero",
                           "training-stack.md#joint-training-control-exit-zero"),
                 Assertion("joint-training.epoch-time-order",
                           "training-stack.md#joint-training-epoch-time-order"),
                 Assertion("joint-training.handoff-loss-continuity",
                           "training-stack.md#joint-training-handoff-loss-continuity")),
       run=gate_gft_c,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.joint-training-config",
                                 "gate_configs.joint-training-control") + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="head-activation-pin",
       spec_code="GHA-F",
       title="Pinned head activation",
       tier=TIER_NEW_FEATURES,
       home="models-and-designs",
       maps="the pinned-head driver exits zero and prints its gated_power text, "
            "the --activation flag run warns that the pin wins, and the invalid "
            "unfrozen-head config is refused with a frozen-trunk message",
       evidence=(Assertion("head-activation-pin.golden-selected-text-equality",
                           "models-and-designs.md#head-activation-pin-golden-selected-text-equality"),
                 Assertion("head-activation-pin.pinned-config-exit-zero",
                           "models-and-designs.md#head-activation-pin-pinned-config-exit-zero"),
                 Assertion("head-activation-pin.gated-power-text-present",
                           "models-and-designs.md#head-activation-pin-gated-power-text-present"),
                 Assertion("head-activation-pin.flag-vs-pin-warning",
                           "models-and-designs.md#head-activation-pin-flag-vs-pin-warning"),
                 Assertion("head-activation-pin.unfrozen-pin-refusal",
                           "models-and-designs.md#head-activation-pin-unfrozen-pin-refusal")),
       run=gate_gha_f,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.head-activation-pin-config",
                                 "gate_configs.head-activation-pin-license") + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="relu-tanh-norm",
       spec_code="GAN-C",
       title="relu/tanh with the norm knob",
       tier=TIER_NEW_FEATURES,
       home="models-and-designs",
       maps="the tanh+per_feature and tanh+affine drivers each exit zero and "
            "name their norm in the banner, while the golden selected-text "
            "equality stays unavailable with no configured base",
       evidence=(Assertion("relu-tanh-norm.golden-selected-text-equality",
                           "models-and-designs.md#relu-tanh-norm-golden-selected-text-equality"),
                 Assertion("relu-tanh-norm.per-feature-config-exit-zero",
                           "models-and-designs.md#relu-tanh-norm-per-feature-config-exit-zero"),
                 Assertion("relu-tanh-norm.per-feature-text-present",
                           "models-and-designs.md#relu-tanh-norm-per-feature-text-present"),
                 Assertion("relu-tanh-norm.affine-config-exit-zero",
                           "models-and-designs.md#relu-tanh-norm-affine-config-exit-zero"),
                 Assertion("relu-tanh-norm.affine-text-present",
                           "models-and-designs.md#relu-tanh-norm-affine-text-present")),
       run=gate_gan_c,
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.relu-tanh-norm-per-feature",
                                 "gate_configs.relu-tanh-norm-affine") + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="weight-decay-census",
       spec_code="GWD-C",
       title="Weight-decay param-group census",
       tier=TIER_NEW_FEATURES,
       home="training-stack",
       maps="the real make_optimizer partitions a toy module tree by role: the "
            "decayed group is exactly the Linear/Conv1d/BinLinear weights, "
            "every non-allowlisted parameter stays undecayed, the two groups "
            "partition every parameter exactly once, and at weight decay 0 both "
            "groups are inert; the golden wd-0 selected-text equality stays "
            "unavailable while its base is null",
       evidence=(Assertion("weight-decay-census.allowed-weight-set",
                           "training-stack.md#weight-decay-census-allowed-weight-set"),
                 Assertion("weight-decay-census.undecayed-role-exclusions",
                           "training-stack.md#weight-decay-census-undecayed-role-exclusions"),
                 Assertion("weight-decay-census.parameter-group-partition",
                           "training-stack.md#weight-decay-census-parameter-group-partition"),
                 Assertion("weight-decay-census.zero-decay-inert",
                           "training-stack.md#weight-decay-census-zero-decay-inert"),
                 Assertion("weight-decay-census.golden-selected-text-equality",
                           "training-stack.md#weight-decay-census-golden-selected-text-equality")),
       run=gate_gwd_c,
       manifest=Manifest(code=(), inputs=()),
       needs=("torch", "gpu")),
  Gate(id="npce-training",
       spec_code="GPC-C",
       title="NPCE training",
       tier=TIER_NEW_FEATURES,
       home="models-and-designs",
       maps="the residual and ratio NPCE drivers each exit zero and name their "
            "pce form, the pce+ia and pce+--rescale combinations are both "
            "refused, and the two-point n_train sweep prints both result lines "
            "with its staging banner; the golden selected-text equality and the "
            "save/rebuild-vs-base round-trip both stay unavailable",
       evidence=(Assertion("npce-training.golden-selected-text-equality",
                           "models-and-designs.md#npce-training-golden-selected-text-equality"),
                 Assertion("npce-training.residual-config-exit-zero",
                           "models-and-designs.md#npce-training-residual-config-exit-zero"),
                 Assertion("npce-training.residual-pce-text-present",
                           "models-and-designs.md#npce-training-residual-pce-text-present"),
                 Assertion("npce-training.ratio-config-exit-zero",
                           "models-and-designs.md#npce-training-ratio-config-exit-zero"),
                 Assertion("npce-training.ratio-pce-text-present",
                           "models-and-designs.md#npce-training-ratio-pce-text-present"),
                 Assertion("npce-training.pce-ia-refusal",
                           "models-and-designs.md#npce-training-pce-ia-refusal"),
                 Assertion("npce-training.pce-rescale-refusal",
                           "models-and-designs.md#npce-training-pce-rescale-refusal"),
                 Assertion("npce-training.sweep-result-lines-and-pce-banner",
                           "models-and-designs.md#npce-training-sweep-result-lines-and-pce-banner"),
                 Assertion("npce-training.rebuild-vs-base",
                           "models-and-designs.md#npce-training-rebuild-vs-base")),
       run=gate_gpc_c,
       manifest=Manifest(code=_CS_TRAIN_CODE
                              + ("cosmic_shear_sweep_ntrain_emulator.py",),
                         inputs=("gate_configs.npce-training-residual",
                                 "gate_configs.npce-training-ratio",
                                 "gate_configs.npce-training-excl-ia",
                                 "gate_configs.npce-training-excl-rescale",
                                 "gate_configs.npce-training-sweep") + _CS_DEPLOY_DATA),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="finetune-identity",
       spec_code="FTW-A",
       title="Fine-tune warm-start identity",
       tier=TIER_NEW_FEATURES,
       home="artifacts-inference-warmstart",
       maps="256-273 (encode + transfer + parity + degenerate + error paths)",
       run=gate_ftw_a,
       manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="transfer-identity",
       spec_code="TPE-A",
       title="Transfer frozen-base identity",
       tier=TIER_NEW_FEATURES,
       home="artifacts-inference-warmstart",
       maps="204-219 (slice + 4x form/space identity + packing + surgery + errors)",
       run=gate_tpe_a,
       manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="scalar-identity",
       spec_code="SPE-A",
       title="Scalar emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-scalar-cmb",
       maps="synthetic scalar artifacts exercise the save/rebuild round trip, "
            "the geometry and schema guards, the Cobaya adapter contract, the "
            "NPCE residual composition, and the fine-tune parity boundary",
       evidence=(Assertion("scalar-identity.artifact-round-trip",
                           "families-scalar-cmb.md#scalar-identity-artifact-round-trip"),
                 Assertion("scalar-identity.geometry-and-schema-guards",
                           "families-scalar-cmb.md#scalar-identity-geometry-and-schema-guards"),
                 Assertion("scalar-identity.scalar-adapter-contract",
                           "families-scalar-cmb.md#scalar-identity-scalar-adapter-contract"),
                 Assertion("scalar-identity.npce-composition",
                           "families-scalar-cmb.md#scalar-identity-npce-composition"),
                 Assertion("scalar-identity.finetune-parity",
                           "families-scalar-cmb.md#scalar-identity-finetune-parity")),
       run=gate_spe_a,
       manifest=Manifest(code=("emulator/designs", "emulator/losses",
                               "cobaya_theory/emul_scalars.py"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="cmb-identity",
       spec_code="CME-A",
       title="CMB emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-scalar-cmb",
       maps="synthetic CMB artifacts exercise the diagonal geometry and "
            "reference schema, the amplitude law and score, the saved "
            "predictor and adapter round trip, the roughness contract, the "
            "model-variant composition, the fine-tune parity boundary, and "
            "the non-Gaussian covariance known answer",
       evidence=(Assertion("cmb-identity.geometry-and-reference-schema",
                           "families-scalar-cmb.md#cmb-identity-geometry-and-reference-schema"),
                 Assertion("cmb-identity.amplitude-law-and-score",
                           "families-scalar-cmb.md#cmb-identity-amplitude-law-and-score"),
                 Assertion("cmb-identity.artifact-and-adapter-round-trip",
                           "families-scalar-cmb.md#cmb-identity-artifact-and-adapter-round-trip"),
                 Assertion("cmb-identity.roughness-contract",
                           "families-scalar-cmb.md#cmb-identity-roughness-contract"),
                 Assertion("cmb-identity.model-variant-composition",
                           "families-scalar-cmb.md#cmb-identity-model-variant-composition"),
                 Assertion("cmb-identity.finetune-parity",
                           "families-scalar-cmb.md#cmb-identity-finetune-parity"),
                 Assertion("cmb-identity.covariance-known-answer",
                           "families-scalar-cmb.md#cmb-identity-covariance-known-answer")),
       run=gate_cme_a,
       manifest=Manifest(code=("emulator/designs", "emulator/losses",
                               "cobaya_theory/emul_cmb.py",
                               "compute_data_vectors/compute_cmb_covariance.py"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="bsn-identity",
       spec_code="BSN-A",
       title="BAOSN grid emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-background-mps",
       maps="synthetic background artifacts exercise the integration rule, "
            "distance construction, grid geometry and saved predictor, the "
            "emul_baosn two-window adapter, the NPCE residual model, and the "
            "fine-tune boundary",
       evidence=(Assertion("bsn-identity.simpson-polynomial-nodes",
                           "families-background-mps.md#bsn-identity-simpson-polynomial-nodes"),
                 Assertion("bsn-identity.distance-pipeline-consistency",
                           "families-background-mps.md#bsn-identity-distance-pipeline-consistency"),
                 Assertion("bsn-identity.geometry-and-artifact-round-trip",
                           "families-background-mps.md#bsn-identity-geometry-and-artifact-round-trip"),
                 Assertion("bsn-identity.adapter-piecewise-contract",
                           "families-background-mps.md#bsn-identity-adapter-piecewise-contract"),
                 Assertion("bsn-identity.npce-composition",
                           "families-background-mps.md#bsn-identity-npce-composition"),
                 Assertion("bsn-identity.finetune-parity",
                           "families-background-mps.md#bsn-identity-finetune-parity")),
       run=gate_bsn_a,
       manifest=Manifest(code=("emulator/designs", "emulator/losses",
                               "cobaya_theory/emul_baosn.py"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="mps-identity",
       spec_code="MPS-A",
       title="MPS grid2d emulator identity",
       tier=TIER_NEW_FEATURES,
       home="families-background-mps",
       maps="the matter-power geometry, bounded staging and its "
            "temporary-file lifecycle, saved model variants, adapter "
            "assembly, config validation, and fine-tuning legs",
       evidence=(Assertion("mps-identity.geometry-laws-and-pins",
                           "families-background-mps.md#mps-identity-geometry-laws-and-pins"),
                 Assertion("mps-identity.bounded-staging-values",
                           "families-background-mps.md#mps-identity-bounded-staging-values"),
                 Assertion("mps-identity.stable-streamed-moments",
                           "families-background-mps.md#mps-identity-stable-streamed-moments"),
                 Assertion("mps-identity.staging-file-lifecycle",
                           "families-background-mps.md#mps-identity-staging-file-lifecycle"),
                 Assertion("mps-identity.saved-model-variants",
                           "families-background-mps.md#mps-identity-saved-model-variants"),
                 Assertion("mps-identity.adapter-assembly-and-defaults",
                           "families-background-mps.md#mps-identity-adapter-assembly-and-defaults"),
                 Assertion("mps-identity.config-and-finetune",
                           "families-background-mps.md#mps-identity-config-and-finetune")),
       run=gate_mps_a,
       manifest=Manifest(code=("emulator/designs", "emulator/losses",
                               "cobaya_theory/emul_mps.py"),
                         inputs=()),
       needs=("torch",)),
  Gate(id="geo-paths",
       spec_code="GEO-A",
       title="Geometry folder is the only geometry home",
       tier=TIER_NEW_FEATURES,
       home="artifacts-inference-warmstart",
       maps="fresh artifacts name geometry classes from the geometry package "
            "(emulator.geometries.*), and the retired flat module paths stay "
            "absent from disk, the import system, and the repository source",
       evidence=(Assertion("geo-paths.fresh-save-uses-folder-paths",
                           "artifacts-inference-warmstart.md#geo-paths-fresh-save-uses-folder-paths"),
                 Assertion("geo-paths.legacy-flat-paths-absent",
                           "artifacts-inference-warmstart.md#geo-paths-legacy-flat-paths-absent"),
                 Assertion("geo-paths.legacy-reference-census",
                           "artifacts-inference-warmstart.md#geo-paths-legacy-reference-census")),
       run=gate_geo_a,
       manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                         inputs=()),
       needs=("torch",)),

  Gate(id="save-rebuild-drift",
       spec_code="GSV-C",
       title="Save/rebuild bitwise + drift",
       tier=TIER_SAVE_AND_SAMPLE,
       home="artifacts-inference-warmstart",
       maps="86-93 (bitwise + drift + v1 refusal); "
            "gates-and-board.md:66-71 (one factored + one NPCE save)",
       run=gate_gsv_c,
       manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                         inputs=()),
       needs=("torch", "cosmolike", "gpu")),
  Gate(id="cobaya-adapter",
       spec_code="GCT-C",
       title="Cobaya adapter parity",
       tier=TIER_SAVE_AND_SAMPLE,
       home="artifacts-inference-warmstart",
       maps="117-123 (parity rtol 1e-6 + evaluate + MCMC); 234-238 (round-trip)",
       run=gate_gct_c,
       manifest=Manifest(code=("emulator/designs", "emulator/losses"),
                         inputs=("evaluate_yaml",)),
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
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.finetune-smoke-config",) + _CS_DEPLOY_DATA),
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
       manifest=Manifest(code=_CS_TRAIN_CODE,
                         inputs=("gate_configs.transfer-smoke-config",) + _CS_DEPLOY_DATA),
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
       manifest=Manifest(code=("emulator/designs", "emulator/losses",
                               "cobaya_theory/emul_scalars.py"),
                         inputs=()),
       needs=("torch", "cobaya")),
  Gate(id="cmb-smoke",
       spec_code="CME-B",
       title="CMB emulator smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="families-scalar-cmb",
       maps="a small real-CAMB fixture drives the CMB spectrum generator, "
            "the Gaussian and eq-6 non-diagonal covariance builders, the "
            "training collapse bar, the real Cobaya provider, and the "
            "family diagnostics pages end to end",
       evidence=(Assertion("cmb-smoke.generated-spectrum-dumps",
                           "families-scalar-cmb.md#cmb-smoke-generated-spectrum-dumps"),
                 Assertion("cmb-smoke.gaussian-covariance",
                           "families-scalar-cmb.md#cmb-smoke-gaussian-covariance"),
                 Assertion("cmb-smoke.nondiagonal-covariance-structure",
                           "families-scalar-cmb.md#cmb-smoke-nondiagonal-covariance-structure"),
                 Assertion("cmb-smoke.training-collapse",
                           "families-scalar-cmb.md#cmb-smoke-training-collapse"),
                 Assertion("cmb-smoke.cobaya-serving",
                           "families-scalar-cmb.md#cmb-smoke-cobaya-serving"),
                 Assertion("cmb-smoke.diagnostics-output",
                           "families-scalar-cmb.md#cmb-smoke-diagnostics-output")),
       run=gate_cme_b,
       manifest=Manifest(
           code=("emulator/designs", "emulator/losses",
                 "cobaya_theory/emul_cmb.py",
                 "compute_data_vectors/dataset_generator_cmb.py",
                 "compute_data_vectors/compute_cmb_covariance.py"),
           inputs=()),
       needs=("torch", "cobaya")),
  Gate(id="bsn-smoke",
       spec_code="BSN-B",
       title="BAOSN emulator smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="families-background-mps",
       maps="the generated background dumps, the dead-network-relative "
            "training collapse, the Cobaya-vs-CAMB comparison, and the "
            "grid diagnostics output legs",
       evidence=(Assertion("bsn-smoke.generated-background-dumps",
                           "families-background-mps.md#bsn-smoke-generated-background-dumps"),
                 Assertion("bsn-smoke.training-collapse",
                           "families-background-mps.md#bsn-smoke-training-collapse"),
                 Assertion("bsn-smoke.cobaya-vs-camb",
                           "families-background-mps.md#bsn-smoke-cobaya-vs-camb"),
                 Assertion("bsn-smoke.diagnostics-output",
                           "families-background-mps.md#bsn-smoke-diagnostics-output")),
       run=gate_bsn_b,
       manifest=Manifest(
           code=("emulator/designs", "emulator/losses",
                 "cobaya_theory/emul_baosn.py",
                 "compute_data_vectors/dataset_generator_background.py"),
           inputs=()),
       needs=("torch", "cobaya")),
  Gate(id="mps-smoke",
       spec_code="MPS-B",
       title="MPS emulator smoke",
       tier=TIER_SAVE_AND_SAMPLE,
       home="families-background-mps",
       maps="the matter-power law-none smoke: generated linear-power and "
            "boost dumps, both trainings collapsing, the grid2d diagnostics "
            "output, and the emul_mps lifecycle vs CAMB's own P(k, z)",
       evidence=(Assertion("mps-smoke.generated-power-dumps",
                           "families-background-mps.md#mps-smoke-generated-power-dumps"),
                 Assertion("mps-smoke.training-collapse",
                           "families-background-mps.md#mps-smoke-training-collapse"),
                 Assertion("mps-smoke.diagnostics-output",
                           "families-background-mps.md#mps-smoke-diagnostics-output"),
                 Assertion("mps-smoke.cobaya-vs-camb",
                           "families-background-mps.md#mps-smoke-cobaya-vs-camb")),
       run=gate_mps_b,
       manifest=Manifest(
           code=("emulator/designs", "emulator/losses",
                 "cobaya_theory/emul_mps.py",
                 "compute_data_vectors/dataset_generator_mps.py"),
           inputs=()),
       needs=("torch", "cobaya")),
]

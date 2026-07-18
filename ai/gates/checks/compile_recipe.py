#!/usr/bin/env python3
"""Prove that a CUDA rebuild consumes its persisted compile mode.

The check has two ordered evidence legs. The CPU leg writes two current
schema-v3 scalar artifacts whose recipes carry distinct modes (``default``
and ``reduce-overhead``). It reads those values independently, exercises the
verdict's bad traces, and proves production refuses a recipe that loses the
field. The CUDA leg first proves that both modes can compile a tiny forward on
this machine. It then rebuilds both artifacts with ``compile_model=True``
while a transparent wrapper records and delegates each real ``torch.compile``
call. Each returned callable must run a finite forward.

This proves mode consumption, not PyTorch's internal optimization strategy.
Without CUDA, the CPU leg still runs but the CUDA leg emits ``UNAVAILABLE``
and the process returns 2. Missing workstation evidence is never a pass.
"""

import sys
import tempfile
from pathlib import Path

import h5py
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from emulator import fixed_facts
from emulator.activations import make_activation
from emulator.model_recipe import set_runtime_compile_mode
from emulator.designs.blocks import make_norm
from emulator.designs.plain import ResMLP
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry
from emulator.results import rebuild_emulator, save_emulator
from ai.gates.checks.artifact_fixtures import one_pass_training_recipe


COMPILE_MODES = ("default", "reduce-overhead")
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_LANE_UNAVAILABLE = 2

LEG_AIDS = (
  "compile-recipe.observation-controls",
  "compile-recipe.cuda-persisted-modes",
)
CPU_AID, CUDA_AID = LEG_AIDS


def report(label, ok, detail):
  """Print one named control and return its boolean verdict."""
  mark = "PASS" if ok else "FAIL"
  print("  [" + mark + "] " + label + "  (" + detail + ")")
  return bool(ok)


def emit_aid(aid, result, reason=""):
  """Emit one board-consumed terminal for an acceptance leg."""
  line = "##AID " + aid + " " + result
  if reason:
    line += " " + reason
  print(line)


def observation_verdict(expected_mode, observed_modes, compiler_inputs,
                        compiler_returns, rebuilt_callable, compile_error):
  """Judge one artifact's recorded compiler calls.

  Exactly one successful call must carry the value independently read from
  that artifact. Its delegated result must differ from the eager input, and
  production must return that exact result. Missing, duplicated, substituted,
  identity, discarded-return, or raising traces fail.
  """
  if compile_error is not None:
    return False, "compile/rebuilt-forward raised: " + compile_error
  if len(observed_modes) != 1:
    return (False,
            "expected one torch.compile call, observed "
            + repr(observed_modes))
  if len(compiler_inputs) != 1 or len(compiler_returns) != 1:
    return (False,
            "expected one compiler input/result pair, observed "
            + str(len(compiler_inputs)) + "/"
            + str(len(compiler_returns)))
  observed_mode = observed_modes[0]
  if observed_mode != expected_mode:
    return (False,
            "saved mode " + repr(expected_mode)
            + ", observed mode " + repr(observed_mode))
  eager_input = compiler_inputs[0]
  delegated_result = compiler_returns[0]
  if delegated_result is eager_input:
    return False, "torch.compile returned its eager input unchanged"
  if rebuilt_callable is not delegated_result:
    return False, "rebuild discarded the delegated compiler result"
  return (True,
          "saved and observed mode " + repr(expected_mode)
          + "; rebuild returned the delegated non-identity result")


def paired_observation_verdict(expected_modes, observed_by_artifact,
                               compiler_inputs_by_artifact,
                               compiler_returns_by_artifact,
                               rebuilt_callables, compile_errors):
  """Judge the ordered observations for both distinct-mode artifacts."""
  if len(observed_by_artifact) != len(expected_modes):
    return (False,
            "expected " + str(len(expected_modes)) + " artifact traces, got "
            + str(len(observed_by_artifact)))
  if len(compile_errors) != len(expected_modes):
    return (False,
            "expected " + str(len(expected_modes)) + " error slots, got "
            + str(len(compile_errors)))
  trace_groups = (
    ("compiler-input", compiler_inputs_by_artifact),
    ("compiler-return", compiler_returns_by_artifact),
    ("rebuild-return", rebuilt_callables),
  )
  for label, group in trace_groups:
    if len(group) != len(expected_modes):
      return (False,
              "expected " + str(len(expected_modes)) + " " + label
              + " traces, got " + str(len(group)))
  details = []
  for expected, observed, inputs, returns, rebuilt, error in zip(
      expected_modes, observed_by_artifact, compiler_inputs_by_artifact,
      compiler_returns_by_artifact, rebuilt_callables, compile_errors):
    ok, detail = observation_verdict(
      expected_mode=expected,
      observed_modes=observed,
      compiler_inputs=inputs,
      compiler_returns=returns,
      rebuilt_callable=rebuilt,
      compile_error=error)
    details.append(expected + ": " + detail)
    if not ok:
      return False, "; ".join(details)
  return True, "; ".join(details)


def model_recipe(compile_mode):
  """Return the complete current rebuild recipe for the tiny fixture."""
  return {
    "cls": "emulator.designs.plain.ResMLP",
    "name": "resmlp",
    "ia": None,
    "input_dim": 2,
    "output_dim": 1,
    "compile_mode": compile_mode,
    "needs_geom": False,
    "kwargs": {
      "int_dim_res": 4,
      "n_blocks": 1,
      "block_opts": {
        "n_layers": 2,
        "act": {"type": "H", "n_gates": 3},
        "norm": "affine",
      },
    },
  }


def save_fixture(path_root, compile_mode, case_label, output_identity=None,
                 resolved_rescale="none", recorded_rescale="none"):
  """Write one current schema-v3 scalar artifact carrying ``compile_mode``."""
  cpu = torch.device("cpu")
  pgeom = ParamGeometry(
    device=cpu,
    names=["p0", "p1"],
    center=[0.0, 0.0],
    evecs=[[1.0, 0.0], [0.0, 1.0]],
    sqrt_ev=[1.0, 1.0])
  geom = ScalarGeometry(
    device=cpu,
    names=["derived"],
    center=[0.0],
    scale=[1.0])
  block_opts = {
    "act": make_activation("H", n_gates=3),
    "norm": make_norm("affine"),
  }
  torch.manual_seed(93 + COMPILE_MODES.index(compile_mode))
  model = ResMLP(
    input_dim=2,
    output_dim=1,
    int_dim_res=4,
    n_blocks=1,
    block_opts=block_opts).to(cpu)
  set_runtime_compile_mode(model, compile_mode)
  config = {
    "data": {},
    "train_args": {"nepochs": 1},
  }
  histories = {
    "train_losses": [0.1],
    "val_medians": [0.1],
    "val_means": [0.1],
    "val_fracs": [torch.tensor([0.5])],
    "thresholds": torch.tensor([1.0]),
  }
  save_emulator(
    path_root=str(path_root),
    model=model,
    param_geometry=pgeom,
    geometry=geom,
    config=config,
    histories=histories,
    train_args=config["train_args"],
    resolved_train=one_pass_training_recipe(
      thresholds=(1.0,), compile_mode=compile_mode),
    resolved_model=model_recipe(compile_mode),
    composition_mode="plain",
    transfer_refined=False,
    resolved_pce=None,
    resolved_transfer=None,
    resolved_rescale=resolved_rescale,
    output_identity=output_identity,
    facts_yaml=fixed_facts.synthetic_sidecar(
      names=pgeom.state()["names"],
      label="unit-93-compile-" + case_label,
      family="scalar",
      support=None),
    attrs={"rescale": recorded_rescale})


def _recipe_mapping(artifact):
  """Decode one artifact's model-recipe YAML as a plain mapping."""
  raw = artifact["model_recipe"][()]
  if isinstance(raw, bytes):
    raw = raw.decode("utf-8")
  recipe = yaml.safe_load(raw)
  if type(recipe) is not dict:
    raise ValueError("model_recipe must decode to a mapping")
  return recipe


def read_saved_mode(path_root):
  """Independently read the current schema and persisted compile mode."""
  with h5py.File(str(path_root) + ".h5", "r") as artifact:
    version = artifact.attrs.get("schema_version")
    if int(version) != fixed_facts.SCHEMA_VERSION:
      raise ValueError(
        "fixture schema " + repr(version) + " is not current schema "
        + repr(fixed_facts.SCHEMA_VERSION))
    recipe = _recipe_mapping(artifact)
  if "compile_mode" not in recipe:
    raise KeyError("saved model_recipe is missing 'compile_mode'")
  return recipe["compile_mode"]


def delete_saved_mode(path_root):
  """Forge one otherwise-current artifact whose recipe loses the field."""
  with h5py.File(str(path_root) + ".h5", "r+") as artifact:
    recipe = _recipe_mapping(artifact)
    if "compile_mode" not in recipe:
      raise KeyError("fixture already lacks compile_mode")
    del recipe["compile_mode"]
    del artifact["model_recipe"]
    artifact.create_dataset(
      "model_recipe",
      data=yaml.safe_dump(recipe, sort_keys=False),
      dtype=h5py.string_dtype(encoding="utf-8"))


def run_trace_controls():
  """Exercise every good and bad two-artifact observation on the CPU."""
  controls_ok = True
  eager_default = object()
  eager_reduce = object()
  compiled_default = object()
  compiled_reduce = object()
  good_inputs = [[eager_default], [eager_reduce]]
  good_returns = [[compiled_default], [compiled_reduce]]
  good_rebuilt = [compiled_default, compiled_reduce]

  def check(label, observed, errors, expected_ok, inputs=None, returns=None,
            rebuilt=None):
    nonlocal controls_ok
    ok, detail = paired_observation_verdict(
      expected_modes=COMPILE_MODES,
      observed_by_artifact=observed,
      compiler_inputs_by_artifact=(good_inputs if inputs is None else inputs),
      compiler_returns_by_artifact=(good_returns
                                    if returns is None else returns),
      rebuilt_callables=(good_rebuilt if rebuilt is None else rebuilt),
      compile_errors=errors)
    controls_ok &= report(label, ok == expected_ok, detail)

  check("one call per artifact receives its saved mode",
        [["default"], ["reduce-overhead"]], [None, None], True)
  check("a lost compiler call is rejected",
        [[], ["reduce-overhead"]], [None, None], False)
  check("a duplicated compiler call is rejected",
        [["default", "default"], ["reduce-overhead"]],
        [None, None], False)
  check("a compiler or rebuilt-forward exception is rejected",
        [["default"], ["reduce-overhead"]],
        [None, "synthetic compiler failure"], False)
  check("substituting the artifacts' modes is rejected",
        [["reduce-overhead"], ["default"]], [None, None], False)
  check("hard-coding default is rejected by the other artifact",
        [["default"], ["default"]], [None, None], False)
  check("hard-coding reduce-overhead is rejected by the other artifact",
        [["reduce-overhead"], ["reduce-overhead"]],
        [None, None], False)
  check("an identity compiler result is rejected",
        [["default"], ["reduce-overhead"]], [None, None], False,
        returns=[[eager_default], [compiled_reduce]],
        rebuilt=[eager_default, compiled_reduce])
  check("a discarded delegated result is rejected",
        [["default"], ["reduce-overhead"]], [None, None], False,
        rebuilt=[eager_default, compiled_reduce])
  return bool(controls_ok)


def run_cpu_controls():
  """Run verdict controls and current-schema save/read/refusal controls."""
  controls_ok = run_trace_controls()
  with tempfile.TemporaryDirectory(prefix="compile-recipe-cpu-") as tmp:
    roots = []
    for case_label, mode in zip(("case-a", "case-b"), COMPILE_MODES):
      root = Path(tmp) / case_label
      roots.append(root)
      try:
        save_fixture(
          path_root=root, compile_mode=mode, case_label=case_label)
        saved_mode = read_saved_mode(path_root=root)
      except Exception as exc:
        controls_ok &= report(
          "current schema-v3 save persists " + mode,
          False,
          type(exc).__name__ + ": " + str(exc))
      else:
        controls_ok &= report(
          "current schema-v3 save persists " + mode,
          saved_mode == mode,
          "saved mode " + repr(saved_mode))

    try:
      delete_saved_mode(path_root=roots[0])
      rebuild_emulator(
        path_root=str(roots[0]),
        device=torch.device("cpu"),
        compile_model=True)
    except (KeyError, ValueError) as exc:
      text = str(exc)
      refused = "compile_mode" in text and "missing" in text
      controls_ok &= report(
        "production refuses a missing persisted compile_mode",
        refused,
        text)
    except Exception as exc:
      controls_ok &= report(
        "production refuses a missing persisted compile_mode",
        False,
        "wrong exception " + type(exc).__name__ + ": " + str(exc))
    else:
      controls_ok &= report(
        "production refuses a missing persisted compile_mode",
        False,
        "rebuild unexpectedly accepted the forged recipe")
  return bool(controls_ok)


def compile_capability():
  """Prove this machine can compile and execute a tiny forward in both modes."""
  if not torch.cuda.is_available():
    return False, "CUDA is not available"
  if not hasattr(torch, "compile"):
    return False, "this PyTorch build has no torch.compile"
  device = torch.device("cuda")
  for mode in COMPILE_MODES:
    try:
      eager = torch.nn.Linear(2, 1).to(device)
      compiled = torch.compile(eager, mode=mode)
      probe = torch.tensor([[1.0, 2.0]], device=device)
      with torch.no_grad():
        output = compiled(probe)
      torch.cuda.synchronize()
      if not bool(torch.isfinite(output).all()):
        return False, mode + " capability forward returned a nonfinite value"
    except Exception as exc:
      return (False,
              mode + " capability failed: "
              + type(exc).__name__ + ": " + str(exc))
  return True, "compiled forwards ran in " + repr(COMPILE_MODES)


def run_cuda_lane():
  """Run both persisted modes through the real compiled CUDA rebuild path."""
  available, capability_detail = compile_capability()
  if not available:
    return "UNAVAILABLE", capability_detail

  try:
    with tempfile.TemporaryDirectory(prefix="compile-recipe-cuda-") as tmp:
      roots = []
      saved_modes = []
      for case_label, mode in zip(("case-a", "case-b"), COMPILE_MODES):
        root = Path(tmp) / case_label
        save_fixture(
          path_root=root, compile_mode=mode, case_label=case_label)
        roots.append(root)
        saved_modes.append(read_saved_mode(path_root=root))
      if tuple(saved_modes) != COMPILE_MODES:
        return ("FAIL",
                "fixtures persisted " + repr(tuple(saved_modes))
                + " instead of " + repr(COMPILE_MODES))

      original_compile = torch.compile
      observed_by_artifact = []
      compiler_inputs_by_artifact = []
      compiler_returns_by_artifact = []
      rebuilt_callables = []
      compile_errors = []
      device = torch.device("cuda")
      for root in roots:
        observed = []
        compiler_inputs = []
        compiler_returns = []
        rebuilt_callable = None
        compile_error = None

        def recording_compile(model, **kwargs):
          observed.append(kwargs.get("mode"))
          compiler_inputs.append(model)
          delegated = original_compile(model, **kwargs)
          compiler_returns.append(delegated)
          return delegated

        try:
          torch.compile = recording_compile
          rebuilt, pgeom, _geom, _info = rebuild_emulator(
            path_root=str(root),
            device=device,
            compile_model=True)
          rebuilt_callable = rebuilt
          probe = torch.tensor(
            [[0.25, -0.5], [1.0, 2.0]],
            dtype=torch.float32,
            device=device)
          with torch.no_grad():
            output = rebuilt(pgeom.encode(probe))
          torch.cuda.synchronize()
          if not bool(torch.isfinite(output).all()):
            compile_error = "rebuilt callable returned a nonfinite value"
        except Exception as exc:
          compile_error = type(exc).__name__ + ": " + str(exc)
        finally:
          torch.compile = original_compile
        observed_by_artifact.append(observed)
        compiler_inputs_by_artifact.append(compiler_inputs)
        compiler_returns_by_artifact.append(compiler_returns)
        rebuilt_callables.append(rebuilt_callable)
        compile_errors.append(compile_error)

      ok, detail = paired_observation_verdict(
        expected_modes=saved_modes,
        observed_by_artifact=observed_by_artifact,
        compiler_inputs_by_artifact=compiler_inputs_by_artifact,
        compiler_returns_by_artifact=compiler_returns_by_artifact,
        rebuilt_callables=rebuilt_callables,
        compile_errors=compile_errors)
      if not ok:
        return "FAIL", detail
      return "PASS", detail + "; " + capability_detail
  except Exception as exc:
    return "FAIL", type(exc).__name__ + ": " + str(exc)


def main():
  """Run both evidence legs and return 0, 1, or non-green unavailable 2."""
  print("== compile-recipe: persisted modes reach CUDA torch.compile ==")
  print("\n-- CPU observation and schema controls --")
  try:
    cpu_ok = run_cpu_controls()
  except Exception as exc:
    report("CPU controls completed", False,
           type(exc).__name__ + ": " + str(exc))
    cpu_ok = False
  emit_aid(CPU_AID, "PASS" if cpu_ok else "FAIL")
  if not cpu_ok:
    emit_aid(CUDA_AID, "UNAVAILABLE", "cpu-controls-failed")
    return EXIT_FAIL

  print("\n-- CUDA persisted-mode observations --")
  try:
    cuda_result, cuda_detail = run_cuda_lane()
  except Exception as exc:
    cuda_result = "FAIL"
    cuda_detail = type(exc).__name__ + ": " + str(exc)
  print("  [" + cuda_result + "] " + cuda_detail)
  reason = "workstation-cuda-proof-owed" \
    if cuda_result == "UNAVAILABLE" else ""
  emit_aid(CUDA_AID, cuda_result, reason)
  if cuda_result == "PASS":
    print("\ncompile-recipe: ALL REQUIRED LANES PASS")
    return EXIT_PASS
  if cuda_result == "UNAVAILABLE":
    print("\ncompile-recipe: CUDA LANE UNAVAILABLE; status 2 is non-green")
    return EXIT_LANE_UNAVAILABLE
  print("\ncompile-recipe: CUDA LANE FAILED")
  return EXIT_FAIL


if __name__ == "__main__":
  sys.exit(main())

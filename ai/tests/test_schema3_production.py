"""CPU checks for the required scientific record on newly saved emulators."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from ai.gates.checks.compile_recipe import save_fixture
from emulator import data_staging, experiment, fixed_facts, results


class _StateDictMustNotRun:
  """Model stand-in that exposes an early-validation ordering mistake."""

  def state_dict(self):
    raise AssertionError("model.state_dict must not run after early refusal")


class _NamedGeometry:
  """Input-geometry stand-in used only by pre-serialization checks."""

  def __init__(self, names):
    self.names = list(names)

  def state(self):
    return {"names": list(self.names)}


def _reader_shape_recipe():
  """Return the smallest recipe carrying every field the reader consumes."""
  return {
    "cls": "emulator.designs.plain.ResMLP",
    "ia": None,
    "input_dim": 1,
    "output_dim": 1,
    "compile_mode": None,
    "needs_geom": False,
    "kwargs": {
      "block_opts": {
        "act": {"type": "H", "n_gates": 3},
        "norm": "affine",
      },
    },
  }


def _save_attempt(root, *, facts_yaml, resolved_train, resolved_model,
                  names=("p0",)):
  """Call save_emulator with sentinels that must remain untouched."""
  return results.save_emulator(
    path_root=str(root),
    model=_StateDictMustNotRun(),
    param_geometry=_NamedGeometry(names),
    geometry=None,
    config={},
    histories={},
    resolved_train=resolved_train,
    resolved_model=resolved_model,
    facts_yaml=facts_yaml,
    composition_mode="plain",
    transfer_refined=False,
    resolved_pce=None,
    resolved_transfer=None)


def _facts_path(params_path):
  """Return the shared facts path for the numbered chain used in tests."""
  base = os.path.splitext(os.fspath(params_path))[0]
  root, chain_ext = os.path.splitext(base)
  if chain_ext[1:].isdigit():
    base = root
  return base + fixed_facts.SIDECAR_SUFFIX


def _write_facts(params_path, *, names, label, text=None):
  """Write one honest synthetic record beside a test parameter chain."""
  if text is None:
    text = fixed_facts.synthetic_sidecar(
      names=names, label=label, family="scalar", support=None)
  path = _facts_path(params_path)
  with open(path, "w", encoding="utf-8", newline="") as handle:
    handle.write(text)
  return text


def _preflight_config(directory):
  """Return the smallest config that reaches the facts preflight."""
  return {
    "data": {
      "train_params": os.path.join(directory, "train.1.txt"),
      "val_params": os.path.join(directory, "val.1.txt"),
      "train_covmat": os.path.join(directory, "params.covmat"),
      "outputs": ["derived"],
      "n_train": 1,
      "n_val": 1,
      "split_seed": 7,
    },
    "train_args": {
      "model": {
        "name": "resmlp",
        "mlp": {"width": 4, "n_blocks": 0},
      },
    },
  }


class Schema3ProductionTests(unittest.TestCase):
  """Prove refusals happen before files and valid schema-3 output rebuilds."""

  @staticmethod
  def _assert_no_artifacts(directory):
    """No final file, temporary stage, or pair marker may survive refusal."""
    entries = os.listdir(directory)
    if entries:
      raise AssertionError("early refusal created output: " + repr(entries))

  def test_missing_facts_refuses_before_model_or_output(self):
    """There is no new schema-less or schema-2 writer path."""
    with tempfile.TemporaryDirectory(prefix="schema3-missing-facts-") as tmp:
      root = Path(tmp) / "artifact"
      with mock.patch.object(
          results, "_new_staging_path",
          side_effect=AssertionError("no temporary path may be reserved")) \
          as reserve:
        with self.assertRaisesRegex(
            ValueError, r"requires.*\.facts\.yaml.*Re-generate"):
          _save_attempt(
            root,
            facts_yaml=None,
            resolved_train={"nepochs": 1},
            resolved_model=_reader_shape_recipe())

      reserve.assert_not_called()
      self._assert_no_artifacts(tmp)

  def test_nontext_facts_refuse_before_model_or_output(self):
    """Parsed mappings and undecoded bytes cannot masquerade as exact text."""
    with tempfile.TemporaryDirectory(prefix="schema3-nontext-facts-") as tmp:
      for index, value in enumerate((b"fixed_facts: {}", {"fixed_facts": {}})):
        with self.subTest(value_type=type(value).__name__):
          root = Path(tmp) / ("artifact-" + str(index))
          with self.assertRaisesRegex(TypeError, r"must be.*text"):
            _save_attempt(
              root,
              facts_yaml=value,
              resolved_train={"nepochs": 1},
              resolved_model=_reader_shape_recipe())
      self._assert_no_artifacts(tmp)

  def test_each_missing_recipe_refuses_before_model_or_output(self):
    """Both consumed recipes are required before serialization starts."""
    facts = fixed_facts.synthetic_sidecar(
      names=["p0"], label="missing-recipe", support=None)
    cases = (
      (None, _reader_shape_recipe(), "resolved_train"),
      ({"nepochs": 1}, None, "resolved_model"),
      (None, None, "resolved_train and resolved_model"),
    )
    with tempfile.TemporaryDirectory(prefix="schema3-missing-recipe-") as tmp:
      for index, (train, model, diagnostic) in enumerate(cases):
        with self.subTest(missing=diagnostic):
          root = Path(tmp) / ("artifact-" + str(index))
          with self.assertRaisesRegex(ValueError, diagnostic):
            _save_attempt(
              root,
              facts_yaml=facts,
              resolved_train=train,
              resolved_model=model)
      self._assert_no_artifacts(tmp)

  def test_nonmapping_recipes_refuse_before_model_or_output(self):
    """A list cannot be saved as model-building instructions."""
    facts = fixed_facts.synthetic_sidecar(
      names=["p0"], label="nonmapping-recipe", support=None)
    cases = (
      ([], _reader_shape_recipe(), "resolved_train"),
      ({"nepochs": 1}, ["fixture"], "resolved_model"),
      ("nepochs: 1", _reader_shape_recipe(), "resolved_train"),
    )
    with tempfile.TemporaryDirectory(prefix="schema3-recipe-type-") as tmp:
      for index, (train, model, diagnostic) in enumerate(cases):
        with self.subTest(recipe=diagnostic, value_type=type(
            train if diagnostic == "resolved_train" else model).__name__):
          root = Path(tmp) / ("artifact-" + str(index))
          with self.assertRaisesRegex(
              TypeError, diagnostic + r" must be a plain mapping"):
            _save_attempt(
              root,
              facts_yaml=facts,
              resolved_train=train,
              resolved_model=model)
      self._assert_no_artifacts(tmp)

  def test_malformed_and_name_mismatched_facts_precede_model_state(self):
    """Invalid producer records fail without touching the model or disk."""
    mismatched = fixed_facts.synthetic_sidecar(
      names=["other"], label="wrong-name", support=None)
    cases = (
      ("fixed_facts: [\n", r"does not parse as YAML"),
      (mismatched, r"record disagree.*sampled parameters"),
    )
    with tempfile.TemporaryDirectory(prefix="schema3-invalid-facts-") as tmp:
      for index, (facts, diagnostic) in enumerate(cases):
        with self.subTest(diagnostic=diagnostic):
          root = Path(tmp) / ("artifact-" + str(index))
          with self.assertRaisesRegex(ValueError, diagnostic):
            _save_attempt(
              root,
              facts_yaml=facts,
              resolved_train={"nepochs": 1},
              resolved_model=_reader_shape_recipe())
      self._assert_no_artifacts(tmp)

  def test_incomplete_model_recipes_refuse_before_model_or_output(self):
    """The writer cannot publish YAML that its reader immediately rejects."""
    facts = fixed_facts.synthetic_sidecar(
      names=["p0"], label="incomplete-model-recipe", support=None)
    missing_cls = _reader_shape_recipe()
    missing_cls.pop("cls")
    bad_block = _reader_shape_recipe()
    bad_block["kwargs"]["block_opts"].pop("norm")
    null_block = _reader_shape_recipe()
    null_block["kwargs"]["block_opts"] = None
    missing_block_gate = _reader_shape_recipe()
    missing_block_gate["kwargs"]["block_opts"]["act"].pop("n_gates")
    missing_head_gate = _reader_shape_recipe()
    missing_head_gate["kwargs"]["head_act"] = {"type": "H"}
    cases = (
      ({}, ValueError, r"missing reader-required key"),
      (missing_cls, ValueError, r"missing reader-required key.*cls"),
      (bad_block, ValueError, r"block_opts is missing.*norm"),
      (null_block, TypeError, r"block_opts must be a plain mapping"),
      (missing_block_gate, ValueError,
       r"block_opts\.act is missing.*n_gates"),
      (missing_head_gate, ValueError, r"head_act is missing.*n_gates"),
    )
    with tempfile.TemporaryDirectory(prefix="schema3-recipe-shape-") as tmp:
      for index, (model_recipe, error_type, diagnostic) in enumerate(cases):
        with self.subTest(case=index):
          root = Path(tmp) / ("artifact-" + str(index))
          with self.assertRaisesRegex(error_type, diagnostic):
            _save_attempt(
              root,
              facts_yaml=facts,
              resolved_train={"nepochs": 1},
              resolved_model=model_recipe)
      self._assert_no_artifacts(tmp)

  def test_valid_save_is_current_schema_and_rebuilds(self):
    """The supported writer path produces one complete readable file pair."""
    with tempfile.TemporaryDirectory(prefix="schema3-valid-") as tmp:
      root = Path(tmp) / "artifact"
      save_fixture(path_root=root,
                   compile_mode="default",
                   case_label="schema3-production-roundtrip")

      with h5py.File(str(root) + ".h5", "r") as artifact:
        self.assertEqual(
          int(artifact.attrs["schema_version"]),
          fixed_facts.SCHEMA_VERSION)
        self.assertIn(fixed_facts.FIXED_FACTS_GROUP, artifact)
        self.assertIn(fixed_facts.INPUT_DOMAIN_GROUP, artifact)
        self.assertIn(fixed_facts.SIDECAR_DATASET, artifact)

      model, pgeom, geometry, info = results.rebuild_emulator(
        path_root=str(root),
        device=torch.device("cpu"),
        compile_model=False)
      self.assertIsNotNone(model)
      self.assertEqual(pgeom.names, ["p0", "p1"])
      self.assertEqual(geometry.names, ["derived"])
      self.assertEqual(info["composition_mode"], "plain")

  def test_missing_or_invalid_val_facts_precede_device_and_warmstart(self):
    """The second small record is checked before any expensive run setup."""
    cases = (
      ("missing", None, r"\.facts\.yaml"),
      ("malformed", "fixed_facts: [\n", r"does not parse as YAML"),
      ("wrong names",
       fixed_facts.synthetic_sidecar(
         names=["other"], label="wrong-val-names", support=None),
       r"record disagree.*sampled parameters"),
    )
    for label, val_text, diagnostic in cases:
      with self.subTest(case=label), tempfile.TemporaryDirectory(
          prefix="schema3-preflight-") as tmp:
        cfg = _preflight_config(tmp)
        with open(cfg["data"]["train_covmat"], "w") as handle:
          handle.write("# p0\n1.0\n")
        _write_facts(
          cfg["data"]["train_params"], names=["p0"], label="train")
        if val_text is not None:
          _write_facts(
            cfg["data"]["val_params"], names=["p0"], label="val",
            text=val_text)

        with mock.patch.object(
            experiment, "pick_device",
            side_effect=AssertionError("device selection must not run")) \
            as pick, mock.patch.object(
              experiment.warmstart, "load_source",
              side_effect=AssertionError("warm-start must not run")) \
            as warm, mock.patch.object(
              data_staging.np, "load",
              side_effect=AssertionError("NumPy data must not open")) \
            as open_array, mock.patch.object(
              experiment.ResMLP, "__init__",
              side_effect=AssertionError("model construction must not run")) \
            as build_model:
          with self.assertRaisesRegex(ValueError, diagnostic):
            experiment.EmulatorExperiment.from_config(cfg)

        pick.assert_not_called()
        warm.assert_not_called()
        open_array.assert_not_called()
        build_model.assert_not_called()

  def test_scalar_model_error_precedes_missing_facts(self):
    """A scalar correction-head mistake is explained before file checks."""
    with tempfile.TemporaryDirectory(prefix="schema3-model-first-") as tmp:
      cfg = _preflight_config(tmp)
      cfg["train_args"]["model"]["name"] = "rescnn"
      with mock.patch.object(
          experiment, "validated_facts_sidecar",
          side_effect=AssertionError("facts must follow pure model checks")) \
          as check_facts:
        with self.assertRaisesRegex(ValueError, r"correction head"):
          experiment.EmulatorExperiment.from_config(cfg)
      check_facts.assert_not_called()

  def test_scalar_pce_error_precedes_missing_facts(self):
    """An invalid scalar PCE form is explained before file checks."""
    with tempfile.TemporaryDirectory(prefix="schema3-pce-first-") as tmp:
      cfg = _preflight_config(tmp)
      cfg["pce"] = {"form": "ratio"}
      with mock.patch.object(
          experiment, "validated_facts_sidecar",
          side_effect=AssertionError("facts must follow pure PCE checks")) \
          as check_facts:
        with self.assertRaisesRegex(ValueError, r"only on the cosmolike"):
          experiment.EmulatorExperiment.from_config(cfg)
      check_facts.assert_not_called()

  def test_staging_reuses_the_exact_record_approved_by_preflight(self):
    """Changing the sidecar path later cannot swap the retained record."""
    with tempfile.TemporaryDirectory(prefix="schema3-pinned-facts-") as tmp:
      params = os.path.join(tmp, "train.1.txt")
      np.savetxt(
        params,
        np.asarray([[1.0, 0.0, 0.25, 3.0]], dtype=np.float32))
      with open(os.path.join(tmp, "train.paramnames"), "w") as handle:
        handle.write("p0 p0\n")
        handle.write("derived* derived\n")
      approved = _write_facts(
        params, names=["p0"], label="approved-before-device")
      retained = data_staging.validated_facts_sidecar(
        params_path=params, names=["p0"])
      self.assertEqual(retained, approved)

      # Replace the path with a valid but differently named record. The staged
      # call receives the retained text and must neither reread nor adopt it.
      _write_facts(params, names=["other"], label="late-path-replacement")
      staged = data_staging.load_scalar_source(
        params_path=params,
        in_names=["p0"],
        out_names=["derived"],
        n_keep=1,
        gen=torch.Generator().manual_seed(1),
        verbose=False,
        facts_yaml=retained)
      self.assertEqual(staged["facts_yaml"], approved)

  def test_early_refusal_preserves_an_existing_valid_pair(self):
    """A failed replacement cannot alter either member or create a marker."""
    with tempfile.TemporaryDirectory(prefix="schema3-preserve-pair-") as tmp:
      root = Path(tmp) / "artifact"
      save_fixture(path_root=root,
                   compile_mode="default",
                   case_label="schema3-preserve-existing")
      before = ((Path(str(root) + ".emul").read_bytes()),
                (Path(str(root) + ".h5").read_bytes()))

      with self.assertRaisesRegex(ValueError, r"requires.*\.facts\.yaml"):
        _save_attempt(
          root,
          facts_yaml=None,
          resolved_train={"nepochs": 1},
          resolved_model=_reader_shape_recipe())

      after = ((Path(str(root) + ".emul").read_bytes()),
               (Path(str(root) + ".h5").read_bytes()))
      self.assertEqual(after, before)
      self.assertFalse(Path(str(root) + ".pair-pending").exists())


if __name__ == "__main__":
  unittest.main()

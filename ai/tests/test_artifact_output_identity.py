"""CPU checks for names that distinguish scientifically different artifacts.

An emulator filename contains a readable family/product prefix and a digest.
The digest must change when the executed science changes, yet remain unchanged
when only dictionary insertion order or local path spelling changes.  These
tests build very small schema-1 dataset pins; they do not read scientific data,
train a network, or write an artifact.
"""

import ast
import copy
import json
from pathlib import Path
import re
import types
import unittest
from unittest import mock

from emulator.output_identity import (
  build_experiment_output_identity,
  build_output_identity,
  digest_cmb_covariance_inputs,
  require_same_output_identity,
  validate_saved_output_identity,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _driver_run_tag(relative_path):
  """Load only ``run_tag`` from a driver without importing training tools.

  The driver modules normally import YAML, Torch, and the full experiment
  stack.  Filename wiring itself needs none of those dependencies, so this
  helper compiles the real function definition in a tiny isolated namespace.
  """
  source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
  tree = ast.parse(source, filename=relative_path)
  definitions = [
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "run_tag"
  ]
  if len(definitions) != 1:
    raise AssertionError(relative_path + " must define exactly one run_tag")
  namespace = {"build_experiment_output_identity": mock.Mock()}
  executable = ast.Module(body=definitions, type_ignores=[])
  ast.fix_missing_locations(executable)
  exec(compile(executable, relative_path, "exec"), namespace)
  return namespace["run_tag"], namespace


def _digest(character):
  """Return one syntactically valid SHA-256 digest for a tiny fixture."""
  return character * 64


def _source_pin(*, split, probe="cs", generation=None,
                row_order_character=None):
  """Return one path-rich source pin with an authenticated row selection."""
  if split == "train":
    defaults = ("train-generation", "1", "2", "3", "4", "5")
  else:
    defaults = ("validation-generation", "6", "7", "8", "9", "a")
  (default_generation, active_character, manifest_character,
   parameter_character, payload_character, default_row_character) = defaults
  generation = default_generation if generation is None else generation
  row_character = (default_row_character if row_order_character is None
                   else row_order_character)
  return {
    "schema": 1,
    "slot_id": split + "-slot-id",
    "slot": split,
    "generation": generation,
    # These path fields model resolver bookkeeping.  Output identity must use
    # the authenticated facts below, not the spelling of a checkout path.
    "publication_root": "/machine-a/publications/" + split,
    "active_path": "/machine-a/ACTIVE-" + split + ".json",
    "active_sha256": _digest(active_character),
    "manifest_sha256": _digest(manifest_character),
    "identity": {
      "probe": probe,
      "dataset_mode": "published",
      "sampling": "latin-hypercube",
    },
    "members": {
      "parameters.chain": {
        "path": "/machine-a/generations/" + generation + "/params.txt",
        "size": 120,
        "sha256": _digest(parameter_character),
      },
      "payload.vector": {
        "path": "/machine-a/generations/" + generation + "/values.npy",
        "size": 240,
        "sha256": _digest(payload_character),
      },
    },
    "selection": {
      "schema": 1,
      "source_rows": 10,
      "selected_rows": 3,
      "row_order_encoding": "uint64-big-endian-v1",
      "row_order_sha256": _digest(row_character),
      "split_seed": 17,
      "param_cuts": {"omegabh2_hi": 0.024, "omegabh2_lo": 0.019},
    },
  }


def _data(*, probe="cs", block=None, outputs=None):
  """Return a resolved data mapping with both production source pins."""
  data = {
    "train_params": "/machine-a/chains/train.1.txt",
    "val_params": "/machine-a/chains/validation.1.txt",
    "train_dv": "/machine-a/chains/train.npy",
    "val_dv": "/machine-a/chains/validation.npy",
    "train_covmat": "/machine-a/chains/train.covmat",
    "cosmolike_data_dir": "lsst_y1",
    "cosmolike_dataset": "3x2pt.dataset",
    "_dataset_sources": {
      "schema": 1,
      "train": _source_pin(split="train", probe=probe),
      "validation": _source_pin(split="validation", probe=probe),
    },
  }
  if outputs is not None:
    data["outputs"] = list(outputs)
  if block is not None:
    name, record = block
    data[name] = copy.deepcopy(record)
    if name == "cmb":
      data[name].setdefault("_covariance_input_sha256", _digest("b"))
  return data


def _resolved_train():
  """Return a small consumed training recipe with nested mappings."""
  return {
    "nepochs": 4,
    "batch_size": 8,
    "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
    "scheduler": {"factor": 0.5, "patience": 2},
  }


def _resolved_model():
  """Return a small consumed model recipe with an explicit activation."""
  return {
    "cls": "emulator.designs.plain.ResMLP",
    "activation": "H",
    "compile_mode": None,
    "kwargs": {"int_dim_res": 16, "n_blocks": 2, "n_gates": 3},
  }


def _arguments(*, data=None):
  """Return the complete plain-run argument mapping for the builder."""
  return {
    "data": _data(outputs=["sigma8"]) if data is None else data,
    "resolved_train": _resolved_train(),
    "resolved_model": _resolved_model(),
    "resolved_rescale": "none",
    "composition_mode": "plain",
    "transfer_refined": False,
    "resolved_pce": None,
    "resolved_transfer": None,
    "require_published_selection": True,
  }


def _build(**changes):
  """Build one identity after replacing selected top-level arguments."""
  arguments = _arguments()
  arguments.update(changes)
  return build_output_identity(**arguments)


def _reverse_mapping_order(value):
  """Recreate every mapping in reverse insertion order without other edits."""
  if type(value) is dict:
    return {
      key: _reverse_mapping_order(item)
      for key, item in reversed(list(value.items()))
    }
  if type(value) is list:
    return [_reverse_mapping_order(item) for item in value]
  return value


class CanonicalIdentityTests(unittest.TestCase):
  """Check stable encoding and the saved digest boundary."""

  def test_reordered_mappings_produce_the_same_identity(self):
    """Dictionary insertion order cannot rename an unchanged run."""
    arguments = _arguments()
    first = build_output_identity(**arguments)
    reordered = {
      key: _reverse_mapping_order(value)
      for key, value in reversed(list(arguments.items()))
    }

    second = build_output_identity(**reordered)

    self.assertEqual(second, first)

  def test_tag_uses_a_sixteen_byte_lowercase_digest_suffix(self):
    """The readable tag retains 16 digest bytes, written as 32 hex digits."""
    identity = _build()
    suffix = identity["tag"].rsplit("-", 1)[-1]
    self.assertEqual(len(suffix), 32)
    self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{32}", suffix))
    self.assertEqual(suffix, identity["sha256"][:32])

  def test_saved_identity_validates_and_detects_a_changed_subject(self):
    """The saved canonical record and its digest must remain one pair."""
    identity = _build()
    subject = validate_saved_output_identity(
      identity["canonical_json"], identity["sha256"])
    self.assertEqual(subject["schema"], 1)

    changed = json.loads(identity["canonical_json"])
    changed["training_recipe"]["nepochs"] += 1
    changed_json = json.dumps(
      changed, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    with self.assertRaisesRegex(ValueError, "digest does not match"):
      validate_saved_output_identity(changed_json, identity["sha256"])

  def test_identity_comparison_checks_the_complete_public_record(self):
    """A matching digest cannot excuse a changed tag or canonical record."""
    expected = _build()
    observed = dict(expected)
    observed["tag"] = "scalar-other-" + expected["sha256"][:16]
    with self.assertRaisesRegex(ValueError, "disagrees.*tag"):
      require_same_output_identity(expected, observed)

class ScientificProductIdentityTests(unittest.TestCase):
  """Check the product pairs that previously shared readable filenames."""

  def _identities_for_blocks(self, family, records):
    return [
      _build(data=_data(block=(family, record)))
      for record in records
    ]

  def test_cmb_tt_and_ee_have_distinct_identities(self):
    """Two spectra trained from equal-size tables still need different roots."""
    tt, ee = self._identities_for_blocks("cmb", (
      {"spectrum": "tt", "amplitude_law": "none"},
      {"spectrum": "ee", "amplitude_law": "none"},
    ))
    self.assertEqual(tt["family"], "cmb")
    self.assertEqual(ee["family"], "cmb")
    self.assertNotEqual(tt["tag"], ee["tag"])

  def test_cmb_covariance_contents_change_the_identity_but_path_does_not(self):
    """Noise or fiducial arrays matter; the local NPZ filename does not."""
    first_record = {
      "spectrum": "tt",
      "amplitude_law": "none",
      "covariance": "/machine-a/covariance.npz",
      "_covariance_input_sha256": _digest("b"),
    }
    moved_record = dict(first_record)
    moved_record["covariance"] = "/machine-b/moved-covariance.npz"
    changed_record = dict(first_record)
    changed_record["_covariance_input_sha256"] = _digest("c")

    first = _build(data=_data(block=("cmb", first_record)))
    moved = _build(data=_data(block=("cmb", moved_record)))
    changed = _build(data=_data(block=("cmb", changed_record)))

    self.assertEqual(first, moved)
    self.assertNotEqual(first["sha256"], changed["sha256"])

  def test_cmb_covariance_digest_uses_the_executed_arrays(self):
    """Changing an ell, whitening scale, or fiducial value changes the pin."""
    ell = [2, 3, 4]
    sigma = [1.0, 2.0, 3.0]
    fiducial = [4.0, 5.0, 6.0]
    baseline = digest_cmb_covariance_inputs(ell, sigma, fiducial)
    cases = (
      ([2, 3, 5], sigma, fiducial),
      (ell, [1.0, 2.0, 4.0], fiducial),
      (ell, sigma, [4.0, 5.0, 7.0]),
    )
    for changed in cases:
      with self.subTest(changed=changed):
        self.assertNotEqual(
          baseline, digest_cmb_covariance_inputs(*changed))

  def test_background_hubble_and_distance_have_distinct_identities(self):
    """Hubble and transverse-distance tables cannot overwrite each other."""
    hubble, distance = self._identities_for_blocks("grid", (
      {"quantity": "Hubble", "units": "km/s/Mpc", "law": "none"},
      {"quantity": "D_M", "units": "Mpc", "law": "none"},
    ))
    self.assertEqual(hubble["family"], "baosn")
    self.assertEqual(distance["family"], "baosn")
    self.assertNotEqual(hubble["tag"], distance["tag"])

  def test_matter_power_pklin_and_boost_have_distinct_identities(self):
    """A linear spectrum and a nonlinear boost receive different roots."""
    pklin, boost = self._identities_for_blocks("grid2d", (
      {"quantity": "pklin", "units": "Mpc^3", "law": "none", "k_stride": 1},
      {"quantity": "boost", "units": "dimensionless", "law": "none",
       "k_stride": 1},
    ))
    self.assertEqual(pklin["family"], "mps")
    self.assertEqual(boost["family"], "mps")
    self.assertNotEqual(pklin["tag"], boost["tag"])

  def test_cosmolike_cs_ggl_and_gc_have_three_distinct_identities(self):
    """The three CosmoLike probes cannot reuse one generic data-vector root."""
    identities = [_build(data=_data(probe=probe))
                  for probe in ("cs", "ggl", "gc")]
    self.assertEqual(
      [identity["product"] for identity in identities],
      ["cs", "ggl", "gc"])
    self.assertEqual(len({identity["tag"] for identity in identities}), 3)

  def test_scalar_output_order_is_part_of_the_identity(self):
    """Scalar columns with reversed meanings cannot share one artifact."""
    first = _build(data=_data(outputs=["sigma8", "H0"]))
    reversed_outputs = _build(data=_data(outputs=["H0", "sigma8"]))
    self.assertNotEqual(first["sha256"], reversed_outputs["sha256"])
    self.assertNotEqual(first["tag"], reversed_outputs["tag"])

  def test_scalar_delimiters_cannot_create_an_ambiguous_concatenation(self):
    """Equal readable products remain distinct through their structured list."""
    left = _build(data=_data(outputs=["a-b", "c"]))
    right = _build(data=_data(outputs=["a", "b-c"]))

    self.assertEqual(left["product"], "a-b-c")
    self.assertEqual(right["product"], "a-b-c")
    self.assertNotEqual(left["sha256"], right["sha256"])
    self.assertNotEqual(left["tag"], right["tag"])


class RecipeAndDatasetIdentityTests(unittest.TestCase):
  """Check consumed model, training, generation, and selection facts."""

  def test_activation_and_model_kwargs_change_the_identity(self):
    """A new nonlinearity or hidden width describes a different model."""
    baseline_model = _resolved_model()
    baseline = _build(resolved_model=baseline_model)

    activation_model = copy.deepcopy(baseline_model)
    activation_model["activation"] = "gated_power"
    width_model = copy.deepcopy(baseline_model)
    width_model["kwargs"]["int_dim_res"] = 32

    for label, model in (
        ("activation", activation_model),
        ("model kwargs", width_model)):
      with self.subTest(changed=label):
        self.assertNotEqual(
          baseline["sha256"], _build(resolved_model=model)["sha256"])

  def test_training_generation_and_row_order_change_the_identity(self):
    """A republished dataset or a new selected-row order receives a new root."""
    baseline_data = _data(outputs=["sigma8"])
    baseline = _build(data=baseline_data)

    cases = []
    for split in ("train", "validation"):
      changed_generation = copy.deepcopy(baseline_data)
      changed_generation["_dataset_sources"][split]["generation"] += "-new"
      cases.append((split + " generation", changed_generation))

      changed_order = copy.deepcopy(baseline_data)
      changed_order["_dataset_sources"][split]["selection"][
        "row_order_sha256"] = _digest("b" if split == "train" else "c")
      cases.append((split + " row order", changed_order))

    for label, data in cases:
      with self.subTest(changed=label):
        self.assertNotEqual(baseline["sha256"], _build(data=data)["sha256"])

  def test_plain_and_npce_compositions_change_the_identity(self):
    """Adding a fitted polynomial base cannot retain the plain-run name."""
    plain = _build()
    npce = _build(
      composition_mode="npce",
      resolved_pce={"form": "residual", "degree": 2, "alpha": 1.0e-6})
    self.assertNotEqual(plain["sha256"], npce["sha256"])

  def test_finetune_identity_binds_both_source_pair_values(self):
    """Fine-tuning distinguishes the source artifact and checkpoint bytes."""
    recipe = _resolved_train()
    recipe["finetune"] = {
      "from": "/machine-a/models/source",
      "source_artifact_id": "1" * 32,
      "source_checkpoint_sha256": _digest("2"),
      "compile_mode": None,
      "extra_names": "w0 wa",
    }
    baseline = _build(resolved_train=recipe)

    cases = []
    changed_artifact = copy.deepcopy(recipe)
    changed_artifact["finetune"]["source_artifact_id"] = "3" * 32
    cases.append(("artifact", changed_artifact))
    changed_checkpoint = copy.deepcopy(recipe)
    changed_checkpoint["finetune"]["source_checkpoint_sha256"] = _digest("4")
    cases.append(("checkpoint", changed_checkpoint))

    for label, changed in cases:
      with self.subTest(changed=label):
        self.assertNotEqual(
          baseline["sha256"], _build(resolved_train=changed)["sha256"])

  def test_transfer_identity_binds_source_form_and_refinement(self):
    """Transfer source, combine rule, and base refinement all affect meaning."""
    transfer = {
      "from": "/machine-a/models/base",
      "source_artifact_id": "5" * 32,
      "source_checkpoint_sha256": _digest("6"),
      "form": "gain",
      "space": "physical",
      "extra_names": "w0 wa",
    }
    recipe = _resolved_train()
    recipe["transfer"] = copy.deepcopy(transfer)
    baseline = _build(
      resolved_train=recipe,
      composition_mode="transfer",
      resolved_transfer=transfer)

    cases = []
    changed_source = copy.deepcopy(transfer)
    changed_source["source_checkpoint_sha256"] = _digest("7")
    cases.append(("source", changed_source, False))
    changed_form = copy.deepcopy(transfer)
    changed_form["form"] = "delta"
    cases.append(("form", changed_form, False))
    cases.append(("refinement", transfer, True))

    for label, changed_transfer, refined in cases:
      with self.subTest(changed=label):
        changed_recipe = copy.deepcopy(recipe)
        changed_recipe["transfer"] = copy.deepcopy(changed_transfer)
        changed = _build(
          resolved_train=changed_recipe,
          composition_mode="transfer",
          transfer_refined=refined,
          resolved_transfer=changed_transfer)
        self.assertNotEqual(baseline["sha256"], changed["sha256"])


class PathIndependenceTests(unittest.TestCase):
  """Check that moving identical authenticated inputs does not rename a run."""

  def test_dataset_and_member_paths_do_not_change_the_identity(self):
    """Only member bytes and staged rows matter, not local source locations."""
    first_data = _data(outputs=["sigma8"])
    moved_data = copy.deepcopy(first_data)
    for key in ("train_params", "val_params", "train_dv", "val_dv",
                "train_covmat"):
      moved_data[key] = "/machine-b/relocated/" + key
    for split in ("train", "validation"):
      pin = moved_data["_dataset_sources"][split]
      pin["publication_root"] = "/machine-b/publications/" + split
      pin["active_path"] = "/machine-b/ACTIVE-" + split + ".json"
      for role, member in pin["members"].items():
        member["path"] = "/machine-b/relocated/" + split + "/" + role

    self.assertEqual(_build(data=first_data), _build(data=moved_data))

  def test_finetune_source_path_does_not_replace_pair_authentication(self):
    """Moving one authenticated fine-tune source leaves its identity stable."""
    first = _resolved_train()
    first["finetune"] = {
      "from": "/machine-a/model/source",
      "source_artifact_id": "1" * 32,
      "source_checkpoint_sha256": _digest("2"),
    }
    moved = copy.deepcopy(first)
    moved["finetune"]["from"] = "/machine-b/model/source"
    self.assertEqual(
      _build(resolved_train=first),
      _build(resolved_train=moved))

  def test_transfer_source_path_does_not_replace_pair_authentication(self):
    """Moving one authenticated transfer base leaves its identity stable."""
    first_transfer = {
      "from": "/machine-a/model/base",
      "source_artifact_id": "3" * 32,
      "source_checkpoint_sha256": _digest("4"),
      "form": "gain",
      "space": "physical",
    }
    moved_transfer = copy.deepcopy(first_transfer)
    moved_transfer["from"] = "/machine-b/model/base"
    first_recipe = _resolved_train()
    first_recipe["transfer"] = copy.deepcopy(first_transfer)
    moved_recipe = _resolved_train()
    moved_recipe["transfer"] = copy.deepcopy(moved_transfer)

    first = _build(
      resolved_train=first_recipe,
      composition_mode="transfer",
      resolved_transfer=first_transfer)
    moved = _build(
      resolved_train=moved_recipe,
      composition_mode="transfer",
      resolved_transfer=moved_transfer)
    self.assertEqual(first, moved)


class ProductionPinRefusalTests(unittest.TestCase):
  """Check that a production identity cannot silently use fixture fallback."""

  def test_production_refuses_missing_dataset_sources(self):
    """A completed production run needs two resolver-published source pins."""
    data = _data(outputs=["sigma8"])
    del data["_dataset_sources"]
    with self.assertRaisesRegex(ValueError, "production output identity"):
      _build(data=data)

  def test_production_refuses_a_missing_split_pin(self):
    """Neither training nor validation provenance may be omitted."""
    for missing in ("train", "validation"):
      with self.subTest(missing=missing):
        data = _data(outputs=["sigma8"])
        del data["_dataset_sources"][missing]
        with self.assertRaisesRegex(ValueError, missing):
          _build(data=data)

  def test_production_refuses_a_split_without_staged_selection(self):
    """A generation pin alone does not say which rows were trained or tested."""
    for split in ("train", "validation"):
      with self.subTest(split=split):
        data = _data(outputs=["sigma8"])
        del data["_dataset_sources"][split]["selection"]
        with self.assertRaisesRegex(ValueError, "staged row selection"):
          _build(data=data)

  def test_production_refuses_missing_authenticated_pin_facts(self):
    """Active record, manifest, members, and row order are all mandatory."""
    cases = (
      ("active_sha256", "active record"),
      ("manifest_sha256", "dataset manifest"),
      ("members", "authenticated members"),
    )
    for key, message in cases:
      with self.subTest(missing=key):
        data = _data(outputs=["sigma8"])
        del data["_dataset_sources"]["train"][key]
        with self.assertRaisesRegex(ValueError, message):
          _build(data=data)

    data = _data(outputs=["sigma8"])
    del data["_dataset_sources"]["train"]["selection"][
      "row_order_sha256"]
    with self.assertRaisesRegex(ValueError, "staged row order"):
      _build(data=data)

  def test_direct_library_fixture_may_explicitly_omit_published_sources(self):
    """The documented low-level fixture allowance remains separate and visible."""
    data = _data(outputs=["sigma8"])
    del data["_dataset_sources"]
    identity = _build(data=data, require_published_selection=False)
    subject = json.loads(identity["canonical_json"])
    self.assertEqual(
      subject["staged_selection"],
      {"fixture_without_published_dataset": True})

  def test_finetune_and_transfer_sources_refuse_missing_pair_bindings(self):
    """A reusable source path is never accepted as its scientific identity."""
    for label in ("finetune", "transfer"):
      with self.subTest(source=label):
        source = {
          "from": "/machine-a/model/source",
          "source_artifact_id": "1" * 32,
          "source_checkpoint_sha256": _digest("2"),
          "form": "gain",
          "space": "physical",
        }
        del source["source_checkpoint_sha256"]
        recipe = _resolved_train()
        recipe[label] = copy.deepcopy(source)
        changes = {"resolved_train": recipe}
        if label == "transfer":
          changes.update(
            composition_mode="transfer", resolved_transfer=source)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
          _build(**changes)


class ExperimentAndDriverWiringTests(unittest.TestCase):
  """Check the production wrapper and both public filename-tag helpers."""

  def _experiment(self):
    return types.SimpleNamespace(
      data=_data(outputs=["sigma8"]),
      resolved_train=_resolved_train(),
      resolved_model=_resolved_model(),
      rescale="none",
      pce_opts=None,
      _transfer_base=None,
      _transfer_pretrained_base=None,
    )

  def test_experiment_wrapper_requires_final_pins_and_selects_plain_npce(self):
    """The production wrapper both requires staging and records NPCE use."""
    experiment = self._experiment()
    plain = build_experiment_output_identity(experiment)
    experiment.pce_opts = {"form": "residual", "degree": 2}
    npce = build_experiment_output_identity(experiment)
    self.assertNotEqual(plain["sha256"], npce["sha256"])

    del experiment.data["_dataset_sources"]["validation"]["selection"]
    with self.assertRaisesRegex(ValueError, "staged row selection"):
      build_experiment_output_identity(experiment)

  def test_experiment_wrapper_refuses_npce_and_transfer_together(self):
    """One run cannot claim two exclusive composition modes."""
    experiment = self._experiment()
    experiment.pce_opts = {"form": "residual"}
    experiment._transfer_base = object()
    with self.assertRaisesRegex(ValueError, "both NPCE and transfer"):
      build_experiment_output_identity(experiment)

  def test_experiment_wrapper_distinguishes_every_rescale_mode(self):
    """The three executed CosmoLike loss modes need three output roots."""
    experiment = self._experiment()
    identities = []
    for mode in ("none", "rescaled", "residual"):
      experiment.rescale = mode
      identities.append(build_experiment_output_identity(experiment))

    self.assertEqual(len({item["sha256"] for item in identities}), 3)
    for identity, mode in zip(identities, ("none", "rescaled", "residual")):
      subject = json.loads(identity["canonical_json"])
      self.assertEqual(subject["loss_recipe"], {"rescale": mode})

  def test_shared_and_scalar_run_tags_use_the_supplied_identity(self):
    """A driver must not rebuild a second identity after naming its outputs."""
    for relative_path in (
        "cosmic_shear_train_emulator.py",
        "scalar_train_emulator.py"):
      with self.subTest(driver=relative_path):
        run_tag, namespace = _driver_run_tag(relative_path)
        builder = namespace["build_experiment_output_identity"]
        supplied = {"tag": relative_path + "-exact-tag"}
        actual = run_tag(object(), object(), output_identity=supplied)
        self.assertEqual(actual, supplied["tag"])
        builder.assert_not_called()

  def test_shared_and_scalar_run_tags_build_once_when_not_supplied(self):
    """The convenience path passes the exact completed experiment to builder."""
    experiment = object()
    for relative_path in (
        "cosmic_shear_train_emulator.py",
        "scalar_train_emulator.py"):
      with self.subTest(driver=relative_path):
        run_tag, namespace = _driver_run_tag(relative_path)
        builder = namespace["build_experiment_output_identity"]
        built = {"tag": relative_path + "-built-tag"}
        builder.return_value = built
        actual = run_tag(object(), experiment)
        self.assertEqual(actual, built["tag"])
        builder.assert_called_once_with(experiment)


if __name__ == "__main__":
  unittest.main()

"""CPU tests that keep failed generator rows out of staged training data."""

import os
import tempfile
import unittest

import h5py
import numpy as np
import torch
import yaml

from emulator import data_staging, fixed_facts, results
from emulator.experiment import EmulatorExperiment
from emulator.geometries.parameter import ParamGeometry
from emulator.geometries.scalar import ScalarGeometry


class FailedRowStagingTests(unittest.TestCase):
  """Require the published failure mask to control row selection."""

  def _write_source(self, directory):
    """Write one small aligned parameter, payload, and metadata source.

    Arguments:
      directory = temporary folder that owns every returned file.

    Returns:
      A mapping with the parameter, payload, and failure-mask paths, plus the
      modeled parameter and payload arrays used to check row alignment.
    """
    params_path = os.path.join(directory, "source.1.txt")
    payload_path = os.path.join(directory, "source.npy")
    failure_path = os.path.join(directory, "source_fail.txt")
    inputs = np.asarray([
      [0.020, 60.0],
      [0.021, 61.0],
      [0.022, 62.0],
      [0.023, 63.0],
    ], dtype=np.float32)
    payload = np.asarray([
      [100.0, 101.0],
      [200.0, 201.0],
      [300.0, 301.0],
      [400.0, 401.0],
    ], dtype=np.float32)
    bookkeeping = np.zeros((inputs.shape[0], 2), dtype=np.float32)
    diagnostic = np.zeros((inputs.shape[0], 1), dtype=np.float32)
    table = np.concatenate((bookkeeping, inputs, diagnostic), axis=1)
    np.savetxt(params_path, table)
    np.save(payload_path, payload)

    paramnames_path = os.path.join(directory, "source.paramnames")
    with open(paramnames_path, "w", encoding="utf-8") as paramnames:
      paramnames.write("omegab omegab\n")
      paramnames.write("H0 H0\n")
      paramnames.write("chi2* chi2\n")
    facts_path = os.path.join(directory, "source.facts.yaml")
    facts_text = fixed_facts.synthetic_sidecar(
      names=["omegab", "H0"],
      label="failed-row-staging",
      family="cosmolike",
      support=None)
    with open(facts_path, "w", encoding="utf-8") as facts_file:
      facts_file.write(facts_text)
    with open(failure_path, "w", encoding="ascii") as failure_file:
      failure_file.write("0\n1\n0\n1\n")

    return {
      "failure": failure_path,
      "inputs": inputs,
      "params": params_path,
      "payload": payload,
      "payload_path": payload_path,
    }

  def test_failed_rows_are_removed_before_seeded_selection(self):
    """Only successful rows may enter the shuffled physical pool."""
    with tempfile.TemporaryDirectory() as directory:
      source = self._write_source(directory)
      seed = 19
      expected_order = data_staging.torch.randperm(
        4,
        generator=data_staging.torch.Generator().manual_seed(seed)).numpy()
      expected_rows = expected_order[np.isin(expected_order, [0, 2])]

      staged = data_staging.load_source(
        dv_path=source["payload_path"],
        params_path=source["params"],
        names=["omegab", "H0"],
        omegabh2_hi=None,
        n_keep=2,
        gen=data_staging.torch.Generator().manual_seed(seed),
        ram_frac=1.0,
        verbose=False,
        failure_mask_path=source["failure"])

      np.testing.assert_array_equal(staged["selected_rows"], expected_rows)
      np.testing.assert_array_equal(
        staged["C"][staged["idx"]],
        source["inputs"][expected_rows])
      np.testing.assert_array_equal(
        staged["dv"][staged["idx"]],
        source["payload"][expected_rows])
      self.assertEqual(staged["source_n_rows"], 4)

  def test_failed_rows_reduce_the_available_pool(self):
    """A requested count cannot be filled with rows marked as failed."""
    with tempfile.TemporaryDirectory() as directory:
      source = self._write_source(directory)

      with self.assertRaisesRegex(ValueError, "pool too small"):
        data_staging.load_source(
          dv_path=source["payload_path"],
          params_path=source["params"],
          names=["omegab", "H0"],
          omegabh2_hi=None,
          n_keep=3,
          gen=data_staging.torch.Generator().manual_seed(3),
          verbose=False,
          failure_mask_path=source["failure"])

      experiment = EmulatorExperiment.__new__(EmulatorExperiment)
      experiment.data = {
        "train_failure_mask": source["failure"],
        "train_params": source["params"],
      }
      experiment.names = ["omegab", "H0"]
      experiment.outputs = []
      experiment._scalar = False
      experiment._cmb = True
      experiment._grid = False
      experiment._grid2d = False
      self.assertEqual(experiment.pool_size(), 2)

  def test_failure_mask_refuses_bad_tokens_and_row_counts(self):
    """The mask cannot silently coerce text or omit a parameter row."""
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, "requires an authenticated"):
        data_staging._load_failure_mask(
          path=None,
          expected_rows=2)

      invalid = os.path.join(directory, "invalid.txt")
      with open(invalid, "w", encoding="ascii") as failure_file:
        failure_file.write("0\ntrue\n")
      with self.assertRaisesRegex(ValueError, "literal '0' or '1'"):
        data_staging._load_failure_mask(
          path=invalid,
          expected_rows=2)

      short = os.path.join(directory, "short.txt")
      with open(short, "w", encoding="ascii") as failure_file:
        failure_file.write("0\n")
      with self.assertRaisesRegex(ValueError, "has 1 rows"):
        data_staging._load_failure_mask(
          path=short,
          expected_rows=2)

  def test_scalar_selection_records_original_disk_rows(self):
    """Chain-only staging records the same ordered row identity."""
    with tempfile.TemporaryDirectory() as directory:
      source = self._write_source(directory)
      seed = 23
      expected = data_staging.torch.randperm(
        4,
        generator=data_staging.torch.Generator().manual_seed(seed)).numpy()[:2]

      staged = data_staging.load_scalar_source(
        params_path=source["params"],
        in_names=["omegab", "H0"],
        out_names=["chi2"],
        n_keep=2,
        gen=data_staging.torch.Generator().manual_seed(seed),
        verbose=False)

      self.assertEqual(staged["source_n_rows"], 4)
      np.testing.assert_array_equal(staged["selected_rows"], expected)

  def test_saved_source_pin_binds_exact_staged_row_order(self):
    """The saved config fingerprints order, cuts, seed, and source size."""
    experiment = EmulatorExperiment.__new__(EmulatorExperiment)
    experiment.data = {
      "_dataset_sources": {
        "schema": 1,
        "train": {"generation": "generation-a"},
        "validation": {"generation": "generation-b"},
      },
      "param_cuts": {"omegabh2_hi": 0.024},
      "split_seed": 19,
    }
    source = {
      "source_n_rows": 6,
      "selected_rows": np.asarray([4, 1, 5], dtype=np.int64),
      "dump_rows": np.asarray([1, 4, 5], dtype=np.int64),
      "idx": np.asarray([1, 0, 2], dtype=np.int64),
      "C": np.zeros((3, 1), dtype=np.float32),
      "dv": np.zeros((3, 1), dtype=np.float32),
    }
    experiment._record_staged_selection("train", source)
    first = dict(
      experiment.data["_dataset_sources"]["train"]["selection"])

    self.assertEqual(first["source_rows"], 6)
    self.assertEqual(first["selected_rows"], 3)
    self.assertEqual(first["split_seed"], 19)
    self.assertEqual(first["param_cuts"], {"omegabh2_hi": 0.024})
    self.assertEqual(len(first["row_order_sha256"]), 64)

    reversed_source = {
      "source_n_rows": 6,
      "selected_rows": np.asarray([5, 1, 4], dtype=np.int64),
      "dump_rows": np.asarray([1, 4, 5], dtype=np.int64),
      "idx": np.asarray([2, 0, 1], dtype=np.int64),
      "C": np.zeros((3, 1), dtype=np.float32),
      "dv": np.zeros((3, 1), dtype=np.float32),
    }
    experiment._record_staged_selection("train", reversed_source)
    second = experiment.data["_dataset_sources"]["train"]["selection"]
    self.assertNotEqual(
      first["row_order_sha256"], second["row_order_sha256"])
    self.assertNotIn(
      "selection",
      experiment.data["_dataset_sources"]["validation"])

  def test_saved_source_pin_refuses_a_partial_staged_identity(self):
    """The recorded row list must cover every loader index."""
    experiment = EmulatorExperiment.__new__(EmulatorExperiment)
    experiment.data = {
      "_dataset_sources": {
        "schema": 1,
        "train": {"generation": "generation-a"},
        "validation": {"generation": "generation-b"},
      },
      "param_cuts": {},
      "split_seed": 19,
    }
    source = {
      "source_n_rows": 6,
      "selected_rows": np.asarray([4, 1, 5], dtype=np.int64),
      "dump_rows": np.asarray([1, 4, 5], dtype=np.int64),
      "idx": np.asarray([0, 1], dtype=np.int64),
      "C": np.zeros((3, 1), dtype=np.float32),
      "dv": np.zeros((3, 1), dtype=np.float32),
    }

    with self.assertRaisesRegex(ValueError, "staged idx supplies 2"):
      experiment._record_staged_selection("train", source)

  def test_saved_source_pin_refuses_an_equal_count_wrong_row_order(self):
    """A permutation mismatch cannot receive a truthful saved fingerprint."""
    experiment = EmulatorExperiment.__new__(EmulatorExperiment)
    experiment.data = {
      "_dataset_sources": {
        "schema": 1,
        "train": {"generation": "generation-a"},
        "validation": {"generation": "generation-b"},
      },
      "param_cuts": {},
      "split_seed": 19,
    }
    source = {
      "source_n_rows": 6,
      "selected_rows": np.asarray([4, 1, 5], dtype=np.int64),
      "dump_rows": np.asarray([1, 4, 5], dtype=np.int64),
      "idx": np.asarray([2, 1, 0], dtype=np.int64),
      "C": np.zeros((3, 1), dtype=np.float32),
      "dv": np.zeros((3, 1), dtype=np.float32),
    }

    with self.assertRaisesRegex(ValueError, "does not match the disk-row"):
      experiment._record_staged_selection("train", source)

  def test_saved_emulator_keeps_the_staged_source_identity(self):
    """Both saved config records retain the exact selection fingerprint."""
    experiment = EmulatorExperiment.__new__(EmulatorExperiment)
    experiment.data = {
      "_dataset_sources": {
        "schema": 1,
        "train": {"generation": "generation-a"},
        "validation": {"generation": "generation-b"},
      },
      "param_cuts": {},
      "split_seed": 19,
    }
    source = {
      "source_n_rows": 6,
      "selected_rows": np.asarray([4, 1, 5], dtype=np.int64),
      "dump_rows": np.asarray([1, 4, 5], dtype=np.int64),
      "idx": np.asarray([1, 0, 2], dtype=np.int64),
      "C": np.zeros((3, 1), dtype=np.float32),
      "dv": np.zeros((3, 1), dtype=np.float32),
    }
    experiment._record_staged_selection("train", source)
    expected = dict(
      experiment.data["_dataset_sources"]["train"]["selection"])

    device = torch.device("cpu")
    model = torch.nn.Linear(1, 1)
    parameter_geometry = ParamGeometry(
      device=device,
      names=["p0"],
      center=[0.0],
      evecs=[[1.0]],
      sqrt_ev=[1.0])
    output_geometry = ScalarGeometry(
      device=device,
      names=["derived"],
      center=[0.0],
      scale=[1.0])
    histories = {
      "train_losses": [0.1],
      "val_medians": [0.1],
      "val_means": [0.1],
      "val_fracs": [torch.tensor([0.5])],
      "thresholds": torch.tensor([1.0]),
    }
    recipe = {
      "cls": "torch.nn.Linear",
      "ia": None,
      "input_dim": 1,
      "output_dim": 1,
      "compile_mode": None,
      "needs_geom": False,
      "kwargs": {},
    }
    config = {"data": experiment.data, "train_args": {"nepochs": 1}}
    facts = fixed_facts.synthetic_sidecar(
      names=["p0"], label="staged-source-artifact", support=None)

    with tempfile.TemporaryDirectory() as directory:
      root = os.path.join(directory, "artifact")
      _, h5_path = results.save_emulator(
        path_root=root,
        model=model,
        param_geometry=parameter_geometry,
        geometry=output_geometry,
        config=config,
        histories=histories,
        train_args=config["train_args"],
        resolved_train={"nepochs": 1},
        resolved_model=recipe,
        facts_yaml=facts,
        composition_mode="plain",
        transfer_refined=False,
        resolved_pce=None,
        resolved_transfer=None)
      with h5py.File(h5_path, "r") as artifact:
        for dataset in ("config_yaml", "config_resolved_yaml"):
          saved = yaml.safe_load(artifact[dataset][()].decode("utf-8"))
          observed = saved["data"]["_dataset_sources"]["train"][
            "selection"]
          self.assertEqual(observed, expected)


if __name__ == "__main__":
  unittest.main()

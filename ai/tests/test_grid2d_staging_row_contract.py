"""CPU checks for Grid2D source row identity and exact sibling sizes."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from emulator.experiment import EmulatorExperiment


class Grid2DStagingRowContractTests(unittest.TestCase):
  """Keep one seeded selection independent of RAM and reject wrong sizes."""

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.n_source = 7
    self.n_width = 6
    self.seeded_disk_rows = np.array([5, 1, 6, 2], dtype="int64")
    self.dump_rows = np.sort(self.seeded_disk_rows)
    self.local_seeded_rows = np.searchsorted(
      self.dump_rows,
      self.seeded_disk_rows,
    )
    self.raw = (
      np.arange(self.n_source * self.n_width, dtype="float32") + 20.0
    ).reshape(self.n_source, self.n_width)
    self.base = (
      np.arange(self.n_source * self.n_width, dtype="float32") / 10.0
      + 2.0
    ).reshape(self.n_source, self.n_width)
    self.parameters = np.arange(
      self.n_source * 3,
      dtype="float32",
    ).reshape(self.n_source, 3)
    np.save(self.root / "z.npy", np.array([0.0, 1.0]))
    np.save(self.root / "k.npy", np.array([0.1, 0.2, 0.3]))
    np.save(self.root / "base.npy", self.base)

  def tearDown(self):
    self.temporary.cleanup()

  def _experiment(self, law="syren_linear"):
    """Return the smallest production experiment needed by the transform."""
    experiment = EmulatorExperiment.__new__(EmulatorExperiment)
    experiment.grid2d = {
      "quantity": "pklin",
      "units": "Mpc3",
      "law": law,
      "z_file": str(self.root / "z.npy"),
      "k_file": str(self.root / "k.npy"),
      "k_stride": 1,
    }
    experiment.data = {"ram_frac": 0.9}
    experiment.log = lambda *args, **kwargs: None
    experiment._grid2d_z = None
    experiment._grid2d_k = None
    experiment._grid2d_center = None
    experiment._grid2d_scale = None
    return experiment

  def _resident_source(self):
    """Return sorted compact storage with a nontrivial local row mapping."""
    return {
      "C": self.parameters[self.dump_rows].copy(),
      "dv": self.raw[self.dump_rows].copy(),
      "idx": self.local_seeded_rows.copy(),
      "dump_rows": self.dump_rows.copy(),
      "source_n_rows": self.n_source,
    }

  def _disk_source(self, raw=None, source_n_rows=None):
    """Return complete disk-backed storage with global seeded coordinates."""
    if raw is None:
      raw = self.raw
    raw_path = self.root / ("raw_" + str(int(raw.shape[0])) + ".npy")
    np.save(raw_path, raw)
    return {
      "C": self.parameters.copy(),
      "dv": np.load(raw_path, mmap_mode="r", allow_pickle=False),
      "idx": self.seeded_disk_rows.copy(),
      "dump_rows": self.dump_rows.copy(),
      "source_n_rows": (
        self.n_source if source_n_rows is None else source_n_rows
      ),
    }

  def _expected_seeded_law_rows(self):
    """Compute the selected rows without calling the production transform."""
    return np.log(
      self.raw[self.seeded_disk_rows].astype("float64")
      / self.base[self.seeded_disk_rows].astype("float64")
    ).astype("float32")

  def test_resident_and_memmap_preserve_the_same_seeded_order(self):
    expected_targets = self._expected_seeded_law_rows()
    expected_parameters = self.parameters[self.seeded_disk_rows]

    for representation in ("resident", "memmap"):
      with self.subTest(representation=representation):
        source = (
          self._resident_source()
          if representation == "resident"
          else self._disk_source()
        )
        experiment = self._experiment()
        experiment._grid2d_law_rows(
          src=source,
          base_path=str(self.root / "base.npy"),
          with_means=True,
        )

        self.assertEqual(source["idx"].dtype.kind, "i")
        self.assertTrue(
          np.array_equal(source["idx"], self.local_seeded_rows)
        )
        self.assertFalse(
          np.array_equal(
            source["idx"],
            np.arange(self.seeded_disk_rows.size),
          ),
          "plain arange would silently replace the seeded order",
        )
        self.assertTrue(
          np.array_equal(
            source["dump_rows"][source["idx"]],
            self.seeded_disk_rows,
          )
        )
        self.assertTrue(
          np.array_equal(source["C"][source["idx"]], expected_parameters)
        )
        self.assertTrue(
          np.array_equal(source["dv"][source["idx"]], expected_targets)
        )

  def test_base_requires_exact_original_row_count_before_allocation(self):
    for delta in (-1, 0, 1):
      with self.subTest(base_rows=self.n_source + delta):
        base_path = self.root / ("base_" + str(delta) + ".npy")
        np.save(base_path, self.base[:self.n_source + delta])
        if delta == 1:
          extra = np.full((1, self.n_width), 3.0, dtype="float32")
          np.save(base_path, np.concatenate([self.base, extra], axis=0))

        source = self._disk_source()
        experiment = self._experiment()
        if delta == 0:
          experiment._grid2d_law_rows(
            src=source,
            base_path=str(base_path),
            with_means=False,
          )
          self.assertEqual(source["dv"].shape[0], self.dump_rows.size)
        else:
          with mock.patch(
            "emulator.experiment.np.empty",
            side_effect=AssertionError("transformed target allocated"),
          ):
            with self.assertRaisesRegex(ValueError, "exactly the raw"):
              experiment._grid2d_law_rows(
                src=source,
                base_path=str(base_path),
                with_means=False,
              )

  def test_raw_requires_exact_original_row_count_before_allocation(self):
    extra = np.full((1, self.n_width), 30.0, dtype="float32")
    raw_cases = {
      self.n_source - 1: self.raw[:-1],
      self.n_source: self.raw,
      self.n_source + 1: np.concatenate([self.raw, extra], axis=0),
    }
    for n_rows, raw in raw_cases.items():
      with self.subTest(raw_rows=n_rows):
        source = self._disk_source(raw=raw)
        experiment = self._experiment(law="none")
        if n_rows == self.n_source:
          experiment._grid2d_law_rows(
            src=source,
            base_path=None,
            with_means=False,
          )
          self.assertEqual(source["dv"].shape[0], self.dump_rows.size)
        else:
          with mock.patch(
            "emulator.experiment.np.empty",
            side_effect=AssertionError("transformed target allocated"),
          ):
            with self.assertRaisesRegex(ValueError, "original source"):
              experiment._grid2d_law_rows(
                src=source,
                base_path=None,
                with_means=False,
              )

  def test_source_row_count_must_be_a_native_positive_integer(self):
    for value in (True, np.int64(self.n_source), 0, -1, "7"):
      with self.subTest(source_n_rows=value):
        source = self._disk_source(source_n_rows=value)
        with self.assertRaisesRegex(ValueError, "native integer"):
          self._experiment(law="none")._grid2d_law_rows(
            src=source,
            base_path=None,
            with_means=False,
          )


if __name__ == "__main__":
  unittest.main()

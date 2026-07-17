"""Check that CMB covariance publication is complete and non-overwriting.

The tests use tiny archives and do not run CAMB. They interrupt the private
write and final-name step, where partial or replaced results could otherwise
become visible.
"""

import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from compute_data_vectors import compute_cmb_covariance as covariance


class CmbCovariancePublicationTests(unittest.TestCase):
  """Exercise the private write and non-overwriting final-name boundary."""

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.output = self.root / "chains" / "cmb_covariance.npz"
    self.arrays = {
      "ell": np.array([2, 3, 4], dtype="int64"),
      "sigma_tt": np.array([1.0, 0.5, 0.25], dtype="float64"),
      "provenance": '{"source":"publication-test"}',
    }

  def tearDown(self):
    self.temporary.cleanup()

  def _temporary_files(self, output=None):
    """Return hidden staging files associated with one final name."""
    final = self.output if output is None else Path(output)
    if not final.parent.exists():
      return []
    return list(final.parent.glob("." + final.name + ".*.tmp"))

  def _assert_no_publication_debris(self, output, final_expected):
    """Reject both the intended ``.tmp`` name and NumPy's ``.tmp.npz`` trap."""
    final = Path(output)
    if not final.parent.exists():
      return
    expected = {final.name} if final_expected else set()
    observed = {entry.name for entry in final.parent.iterdir()}
    self.assertEqual(observed, expected)

  @staticmethod
  def _archive_bytes(arrays):
    """Build one small valid competitor archive entirely in memory."""
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()

  def _assert_archive_equals(self, path, arrays):
    """Require exact names, dtypes, shapes, and values after publication."""
    with np.load(path, allow_pickle=False) as archive:
      self.assertEqual(set(archive.files), set(arrays))
      for name, expected_value in arrays.items():
        expected = np.asarray(expected_value)
        observed = np.asarray(archive[name])
        self.assertEqual(observed.dtype, expected.dtype)
        self.assertEqual(observed.shape, expected.shape)
        np.testing.assert_array_equal(observed, expected)

  def test_complete_archive_is_published_without_staging(self):
    """A successful run publishes exact arrays and removes its private name."""
    returned = covariance.publish_covariance_archive(
      self.output, self.arrays)

    self.assertEqual(Path(returned), self.output)
    self._assert_archive_equals(self.output, self.arrays)
    self.assertEqual(self._temporary_files(), [])
    self._assert_no_publication_debris(self.output, final_expected=True)

  def test_existing_archive_refuses_before_any_write(self):
    """A rerun cannot replace or rewrite the file readers already trust."""
    self.output.parent.mkdir(parents=True)
    previous_arrays = {
      "ell": np.array([2], dtype="int64"),
      "sigma_tt": np.array([9.0], dtype="float64"),
    }
    previous_bytes = self._archive_bytes(previous_arrays)
    self.output.write_bytes(previous_bytes)

    with mock.patch.object(covariance.np, "savez") as save:
      with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
        covariance.publish_covariance_archive(self.output, self.arrays)

    save.assert_not_called()
    self.assertEqual(self.output.read_bytes(), previous_bytes)
    self._assert_archive_equals(self.output, previous_arrays)
    self.assertEqual(self._temporary_files(), [])

  def test_faults_before_final_name_leave_no_output_or_staging_file(self):
    """Write and link failures cannot publish partial data."""
    def write_part_then_fail(stream, **arrays):
      del arrays
      stream.write(b"partial zip bytes")
      raise OSError("injected partial-write fault")

    fault_factories = (
      ("write", lambda: mock.patch.object(
        covariance.np, "savez", side_effect=write_part_then_fail)),
      ("final-name", lambda: mock.patch.object(
        covariance.os, "link", side_effect=OSError("injected link fault"))),
    )
    for label, patch_factory in fault_factories:
      output = self.root / label / "covariance.npz"
      output.parent.mkdir(parents=True)
      with self.subTest(boundary=label), patch_factory():
        with self.assertRaises((OSError, RuntimeError)):
          covariance.publish_covariance_archive(output, self.arrays)
      self.assertFalse(os.path.lexists(output))
      self.assertEqual(self._temporary_files(output), [])
      self._assert_no_publication_debris(output, final_expected=False)

  def test_keyboard_interrupt_before_publication_cleans_partial_bytes(self):
    """Ctrl-C during the write leaves neither a trusted final nor debris."""
    def interrupt_after_bytes(stream, **arrays):
      del arrays
      stream.write(b"partial zip bytes")
      raise KeyboardInterrupt("injected interruption")

    with mock.patch.object(
        covariance.np, "savez", side_effect=interrupt_after_bytes):
      with self.assertRaises(KeyboardInterrupt):
        covariance.publish_covariance_archive(self.output, self.arrays)

    self.assertFalse(os.path.lexists(self.output))
    self._assert_no_publication_debris(self.output, final_expected=False)

  def test_late_competing_archive_is_preserved(self):
    """A file created before the final link wins unchanged."""
    competitor_arrays = {
      "ell": np.array([7], dtype="int64"),
      "sigma_tt": np.array([3.0], dtype="float64"),
    }
    competitor_bytes = self._archive_bytes(competitor_arrays)
    original_link = covariance.os.link

    def create_competitor_then_link(source, destination):
      self.output.write_bytes(competitor_bytes)
      return original_link(source, destination)

    with mock.patch.object(
        covariance.os, "link", side_effect=create_competitor_then_link):
      with self.assertRaisesRegex(FileExistsError, "created during this run"):
        covariance.publish_covariance_archive(self.output, self.arrays)

    self.assertEqual(self.output.read_bytes(), competitor_bytes)
    self._assert_archive_equals(self.output, competitor_arrays)
    self.assertEqual(self._temporary_files(), [])

  @unittest.skipIf(os.name == "nt", "symlink setup is not portable on Windows")
  def test_dangling_link_also_reserves_the_output_name(self):
    """A broken link is still user-owned and must not be replaced."""
    self.output.parent.mkdir(parents=True)
    self.output.symlink_to(self.output.parent / "missing-target.npz")

    with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
      covariance.publish_covariance_archive(self.output, self.arrays)

    self.assertTrue(self.output.is_symlink())
    self.assertFalse(self.output.exists())
    self.assertEqual(os.readlink(self.output),
                     str(self.output.parent / "missing-target.npz"))

  def test_main_refuses_existing_output_before_yaml_or_camb(self):
    """An expensive command rerun stops before reading config or solving."""
    self.output.parent.mkdir(parents=True)
    previous_bytes = self._archive_bytes(self.arrays)
    self.output.write_bytes(previous_bytes)
    arguments = [
      "compute_cmb_covariance",
      "--root", ".",
      "--fileroot", "generator",
      "--yaml", "missing.yaml",
      "--output", "cmb_covariance",
    ]
    with mock.patch.dict(os.environ, {"ROOTDIR": str(self.root)}), \
         mock.patch.object(sys, "argv", arguments), \
         mock.patch.object(
           covariance, "fiducial_spectra",
           side_effect=AssertionError("CAMB must not run")) as solve:
      with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
        covariance.main()

    solve.assert_not_called()
    self.assertEqual(self.output.read_bytes(), previous_bytes)
    self.assertEqual(self._temporary_files(), [])


if __name__ == "__main__":
  unittest.main()

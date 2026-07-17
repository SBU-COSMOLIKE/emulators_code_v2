"""Check that a CMB covariance file appears only after it is complete.

The real covariance calculation can take a long time.  These CPU tests use a
two-member archive so they can interrupt the write, file synchronization,
readback, and final-name steps without running CAMB.  A failure before the
final-name step must leave no trusted output.  A file that already owns the
requested name must remain byte-for-byte unchanged.
"""

import io
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from compute_data_vectors import compute_cmb_covariance as covariance


class CmbCovariancePublicationTests(unittest.TestCase):
  """Exercise every boundary between an in-memory result and its final name."""

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

  def test_complete_archive_is_read_back_before_publication(self):
    """A successful run publishes exact bytes and removes its staging name."""
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
    """Write, sync, readback, and link failures cannot publish partial data."""
    def write_part_then_fail(stream, **arrays):
      del arrays
      stream.write(b"partial zip bytes")
      raise OSError("injected partial-write fault")

    fault_factories = (
      ("write", lambda: mock.patch.object(
        covariance.np, "savez", side_effect=write_part_then_fail)),
      ("sync", lambda: mock.patch.object(
        covariance.os, "fsync", side_effect=OSError("injected sync fault"))),
      ("readback", lambda: mock.patch.object(
        covariance, "_validate_covariance_archive",
        side_effect=RuntimeError("injected readback fault"))),
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

  def test_bad_readback_never_receives_the_final_name(self):
    """Missing members and corrupt ZIP bytes fail the exact reopen check."""
    original_savez = covariance.np.savez

    def write_incomplete(stream, **arrays):
      original_savez(stream, ell=arrays["ell"])

    def write_corrupt(stream, **arrays):
      del arrays
      stream.write(b"this is not a NumPy archive")

    for label, writer in (("missing-member", write_incomplete),
                          ("corrupt-zip", write_corrupt)):
      output = self.root / label / "covariance.npz"
      with self.subTest(archive=label), \
           mock.patch.object(covariance.np, "savez", side_effect=writer):
        with self.assertRaisesRegex(RuntimeError, "readback validation"):
          covariance.publish_covariance_archive(output, self.arrays)
      self.assertFalse(os.path.lexists(output))
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
    """A file created during readback wins; the completed temporary loses."""
    competitor_arrays = {
      "ell": np.array([7], dtype="int64"),
      "sigma_tt": np.array([3.0], dtype="float64"),
    }
    competitor_bytes = self._archive_bytes(competitor_arrays)
    original_validate = covariance._validate_covariance_archive

    def validate_then_create_competitor(path, arrays):
      original_validate(path, arrays)
      self.output.write_bytes(competitor_bytes)

    with mock.patch.object(
        covariance, "_validate_covariance_archive",
        side_effect=validate_then_create_competitor):
      with self.assertRaisesRegex(FileExistsError, "created during this run"):
        covariance.publish_covariance_archive(self.output, self.arrays)

    self.assertEqual(self.output.read_bytes(), competitor_bytes)
    self._assert_archive_equals(self.output, competitor_arrays)
    self.assertEqual(self._temporary_files(), [])

  def test_file_sync_and_readback_happen_before_final_name(self):
    """The final name is last: write, file sync, and readback come first."""
    self.output.parent.mkdir(parents=True)
    events = []
    original_savez = covariance.np.savez
    original_fsync = covariance.os.fsync
    original_validate = covariance._validate_covariance_archive
    original_link = covariance.os.link

    def record_save(stream, **arrays):
      events.append("write")
      return original_savez(stream, **arrays)

    def record_sync(descriptor):
      mode = os.fstat(descriptor).st_mode
      events.append("file-sync" if stat.S_ISREG(mode) else "directory-sync")
      return original_fsync(descriptor)

    def record_validate(path, arrays):
      events.append("readback")
      return original_validate(path, arrays)

    def record_link(source, destination):
      events.append("final-name")
      return original_link(source, destination)

    with mock.patch.object(covariance.np, "savez", side_effect=record_save), \
         mock.patch.object(covariance.os, "fsync", side_effect=record_sync), \
         mock.patch.object(
           covariance, "_validate_covariance_archive",
           side_effect=record_validate), \
         mock.patch.object(covariance.os, "link", side_effect=record_link):
      covariance.publish_covariance_archive(self.output, self.arrays)

    self.assertLess(events.index("write"), events.index("file-sync"))
    self.assertLess(events.index("file-sync"), events.index("readback"))
    self.assertLess(events.index("readback"), events.index("final-name"))
    self.assertLess(events.index("final-name"), events.index("directory-sync"))
    self._assert_archive_equals(self.output, self.arrays)

  def test_transient_post_link_faults_are_retried(self):
    """One cleanup or directory-sync failure cannot hide a complete result."""
    self.output.parent.mkdir(parents=True)
    original_sync = covariance._sync_directory
    original_unlink = covariance.os.unlink
    sync_calls = 0
    unlink_calls = 0

    def fail_first_sync(directory):
      nonlocal sync_calls
      sync_calls += 1
      if sync_calls == 1:
        raise OSError("injected transient directory-sync fault")
      return original_sync(directory)

    def fail_first_unlink(path):
      nonlocal unlink_calls
      unlink_calls += 1
      if unlink_calls == 1:
        raise OSError("injected transient unlink fault")
      return original_unlink(path)

    with mock.patch.object(
        covariance, "_sync_directory", side_effect=fail_first_sync), \
         mock.patch.object(
           covariance.os, "unlink", side_effect=fail_first_unlink):
      covariance.publish_covariance_archive(self.output, self.arrays)

    self.assertGreaterEqual(sync_calls, 2)
    self.assertEqual(unlink_calls, 2)
    self._assert_archive_equals(self.output, self.arrays)
    self._assert_no_publication_debris(self.output, final_expected=True)

  def test_persistent_post_link_sync_fault_reports_committed_file(self):
    """A durability warning never invites a rerun over an installed archive."""
    self.output.parent.mkdir(parents=True)
    with mock.patch.object(
        covariance, "_sync_directory",
        side_effect=OSError("injected persistent directory-sync fault")):
      with self.assertRaisesRegex(
          covariance.CovariancePublicationCommittedError,
          "complete CMB covariance archive now exists"):
        covariance.publish_covariance_archive(self.output, self.arrays)

    self._assert_archive_equals(self.output, self.arrays)
    self._assert_no_publication_debris(self.output, final_expected=True)

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

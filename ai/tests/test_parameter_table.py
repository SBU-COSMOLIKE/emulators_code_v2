"""CPU tests for the strict shared parameter-table resolver."""

from dataclasses import FrozenInstanceError
import os
import tempfile
import unittest

import numpy as np

from emulator.parameter_table import resolve_parameter_table


class ParameterTableTest(unittest.TestCase):
  """Pin sidecar lookup, declaration law, and named numeric slicing."""

  @staticmethod
  def _write_sidecar(path, tokens):
    with open(path, "w") as handle:
      for token in tokens:
        handle.write(f"{token} latex-{token}\n")

  @staticmethod
  def _write_table(path, rows):
    np.savetxt(path, np.asarray(rows, dtype=np.float32), fmt="%.9e")

  def _valid(self, tmp, stem="sample", tokens=("a", "b", "chi2*"),
             rows=None, inputs=("a", "b"), outputs=("chi2",)):
    if rows is None:
      rows = [[1.0, 2.0, 3.0, 4.0, 5.0],
              [6.0, 7.0, 8.0, 9.0, 10.0]]
    params = os.path.join(tmp, stem + ".txt")
    sidecar = os.path.join(tmp, stem + ".paramnames")
    self._write_table(params, rows)
    self._write_sidecar(sidecar, tokens)
    return resolve_parameter_table(params, inputs, outputs), params, sidecar

  def test_plain_stem_and_frozen_record(self):
    with tempfile.TemporaryDirectory() as tmp:
      resolved, _, sidecar = self._valid(tmp)
      self.assertEqual(resolved.sidecar_path, sidecar)
      self.assertEqual(
        resolved.declarations,
        (("a", False, 2), ("b", False, 3), ("chi2", True, 4)))
      self.assertEqual(resolved.inputs.dtype, np.float32)
      self.assertEqual(resolved.outputs.dtype, np.float32)
      with self.assertRaises(FrozenInstanceError):
        resolved.sidecar_path = "changed"

  def test_numeric_chain_falls_back_to_shared_root(self):
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.7.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_table(params, [[1, 2, 3]])
      self._write_sidecar(sidecar, ["a"])
      resolved = resolve_parameter_table(params, ["a"])
      self.assertEqual(resolved.sidecar_path, sidecar)

  def test_numeric_chain_prefers_exact_stem(self):
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.7.txt")
      exact = os.path.join(tmp, "chain.7.paramnames")
      root = os.path.join(tmp, "chain.paramnames")
      self._write_table(params, [[1, 2, 3]])
      self._write_sidecar(exact, ["right"])
      self._write_sidecar(root, ["wrong"])
      resolved = resolve_parameter_table(params, ["right"])
      self.assertEqual(resolved.sidecar_path, exact)

  def test_nonnumeric_dotted_stem_is_not_stripped(self):
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "lcdm.v2.txt")
      exact = os.path.join(tmp, "lcdm.v2.paramnames")
      wrong = os.path.join(tmp, "lcdm.paramnames")
      self._write_table(params, [[1, 2, 3]])
      self._write_sidecar(exact, ["a"])
      self._write_sidecar(wrong, ["wrong"])
      resolved = resolve_parameter_table(params, ["a"])
      self.assertEqual(resolved.sidecar_path, exact)

  def test_utf8_bom_is_not_part_of_the_first_logical_name(self):
    """GetDist's UTF-8-with-BOM sidecars resolve like plain UTF-8."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "bom.txt")
      sidecar = os.path.join(tmp, "bom.paramnames")
      self._write_table(params, [[1, 2, 3]])
      with open(sidecar, "w", encoding="utf-8-sig") as handle:
        handle.write("a A\n")

      resolved = resolve_parameter_table(params, ["a"])

      self.assertEqual(resolved.declarations, (("a", False, 2),))
      np.testing.assert_array_equal(
        resolved.inputs, np.asarray([[3]], dtype=np.float32))

  def test_zero_derived_columns_returns_two_dimensional_empty_output(self):
    with tempfile.TemporaryDirectory() as tmp:
      resolved, _, _ = self._valid(
        tmp, tokens=("a", "b"),
        rows=[[1, 2, 3, 4], [5, 6, 7, 8]], outputs=())
      self.assertEqual(resolved.outputs.shape, (2, 0))
      self.assertEqual(resolved.outputs.dtype, np.float32)

  def test_one_row_stays_two_dimensional(self):
    with tempfile.TemporaryDirectory() as tmp:
      resolved, _, _ = self._valid(tmp, rows=[[1, 2, 3, 4, 5]])
      self.assertEqual(resolved.inputs.shape, (1, 2))
      self.assertEqual(resolved.outputs.shape, (1, 1))

  def test_multiple_interleaved_derived_columns_map_by_name(self):
    with tempfile.TemporaryDirectory() as tmp:
      resolved, _, _ = self._valid(
        tmp,
        tokens=("d0*", "a", "d1*", "b", "d2*"),
        rows=[[1, 2, 30, 10, 31, 11, 32],
              [3, 4, 40, 20, 41, 21, 42]],
        outputs=("d2", "d0", "d1"))
      np.testing.assert_array_equal(
        resolved.inputs, np.asarray([[10, 11], [20, 21]], np.float32))
      np.testing.assert_array_equal(
        resolved.outputs,
        np.asarray([[32, 30, 31], [42, 40, 41]], np.float32))

  def test_current_generator_layout_maps_without_numeric_drift(self):
    with tempfile.TemporaryDirectory() as tmp:
      rows = np.asarray([
        [1.0, -7.0, 0.02237, 0.1200, 67.4, 13.5],
        [2.0, -8.0, 0.02241, 0.1192, 68.1, 15.25],
      ], dtype=np.float32)
      resolved, _, _ = self._valid(
        tmp, tokens=("omegabh2", "omegach2", "H0", "chi2*"),
        rows=rows, inputs=("omegabh2", "omegach2", "H0"),
        outputs=("chi2",))
      np.testing.assert_array_equal(resolved.inputs, rows[:, 2:5])
      np.testing.assert_array_equal(resolved.outputs, rows[:, 5:6])

  def test_plain_and_starred_name_are_duplicate_declarations(self):
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "x.txt")
      self._write_table(params, [[1, 2, 3, 4]])
      self._write_sidecar(os.path.join(tmp, "x.paramnames"), ["a", "a*"])
      with self.assertRaisesRegex(ValueError, "duplicate normalized"):
        resolve_parameter_table(params, ["a"])

  def test_invalid_or_repeated_getdist_markers_refuse(self):
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "markers.txt")
      sidecar = os.path.join(tmp, "markers.paramnames")
      self._write_table(params, [[1, 2, 3]])
      for token in ("a**", "a?"):
        with self.subTest(token=token):
          self._write_sidecar(sidecar, [token])
          with self.assertRaisesRegex(ValueError, "allowed exactly once"):
            resolve_parameter_table(params, ["a"])

  def test_duplicate_requested_input_and_output_names_refuse(self):
    with tempfile.TemporaryDirectory() as tmp:
      _, params, _ = self._valid(tmp)
      with self.assertRaisesRegex(ValueError, "input names contain duplicates"):
        resolve_parameter_table(params, ["a", "a"])
      with self.assertRaisesRegex(ValueError, "output names contain duplicates"):
        resolve_parameter_table(params, ["a", "b"], ["chi2", "chi2"])

  def test_input_output_overlap_and_nonderived_output_refuse(self):
    with tempfile.TemporaryDirectory() as tmp:
      _, params, _ = self._valid(tmp)
      with self.assertRaisesRegex(ValueError, "overlap.*not derived"):
        resolve_parameter_table(params, ["a", "b"], ["a"])

  def test_missing_input_and_output_names_refuse(self):
    with tempfile.TemporaryDirectory() as tmp:
      _, params, _ = self._valid(tmp)
      with self.assertRaisesRegex(ValueError, r"inputs=\['missing'\]"):
        resolve_parameter_table(params, ["a", "b", "missing"], ["chi2"])
      with self.assertRaisesRegex(ValueError, r"outputs=\['missing'\]"):
        resolve_parameter_table(params, ["a", "b"], ["missing"])

  def test_permuted_or_extra_nonderived_declarations_refuse(self):
    with tempfile.TemporaryDirectory() as tmp:
      _, params, sidecar = self._valid(tmp)
      self._write_sidecar(sidecar, ["b", "a", "chi2*"])
      with self.assertRaisesRegex(ValueError, "order included"):
        resolve_parameter_table(params, ["a", "b"], ["chi2"])
      self._write_sidecar(sidecar, ["a", "b", "extra", "chi2*"])
      self._write_table(params, [[1, 2, 3, 4, 5, 6]])
      with self.assertRaisesRegex(ValueError, "declarations=.*extra"):
        resolve_parameter_table(params, ["a", "b"], ["chi2"])

  def test_width_minus_or_plus_one_refuses(self):
    with tempfile.TemporaryDirectory() as tmp:
      _, params, _ = self._valid(tmp)
      self._write_table(params, [[1, 2, 3, 4]])
      with self.assertRaisesRegex(ValueError, "width 4.*expected 5"):
        resolve_parameter_table(params, ["a", "b"], ["chi2"])
      self._write_table(params, [[1, 2, 3, 4, 5, 6]])
      with self.assertRaisesRegex(ValueError, "width 6.*expected 5"):
        resolve_parameter_table(params, ["a", "b"], ["chi2"])

  def test_empty_numeric_table_refuses(self):
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "empty.txt")
      sidecar = os.path.join(tmp, "empty.paramnames")
      open(params, "w").close()
      self._write_sidecar(sidecar, ["a"])
      with self.assertRaisesRegex(ValueError, "is empty"):
        resolve_parameter_table(params, ["a"])

  def test_missing_sidecar_names_candidates_and_migration(self):
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "legacy.3.txt")
      self._write_table(params, [[1, 2, 3]])
      with self.assertRaisesRegex(ValueError, "legacy\\.3\\.paramnames") as ctx:
        resolve_parameter_table(params, ["a"])
      message = str(ctx.exception)
      self.assertIn(os.path.join(tmp, "legacy.paramnames"), message)
      self.assertIn("cannot be mapped safely by position", message)


if __name__ == "__main__":
  unittest.main()

"""Focused tests for parameter-dump to .paramnames sidecar resolution."""

import os
import tempfile
import unittest

from emulator import data_staging


class ParamnamesResolutionTest(unittest.TestCase):
  """Exercise every supported parameter-dump naming convention."""

  @staticmethod
  def _write_sidecar(path, names):
    """Write a minimal getdist sidecar plus one ignored derived column.

    Arguments:
      path  = destination .paramnames path.
      names = non-derived names in whitening order.

    Returns:
      None.
    """
    with open(path, "w") as handle:
      for name in names:
        handle.write(name + " " + name + "\n")
      handle.write("chi2* chi2\n")

  def test_numeric_chain_uses_shared_root_sidecar(self):
    """X.1.txt checks X.paramnames when no exact-stem sidecar exists."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_sidecar(sidecar, ["omegab", "H0"])

      found = data_staging.check_source_paramnames(
        params_path=params, covmat_names=["omegab", "H0"])

      self.assertEqual(found, sidecar)

  def test_plain_dump_uses_exact_stem_sidecar(self):
    """X.txt checks X.paramnames."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_sidecar(sidecar, ["omegab", "H0"])

      found = data_staging.check_source_paramnames(
        params_path=params, covmat_names=["omegab", "H0"])

      self.assertEqual(found, sidecar)

  def test_nonnumeric_dotted_stem_is_not_stripped(self):
    """lcdm.v2.txt pairs with lcdm.v2.paramnames, never lcdm.paramnames."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "lcdm.v2.txt")
      exact = os.path.join(tmp, "lcdm.v2.paramnames")
      wrong_root = os.path.join(tmp, "lcdm.paramnames")
      self._write_sidecar(exact, ["omegab", "H0"])
      self._write_sidecar(wrong_root, ["wrong"])

      found = data_staging.check_source_paramnames(
        params_path=params, covmat_names=["omegab", "H0"])

      self.assertEqual(found, exact)

  def test_absent_sidecar_remains_compatible(self):
    """The ordinary staging path explicitly skips a missing sidecar."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "legacy.1.txt")

      found = data_staging.check_source_paramnames(
        params_path=params, covmat_names=["omegab", "H0"])

      self.assertIsNone(found)

  def test_numeric_chain_root_mismatch_refuses(self):
    """A mismatched X.paramnames is no longer hidden by the .1 suffix."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_sidecar(sidecar, ["H0", "omegab"])

      with self.assertRaisesRegex(ValueError, "whitening would pair wrong"):
        data_staging.check_source_paramnames(
          params_path=params, covmat_names=["omegab", "H0"])

  def test_load_source_checks_chain_root_before_loading_arrays(self):
    """The production loader reaches the root-sidecar integrity check."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_sidecar(sidecar, ["H0", "omegab"])

      with self.assertRaisesRegex(ValueError, "whitening would pair wrong"):
        data_staging.load_source(
          dv_path=os.path.join(tmp, "not-read.npy"),
          params_path=params,
          names=["omegab", "H0"],
          omegabh2_hi=None,
          n_keep=1,
          gen=data_staging.torch.Generator())


if __name__ == "__main__":
  unittest.main()

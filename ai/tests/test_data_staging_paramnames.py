"""Focused tests for strict parameter-table / .paramnames resolution."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from emulator import data_staging, fixed_facts
from emulator.experiment import EmulatorExperiment
from emulator.parameter_table import resolve_parameter_table


class ParamnamesResolutionTest(unittest.TestCase):
  """Exercise consumer-visible sidecar resolution and refusal semantics."""

  @staticmethod
  def _write_table(path, rows=None):
    """Write weight, log-posterior, two inputs, and one diagnostic."""
    if rows is None:
      rows = [[1.0, 0.0, 0.022, 70.0, 3.0],
              [1.0, 0.0, 0.023, 68.0, 4.0]]
    np.savetxt(path, np.asarray(rows, dtype=np.float32))

  @staticmethod
  def _write_sidecar(path, names):
    with open(path, "w") as handle:
      for name in names:
        handle.write(name + " " + name.rstrip("*") + "\n")

  @staticmethod
  def _write_facts(params_path, names, *, text=None):
    """Write the producer record paired with a plain or numbered chain."""
    base = os.path.splitext(params_path)[0]
    root, chain_ext = os.path.splitext(base)
    if chain_ext[1:].isdigit():
      base = root
    sidecar = base + fixed_facts.SIDECAR_SUFFIX
    if text is None:
      text = fixed_facts.synthetic_sidecar(
        names=names,
        label="parameter-table-" + os.path.basename(base),
        family="scalar",
        support=None)
    with open(sidecar, "w", encoding="utf-8", newline="") as handle:
      handle.write(text)
    return sidecar, text

  def test_numeric_chain_uses_shared_root_sidecar(self):
    """X.1.txt resolves X.paramnames and returns a stable 2-D table."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])

      found = resolve_parameter_table(params, ["omegab", "H0"])

      self.assertEqual(found.sidecar_path, sidecar)
      self.assertEqual(found.inputs.shape, (2, 2))
      np.testing.assert_array_equal(
        found.inputs,
        np.asarray([[0.022, 70.0], [0.023, 68.0]], dtype=np.float32))

  def test_plain_dump_uses_exact_stem_sidecar(self):
    """X.txt resolves X.paramnames."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])

      found = resolve_parameter_table(params, ["omegab", "H0"])

      self.assertEqual(found.sidecar_path, sidecar)

  def test_nonnumeric_dotted_stem_is_not_stripped(self):
    """lcdm.v2.txt uses lcdm.v2.paramnames, never lcdm.paramnames."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "lcdm.v2.txt")
      exact = os.path.join(tmp, "lcdm.v2.paramnames")
      wrong_root = os.path.join(tmp, "lcdm.paramnames")
      self._write_table(params)
      self._write_sidecar(exact, ["omegab", "H0", "chi2*"])
      self._write_sidecar(wrong_root, ["wrong", "H0", "chi2*"])

      found = resolve_parameter_table(params, ["omegab", "H0"])

      self.assertEqual(found.sidecar_path, exact)

  def test_exact_numeric_stem_takes_precedence_over_chain_root(self):
    """X.1.paramnames wins over X.paramnames when both exist."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      exact = os.path.join(tmp, "chain.1.paramnames")
      root = os.path.join(tmp, "chain.paramnames")
      self._write_table(params)
      self._write_sidecar(exact, ["omegab", "H0", "chi2*"])
      self._write_sidecar(root, ["wrong", "H0", "chi2*"])

      found = resolve_parameter_table(params, ["omegab", "H0"])

      self.assertEqual(found.sidecar_path, exact)

  def test_absent_sidecar_is_a_migration_refusal(self):
    """Ordinary staging no longer guesses legacy positional columns."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "legacy.1.txt")
      self._write_table(params)

      with self.assertRaisesRegex(ValueError, r"\.paramnames"):
        resolve_parameter_table(params, ["omegab", "H0"])

  def test_numeric_chain_root_mismatch_refuses(self):
    """A reordered sampled block cannot silently pair with the covmat."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["H0", "omegab", "chi2*"])

      with self.assertRaises(ValueError):
        resolve_parameter_table(params, ["omegab", "H0"])

  def test_load_source_refuses_before_opening_data_vector(self):
    """A bad declaration wins over the deliberately absent DV path."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      self._write_table(params)

      with self.assertRaisesRegex(ValueError, r"\.paramnames"):
        data_staging.load_source(
          dv_path=os.path.join(tmp, "must-not-open.npy"),
          params_path=params,
          names=["omegab", "H0"],
          omegabh2_hi=None,
          n_keep=1,
          gen=data_staging.torch.Generator())

  def test_missing_facts_refuses_before_opening_data_vector(self):
    """A valid parameter schema cannot make a record-less dump trainable."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])

      with mock.patch.object(
          data_staging.np, "load",
          side_effect=AssertionError("the DV must not be opened")) as open_dv:
        with self.assertRaisesRegex(
            ValueError, r"(?s)\.facts\.yaml.*[Rr]e-generate"):
          data_staging.load_source(
            dv_path=os.path.join(tmp, "must-not-open.npy"),
            params_path=params,
            names=["omegab", "H0"],
            omegabh2_hi=None,
            n_keep=1,
            gen=data_staging.torch.Generator())

      open_dv.assert_not_called()

  def test_malformed_facts_refuses_before_opening_data_vector(self):
    """Malformed producer YAML is diagnosed before a large DV is mapped."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])
      self._write_facts(params, ["omegab", "H0"], text="fixed_facts: [\n")

      with mock.patch.object(
          data_staging.np, "load",
          side_effect=AssertionError("the DV must not be opened")) as open_dv:
        with self.assertRaisesRegex(
            ValueError, r"(?s)does not parse as YAML.*Re-generate"):
          data_staging.load_source(
            dv_path=os.path.join(tmp, "must-not-open.npy"),
            params_path=params,
            names=["omegab", "H0"],
            omegabh2_hi=None,
            n_keep=1,
            gen=data_staging.torch.Generator())

      open_dv.assert_not_called()

  def test_facts_name_order_refuses_before_opening_data_vector(self):
    """Producer names cannot silently reverse the table's sampled columns."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])
      self._write_facts(params, ["H0", "omegab"])

      with mock.patch.object(
          data_staging.np, "load",
          side_effect=AssertionError("the DV must not be opened")) as open_dv:
        with self.assertRaisesRegex(
            ValueError, r"record disagree.*sampled parameters"):
          data_staging.load_source(
            dv_path=os.path.join(tmp, "must-not-open.npy"),
            params_path=params,
            names=["omegab", "H0"],
            omegabh2_hi=None,
            n_keep=1,
            gen=data_staging.torch.Generator())

      open_dv.assert_not_called()

  def test_valid_facts_text_is_preserved_byte_for_byte_as_text(self):
    """Validation never rewrites the producer's line endings or wording."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "chain.1.txt")
      sidecar = os.path.join(tmp, "chain.paramnames")
      dv = os.path.join(tmp, "chain.npy")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])
      np.save(dv, np.ones((2, 2), dtype=np.float32))
      text = fixed_facts.synthetic_sidecar(
        names=["omegab", "H0"],
        label="exact-sidecar-text",
        family="cmb",
        support=None).replace("\n", "\r\n")
      self._write_facts(params, ["omegab", "H0"], text=text)

      staged = data_staging.load_source(
        dv_path=dv,
        params_path=params,
        names=["omegab", "H0"],
        omegabh2_hi=None,
        n_keep=1,
        gen=data_staging.torch.Generator().manual_seed(4),
        verbose=False)

      self.assertEqual(staged["facts_yaml"], text)

  def test_scalar_missing_facts_refuses_before_staging(self):
    """The scalar loader has the same required-record boundary."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "scalar.1.txt")
      sidecar = os.path.join(tmp, "scalar.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])

      with mock.patch.object(
          data_staging, "stage_source",
          side_effect=AssertionError("staging must not run")) as stage:
        with self.assertRaisesRegex(ValueError, r"\.facts\.yaml"):
          data_staging.load_scalar_source(
            params_path=params,
            in_names=["omegab", "H0"],
            out_names=["chi2"],
            n_keep=1,
            gen=data_staging.torch.Generator())

      stage.assert_not_called()

  def test_scalar_pool_size_and_staging_share_the_resolver(self):
    """Pool sizing and scalar staging accept and count the same table."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "scalar.1.txt")
      sidecar = os.path.join(tmp, "scalar.paramnames")
      self._write_table(params)
      self._write_sidecar(sidecar, ["omegab", "H0", "chi2*"])
      self._write_facts(params, ["omegab", "H0"])
      cuts = {"omegabh2_hi": 0.03}

      exp = EmulatorExperiment.__new__(EmulatorExperiment)
      exp.data = {"train_params": params, "param_cuts": cuts}
      exp.names = ["omegab", "H0"]
      exp.outputs = ["chi2"]
      exp._scalar = True

      pool = exp.pool_size()
      staged = data_staging.load_scalar_source(
        params_path=params,
        in_names=exp.names,
        out_names=exp.outputs,
        n_keep=pool,
        omegabh2_hi=cuts["omegabh2_hi"],
        gen=data_staging.torch.Generator().manual_seed(7),
        verbose=False)

      self.assertEqual(pool, 2)
      self.assertEqual(staged["C"].shape, (pool, 2))
      self.assertEqual(staged["dv"].shape, (pool, 1))

  def test_optional_cut_families_share_the_stageable_ceiling(self):
    """No-cut and active-cut pools are the exact real staging ceilings."""
    with tempfile.TemporaryDirectory() as tmp:
      params = os.path.join(tmp, "families.1.txt")
      sidecar = os.path.join(tmp, "families.paramnames")
      dv = os.path.join(tmp, "families.npy")
      self._write_table(params, rows=[
        [1, 0, 0.040, 50, 101],
        [1, 0, 0.050, 80, 202],
        [1, 0, 0.030, 60, 303],
        [1, 0, 0.060, 90, 404],
      ])
      self._write_sidecar(sidecar, ["omegab", "H0", "target*"])
      self._write_facts(params, ["omegab", "H0"])
      np.save(dv, np.asarray([
        [1001, 1002],
        [2001, 2002],
        [3001, 3002],
        [4001, 4002],
      ], dtype=np.float32))

      # omega_b h^2 is .010, .032, .0108, .0486 respectively, so the
      # active window retains exactly rows 0 and 2.  None means the
      # param_cuts key is genuinely absent, not present with null values.
      scenarios = (("no-cuts", None, 4),
                   ("active-cut", {"omegabh2_hi": 0.02}, 2))
      for family in ("scalar", "cmb", "grid", "grid2d"):
        for label, cuts, expected_pool in scenarios:
          with self.subTest(family=family, scenario=label):
            exp = EmulatorExperiment.__new__(EmulatorExperiment)
            exp.data = {
              "train_params": params,
              "split_seed": 29,
              "n_train": expected_pool,
              "ram_frac": 1.0,
            }
            if family != "scalar":
              exp.data["train_dv"] = dv
            if cuts is not None:
              exp.data["param_cuts"] = cuts
            exp.names = ["omegab", "H0"]
            exp.outputs = ["target"] if family == "scalar" else []
            exp._scalar = family == "scalar"
            exp._cmb = family == "cmb"
            exp._grid = family == "grid"
            exp._grid2d = family == "grid2d"
            exp.quiet = True

            # Grid2d's named loader and pool are real. Only its downstream
            # law-space transform is stubbed; the witness records that it saw
            # the real disk-backed dump and the exact selected ceiling.
            law_calls = []
            if exp._grid2d:
              exp._grid2d_train_tmp = None
              exp.grid2d = {"train_base": None}

              def law_stub(*, src, base_path, with_means):
                law_calls.append((len(src["idx"]),
                                  isinstance(src["dv"], np.memmap),
                                  base_path, with_means))
                return None

              exp._grid2d_law_rows = law_stub

            pool = exp.pool_size()
            staged = exp.stage_train(n_train=pool)

            self.assertEqual(pool, expected_pool)
            self.assertEqual(len(staged["idx"]), pool)
            with self.assertRaises(ValueError):
              exp.stage_train(n_train=pool + 1)
            if exp._grid2d:
              self.assertEqual(
                law_calls, [(pool, True, None, True)],
                "the refused pool+1 call must not reach the law transform")


if __name__ == "__main__":
  unittest.main()

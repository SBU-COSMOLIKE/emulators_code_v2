#!/usr/bin/env python3
"""Challenge the fixed flat-LCDM parameter boundary without running CAMB.

This adversarial-review witness exercises every required parameter refusal.
It also runs the shipped configuration through the real producer entry point
with the expensive numerical work replaced by finite local fixtures. The
entry-point leg proves that Cobaya consumption and persisted provenance see
the same complete mapping object. Four malformed entry-point runs make the
producer's schema, PSD, and final-finiteness validator calls load-bearing.

This is not a board acceptance gate. A passing run means the parameter schema
rejects known omissions and ambiguities while preserving the shipped mapping.

PS: Resolved means that every value needed to identify the fiducial cosmology
is explicit before the covariance producer constructs Cobaya or writes output.
"""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from compute_data_vectors import compute_cmb_covariance as covariance


def make_valid_params():
  """Build one complete fixed flat-LCDM parameter mapping.

  Arguments:
    None.

  Returns:
    New mapping with the shipped fiducial parameter values.
  """
  params = {}
  params["As"] = 2.1e-9
  params["ns"] = 0.9660
  params["H0"] = 67.36
  params["omegabh2"] = 0.02237
  params["omegach2"] = 0.1200
  params["tau"] = 0.0544
  params["mnu"] = 0.06
  params["omk"] = 0.0
  params["w"] = -1.0
  params["wa"] = 0.0
  return params


def expect_refusal(label, params):
  """Require the public parameter validator to refuse one mapping.

  Arguments:
    label  = plain-language description printed for the check.
    params = candidate fiducial parameter mapping.

  Returns:
    None.
  """
  try:
    covariance.validate_lcdm_params(params=params)
  except ValueError as error:
    print("  [PASS] " + label + " (" + str(error) + ")")
    return
  raise AssertionError(label + " was accepted")


def check_refusals():
  """Exercise omissions, non-finite values, and ambiguous alternatives.

  Arguments:
    None.

  Returns:
    None.
  """
  expect_refusal(label="empty params mapping refuses", params={})

  candidate = make_valid_params()
  candidate["As"] = True
  expect_refusal(label="boolean numeric value refuses", params=candidate)

  candidate = make_valid_params()
  candidate["As"] = float("nan")
  expect_refusal(label="NaN value refuses", params=candidate)

  candidate = make_valid_params()
  candidate["ns"] = float("inf")
  expect_refusal(label="infinite value refuses", params=candidate)

  candidate = make_valid_params()
  del candidate["As"]
  expect_refusal(label="missing amplitude refuses", params=candidate)

  candidate = make_valid_params()
  candidate["logA"] = 3.0
  expect_refusal(label="double amplitude refuses", params=candidate)

  candidate = make_valid_params()
  del candidate["H0"]
  expect_refusal(label="missing expansion parameter refuses", params=candidate)

  candidate = make_valid_params()
  candidate["thetastar"] = 0.0104
  expect_refusal(label="double expansion parameter refuses", params=candidate)

  candidate = make_valid_params()
  candidate["omk"] = float("nan")
  expect_refusal(label="NaN curvature refuses before the flatness pin",
                 params=candidate)

  fixed_names = ("omk", "w", "wa")
  for name in fixed_names:
    candidate = make_valid_params()
    del candidate[name]
    expect_refusal(label="missing explicit " + name + " refuses",
                   params=candidate)

  candidate = make_valid_params()
  candidate["omk"] = 0.01
  expect_refusal(label="non-flat curvature refuses", params=candidate)


def load_shipped_params():
  """Load scalar values from the shipped YAML ``params`` block.

  Arguments:
    None.

  Returns:
    Parsed fiducial parameter mapping from the repository example.
  """
  config_path = REPO_ROOT / "example_yamls" / "cmb_covariance_lcdm.yaml"
  params = {}
  in_params = False
  with config_path.open(encoding="utf-8") as stream:
    for raw_line in stream:
      line = raw_line.strip()
      if line == "params:":
        in_params = True
        continue
      if in_params and line == "cov_args:":
        break
      if not in_params or line == "" or line.startswith("#"):
        continue
      name, scalar = line.split(":", maxsplit=1)
      params[name.strip()] = float(scalar.strip())
  return params


def make_instrumented_info(params):
  """Build the non-parameter blocks needed by the instrumented entry point.

  Arguments:
    params = shipped fiducial parameter mapping.

  Returns:
    Complete top-level producer configuration using the shipped experiment.
  """
  noise = {}
  noise["delta_tt"] = 1.0
  noise["delta_ee"] = 1.4
  noise["delta_te"] = 0.0
  noise["beam_fwhm"] = 1.0
  nongaussian = {}
  nongaussian["enabled"] = False
  nongaussian["lens_lmax"] = 3000
  nongaussian["band_width"] = 20
  nongaussian["step_fracs"] = [0.01, 0.02, 0.04]
  nongaussian["converge_rtol"] = 0.05
  cov_args = {}
  cov_args["lmax"] = 5000
  cov_args["fsky"] = 1.0
  cov_args["noise"] = noise
  cov_args["nongaussian"] = nongaussian
  camb = {}
  camb["extra_args"] = {}
  theory = {}
  theory["camb"] = camb
  info = {}
  info["theory"] = theory
  info["params"] = params
  info["cov_args"] = cov_args
  return info


def finite_cls(lmax):
  """Build finite zero spectra for the instrumented entry-point run.

  Arguments:
    lmax = largest requested covariance multipole.

  Returns:
    Mapping of four arrays over multipoles zero through ``lmax``.
  """
  values = np.zeros(shape=(int(lmax) + 1,), dtype="float64")
  spectra = {}
  spectra["tt"] = values.copy()
  spectra["te"] = values.copy()
  spectra["ee"] = values.copy()
  spectra["pp"] = values.copy()
  return spectra


def finite_gaussian_blocks(ell, cls, noise, fsky):
  """Build finite Gaussian arrays for the instrumented entry-point run.

  Arguments:
    ell   = covariance multipoles whose length fixes every output width.
    cls   = unused finite fiducial spectra mapping.
    noise = unused finite instrumental-noise mapping.
    fsky  = unused finite observed sky fraction.

  Returns:
    Mapping containing every Gaussian array consumed by ``main``.
  """
  del cls, noise, fsky
  values = np.ones(shape=np.asarray(ell).shape, dtype="float64")
  blocks = {}
  block_names = (
    "var_tt",
    "var_te",
    "var_ee",
    "var_pp",
    "gauss_tt_te",
    "gauss_tt_ee",
    "gauss_te_ee",
  )
  for name in block_names:
    blocks[name] = values.copy()
  return blocks


def finite_fiducial_spectra(info, lmax):
  """Build the finite fiducial return used by instrumented main runs.

  Arguments:
    info = unused producer configuration.
    lmax = largest requested covariance multipole.

  Returns:
    Pair containing finite spectra and an unused CAMB-data sentinel.
  """
  del info
  return finite_cls(lmax=lmax), object()


def run_instrumented_main(info, fiducial_fixture=finite_fiducial_spectra,
                          gaussian_fixture=finite_gaussian_blocks):
  """Drive the real producer entry point under finite local fixtures.

  The real entry point is used, but its CAMB solve and Gaussian arithmetic are
  replaced with finite fixtures. ``json.dumps`` is instrumented before it
  serializes provenance, and ``np.savez`` is instrumented before publication.

  Arguments:
    info             = complete candidate producer configuration.
    fiducial_fixture = replacement for the CAMB-backed spectra producer.
    gaussian_fixture = replacement for Gaussian covariance arithmetic.

  Returns:
    Captured consumption, provenance, and publication surfaces.
  """
  captures = {}
  original_fiducial = covariance.fiducial_spectra
  original_gaussian = covariance.gaussian_blocks
  original_noise = covariance.noise_spectrum
  original_dumps = covariance.json.dumps
  original_savez = covariance.np.savez
  original_argv = sys.argv
  original_rootdir = os.environ.get("ROOTDIR")
  original_yaml = sys.modules.get("yaml")

  fake_yaml = types.ModuleType("yaml")

  def fake_safe_load(stream):
    """Return the caller's candidate producer configuration."""
    del stream
    return info

  fake_yaml.safe_load = fake_safe_load

  def fake_fiducial_spectra(info, lmax):
    """Capture the mapping delivered to the Cobaya-facing boundary."""
    captures["consumed"] = info["params"]
    return fiducial_fixture(info=info, lmax=lmax)

  def fake_noise_spectrum(ell, delta_arcmin, beam_fwhm_arcmin):
    """Return finite local noise with the requested multipole shape."""
    del delta_arcmin, beam_fwhm_arcmin
    return np.zeros(shape=np.asarray(ell).shape, dtype="float64")

  def capture_dumps(value, *args, **kwargs):
    """Capture provenance immediately before JSON serialization."""
    captures["persisted"] = value["fiducial_params"]
    return original_dumps(value, *args, **kwargs)

  def capture_savez(path, **arrays):
    """Capture publication without writing the instrumented fixture."""
    captures["output_path"] = path
    captures["output"] = arrays

  with tempfile.TemporaryDirectory() as temp_dir:
    config_dir = Path(temp_dir) / "project" / "config"
    config_dir.mkdir(parents=True)
    source = REPO_ROOT / "example_yamls" / "cmb_covariance_lcdm.yaml"
    destination = config_dir / source.name
    shutil.copyfile(source, destination)
    try:
      covariance.fiducial_spectra = fake_fiducial_spectra
      covariance.gaussian_blocks = gaussian_fixture
      covariance.noise_spectrum = fake_noise_spectrum
      covariance.json.dumps = capture_dumps
      covariance.np.savez = capture_savez
      sys.modules["yaml"] = fake_yaml
      os.environ["ROOTDIR"] = temp_dir
      sys.argv = [
        "compute_cmb_covariance",
        "--root",
        "project",
        "--fileroot",
        "config",
        "--yaml",
        source.name,
        "--output",
        "identity_witness",
      ]
      covariance.main()
    finally:
      covariance.fiducial_spectra = original_fiducial
      covariance.gaussian_blocks = original_gaussian
      covariance.noise_spectrum = original_noise
      covariance.json.dumps = original_dumps
      covariance.np.savez = original_savez
      sys.argv = original_argv
      if original_yaml is None:
        del sys.modules["yaml"]
      else:
        sys.modules["yaml"] = original_yaml
      if original_rootdir is None:
        del os.environ["ROOTDIR"]
      else:
        os.environ["ROOTDIR"] = original_rootdir

  return captures


def expect_main_refusal(label, info, message,
                        fiducial_fixture=finite_fiducial_spectra,
                        gaussian_fixture=finite_gaussian_blocks):
  """Require one malformed case to refuse through the real entry point.

  Arguments:
    label            = plain-language description printed for the check.
    info             = candidate producer configuration.
    message          = required fragment of the entry-point refusal.
    fiducial_fixture = replacement for the CAMB-backed spectra producer.
    gaussian_fixture = replacement for Gaussian covariance arithmetic.

  Returns:
    None.
  """
  try:
    run_instrumented_main(info=info,
                          fiducial_fixture=fiducial_fixture,
                          gaussian_fixture=gaussian_fixture)
  except ValueError as error:
    if message not in str(error):
      raise AssertionError(
        label + " refused for the wrong reason: " + str(error)) from error
    print("  [PASS] " + label + " (" + str(error) + ")")
    return
  raise AssertionError(label + " was accepted through main()")


def check_main_refusals():
  """Prove all four validator calls in ``main`` are load-bearing.

  Arguments:
    None.

  Returns:
    None.
  """
  info = make_instrumented_info(params=make_valid_params())
  info["cov_args"]["unexpected"] = 1.0
  expect_main_refusal(label="cov_args schema violation refuses in main",
                      info=info, message="unexpected")

  params = make_valid_params()
  params["unexpected"] = 1.0
  info = make_instrumented_info(params=params)
  expect_main_refusal(label="params schema violation refuses in main",
                      info=info, message="non-LCDM")

  def non_psd_fiducial(info, lmax):
    """Return finite spectra whose T/E block is not PSD."""
    del info
    spectra = finite_cls(lmax=lmax)
    spectra["te"][:] = 1.0
    return spectra, object()

  info = make_instrumented_info(params=make_valid_params())
  expect_main_refusal(
    label="post-signal PSD violation refuses in main", info=info,
    message="signal plus noise is not positive semidefinite",
    fiducial_fixture=non_psd_fiducial)

  def nonfinite_gaussian(ell, cls, noise, fsky):
    """Return one non-finite output outside the joint T/E PSD block."""
    blocks = finite_gaussian_blocks(ell=ell, cls=cls, noise=noise,
                                    fsky=fsky)
    blocks["var_pp"][0] = float("nan")
    return blocks

  info = make_instrumented_info(params=make_valid_params())
  expect_main_refusal(
    label="non-finite Gaussian output refuses in main", info=info,
    message="non-finite covariance output 'sigma_pp'",
    gaussian_fixture=nonfinite_gaussian)


def check_shipped_identity():
  """Prove shipped validation, completeness, and consume/persist identity.

  Arguments:
    None.

  Returns:
    None.
  """
  shipped_params = load_shipped_params()
  validated = covariance.validate_lcdm_params(params=shipped_params)
  if validated is not shipped_params:
    raise AssertionError("validator returned a different mapping object")

  expected_names = set(make_valid_params())
  if set(validated) != expected_names:
    raise AssertionError("shipped params are incomplete: " + repr(validated))

  info = make_instrumented_info(params=load_shipped_params())
  captures = run_instrumented_main(info=info)

  consumed = captures.get("consumed")
  persisted = captures.get("persisted")
  if consumed is None or persisted is None:
    raise AssertionError("entry point did not reach both identity surfaces")
  if consumed is not persisted:
    raise AssertionError("consumed and persisted mappings are different objects")
  if set(consumed) != expected_names:
    raise AssertionError("consumed and persisted mapping is incomplete")
  print("  [PASS] shipped config validates as the same complete mapping")
  print("  [PASS] Cobaya consumption and provenance persist the same object")


def main():
  """Run every pure parameter-schema challenge.

  Arguments:
    None.

  Returns:
    None.
  """
  print("CMB covariance fiducial-parameter witness")
  check_refusals()
  check_main_refusals()
  check_shipped_identity()
  print("CMB covariance fiducial-parameter witness: PASS")


if __name__ == "__main__":
  main()

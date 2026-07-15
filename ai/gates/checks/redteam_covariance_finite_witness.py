#!/usr/bin/env python3
"""Challenge finite CMB covariance arithmetic without running CAMB.

This adversarial-review witness drives the public pure-numpy boundaries in
the covariance producer. It checks that finite configuration values cannot
overflow the derived instrumental noise or its Gaussian covariance products,
and that a non-finite computed array cannot reach the output file.

This is not a board acceptance gate. A passing run means the repaired
producer refuses both known overflow cases, preserves the shipped one-arcmin
arithmetic byte for byte, and catches a mutation that validates inputs only.

PS: Representable means that float64 can store the value without replacing
it by infinity. A just-inside boundary is the nearest smaller float64 value,
found with ``nextafter`` rather than a guessed decimal beam limit.
"""

import math
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from compute_data_vectors import compute_cmb_covariance as covariance


def report(failures, label, passed, detail):
  """Print one result and retain any failed assertion.

  Arguments:
    failures = list that owns the failed result labels.
    label = short description of the scientific assertion.
    passed = whether the assertion succeeded.
    detail = measured value or refusal message.

  Returns:
    None.
  """
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + label + " (" + detail + ")")
  if not passed:
    failures.append(label)


def make_noise_config(beam_fwhm):
  """Build a finite equal-noise fixture at one beam width.

  Arguments:
    beam_fwhm = beam full width at half maximum in arcminutes.

  Returns:
    Noise configuration mapping with all producer keys explicit.
  """
  noise_cfg = {}
  noise_cfg["delta_tt"] = 1.0
  noise_cfg["delta_ee"] = 1.0
  noise_cfg["delta_te"] = 1.0
  noise_cfg["beam_fwhm"] = float(beam_fwhm)
  return noise_cfg


def covariance_product_beam_boundary(lmax, delta_arcmin, fsky):
  """Derive the beam where the largest Gaussian variance reaches float64.

  For equal noise amplitudes and zero signal, the limiting variance obeys

      log(2) - log((2 ell + 1) fsky)
        + 4 log(delta_rad) + 2 x = log(float64_max),

  where ``x = ell(ell+1) theta_rad^2 / (8 log(2))``.

  Arguments:
    lmax = largest requested covariance multipole.
    delta_arcmin = finite positive noise amplitude in microkelvin-arcminutes.
    fsky = observed sky fraction.

  Returns:
    Beam full width at half maximum in arcminutes at the analytic boundary.
  """
  arcmin_to_rad = math.pi / (180.0 * 60.0)
  delta_rad = float(delta_arcmin) * arcmin_to_rad
  log_limit = math.log(np.finfo(np.float64).max)
  mode_count = (2.0 * float(lmax) + 1.0) * float(fsky)
  exponent_limit = 0.5 * (
    log_limit
    - math.log(2.0)
    + math.log(mode_count)
    - 4.0 * math.log(delta_rad)
  )
  ell_factor = float(lmax) * (float(lmax) + 1.0)
  theta_rad = math.sqrt(
    exponent_limit * 8.0 * math.log(2.0) / ell_factor
  )
  return theta_rad / arcmin_to_rad


def legacy_noise_spectrum(ell, delta_arcmin, beam_fwhm_arcmin):
  """Evaluate the shipped noise formula in its original operation order.

  Arguments:
    ell = one-dimensional multipole array.
    delta_arcmin = noise amplitude in microkelvin-arcminutes.
    beam_fwhm_arcmin = beam full width at half maximum in arcminutes.

  Returns:
    Float64 instrumental-noise array.
  """
  arcmin_to_rad = math.pi / (180.0 * 60.0)
  multipoles = np.asarray(ell, dtype="float64")
  delta = float(delta_arcmin) * arcmin_to_rad
  theta = float(beam_fwhm_arcmin) * arcmin_to_rad
  exponent = (
    multipoles
    * (multipoles + 1.0)
    * theta ** 2
    / (8.0 * math.log(2.0))
  )
  return (delta ** 2) * np.exp(exponent)


def legacy_gaussian_blocks(ell, cls, noise, fsky):
  """Evaluate the shipped Gaussian formulas in their original order.

  Arguments:
    ell = one-dimensional multipole array.
    cls = mapping of tt, te, ee, and pp fiducial spectra.
    noise = mapping of tt, te, and ee instrumental-noise spectra.
    fsky = observed sky fraction.

  Returns:
    Mapping of the seven Gaussian variance and cross-spectrum arrays.
  """
  multipoles = np.asarray(ell, dtype="float64")
  norm = 1.0 / ((2.0 * multipoles + 1.0) * float(fsky))
  tt = np.asarray(cls["tt"], dtype="float64") + noise["tt"]
  te = np.asarray(cls["te"], dtype="float64") + noise["te"]
  ee = np.asarray(cls["ee"], dtype="float64") + noise["ee"]
  pp = np.asarray(cls["pp"], dtype="float64")
  blocks = {}
  blocks["var_tt"] = norm * 2.0 * tt * tt
  blocks["var_te"] = norm * (tt * ee + te * te)
  blocks["var_ee"] = norm * 2.0 * ee * ee
  blocks["var_pp"] = norm * 2.0 * pp * pp
  blocks["gauss_tt_te"] = norm * 2.0 * tt * te
  blocks["gauss_tt_ee"] = norm * 2.0 * te * te
  blocks["gauss_te_ee"] = norm * 2.0 * ee * te
  return blocks


def refusal_message(lmax, noise_cfg):
  """Run the beam preflight and return its refusal text.

  Arguments:
    lmax = largest requested covariance multipole.
    noise_cfg = finite noise configuration mapping.

  Returns:
    Empty text when accepted, or the ValueError text when refused.
  """
  try:
    covariance.validate_beam_representability(
      lmax=lmax,
      noise_cfg=noise_cfg,
    )
  except ValueError as error:
    return str(error)
  return ""


def check_overflow_preflights(failures):
  """Check the direct-noise and covariance-product overflow refusals.

  Arguments:
    failures = list that owns failed result labels.

  Returns:
    None.
  """
  lmax = 5000
  ell = np.asarray([lmax], dtype="int64")

  wide_cfg = make_noise_config(beam_fwhm=60.0)
  wide_message = refusal_message(lmax=lmax, noise_cfg=wide_cfg)
  with np.errstate(over="ignore", invalid="ignore"):
    wide_noise = covariance.noise_spectrum(
      ell=ell,
      delta_arcmin=wide_cfg["delta_tt"],
      beam_fwhm_arcmin=wide_cfg["beam_fwhm"],
    )
  wide_passed = wide_message != "" and not np.isfinite(wide_noise[0])
  report(
    failures=failures,
    label="60 arcmin at ell 5000 refuses before covariance work",
    passed=wide_passed,
    detail="message=" + repr(wide_message),
  )

  square_cfg = make_noise_config(beam_fwhm=32.0)
  square_message = refusal_message(lmax=lmax, noise_cfg=square_cfg)
  square_noise = covariance.noise_spectrum(
    ell=ell,
    delta_arcmin=square_cfg["delta_tt"],
    beam_fwhm_arcmin=square_cfg["beam_fwhm"],
  )
  square_cls = {}
  square_cls["tt"] = np.zeros(ell.shape, dtype="float64")
  square_cls["te"] = np.zeros(ell.shape, dtype="float64")
  square_cls["ee"] = np.zeros(ell.shape, dtype="float64")
  square_cls["pp"] = np.zeros(ell.shape, dtype="float64")
  square_noise_by_spectrum = {}
  square_noise_by_spectrum["tt"] = square_noise
  square_noise_by_spectrum["te"] = square_noise
  square_noise_by_spectrum["ee"] = square_noise
  with np.errstate(over="ignore", invalid="ignore"):
    square_blocks = covariance.gaussian_blocks(
      ell=ell,
      cls=square_cls,
      noise=square_noise_by_spectrum,
      fsky=1.0,
    )
  square_product = square_blocks["var_tt"]
  square_passed = (
    square_message != ""
    and np.isfinite(square_noise[0])
    and not np.isfinite(square_product[0])
  )
  square_detail = (
    "noise="
    + repr(float(square_noise[0]))
    + ", product="
    + repr(float(square_product[0]))
    + ", message="
    + repr(square_message)
  )
  report(
    failures=failures,
    label="32 arcmin finite noise refuses its overflowing square",
    passed=square_passed,
    detail=square_detail,
  )


def check_just_inside_boundary(failures):
  """Check the nearest beam below the derived product boundary.

  Arguments:
    failures = list that owns failed result labels.

  Returns:
    None.
  """
  lmax = 5000
  boundary = covariance_product_beam_boundary(
    lmax=lmax,
    delta_arcmin=1.0,
    fsky=1.0,
  )
  inside = np.nextafter(boundary, 0.0)
  noise_cfg = make_noise_config(beam_fwhm=inside)
  message = refusal_message(lmax=lmax, noise_cfg=noise_cfg)
  ell = np.asarray([lmax], dtype="int64")
  noise = covariance.noise_spectrum(
    ell=ell,
    delta_arcmin=1.0,
    beam_fwhm_arcmin=inside,
  )
  cls = {}
  cls["tt"] = np.zeros(ell.shape, dtype="float64")
  cls["te"] = np.zeros(ell.shape, dtype="float64")
  cls["ee"] = np.zeros(ell.shape, dtype="float64")
  cls["pp"] = np.zeros(ell.shape, dtype="float64")
  noise_by_spectrum = {}
  noise_by_spectrum["tt"] = noise
  noise_by_spectrum["te"] = noise
  noise_by_spectrum["ee"] = noise
  with np.errstate(over="ignore", invalid="ignore"):
    blocks = covariance.gaussian_blocks(
      ell=ell,
      cls=cls,
      noise=noise_by_spectrum,
      fsky=1.0,
    )
  product = blocks["var_tt"]
  passed = message == "" and np.isfinite(noise[0]) and np.isfinite(product[0])
  detail = (
    "boundary="
    + repr(boundary)
    + ", inside="
    + repr(float(inside))
    + ", product="
    + repr(float(product[0]))
  )
  report(
    failures=failures,
    label="nextafter beam inside the derived boundary stays finite",
    passed=passed,
    detail=detail,
  )


def check_shipped_arithmetic(failures):
  """Check that the one-arcmin producer arithmetic remains byte-identical.

  Arguments:
    failures = list that owns failed result labels.

  Returns:
    None.
  """
  ell = np.arange(2, 5001, dtype="int64")
  noise_cfg = {}
  noise_cfg["delta_tt"] = 1.0
  noise_cfg["delta_ee"] = 1.4
  noise_cfg["delta_te"] = 0.0
  noise_cfg["beam_fwhm"] = 1.0
  message = refusal_message(lmax=5000, noise_cfg=noise_cfg)

  noise = {}
  reference_noise = {}
  for spectrum in ("tt", "te", "ee"):
    delta_key = "delta_" + spectrum
    noise[spectrum] = covariance.noise_spectrum(
      ell=ell,
      delta_arcmin=noise_cfg[delta_key],
      beam_fwhm_arcmin=noise_cfg["beam_fwhm"],
    )
    reference_noise[spectrum] = legacy_noise_spectrum(
      ell=ell,
      delta_arcmin=noise_cfg[delta_key],
      beam_fwhm_arcmin=noise_cfg["beam_fwhm"],
    )

  cls = {}
  cls["tt"] = np.full(ell.shape, 2.0e-4, dtype="float64")
  cls["te"] = np.full(ell.shape, 3.0e-5, dtype="float64")
  cls["ee"] = np.full(ell.shape, 8.0e-5, dtype="float64")
  cls["pp"] = np.full(ell.shape, 4.0e-9, dtype="float64")
  blocks = covariance.gaussian_blocks(
    ell=ell,
    cls=cls,
    noise=noise,
    fsky=1.0,
  )
  reference_blocks = legacy_gaussian_blocks(
    ell=ell,
    cls=cls,
    noise=reference_noise,
    fsky=1.0,
  )

  mismatches = []
  for key in noise:
    if noise[key].tobytes() != reference_noise[key].tobytes():
      mismatches.append("noise." + key)
  for key in blocks:
    if blocks[key].tobytes() != reference_blocks[key].tobytes():
      mismatches.append(key)
  passed = message == "" and mismatches == []
  report(
    failures=failures,
    label="shipped one-arcmin noise and Gaussian bytes are unchanged",
    passed=passed,
    detail="mismatches=" + repr(mismatches),
  )


def check_postcompute_finiteness(failures):
  """Check the first non-finite output is named before publication.

  Arguments:
    failures = list that owns failed result labels.

  Returns:
    None.
  """
  finite_arrays = {}
  finite_arrays["ell"] = np.asarray([2, 3], dtype="int64")
  finite_arrays["sigma_tt"] = np.asarray([1.0, 2.0], dtype="float64")
  covariance.require_finite_arrays(arrays=finite_arrays)

  bad_arrays = {}
  bad_arrays["ell"] = np.asarray([2, 3], dtype="int64")
  bad_arrays["sigma_tt"] = np.asarray([1.0, 2.0], dtype="float64")
  bad_arrays["sigma_te"] = np.asarray([1.0, np.inf], dtype="float64")
  bad_arrays["sigma_ee"] = np.asarray([np.nan, 2.0], dtype="float64")
  message = ""
  try:
    covariance.require_finite_arrays(arrays=bad_arrays)
  except ValueError as error:
    message = str(error)
  names_first = (
    "sigma_te" in message
    and "index" in message
    and "1" in message
    and "inf" in message.lower()
    and "sigma_ee" not in message
  )
  report(
    failures=failures,
    label="postcompute refusal names the first key, ell index, and value",
    passed=names_first,
    detail="message=" + repr(message),
  )


def mutation_inputs_only_reaches_save(path):
  """Restore input-only checks and save the resulting infinite noise.

  Arguments:
    path = destination for the deliberately invalid numpy archive.

  Returns:
    The saved noise array read back from the archive.
  """
  lmax = 5000
  delta_arcmin = 1.0
  beam_fwhm = 60.0
  for value in (float(lmax), delta_arcmin, beam_fwhm):
    if not np.isfinite(value):
      raise ValueError("mutation fixture inputs must be finite")
  if delta_arcmin < 0.0 or beam_fwhm <= 0.0:
    raise ValueError("mutation fixture inputs have invalid signs")
  ell = np.asarray([lmax], dtype="int64")
  with np.errstate(over="ignore", invalid="ignore"):
    noise = covariance.noise_spectrum(
      ell=ell,
      delta_arcmin=delta_arcmin,
      beam_fwhm_arcmin=beam_fwhm,
    )
  np.savez(path, ell=ell, sigma_tt=noise)
  with np.load(path) as archive:
    return archive["sigma_tt"].copy()


def check_input_only_mutation(failures):
  """Check that input-only validation publishes an infinite member.

  Arguments:
    failures = list that owns failed result labels.

  Returns:
    None.
  """
  with tempfile.TemporaryDirectory(prefix="covariance-finite-") as directory:
    path = Path(directory) / "mutation.npz"
    saved = mutation_inputs_only_reaches_save(path=path)
    passed = path.is_file() and not np.isfinite(saved[0])
    detail = "saved sigma_tt=" + repr(float(saved[0]))
  report(
    failures=failures,
    label="input-only mutation reaches savez with infinite data",
    passed=passed,
    detail=detail,
  )


def main():
  """Run every pure-CPU witness leg and return one process verdict.

  Arguments:
    None.

  Returns:
    Zero when every leg passes, otherwise one.
  """
  failures = []
  print("covariance finite-arithmetic adversarial witness")
  check_overflow_preflights(failures=failures)
  check_just_inside_boundary(failures=failures)
  check_shipped_arithmetic(failures=failures)
  check_postcompute_finiteness(failures=failures)
  check_input_only_mutation(failures=failures)
  if failures:
    print("FAIL: " + str(len(failures)) + " witness leg(s): " + repr(failures))
    return 1
  print("PASS: all covariance finite-arithmetic witness legs passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

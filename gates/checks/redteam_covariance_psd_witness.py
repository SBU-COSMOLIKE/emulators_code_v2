#!/usr/bin/env python3
"""Challenge the physical validity checks for the joint T/E covariance.

This pure-CPU witness exercises the public validation helpers and the real
Gaussian covariance assembly. It checks the input-amplitude boundary, each
signal-plus-noise matrix, and a tiled covariance matrix before publication.
The last leg deliberately restores the old individual-nonnegativity check and
must recover the documented negative eigenvalue.

PS: PSD means positive semidefinite. A PSD covariance has no negative
eigenvalues, apart from a declared rounding allowance.
"""

import math
from pathlib import Path
import sys

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from compute_data_vectors import compute_cmb_covariance as covariance_module


def _expect_value_error(function, label, **kwargs):
  """Require one validation call to refuse its input.

  Arguments:
    function = validation function under test.
    label    = plain-language name printed for this check.
    **kwargs = named arguments forwarded to ``function``.

  Returns:
    None.
  """
  try:
    function(**kwargs)
  except ValueError as error:
    print("  [PASS] " + label + " (" + str(error) + ")")
    return
  raise AssertionError(label + " was accepted")


def _expect_pass(function, label, **kwargs):
  """Require one validation call to accept its input without mutation.

  Arguments:
    function = validation function under test.
    label    = plain-language name printed for this check.
    **kwargs = named arguments forwarded to ``function``.

  Returns:
    None.
  """
  result = function(**kwargs)
  if result is not None:
    raise AssertionError(label + " returned " + repr(result) + ", not None")
  print("  [PASS] " + label)


def _gaussian_matrix(blocks, ell_index):
  """Assemble the joint TT, TE, and EE covariance at one multipole.

  Arguments:
    blocks    = mapping returned by ``gaussian_blocks``.
    ell_index = index of the multipole to assemble.

  Returns:
    Symmetric array with shape ``(3, 3)`` in TT, TE, EE order.
  """
  matrix = np.empty(shape=(3, 3), dtype="float64")
  matrix[0, 0] = blocks["var_tt"][ell_index]
  matrix[0, 1] = blocks["gauss_tt_te"][ell_index]
  matrix[0, 2] = blocks["gauss_tt_ee"][ell_index]
  matrix[1, 0] = matrix[0, 1]
  matrix[1, 1] = blocks["var_te"][ell_index]
  matrix[1, 2] = blocks["gauss_te_ee"][ell_index]
  matrix[2, 0] = matrix[0, 2]
  matrix[2, 1] = matrix[1, 2]
  matrix[2, 2] = blocks["var_ee"][ell_index]
  return matrix


def _tiled_gaussian_matrix(blocks, n_ell):
  """Tile independent per-multipole blocks into one joint covariance.

  Shape flow::

    seven arrays (L,) -> L blocks (3, 3) -> tiled matrix (3L, 3L)

  (legend: L = number of multipoles; 3 = TT, TE, and EE spectra.)

  Arguments:
    blocks = mapping returned by ``gaussian_blocks``.
    n_ell  = number of multipoles represented by each mapping value.

  Returns:
    Symmetric array with shape ``(3 * n_ell, 3 * n_ell)``.
  """
  matrix = np.zeros(shape=(3 * n_ell, 3 * n_ell), dtype="float64")
  for ell_index in range(n_ell):
    local = _gaussian_matrix(
      blocks=blocks,
      ell_index=ell_index)
    start = 3 * ell_index
    stop = start + 3
    matrix[start:stop, start:stop] = local
  return matrix


def _realistic_fixture(noise_spectrum, gaussian_blocks):
  """Build a finite signal-plus-noise control over representative multipoles.

  Arguments:
    noise_spectrum = production instrumental-noise function.
    gaussian_blocks = production Gaussian covariance function.

  Returns:
    Tuple containing the signal mapping, noise mapping, and tiled covariance.
  """
  ell = np.array(object=(2, 30, 100, 500), dtype="int64")
  cls = {}
  cls["tt"] = np.array(object=(1100.0, 90.0, 1.8, 0.06), dtype="float64")
  cls["te"] = np.array(object=(35.0, 4.0, 0.14, 0.006), dtype="float64")
  cls["ee"] = np.array(object=(18.0, 2.0, 0.10, 0.012), dtype="float64")
  cls["pp"] = np.zeros(shape=ell.shape, dtype="float64")

  noise = {}
  noise["tt"] = noise_spectrum(
    ell=ell,
    delta_arcmin=1.0,
    beam_fwhm_arcmin=1.0)
  noise["te"] = noise_spectrum(
    ell=ell,
    delta_arcmin=0.0,
    beam_fwhm_arcmin=1.0)
  noise["ee"] = noise_spectrum(
    ell=ell,
    delta_arcmin=1.4,
    beam_fwhm_arcmin=1.0)
  blocks = gaussian_blocks(
    ell=ell,
    cls=cls,
    noise=noise,
    fsky=1.0)
  covariance = _tiled_gaussian_matrix(
    blocks=blocks,
    n_ell=ell.size)
  return cls, noise, covariance


def _check_input_boundary(validate_noise_psd):
  """Check refusal, equality, rounding boundary, and the shipped control.

  Arguments:
    validate_noise_psd = public noise-amplitude PSD validator.

  Returns:
    None.
  """
  _expect_value_error(
    function=validate_noise_psd,
    label="1/1/10 amplitudes refuse before CAMB",
    delta_tt=1.0,
    delta_te=10.0,
    delta_ee=1.0)
  _expect_pass(
    function=validate_noise_psd,
    label="exact PSD equality passes",
    delta_tt=1.0,
    delta_te=1.0,
    delta_ee=1.0)

  just_over = np.nextafter(1.0, math.inf)
  _expect_value_error(
    function=validate_noise_psd,
    label="first representable amplitude above equality refuses",
    delta_tt=1.0,
    delta_te=just_over,
    delta_ee=1.0)

  shipped = np.array(object=(1.0, 0.0, 1.4), dtype="float64")
  before = shipped.tobytes()
  _expect_pass(
    function=validate_noise_psd,
    label="shipped uncorrelated-noise control passes",
    delta_tt=shipped[0],
    delta_te=shipped[1],
    delta_ee=shipped[2])
  after = shipped.tobytes()
  if after != before:
    raise AssertionError("the shipped delta_te=0 control was mutated")
  print("  [PASS] shipped delta_te=0 inputs remain byte-identical")


def _check_post_signal(
    validate_signal_noise_psd,
    validate_covariance_psd,
    noise_spectrum,
    gaussian_blocks):
  """Check realistic and deliberately invalid post-signal matrices.

  Arguments:
    validate_signal_noise_psd = per-multipole signal-plus-noise validator.
    validate_covariance_psd   = assembled symmetric-matrix validator.
    noise_spectrum            = production instrumental-noise function.
    gaussian_blocks           = production Gaussian covariance function.

  Returns:
    None.
  """
  cls, noise, covariance = _realistic_fixture(
    noise_spectrum=noise_spectrum,
    gaussian_blocks=gaussian_blocks)
  _expect_pass(
    function=validate_signal_noise_psd,
    label="realistic signal-plus-noise matrices pass at every ell",
    cls=cls,
    noise=noise)
  _expect_pass(
    function=validate_covariance_psd,
    label="realistic tiled covariance passes",
    covariance=covariance,
    name="realistic tiled covariance")

  invalid_cls = {}
  invalid_cls["tt"] = np.array(object=(1.0,), dtype="float64")
  invalid_cls["te"] = np.array(object=(2.0,), dtype="float64")
  invalid_cls["ee"] = np.array(object=(1.0,), dtype="float64")
  invalid_noise = {}
  invalid_noise["tt"] = np.zeros(shape=(1,), dtype="float64")
  invalid_noise["te"] = np.zeros(shape=(1,), dtype="float64")
  invalid_noise["ee"] = np.zeros(shape=(1,), dtype="float64")
  _expect_value_error(
    function=validate_signal_noise_psd,
    label="constructed post-signal violation refuses",
    cls=invalid_cls,
    noise=invalid_noise)


def _individual_nonnegative_only(delta_tt, delta_te, delta_ee):
  """Model the old mutation that checks three amplitudes separately.

  Arguments:
    delta_tt = temperature auto-noise amplitude.
    delta_te = temperature-polarization cross-noise amplitude.
    delta_ee = polarization auto-noise amplitude.

  Returns:
    True when all individual amplitudes are finite and nonnegative.
  """
  values = (delta_tt, delta_te, delta_ee)
  for value in values:
    if not math.isfinite(value) or value < 0.0:
      return False
  return True


def _check_nonnegative_mutation(noise_spectrum, gaussian_blocks):
  """Require the individual-nonnegativity mutation to publish indefiniteness.

  Arguments:
    noise_spectrum = production instrumental-noise function.
    gaussian_blocks = production Gaussian covariance function.

  Returns:
    None.
  """
  if not _individual_nonnegative_only(
      delta_tt=1.0,
      delta_te=10.0,
      delta_ee=1.0):
    raise AssertionError("the intended individual-check mutation did not pass")

  ell = np.array(object=(2,), dtype="int64")
  cls = {}
  cls["tt"] = np.zeros(shape=(1,), dtype="float64")
  cls["te"] = np.zeros(shape=(1,), dtype="float64")
  cls["ee"] = np.zeros(shape=(1,), dtype="float64")
  cls["pp"] = np.zeros(shape=(1,), dtype="float64")
  noise = {}
  noise["tt"] = noise_spectrum(
    ell=ell,
    delta_arcmin=1.0,
    beam_fwhm_arcmin=1.0)
  noise["te"] = noise_spectrum(
    ell=ell,
    delta_arcmin=10.0,
    beam_fwhm_arcmin=1.0)
  noise["ee"] = noise_spectrum(
    ell=ell,
    delta_arcmin=1.0,
    beam_fwhm_arcmin=1.0)
  blocks = gaussian_blocks(
    ell=ell,
    cls=cls,
    noise=noise,
    fsky=1.0)
  covariance = _gaussian_matrix(
    blocks=blocks,
    ell_index=0)
  eigenvalues = np.linalg.eigvalsh(covariance)
  expected = np.array(
    object=(-2.86365772e-11, 1.43097071e-11, 2.86537506e-11),
    dtype="float64")
  if not np.allclose(eigenvalues, expected, rtol=5.0e-9, atol=1.0e-18):
    raise AssertionError(
      "mutation eigenvalues changed: " + repr(eigenvalues))
  sigmas = np.sqrt(
    np.array(
      object=(blocks["var_tt"][0],
              blocks["var_te"][0],
              blocks["var_ee"][0]),
      dtype="float64"))
  if not np.all(np.isfinite(sigmas)):
    raise AssertionError("mutation did not reach finite published sigmas")
  print(
    "  [PASS] individual-only mutation reaches negative eigenvalue "
    + repr(eigenvalues))


def main():
  """Run every pure-CPU physical-covariance witness.

  Arguments:
    None.

  Returns:
    None.
  """
  validate_noise_psd = covariance_module.validate_noise_psd
  validate_signal_noise_psd = covariance_module.validate_signal_noise_psd
  validate_covariance_psd = covariance_module.validate_covariance_psd
  noise_spectrum = covariance_module.noise_spectrum
  gaussian_blocks = covariance_module.gaussian_blocks

  print("joint T/E covariance physical-validity witness")
  _check_input_boundary(
    validate_noise_psd=validate_noise_psd)
  _check_post_signal(
    validate_signal_noise_psd=validate_signal_noise_psd,
    validate_covariance_psd=validate_covariance_psd,
    noise_spectrum=noise_spectrum,
    gaussian_blocks=gaussian_blocks)
  _check_nonnegative_mutation(
    noise_spectrum=noise_spectrum,
    gaussian_blocks=gaussian_blocks)
  print("PASS: every covariance PSD witness discriminated")


if __name__ == "__main__":
  main()

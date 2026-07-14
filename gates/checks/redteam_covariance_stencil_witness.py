#!/usr/bin/env python3
"""Challenge the covariance stencil's float64 perturbation boundary.

The lensing-potential stencil multiplies each physical band by four factors.
This witness checks that all four factors are distinct and ordered in float64,
and that a nonzero physical band changes on both sides of the fiducial value.
An exactly zero physical band remains legal because multiplication cannot move
zero away from zero.

The checks use small NumPy fixtures and a fake re-lensing object.  No CAMB,
Cobaya, Torch, network, or accelerator is needed.

PS: A stencil estimates a derivative from values on both sides of a central
value.  A representable factor is a float64 number that differs from its
neighbor after rounding to the finite set of numbers float64 can store.
"""

import math
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from compute_data_vectors import compute_cmb_covariance as covariance


class Results:
  """Collect witness verdicts and print one readable line per check."""

  def __init__(self):
    """Create an empty result collection.

    Returns:
      A result collection with no recorded failures.
    """
    self.failures = []

  def report(self, label, ok, detail):
    """Print one verdict and retain its label when it fails.

    Arguments:
      label  = plain-language name of the behavior under test.
      ok     = True when the observed behavior matches the requirement.
      detail = short description of the observed value.

    Returns:
      None.
    """
    mark = "PASS" if ok else "FAIL"
    print("  [" + mark + "] " + label + "  (" + detail + ")")
    if not ok:
      self.failures.append(label)


class FakeCambData:
  """Supply deterministic lensing responses and count every CAMB-like call."""

  def __init__(self, clpp_fid):
    """Store one fiducial scaled lensing-potential spectrum.

    Arguments:
      clpp_fid = one-dimensional float64 fiducial spectrum.

    Returns:
      A fake re-lensing object with zero calls recorded.
    """
    self.clpp_fid = np.asarray(clpp_fid, dtype="float64")
    self.potential_calls = 0
    self.relensing_calls = 0

  def get_lens_potential_cls(self, raw_cl):
    """Return the stored spectrum in CAMB's expected two-dimensional shape.

    Arguments:
      raw_cl = False when the caller requests CAMB's scaled convention.

    Returns:
      A two-dimensional array whose first column is the stored spectrum.
    """
    self.potential_calls += 1
    if raw_cl is not False:
      raise AssertionError("the covariance path requested raw lensing power")
    answer = np.zeros((len(self.clpp_fid), 1), dtype="float64")
    answer[:, 0] = self.clpp_fid
    return answer

  def get_lensed_cls_with_spectrum(self, clpp, lmax, CMB_unit, raw_cl):
    """Return a response linear in the perturbed lensing-potential sum.

    Arguments:
      clpp    = scaled lensing-potential spectrum supplied for re-lensing.
      lmax    = largest requested CMB multipole.
      CMB_unit = requested temperature unit, which must be ``muK``.
      raw_cl  = True when raw CMB power spectra are requested.

    Returns:
      A ``(lmax + 1, 4)`` array with deterministic TT, EE, and TE columns.
    """
    self.relensing_calls += 1
    if CMB_unit != "muK" or raw_cl is not True:
      raise AssertionError("the covariance path changed its CMB convention")
    delta = np.asarray(clpp, dtype="float64") - self.clpp_fid
    amplitude = math.fsum(delta.tolist())
    answer = np.zeros((int(lmax) + 1, 4), dtype="float64")
    ell = np.arange(int(lmax) + 1, dtype="float64")
    answer[:, 0] = amplitude * (ell + 1.0)
    answer[:, 1] = amplitude * (ell + 2.0)
    answer[:, 3] = amplitude * (ell + 3.0)
    return answer


def nextafter_step_boundary():
  """Derive the first step above the float64 tie for two positive arms.

  Returns:
    The first float64 step above the tie where the ``+1`` and ``+2``
    stencil factors would otherwise round to the same value.
  """
  next_up = np.nextafter(np.float64(1.0), np.float64(np.inf))
  upper_spacing = next_up - np.float64(1.0)
  two_arm_tie = np.float64(3.0) * upper_spacing / np.float64(4.0)
  return two_arm_tie


def require_refusal(callback, message_part):
  """Run one call and require a ValueError that names the failed rule.

  Arguments:
    callback     = zero-argument callable that must refuse its input.
    message_part = text that must occur in the ValueError message.

  Returns:
    A pair containing the verdict and the observed detail.
  """
  try:
    callback()
  except ValueError as error:
    text = str(error)
    return message_part in text, text
  except Exception as error:
    return False, type(error).__name__ + ": " + str(error)
  return False, "call returned without refusing the input"


def run_factor_checks(results):
  """Check exact factor ordering, the boundary, and shipped step values.

  Arguments:
    results = collection that receives each factor-check verdict.

  Returns:
    None.
  """
  shipped_steps = (0.01, 0.02, 0.04)
  for step_frac in shipped_steps:
    factors = covariance.stencil_factors(step_frac=step_frac)
    expected = (
      np.float64(1.0) - np.float64(2.0) * np.float64(step_frac),
      np.float64(1.0) - np.float64(step_frac),
      np.float64(1.0) + np.float64(step_frac),
      np.float64(1.0) + np.float64(2.0) * np.float64(step_frac),
    )
    ordered = factors[0] < factors[1] < 1.0 < factors[2] < factors[3]
    unchanged = np.array_equal(
      np.asarray(factors, dtype="float64"),
      np.asarray(expected, dtype="float64"))
    results.report(
      "shipped step has ordered factors and unchanged arithmetic",
      ordered and unchanged,
      "step_frac=" + repr(step_frac) + ", factors=" + repr(factors))

  boundary = nextafter_step_boundary()
  below = np.nextafter(boundary, np.float64(0.0))
  factors = covariance.stencil_factors(step_frac=boundary)
  ordered = factors[0] < factors[1] < 1.0 < factors[2] < factors[3]
  results.report(
    "nextafter-derived boundary is accepted and ordered",
    ordered,
    "boundary=" + repr(float(boundary)) + ", factors=" + repr(factors))

  def below_boundary_call():
    """Request factors at the round-to-even tie below the boundary.

    Returns:
      The factor tuple when production incorrectly accepts the step.
    """
    return covariance.stencil_factors(step_frac=below)

  refused, detail = require_refusal(
    callback=below_boundary_call,
    message_part="representable")
  results.report(
    "one float64 step below the boundary is refused",
    refused,
    detail)


def run_band_checks(results):
  """Check changed-value counts for nonzero and physical-zero bands.

  Arguments:
    results = collection that receives each band-check verdict.

  Returns:
    None.
  """
  fiducial = np.asarray([0.0, 0.0, 2.0, 4.0, 0.0], dtype="float64")
  factors = covariance.stencil_factors(step_frac=0.01)
  counts = []
  for factor in factors:
    perturbed, changed_count = covariance.scaled_lensing_band(
      clpp_fid=fiducial,
      band_lo=2,
      band_hi=3,
      factor=factor)
    counts.append(changed_count)
    outside_same = np.array_equal(perturbed[:2], fiducial[:2])
    outside_same = outside_same and np.array_equal(
      perturbed[4:],
      fiducial[4:])
    results.report(
      "nonzero band changes only inside the selected band",
      changed_count == 2 and outside_same,
      "factor=" + repr(factor) + ", changed=" + str(changed_count))

  both_signs = counts[0] > 0 and counts[1] > 0
  both_signs = both_signs and counts[2] > 0 and counts[3] > 0
  results.report(
    "nonzero band changes on both stencil signs",
    both_signs,
    "changed counts=" + repr(counts))

  zero_fiducial = np.zeros(5, dtype="float64")
  zero_counts = []
  for factor in factors:
    perturbed, changed_count = covariance.scaled_lensing_band(
      clpp_fid=zero_fiducial,
      band_lo=2,
      band_hi=3,
      factor=factor)
    zero_counts.append(changed_count)
    if not np.array_equal(perturbed, zero_fiducial):
      changed_count = -1
  results.report(
    "an exactly zero physical band remains legal",
    zero_counts == [0, 0, 0, 0],
    "changed counts=" + repr(zero_counts))


def covariance_fixture(step_fracs):
  """Build the small inputs used to execute the production covariance path.

  Arguments:
    step_fracs = list of fractional stencil steps.

  Returns:
    A tuple containing fake CAMB data, spectra, multipoles, and configuration.
  """
  clpp_fid = np.asarray([0.0, 0.0, 3.0, 0.0], dtype="float64")
  cambdata = FakeCambData(clpp_fid=clpp_fid)
  spectra = {
    "tt": np.zeros(4, dtype="float64"),
    "te": np.zeros(4, dtype="float64"),
    "ee": np.zeros(4, dtype="float64"),
    "pp": np.asarray([0.0, 0.0, 2.0, 0.0], dtype="float64"),
  }
  ell = np.asarray([2, 3], dtype="int64")
  config = {
    "lens_lmax": 2,
    "band_width": 1,
    "step_fracs": list(step_fracs),
    "converge_rtol": 1.0e-8,
  }
  return cambdata, spectra, ell, config


def run_production_path(cambdata, spectra, ell, config):
  """Execute the production non-Gaussian covariance body on one fixture.

  Arguments:
    cambdata = fake re-lensing object.
    spectra  = fiducial raw power-spectrum mapping.
    ell      = covariance multipole grid.
    config   = non-Gaussian covariance configuration.

  Returns:
    The production ``(blocks, study)`` result.
  """
  def discard_log(message):
    """Accept a production progress message without printing it.

    Arguments:
      message = progress text emitted by the covariance calculation.

    Returns:
      None.
    """
    del message

  return covariance.nongaussian_blocks(
    cambdata=cambdata,
    cls=spectra,
    ell=ell,
    ng_cfg=config,
    fsky=1.0,
    log=discard_log)


def run_false_green_refusal(results):
  """Require tiny no-op steps to fail before any CAMB-like call.

  Arguments:
    results = collection that receives the refusal verdict.

  Returns:
    None.
  """
  cambdata, spectra, ell, config = covariance_fixture(
    step_fracs=[1.0e-20, 2.0e-20])

  def tiny_step_call():
    """Execute the production path with two unrepresentable steps.

    Returns:
      The production covariance result when validation fails to refuse.
    """
    return run_production_path(
      cambdata=cambdata,
      spectra=spectra,
      ell=ell,
      config=config)

  refused, detail = require_refusal(
    callback=tiny_step_call,
    message_part="representable")
  before_camb = cambdata.potential_calls == 0
  before_camb = before_camb and cambdata.relensing_calls == 0
  results.report(
    "tiny false-green steps refuse before re-lensing",
    refused and before_camb,
    detail + "; calls=" + repr(
      (cambdata.potential_calls, cambdata.relensing_calls)))


def run_provenance_checks(results):
  """Require the production study to persist factors and change counts.

  Arguments:
    results = collection that receives the provenance verdicts.

  Returns:
    None.
  """
  cambdata, spectra, ell, config = covariance_fixture(
    step_fracs=[0.01, 0.02])
  blocks, study = run_production_path(
    cambdata=cambdata,
    spectra=spectra,
    ell=ell,
    config=config)
  del blocks

  expected_factors = []
  for step_frac in config["step_fracs"]:
    factors = covariance.stencil_factors(step_frac=step_frac)
    expected_factors.append(list(factors))
  factors_match = study["factor_lists"] == expected_factors
  results.report(
    "production study persists the exact float64 factor lists",
    factors_match,
    "factor lists=" + repr(study["factor_lists"]))

  counts = study["per_band_changed_value_counts"]
  counts_match = len(counts) == 1 and len(counts[0]) == 2
  if counts_match:
    for step_counts in counts[0]:
      if len(step_counts) != 4:
        counts_match = False
        continue
      for changed_count in step_counts:
        if changed_count <= 0:
          counts_match = False
  results.report(
    "production study persists changes on both signs",
    counts_match,
    "changed counts=" + repr(counts))


def run_mutation_arm(results):
  """Show that positive-only validation returns zero blocks and is caught.

  Arguments:
    results = collection that receives the mutation verdict.

  Returns:
    None.
  """
  def positive_only_factors(step_frac):
    """Model the defective rule that checks positivity but not rounding.

    Arguments:
      step_frac = positive fractional stencil step.

    Returns:
      Four rounded float64 factors without an ordering check.
    """
    value = np.float64(step_frac)
    if not np.isfinite(value) or value <= 0.0:
      raise ValueError("step fraction must be finite and positive")
    return (
      np.float64(1.0) - np.float64(2.0) * value,
      np.float64(1.0) - value,
      np.float64(1.0) + value,
      np.float64(1.0) + np.float64(2.0) * value,
    )

  def positive_only_scaled_band(clpp_fid, band_lo, band_hi, factor):
    """Model scaling that records changes without requiring any.

    Arguments:
      clpp_fid = full fiducial lensing-potential array.
      band_lo  = inclusive first band index.
      band_hi  = inclusive final band index.
      factor   = positive stencil multiplier.

    Returns:
      A pair containing the scaled copy and its changed-value count.
    """
    scaled = np.asarray(clpp_fid, dtype="float64").copy()
    before = scaled[band_lo:band_hi + 1].copy()
    scaled[band_lo:band_hi + 1] *= np.float64(factor)
    after = scaled[band_lo:band_hi + 1]
    changed_count = int(np.count_nonzero(after != before))
    return scaled, changed_count

  clpp_fid = np.asarray([0.0, 0.0, 3.0, 0.0], dtype="float64")
  cambdata = FakeCambData(clpp_fid=clpp_fid)
  factors = positive_only_factors(step_frac=1.0e-20)
  lensed = []
  changed_counts = []
  for factor in factors:
    perturbed, changed_count = positive_only_scaled_band(
      clpp_fid=clpp_fid,
      band_lo=2,
      band_hi=2,
      factor=factor)
    changed_counts.append(changed_count)
    response = cambdata.get_lensed_cls_with_spectrum(
      clpp=perturbed,
      lmax=3,
      CMB_unit="muK",
      raw_cl=True)
    lensed.append(response)

  derivative = {}
  columns = {
    "tt": 0,
    "te": 3,
    "ee": 1,
  }
  for spectrum, column in columns.items():
    values = []
    for response in lensed:
      values.append(response[2:4, column])
    estimate = covariance.stencil_derivative(
      f_m2=values[0],
      f_m1=values[1],
      f_p1=values[2],
      f_p2=values[3],
      step_frac=1.0e-20)
    derivative[spectrum] = estimate.reshape(1, 2)
  blocks = covariance.assemble_lensing_blocks(
    deriv=derivative,
    w=np.asarray([0.4], dtype="float64"))

  every_zero = True
  for block in blocks.values():
    if np.count_nonzero(block) != 0:
      every_zero = False
  mutation_rejected = every_zero and changed_counts == [0, 0, 0, 0]
  results.report(
    "both-sign count rule catches the positive-only mutation",
    mutation_rejected,
    "zero blocks=" + repr(every_zero)
    + ", changed counts=" + repr(changed_counts))


def run():
  """Execute every pure-CPU stencil witness.

  Returns:
    Process status 0 when every witness passes, otherwise 1.
  """
  results = Results()
  print("covariance stencil float64 witness")
  run_factor_checks(results=results)
  run_band_checks(results=results)
  run_false_green_refusal(results=results)
  run_provenance_checks(results=results)
  run_mutation_arm(results=results)
  if results.failures:
    print("FAIL: " + repr(results.failures))
    return 1
  print("PASS: covariance stencil rejects no-op perturbations")
  return 0


if __name__ == "__main__":
  raise SystemExit(run())

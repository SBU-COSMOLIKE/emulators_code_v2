#!/usr/bin/env python3
"""Challenge the covariance configuration boundary with malformed inputs.

The covariance calculation is expensive, so every malformed experiment must
stop at its pure configuration boundary. This witness also checks that missing
lensing-power ranges stop before any relensing call.
"""

from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from compute_data_vectors import compute_cmb_covariance as covariance


FAILURES = []


def report(label, passed, detail):
  """Print one result and retain any failure.

  Arguments:
    label  = plain-language acceptance claim.
    passed = whether the claim held.
    detail = measured value or refusal message.

  Returns:
    None.
  """
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + label + " (" + detail + ")")
  if not passed:
    FAILURES.append(label)


def valid_covariance_args():
  """Build one complete shipped-value covariance configuration.

  Returns:
    a fresh nested mapping accepted by the producer boundary.
  """
  return {
    "lmax": 5000,
    "fsky": 1.0,
    "noise": {
      "delta_tt": 1.0,
      "delta_ee": 1.4,
      "delta_te": 0.0,
      "beam_fwhm": 1.0,
    },
    "nongaussian": {
      "enabled": False,
      "lens_lmax": 3000,
      "band_width": 20,
      "step_fracs": [0.01, 0.02, 0.04],
      "converge_rtol": 0.05,
    },
  }


def expect_refusal(label, mutate, needle):
  """Apply one invalid change and require the named schema refusal.

  Arguments:
    label  = acceptance claim.
    mutate = callable that changes one fresh valid mapping.
    needle = diagnostic substring owned by the tested rule.

  Returns:
    None.
  """
  config = valid_covariance_args()
  mutate(config)
  try:
    covariance.validate_cov_args(config)
  except ValueError as error:
    text = str(error)
    report(label, needle in text, text)
    return
  report(label, False, "invalid configuration was accepted")


def replace_nested(config, path, value):
  """Replace one nested configuration value by an explicit key path.

  Arguments:
    config = nested covariance configuration mapping.
    path   = tuple of keys ending at the value to replace.
    value  = replacement used by one refusal arm.

  Returns:
    None.
  """
  parent = config
  for key in path[:-1]:
    parent = parent[key]
  parent[path[-1]] = value


def expect_value_refusal(label, path, value, needle):
  """Require refusal after replacing one nested scalar or list.

  Arguments:
    label  = acceptance claim.
    path   = tuple locating the replaced configuration value.
    value  = invalid replacement.
    needle = diagnostic substring owned by the tested rule.

  Returns:
    None.
  """
  config = valid_covariance_args()
  replace_nested(config=config, path=path, value=value)
  try:
    covariance.validate_cov_args(config)
  except ValueError as error:
    text = str(error)
    report(label, needle in text, text)
    return
  report(label, False, "invalid configuration was accepted")


def check_scalar_ranges():
  """Check scalar types, ranges, and explicit Boolean semantics.

  Returns:
    None.
  """
  cases = []
  cases.append(("zero sky fraction refuses", ("fsky",), 0.0,
                "0 < fsky"))
  cases.append(("non-finite sky fraction refuses", ("fsky",), np.nan,
                "finite"))
  cases.append(("negative noise refuses", ("noise", "delta_tt"), -1.0,
                "must be >= 0"))
  cases.append(("non-finite noise refuses", ("noise", "delta_ee"),
                np.inf, "must be finite"))
  cases.append(("zero beam refuses", ("noise", "beam_fwhm"), 0.0,
                "must be > 0"))
  cases.append(("quoted false refuses", ("nongaussian", "enabled"),
                "false", "real boolean"))
  cases.append(("zero band width refuses",
                ("nongaussian", "band_width"), 0, "integer >= 1"))
  cases.append(("boolean lmax refuses", ("lmax",), True,
                "non-boolean integer"))
  for label, path, value, needle in cases:
    expect_value_refusal(label=label,
                         path=path,
                         value=value,
                         needle=needle)


def check_step_ranges():
  """Check the complete ordering and finiteness contract for steps.

  Returns:
    None.
  """
  values = []
  values.append(("zero step", [0.0, 0.02]))
  values.append(("duplicate steps", [0.01, 0.01]))
  values.append(("unordered steps", [0.02, 0.01]))
  values.append(("NaN step", [0.01, np.nan]))
  values.append(("infinite step", [0.01, np.inf]))
  for label, step_fracs in values:
    expect_value_refusal(label=label + " refuses",
                         path=("nongaussian", "step_fracs"),
                         value=step_fracs,
                         needle="step_fracs")


def check_exact_keys():
  """Check unknown and missing keys at every nested schema level.

  Returns:
    None.
  """
  def extra_top(config):
    """Add one mistyped top-level covariance key."""
    config["f_sky"] = 1.0

  def extra_noise(config):
    """Add one mistyped noise key."""
    config["noise"]["delta_et"] = 0.0

  def extra_nongaussian(config):
    """Add one mistyped non-Gaussian key."""
    config["nongaussian"]["step_frac"] = [0.01, 0.02]

  def missing_noise(config):
    """Remove one formerly defaulted noise key."""
    del config["noise"]["delta_te"]

  expect_refusal("unknown cov_args key refuses", extra_top, "f_sky")
  expect_refusal("unknown noise key refuses", extra_noise, "delta_et")
  expect_refusal("unknown non-Gaussian key refuses",
                 extra_nongaussian,
                 "step_frac")
  expect_refusal("missing explicit noise key refuses",
                 missing_noise,
                 "missing required")


class ShortScaledLensing:
  """Expose a deliberately short scaled lensing spectrum."""

  def get_lens_potential_cls(self, raw_cl=False):
    """Return a scaled spectrum ending below the requested range.

    Arguments:
      raw_cl = convention selector, which must remain false.

    Returns:
      a short two-column array.
    """
    if raw_cl:
      raise AssertionError("the scaled lensing convention was not requested")
    return np.zeros((3, 2), dtype="float64")


def ignore_log(message):
  """Discard one progress message from a deliberately failing run.

  Arguments:
    message = progress text, unused because the run must fail first.

  Returns:
    None.
  """


def check_lensing_range_completeness():
  """Require both raw and scaled lensing arrays to cover lens_lmax.

  Returns:
    None.
  """
  config = valid_covariance_args()["nongaussian"]
  config["enabled"] = True
  config["lens_lmax"] = 4
  config["band_width"] = 1
  ell = np.arange(2, 5, dtype="int64")
  short_raw = {"pp": np.ones(4, dtype="float64")}
  try:
    covariance.nongaussian_blocks(
      cambdata=ShortScaledLensing(),
      cls=short_raw,
      ell=ell,
      ng_cfg=config,
      fsky=1.0,
      log=ignore_log)
  except ValueError as error:
    report("short raw lensing range refuses",
           "raw lensing spectrum is too short" in str(error),
           str(error))
  else:
    report("short raw lensing range refuses", False, "accepted")

  complete_raw = {"pp": np.ones(5, dtype="float64")}
  try:
    covariance.nongaussian_blocks(
      cambdata=ShortScaledLensing(),
      cls=complete_raw,
      ell=ell,
      ng_cfg=config,
      fsky=1.0,
      log=ignore_log)
  except ValueError as error:
    report("short scaled lensing range refuses",
           "scaled lensing spectrum is too short" in str(error),
           str(error))
  else:
    report("short scaled lensing range refuses", False, "accepted")


def main():
  """Run every pure covariance-schema witness.

  Returns:
    process exit status.
  """
  print("covariance configuration boundary witness")
  valid = valid_covariance_args()
  returned = covariance.validate_cov_args(valid)
  report("complete shipped-value configuration passes",
         returned is valid,
         "validated mapping identity is preserved")
  check_scalar_ranges()
  check_step_ranges()
  check_exact_keys()
  check_lensing_range_completeness()
  if FAILURES:
    print("FAIL: " + str(len(FAILURES)) + " covariance schema leg(s)")
    return 1
  print("PASS: covariance schema rejects every malformed input")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

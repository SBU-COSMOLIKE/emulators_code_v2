"""Small validation predicates shared by configuration and geometry code."""

import math


def _is_finite_real(value):
  """Return whether a value is a finite, non-Boolean Python real number.

  Validation checks the original type before any conversion. This rejects
  Boolean values and numeric strings instead of turning them into 1.0 or a
  parsed float.

  Arguments:
    value = candidate configuration or saved numeric value.

  Returns:
    True for a finite int or float other than bool; False otherwise.
  """
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return False
  return math.isfinite(value)

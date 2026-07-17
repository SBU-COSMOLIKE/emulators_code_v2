"""Small validation helpers shared by configuration and model code."""

import math
from numbers import Integral
import struct


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


def require_exact_bool(value, name):
  """Return one real Boolean setting, without truth-value coercion.

  A quoted YAML value such as ``"false"`` is a nonempty Python string and
  therefore evaluates as true.  Model switches must reject that spelling
  instead of silently enabling the feature.
  """
  if type(value) is not bool:
    raise TypeError(
      name + " must be true or false without quotes; got " + repr(value))
  return value


def require_exact_int(value, name, *, minimum):
  """Return one native integer at or above ``minimum`` without coercion."""
  if type(value) is not int:
    raise TypeError(
      name + " must be an integer, not a Boolean, string, or fraction; got "
      + repr(value))
  if value < minimum:
    raise ValueError(
      name + " must be >= " + str(minimum) + "; got " + repr(value))
  return value


def require_finite_real(value, name):
  """Return one finite Python ``int`` or ``float``, excluding Booleans."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(
      name + " must be a finite real number, not a Boolean or string; got "
      + repr(value))
  if not math.isfinite(value):
    raise ValueError(name + " must be finite; got " + repr(value))
  return value


def require_nonzero_float32(value, name):
  """Return a finite gate value that stays nonzero in model precision.

  Correction gates are stored as float32 parameters.  A nonzero Python
  number such as ``1e-50`` rounds to zero in that format and creates the same
  dead head as an explicit zero.  Overflow is refused for the same reason:
  the stored parameter would no longer be finite.
  """
  value = require_finite_real(value, name)
  try:
    stored = struct.unpack("f", struct.pack("f", float(value)))[0]
  except OverflowError:
    stored = math.copysign(math.inf, float(value))
  if not math.isfinite(stored) or stored == 0.0:
    raise ValueError(
      name + " must remain finite and nonzero after conversion to float32; "
      "got " + repr(value))
  return value


def require_positive_int_list(values, name):
  """Return a nonempty list of positive integral layout sizes.

  Output geometries may store ordinary Python integers or integer scalars
  supplied by a numerical library.  Booleans, text, fractions, empty lists,
  and zero-width bins have no physical layout meaning and are refused before
  a model allocates learned layers.
  """
  if isinstance(values, (str, bytes)):
    raise TypeError(name + " must be a sequence of positive integers")
  try:
    candidates = list(values)
  except TypeError as error:
    raise TypeError(
      name + " must be a sequence of positive integers") from error
  if not candidates:
    raise ValueError(name + " must contain at least one bin")
  resolved = []
  for index, value in enumerate(candidates):
    item_name = name + "[" + str(index) + "]"
    if isinstance(value, bool) or not isinstance(value, Integral):
      raise TypeError(item_name + " must be an integer; got " + repr(value))
    value = int(value)
    if value < 1:
      raise ValueError(item_name + " must be >= 1; got " + repr(value))
    resolved.append(value)
  return resolved

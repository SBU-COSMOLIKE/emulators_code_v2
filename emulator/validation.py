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
  """Read one real Boolean model setting, without truth-value coercion.

  A quoted YAML value such as ``"false"`` is a nonempty Python string and
  therefore evaluates as true.  A model switch must reject that spelling
  instead of silently enabling the feature.

  Arguments:
    value = the configured setting, expected to be the YAML Boolean
            true or false.
    name  = the configuration key, named in the refusal.

  Returns:
    the value unchanged, when it is exactly a Python bool.

  Raises:
    TypeError naming the key when the value is anything but a bool.
  """
  if type(value) is not bool:
    raise TypeError(
      name + " must be true or false without quotes; got " + repr(value))
  return value


def require_exact_int(value, name, *, minimum):
  """Read one native integer at or above ``minimum``, without coercion.

  Refused by type, not by value: True (which equals 1 in Python), the
  string "3", and the float 3.0 are all configuration mistakes worth
  stopping, even though each would convert to a usable integer.

  Arguments:
    value   = the configured setting, expected to be a plain int.
    name    = the configuration key, named in the refusal.
    minimum = the smallest accepted value (inclusive).

  Returns:
    the value unchanged, when it is a plain int >= minimum.

  Raises:
    TypeError when the value is not a plain int; ValueError when it is
    below the minimum.
  """
  if type(value) is not int:
    raise TypeError(
      name + " must be an integer, not a Boolean, string, or fraction; got "
      + repr(value))
  if value < minimum:
    raise ValueError(
      name + " must be >= " + str(minimum) + "; got " + repr(value))
  return value


def require_finite_real(value, name):
  """Read one finite real number, refusing Booleans, strings, and NaN.

  Arguments:
    value = the configured setting, expected to be an int or float.
    name  = the configuration key, named in the refusal.

  Returns:
    the value unchanged, when it is a finite int or float.

  Raises:
    TypeError when the value is a Boolean or not a number; ValueError
    when it is NaN or infinite.
  """
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(
      name + " must be a finite real number, not a Boolean or string; got "
      + repr(value))
  if not math.isfinite(value):
    raise ValueError(name + " must be finite; got " + repr(value))
  return value


def require_nonzero_float32(value, name):
  """Read a gate value that stays finite and nonzero in model precision.

  Correction gates are stored as float32 parameters.  A nonzero Python
  number such as ``1e-50`` rounds to zero in that format and creates the
  same dead head as an explicit zero; a huge value overflows to infinity.
  The value is therefore round-tripped through the 4-byte float format
  first, and judged in that precision.

  Arguments:
    value = the configured gate value, an int or float.
    name  = the configuration key, named in the refusal.

  Returns:
    the ORIGINAL Python value (not the float32 rounding), once its
    float32 image is known to be finite and nonzero.

  Raises:
    TypeError / ValueError from the finite-real check; ValueError when
    the float32 image is zero or infinite.
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
  """Read a nonempty list of positive layout sizes (bin widths).

  Output geometries may store ordinary Python integers or integer scalars
  supplied by a numerical library, so integral numpy scalars are accepted
  and converted.  Booleans, text, fractions, empty lists, and zero-width
  bins have no physical layout meaning and are refused before a model
  allocates learned layers around them.

  Arguments:
    values = the layout sizes, a sequence of positive integers (plain or
             numpy integral scalars).
    name   = what the sequence describes, named in every refusal; each
             refusal also names the offending index as name[i].

  Returns:
    a new list of plain Python ints, every entry >= 1.

  Raises:
    TypeError when the input is not a sequence or an entry is not an
    integer; ValueError when the list is empty or an entry is below 1.
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

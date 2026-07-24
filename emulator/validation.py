"""Small validation helpers shared by configuration and model code.

Each helper reads one configured value, refuses the look-alikes Python
would silently accept in its place, and returns the value unchanged.
The refusals exist because most settings arrive from YAML text files,
and a YAML typo usually produces a valid Python value of the wrong type
rather than an error: quoting a number yields a string, quoting a
Boolean yields a nonempty (therefore truthy) string, and Python itself
treats ``True`` as the number 1.  Catching those at the configuration
boundary keeps the mistake out of the model, where it would surface much
later as wrong numerics instead of a named error.
"""

import math
from numbers import Integral
import struct

import numpy as np


def _is_finite_real(value):
  """Return whether a value is a finite, non-Boolean Python real number.

  The check reads the original type before any conversion, so Boolean
  values and numeric strings are rejected instead of being turned into
  1.0 or a parsed float.  The Boolean test must run first because Python
  defines ``bool`` as a subclass of ``int``: ``isinstance(True, int)``
  is true and ``True`` behaves as the number 1, so without that first
  test a Boolean would pass as a valid number.  ``math.isfinite`` then
  rejects the two special floating-point values NaN (not-a-number) and
  infinity, which arithmetic propagates silently instead of failing on.

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

  Truth-value coercion is Python's willingness to use any value as a
  condition: a nonempty string counts as true, an empty one as false.
  The quoted YAML value ``"false"`` is a five-character string, not the
  Boolean false, so used as a condition it would silently ENABLE the
  feature it names.  The check ``type(value) is bool`` asks for the
  value's exact type instead of asking whether the value is usable as a
  condition, so every such look-alike is refused with the key's name.

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

  The check ``type(value) is int`` compares the value's exact type, so
  the value is refused by type, not by whether it could be converted:
  ``True`` (which Python defines as a subclass of ``int`` equal to 1),
  the string ``"3"``, and the float ``3.0`` are all configuration
  mistakes worth stopping, even though ``int()`` would turn each of
  them into a usable integer.  Only after the type is right is the
  value itself compared against the minimum.

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

  The Boolean test runs before the number test because ``bool`` is a
  subclass of ``int`` in Python, so ``isinstance(True, (int, float))``
  is true and a Boolean would otherwise pass as the number 1.  The
  finiteness test refuses NaN (not-a-number) and infinity: both are
  legal floats that later arithmetic propagates silently, so a setting
  holding one is a mistake to stop here, with the key's name attached.

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


def whitening_scale_from_eigenvalues(eigenvalues, source):
  """Square-root a covariance's eigenvalues, refusing a non-positive one.

  A whitening transform divides each direction by the square root of the
  covariance's eigenvalue in that direction (see the geometry modules).
  A valid covariance is positive definite, so every eigenvalue is finite
  and strictly greater than zero.  A zero eigenvalue would make the
  whitening divide by zero (an infinity); a negative one would take the
  square root of a negative number (a not-a-number).  Either means the
  covariance is degenerate -- a pinned or duplicated parameter, or a
  corrupt covariance file -- so refuse it here, at the build boundary,
  naming its source, instead of letting the bad scale surface much later
  as a non-finite training loss with no covariance named.

  Arguments:
    eigenvalues = the covariance eigenvalues, a NumPy array (for example
                  the ascending eigenvalues numpy.linalg.eigh returns).
    source      = a short label for where the covariance came from (a
                  file path or a probe name), quoted in the refusal.

  Returns:
    a NumPy array of the eigenvalue square roots: the per-direction
    whitening scale.

  Raises:
    ValueError when any eigenvalue is not a finite, strictly positive
    number.
  """
  eigenvalues = np.asarray(eigenvalues, dtype="float64")
  finite_positive = np.isfinite(eigenvalues) & (eigenvalues > 0.0)
  if not bool(finite_positive.all()):
    smallest = float(eigenvalues.min())
    raise ValueError(
      "the covariance from " + str(source) + " is not positive definite: "
      "its smallest eigenvalue is " + repr(smallest) + " (a valid "
      "covariance has every eigenvalue finite and greater than zero). A "
      "zero or negative eigenvalue is a degenerate direction -- a pinned "
      "or duplicated parameter, or a corrupt covariance file -- and would "
      "make the whitening transform divide by zero. Drop the degenerate "
      "parameter from the covariance and rebuild.")
  return np.sqrt(eigenvalues)


def require_nonzero_float32(value, name):
  """Read a gate value that stays finite and nonzero in model precision.

  A gate here is a small learned multiplier that scales a model head's
  output; it is stored inside the model as a float32 parameter, the
  4-byte floating-point format the emulators train in.  A Python float
  is the wider 8-byte format, so a value can be perfectly valid in
  Python yet degenerate once stored: ``1e-50`` is below the smallest
  magnitude float32 can represent and rounds to exactly zero, creating
  the same dead head (a head whose output is identically zero, which a
  multiplicative gate cannot train back to life) as an explicit zero,
  while a huge value becomes infinity.

  The check therefore judges the value in the model's precision, not
  Python's.  ``struct.pack("f", x)`` encodes the number into the exact
  4 bytes a float32 parameter would hold, and ``struct.unpack`` decodes
  those bytes back into a Python float; the round trip yields the value
  the model would actually store.  ``struct.pack`` raises OverflowError
  for a magnitude too large for the 4-byte format, and the handler maps
  that case to infinity of the same sign so it fails the finiteness
  test like any other overflow.

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

  Output geometries may store ordinary Python integers or the integer
  scalars a numerical library hands back (a numpy array element is a
  numpy scalar, not a plain ``int``).  The entry test uses
  ``numbers.Integral``, Python's umbrella type that plain ints and
  numpy integer scalars both register under, so either spelling is
  accepted and converted to a plain ``int``.  Booleans still fail the
  entry test through the explicit ``bool`` check that runs first,
  because ``bool`` registers under ``Integral`` too.

  A bare string is refused before the list conversion because Python
  treats a string as a sequence of its own characters: ``list("12")``
  is ``["1", "2"]``, not the number 12.  The ``list(values)`` call
  itself raises TypeError for a value that cannot be iterated at all,
  and that error is re-raised with the sequence's name attached.
  Empty lists and zero-width bins have no physical layout meaning and
  are refused before a model allocates learned layers around them.

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

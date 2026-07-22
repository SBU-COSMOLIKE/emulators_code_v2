"""The smallest cobaya.log surface the adapter's import needs."""


class LoggedError(Exception):
  """Stand-in for cobaya's logged exception type."""


class _SilentLogger:
  """A logger whose debug output goes nowhere."""

  def debug(self, *args, **kwargs):
    """Discard a debug message."""
    del args, kwargs


def get_logger(name):
  """Return a silent logger for any requested name."""
  del name
  return _SilentLogger()

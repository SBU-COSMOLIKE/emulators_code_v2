"""The smallest cobaya.theory surface the adapter's import needs."""


class Theory:
  """Base-class stand-in: bare construction plus the two hooks used.

  The real cobaya Theory configures logging, requirements, and timing
  at construction. The child checks build adapter instances directly
  and never run cobaya's model machinery, so this stand-in keeps
  construction empty and gives the two methods the adapter overrides
  and then calls upward (initialize and must_provide) do-nothing
  bodies.
  """

  extra_args = {}
  output_params = []

  def initialize(self):
    """Do nothing; the adapter's own initialize is tested elsewhere."""
    return None

  def must_provide(self, **requirements):
    """Accept any requirement mapping and request nothing extra."""
    del requirements
    return None

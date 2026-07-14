"""This module resolves stable Optuna study names from emulator families."""


def resolve_study_name(family):
  """Return the stable Optuna study name owned by one emulator family.

  The command name is deliberately not an input. A wrapper may be renamed
  without changing the study that an existing journal resumes.

  Arguments:
    family = explicit emulator family identity.

  Returns:
    Stable study name for that family.
  """
  if family == "cosmolike":
    return "cosmic_shear_tune"
  if family == "outputs":
    return "scalar_tune_emulator"
  if family == "cmb":
    return "cmb_tune_emulator"
  if family == "grid":
    return "baosn_tune_emulator"
  if family == "grid2d":
    return "mps_tune_emulator"

  supported = "cosmolike, outputs, cmb, grid, grid2d"
  raise ValueError(
    "Unknown emulator family " + repr(family) + ". Supported families are: "
    + supported + "."
  )

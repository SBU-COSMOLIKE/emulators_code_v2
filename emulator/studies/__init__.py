"""Identity, manifest, and naming support for tuning studies.

The modules are kept separate because they own distinct contracts:

  implementation.py   versioned semantic implementation identities.
  manifest.py         scientific identity records and journal binding.
  manifest_digest.py  strict canonical JSON and SHA-256 helpers.
  name.py             stable per-family Optuna study names.

Import the specific owner needed; this package deliberately re-exports
nothing.
"""

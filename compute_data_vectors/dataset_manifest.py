"""Pure dataset-publication identity and run-control surfaces.

This module starts the dataset-manifest boundary with the command state that
decides whether a generator creates, resumes, or appends a dataset.  It imports
no generator, Cobaya, MPI, filesystem, or numerical code, so callers can reject
an invalid state before looking up paths or touching output.
"""

from dataclasses import dataclass
import math
import re
from types import MappingProxyType


DATASET_REQUEST_SCHEMA = 1
UNIFORM_BOUNDARY_INTERIOR_POLICY = (
  "nextafter-toward-interval-interior-v1")
DATASET_PROBE_FAMILIES = MappingProxyType({
  "cs": "cosmolike",
  "ggl": "cosmolike",
  "gc": "cosmolike",
  "cmblensed": "cmb",
  "cmbunlensed": "cmb",
  "background": "grid",
  "mps": "grid2d",
})
DATASET_PROBE_GENERATORS = MappingProxyType({
  "cs": "dataset_generator_lensing",
  "ggl": "dataset_generator_lensing",
  "gc": "dataset_generator_lensing",
  "cmblensed": "dataset_generator_cmb",
  "cmbunlensed": "dataset_generator_cmb",
  "background": "dataset_generator_background",
  "mps": "dataset_generator_mps",
})
DATASET_SAMPLING_POLICIES = MappingProxyType({
  "uniform": MappingProxyType({
    "algorithm": "uniform-box-v1",
    "bit_generator": "PCG64",
    "emcee_random": None,
    "owner": "numpy.random.Generator",
    "policy": "persist-complete-state-v1",
  }),
  "gaussian-mcmc": MappingProxyType({
    "algorithm": "emcee-de-snooker-v1",
    "bit_generator": "PCG64",
    "emcee_random": "MT19937",
    "owner": "numpy.random.Generator",
    "policy": "persist-complete-state-v1",
  }),
})

_REQUEST_KEYS = (
  "dataset_mode", "family", "family_variant", "generator", "parameters",
  "probe", "sampling", "schema", "scientific_contract_sha256")
_SAMPLING_KEYS = (
  "algorithm", "boundary_factor", "boundary_interior_policy",
  "max_correlation", "mode", "rng", "seed", "temperature")
_RNG_KEYS = ("bit_generator", "emcee_random", "owner", "policy")
_PARAMETER_KEYS = ("configuration_sha256", "dtype", "names")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_PORTABLE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MAX_STEM_LENGTH = 200
_MAX_JSON_INTEGER_BITS = 3402
_MAX_JSON_INTEGER_DIGITS = 1024


def build_dataset_request_identity(*, dataset_mode, family, family_variant,
                                   generator, probe, sampling_mode,
                                   temperature, boundary_factor,
                                   max_correlation, sampling_algorithm, seed,
                                   rng_bit_generator, rng_emcee_random,
                                   rng_policy,
                                   boundary_interior_policy, ordered_names,
                                   configuration_sha256,
                                   scientific_contract_sha256):
  """Build the immutable scientific request for one logical dataset.

  This record says *which* rows and payload semantics a caller requested.  It
  deliberately excludes run-control flags, requested batch size, committed row
  count, and mutable random-number-generator state.  Those values describe a
  transaction or one committed generation, not the invariant dataset request.
  A later integration slice must publish complete continuation state as its own
  authenticated member.

  Requested and resolved support are also not copied into this object.  The
  producer-authored ``.facts.yaml`` sidecar is their single source of truth.
  ``scientific_contract_sha256`` binds the versioned stable projection owned by
  ``emulator.fixed_facts.scientific_contract_digest``.  That projection omits
  only the generation-specific chain digest, so a valid append keeps the same
  request while the full sidecar member remains authenticated separately.
  """
  if type(ordered_names) is not list:
    raise ValueError(
      "ordered_names must be a nonempty native list, not an iterable that "
      "could be split or reordered; got " + repr(ordered_names))
  identity = {
    "dataset_mode": dataset_mode,
    "family": family,
    "family_variant": family_variant,
    "generator": generator,
    "parameters": {
      "configuration_sha256": configuration_sha256,
      "dtype": "float32",
      "names": list(ordered_names),
    },
    "probe": probe,
    "sampling": {
      "algorithm": sampling_algorithm,
      "boundary_factor": boundary_factor,
      "boundary_interior_policy": boundary_interior_policy,
      "max_correlation": max_correlation,
      "mode": sampling_mode,
      "rng": {
        "bit_generator": rng_bit_generator,
        "emcee_random": rng_emcee_random,
        "owner": "numpy.random.Generator",
        "policy": rng_policy,
      },
      "seed": seed,
      "temperature": temperature,
    },
    "schema": DATASET_REQUEST_SCHEMA,
    "scientific_contract_sha256": scientific_contract_sha256,
  }
  validate_dataset_request_identity(identity)
  return identity


def validate_dataset_request_identity(identity):
  """Validate one request identity without supplying any missing defaults.

  The accepted object is intentionally strict enough to validate a record read
  back from canonical JSON.  Unknown, missing, or wrongly typed fields refuse;
  booleans never stand in for integers and nonfinite controls never enter a
  manifest.
  """
  _require_exact_keys(identity, _REQUEST_KEYS, "dataset request identity")
  if type(identity["schema"]) is not int \
      or identity["schema"] != DATASET_REQUEST_SCHEMA:
    raise ValueError(
      "dataset request schema must be the native integer "
      + str(DATASET_REQUEST_SCHEMA) + "; got " + repr(identity["schema"]))

  _validate_dataset_route(
    dataset_mode=identity["dataset_mode"],
    family=identity["family"],
    family_variant=identity["family_variant"],
    generator=identity["generator"],
    probe=identity["probe"])

  _require_digest(identity["scientific_contract_sha256"],
                  "invariant scientific contract")

  parameters = identity["parameters"]
  _require_exact_keys(parameters, _PARAMETER_KEYS, "dataset parameters")
  if parameters["dtype"] != "float32":
    raise ValueError(
      "dataset parameter dtype must be 'float32'; got "
      + repr(parameters["dtype"]))
  _require_digest(parameters["configuration_sha256"],
                  "resolved configuration")
  names = parameters["names"]
  if type(names) is not list or not names:
    raise ValueError(
      "dataset parameter names must be a nonempty ordered JSON list")
  if len(names) > 4096:
    raise ValueError("dataset parameter-name list exceeds 4096 entries")
  seen = set()
  for name in names:
    if type(name) is not str or len(name) > 256 \
        or not _PARAMETER_NAME_RE.fullmatch(name):
      raise ValueError(
        "dataset parameter name must be a portable identifier; got "
        + repr(name))
    if name in seen:
      raise ValueError("dataset parameter name is repeated: " + repr(name))
    seen.add(name)

  sampling = identity["sampling"]
  _require_exact_keys(sampling, _SAMPLING_KEYS, "dataset sampling record")
  mode = sampling["mode"]
  if type(mode) is not str or mode not in DATASET_SAMPLING_POLICIES:
    raise ValueError(
      "sampling mode must be 'uniform' or 'gaussian-mcmc'; got "
      + repr(mode))
  policy = DATASET_SAMPLING_POLICIES[mode]
  if sampling["algorithm"] != policy["algorithm"]:
    raise ValueError(
      "sampling mode " + repr(mode) + " requires algorithm "
      + repr(policy["algorithm"]) + "; got "
      + repr(sampling["algorithm"]))
  _native_integer(sampling["temperature"], "sampling temperature", minimum=1)
  _native_integer(sampling["seed"], "sampling seed", minimum=0)
  boundary = _native_finite_float(
    sampling["boundary_factor"], "sampling boundary factor")
  if not 0.0 < boundary <= 1.0:
    raise ValueError(
      "sampling boundary factor must be in (0, 1]; got " + repr(boundary))

  if mode == "uniform":
    if sampling["max_correlation"] is not None:
      raise ValueError(
        "uniform sampling requires max_correlation=null; got "
        + repr(sampling["max_correlation"]))
    if sampling["boundary_interior_policy"] \
        != UNIFORM_BOUNDARY_INTERIOR_POLICY:
      raise ValueError(
        "uniform sampling requires boundary-interior policy "
        + repr(UNIFORM_BOUNDARY_INTERIOR_POLICY) + "; got "
        + repr(sampling["boundary_interior_policy"]))
  else:
    correlation = _native_finite_float(
      sampling["max_correlation"], "maximum sampling correlation")
    if not 0.01 < correlation <= 1.0:
      raise ValueError(
        "maximum sampling correlation must be in (0.01, 1]; got "
        + repr(correlation))
    if sampling["boundary_interior_policy"] is not None:
      raise ValueError(
        "gaussian-mcmc sampling requires boundary_interior_policy=null; got "
        + repr(sampling["boundary_interior_policy"]))

  rng = sampling["rng"]
  _require_exact_keys(rng, _RNG_KEYS, "dataset RNG policy")
  for field in ("bit_generator", "emcee_random", "owner", "policy"):
    if rng[field] != policy[field]:
      raise ValueError(
        "sampling mode " + repr(mode) + " requires RNG " + field + "="
        + repr(policy[field]) + "; got " + repr(rng[field]))
  return identity


def _validate_dataset_route(*, dataset_mode, family, family_variant,
                            generator, probe):
  """Return the validated fields that route one dataset to its driver."""
  if dataset_mode not in ("full", "chain-only"):
    raise ValueError(
      "dataset request mode must be 'full' or 'chain-only'; got "
      + repr(dataset_mode))

  probe = _portable_string(probe, "dataset probe")
  if probe not in DATASET_PROBE_FAMILIES:
    raise ValueError("unknown dataset probe " + repr(probe))
  expected_family = DATASET_PROBE_FAMILIES[probe]
  if family != expected_family:
    raise ValueError(
      "dataset probe " + repr(probe) + " belongs to family "
      + repr(expected_family) + ", not " + repr(family))

  variant = family_variant
  if family == "grid2d":
    if variant not in ("native", "syren-base"):
      raise ValueError(
        "grid2d family variant must be 'native' or 'syren-base'; got "
        + repr(variant))
  elif variant != "standard":
    raise ValueError(
      "family " + repr(family) + " requires variant 'standard'; got "
      + repr(variant))

  generator = _portable_string(generator, "dataset generator")
  expected_generator = DATASET_PROBE_GENERATORS[probe]
  if generator != expected_generator:
    raise ValueError(
      "dataset probe " + repr(probe) + " requires generator "
      + repr(expected_generator) + "; got " + repr(generator))
  return {
    "dataset_mode": dataset_mode,
    "family": family,
    "family_variant": variant,
    "generator": generator,
    "probe": probe,
  }


@dataclass(frozen=True)
class DatasetMemberCensus:
  """Validated route and member names for one generator checkpoint."""

  route: MappingProxyType
  members: MappingProxyType


def build_dataset_member_census(*, dataset_mode, family, family_variant,
                                generator, probe, params_stem, dvs_stem,
                                fail_stem):
  """Build the immutable route and member names known before publication.

  This census deliberately excludes configuration and scientific digests. It
  lets a generator bind its validated driver route and exact checkpoint names
  before any checkpoint file is inspected. The complete request identity is a
  later publication input and remains mandatory at that boundary.
  """
  route = _validate_dataset_route(
    dataset_mode=dataset_mode,
    family=family,
    family_variant=family_variant,
    generator=generator,
    probe=probe)
  stems = _validate_dataset_stems(
    params_stem=params_stem,
    dvs_stem=dvs_stem,
    fail_stem=fail_stem)
  members = _build_dataset_member_map(route=route, stems=stems)
  return DatasetMemberCensus(
    route=MappingProxyType(dict(route)),
    members=MappingProxyType(dict(members)))


def build_dataset_member_map(identity, *, params_stem, dvs_stem, fail_stem):
  """Return the exact semantic role-to-basename map for one request.

  The stems are the already mode-scoped basenames used in the slot descriptor.
  Full CMB publication intentionally requires a persisted integer multipole
  axis.  The current generator does not yet write that member, so later
  integration must add it or fail closed rather than infer coordinates from
  array width.
  """
  validate_dataset_request_identity(identity)
  route = {
    "dataset_mode": identity["dataset_mode"],
    "family": identity["family"],
    "family_variant": identity["family_variant"],
    "generator": identity["generator"],
    "probe": identity["probe"],
  }
  stems = _validate_dataset_stems(
    params_stem=params_stem,
    dvs_stem=dvs_stem,
    fail_stem=fail_stem)
  return _build_dataset_member_map(route=route, stems=stems)


def _validate_dataset_stems(*, params_stem, dvs_stem, fail_stem):
  """Return three portable, case-distinct publication basenames."""
  stems = {}
  for label, value in (("parameter", params_stem),
                       ("data-vector", dvs_stem),
                       ("failure", fail_stem)):
    if type(value) is not str or len(value) > _MAX_STEM_LENGTH \
        or not _STEM_RE.fullmatch(value) \
        or value in (".", "..") or value.startswith("."):
      raise ValueError(
        label + " stem must be one portable, visible basename; got "
        + repr(value))
    stems[label] = value
  if len({value.casefold() for value in stems.values()}) != 3:
    raise ValueError(
      "parameter, data-vector, and failure stems must be distinct on "
      "case-insensitive filesystems")
  return stems


def _build_dataset_member_map(*, route, stems):
  """Build one role-to-basename map from an already validated route."""
  params = stems["parameter"]
  dvs = stems["data-vector"]
  fail = stems["failure"]
  members = {
    "parameters.chain": params + ".1.txt",
    "parameters.schema": params + ".paramnames",
    "parameters.covariance": params + ".covmat",
    "parameters.ranges": params + ".ranges",
    "metadata.scientific-facts": params + ".facts.yaml",
  }
  if route["dataset_mode"] == "chain-only":
    _require_unique_member_paths(members)
    return members

  members["rows.failure-mask"] = fail + ".txt"
  family = route["family"]
  if family == "cosmolike":
    members["payload.cosmolike.vector"] = dvs + ".npy"
  elif family == "cmb":
    for spectrum in ("tt", "te", "ee", "pp"):
      members["payload.cmb." + spectrum] = dvs + "_" + spectrum + ".npy"
    members["axis.cmb.multipole"] = dvs + "_ell.npy"
  elif family == "grid":
    for quantity in ("h", "dm"):
      members["payload.grid." + quantity] = dvs + "_" + quantity + ".npy"
      members["axis.grid." + quantity + ".redshift"] = (
        dvs + "_" + quantity + "_z.npy")
  elif family == "grid2d":
    members["payload.grid2d.pklin"] = dvs + "_pklin.npy"
    members["payload.grid2d.boost"] = dvs + "_boost.npy"
    members["axis.grid2d.redshift"] = dvs + "_z.npy"
    members["axis.grid2d.wavenumber"] = dvs + "_k.npy"
    if route["family_variant"] == "syren-base":
      members["base.grid2d.pklin"] = dvs + "_pklin_base.npy"
      members["base.grid2d.boost"] = dvs + "_boost_base.npy"
  else:
    raise AssertionError("validated dataset family was not routed")
  _require_unique_member_paths(members)
  return members


def _require_exact_keys(value, keys, label):
  if type(value) is not dict:
    raise ValueError(label + " must be a JSON object")
  if any(type(key) is not str for key in value):
    raise ValueError(label + " field names must all be strings")
  missing = sorted(set(keys) - set(value))
  unknown = sorted(set(value) - set(keys))
  if missing or unknown:
    raise ValueError(
      label + " fields differ from the schema: missing=" + repr(missing)
      + ", unknown=" + repr(unknown))


def _portable_string(value, label):
  if type(value) is not str or len(value) > 256 \
      or not _PORTABLE_TOKEN_RE.fullmatch(value):
    raise ValueError(label + " must be a bounded portable token; got "
                     + repr(value))
  return value


def _require_digest(value, label):
  if type(value) is not str or not _HEX64_RE.fullmatch(value):
    raise ValueError(label + " SHA-256 must be 64 lower-case hex digits; got "
                     + repr(value))


def _native_integer(value, label, minimum):
  if type(value) is not int:
    raise ValueError(label + " must be a native integer >= " + str(minimum)
                     + " (not bool); got " + repr(value))
  if value.bit_length() > _MAX_JSON_INTEGER_BITS:
    raise ValueError(
      label + " exceeds the publication limit of "
      + str(_MAX_JSON_INTEGER_BITS) + " bits")
  if len(str(abs(value))) > _MAX_JSON_INTEGER_DIGITS:
    raise ValueError(
      label + " exceeds the publication limit of "
      + str(_MAX_JSON_INTEGER_DIGITS) + " decimal digits")
  if value < minimum:
    raise ValueError(label + " must be a native integer >= " + str(minimum)
                     + "; got " + repr(value))
  return value


def _require_unique_member_paths(members):
  folded = {}
  for role, relative in members.items():
    key = relative.casefold()
    if key in folded:
      raise ValueError(
        "dataset member paths collide on a case-insensitive filesystem: "
        + repr(folded[key]) + " and " + repr(role) + " both select "
        + repr(relative))
    folded[key] = role


def _native_finite_float(value, label):
  if type(value) is not float or not math.isfinite(value):
    raise ValueError(label + " must be a finite native float; got "
                     + repr(value))
  return value


class CheckpointLoadError(RuntimeError):
  """A requested checkpoint cannot be loaded without risking old output."""


def scope_dataset_stem(stem, dataset_mode):
  """Place chain-only outputs in a namespace distinct from full datasets.

  Full datasets keep the historical stem supplied by the caller. Chain-only
  datasets add one explicit suffix to every parameter, failure, and
  data-vector stem.  Even though chain-only generation writes only parameter
  members, scoping all three stems prevents later code from borrowing a full
  dataset's payload or failure mask by path coincidence.

  Arguments:
    stem = nonempty output stem, including any parent directory.
    dataset_mode = normalized ``full`` or ``chain-only`` mode.

  Returns:
    the unchanged full stem or the chain-only-scoped stem.

  Raises:
    ValueError when the stem or normalized mode is invalid.
  """
  if type(stem) is not str or not stem:
    raise ValueError("dataset output stem must be a nonempty string; got "
                     + repr(stem))
  if dataset_mode == "full":
    return stem
  if dataset_mode == "chain-only":
    return stem + "_chain_only"
  raise ValueError("Unknown normalized generator dataset mode: "
                   + repr(dataset_mode))


@dataclass(frozen=True)
class RunControl:
  """One normalized generator operation and dataset mode.

  Attributes:
    loadchk = 1 when an existing validated dataset is requested, else 0.
    append = 1 when new rows extend that dataset, else 0.
    chain = 1 for a chain-only dataset, else 0 for a full dataset.
    operation = ``fresh``, ``resume``, or ``append``.
    dataset_mode = ``full`` or ``chain-only``.
  """

  loadchk: int
  append: int
  chain: int
  operation: str
  dataset_mode: str


def _binary_flag(name, value, default):
  """Normalize one native-integer 0/1 flag.

  Arguments:
    name = command-line flag name without its leading dashes.
    value = value supplied by argparse or a programmatic caller.
    default = replacement for None, or None when absence is invalid.

  Returns:
    the native integer 0 or 1.

  Raises:
    ValueError when the value is absent without a default, is not a native
    integer, or is outside the two legal values.
  """
  if value is None:
    if default is None:
      raise ValueError(
        "--" + name + " must be a native integer 0 or 1 (not bool); got None")
    return default
  if type(value) is not int or value not in (0, 1):
    raise ValueError(
      "--" + name + " must be a native integer 0 or 1 (not bool); got "
      + repr(value))
  return value


def validate_run_control(loadchk, append, chain):
  """Validate and normalize the generator's run-control state.

  The three legal operations are fresh ``loadchk=0, append=0``, resume
  ``loadchk=1, append=0``, and append ``loadchk=1, append=1``.  Appending
  without loading is refused because append extends a validated prior dataset;
  it never means fresh generation at the same output path.  The independent
  chain axis records whether the operation targets a full or chain-only
  dataset.

  Arguments:
    loadchk = optional native integer 0/1; None means 0.
    append = optional native integer 0/1; None means 0.
    chain = optional native integer 0/1; None means 0.

  Returns:
    an immutable ``RunControl`` with normalized flags, operation, and dataset
    mode.

  Raises:
    ValueError when a flag is not a native integer 0/1, or append is requested
    without loading the prior dataset first.
  """
  normalized_loadchk = _binary_flag(
    name="loadchk", value=loadchk, default=0)
  normalized_append = _binary_flag(
    name="append", value=append, default=0)
  normalized_chain = _binary_flag(
    name="chain", value=chain, default=0)

  if normalized_append == 1 and normalized_loadchk != 1:
    raise ValueError(
      "--append=1 requires --loadchk=1; append extends a validated prior "
      "dataset and never starts fresh generation at the same path. Got "
      "--loadchk=" + str(normalized_loadchk)
      + " and --append=" + str(normalized_append) + ".")

  if normalized_loadchk == 0:
    operation = "fresh"
  elif normalized_append == 0:
    operation = "resume"
  else:
    operation = "append"
  dataset_mode = "chain-only" if normalized_chain == 1 else "full"

  return RunControl(
    loadchk=normalized_loadchk,
    append=normalized_append,
    chain=normalized_chain,
    operation=operation,
    dataset_mode=dataset_mode)


def require_checkpoint_members(operation, members, is_file):
  """Require every named member when resume or append was requested.

  ``fresh`` is the only operation allowed to proceed without an existing
  checkpoint.  The filesystem predicate is supplied by the caller so this
  module stays importable in small CPU-only checks without acquiring a path or
  generator dependency.

  Arguments:
    operation = ``fresh``, ``resume``, or ``append``.
    members = ordered paths that form the current checkpoint census.
    is_file = callable returning whether one path is an existing file.

  Returns:
    the ordered member tuple.  Returning the census makes it straightforward
    for later manifest work to consume the exact same list.

  Raises:
    CheckpointLoadError when a requested checkpoint member is missing.
    ValueError when the operation is not one of the normalized operations.
  """
  if operation not in ("fresh", "resume", "append"):
    raise ValueError("Unknown normalized generator operation: "
                     + repr(operation))

  checkpoint_members = tuple(members)
  if operation == "fresh":
    return checkpoint_members

  missing = [path for path in checkpoint_members if not is_file(path)]
  if missing:
    raise CheckpointLoadError(
      "Cannot " + operation + " the requested dataset because checkpoint "
      "members are missing: " + ", ".join(str(path) for path in missing)
      + ". No existing dataset file was changed.")
  return checkpoint_members


def load_checkpoint_or_refuse(operation, loader):
  """Run one checkpoint loader without converting failure into a fresh run.

  A historical broad exception handler treated every missing, truncated, or
  shape-incompatible checkpoint as if the user had requested fresh generation.
  This boundary keeps intent explicit: only ``fresh`` may produce the false
  ``not loaded`` result; resume and append either return ``True`` or raise.
  """
  try:
    loaded = loader()
  except Exception as exc:
    if operation == "fresh":
      raise
    raise CheckpointLoadError(
      "Cannot " + operation + " the requested dataset because its checkpoint "
      "could not be validated: " + str(exc)
      + ". No existing dataset file was changed.") from exc

  if operation == "fresh":
    if loaded:
      raise CheckpointLoadError(
        "Fresh generation unexpectedly loaded an existing checkpoint.")
    return False
  if loaded is not True:
    raise CheckpointLoadError(
      "Cannot " + operation + " the requested dataset because the checkpoint "
      "loader did not confirm a complete checkpoint. No existing dataset "
      "file was changed.")
  return True

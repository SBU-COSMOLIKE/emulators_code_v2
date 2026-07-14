"""The scientific identity a dataset and an emulator are born under.

A trained emulator answers questions about one cosmology, over one region of
one parameter space. Nothing in a saved emulator used to record either fact.
The file kept the names of the parameters that were sampled and nothing about
the world those names lived in: not the parameters held fixed while they
varied, not the interval each was allowed to range over, not which dataset the
weights were fitted to. A consumer could therefore hand a w-varying emulator to
a cosmological-constant likelihood, or ask an emulator trained on 0.1 < omegam
< 0.5 about omegam = 0.7, and get a confident number back with no warning.

This module is the record that closes that hole. It defines two blocks of
persisted truth and the laws that read them:

  fixed_facts    the facts that must be EQUAL. The cosmology the dataset was
                 generated under: what was held fixed and at what value, which
                 neutrino convention, which dark-energy law, what the spectra
                 are measured in. Two artifacts that disagree on any one of
                 these describe different universes and may not be combined.

  input_domain   the facts that legally INTERSECT. The parameters that were
                 sampled, in their canonical order, with the interval each was
                 drawn from. Two artifacts asked about one point need only both
                 contain that point, so the served region is the overlap of the
                 two, not their union.

The blocks are siblings, never one block, because those two laws differ in
kind: an equality law and an overlap law inside a single block would need a
per-key table of exceptions saying which keys are compared how, and that table
is the ad-hoc mechanism this design exists to avoid.

The facts are born in the generator, must survive training, and are read at
inference:

    the resolved Cobaya model                 the FACT, not the YAML request
          |  the generator writes, once, at publication
          v
    <paramsf>.facts.yaml                      the producer sidecar (this module
          |                                   writes and parses it)
          |  COPIED VERBATIM by the training loader, never re-derived
          v
    the emulator .h5: fixed_facts/ + input_domain/ groups, plus the sidecar
          |            text itself, stored beside them
          |  read once, by one reader, refused loudly if absent
          v
    the emulator's consumers                  the comparison laws execute there

  (legend: <paramsf> = the generator's parameter-file stem, the one that
   already carries .1.txt / .paramnames / .ranges / .covmat; "verbatim" =
   the sidecar's own text, copied without being regenerated, so that training
   never becomes a second author of a scientific fact.)

Training copies the sidecar rather than rebuilding it from its own view of the
run. A derived copy is a second author, and two authors of one fact are how the
two halves of that fact drift apart. The artifact therefore carries BOTH the
sidecar text and the parsed blocks, and the reader checks them against each
other in both directions: the blocks must be exactly what parsing the stored
text produces. A block edited after the fact, or a text swapped under blocks
that no longer match it, is refused rather than believed.

PS: a "sidecar" is a small companion file written next to a data file and
sharing its name stem, the way GetDist expects a chain's .paramnames to sit
beside its .1.txt. "Sampled" parameters are the ones the generator drew values
for; "fixed" parameters are the ones it held at a constant value while
sampling. The "support" of a sampled parameter is the interval it was drawn
from. "Shortest-roundtrip decimal" is the shortest decimal string that reads
back as exactly the same 32-bit float it was written from.

Spec and rulings: notes/gates-and-board.md.
"""
import hashlib


# The version of the two-block schema as a whole, carried by the emulator .h5
# as its schema_version root attribute. A file that does not announce a version
# this module knows is refused, never guessed at: the reader cannot tell a file
# written before a key existed from a file whose writer forgot that key, and
# only one of those two is safe to read.
SCHEMA_VERSION = 3

# Each block also carries its own grammar version. The sidecar is a standalone
# file that outlives the artifact it was copied into, so it must be able to say
# what grammar it is written in without the artifact's help; and the copy is
# verbatim, so a key cannot be stripped on the way in.
FIXED_FACTS_BLOCK_VERSION  = 1
INPUT_DOMAIN_BLOCK_VERSION = 1

# The (schema, fixed_facts grammar, input_domain grammar) triples this module
# knows how to read. Any other combination is refused. Versions are values like
# any other, and the house rule is that a value the file does not carry is
# never supplied from a code default.
KNOWN_VERSIONS = ((SCHEMA_VERSION,
                   FIXED_FACTS_BLOCK_VERSION,
                   INPUT_DOMAIN_BLOCK_VERSION),)

# Every key each block must carry. Every one is REQUIRED. A fact that does not
# apply to a family is written "n/a" explicitly and never omitted, because
# "this family has no such fact" and "the writer forgot this fact" are
# different statements and only the first is safe to read.
FIXED_FACTS_KEYS = ("block_version",
                    "dataset_id",
                    "generator",
                    "family",
                    "cosmology_fixed",
                    "neutrino_convention",
                    "flat_only",
                    "dark_energy_law",
                    "dark_energy_inputs",
                    "cl_units",
                    "base_identity",
                    "param_dtype",
                    "decimal_policy")

INPUT_DOMAIN_KEYS = ("block_version",
                     "source",
                     "constraint",
                     "names",
                     "requested",
                     "resolved")

# The cosmology coordinates a fixed-facts block reports on. A coordinate the
# run SAMPLED is dropped from this roster rather than pinned, because it is
# validated through the input domain instead; a coordinate the model cannot
# resolve is reported "n/a".
COSMOLOGY_FIXED_KEYS = ("mnu",
                        "w",
                        "wa",
                        "omk",
                        "TCMB",
                        "nnu")

# The value written for a fact that does not apply, or that the resolved model
# cannot supply. It is a value, not an absence.
NOT_APPLICABLE = "n/a"

# How a parameter value is turned into text, everywhere it is written as text.
# The shortest decimal string that reads back as exactly the same float32 is
# repr(numpy.float32(x)). It is exact, it is legible to a cosmologist reading a
# support interval, and GetDist parses it unchanged.
PARAM_DTYPE    = "float32"
DECIMAL_POLICY = "shortest-roundtrip"

# The sidecar's file extension, appended to the generator's parameter-file stem.
SIDECAR_SUFFIX = ".facts.yaml"

# Where the emulator .h5 keeps each half of the record.
FIXED_FACTS_GROUP  = "fixed_facts"
INPUT_DOMAIN_GROUP = "input_domain"
SIDECAR_DATASET    = "facts_sidecar_yaml"

# What a refusal tells the reader to do about a file the current schema cannot
# honour. Every refusal in this module ends with it: a message that says a file
# is incompatible and stops is not a refusal, it is a shrug.
MIGRATION = ("Re-generate the dataset so the producer writes its "
             + SIDECAR_SUFFIX + " sidecar, then retrain and re-save the "
             "emulator; the saved run then carries the cosmology it was born "
             "under and the domain it may be asked about. Files saved before "
             "this record existed cannot be upgraded in place, because the "
             "facts they would need were never written down.")


def chain_digest(chain_path):
  """Digest the published chain file, so a dataset can prove which run it is.

  Two independent generator runs can agree on every fixed fact and every
  bound and still be different datasets: same cosmology, same priors, a
  different seed. Comparing the facts cannot tell them apart, and a pair of
  emulators trained on two such runs must not be served together. The chain's
  own bytes can tell them apart, because they are the drawn sample: they are
  unique to the run by construction, and two emulators trained off ONE dump
  share them exactly.

  The digest is taken from the chain rather than from the sidecar because the
  sidecar records only facts and bounds, which a re-run with a fresh seed
  reproduces byte for byte.

  Arguments:
    chain_path = path to the published chain file (<paramsf>.1.txt).

  Returns:
    the digest as the string "sha256:<hex>".
  """
  digest = hashlib.sha256()
  with open(chain_path, "rb") as f:
    while True:
      chunk = f.read(1 << 20)
      if not chunk:
        break
      digest.update(chunk)
  return "sha256:" + digest.hexdigest()


def _plain_fact(value):
  """Reduce one value read off the resolved model to a plain, storable fact.

  The model hands its values back in whatever type cobaya, CAMB, or the YAML
  parser produced: a Python float, a numpy scalar, an integer, a string, a
  boolean. The record is written as YAML and later copied into an HDF5 file,
  and both of those store plain values, so a fact is reduced once here rather
  than at each of the places that write it.

  A value that is neither a number, a string, nor a boolean is recorded as the
  text the model would print for it. It is still recorded: a fact the writer
  dropped and a fact the family does not have read the same way on the way back
  in, and only one of the two is safe to read.

  Arguments:
    value = one value as the resolved model handed it back.

  Returns:
    the plain bool, float, or str the record stores.
  """
  # booleans are tested before numbers on purpose: in Python True equals 1, so a
  # flag tested as a number would be stored as the float 1.0 and read back as a
  # number that was never a flag.
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    return value
  try:
    return float(value)
  except (TypeError, ValueError):
    return repr(value)


def _theory_components(model):
  """List the components of the resolved model that carry theory settings.

  The record's second source of a fixed fact is the theory block's extra_args:
  the settings a run hands the Boltzmann code directly (the neutrino splitting,
  the radiation temperature, the effective number of species) rather than
  through the params block. Those settings live on the component object cobaya
  built, so they are read from the model, never from the YAML.

  Two ways in are tried, because a cobaya that does not expose one of them must
  leave a fact unresolved rather than kill a run whose data vectors are already
  computed: the model's own theory collection, and the component walk the
  per-sample drivers already use (dataset_generator_cmb.py finds its Boltzmann
  code that way).

  Arguments:
    model = the resolved Cobaya model. It is duck-typed, never imported: this
            module reads the surfaces named below and nothing else, so it stays
            free of cobaya just as it stays free of torch.

  Returns:
    the list of components that carry an extra_args mapping, possibly empty.
  """
  found = []
  try:
    theory = model.theory
    for name in theory:
      found.append(theory[name])
  except Exception:
    found = []
  if len(found) == 0:
    try:
      for component, _ in model._component_order.items():
        found.append(component)
    except Exception:
      found = []

  components = []
  for component in found:
    extra = getattr(component, "extra_args", None)
    if isinstance(extra, dict):
      components.append(component)
  return components


def resolved_constants(model):
  """Read every value the resolved Cobaya model pins to a constant.

  The YAML is the request; the model is the fact. A default the YAML left
  unstated has been materialized by the time get_model returns, and it is that
  materialized value the dataset was generated under. Two sources are read, in
  this order:

    the params block   parameterization.constant_params(): every parameter the
                       run wrote as a number (or as value: <number>). A
                       parameter given as a function of other parameters is not
                       a constant, and is deliberately absent here: its value
                       changes with the sample, so it is not a fixed fact.
    the theory block   each theory component's extra_args: the settings the run
                       hands the Boltzmann code directly.

  The params block wins a name both blocks state, because it is the model's own
  parameterization of the cosmology. Between two theory components that state
  one name (a configuration nothing in this program produces), the first
  component the model lists wins.

  Every lookup is wrapped: a cobaya that does not expose one of these surfaces
  leaves the facts it would have supplied unresolved, and they are published as
  "n/a" rather than crashing a run whose data vectors are already computed.

  There is one reader, and it lives here rather than in the generator, because
  the generator is not its only caller. The producer reads the model to WRITE
  the record; each cobaya adapter reads the model to CHECK an artifact's record
  against the cosmology the chain is sampling. Those two must read the model the
  same way, down to which block wins a name both blocks state. A second copy of
  this function would be a second author of the same scientific fact, and two
  authors of one fact are how the two halves of that fact drift apart.

  Arguments:
    model = the resolved Cobaya model (cobaya.model.get_model's return value),
            duck-typed rather than imported.

  Returns:
    a mapping of name to plain value, holding every constant the model states.
    It is a superset of the coordinates the record reports on; the caller reads
    the names it needs.
  """
  pinned = {}
  for component in _theory_components(model=model):
    extra = component.extra_args
    for key in extra:
      if key not in pinned:
        pinned[key] = _plain_fact(value=extra[key])

  try:
    constants = model.parameterization.constant_params()
  except Exception:
    constants = {}
  for key in constants:
    pinned[key] = _plain_fact(value=constants[key])
  return pinned


def format_value(value):
  """Write one parameter value as text, under the one decimal policy.

  Every place a parameter value becomes text (the sidecar's bounds, the chain,
  the GetDist .ranges view) writes it the same way, so the same number never
  reads back as two different numbers depending on which file it was read from.

  A bound that was never declared passes through as the "n/a" that says so. It
  is not a number, and turning it into one would invent the very fact it exists
  to report the absence of.

  Arguments:
    value = the number to write, or "n/a" for a bound that was never declared.

  Returns:
    the shortest decimal string that reads back as exactly the same float32,
    or "n/a" unchanged.
  """
  # numpy is imported here rather than at module scope so that the pure schema
  # laws below stay importable by a reader that has no numpy.
  import numpy as np
  if value == NOT_APPLICABLE:
    return NOT_APPLICABLE
  return repr(np.float32(value))


def build_sidecar(dataset_id,
                  generator,
                  family,
                  cosmology_fixed,
                  neutrino_convention,
                  flat_only,
                  dark_energy_law,
                  dark_energy_inputs,
                  cl_units,
                  base_identity,
                  names,
                  requested,
                  resolved,
                  source="declared-prior",
                  constraint="box"):
  """Compose the producer sidecar's text: the two blocks, in block-style YAML.

  Called by the generator at publication, once, with facts read from the
  RESOLVED Cobaya model rather than from the YAML that requested it. The YAML
  is the request; the model is the fact. A default the YAML left unstated has
  been materialized by the time the model exists, and it is the materialized
  value the dataset was actually generated under.

  Arguments:
    dataset_id          = the published chain's digest, from chain_digest.
    generator           = the generator that produced the dataset, by name.
    family              = the output family the dataset feeds.
    cosmology_fixed     = mapping of every cosmology coordinate this run held
                          fixed, to its resolved value; a coordinate the model
                          cannot supply maps to "n/a". A SAMPLED coordinate
                          must not appear here at all.
    neutrino_convention = how the neutrino masses are split, by name.
    flat_only           = True when the run admits no spatial curvature.
    dark_energy_law     = the equation-of-state law, by name.
    dark_energy_inputs  = the parameter names that law consumes.
    cl_units            = the units the angular power spectra are measured in,
                          or "n/a" for a family that has no spectra.
    base_identity       = the frozen base model the dataset was built on top
                          of, by name and version, or "n/a" when there is none.
    names               = the sampled parameters, in the canonical order the
                          generator declared. This list is the authority on
                          that order.
    requested           = mapping name -> (low, high), the support as the prior
                          declared it.
    resolved            = mapping name -> (low, high), the support the sampler
                          actually drew from, after the endpoint stretch and
                          the accuracy margin.
    source              = where the support came from. The generator declares
                          "declared-prior": the bounds are the prior's, never
                          the smallest box the drawn points happened to fall
                          in, because the support is a contract and not an
                          observation. A test double declares "synthetic".
    constraint          = the shape of the support. "box" is a per-parameter
                          interval; a test double that declares no support at
                          all says "undeclared".

  Returns:
    the sidecar's text, ready to write to <paramsf>.facts.yaml.

  Raises:
    ValueError when a coordinate is both sampled and fixed, when a required
    fact is missing, or when the two supports do not cover exactly the
    sampled names.
  """
  import yaml

  facts = {"block_version":       FIXED_FACTS_BLOCK_VERSION,
           "dataset_id":          dataset_id,
           "generator":           generator,
           "family":              family,
           "cosmology_fixed":     dict(cosmology_fixed),
           "neutrino_convention": neutrino_convention,
           "flat_only":           flat_only,
           "dark_energy_law":     dark_energy_law,
           "dark_energy_inputs":  list(dark_energy_inputs),
           "cl_units":            cl_units,
           "base_identity":       base_identity,
           "param_dtype":         PARAM_DTYPE,
           "decimal_policy":      DECIMAL_POLICY}

  # the bounds are written under the one decimal policy, so the number a
  # cosmologist reads out of the sidecar is the number the sampler drew from.
  req_text = {}
  res_text = {}
  for name in names:
    if name not in requested:
      raise ValueError(
        "the sampled parameter " + repr(name) + " has no requested support; "
        "every sampled parameter must declare the interval its prior asked "
        "for. " + MIGRATION)
    if name not in resolved:
      raise ValueError(
        "the sampled parameter " + repr(name) + " has no resolved support; "
        "every sampled parameter must declare the interval it was actually "
        "drawn from. " + MIGRATION)
    lo_req, hi_req = requested[name]
    lo_res, hi_res = resolved[name]
    req_text[name] = [format_value(lo_req), format_value(hi_req)]
    res_text[name] = [format_value(lo_res), format_value(hi_res)]

  domain = {"block_version": INPUT_DOMAIN_BLOCK_VERSION,
            "source":        source,
            "constraint":    constraint,
            "names":         list(names),
            "requested":     req_text,
            "resolved":      res_text}

  blocks = {FIXED_FACTS_GROUP:  facts,
            INPUT_DOMAIN_GROUP: domain}
  # validate before publishing, so a sidecar that would be refused on the way
  # in is never written on the way out.
  validate(blocks=blocks, where="the sidecar being written")
  return yaml.safe_dump(blocks,
                        default_flow_style=False,
                        sort_keys=False)


def synthetic_sidecar(names, label, family=NOT_APPLICABLE, support=None):
  """Compose the record for an emulator with no producer dataset behind it.

  The gates build emulators in memory, out of hand-made geometries and a few
  hundred rows of noise, to prove that saving and rebuilding one is faithful.
  Such an emulator was generated by nobody, describes no cosmology, and is
  valid over no region: it is a test double, not a prediction. It still has to
  say so, because the alternative is a file that carries no record at all, and
  a file with no record is exactly what this whole design exists to refuse.

  So a test double declares itself one. Its generator is "synthetic" and every
  cosmological fact reads "n/a". A consumer comparing it against a real model
  finds facts that do not match and refuses it, which is the correct answer: a
  test double must never be served to a likelihood.

  What the double says about its SUPPORT depends on what the double is for, and
  the two answers are both honest:

    support=None   the double declares no support at all. Its bounds are not
                   wide, they are absent, and every prediction asked of it is
                   refused. This is the double that exists to be round-tripped
                   through a file and compared byte for byte, never asked a
                   question.

    support given  the double declares the box it stands for, and may be asked
                   about a point inside it. A gate that PREDICTS through a
                   double is standing that double in for a real emulator, and a
                   real emulator was drawn from an interval. The bounds are
                   written by format_value, the same decimal policy the
                   generator publishes under, because a support written by any
                   other hand would be a second author of the interval.

  The identity is derived from the caller's label, so two doubles built for
  different purposes carry different identities and a pair built to match on
  purpose carries one identity. It is a real digest, of the label rather than
  of a chain, and the "synthetic" generator says as much.

  Arguments:
    names   = the sampled parameter names, in the order the emulator's input
              geometry holds them. They must equal that geometry's names.
    label   = what this double is for. Two doubles with the same label carry the
              same identity; two with different labels do not.
    family  = the output family the double stands in for, when it stands in for
              one; "n/a" otherwise.
    support = the interval each sampled name stands for, as a mapping
              name -> (low, high) of numbers, or None for a double that
              declares no support and refuses every point.

  Returns:
    the sidecar's text, as build_sidecar returns it.

  Raises:
    ValueError when a declared support does not cover exactly the sampled names
    (build_sidecar's law, unchanged: a support is a per-name contract).
  """
  fixed = {}
  for key in COSMOLOGY_FIXED_KEYS:
    if key not in names:
      fixed[key] = NOT_APPLICABLE

  if support is None:
    bounds     = _undeclared_support(names=names)
    constraint = "undeclared"
  else:
    bounds     = dict(support)
    constraint = "box"

  digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
  text = build_sidecar(dataset_id="sha256:" + digest,
                       generator="synthetic",
                       family=family,
                       cosmology_fixed=fixed,
                       neutrino_convention=NOT_APPLICABLE,
                       flat_only=False,
                       dark_energy_law=NOT_APPLICABLE,
                       dark_energy_inputs=[],
                       cl_units=NOT_APPLICABLE,
                       base_identity=NOT_APPLICABLE,
                       names=names,
                       requested=bounds,
                       resolved=bounds,
                       source="synthetic",
                       constraint=constraint)
  return text


def _undeclared_support(names):
  """State, for every sampled name, that no support was declared for it.

  A test double has no prior, so it has no interval. Writing the widest
  possible interval would be a lie in the direction that never refuses, and
  omitting the key would be the absence this schema forbids. The support is
  therefore recorded as a pair of "n/a" values, which is a declaration and
  reads as one.

  Arguments:
    names = the sampled parameter names.

  Returns:
    a mapping name -> ("n/a", "n/a").
  """
  support = {}
  for name in names:
    support[name] = (NOT_APPLICABLE, NOT_APPLICABLE)
  return support


def parse_sidecar(text, where):
  """Read a sidecar's text back into its two blocks, and validate them.

  Arguments:
    text  = the sidecar's text, as written by build_sidecar.
    where = what is being read, named in any refusal (a path, usually).

  Returns:
    a mapping with exactly two keys, "fixed_facts" and "input_domain", each
    the block's own mapping.

  Raises:
    ValueError naming the first law the text breaks.
  """
  import yaml

  try:
    blocks = yaml.safe_load(text)
  except yaml.YAMLError as exc:
    raise ValueError(
      where + " does not parse as YAML: " + str(exc) + " " + MIGRATION)
  if not isinstance(blocks, dict):
    raise ValueError(
      where + " does not hold the two blocks of the scientific record; it "
      "parsed as " + type(blocks).__name__ + ". " + MIGRATION)
  validate(blocks=blocks, where=where)
  return blocks


def validate(blocks, where, schema_version=SCHEMA_VERSION):
  """Enforce every law the two blocks must satisfy, wherever they are read.

  One validator, called by the producer before it writes, by the parser after
  it reads, and by the artifact reader on both of its paths, so a block cannot
  be refused on one path and accepted on another.

  Arguments:
    blocks         = the mapping of the two blocks, keyed "fixed_facts" and
                     "input_domain"; each block is its own mapping, with the
                     keys FIXED_FACTS_KEYS and INPUT_DOMAIN_KEYS respectively.
    where          = what is being validated, named in any refusal.
    schema_version = the schema version the blocks arrived under. The producer
                     leaves this at the current version, because a block it is
                     about to write is by definition written in the current
                     grammar; the artifact reader passes the version the FILE
                     announced, which is the whole point of announcing it.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError naming the first law the blocks break, and what to do about it.
  """
  for group in (FIXED_FACTS_GROUP, INPUT_DOMAIN_GROUP):
    if group not in blocks:
      raise ValueError(
        where + " is missing its " + group + " record, so it cannot say "
        + ("which cosmology it was generated under"
           if group == FIXED_FACTS_GROUP
           else "which parameter region it may be asked about")
        + ". " + MIGRATION)
    if not isinstance(blocks[group], dict):
      raise ValueError(
        where + ": the " + group + " record is not a mapping of facts; it "
        "read as " + type(blocks[group]).__name__ + ". " + MIGRATION)

  facts  = blocks[FIXED_FACTS_GROUP]
  domain = blocks[INPUT_DOMAIN_GROUP]

  for key in FIXED_FACTS_KEYS:
    if key not in facts:
      raise ValueError(
        where + ": the fixed_facts record is missing " + repr(key) + ". "
        "Every fact is required; one that does not apply is written "
        + repr(NOT_APPLICABLE) + " rather than left out, so a fact the writer "
        "forgot cannot read as a fact the family does not have. " + MIGRATION)
  for key in INPUT_DOMAIN_KEYS:
    if key not in domain:
      raise ValueError(
        where + ": the input_domain record is missing " + repr(key) + ". "
        "Every key is required. " + MIGRATION)

  versions = (schema_version,
              facts["block_version"],
              domain["block_version"])
  if versions not in KNOWN_VERSIONS:
    raise ValueError(
      where + " is written in a grammar this code does not know "
      "(schema_version=" + repr(schema_version)
      + ", fixed_facts block_version=" + repr(facts["block_version"])
      + ", input_domain block_version=" + repr(domain["block_version"])
      + "); the versions it knows are " + repr(KNOWN_VERSIONS) + ". A record "
      "whose grammar is unknown is refused rather than guessed at. "
      + MIGRATION)

  names = domain["names"]
  if not isinstance(names, list) or len(names) == 0:
    raise ValueError(
      where + ": input_domain names must be the non-empty ordered list of "
      "sampled parameters; it read as " + repr(names) + ". " + MIGRATION)
  seen = set()
  for name in names:
    if name in seen:
      raise ValueError(
        where + ": input_domain names repeats " + repr(name) + "; the "
        "canonical order cannot name one parameter twice. " + MIGRATION)
    seen.add(name)

  fixed = facts["cosmology_fixed"]
  if not isinstance(fixed, dict):
    raise ValueError(
      where + ": cosmology_fixed must be a mapping of the coordinates held "
      "fixed to their values; it read as " + type(fixed).__name__ + ". "
      + MIGRATION)

  # A coordinate cannot be both sampled and fixed. If it were, the two records
  # would disagree about the same number: the fixed block would pin it, and the
  # domain block would let it range. The file would then answer the question
  # "what was omegam?" two different ways depending on which half was read.
  for name in names:
    if name in fixed:
      raise ValueError(
        where + ": " + repr(name) + " is both sampled and held fixed (fixed "
        "at " + repr(fixed[name]) + ", and sampled over "
        + repr(domain["resolved"].get(name)) + "). A coordinate is one or the "
        "other. A sampled coordinate is validated through the input domain "
        "and must not also be pinned in cosmology_fixed. " + MIGRATION)

  for key in ("requested", "resolved"):
    support = domain[key]
    if not isinstance(support, dict):
      raise ValueError(
        where + ": the " + key + " support must be a mapping of parameter "
        "name to its interval; it read as " + type(support).__name__ + ". "
        + MIGRATION)
    for name in names:
      if name not in support:
        raise ValueError(
          where + ": the " + key + " support does not cover the sampled "
          "parameter " + repr(name) + ". " + MIGRATION)
    for name in support:
      if name not in seen:
        raise ValueError(
          where + ": the " + key + " support covers " + repr(name) + ", which "
          "is not a sampled parameter. The support describes the sampled "
          "coordinates and nothing else. " + MIGRATION)


def check_names_match(geometry_names, blocks, where):
  """Prove the whitening geometry and the record agree on the parameter order.

  The generator declares the canonical order of the sampled parameters, and the
  record carries that order. The emulator's input geometry carries its own copy
  of the same names, because that is the order its whitening matrices were
  built in. If the two ever disagree, the emulator pairs each incoming value
  with the wrong parameter's column: every prediction is then confidently
  wrong, and nothing about the numbers looks unusual.

  Nothing enforced this before. The two lists were both present, in one file,
  and never compared.

  Arguments:
    geometry_names = the input geometry's names, in its own order.
    blocks         = the two blocks, as validate accepts them.
    where          = the file's identity, named in any refusal.

  Returns:
    None. The function is called for its refusal.

  Raises:
    ValueError when the two orders differ, naming both.
  """
  declared = blocks[INPUT_DOMAIN_GROUP]["names"]
  geometry = list(geometry_names)
  if geometry != declared:
    raise ValueError(
      where + ": the emulator's input geometry and its record disagree about "
      "the sampled parameters, so every value handed in would be whitened "
      "against the wrong parameter's column and every prediction would be "
      "wrong without looking wrong.\n"
      "  the geometry holds: " + repr(geometry) + "\n"
      "  the record declares: " + repr(declared) + "\n"
      "The order is part of the record, not an incidental detail of it. "
      + MIGRATION)


def _plain(value):
  """Coerce one value read back out of HDF5 into the Python value it was.

  HDF5 has no Python types: a bool comes back as a numpy bool, a string may
  come back as bytes, an int as a numpy integer. The two-way check below
  compares the blocks stored in the file against the blocks parsed from the
  text stored beside them, and it must compare VALUES, not the incidental
  types HDF5 chose to store them in.

  Booleans are tested before integers on purpose: in Python True == 1, so an
  integer 1 that slipped in where a boolean belongs would compare equal to
  True and the mismatch would go unseen.

  Arguments:
    value = one value as h5py handed it back.

  Returns:
    the plain Python bool, int, float, str, or list of those.
  """
  import numpy as np
  if isinstance(value, (bool, np.bool_)):
    return bool(value)
  if isinstance(value, bytes):
    return value.decode()
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.floating):
    return float(value)
  if isinstance(value, (list, tuple, np.ndarray)):
    out = []
    for item in value:
      out.append(_plain(item))
    return out
  return value


def _write_block(group, block):
  """Write one block's facts into an open HDF5 group.

  Scalars become attributes, ordered name lists become string datasets (an
  attribute would not promise to keep their order, and the order of the
  sampled names is itself one of the facts), and a mapping of facts becomes a
  subgroup.

  Arguments:
    group = the open h5py group to write into.
    block = the block's mapping of facts.

  Returns:
    None.
  """
  import h5py
  import numpy as np

  str_dt = h5py.string_dtype(encoding="utf-8")
  for key in block:
    value = block[key]
    if isinstance(value, dict):
      sub = group.create_group(key)
      for name in value:
        item = value[name]
        if isinstance(item, list):
          group_data = np.asarray(item, dtype=object)
          sub.create_dataset(name, data=group_data, dtype=str_dt)
        else:
          sub.attrs[name] = item
    elif isinstance(value, list):
      group.create_dataset(key,
                           data=np.asarray(value, dtype=object),
                           dtype=str_dt)
    else:
      group.attrs[key] = value


def _read_block(group):
  """Read one block's facts back out of an open HDF5 group.

  The exact inverse of _write_block, so that a block written and read back is
  the block that was written.

  Arguments:
    group = the open h5py group to read.

  Returns:
    the block's mapping of facts, in plain Python types.
  """
  import h5py

  block = {}
  for key in group.attrs:
    block[key] = _plain(group.attrs[key])
  for key in group:
    item = group[key]
    if isinstance(item, h5py.Group):
      sub = {}
      for name in item.attrs:
        sub[name] = _plain(item.attrs[name])
      for name in item:
        sub[name] = _plain(item[name][()])
      block[key] = sub
    else:
      block[key] = _plain(item[()])
  return block


def write_h5(f, sidecar_text):
  """Persist the scientific record into an open emulator .h5.

  The blocks written are the blocks PARSED FROM THE SIDECAR TEXT, and the text
  itself is stored beside them. Training therefore cannot become a second
  author of a scientific fact even by accident: it has nothing to author from.
  The file carries the producer's own words and the reading of those words, and
  the reader below checks that they still agree.

  Arguments:
    f            = the open h5py File to write into.
    sidecar_text = the producer sidecar's text, copied verbatim from
                   <paramsf>.facts.yaml.

  Returns:
    the two blocks, as parsed and written.

  Raises:
    ValueError when the sidecar breaks any law in validate.
  """
  import h5py

  blocks = parse_sidecar(text=sidecar_text,
                         where="the producer sidecar being copied into the "
                               "emulator file")
  str_dt = h5py.string_dtype(encoding="utf-8")
  f.create_dataset(SIDECAR_DATASET, data=sidecar_text, dtype=str_dt)
  _write_block(group=f.create_group(FIXED_FACTS_GROUP),
               block=blocks[FIXED_FACTS_GROUP])
  _write_block(group=f.create_group(INPUT_DOMAIN_GROUP),
               block=blocks[INPUT_DOMAIN_GROUP])
  return blocks


def read_h5(f, schema_version, where):
  """Read the scientific record back, and prove it was not rewritten.

  The record is read TWICE and the two readings are checked against each other:
  once from the blocks stored in the file, and once by parsing the producer's
  text stored beside them. They must agree in both directions. A block edited
  after the file was written, a text swapped under blocks that no longer match
  it, or a group quietly dropped, all disagree here and are refused.

  That is what makes "copied verbatim, never re-derived" a checkable statement
  rather than a promise: the file carries the evidence to check it against.

  Arguments:
    f              = the open h5py File to read.
    schema_version = the version the file announced, read by the caller (the
                     one shared reader in results.py owns that attribute).
    where          = the file's identity, named in any refusal.

  Returns:
    the two blocks, in plain Python types, keyed "fixed_facts" and
    "input_domain".

  Raises:
    ValueError when a group or the text is missing, when the record breaks any
    law in validate, or when the stored blocks and the stored text disagree.
  """
  for group in (FIXED_FACTS_GROUP, INPUT_DOMAIN_GROUP):
    if group not in f:
      raise ValueError(
        where + " carries no " + group + " record, so it cannot say "
        + ("which cosmology it was trained under"
           if group == FIXED_FACTS_GROUP
           else "which parameter region it may be asked about")
        + ". It was saved before the emulator recorded the science it was born "
        "under. " + MIGRATION)
  if SIDECAR_DATASET not in f:
    raise ValueError(
      where + " carries the scientific record but not the producer's own text "
      "it was copied from, so the record cannot be checked against its "
      "source. " + MIGRATION)

  stored = {FIXED_FACTS_GROUP:  _read_block(f[FIXED_FACTS_GROUP]),
            INPUT_DOMAIN_GROUP: _read_block(f[INPUT_DOMAIN_GROUP])}
  validate(blocks=stored, where=where, schema_version=schema_version)

  text = _plain(f[SIDECAR_DATASET][()])
  parsed = parse_sidecar(text=text,
                         where=where + " (the producer text it carries)")

  # the two-way check. Both directions, because one direction alone would miss
  # half the ways they can disagree: a fact added to the stored blocks that the
  # text never had, and a fact the text has that the stored blocks dropped.
  for group in (FIXED_FACTS_GROUP, INPUT_DOMAIN_GROUP):
    if stored[group] != parsed[group]:
      raise ValueError(
        where + ": the " + group + " record does not match the producer text "
        "stored beside it, so the emulator's copy of the science was rewritten "
        "somewhere between the generator and this file. The record is refused "
        "rather than believed.\n"
        "  the file's record: " + repr(stored[group]) + "\n"
        "  the producer said: " + repr(parsed[group]) + "\n"
        + MIGRATION)
  return stored


def domain_bounds(blocks, name):
  """Read one sampled parameter's resolved support back as two numbers.

  The sidecar stores the bounds as text, under the one decimal policy, so that
  the file a cosmologist reads and the number the code compares against are the
  same value. This is the one place that turns that text back into numbers.

  Legal on a "box" constraint only. A record that declares no support has no
  interval to read, and its two bounds are the string "n/a"; float("n/a") is
  a crash, not a refusal. compile_support below reads the constraint FIRST and
  never reaches this function on an undeclared record.

  Arguments:
    blocks = the two blocks, as validate accepts them.
    name   = the sampled parameter to read.

  Returns:
    (low, high), the resolved interval, as Python floats.
  """
  low_text, high_text = blocks[INPUT_DOMAIN_GROUP]["resolved"][name]
  return float(low_text), float(high_text)


# ---------------------------------------------------------------------------
# The three comparison laws.
#
# Three different questions, and conflating them is how the consumers of these
# artifacts ended up comparing parameter-axis names and calling it a cosmology
# check:
#
#   VERTICAL    does this artifact belong to the cosmology being sampled?
#               every coordinate the artifact HELD FIXED must equal the value
#               the sampling cosmology resolved for it. An equality law.
#
#   HORIZONTAL  do these two artifacts belong to each other?
#               same dataset, same cosmology, same sampled coordinates. An
#               equality law.
#
#   DOMAIN      may this artifact be asked about this point?
#               the point must lie inside the support the artifact was drawn
#               over. For a pair, the served support is the INTERSECTION of the
#               two, never the union. The only law that may intersect, which is
#               exactly why it cannot live in a block compared by equality.
#
# Every refusal names the artifact's value, the value it was asked about, and
# what to do about it. A message that says "incompatible artifact" and stops is
# not a refusal, it is a shrug.
# ---------------------------------------------------------------------------


def _same_fact(left, right):
  """Compare two persisted facts for equality, without lying about types.

  A fact read back out of YAML or HDF5 is a plain Python value, and two records
  that agree spell it the same way. Numbers are compared as numbers so that an
  integer 0 and a float 0.0 for the same pinned coordinate agree; booleans are
  tested before numbers, because in Python True == 1 and a bool that slipped in
  where a number belongs would otherwise compare equal to it.

  Arguments:
    left, right = the two values to compare.

  Returns:
    True when the two facts say the same thing.
  """
  if isinstance(left, bool) or isinstance(right, bool):
    return isinstance(left, bool) and isinstance(right, bool) and left == right
  if isinstance(left, (int, float)) and isinstance(right, (int, float)):
    return float(left) == float(right)
  if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
    return list(left) == list(right)
  return left == right


def check_vertical(blocks, resolved_model, where):
  """Does this artifact belong to the cosmology being sampled?

  The artifact was generated with some coordinates held fixed: a neutrino mass,
  an equation of state, a curvature. Those are not free parameters of the
  emulator, they are properties of the universe it learned. If the chain now
  being sampled holds any of them at a different value, the emulator is
  answering about a different universe, and it will answer confidently.

  The law reads the coordinates the ARTIFACT pinned and asks the sampling
  cosmology what it says about each. It never reads the emulator's input names:
  a parameter the emulator takes as input is validated by the domain law
  instead, and an artifact that pins nothing pins nothing. Comparing the input
  axes and calling it a cosmology check is the exact confusion this law exists
  to end.

  A synthetic record (a gate's test double) pins every coordinate at "n/a" and
  therefore fails this law against any real model, which is the correct answer:
  a test double must never be served to a likelihood.

  Arguments:
    blocks         = the artifact's two blocks, as validate accepts them.
    resolved_model = the RESOLVED global model as a plain mapping, coordinate
                     name -> value. Resolved, not requested: a default the YAML
                     left unstated has been materialized by the time the model
                     exists, and it is the materialized value the chain is
                     actually being sampled under. The consumer resolves it and
                     hands it in; this module never imports cobaya.
    where          = the artifact's identity, named in any refusal.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError naming the coordinate, the artifact's value, the sampled value,
    and the remediation.
  """
  held = blocks[FIXED_FACTS_GROUP]["cosmology_fixed"]
  for name in sorted(held):
    artifact_value = held[name]
    if name not in resolved_model:
      raise ValueError(
        where + " was generated with " + name + " held fixed at "
        + repr(artifact_value) + ", but the cosmology being sampled does not "
        "say what " + name + " is, so the two cannot be compared and the "
        "emulator cannot be shown to belong here. Resolve " + name + " in the "
        "model being sampled, or serve an emulator that was generated without "
        "it pinned.")
    sampled_value = resolved_model[name]
    if not _same_fact(artifact_value, sampled_value):
      raise ValueError(
        where + " was generated with " + name + " held fixed at "
        + repr(artifact_value) + ", but the cosmology being sampled has "
        + name + " = " + repr(sampled_value) + ". The emulator was never shown "
        "that universe, so every answer it gives here would be confident and "
        "wrong, and nothing about the numbers would look unusual. Serve an "
        "emulator generated under this cosmology, or sample the cosmology this "
        "emulator was generated under.")


def check_horizontal(blocks_a, blocks_b, where_a, where_b):
  """Do these two artifacts belong to each other?

  Two emulators are served together all the time: a Hubble rate beside an
  angular diameter distance, a linear power spectrum beside its nonlinear
  boost, a TT spectrum beside an EE. Each pair is combined into one prediction,
  so each pair must come from ONE dataset and ONE cosmology. Nothing checked
  that. The consumers compared parameter-axis names, which agree between two
  runs that share a prior and differ in every number that matters.

  The law is equality, and it starts with identity. Two artifacts trained off
  one generator dump carry the same dataset_id, because it is the digest of the
  chain they were both fitted to; two independent runs of the same YAML, with
  the same priors and a different seed, agree on every fixed fact and every
  bound and still carry different identities. Comparing the facts alone cannot
  tell those apart. The identity can, and it is compared as an opaque string:
  no parsing, no special case for a synthetic double, equality is equality.

  Arguments:
    blocks_a, blocks_b = the two artifacts' blocks, as validate accepts them.
    where_a, where_b   = the two artifacts' identities, named in any refusal.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError naming the disagreeing fact, both values, and the remediation.
  """
  facts_a = blocks_a[FIXED_FACTS_GROUP]
  facts_b = blocks_b[FIXED_FACTS_GROUP]

  # identity first: the sharpest question, and the one the facts cannot answer.
  if facts_a["dataset_id"] != facts_b["dataset_id"]:
    raise ValueError(
      "these two emulators were fitted to different datasets and may not be "
      "served together:\n"
      "  " + where_a + " was trained on " + repr(facts_a["dataset_id"]) + "\n"
      "  " + where_b + " was trained on " + repr(facts_b["dataset_id"]) + "\n"
      "The identity is the digest of the chain each was fitted to, so two "
      "emulators trained off one generator dump share it and two independent "
      "runs never do — even when the two runs agree on every fixed fact and "
      "every bound, because they still drew different points. Train the pair "
      "off one dump, or serve them separately.")

  for key in FIXED_FACTS_KEYS:
    if key == "dataset_id":
      continue
    if not _same_fact(facts_a[key], facts_b[key]):
      # Name the COORDINATE, not just the block it sits in. cosmology_fixed is
      # a mapping of six coordinates, and printing two six-key dicts side by
      # side and leaving the reader to diff them is not a refusal, it is a
      # puzzle. The one that differs is the one worth naming.
      where_it_differs = key
      said_a = facts_a[key]
      said_b = facts_b[key]
      if isinstance(said_a, dict) and isinstance(said_b, dict):
        for name in sorted(set(said_a) | set(said_b)):
          if not _same_fact(said_a.get(name), said_b.get(name)):
            where_it_differs = key + "[" + repr(name) + "]"
            said_a = said_a.get(name)
            said_b = said_b.get(name)
            break
      raise ValueError(
        "these two emulators describe different universes and may not be "
        "served together: they disagree about " + where_it_differs + ".\n"
        "  " + where_a + " says " + repr(said_a) + "\n"
        "  " + where_b + " says " + repr(said_b) + "\n"
        "A pair is combined into one prediction, so the pair must come from "
        "one cosmology. Regenerate both halves from one generator run.")

  names_a = list(blocks_a[INPUT_DOMAIN_GROUP]["names"])
  names_b = list(blocks_b[INPUT_DOMAIN_GROUP]["names"])
  if names_a != names_b:
    raise ValueError(
      "these two emulators were not sampled over the same coordinates and may "
      "not be served together:\n"
      "  " + where_a + " sampled " + repr(names_a) + "\n"
      "  " + where_b + " sampled " + repr(names_b) + "\n"
      "A coordinate sampled by one half only is not a coordinate the pair can "
      "be asked about: the half that never saw it would answer as though it "
      "were held fixed, and the two halves would be evaluated at different "
      "points in the same prediction. The served coordinates are never the "
      "union of the two. Regenerate both halves from one generator run.")


def compile_support(blocks, where):
  """Read the artifact's support once, into the form a point is compared against.

  The record stores its bounds as TEXT, under the one decimal policy, because
  the file a cosmologist reads and the number the code compares against must be
  the same value. Turning that text back into numbers costs a parse per bound,
  and the domain law now runs on every prediction: parsing six strings on every
  step of a chain, to compare against numbers that cannot change while the chain
  runs, would be paid a million times over for no information.

  So the text is read ONCE, here, and the result is what the law compares
  against afterwards. The compiled form is a plain mapping and carries
  everything a refusal needs to name, so that the comparison and its words stay
  in this module and the caller holds nothing but the parse it was handed.

  Nothing is refused here. A double that declares no support must still LOAD (a
  gate saves one, rebuilds it, and compares it byte for byte); what it may not
  do is answer a question, and that is check_support's refusal, not this one.
  Compiling a record is not asking it anything.

  Arguments:
    blocks = the artifact's two blocks, as validate accepts them.
    where  = the artifact's identity, carried into every refusal the compiled
             support later raises.

  Returns:
    the compiled support: a mapping with the artifact's identity, its
    constraint, its generator, its sampled names in order, and the low/high
    bound of each name as Python floats (empty when the constraint is not a
    box).
  """
  domain   = blocks[INPUT_DOMAIN_GROUP]
  compiled = {"where":      where,
              "constraint": domain["constraint"],
              "generator":  blocks[FIXED_FACTS_GROUP]["generator"],
              "names":      list(domain["names"]),
              "low":        {},
              "high":       {}}
  # the constraint is read FIRST, and only a box has intervals to parse. An
  # undeclared record's bounds are the string "n/a", and float("n/a") is a
  # crash, not a refusal.
  if compiled["constraint"] == "box":
    for name in compiled["names"]:
      low, high = domain_bounds(blocks=blocks, name=name)
      compiled["low"][name]  = low
      compiled["high"][name] = high
  return compiled


def check_support(compiled, point):
  """May this artifact be asked about this point? The law, on a compiled support.

  An emulator interpolates inside the region it was trained over, and outside
  it, it extrapolates: it returns a number of the right shape, with the right
  sign, and no warning. Nothing in a saved emulator used to record the region
  at all, so nothing could refuse. This is the refusal, and it is the ONE author
  of it: check_domain below and predict() in emulator/inference.py both arrive
  here, so a point cannot be refused in one place and served in another, and the
  words a cosmologist reads are the same words whichever door was walked
  through.

  The constraint is read FIRST, because it says whether there is an interval to
  compare against at all. A record that declares no support ("undeclared")
  cannot be asked about any point: its bounds are not wide, they are absent.
  That is the shape of a test double, and a test double must never be served.

  Arguments:
    compiled = the artifact's support, from compile_support.
    point    = the point being asked about, a mapping name -> value. Only the
               sampled coordinates are read; a mapping that carries more (a
               cobaya parameter block, say) is fine.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError naming the coordinate, the artifact's interval, the requested
    value, and the remediation.
  """
  where      = compiled["where"]
  constraint = compiled["constraint"]

  if constraint == "undeclared":
    raise ValueError(
      where + " declares no support: it was generated by "
      + repr(compiled["generator"]) + " and records no "
      "interval for any coordinate, so there is no region it may be asked "
      "about. An emulator with no declared support is a test double, not a "
      "prediction, and it must not be served. Serve an emulator generated by "
      "the dataset generator, which publishes the interval each coordinate was "
      "drawn from.")
  if constraint != "box":
    raise ValueError(
      where + " declares its support as " + repr(constraint) + ", which this "
      "code does not know how to compare a point against; the supports it "
      "knows are 'box' (a per-coordinate interval) and 'undeclared'. A support "
      "whose shape is unknown is refused rather than guessed at. " + MIGRATION)

  # the accept path: one dictionary lookup and two float compares per sampled
  # coordinate, and not one string parsed. This runs on every prediction.
  for name in compiled["names"]:
    if name not in point:
      raise ValueError(
        where + " was trained over " + name + " and cannot be asked about a "
        "point that does not say what " + name + " is. The point carries "
        + repr(sorted(point)) + ". Hand in every coordinate the emulator "
        "sampled.")
    low   = compiled["low"][name]
    high  = compiled["high"][name]
    value = float(point[name])
    if value < low or value > high:
      raise ValueError(
        where + " was trained over " + name + " in ["
        + format_value(low) + ", " + format_value(high) + "] and is being "
        "asked about " + name + " = " + repr(value) + ", which is outside it. "
        "An emulator does not fail outside its training region; it "
        "extrapolates, and it returns a confident number that is wrong. The "
        "region is the contract the dataset was generated under. Sample inside "
        "it, or generate a dataset that covers the region you mean to sample.")


def check_domain(blocks, point, where):
  """May this artifact be asked about this point? The law, from the raw blocks.

  The same law as check_support, for a caller that holds the record and has
  compiled nothing: it compiles the support and applies it, in that order. A
  caller that asks this question more than once (a predictor, a sampler) keeps
  the compiled support instead and calls check_support, so the record's text is
  parsed once rather than once per point.

  Arguments:
    blocks = the artifact's two blocks, as validate accepts them.
    point  = the point being asked about, a mapping name -> value.
    where  = the artifact's identity, named in any refusal.

  Returns:
    None. The function is called for its refusals.

  Raises:
    ValueError naming the coordinate, the artifact's interval, the requested
    value, and the remediation — check_support's refusals, unchanged.
  """
  check_support(compiled=compile_support(blocks=blocks, where=where),
                point=point)


def served_support(blocks_a, blocks_b, where_a, where_b):
  """The support a PAIR of artifacts may be served over: their intersection.

  Two emulators combined into one prediction can only be asked about a point
  both of them were trained over. The served region is therefore the overlap of
  the two, never their union: a point inside one half and outside the other is
  a point where one half is extrapolating, and the combined answer inherits
  that silently. A pair whose regions do not overlap at all has no point it can
  be asked about, and says so rather than serving the empty set.

  This is the one law that intersects, which is why the support cannot live in
  the block that is compared by equality.

  Arguments:
    blocks_a, blocks_b = the two artifacts' blocks, as validate accepts them.
    where_a, where_b   = the two artifacts' identities, named in any refusal.

  Returns:
    a mapping name -> (low, high), the intersected support, as Python floats.

  Raises:
    ValueError when either record declares no box support, or when the two
    supports do not overlap on some coordinate.
  """
  for blocks, where in ((blocks_a, where_a), (blocks_b, where_b)):
    constraint = blocks[INPUT_DOMAIN_GROUP]["constraint"]
    if constraint != "box":
      raise ValueError(
        where + " declares its support as " + repr(constraint) + ", not a "
        "box of per-coordinate intervals, so there is no region to intersect "
        "with the emulator it is served beside. An emulator with no declared "
        "support must not be served.")

  # The two must be sampled over the same coordinates before their regions can
  # be intersected at all: an interval belonging to a coordinate the other half
  # never sampled has nothing to be intersected WITH. check_horizontal refuses
  # such a pair first, but this function is called by name and must not depend
  # on the caller having asked the other law first — unguarded, the loop below
  # would raise KeyError, which is a crash, not a refusal.
  names_a = list(blocks_a[INPUT_DOMAIN_GROUP]["names"])
  names_b = list(blocks_b[INPUT_DOMAIN_GROUP]["names"])
  if names_a != names_b:
    raise ValueError(
      "these two emulators were not sampled over the same coordinates, so "
      "their regions cannot be intersected and the pair has no support:\n"
      "  " + where_a + " sampled " + repr(names_a) + "\n"
      "  " + where_b + " sampled " + repr(names_b) + "\n"
      "Regenerate both halves from one generator run.")

  support = {}
  for name in names_a:
    low_a, high_a = domain_bounds(blocks=blocks_a, name=name)
    low_b, high_b = domain_bounds(blocks=blocks_b, name=name)
    low  = max(low_a, low_b)
    high = min(high_a, high_b)
    if low > high:
      raise ValueError(
        "these two emulators were trained over regions that do not overlap in "
        + name + ", so there is no point the pair can be asked about:\n"
        "  " + where_a + " covers [" + format_value(low_a) + ", "
        + format_value(high_a) + "]\n"
        "  " + where_b + " covers [" + format_value(low_b) + ", "
        + format_value(high_b) + "]\n"
        "The served region of a pair is the overlap of the two, never the "
        "union: a point inside one half only is a point where the other half "
        "extrapolates. Regenerate both halves from one generator run.")
    support[name] = (low, high)
  return support

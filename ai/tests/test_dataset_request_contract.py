"""Focused CPU tests for immutable dataset request identity and members."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from compute_data_vectors.dataset_manifest import (
  DATASET_PROBE_FAMILIES,
  DATASET_PROBE_GENERATORS,
  DATASET_SAMPLING_POLICIES,
  UNIFORM_BOUNDARY_INTERIOR_POLICY,
  build_dataset_member_map,
  build_dataset_request_identity,
  validate_dataset_request_identity,
)
from emulator import fixed_facts
from compute_data_vectors.dataset_publication import (
  begin_dataset_generation,
  canonical_json_bytes,
  derive_dataset_slot,
  load_active_generation,
  publish_dataset_generation,
)
CONFIG_A = "1" * 64
CONFIG_B = "2" * 64
CONTRACT_A = "a" * 64
CONTRACT_B = "b" * 64


def _uniform_identity(**changes):
  values = {
    "dataset_mode": "full",
    "family": "cosmolike",
    "family_variant": "standard",
    "generator": "dataset_generator_lensing",
    "probe": "cs",
    "sampling_mode": "uniform",
    "temperature": 128,
    "boundary_factor": 1.0,
    "max_correlation": None,
    "sampling_algorithm": "uniform-box-v1",
    "seed": 314159,
    "rng_bit_generator": "PCG64",
    "rng_emcee_random": None,
    "rng_policy": "persist-complete-state-v1",
    "boundary_interior_policy": UNIFORM_BOUNDARY_INTERIOR_POLICY,
    "ordered_names": ["omegabh2", "H0", "ns"],
    "configuration_sha256": CONFIG_A,
    "scientific_contract_sha256": CONTRACT_A,
  }
  values.update(changes)
  return build_dataset_request_identity(**values)


def _gaussian_identity(**changes):
  values = {
    "dataset_mode": "full",
    "family": "cmb",
    "family_variant": "standard",
    "generator": "dataset_generator_cmb",
    "probe": "cmblensed",
    "sampling_mode": "gaussian-mcmc",
    "temperature": 64,
    "boundary_factor": 0.9,
    "max_correlation": 0.15,
    "sampling_algorithm": "emcee-de-snooker-v1",
    "seed": 2718,
    "rng_bit_generator": "PCG64",
    "rng_emcee_random": "MT19937",
    "rng_policy": "persist-complete-state-v1",
    "boundary_interior_policy": None,
    "ordered_names": ["omegabh2", "H0", "logA"],
    "configuration_sha256": CONFIG_A,
    "scientific_contract_sha256": CONTRACT_A,
  }
  values.update(changes)
  return build_dataset_request_identity(**values)


def _replace(identity, path, value):
  changed = copy.deepcopy(identity)
  target = changed
  for key in path[:-1]:
    target = target[key]
  target[path[-1]] = value
  return changed


def _contains_key(value, name):
  if isinstance(value, dict):
    return name in value or any(
      _contains_key(child, name) for child in value.values())
  if isinstance(value, list):
    return any(_contains_key(child, name) for child in value)
  return False


def _scientific_blocks(dataset_id, resolved_high="90.9"):
  return {
    fixed_facts.FIXED_FACTS_GROUP: {
      "block_version": fixed_facts.FIXED_FACTS_BLOCK_VERSION,
      "dataset_id": dataset_id,
      "generator": "dataset_generator_lensing",
      "family": "cosmolike",
      "cosmology_fixed": {},
      "neutrino_convention": "n/a",
      "flat_only": True,
      "dark_energy_law": "constant-w",
      "dark_energy_inputs": [],
      "cl_units": "n/a",
      "base_identity": "n/a",
      "param_dtype": fixed_facts.PARAM_DTYPE,
      "decimal_policy": fixed_facts.DECIMAL_POLICY,
    },
    fixed_facts.INPUT_DOMAIN_GROUP: {
      "block_version": fixed_facts.INPUT_DOMAIN_BLOCK_VERSION,
      "source": "declared-prior",
      "constraint": "box",
      "names": ["H0"],
      "requested": {"H0": ["55.0", "91.0"]},
      "resolved": {"H0": ["55.1", resolved_high]},
    },
  }


class DatasetRequestIdentityTests(unittest.TestCase):

  def test_uniform_and_gaussian_records_roundtrip_canonical_json(self):
    for identity in (_uniform_identity(), _gaussian_identity()):
      payload = canonical_json_bytes(identity)
      parsed = json.loads(payload.decode("ascii"))
      self.assertEqual(validate_dataset_request_identity(parsed), parsed)
      self.assertEqual(canonical_json_bytes(parsed), payload)

  def test_publication_roundtrip_preserves_validated_identity_and_members(self):
    facts = b"producer-authored-facts-sidecar\n"
    identity = _uniform_identity()
    members = build_dataset_member_map(
      identity, params_stem="params", dvs_stem="dvs", fail_stem="fail")
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      slot = derive_dataset_slot(
        chains, params_stem=str(chains / "params"),
        dvs_stem=str(chains / "dvs"), fail_stem=str(chains / "fail"),
        dataset_mode=identity["dataset_mode"], family=identity["family"])
      draft = begin_dataset_generation(slot)
      for role, relative in members.items():
        path = draft.member_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = facts if role == "metadata.scientific-facts" \
          else ("member:" + role + "\n").encode("ascii")
        path.write_bytes(payload)
      publish_dataset_generation(
        draft, identity=identity, members=members,
        expected_active_sha256=None)
      loaded = load_active_generation(
        slot, expected_identity=identity, expected_members=members)
      self.assertEqual(loaded.identity, identity)
      self.assertEqual({member.role for member in loaded.members}, set(members))

  def test_each_supported_request_change_changes_canonical_bytes(self):
    original = _uniform_identity()
    controls = [
      _replace(original, ("dataset_mode",), "chain-only"),
      _uniform_identity(family="cmb", probe="cmbunlensed",
                        generator="dataset_generator_cmb"),
      _uniform_identity(probe="ggl"),
      _replace(original, ("sampling", "temperature"), 64),
      _replace(original, ("sampling", "boundary_factor"), 0.75),
      _replace(original, ("sampling", "seed"), 314160),
      _replace(original, ("parameters", "names"),
               ["H0", "omegabh2", "ns"]),
      _replace(original, ("parameters", "configuration_sha256"), CONFIG_B),
      _replace(original, ("scientific_contract_sha256",), CONTRACT_B),
      _gaussian_identity(),
      _uniform_identity(family="grid2d", probe="mps",
                        generator="dataset_generator_mps",
                        family_variant="native"),
      _uniform_identity(family="grid2d", probe="mps",
                        generator="dataset_generator_mps",
                        family_variant="syren-base"),
    ]
    original_bytes = canonical_json_bytes(original)
    for changed in controls:
      with self.subTest(changed=changed):
        validate_dataset_request_identity(changed)
        self.assertNotEqual(canonical_json_bytes(changed), original_bytes)

  def test_temperature_remains_identity_when_support_bytes_match(self):
    low = _uniform_identity(temperature=64)
    high = _uniform_identity(temperature=128)
    self.assertEqual(low["scientific_contract_sha256"],
                     high["scientific_contract_sha256"])
    self.assertNotEqual(canonical_json_bytes(low), canonical_json_bytes(high))

  def test_order_is_preserved_and_duplicate_names_refuse(self):
    names = ["wa", "w", "H0"]
    identity = _uniform_identity(ordered_names=names)
    self.assertEqual(identity["parameters"]["names"], names)
    self.assertNotEqual(
      canonical_json_bytes(identity),
      canonical_json_bytes(_uniform_identity(ordered_names=list(reversed(names)))))
    with self.assertRaisesRegex(ValueError, "repeated"):
      _uniform_identity(ordered_names=["H0", "H0"])
    for wrong in ("wa", ("wa",), b"wa"):
      with self.subTest(wrong=wrong):
        with self.assertRaisesRegex(ValueError, "native list"):
          _uniform_identity(ordered_names=wrong)

  def test_sampling_modes_require_their_exact_policy_fields(self):
    malformed_mode = _uniform_identity()
    malformed_mode["sampling"]["mode"] = []
    with self.assertRaisesRegex(ValueError, "sampling mode"):
      validate_dataset_request_identity(malformed_mode)
    with self.assertRaisesRegex(ValueError, "max_correlation=null"):
      _uniform_identity(max_correlation=0.15)
    for policy in (None, "nextafter-v0", ""):
      with self.subTest(policy=policy):
        with self.assertRaisesRegex(ValueError, "boundary-interior policy"):
          _uniform_identity(boundary_interior_policy=policy)
    with self.assertRaisesRegex(ValueError, "maximum sampling correlation"):
      _gaussian_identity(max_correlation=None)
    with self.assertRaisesRegex(ValueError, "requires.*null"):
      _gaussian_identity(
        boundary_interior_policy=UNIFORM_BOUNDARY_INTERIOR_POLICY)
    with self.assertRaisesRegex(ValueError, "requires algorithm"):
      _uniform_identity(sampling_algorithm="emcee-de-snooker-v1")
    with self.assertRaisesRegex(ValueError, "RNG bit_generator"):
      _uniform_identity(rng_bit_generator="PCG64DXSM")
    with self.assertRaisesRegex(ValueError, "RNG emcee_random"):
      _uniform_identity(rng_emcee_random="MT19937")
    with self.assertRaisesRegex(ValueError, "RNG emcee_random"):
      _gaussian_identity(rng_emcee_random=None)

  def test_native_numeric_and_finite_boundaries_refuse_coercion(self):
    for field, value in (("temperature", True), ("seed", False),
                         ("temperature", 1.0), ("seed", "1")):
      with self.subTest(field=field, value=value):
        with self.assertRaisesRegex(ValueError, "native integer"):
          _uniform_identity(**{field: value})
    for value in (True, 1, float("nan"), float("inf"), -0.0, 1.01):
      with self.subTest(boundary=value):
        with self.assertRaisesRegex(ValueError, "boundary factor"):
          _uniform_identity(boundary_factor=value)
    for value in (True, 1, float("nan"), float("inf"), 0.01, 1.01):
      with self.subTest(correlation=value):
        with self.assertRaisesRegex(ValueError, "correlation"):
          _gaussian_identity(max_correlation=value)
    for field in ("temperature", "seed"):
      with self.subTest(field=field):
        with self.assertRaisesRegex(ValueError, "3402 bits"):
          _uniform_identity(**{field: 1 << 3402})
        with self.assertRaisesRegex(ValueError, "1024 decimal digits"):
          _uniform_identity(**{field: 10 ** 1024})

  def test_schema_keys_digests_family_and_variant_are_exact(self):
    identity = _uniform_identity()
    for path in (("sampling", "temperature"),
                 ("parameters", "configuration_sha256"),
                 ("scientific_contract_sha256",)):
      changed = copy.deepcopy(identity)
      target = changed
      for key in path[:-1]:
        target = target[key]
      if path[-1] in target:
        del target[path[-1]]
      with self.subTest(path=path):
        with self.assertRaisesRegex(ValueError, "fields differ"):
          validate_dataset_request_identity(changed)
    for digest in (None, "A" * 64, "a" * 63, "g" * 64):
      with self.subTest(digest=digest):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
          _uniform_identity(scientific_contract_sha256=digest)
    with self.assertRaisesRegex(ValueError, "belongs to family"):
      _uniform_identity(family="cmb")
    with self.assertRaisesRegex(ValueError, "requires variant"):
      _uniform_identity(family_variant="native")
    with self.assertRaisesRegex(ValueError, "grid2d family variant"):
      _uniform_identity(family="grid2d", probe="mps",
                        family_variant="standard")
    with self.assertRaisesRegex(ValueError, "requires generator"):
      _uniform_identity(generator="dataset_generator_cmb")
    mixed = copy.deepcopy(identity)
    mixed[1] = "not a string key"
    with self.assertRaisesRegex(ValueError, "field names.*strings"):
      validate_dataset_request_identity(mixed)

  def test_transaction_controls_are_not_scientific_identity_fields(self):
    identity = _uniform_identity()
    for field in ("loadchk", "append", "operation", "freqchk", "nparams"):
      changed = copy.deepcopy(identity)
      changed[field] = 1
      with self.subTest(field=field):
        with self.assertRaisesRegex(ValueError, "unknown"):
          validate_dataset_request_identity(changed)
    self.assertNotIn("state", identity)
    self.assertNotIn("row_count", identity)

  def test_scientific_contract_is_append_stable_and_binds_support(self):
    first = _scientific_blocks("sha256:" + "1" * 64)
    appended = _scientific_blocks("sha256:" + "2" * 64)
    changed_support = _scientific_blocks(
      "sha256:" + "2" * 64, resolved_high="90.8")
    digest_a = fixed_facts.scientific_contract_digest(first)
    self.assertEqual(
      digest_a, fixed_facts.scientific_contract_digest(appended))
    digest_b = fixed_facts.scientific_contract_digest(changed_support)
    self.assertNotEqual(digest_a, digest_b)
    identity_a = _uniform_identity(
      ordered_names=["H0"], scientific_contract_sha256=digest_a)
    identity_b = _uniform_identity(
      ordered_names=["H0"], scientific_contract_sha256=digest_b)
    self.assertNotEqual(canonical_json_bytes(identity_a),
                        canonical_json_bytes(identity_b))
    for identity in (identity_a, identity_b):
      self.assertFalse(_contains_key(identity, "requested"))
      self.assertFalse(_contains_key(identity, "resolved"))
    hostile = _scientific_blocks("sha256:" + "3" * 64)
    hostile[fixed_facts.FIXED_FACTS_GROUP]["cosmology_fixed"]["mnu"] = (
      10 ** 1024)
    with self.assertRaisesRegex(ValueError, "1024 decimal digits"):
      fixed_facts.scientific_contract_digest(hostile)
    nonfinite = _scientific_blocks("sha256:" + "3" * 64)
    nonfinite[fixed_facts.FIXED_FACTS_GROUP]["cosmology_fixed"]["mnu"] = (
      float("nan"))
    with self.assertRaisesRegex(ValueError, "non-finite"):
      fixed_facts.scientific_contract_digest(nonfinite)
    oversized = _scientific_blocks("sha256:" + "3" * 64)
    oversized[fixed_facts.FIXED_FACTS_GROUP]["dark_energy_inputs"] = (
      ["w"] * 4097)
    with self.assertRaisesRegex(ValueError, "list longer than 4096"):
      fixed_facts.scientific_contract_digest(oversized)
    extra = _scientific_blocks("sha256:" + "3" * 64)
    extra[fixed_facts.FIXED_FACTS_GROUP]["future_default"] = "unsafe"
    with self.assertRaisesRegex(ValueError, "fields differ"):
      fixed_facts.scientific_contract_digest(extra)


class DatasetMemberMapTests(unittest.TestCase):

  def test_chain_only_has_exact_common_five_for_every_family(self):
    cases = (
      ("cosmolike", "standard", "cs", "dataset_generator_lensing"),
      ("cmb", "standard", "cmblensed", "dataset_generator_cmb"),
      ("grid", "standard", "background", "dataset_generator_background"),
      ("grid2d", "native", "mps", "dataset_generator_mps"),
      ("grid2d", "syren-base", "mps", "dataset_generator_mps"),
    )
    expected = {
      "parameters.chain": "params_chain_only.1.txt",
      "parameters.schema": "params_chain_only.paramnames",
      "parameters.covariance": "params_chain_only.covmat",
      "parameters.ranges": "params_chain_only.ranges",
      "metadata.scientific-facts": "params_chain_only.facts.yaml",
    }
    for family, variant, probe, generator in cases:
      identity = _uniform_identity(
        dataset_mode="chain-only", family=family, family_variant=variant,
        probe=probe, generator=generator)
      with self.subTest(family=family, variant=variant):
        self.assertEqual(
          build_dataset_member_map(
            identity, params_stem="params_chain_only",
            dvs_stem="dvs_chain_only", fail_stem="fail_chain_only"),
          expected)

  def test_full_family_maps_have_exact_roles_paths_and_counts(self):
    cases = (
      (_uniform_identity(), 7, {
        "payload.cosmolike.vector": "dvs.npy",
      }),
      (_uniform_identity(family="cmb", probe="cmblensed",
                         generator="dataset_generator_cmb"), 11, {
        "payload.cmb.tt": "dvs_tt.npy",
        "payload.cmb.te": "dvs_te.npy",
        "payload.cmb.ee": "dvs_ee.npy",
        "payload.cmb.pp": "dvs_pp.npy",
        "axis.cmb.multipole": "dvs_ell.npy",
      }),
      (_uniform_identity(family="grid", probe="background",
                         generator="dataset_generator_background"), 10, {
        "payload.grid.h": "dvs_h.npy",
        "axis.grid.h.redshift": "dvs_h_z.npy",
        "payload.grid.dm": "dvs_dm.npy",
        "axis.grid.dm.redshift": "dvs_dm_z.npy",
      }),
      (_uniform_identity(family="grid2d", family_variant="native", probe="mps",
                         generator="dataset_generator_mps"), 10, {
        "payload.grid2d.pklin": "dvs_pklin.npy",
        "payload.grid2d.boost": "dvs_boost.npy",
        "axis.grid2d.redshift": "dvs_z.npy",
        "axis.grid2d.wavenumber": "dvs_k.npy",
      }),
      (_uniform_identity(family="grid2d", family_variant="syren-base",
                         probe="mps", generator="dataset_generator_mps"), 12, {
        "payload.grid2d.pklin": "dvs_pklin.npy",
        "payload.grid2d.boost": "dvs_boost.npy",
        "axis.grid2d.redshift": "dvs_z.npy",
        "axis.grid2d.wavenumber": "dvs_k.npy",
        "base.grid2d.pklin": "dvs_pklin_base.npy",
        "base.grid2d.boost": "dvs_boost_base.npy",
      }),
    )
    common = {
      "parameters.chain": "params.1.txt",
      "parameters.schema": "params.paramnames",
      "parameters.covariance": "params.covmat",
      "parameters.ranges": "params.ranges",
      "metadata.scientific-facts": "params.facts.yaml",
      "rows.failure-mask": "fail.txt",
    }
    for identity, count, additions in cases:
      members = build_dataset_member_map(
        identity, params_stem="params", dvs_stem="dvs", fail_stem="fail")
      with self.subTest(family=identity["family"],
                        variant=identity["family_variant"]):
        self.assertEqual(len(members), count)
        self.assertEqual(members, dict(common, **additions))

  def test_cmb_axis_and_grid2d_base_pair_are_mandatory_not_optional(self):
    cmb = build_dataset_member_map(
      _uniform_identity(family="cmb", probe="cmbunlensed",
                        generator="dataset_generator_cmb"),
      params_stem="p", dvs_stem="d", fail_stem="f")
    self.assertEqual(cmb["axis.cmb.multipole"], "d_ell.npy")
    self.assertEqual(len([role for role in cmb if role.startswith("axis.cmb")]),
                     1)

    native = build_dataset_member_map(
      _uniform_identity(family="grid2d", family_variant="native", probe="mps",
                        generator="dataset_generator_mps"),
      params_stem="p", dvs_stem="d", fail_stem="f")
    based = build_dataset_member_map(
      _uniform_identity(family="grid2d", family_variant="syren-base",
                        probe="mps", generator="dataset_generator_mps"),
      params_stem="p", dvs_stem="d", fail_stem="f")
    self.assertFalse(any(role.startswith("base.") for role in native))
    self.assertEqual(
      {role for role in based if role.startswith("base.")},
      {"base.grid2d.pklin", "base.grid2d.boost"})

  def test_stems_are_visible_portable_basenames_and_case_distinct(self):
    identity = _uniform_identity()
    for field, value in (("params_stem", "../params"),
                         ("dvs_stem", ".hidden"),
                         ("fail_stem", "fail/path"),
                         ("params_stem", "")):
      kwargs = {"params_stem": "params", "dvs_stem": "dvs",
                "fail_stem": "fail"}
      kwargs[field] = value
      with self.subTest(field=field, value=value):
        with self.assertRaisesRegex(ValueError, "portable"):
          build_dataset_member_map(identity, **kwargs)
    with self.assertRaisesRegex(ValueError, "portable"):
      build_dataset_member_map(
        identity, params_stem="p" * 201, dvs_stem="d", fail_stem="f")
    with self.assertRaisesRegex(ValueError, "case-insensitive"):
      build_dataset_member_map(
        identity, params_stem="DATA", dvs_stem="data", fail_stem="fail")
    for params, fail in (("p", "p.1"), ("P", "p.1")):
      with self.subTest(params=params, fail=fail):
        with self.assertRaisesRegex(ValueError, "member paths collide"):
          build_dataset_member_map(
            identity, params_stem=params, dvs_stem="d", fail_stem=fail)

  def test_probe_generator_and_sampling_registries_are_exact_and_immutable(self):
    self.assertEqual(DATASET_PROBE_FAMILIES, {
      "cs": "cosmolike", "ggl": "cosmolike", "gc": "cosmolike",
      "cmblensed": "cmb", "cmbunlensed": "cmb",
      "background": "grid", "mps": "grid2d",
    })
    self.assertEqual(DATASET_PROBE_GENERATORS, {
      "cs": "dataset_generator_lensing",
      "ggl": "dataset_generator_lensing",
      "gc": "dataset_generator_lensing",
      "cmblensed": "dataset_generator_cmb",
      "cmbunlensed": "dataset_generator_cmb",
      "background": "dataset_generator_background",
      "mps": "dataset_generator_mps",
    })
    self.assertEqual(set(DATASET_SAMPLING_POLICIES),
                     {"uniform", "gaussian-mcmc"})
    self.assertEqual(DATASET_SAMPLING_POLICIES["uniform"]["emcee_random"],
                     None)
    self.assertEqual(
      DATASET_SAMPLING_POLICIES["gaussian-mcmc"]["emcee_random"],
      "MT19937")
    for registry, key in ((DATASET_PROBE_FAMILIES, "cs"),
                          (DATASET_PROBE_GENERATORS, "cs"),
                          (DATASET_SAMPLING_POLICIES, "uniform"),
                          (DATASET_SAMPLING_POLICIES["uniform"], "algorithm")):
      with self.subTest(registry=registry, key=key):
        with self.assertRaises(TypeError):
          registry[key] = "poisoned"


if __name__ == "__main__":
  unittest.main()

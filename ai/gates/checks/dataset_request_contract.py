#!/usr/bin/env python3
"""Prove the pure Unit-8 request identity and family-member contract."""

import copy
import io
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from ai.tests.test_dataset_request_contract import DatasetMemberMapTests
from ai.tests.test_dataset_request_contract import DatasetRequestIdentityTests


MANIFEST_SOURCE = ROOT / "compute_data_vectors" / "dataset_manifest.py"
FIXED_FACTS_SOURCE = ROOT / "emulator" / "fixed_facts.py"
LEG_AIDS = (
  "dataset-request-contract.identity",
  "dataset-request-contract.family-members",
  "dataset-request-contract.mutation-controls",
)

IDENTITY_TESTS = tuple(unittest.defaultTestLoader.getTestCaseNames(
  DatasetRequestIdentityTests))
MEMBER_TESTS = tuple(unittest.defaultTestLoader.getTestCaseNames(
  DatasetMemberMapTests))


def _run_suite(case, methods):
  suite = unittest.TestSuite(case(method) for method in methods)
  stream = io.StringIO()
  result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
  transcript = stream.getvalue().rstrip()
  if transcript:
    print(transcript)
  return result.wasSuccessful() and result.testsRun == len(methods)


def _mutate(source, old, new, label):
  count = source.count(old)
  if count != 1:
    raise AssertionError(
      label + " mutation expected one source match, found " + str(count))
  return source.replace(old, new)


def _load_mutant(source, label, source_path=MANIFEST_SOURCE):
  name = "_dataset_contract_mutant_" + label.replace("-", "_")
  module = types.ModuleType(name)
  module.__file__ = str(source_path)
  sys.modules[name] = module
  try:
    exec(compile(source, str(source_path), "exec"), module.__dict__)
  except Exception:
    sys.modules.pop(name, None)
    raise
  return name, module


def _identity(module, **changes):
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
    "seed": 7,
    "rng_bit_generator": "PCG64",
    "rng_emcee_random": None,
    "rng_policy": "persist-complete-state-v1",
    "boundary_interior_policy": module.UNIFORM_BOUNDARY_INTERIOR_POLICY,
    "ordered_names": ["wa", "H0"],
    "configuration_sha256": "1" * 64,
    "scientific_contract_sha256": "a" * 64,
  }
  values.update(changes)
  return module.build_dataset_request_identity(**values)


def _raises(call):
  try:
    call()
  except Exception:
    return True
  return False


def _scientific_blocks(module, dataset_id):
  return {
    module.FIXED_FACTS_GROUP: {
      "block_version": module.FIXED_FACTS_BLOCK_VERSION,
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
      "param_dtype": module.PARAM_DTYPE,
      "decimal_policy": module.DECIMAL_POLICY,
    },
    module.INPUT_DOMAIN_GROUP: {
      "block_version": module.INPUT_DOMAIN_BLOCK_VERSION,
      "source": "declared-prior",
      "constraint": "box",
      "names": ["H0"],
      "requested": {"H0": ["55.0", "91.0"]},
      "resolved": {"H0": ["55.1", "90.9"]},
    },
  }


def _mutation_controls():
  source = MANIFEST_SOURCE.read_text(encoding="utf-8")
  mutations = (
    (
      "drop-temperature",
      '      "temperature": temperature,\n',
      "",
      lambda module: _raises(lambda: _identity(module)),
    ),
    (
      "sort-names",
      '      "names": list(ordered_names),\n',
      '      "names": sorted(ordered_names),\n',
      lambda module: _identity(module)["parameters"]["names"]
                     != ["wa", "H0"],
    ),
    (
      "accept-wrong-interior-policy",
      '    if sampling["boundary_interior_policy"] \\\n'
      '        != UNIFORM_BOUNDARY_INTERIOR_POLICY:\n',
      '    if False and sampling["boundary_interior_policy"] \\\n'
      '        != UNIFORM_BOUNDARY_INTERIOR_POLICY:\n',
      lambda module: not _raises(lambda: module.validate_dataset_request_identity(
        dict(_identity(module), sampling=dict(
          _identity(module)["sampling"],
          boundary_interior_policy="wrong-policy-v0")))),
    ),
    (
      "coerce-bool-seed",
      '  if type(value) is not int:\n',
      '  if not isinstance(value, int):\n',
      lambda module: not _raises(lambda: _identity(module, seed=True)),
    ),
    (
      "omit-scientific-contract-digest-check",
      '  _require_digest(identity["scientific_contract_sha256"],\n'
      '                  "invariant scientific contract")\n',
      "",
      lambda module: not _raises(lambda: _identity(
        module, scientific_contract_sha256="not-a-digest")),
    ),
    (
      "drop-cmb-axis",
      '    members["axis.cmb.multipole"] = dvs + "_ell.npy"\n',
      "",
      lambda module: "axis.cmb.multipole" not in module.build_dataset_member_map(
        _identity(module, family="cmb", probe="cmblensed",
                  generator="dataset_generator_cmb"),
        params_stem="p", dvs_stem="d", fail_stem="f"),
    ),
    (
      "drop-one-base-member",
      '      members["base.grid2d.boost"] = dvs + "_boost_base.npy"\n',
      "",
      lambda module: len([
        role for role in module.build_dataset_member_map(
          _identity(module, family="grid2d", family_variant="syren-base",
                    probe="mps", generator="dataset_generator_mps"),
          params_stem="p", dvs_stem="d", fail_stem="f")
        if role.startswith("base.")]) == 1,
    ),
    (
      "borrow-full-members-in-chain-only",
      '  if identity["dataset_mode"] == "chain-only":\n'
      '    _require_unique_member_paths(members)\n'
      '    return members\n',
      '  if identity["dataset_mode"] == "chain-only" and False:\n'
      '    _require_unique_member_paths(members)\n'
      '    return members\n',
      lambda module: len(module.build_dataset_member_map(
        _identity(module, dataset_mode="chain-only"),
        params_stem="p", dvs_stem="d", fail_stem="f")) != 5,
    ),
    (
      "split-string-parameter-names",
      '  if type(ordered_names) is not list:\n',
      '  if False and type(ordered_names) is not list:\n',
      lambda module: not _raises(lambda: _identity(
        module, ordered_names="wa")),
    ),
    (
      "accept-wrong-generator",
      '  if generator != expected_generator:\n',
      '  if False and generator != expected_generator:\n',
      lambda module: not _raises(lambda: _identity(
        module, generator="dataset_generator_cmb")),
    ),
    (
      "accept-wrong-sampling-algorithm",
      '  if sampling["algorithm"] != policy["algorithm"]:\n',
      '  if False and sampling["algorithm"] != policy["algorithm"]:\n',
      lambda module: not _raises(lambda: _identity(
        module, sampling_algorithm="emcee-de-snooker-v1")),
    ),
    (
      "omit-final-member-path-collision",
      '  else:\n'
      '    raise AssertionError("validated dataset family was not routed")\n'
      '  _require_unique_member_paths(members)\n'
      '  return members\n',
      '  else:\n'
      '    raise AssertionError("validated dataset family was not routed")\n'
      '  return members\n',
      lambda module: not _raises(lambda: module.build_dataset_member_map(
        _identity(module), params_stem="p", dvs_stem="d", fail_stem="p.1")),
    ),
    (
      "omit-decimal-integer-bound",
      '  if len(str(abs(value))) > _MAX_JSON_INTEGER_DIGITS:\n',
      '  if False and len(str(abs(value))) > _MAX_JSON_INTEGER_DIGITS:\n',
      lambda module: not _raises(lambda: _identity(
        module, seed=10 ** 1024)),
    ),
  )

  results = []
  for label, old, new, witness in mutations:
    module_name = None
    try:
      mutated = _mutate(source, old, new, label)
      module_name, module = _load_mutant(mutated, label)
      killed = bool(witness(module))
    except Exception as exc:
      print("  [FAIL] mutation " + label + " could not run: " + str(exc))
      killed = False
    finally:
      if module_name is not None:
        sys.modules.pop(module_name, None)
    print("  [" + ("PASS" if killed else "FAIL") + "] mutation " + label)
    results.append(killed)
  fixed_source = FIXED_FACTS_SOURCE.read_text(encoding="utf-8")
  fixed_mutations = (
    (
      "include-generation-id-in-scientific-contract",
      '  del facts["dataset_id"]\n',
      '  # mutation keeps the generation-specific dataset_id\n',
      lambda module: module.scientific_contract_digest(
        _scientific_blocks(module, "sha256:" + "1" * 64))
        != module.scientific_contract_digest(
          _scientific_blocks(module, "sha256:" + "2" * 64)),
    ),
    (
      "omit-scientific-contract-resource-bounds",
      '  _validate_scientific_contract_value(\n'
      '    projection, path="$", depth=0, seen=set(), budget=budget)\n',
      "",
      lambda module: not _raises(lambda: _digest_hostile_contract(module)),
    ),
  )
  for label, old, new, witness in fixed_mutations:
    module_name = None
    try:
      mutated = _mutate(fixed_source, old, new, label)
      module_name, module = _load_mutant(
        mutated, label, source_path=FIXED_FACTS_SOURCE)
      killed = bool(witness(module))
    except Exception as exc:
      print("  [FAIL] mutation " + label + " could not run: " + str(exc))
      killed = False
    finally:
      if module_name is not None:
        sys.modules.pop(module_name, None)
    print("  [" + ("PASS" if killed else "FAIL") + "] mutation " + label)
    results.append(killed)
  return all(results) and len(results) == 15


def _digest_hostile_contract(module):
  blocks = _scientific_blocks(module, "sha256:" + "3" * 64)
  blocks[module.FIXED_FACTS_GROUP]["cosmology_fixed"]["mnu"] = 10 ** 1024
  return module.scientific_contract_digest(blocks)


def _terminal(aid, passed, witnesses):
  mark = "PASS" if passed else "FAIL"
  print("  [" + mark + "] " + aid + " (" + witnesses + ")")
  print("##AID " + aid + " " + mark)
  return passed


def main():
  identity = _run_suite(DatasetRequestIdentityTests, IDENTITY_TESTS)
  member = _run_suite(DatasetMemberMapTests, MEMBER_TESTS)
  mutations = _mutation_controls()
  passed = (
    _terminal(LEG_AIDS[0], identity,
              str(len(IDENTITY_TESTS)) + " focused witnesses"),
    _terminal(LEG_AIDS[1], member,
              str(len(MEMBER_TESTS)) + " focused witnesses"),
    _terminal(LEG_AIDS[2], mutations, "15/15 source mutations killed"),
  )
  if all(passed):
    print("dataset-request-contract: ALL PASS")
    return 0
  print("dataset-request-contract: FAIL")
  return 1


if __name__ == "__main__":
  raise SystemExit(main())

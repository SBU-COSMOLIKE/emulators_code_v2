"""CPU tests for the generator's immutable-publication bridge.

The production generator imports Cobaya, emcee, GetDist, and MPI. These tests
compile only the real bridge methods from its syntax tree, then exercise them
against the real filesystem publication API. No scientific model is started.
"""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from numpy.lib.format import open_memmap

from compute_data_vectors.dataset_manifest import (
  DATASET_SAMPLING_POLICIES,
  UNIFORM_BOUNDARY_INTERIOR_POLICY,
  build_dataset_member_census,
  build_dataset_request_identity,
)
from compute_data_vectors.dataset_publication import (
  DatasetPublicationError,
  begin_dataset_continuation,
  begin_dataset_generation,
  canonical_json_bytes,
  discard_dataset_draft,
  derive_dataset_slot,
  install_dataset_locator,
  load_active_generation,
  publish_dataset_generation,
)
from emulator import fixed_facts


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "compute_data_vectors" / "generator_core.py"
BRIDGE_METHODS = {
  "_facts_sidecar_text",
  "_build_dataset_request_identity",
  "_bind_dataset_publication_request",
  "_prepare_dataset_publication",
  "_preflight_active_checkpoint",
  "_close_dataset_memmaps",
  "_require_publishable_failure_mask",
  "_publish_dataset_generation",
}


def _compile_bridge():
  """Compile the selected production methods without heavy dependencies."""
  tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
  helper = [
    copy.deepcopy(node) for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "resolve_uniform_sampling_support"
  ]
  classes = [
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "GeneratorCore"
  ]
  if len(helper) != 1 or len(classes) != 1:
    raise AssertionError("generator source has an unexpected class/helper shape")
  methods = [
    copy.deepcopy(node) for node in classes[0].body
    if isinstance(node, ast.FunctionDef) and node.name in BRIDGE_METHODS
  ]
  if {method.name for method in methods} != BRIDGE_METHODS:
    raise AssertionError("generator publication bridge methods are incomplete")
  fixture = ast.ClassDef(
    name="BridgeCore",
    bases=[],
    keywords=[],
    body=methods,
    decorator_list=[])
  module = ast.Module(body=helper + [fixture], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace = {
    "np": np,
    "os": os,
    "Path": Path,
    "hashlib": hashlib,
    "fixed_facts": fixed_facts,
    "DATASET_SAMPLING_POLICIES": DATASET_SAMPLING_POLICIES,
    "DATASET_UNIFORM_BOUNDARY_INTERIOR_POLICY": (
      UNIFORM_BOUNDARY_INTERIOR_POLICY),
    "UNIFORM_BOUNDARY_INTERIOR_POLICY": UNIFORM_BOUNDARY_INTERIOR_POLICY,
    "build_dataset_request_identity": build_dataset_request_identity,
    "canonical_json_bytes": canonical_json_bytes,
    "derive_dataset_slot": derive_dataset_slot,
    "install_dataset_locator": install_dataset_locator,
    "begin_dataset_generation": begin_dataset_generation,
    "begin_dataset_continuation": begin_dataset_continuation,
    "discard_dataset_draft": discard_dataset_draft,
    "load_active_generation": load_active_generation,
    "publish_dataset_generation": publish_dataset_generation,
  }
  exec(compile(module, str(GENERATOR), "exec"), namespace)
  return namespace["BridgeCore"]


def _parsed_yaml(reverse=False):
  """Return equal parsed YAML data with either insertion order."""
  params = {"H0": {"prior": {"min": 55.0, "max": 91.0}}}
  train = {"probe": "cs", "ord": [["H0"]]}
  if reverse:
    return {"train_args": train, "params": params, "likelihood": {}}
  return {"likelihood": {}, "params": params, "train_args": train}


def _resolved_facts():
  """Return a small valid scientific record owned by the test fixture."""
  return {
    "generator": "dataset_generator_lensing",
    "family": "cosmolike",
    "cosmology_fixed": {},
    "neutrino_convention": fixed_facts.NOT_APPLICABLE,
    "flat_only": True,
    "dark_energy_law": "constant-w",
    "dark_energy_inputs": ["w"],
    "cl_units": fixed_facts.NOT_APPLICABLE,
    "base_identity": fixed_facts.NOT_APPLICABLE,
  }


def _yaml_test_module():
  """Supply the JSON subset of YAML when the small test runtime lacks PyYAML."""
  module = types.ModuleType("yaml")
  module.YAMLError = ValueError
  module.safe_dump = lambda value, **kwargs: json.dumps(
    value, sort_keys=kwargs.get("sort_keys", False))
  module.safe_load = json.loads
  return module


def _new_bridge(core_class, chains, operation):
  """Build one lightweight chain-only generator around logical stems."""
  instance = object.__new__(core_class)
  census = build_dataset_member_census(
    dataset_mode="chain-only",
    family="cosmolike",
    family_variant="standard",
    generator="dataset_generator_lensing",
    probe="cs",
    params_stem="params_cs_unifs_chain",
    dvs_stem="dvs_cs_unifs_chain",
    fail_stem="fail_cs_unifs_chain")
  instance.run_control = types.SimpleNamespace(
    dataset_mode="chain-only", operation=operation)
  instance.dataset_route = census.route
  instance.dataset_members = census.members
  instance.dataset_member_directory = chains
  instance.paramsf = str(chains / "params_cs_unifs_chain")
  instance.dvsf = str(chains / "dvs_cs_unifs_chain")
  instance.failf = str(chains / "fail_cs_unifs_chain")
  instance.sampled_params = ["H0"]
  instance.bounds_requested = np.asarray([[55.0, 91.0]], dtype=np.float32)
  instance.bounds = np.asarray([[55.0, 91.0]], dtype=np.float32)
  instance.unif = 1
  instance.temp = 128
  instance.maxcorr = 0.15
  instance.seed = 17
  instance.bounds_adj = 1.0
  instance._resolve_fixed_facts = _resolved_facts
  configuration_sha256 = hashlib.sha256(
    canonical_json_bytes(_parsed_yaml())).hexdigest()
  instance._bind_dataset_publication_request(configuration_sha256)
  instance.preflight_observations = []

  def load_checkpoint_read_only():
    instance.preflight_observations.append({
      "paramsf": instance.paramsf,
      "dvsf": instance.dvsf,
      "failf": instance.failf,
      "member_directory": instance.dataset_member_directory,
      "read_only": getattr(instance, "_checkpoint_read_only", False),
    })
    return True

  instance._BridgeCore__load_chk = load_checkpoint_read_only
  return instance


def _write_all_members(instance, prefix=b"fresh"):
  """Populate a prepared draft with exactly its required member census."""
  for role, relative in instance.dataset_locator.members.items():
    path = instance.dataset_draft.member_path(relative)
    path.write_bytes(prefix + b":" + role.encode("ascii") + b"\n")


class GeneratorPublicationBridgeTests(unittest.TestCase):
  """Keep generator output private until one authenticated publication."""

  @classmethod
  def setUpClass(cls):
    cls.core_class = _compile_bridge()

  def setUp(self):
    self.yaml_patch = mock.patch.dict(
      sys.modules, {"yaml": _yaml_test_module()})
    self.yaml_patch.start()
    self.addCleanup(self.yaml_patch.stop)

  def test_identity_hashes_canonical_yaml_and_final_uniform_support(self):
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      instance = _new_bridge(self.core_class, chains, "fresh")
      reordered_digest = hashlib.sha256(
        canonical_json_bytes(_parsed_yaml(reverse=True))).hexdigest()
      reordered = instance._build_dataset_request_identity(reordered_digest)
      self.assertEqual(instance.dataset_identity, reordered)
      self.assertEqual(
        instance.dataset_identity["parameters"]["configuration_sha256"],
        hashlib.sha256(canonical_json_bytes(_parsed_yaml())).hexdigest())

      support = instance._facts_sidecar_text(
        names=["H0"],
        dataset_id="f" * 64,
        resolved_bounds=np.asarray([
          [np.nextafter(np.float32(55.0), np.float32(91.0)),
           np.nextafter(np.float32(91.0), np.float32(55.0))]
        ], dtype=np.float32))
      blocks = fixed_facts.parse_sidecar(support, "the final support witness")
      self.assertEqual(
        instance.dataset_identity["scientific_contract_sha256"],
        fixed_facts.scientific_contract_digest(blocks))

  def test_fresh_writes_only_a_private_draft_then_publishes(self):
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      instance = _new_bridge(self.core_class, chains, "fresh")
      instance._prepare_dataset_publication()
      self.assertEqual(instance.paramsf,
                       str(instance.dataset_draft.files_path
                           / Path(instance.logical_paramsf).name))
      self.assertEqual(instance.dataset_member_directory,
                       instance.dataset_draft.files_path)
      self.assertFalse(Path(instance.logical_paramsf + ".1.txt").exists())
      self.assertTrue(instance.dataset_locator.path.is_file())

      _write_all_members(instance)
      instance._publish_dataset_generation()
      active = load_active_generation(
        instance.dataset_slot,
        expected_identity=instance.dataset_identity,
        expected_members=dict(instance.dataset_members))
      self.assertEqual({member.role for member in active.members},
                       set(instance.dataset_members))
      self.assertFalse(instance.dataset_draft.path.exists())
      self.assertFalse(Path(instance.logical_paramsf + ".1.txt").exists())

  def test_resume_copies_authenticated_active_and_publishes_with_cas(self):
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      first = _new_bridge(self.core_class, chains, "fresh")
      first._prepare_dataset_publication()
      _write_all_members(first)
      first._publish_dataset_generation()
      old = load_active_generation(
        first.dataset_slot,
        expected_identity=first.dataset_identity,
        expected_members=dict(first.dataset_members))

      resumed = _new_bridge(self.core_class, chains, "resume")
      resumed._prepare_dataset_publication()
      self.assertEqual(len(resumed.preflight_observations), 1)
      preflight = resumed.preflight_observations[0]
      self.assertTrue(preflight["read_only"])
      self.assertEqual(
        Path(preflight["paramsf"]).parent,
        old.member("parameters.chain").path.parent)
      self.assertNotEqual(
        Path(preflight["paramsf"]).parent,
        resumed.dataset_draft.files_path)
      self.assertEqual(resumed.dataset_expected_active_sha256,
                       old.active_sha256)
      chain = resumed.dataset_draft.member_path(
        resumed.dataset_locator.members["parameters.chain"])
      self.assertIn(b"parameters.chain", chain.read_bytes())
      chain.write_bytes(b"resumed:parameters.chain\n")
      resumed._publish_dataset_generation()
      new = load_active_generation(
        resumed.dataset_slot,
        expected_identity=resumed.dataset_identity,
        expected_members=dict(resumed.dataset_members))
      self.assertNotEqual(new.active_sha256, old.active_sha256)
      self.assertTrue(old.member("parameters.chain").path.is_file())
      self.assertEqual(old.member("parameters.chain").path.read_bytes(),
                       b"fresh:parameters.chain\n")

  def test_append_authenticates_active_then_refuses_without_a_draft(self):
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      first = _new_bridge(self.core_class, chains, "fresh")
      first._prepare_dataset_publication()
      _write_all_members(first)
      first._publish_dataset_generation()
      before = load_active_generation(
        first.dataset_slot,
        expected_identity=first.dataset_identity,
        expected_members=dict(first.dataset_members))

      appended = _new_bridge(self.core_class, chains, "append")
      locator_path = first.dataset_locator.path
      locator_path.unlink()
      work_before = sorted(
        path.name for path in appended.dataset_slot.work_path.iterdir())
      with self.assertRaisesRegex(RuntimeError, "exact append is not available"):
        appended._prepare_dataset_publication()
      self.assertEqual(len(appended.preflight_observations), 1)
      self.assertTrue(appended.preflight_observations[0]["read_only"])
      after = load_active_generation(
        appended.dataset_slot,
        expected_identity=appended.dataset_identity,
        expected_members=dict(appended.dataset_members))
      self.assertEqual(after.active_sha256, before.active_sha256)
      self.assertIsNone(appended.dataset_draft)
      self.assertIsNone(appended.dataset_locator)
      self.assertFalse(locator_path.exists())
      self.assertEqual(
        sorted(path.name for path in appended.dataset_slot.work_path.iterdir()),
        work_before)

  def test_resume_semantic_refusal_precedes_locator_or_draft_creation(self):
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      first = _new_bridge(self.core_class, chains, "fresh")
      first._prepare_dataset_publication()
      _write_all_members(first)
      first._publish_dataset_generation()

      resumed = _new_bridge(self.core_class, chains, "resume")
      locator_path = first.dataset_locator.path
      locator_path.unlink()
      work_before = sorted(
        path.name for path in resumed.dataset_slot.work_path.iterdir())

      def refuse_semantic_checkpoint():
        self.assertTrue(resumed._checkpoint_read_only)
        raise ValueError("synthetic semantic checkpoint refusal")

      resumed._BridgeCore__load_chk = refuse_semantic_checkpoint
      with self.assertRaisesRegex(ValueError, "semantic checkpoint refusal"):
        resumed._prepare_dataset_publication()

      self.assertIsNone(resumed.dataset_draft)
      self.assertIsNone(resumed.dataset_locator)
      self.assertFalse(locator_path.exists())
      self.assertEqual(resumed.paramsf, resumed.logical_paramsf)
      self.assertEqual(resumed.dvsf, resumed.logical_dvsf)
      self.assertEqual(resumed.failf, resumed.logical_failf)
      self.assertFalse(resumed._checkpoint_read_only)
      self.assertEqual(
        sorted(path.name for path in resumed.dataset_slot.work_path.iterdir()),
        work_before)

  def test_fresh_refuses_an_existing_authenticated_active_before_work(self):
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      first = _new_bridge(self.core_class, chains, "fresh")
      first._prepare_dataset_publication()
      _write_all_members(first)
      first._publish_dataset_generation()

      repeated = _new_bridge(self.core_class, chains, "fresh")
      with self.assertRaisesRegex(RuntimeError, "already owns this logical"):
        repeated._prepare_dataset_publication()
      self.assertIsNone(repeated.dataset_draft)

  def test_first_fresh_crash_cannot_be_mistaken_for_resumable_active(self):
    with tempfile.TemporaryDirectory() as directory:
      chains = Path(directory) / "chains"
      chains.mkdir()
      crashed = _new_bridge(self.core_class, chains, "fresh")
      crashed._prepare_dataset_publication()
      _write_all_members(crashed, prefix=b"unpublished")

      resumed = _new_bridge(self.core_class, chains, "resume")
      with self.assertRaisesRegex(
          DatasetPublicationError, "active dataset record cannot be opened"):
        resumed._prepare_dataset_publication()
      self.assertTrue(crashed.dataset_draft.path.is_dir())
      self.assertFalse(crashed.dataset_slot.active_path.exists())

  def test_single_and_family_memmaps_are_closed_before_publish_call(self):
    for mapping in (False, True):
      with self.subTest(family_mapping=mapping):
        with tempfile.TemporaryDirectory() as directory:
          path = Path(directory) / "payload.npy"
          store = open_memmap(path, mode="w+", dtype=np.float32, shape=(2, 2))
          store[:] = 3.0
          instance = object.__new__(self.core_class)
          instance.datavectors = {"tt": store} if mapping else store
          instance.run_control = types.SimpleNamespace(
            dataset_mode="chain-only")
          instance.dataset_draft = object()
          instance.dataset_locator = types.SimpleNamespace(
            identity={"request": "test"}, members={"payload": "payload.npy"})
          instance.dataset_expected_active_sha256 = "a" * 64
          observed = []

          def witness(*args, **kwargs):
            observed.append((store._mmap.closed, args, kwargs))

          publication_globals = (
            self.core_class._publish_dataset_generation.__globals__)
          with mock.patch.dict(
              publication_globals, {"publish_dataset_generation": witness}):
            instance._publish_dataset_generation()
          self.assertEqual(len(observed), 1)
          self.assertTrue(observed[0][0])
          self.assertEqual(observed[0][2]["expected_active_sha256"],
                           "a" * 64)

  def test_failed_row_mask_refuses_before_publication(self):
    with tempfile.TemporaryDirectory() as directory:
      files = Path(directory) / "files"
      files.mkdir()
      (files / "failed.txt").write_text("0\n1\n", encoding="ascii")
      instance = object.__new__(self.core_class)
      instance.run_control = types.SimpleNamespace(dataset_mode="full")
      instance.samples = np.zeros((2, 1), dtype=np.float32)
      instance.dataset_draft = types.SimpleNamespace(files_path=files)
      instance.dataset_locator = types.SimpleNamespace(
        identity={"request": "test"},
        members={"rows.failure-mask": "failed.txt"})
      instance.dataset_expected_active_sha256 = None
      calls = []
      publication_globals = self.core_class._publish_dataset_generation.__globals__
      with mock.patch.dict(
          publication_globals,
          {"publish_dataset_generation": lambda *args, **kwargs: calls.append(1)}):
        with self.assertRaisesRegex(RuntimeError, "1 data-vector rows failed"):
          instance._publish_dataset_generation()
      self.assertEqual(calls, [])

  def test_constructor_barrier_precedes_rank_zero_publication(self):
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    core = next(
      node for node in tree.body
      if isinstance(node, ast.ClassDef) and node.name == "GeneratorCore")
    constructor = next(
      node for node in core.body
      if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    direct_calls = []
    for node in ast.walk(constructor):
      if not isinstance(node, ast.Call):
        continue
      if isinstance(node.func, ast.Attribute):
        direct_calls.append((node.lineno, node.func.attr))
    positions = {}
    for index, name in direct_calls:
      positions.setdefault(name, []).append(index)
    self.assertLess(positions["_prepare_dataset_publication"][0],
                    positions["__run_mcmc"][0])
    self.assertLess(positions["__run_mcmc"][0], positions["bcast"][0])
    self.assertLess(positions["bcast"][0],
                    positions["__generate_datavectors"][0])
    self.assertLess(positions["__generate_datavectors"][0],
                    positions["Barrier"][0])
    self.assertLess(positions["Barrier"][0],
                    positions["_publish_dataset_generation"][0])


if __name__ == "__main__":
  unittest.main()

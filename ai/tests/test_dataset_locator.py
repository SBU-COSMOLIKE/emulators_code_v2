"""CPU tests for stable logical-filename dataset locators."""

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest

from compute_data_vectors.dataset_manifest import (
  UNIFORM_BOUNDARY_INTERIOR_POLICY,
  build_dataset_member_map,
  build_dataset_request_identity,
)
from compute_data_vectors.dataset_publication import (
  DatasetLocator,
  DatasetPublicationError,
  begin_dataset_generation,
  derive_dataset_slot,
  install_dataset_locator,
  load_dataset_locator,
  load_located_generation,
  publish_dataset_generation,
)


def _identity(*, seed=17, dataset_mode="chain-only"):
  return build_dataset_request_identity(
    dataset_mode=dataset_mode,
    family="cosmolike",
    family_variant="standard",
    generator="dataset_generator_lensing",
    probe="cs",
    sampling_mode="uniform",
    temperature=64,
    boundary_factor=1.0,
    max_correlation=None,
    sampling_algorithm="uniform-box-v1",
    seed=seed,
    rng_bit_generator="PCG64",
    rng_emcee_random=None,
    rng_policy="persist-complete-state-v1",
    boundary_interior_policy=UNIFORM_BOUNDARY_INTERIOR_POLICY,
    ordered_names=["omegabh2", "H0"],
    configuration_sha256="1" * 64,
    scientific_contract_sha256="a" * 64)


class DatasetLocatorTests(unittest.TestCase):

  def setUp(self):
    self._temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self._temporary.cleanup)
    self.root = Path(self._temporary.name)
    self.chains = self.root / "chains"
    self.chains.mkdir()
    self.addCleanup(self._make_tree_writable)
    self.identity = _identity()
    self.slot = derive_dataset_slot(
      self.chains,
      params_stem=str(self.chains / "params"),
      dvs_stem=str(self.chains / "dvs"),
      fail_stem=str(self.chains / "fail"),
      dataset_mode=self.identity["dataset_mode"],
      family=self.identity["family"])
    self.members = build_dataset_member_map(
      self.identity, params_stem="params", dvs_stem="dvs", fail_stem="fail")

  def _make_tree_writable(self):
    if not self.root.exists():
      return
    for directory, names, files in os.walk(
        str(self.root), topdown=True, followlinks=False):
      directory_path = Path(directory)
      try:
        os.chmod(directory_path, 0o700, follow_symlinks=False)
      except (FileNotFoundError, NotImplementedError, OSError):
        pass
      for name in names + files:
        path = directory_path / name
        try:
          status = os.lstat(path)
          if not stat.S_ISLNK(status.st_mode):
            os.chmod(path, 0o600, follow_symlinks=False)
        except (FileNotFoundError, NotImplementedError, OSError):
          pass

  def _publish(self, label, expected_active_sha256):
    draft = begin_dataset_generation(self.slot)
    for role, relative in self.members.items():
      draft.member_path(relative).write_bytes(
        (label + ":" + role + "\n").encode("ascii"))
    return publish_dataset_generation(
      draft,
      identity=self.identity,
      members=self.members,
      expected_active_sha256=expected_active_sha256)

  def test_install_is_canonical_read_only_and_idempotent(self):
    locator = install_dataset_locator(
      self.slot, identity=self.identity, members=self.members)
    self.assertIsInstance(locator, DatasetLocator)
    self.assertEqual(locator.logical_parameter, "params.1.txt")
    self.assertEqual(
      locator.path,
      self.chains / ".datasets" / "locators" / "params.1.txt.json")
    self.assertEqual(stat.S_IMODE(os.lstat(locator.path).st_mode), 0o444)

    payload = locator.path.read_bytes()
    record = json.loads(payload.decode("ascii"))
    self.assertEqual(
      set(record), {"identity", "members", "schema", "slot", "slot_id"})
    self.assertNotIn(b"generation", payload)
    self.assertNotIn(b"active_sha256", payload)
    before = os.lstat(locator.path)
    repeated = install_dataset_locator(
      self.slot, identity=self.identity, members=self.members)
    after = os.lstat(locator.path)
    self.assertEqual(repeated, locator)
    self.assertEqual((after.st_ino, after.st_mtime_ns, locator.path.read_bytes()),
                     (before.st_ino, before.st_mtime_ns, payload))

    identity_copy = locator.identity
    members_copy = locator.members
    identity_copy["sampling"]["seed"] = 999
    members_copy["parameters.chain"] = "changed.1.txt"
    self.assertEqual(locator.identity["sampling"]["seed"], 17)
    self.assertEqual(locator.members["parameters.chain"], "params.1.txt")

  def test_existing_logical_name_cannot_be_reassigned(self):
    installed = install_dataset_locator(
      self.slot, identity=self.identity, members=self.members)
    before = installed.path.read_bytes()
    different_identity = _identity(seed=18)
    different_members = build_dataset_member_map(
      different_identity,
      params_stem="params", dvs_stem="dvs", fail_stem="fail")
    with self.assertRaisesRegex(DatasetPublicationError,
                                "different dataset locator"):
      install_dataset_locator(
        self.slot, identity=different_identity, members=different_members)
    self.assertEqual(installed.path.read_bytes(), before)

  def test_install_recomputes_slot_and_member_contract(self):
    wrong_members = dict(self.members)
    wrong_members["parameters.chain"] = "other.1.txt"
    with self.assertRaisesRegex(DatasetPublicationError, "exact map"):
      install_dataset_locator(
        self.slot, identity=self.identity, members=wrong_members)

    full_identity = _identity(dataset_mode="full")
    full_members = build_dataset_member_map(
      full_identity, params_stem="params", dvs_stem="dvs", fail_stem="fail")
    with self.assertRaisesRegex(DatasetPublicationError, "slot mode and family"):
      install_dataset_locator(
        self.slot, identity=full_identity, members=full_members)
    self.assertFalse(
      (self.chains / ".datasets" / "locators" / "params.1.txt.json").exists())

  def test_load_accepts_only_the_named_chains_child(self):
    installed = install_dataset_locator(
      self.slot, identity=self.identity, members=self.members)
    by_name = load_dataset_locator(
      self.chains, logical_parameter="params.1.txt")
    by_path = load_dataset_locator(
      self.chains, logical_parameter=self.chains / "params.1.txt")
    self.assertEqual(by_name, installed)
    self.assertEqual(by_path, installed)

    for wrong in ("nested/params.1.txt", "../params.1.txt",
                  self.root / "params.1.txt"):
      with self.subTest(wrong=wrong):
        with self.assertRaisesRegex(DatasetPublicationError,
                                    "basename|direct child"):
          load_dataset_locator(self.chains, logical_parameter=wrong)
    with self.assertRaisesRegex(DatasetPublicationError, "opened safely"):
      load_dataset_locator(self.chains, logical_parameter="missing.1.txt")

  def test_load_refuses_mutable_noncanonical_and_linked_records(self):
    locator = install_dataset_locator(
      self.slot, identity=self.identity, members=self.members)
    os.chmod(locator.path, 0o644)
    with self.assertRaisesRegex(DatasetPublicationError, "still writable"):
      load_dataset_locator(self.chains, logical_parameter="params.1.txt")

    os.chmod(locator.path, 0o600)
    record = json.loads(locator.path.read_text(encoding="ascii"))
    locator.path.write_text(json.dumps(record, indent=2), encoding="ascii")
    os.chmod(locator.path, 0o444)
    with self.assertRaisesRegex(DatasetPublicationError, "canonical"):
      load_dataset_locator(self.chains, logical_parameter="params.1.txt")

    locator.path.unlink()
    target = self.root / "outside.json"
    target.write_text("{}\n", encoding="ascii")
    os.symlink(target, locator.path)
    with self.assertRaisesRegex(DatasetPublicationError, "opened safely"):
      load_dataset_locator(self.chains, logical_parameter="params.1.txt")

  def test_locator_resolves_each_current_generation_without_changing(self):
    locator = install_dataset_locator(
      self.slot, identity=self.identity, members=self.members)
    locator_bytes = locator.path.read_bytes()
    with self.assertRaises(DatasetPublicationError):
      load_located_generation(locator)

    first = self._publish("first", None)
    loaded_first = load_located_generation(
      load_dataset_locator(self.chains, logical_parameter="params.1.txt"))
    self.assertEqual(loaded_first.generation, first.generation)
    self.assertEqual(
      loaded_first.member("parameters.chain").path.read_bytes(),
      b"first:parameters.chain\n")

    second = self._publish("second", first.active_sha256)
    loaded_second = load_located_generation(locator)
    self.assertEqual(loaded_second.generation, second.generation)
    self.assertNotEqual(loaded_second.generation, loaded_first.generation)
    self.assertEqual(
      loaded_first.member("parameters.chain").path.read_bytes(),
      b"first:parameters.chain\n")
    self.assertEqual(locator.path.read_bytes(), locator_bytes)

    with self.assertRaisesRegex(DatasetPublicationError, "DatasetLocator"):
      load_located_generation(self.slot)


if __name__ == "__main__":
  unittest.main()

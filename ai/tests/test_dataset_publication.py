"""CPU-only tests for immutable dataset generation publication."""

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock
import warnings

import compute_data_vectors.dataset_publication as publication
from compute_data_vectors.dataset_publication import begin_dataset_generation
from compute_data_vectors.dataset_publication import canonical_json_bytes
from compute_data_vectors.dataset_publication import DatasetPublicationError
from compute_data_vectors.dataset_publication import derive_dataset_slot
from compute_data_vectors.dataset_publication import load_active_generation
from compute_data_vectors.dataset_publication import publish_dataset_generation


class InjectedPublicationCrash(RuntimeError):
  """A test stopped publication at one documented durability boundary."""


class DatasetPublicationTests(unittest.TestCase):

  def setUp(self):
    self._temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self._temporary.cleanup)
    self.root = Path(self._temporary.name)
    self.chains = self.root / "chains"
    self.chains.mkdir()
    # Published generations deliberately have no write bits.  Make them
    # removable before TemporaryDirectory performs its cross-version cleanup.
    self.addCleanup(self._make_test_tree_writable)

  def _make_test_tree_writable(self):
    if not self.root.exists():
      return
    for directory, names, files in os.walk(
        str(self.root), topdown=True, followlinks=False):
      directory_path = Path(directory)
      try:
        os.chmod(str(directory_path), 0o700, follow_symlinks=False)
      except (FileNotFoundError, NotImplementedError, OSError):
        pass
      for name in names + files:
        path = directory_path / name
        try:
          status = os.lstat(str(path))
          if not stat.S_ISLNK(status.st_mode):
            os.chmod(str(path), 0o600, follow_symlinks=False)
        except (FileNotFoundError, NotImplementedError, OSError):
          pass

  def _slot(self, tag, *, mode="full", family="cmb"):
    return derive_dataset_slot(
      self.chains,
      params_stem=str(self.chains / ("params_" + tag)),
      dvs_stem=str(self.chains / ("dvs_" + tag)),
      fail_stem=str(self.chains / ("fail_" + tag)),
      dataset_mode=mode,
      family=family)

  def _draft(self, slot, payloads, generation=None):
    draft = begin_dataset_generation(slot, generation=generation)
    for relative_path, payload in payloads.items():
      draft.member_path(relative_path).write_bytes(payload)
    return draft

  def _publish(self, slot, tag, *, identity=None, expected=None,
               checkpoint=None):
    payloads = {
      "params.1.txt": ("chain-" + tag).encode("ascii"),
      "metadata/facts.yaml": ("facts-" + tag).encode("ascii"),
    }
    members = {
      "chain": "params.1.txt",
      "facts": "metadata/facts.yaml",
    }
    if identity is None:
      identity = {"dataset": tag, "seed": 17}
    draft = self._draft(slot, payloads)
    active = publish_dataset_generation(
      draft,
      identity=identity,
      members=members,
      expected_active_sha256=expected,
      checkpoint=checkpoint)
    return active, identity, members

  @staticmethod
  def _replace_read_only_file(path, payload):
    os.chmod(str(path), 0o600, follow_symlinks=False)
    try:
      path.write_bytes(payload)
    finally:
      os.chmod(str(path), 0o444, follow_symlinks=False)

  def _manifest_paths(self, active):
    generation_path = (
      active.slot.generations_path / active.generation)
    return generation_path, generation_path / "manifest.json"

  def _replace_manifest(self, active, payload):
    _, manifest_path = self._manifest_paths(active)
    self._replace_read_only_file(manifest_path, payload)
    pointer = json.loads(active.slot.active_path.read_text(encoding="utf-8"))
    pointer["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    self._replace_read_only_file(
      active.slot.active_path, canonical_json_bytes(pointer))

  @staticmethod
  def _sealed_work_entries(slot):
    return sorted(
      path.name for path in slot.work_path.iterdir()
      if path.name.startswith("sealed-"))

  def test_roundtrip_authenticates_nested_members_and_checkpoints(self):
    slot = self._slot("roundtrip")
    checkpoints = []
    active, identity, members = self._publish(
      slot, "roundtrip", checkpoint=checkpoints.append)

    loaded = load_active_generation(
      slot,
      expected_identity={"seed": 17, "dataset": "roundtrip"},
      expected_members={
        "facts": "metadata/facts.yaml",
        "chain": "params.1.txt",
      })

    self.assertEqual(loaded.generation, active.generation)
    self.assertEqual(loaded.active_sha256, active.active_sha256)
    self.assertEqual(loaded.manifest_sha256, active.manifest_sha256)
    self.assertEqual(loaded.identity, identity)
    self.assertEqual([member.role for member in loaded.members],
                     ["chain", "facts"])
    self.assertEqual(loaded.member("chain").path.read_bytes(),
                     b"chain-roundtrip")
    self.assertEqual(loaded.member("facts").relative_path,
                     "metadata/facts.yaml")
    self.assertEqual(set(members), {"chain", "facts"})
    self.assertEqual(
      checkpoints,
      ["draft-durable", "generation-installed",
       "active-temp-durable", "active-replaced",
       "active-directory-durable"])

  def test_slots_with_different_descriptors_are_isolated(self):
    full_slot = self._slot("shared", mode="full")
    chain_slot = self._slot("shared", mode="chain-only")
    self.assertNotEqual(full_slot.slot_id, chain_slot.slot_id)
    self.assertNotEqual(full_slot.path, chain_slot.path)

    full, full_identity, roles = self._publish(full_slot, "full")
    chain, chain_identity, _ = self._publish(chain_slot, "chain")
    loaded_full = load_active_generation(
      full_slot, expected_identity=full_identity,
      expected_members=roles)
    loaded_chain = load_active_generation(
      chain_slot, expected_identity=chain_identity,
      expected_members=roles)

    self.assertEqual(loaded_full.generation, full.generation)
    self.assertEqual(loaded_chain.generation, chain.generation)
    self.assertEqual(loaded_full.member("chain").path.read_bytes(),
                     b"chain-full")
    self.assertEqual(loaded_chain.member("chain").path.read_bytes(),
                     b"chain-chain")
    self.assertFalse(
      loaded_full.member("chain").path.is_relative_to(chain_slot.path)
      if hasattr(Path, "is_relative_to") else
      str(loaded_full.member("chain").path).startswith(str(chain_slot.path)))

  def test_each_slot_axis_is_distinct_and_relocation_is_stable(self):
    def derive(chains, *, params="params_axis", dvs="dvs_axis",
               fail="fail_axis", mode="full", family="cmb"):
      return derive_dataset_slot(
        chains,
        params_stem=str(chains / params),
        dvs_stem=str(chains / dvs),
        fail_stem=str(chains / fail),
        dataset_mode=mode,
        family=family)

    base = derive(self.chains)
    variants = {
      "params": derive(self.chains, params="params_other"),
      "dvs": derive(self.chains, dvs="dvs_other"),
      "fail": derive(self.chains, fail="fail_other"),
      "mode": derive(self.chains, mode="chain-only"),
      "family": derive(self.chains, family="mps"),
    }
    for axis, variant in variants.items():
      with self.subTest(axis=axis):
        self.assertNotEqual(base.slot_id, variant.slot_id)

    relocated_chains = self.root / "relocated" / "chains"
    relocated_chains.mkdir(parents=True)
    relocated = derive(relocated_chains)
    self.assertEqual(base.slot_id, relocated.slot_id)
    self.assertEqual(base.descriptor_json, relocated.descriptor_json)
    self.assertNotEqual(base.path, relocated.path)

  def test_case_only_logical_stem_collision_is_refused(self):
    with self.assertRaisesRegex(DatasetPublicationError,
                                "case-insensitive filesystems"):
      derive_dataset_slot(
        self.chains,
        params_stem=str(self.chains / "Dataset"),
        dvs_stem=str(self.chains / "dataset"),
        fail_stem=str(self.chains / "fail_case"),
        dataset_mode="full",
        family="cmb")

  def test_identity_and_role_path_censuses_match_exactly(self):
    slot = self._slot("identity")
    identity = {"flag": True, "nested": {"count": 1}}
    _, _, roles = self._publish(slot, "identity", identity=identity)

    with self.assertRaisesRegex(DatasetPublicationError, "identity"):
      load_active_generation(
        slot,
        expected_identity={"flag": 1, "nested": {"count": 1}},
        expected_members=roles)
    for requested in (
        {"chain": "params.1.txt"},
        {"chain": "params.1.txt", "facts": "metadata/facts.yaml",
         "extra": "extra.bin"}):
      with self.subTest(requested=requested):
        with self.assertRaisesRegex(DatasetPublicationError, "roles differ"):
          load_active_generation(
            slot, expected_identity=identity,
            expected_members=requested)
    wrong_path = dict(roles)
    wrong_path["chain"] = "renamed.1.txt"
    with self.assertRaisesRegex(DatasetPublicationError,
                                "requested basename/path"):
      load_active_generation(
        slot, expected_identity=identity,
        expected_members=wrong_path)
    with self.assertRaisesRegex(DatasetPublicationError,
                                "more than one member role"):
      load_active_generation(
        slot, expected_identity=identity,
        expected_members={
          "chain": "params.1.txt",
          "facts": "params.1.txt",
        })

  def test_canonical_json_is_stable_and_refuses_nonfinite_values(self):
    self.assertEqual(
      canonical_json_bytes({"z": 2, "a": [True, "caf\N{LATIN SMALL LETTER E WITH ACUTE}"]}),
      b'{"a":[true,"caf\\u00e9"],"z":2}\n')
    for value in (float("nan"), float("inf"), float("-inf")):
      with self.subTest(value=value):
        with self.assertRaisesRegex(DatasetPublicationError, "non-finite"):
          canonical_json_bytes({"value": value})

  def test_canonical_and_loader_refuse_oversized_json_integer(self):
    oversized_integer = 10 ** 1024
    with self.assertRaisesRegex(DatasetPublicationError,
                                "longer than 1024 decimal digits"):
      canonical_json_bytes({"value": oversized_integer})

    slot = self._slot("oversized_integer")
    _, identity, members = self._publish(slot, "integer")
    active_bytes = slot.active_path.read_bytes()
    oversized_token = b"9" * 1025
    attacked = active_bytes.replace(
      b'"schema":1', b'"schema":' + oversized_token)
    self.assertNotEqual(attacked, active_bytes)
    self._replace_read_only_file(slot.active_path, attacked)
    with self.assertRaisesRegex(DatasetPublicationError,
                                "JSON integer exceeds 1024"):
      load_active_generation(
        slot,
        expected_identity=identity,
        expected_members=members)

  def test_active_record_refuses_noncanonical_duplicate_and_unknown_fields(self):
    for attack in ("noncanonical", "duplicate", "unknown"):
      with self.subTest(attack=attack):
        slot = self._slot("active_" + attack)
        active, identity, roles = self._publish(slot, attack)
        pointer = json.loads(slot.active_path.read_text(encoding="utf-8"))
        if attack == "noncanonical":
          payload = json.dumps(pointer, indent=2, sort_keys=True).encode("ascii")
        elif attack == "duplicate":
          payload = (
            '{"generation":' + json.dumps(pointer["generation"])
            + ',"generation":' + json.dumps(pointer["generation"])
            + ',"manifest_sha256":'
            + json.dumps(pointer["manifest_sha256"])
            + ',"schema":1,"slot_id":'
            + json.dumps(pointer["slot_id"]) + '}\n').encode("ascii")
        else:
          pointer["unexpected"] = "field"
          payload = canonical_json_bytes(pointer)
        self._replace_read_only_file(slot.active_path, payload)
        message = {
          "noncanonical": "canonical",
          "duplicate": "duplicate",
          "unknown": "fields must be exactly",
        }[attack]
        with self.assertRaisesRegex(DatasetPublicationError, message):
          load_active_generation(
            slot, expected_identity=identity,
            expected_members=roles)
        self.assertEqual(active.slot, slot)

  def test_manifest_refuses_noncanonical_and_unknown_fields(self):
    for attack in ("noncanonical", "unknown"):
      with self.subTest(attack=attack):
        slot = self._slot("manifest_" + attack)
        active, identity, roles = self._publish(slot, attack)
        _, manifest_path = self._manifest_paths(active)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if attack == "noncanonical":
          payload = json.dumps(manifest, indent=1, sort_keys=True).encode("ascii")
          message = "canonical"
        else:
          manifest["unexpected"] = 1
          payload = canonical_json_bytes(manifest)
          message = "fields must be exactly"
        self._replace_manifest(active, payload)
        with self.assertRaisesRegex(DatasetPublicationError, message):
          load_active_generation(
            slot, expected_identity=identity,
            expected_members=roles)

  def test_declared_and_manifest_paths_cannot_traverse(self):
    slot = self._slot("traversal_draft")
    draft = self._draft(slot, {"member.bin": b"safe"})
    with self.assertRaisesRegex(DatasetPublicationError, "path"):
      publish_dataset_generation(
        draft,
        identity={"kind": "draft"},
        members={"payload": "../member.bin"},
        expected_active_sha256=None)

    slot = self._slot("traversal_manifest")
    active, identity, roles = self._publish(slot, "traversal")
    _, manifest_path = self._manifest_paths(active)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"]["chain"]["path"] = "files/../escape"
    self._replace_manifest(active, canonical_json_bytes(manifest))
    with self.assertRaisesRegex(DatasetPublicationError,
                                "portable|traversal"):
      load_active_generation(
        slot, expected_identity=identity,
        expected_members=roles)

  def test_symlink_member_is_refused(self):
    slot = self._slot("symlink")
    draft = begin_dataset_generation(slot)
    outside = self.root / "outside.bin"
    outside.write_bytes(b"outside")
    os.symlink(str(outside), str(draft.member_path("payload.bin")))
    with self.assertRaisesRegex(DatasetPublicationError, "regular file"):
      publish_dataset_generation(
        draft, identity={"attack": "symlink"},
        members={"payload": "payload.bin"},
        expected_active_sha256=None)
    self.assertFalse(os.path.lexists(slot.active_path))

  def test_hardlinked_member_is_refused(self):
    slot = self._slot("hardlink")
    draft = begin_dataset_generation(slot)
    outside = self.root / "hardlink-source.bin"
    outside.write_bytes(b"linked")
    os.link(str(outside), str(draft.member_path("payload.bin")))
    with self.assertRaisesRegex(DatasetPublicationError, "regular file"):
      publish_dataset_generation(
        draft, identity={"attack": "hardlink"},
        members={"payload": "payload.bin"},
        expected_active_sha256=None)
    self.assertFalse(os.path.lexists(slot.active_path))

  @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO creation is unavailable")
  def test_special_member_is_refused(self):
    slot = self._slot("fifo")
    draft = begin_dataset_generation(slot)
    os.mkfifo(str(draft.member_path("payload.bin")), 0o600)
    with self.assertRaisesRegex(DatasetPublicationError, "regular file"):
      publish_dataset_generation(
        draft, identity={"attack": "fifo"},
        members={"payload": "payload.bin"},
        expected_active_sha256=None)
    self.assertFalse(os.path.lexists(slot.active_path))

  def test_missing_and_extra_draft_members_are_refused(self):
    slot = self._slot("missing")
    draft = begin_dataset_generation(slot)
    with self.assertRaisesRegex(DatasetPublicationError, "missing"):
      publish_dataset_generation(
        draft, identity={"attack": "missing"},
        members={"payload": "missing.bin"},
        expected_active_sha256=None)

    slot = self._slot("extra")
    draft = self._draft(
      slot, {"payload.bin": b"declared", "extra.bin": b"undeclared"})
    with self.assertRaisesRegex(DatasetPublicationError, "extra"):
      publish_dataset_generation(
        draft, identity={"attack": "extra"},
        members={"payload": "payload.bin"},
        expected_active_sha256=None)

  def test_extra_empty_nested_directory_is_refused(self):
    slot = self._slot("empty_directory")
    draft = self._draft(slot, {"payload.bin": b"declared"})
    # member_path creates safe parents but does not create the final file.
    draft.member_path("empty/nested/not-created.bin")
    with self.assertRaisesRegex(DatasetPublicationError,
                                "extra directories"):
      publish_dataset_generation(
        draft,
        identity={"attack": "empty-directory"},
        members={"payload": "payload.bin"},
        expected_active_sha256=None)
    self.assertFalse(os.path.lexists(slot.active_path))

  def test_missing_extra_and_corrupted_published_members_are_refused(self):
    attacks = ("missing", "extra", "corrupt")
    for attack in attacks:
      with self.subTest(attack=attack):
        slot = self._slot("published_" + attack)
        active, identity, roles = self._publish(slot, attack)
        generation_path, _ = self._manifest_paths(active)
        files_path = generation_path / "files"
        chain_path = files_path / "params.1.txt"
        if attack == "missing":
          os.chmod(str(files_path), 0o755, follow_symlinks=False)
          chain_path.unlink()
          os.chmod(str(files_path), 0o555, follow_symlinks=False)
          message = "census"
        elif attack == "extra":
          os.chmod(str(files_path), 0o755, follow_symlinks=False)
          extra = files_path / "extra.bin"
          extra.write_bytes(b"extra")
          os.chmod(str(extra), 0o444, follow_symlinks=False)
          os.chmod(str(files_path), 0o555, follow_symlinks=False)
          message = "census"
        else:
          self._replace_read_only_file(chain_path, b"corrupted")
          message = "size/digest"
        with self.assertRaisesRegex(DatasetPublicationError, message):
          load_active_generation(
            slot, expected_identity=identity,
            expected_members=roles)

  def test_source_mutation_between_member_copies_refuses_whole_dataset(self):
    for target_role in ("chain", "facts"):
      with self.subTest(target_role=target_role):
        slot = self._slot("copy_mutation_" + target_role)
        old, old_identity, members = self._publish(slot, "old")
        draft = self._draft(slot, {
          "params.1.txt": b"chain-pending",
          "metadata/facts.yaml": b"facts-pending",
        })
        target_relative = members[target_role]
        target_path = draft.member_path(target_relative)
        replacement = {
          "chain": b"CHAIN-pending",
          "facts": b"FACTS-pending",
        }[target_role]
        original_copy = publication._copy_open_regular_file
        copy_count = [0]

        def mutate_after_first_copy(source_fd, destination, *, label):
          result = original_copy(source_fd, destination, label=label)
          copy_count[0] += 1
          if copy_count[0] == 1:
            target_path.write_bytes(replacement)
          return result

        with mock.patch.object(
            publication, "_copy_open_regular_file",
            side_effect=mutate_after_first_copy):
          with self.assertRaisesRegex(
              DatasetPublicationError,
              "source changed across the complete dataset copy"):
            publish_dataset_generation(
              draft,
              identity={"dataset": "pending", "target": target_role},
              members=members,
              expected_active_sha256=old.active_sha256)

        self.assertEqual(copy_count[0], 2)
        self.assertTrue(draft.path.exists())
        self.assertEqual(self._sealed_work_entries(slot), [])
        loaded = load_active_generation(
          slot,
          expected_identity=old_identity,
          expected_members=members)
        self.assertEqual(loaded.generation, old.generation)
        self.assertEqual(
          (loaded.member("chain").path.read_bytes(),
           loaded.member("facts").path.read_bytes()),
          (b"chain-old", b"facts-old"))

  def test_compare_and_swap_refuses_a_stale_writer(self):
    slot = self._slot("cas")
    first, _, roles = self._publish(slot, "first")
    stale = self._draft(slot, {"payload.bin": b"stale"})
    winner = self._draft(slot, {"payload.bin": b"winner"})
    winner_identity = {"dataset": "winner"}
    winner_active = publish_dataset_generation(
      winner,
      identity=winner_identity,
      members={"payload": "payload.bin"},
      expected_active_sha256=first.active_sha256)

    with self.assertRaisesRegex(DatasetPublicationError, "changed"):
      publish_dataset_generation(
        stale,
        identity={"dataset": "stale"},
        members={"payload": "payload.bin"},
        expected_active_sha256=first.active_sha256)
    loaded = load_active_generation(
      slot, expected_identity=winner_identity,
      expected_members={"payload": "payload.bin"})
    self.assertEqual(loaded.generation, winner_active.generation)
    self.assertEqual(loaded.member("payload").path.read_bytes(), b"winner")
    self.assertNotEqual(set(roles), {"payload"})

  def test_compare_and_swap_detects_generation_only_active_mutation(self):
    slot = self._slot("cas_active_record")
    first, _, _ = self._publish(slot, "first")
    pending = self._draft(slot, {"payload.bin": b"pending"})
    pointer = json.loads(slot.active_path.read_text(encoding="utf-8"))
    pointer["generation"] = "gen-" + ("0" * 32)
    self._replace_read_only_file(
      slot.active_path, canonical_json_bytes(pointer))

    with self.assertRaisesRegex(DatasetPublicationError,
                                "active generation changed"):
      publish_dataset_generation(
        pending,
        identity={"dataset": "pending"},
        members={"payload": "payload.bin"},
        expected_active_sha256=first.active_sha256)
    self.assertTrue(pending.path.exists())

  def test_successful_publish_removes_mutable_source_draft(self):
    slot = self._slot("source_removed")
    draft = self._draft(slot, {"payload.bin": b"sealed"})
    source_path = draft.path
    active = publish_dataset_generation(
      draft,
      identity={"dataset": "source-removed"},
      members={"payload": "payload.bin"},
      expected_active_sha256=None)

    self.assertFalse(os.path.lexists(source_path))
    loaded = load_active_generation(
      slot,
      expected_identity={"dataset": "source-removed"},
      expected_members={"payload": "payload.bin"})
    self.assertEqual(loaded.generation, active.generation)

  def test_cleanup_failure_and_warning_error_do_not_hide_commit(self):
    slot = self._slot("cleanup_warning")
    draft = self._draft(slot, {"payload.bin": b"committed"})
    with mock.patch.object(
        publication, "discard_dataset_draft",
        side_effect=DatasetPublicationError("cleanup blocked")):
      with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        active = publish_dataset_generation(
          draft,
          identity={"dataset": "cleanup-warning"},
          members={"payload": "payload.bin"},
          expected_active_sha256=None)

    self.assertIsInstance(active, publication.ActiveGeneration)
    self.assertTrue(draft.path.exists())
    loaded = load_active_generation(
      slot,
      expected_identity={"dataset": "cleanup-warning"},
      expected_members={"payload": "payload.bin"})
    self.assertEqual(loaded.generation, active.generation)
    self.assertEqual(loaded.member("payload").path.read_bytes(), b"committed")

    release_slot = self._slot("release_cleanup")
    release_draft = self._draft(
      release_slot, {"payload.bin": b"release-committed"})
    real_release = publication._release_publish_lock

    def release_then_report(descriptor):
      real_release(descriptor)
      raise DatasetPublicationError("injected release diagnostic")

    with mock.patch.object(
        publication, "_release_publish_lock",
        side_effect=release_then_report):
      released_active = publish_dataset_generation(
        release_draft,
        identity={"dataset": "release-cleanup"},
        members={"payload": "payload.bin"},
        expected_active_sha256=None)
    released_loaded = load_active_generation(
      release_slot,
      expected_identity={"dataset": "release-cleanup"},
      expected_members={"payload": "payload.bin"})
    self.assertEqual(released_loaded.generation, released_active.generation)

  def test_retained_writable_source_fd_cannot_change_published_copy(self):
    slot = self._slot("retained_fd")
    draft = self._draft(slot, {"payload.bin": b"authenticated"})
    source_path = draft.member_path("payload.bin")
    descriptor = os.open(str(source_path), os.O_RDWR)
    try:
      active = publish_dataset_generation(
        draft,
        identity={"dataset": "retained-fd"},
        members={"payload": "payload.bin"},
        expected_active_sha256=None)
      os.lseek(descriptor, 0, os.SEEK_SET)
      os.ftruncate(descriptor, 0)
      os.write(descriptor, b"attacker-after-publication")
      os.fsync(descriptor)
    finally:
      os.close(descriptor)

    loaded = load_active_generation(
      slot,
      expected_identity={"dataset": "retained-fd"},
      expected_members={"payload": "payload.bin"})
    self.assertEqual(loaded.generation, active.generation)
    self.assertEqual(loaded.member("payload").path.read_bytes(),
                     b"authenticated")

  def test_pinned_reader_stays_on_a_while_fresh_reader_sees_b(self):
    slot = self._slot("pinned_reader")
    first, first_identity, members = self._publish(slot, "first")
    pinned = load_active_generation(
      slot,
      expected_identity=first_identity,
      expected_members=members)
    pinned_paths = tuple(member.path for member in pinned.members)
    pinned_payloads = tuple(path.read_bytes() for path in pinned_paths)

    second, second_identity, _ = self._publish(
      slot, "second", expected=first.active_sha256)
    fresh = load_active_generation(
      slot,
      expected_identity=second_identity,
      expected_members=members)

    self.assertEqual(tuple(path.read_bytes() for path in pinned_paths),
                     pinned_payloads)
    self.assertEqual(pinned.generation, first.generation)
    self.assertEqual(fresh.generation, second.generation)
    self.assertNotEqual(pinned.generation, fresh.generation)
    self.assertEqual(fresh.member("chain").path.read_bytes(), b"chain-second")

  def test_oversize_manifest_refuses_before_active_switch(self):
    slot = self._slot("oversize")
    old, old_identity, old_members = self._publish(slot, "old")
    pending = self._draft(slot, {"payload.bin": b"pending"})
    checkpoints = []
    oversize_identity = {
      "dataset": "oversize",
      "padding": "x" * (8 * 1024 * 1024),
    }
    with self.assertRaisesRegex(DatasetPublicationError,
                                "manifest exceeds"):
      publish_dataset_generation(
        pending,
        identity=oversize_identity,
        members={"payload": "payload.bin"},
        expected_active_sha256=old.active_sha256,
        checkpoint=checkpoints.append)

    self.assertEqual(checkpoints, [])
    self.assertTrue(pending.path.exists())
    self.assertEqual(self._sealed_work_entries(slot), [])
    loaded = load_active_generation(
      slot,
      expected_identity=old_identity,
      expected_members=old_members)
    self.assertEqual(loaded.generation, old.generation)

  def test_draft_durable_callback_failure_cleans_sealed_not_source(self):
    slot = self._slot("draft_durable_cleanup")
    old, old_identity, members = self._publish(slot, "old")
    pending = self._draft(slot, {
      "params.1.txt": b"chain-pending",
      "metadata/facts.yaml": b"facts-pending",
    })

    def stop_at_draft_durable(name):
      if name == "draft-durable":
        raise InjectedPublicationCrash(name)

    with self.assertRaisesRegex(InjectedPublicationCrash, "draft-durable"):
      publish_dataset_generation(
        pending,
        identity={"dataset": "pending"},
        members=members,
        expected_active_sha256=old.active_sha256,
        checkpoint=stop_at_draft_durable)

    self.assertTrue(pending.path.exists())
    self.assertEqual(self._sealed_work_entries(slot), [])
    loaded = load_active_generation(
      slot,
      expected_identity=old_identity,
      expected_members=members)
    self.assertEqual(loaded.generation, old.generation)

  def test_sync_order_makes_generation_durable_before_active_switch(self):
    slot = self._slot("sync_order")
    identity = {"dataset": "sync-order", "seed": 17}
    members = {
      "chain": "params.1.txt",
      "facts": "metadata/facts.yaml",
    }
    draft = self._draft(slot, {
      "params.1.txt": b"chain-sync-order",
      "metadata/facts.yaml": b"facts-sync-order",
    })
    events = []
    real_file_sync = publication._fsync_regular_file
    real_directory_sync = publication._fsync_directory
    real_fchmod = publication.os.fchmod
    real_rename = publication.os.rename
    real_replace = publication.os.replace

    def descriptor_path(descriptor):
      descriptor_status = os.fstat(descriptor)
      identity = (descriptor_status.st_dev, descriptor_status.st_ino)
      for candidate in slot.path.rglob("*"):
        try:
          candidate_status = os.lstat(candidate)
        except OSError:
          continue
        if (candidate_status.st_dev, candidate_status.st_ino) == identity:
          return candidate, identity
      return None, identity

    def record_file_sync(descriptor):
      result = real_file_sync(descriptor)
      target, identity_key = descriptor_path(descriptor)
      events.append(("file-sync", target, identity_key))
      return result

    def record_directory_sync(path):
      path = Path(path)
      result = real_directory_sync(path)
      events.append(("directory-sync", path, None))
      return result

    def record_fchmod(descriptor, mode):
      result = real_fchmod(descriptor, mode)
      target, identity_key = descriptor_path(descriptor)
      events.append(("file-chmod", target, (identity_key, mode)))
      return result

    def record_rename(source, destination):
      result = real_rename(source, destination)
      events.append(("generation-rename", Path(destination), None))
      return result

    def record_replace(source, destination):
      result = real_replace(source, destination)
      events.append(("active-replace", Path(destination), None))
      return result

    with mock.patch.object(
        publication, "_fsync_regular_file",
        side_effect=record_file_sync), mock.patch.object(
          publication, "_fsync_directory",
          side_effect=record_directory_sync), mock.patch.object(
            publication.os, "fchmod",
            side_effect=record_fchmod), mock.patch.object(
              publication.os, "rename",
              side_effect=record_rename), mock.patch.object(
                publication.os, "replace", side_effect=record_replace):
      active = publish_dataset_generation(
        draft,
        identity=identity,
        members=members,
        expected_active_sha256=None)

    final_path = slot.generations_path / active.generation
    rename_indexes = [
      index for index, event in enumerate(events)
      if event == ("generation-rename", final_path, None)]
    replace_indexes = [
      index for index, event in enumerate(events)
      if event == ("active-replace", slot.active_path, None)]
    self.assertEqual(len(rename_indexes), 1)
    self.assertEqual(len(replace_indexes), 1)
    rename_index = rename_indexes[0]
    replace_index = replace_indexes[0]
    self.assertLess(rename_index, replace_index)

    def require_final_mode_sync(*, filename, parent, before_index):
      chmods = [
        (index, target, detail)
        for index, (kind, target, detail) in enumerate(events[:before_index])
        if kind == "file-chmod" and target is not None
        and target.name == filename and parent in target.parents
        and detail[1] == 0o444
      ]
      self.assertEqual(len(chmods), 1)
      chmod_index, _, (identity_key, _) = chmods[0]
      self.assertTrue(any(
        index > chmod_index and index < before_index
        and kind == "file-sync" and detail == identity_key
        for index, (kind, _, detail) in enumerate(events)))

    for sealed_name in ("params.1.txt", "facts.yaml", "manifest.json"):
      with self.subTest(final_mode_sync=sealed_name):
        require_final_mode_sync(
          filename=sealed_name,
          parent=slot.work_path,
          before_index=rename_index)

    installed_interval = events[rename_index + 1:replace_index]
    directory_syncs = {
      target for kind, target, _ in installed_interval
      if kind == "directory-sync"
    }
    self.assertTrue({
      final_path,
      final_path / "files",
      final_path / "files" / "metadata",
      slot.work_path,
      slot.generations_path,
    }.issubset(directory_syncs))
    active_chmods = [
      (index, target, detail)
      for index, (kind, target, detail) in enumerate(events[:replace_index])
      if kind == "file-chmod" and target is not None
      and target.parent == slot.path
      and target.name.startswith(".active-")
      and target.name.endswith(".tmp") and detail[1] == 0o444
    ]
    self.assertEqual(len(active_chmods), 1)
    active_chmod_index, _, (active_identity, _) = active_chmods[0]
    self.assertGreater(active_chmod_index, rename_index)
    self.assertTrue(any(
      index > active_chmod_index and index < replace_index
      and kind == "file-sync" and detail == active_identity
      for index, (kind, _, detail) in enumerate(events)))
    self.assertEqual(
      events[replace_index + 1], ("directory-sync", slot.path, None))
    loaded = load_active_generation(
      slot, expected_identity=identity, expected_members=members)
    self.assertEqual(loaded.generation, active.generation)

  def test_directory_creation_retry_resyncs_existing_parent(self):
    parent = self.root / "private-parent"
    parent.mkdir(mode=0o700)
    child = parent / "child"
    real_directory_sync = publication._fsync_directory
    first_attempt = []

    def fail_parent_sync(path):
      path = Path(path)
      first_attempt.append(path)
      if path == parent:
        raise DatasetPublicationError("injected parent sync failure")
      return real_directory_sync(path)

    with mock.patch.object(
        publication, "_fsync_directory", side_effect=fail_parent_sync):
      with self.assertRaisesRegex(
          DatasetPublicationError, "injected parent sync failure"):
        publication._ensure_private_directory(child)

    self.assertTrue(child.is_dir())
    self.assertEqual(first_attempt, [child, parent])
    retry = []

    def record_retry(path):
      path = Path(path)
      retry.append(path)
      return real_directory_sync(path)

    with mock.patch.object(
        publication, "_fsync_directory", side_effect=record_retry):
      publication._ensure_private_directory(child)

    self.assertEqual(retry, [child, parent])

  def test_durability_helpers_call_platform_primitives(self):
    file_path = self.root / "durability-primitives.bin"
    descriptor = os.open(file_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    self.addCleanup(os.close, descriptor)
    os.write(descriptor, b"durability-primitives")
    real_fsync = publication.os.fsync
    real_fcntl = publication.fcntl.fcntl
    events = []

    def record_fsync(target_descriptor):
      result = real_fsync(target_descriptor)
      target_status = os.fstat(target_descriptor)
      kind = "directory" if stat.S_ISDIR(target_status.st_mode) else "file"
      events.append(("fsync", kind, target_descriptor))
      return result

    def record_fcntl(target_descriptor, command):
      result = real_fcntl(target_descriptor, command)
      events.append(("fcntl", command, target_descriptor))
      return result

    with mock.patch.object(
        publication.os, "fsync", side_effect=record_fsync), mock.patch.object(
          publication.fcntl, "fcntl", side_effect=record_fcntl):
      publication._fsync_regular_file(descriptor)
      publication._fsync_directory(self.root)

    self.assertIn(("fsync", "file", descriptor), events)
    self.assertTrue(any(
      kind == "fsync" and detail == "directory"
      for kind, detail, _ in events))
    full_fsync = getattr(publication.fcntl, "F_FULLFSYNC", None)
    if full_fsync is not None:
      self.assertIn(("fcntl", full_fsync, descriptor), events)
    else:
      self.assertFalse(any(kind == "fcntl" for kind, _, _ in events))

  def test_published_generation_is_read_only_and_writable_tree_is_refused(self):
    slot = self._slot("readonly")
    active, identity, roles = self._publish(slot, "readonly")
    generation_path, manifest_path = self._manifest_paths(active)
    checked = [
      generation_path,
      generation_path / "files",
      active.member("facts").path.parent,
      manifest_path,
      slot.active_path,
    ]
    checked.extend(member.path for member in active.members)
    for path in checked:
      with self.subTest(path=path):
        self.assertEqual(os.lstat(str(path)).st_mode & 0o222, 0)

    os.chmod(str(generation_path), 0o755, follow_symlinks=False)
    try:
      with self.assertRaisesRegex(DatasetPublicationError, "writable"):
        load_active_generation(
          slot, expected_identity=identity,
          expected_members=roles)
    finally:
      os.chmod(str(generation_path), 0o555, follow_symlinks=False)

  def test_callback_exception_boundaries_expose_no_in_process_hybrid(self):
    # Raising from a callback lets us inspect each operation boundary in this
    # process.  It is not a simulation of power loss or filesystem recovery.
    boundaries = (
      "draft-durable",
      "generation-installed",
      "active-temp-durable",
      "active-replaced",
      "active-directory-durable",
    )
    for index, boundary in enumerate(boundaries):
      with self.subTest(boundary=boundary):
        slot = self._slot("crash_" + str(index))
        old, old_identity, members = self._publish(slot, "old")
        new_identity = {"dataset": "new", "boundary": boundary}
        draft = self._draft(slot, {
          "params.1.txt": b"chain-new",
          "metadata/facts.yaml": b"facts-new",
        })
        reached = []

        def stop_at_boundary(name):
          reached.append(name)
          if name == boundary:
            raise InjectedPublicationCrash(name)

        with self.assertRaisesRegex(InjectedPublicationCrash, boundary):
          publish_dataset_generation(
            draft,
            identity=new_identity,
            members=members,
            expected_active_sha256=old.active_sha256,
            checkpoint=stop_at_boundary)

        exposes_new = boundary in (
          "active-replaced", "active-directory-durable")
        visible_identity = new_identity if exposes_new else old_identity
        loaded = load_active_generation(
          slot,
          expected_identity=visible_identity,
          expected_members=members)
        self.assertIn(loaded.identity, (old_identity, new_identity))
        self.assertEqual(loaded.identity, visible_identity)
        observed_pair = (
          loaded.member("chain").path.read_bytes(),
          loaded.member("facts").path.read_bytes())
        self.assertIn(
          observed_pair,
          ((b"chain-old", b"facts-old"),
           (b"chain-new", b"facts-new")))
        if not exposes_new:
          self.assertEqual(loaded.generation, old.generation)
        else:
          self.assertEqual(observed_pair, (b"chain-new", b"facts-new"))
        self.assertEqual(reached[-1], boundary)


if __name__ == "__main__":
  unittest.main()

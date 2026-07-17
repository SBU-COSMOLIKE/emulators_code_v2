"""CPU tests for resolving logical Cocoa filenames to immutable datasets."""

import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

from compute_data_vectors.dataset_manifest import (
  DATASET_PROBE_GENERATORS,
  UNIFORM_BOUNDARY_INTERIOR_POLICY,
  build_dataset_member_map,
  build_dataset_request_identity,
)
from compute_data_vectors.dataset_publication import (
  begin_dataset_generation,
  derive_dataset_slot,
  install_dataset_locator,
  publish_dataset_generation,
)
from emulator import cocoa


_ROUTES = {
  "cosmolike": ("cs", "dataset_generator_lensing"),
  "cmb": ("cmblensed", "dataset_generator_cmb"),
  "grid": ("background", "dataset_generator_background"),
  "grid2d": ("mps", "dataset_generator_mps"),
}


class CocoaDatasetResolutionTests(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.rootdir = Path(self.temporary.name)
    self.project = self.rootdir / "projects" / "science"
    self.chains = self.project / "chains"
    self.fileroot = self.project / "emulators" / "training"
    self.chains.mkdir(parents=True)
    self.fileroot.mkdir(parents=True)
    self.yaml_path = self.fileroot / "run.yaml"
    self.args = SimpleNamespace(
      root="projects/science",
      fileroot="emulators/training",
      yaml="run.yaml",
    )
    self.environment = mock.patch.dict(
      os.environ, {"ROOTDIR": str(self.rootdir)})
    self.environment.start()
    self.addCleanup(self.environment.stop)
    # Publication deliberately removes write bits. Restore them before the
    # temporary-directory cleanup walks the tree.
    self.addCleanup(self._make_tree_writable)

  def _make_tree_writable(self):
    if not self.rootdir.exists():
      return
    for directory, names, files in os.walk(
        str(self.rootdir), topdown=True, followlinks=False):
      paths = [Path(directory)] + [Path(directory) / name
                                   for name in names + files]
      for path in paths:
        try:
          status = os.lstat(path)
          if stat.S_ISDIR(status.st_mode):
            os.chmod(path, 0o700, follow_symlinks=False)
          elif stat.S_ISREG(status.st_mode):
            os.chmod(path, 0o600, follow_symlinks=False)
        except (FileNotFoundError, NotImplementedError, OSError):
          pass

  def _identity(self, family, *, mode="full", variant="standard", seed=17,
                probe=None, ordered_names=None, scientific_contract=None,
                sampling_mode="uniform"):
    default_probe, _ = _ROUTES[family]
    probe = default_probe if probe is None else probe
    generator = DATASET_PROBE_GENERATORS[probe]
    if sampling_mode == "uniform":
      algorithm = "uniform-box-v1"
      max_correlation = None
      emcee_random = None
      interior_policy = UNIFORM_BOUNDARY_INTERIOR_POLICY
    elif sampling_mode == "gaussian-mcmc":
      algorithm = "emcee-de-snooker-v1"
      max_correlation = 0.15
      emcee_random = "MT19937"
      interior_policy = None
    else:
      raise AssertionError("unsupported test sampling mode "
                           + repr(sampling_mode))
    return build_dataset_request_identity(
      dataset_mode=mode,
      family=family,
      family_variant=variant,
      generator=generator,
      probe=probe,
      sampling_mode=sampling_mode,
      temperature=64,
      boundary_factor=1.0,
      max_correlation=max_correlation,
      sampling_algorithm=algorithm,
      seed=seed,
      rng_bit_generator="PCG64",
      rng_emcee_random=emcee_random,
      rng_policy="persist-complete-state-v1",
      boundary_interior_policy=interior_policy,
      ordered_names=(
        ["omegabh2", "H0", "ns"]
        if ordered_names is None else ordered_names),
      configuration_sha256=("1" if seed == 17 else "2") * 64,
      scientific_contract_sha256=(
        "a" * 64 if scientific_contract is None else scientific_contract),
    )

  def _publish(self, tag, family, *, mode="full", variant="standard",
               seed=17, axis_payload=b"shared-axis\n", probe=None,
               ordered_names=None, scientific_contract=None,
               sampling_mode="uniform"):
    identity = self._identity(
      family,
      mode=mode,
      variant=variant,
      seed=seed,
      probe=probe,
      ordered_names=ordered_names,
      scientific_contract=scientific_contract,
      sampling_mode=sampling_mode,
    )
    params_stem = "params_" + tag
    dvs_stem = "dvs_" + tag
    fail_stem = "fail_" + tag
    members = build_dataset_member_map(
      identity,
      params_stem=params_stem,
      dvs_stem=dvs_stem,
      fail_stem=fail_stem,
    )
    slot = derive_dataset_slot(
      self.chains,
      params_stem=str(self.chains / params_stem),
      dvs_stem=str(self.chains / dvs_stem),
      fail_stem=str(self.chains / fail_stem),
      dataset_mode=mode,
      family=family,
    )
    draft = begin_dataset_generation(slot)
    for role, relative_path in members.items():
      payload = axis_payload if role.startswith("axis.") else (
        tag + ":" + role + "\n").encode("ascii")
      draft.member_path(relative_path).write_bytes(payload)
    active = publish_dataset_generation(
      draft,
      identity=identity,
      members=members,
      expected_active_sha256=None,
    )
    locator = install_dataset_locator(
      slot, identity=identity, members=members)
    return SimpleNamespace(
      identity=identity,
      members=members,
      active=active,
      locator=locator,
      slot=slot,
    )

  def _write_config(self, data):
    self.yaml_path.write_text(
      yaml.safe_dump({"data": data, "train_args": {}}, sort_keys=False),
      encoding="utf-8",
    )

  @staticmethod
  def _common_data(train, validation):
    return {
      "train_params": train.members["parameters.chain"],
      "val_params": validation.members["parameters.chain"],
      "train_covmat": train.members["parameters.covariance"],
      "n_train": 8,
      "n_val": 4,
      "split_seed": 3,
    }

  def _resolved(self):
    return cocoa.resolve_cocoa_config(self.args)[0]

  def test_cosmolike_paths_and_failure_masks_come_from_two_pins(self):
    train = self._publish("train", "cosmolike")
    validation = self._publish("validation", "cosmolike", seed=18)
    data = self._common_data(train, validation)
    data.update({
      "train_dv": train.members["payload.cosmolike.vector"],
      "val_dv": validation.members["payload.cosmolike.vector"],
    })
    self._write_config(data)

    original = cocoa.load_located_generation
    with mock.patch.object(
        cocoa, "load_located_generation", wraps=original) as load:
      resolved = self._resolved()["data"]

    self.assertEqual(load.call_count, 2)
    self.assertEqual(
      resolved["train_params"],
      str(train.active.member("parameters.chain").path))
    self.assertEqual(
      resolved["val_params"],
      str(validation.active.member("parameters.chain").path))
    self.assertEqual(
      resolved["train_covmat"],
      str(train.active.member("parameters.covariance").path))
    self.assertEqual(
      resolved["train_dv"],
      str(train.active.member("payload.cosmolike.vector").path))
    self.assertEqual(
      resolved["val_dv"],
      str(validation.active.member("payload.cosmolike.vector").path))
    self.assertEqual(
      resolved["train_failure_mask"],
      str(train.active.member("rows.failure-mask").path))
    self.assertEqual(
      resolved["val_failure_mask"],
      str(validation.active.member("rows.failure-mask").path))

    pins = resolved["_dataset_sources"]
    self.assertEqual(pins["schema"], 1)
    self.assertEqual(pins["train"]["generation"], train.active.generation)
    self.assertEqual(
      pins["validation"]["generation"], validation.active.generation)
    self.assertEqual(
      pins["train"]["members"]["parameters.chain"]["sha256"],
      train.active.member("parameters.chain").sha256)
    # All injected values remain plain mappings, lists, strings, and numbers.
    yaml.safe_dump(pins)

  def test_missing_locator_refuses_even_when_legacy_flat_files_exist(self):
    logical = {
      "train_params": "params_train.1.txt",
      "val_params": "params_validation.1.txt",
      "train_covmat": "params_train.covmat",
      "train_dv": "dvs_train.npy",
      "val_dv": "dvs_validation.npy",
      "n_train": 8,
      "n_val": 4,
      "split_seed": 3,
    }
    for name in logical.values():
      if isinstance(name, str):
        (self.chains / name).write_bytes(b"legacy-flat-file\n")
    self._write_config(logical)

    with self.assertRaisesRegex(Exception, "publication root|locator"):
      self._resolved()

  def test_source_yaml_cannot_supply_resolver_owned_identity(self):
    data = {
      "train_params": "params_train.1.txt",
      "val_params": "params_validation.1.txt",
      "train_covmat": "params_train.covmat",
      "_dataset_sources": {"forged": True},
    }
    self._write_config(data)
    with self.assertRaisesRegex(ValueError, "must not appear in source YAML"):
      self._resolved()

  def test_chain_only_scalar_rewrites_no_payload_or_failure_mask(self):
    train = self._publish(
      "scalar_train", "cosmolike", mode="chain-only")
    validation = self._publish(
      "scalar_validation", "cosmolike", mode="chain-only", seed=18)
    data = self._common_data(train, validation)
    data["outputs"] = ["sigma8"]
    self._write_config(data)

    resolved = self._resolved()["data"]
    self.assertEqual(
      resolved["train_params"],
      str(train.active.member("parameters.chain").path))
    self.assertEqual(
      resolved["train_covmat"],
      str(train.active.member("parameters.covariance").path))
    for key in ("train_dv", "val_dv", "train_failure_mask",
                "val_failure_mask"):
      self.assertNotIn(key, resolved)
    self.assertEqual(
      resolved["_dataset_sources"]["train"]["identity"]["dataset_mode"],
      "chain-only")

  def test_cmb_selects_one_spectrum_and_requires_shared_multipoles(self):
    train = self._publish("cmb_train", "cmb")
    validation = self._publish("cmb_validation", "cmb", seed=18)
    data = self._common_data(train, validation)
    data.update({
      "train_dv": train.members["payload.cmb.ee"],
      "val_dv": validation.members["payload.cmb.ee"],
      "cmb": {
        "spectrum": "ee",
        "covariance": "cmb_covariance.npz",
        "amplitude_law": "none",
      },
    })
    self._write_config(data)

    resolved = self._resolved()["data"]
    self.assertEqual(
      resolved["train_dv"],
      str(train.active.member("payload.cmb.ee").path))
    self.assertIn(
      "axis.cmb.multipole",
      resolved["_dataset_sources"]["train"]["members"])

  def test_background_and_grid2d_axes_and_bases_are_generation_paths(self):
    cases = []

    grid_train = self._publish("grid_train", "grid")
    grid_val = self._publish("grid_validation", "grid", seed=18)
    grid_data = self._common_data(grid_train, grid_val)
    grid_data.update({
      "train_dv": grid_train.members["payload.grid.dm"],
      "val_dv": grid_val.members["payload.grid.dm"],
      "grid": {
        "quantity": "D_M",
        "units": "Mpc",
        "law": "none",
        "z_file": grid_train.members["axis.grid.dm.redshift"],
      },
    })
    cases.append((grid_data, "grid", "z_file", grid_train,
                  "axis.grid.dm.redshift"))

    mps_train = self._publish(
      "mps_train", "grid2d", variant="syren-base")
    mps_val = self._publish(
      "mps_validation", "grid2d", variant="syren-base", seed=18)
    mps_data = self._common_data(mps_train, mps_val)
    mps_data.update({
      "train_dv": mps_train.members["payload.grid2d.boost"],
      "val_dv": mps_val.members["payload.grid2d.boost"],
      "grid2d": {
        "quantity": "boost",
        "units": "dimensionless",
        "law": "syren_halofit",
        "z_file": mps_train.members["axis.grid2d.redshift"],
        "k_file": mps_train.members["axis.grid2d.wavenumber"],
        "train_base": mps_train.members["base.grid2d.boost"],
        "val_base": mps_val.members["base.grid2d.boost"],
      },
    })
    cases.append((mps_data, "grid2d", "z_file", mps_train,
                  "axis.grid2d.redshift"))

    for data, block, axis_key, train, role in cases:
      with self.subTest(block=block):
        self._write_config(data)
        resolved = self._resolved()["data"]
        self.assertEqual(
          resolved[block][axis_key], str(train.active.member(role).path))
        if block == "grid2d":
          self.assertEqual(
            resolved[block]["k_file"],
            str(train.active.member("axis.grid2d.wavenumber").path))
          self.assertEqual(
            resolved[block]["train_base"],
            str(train.active.member("base.grid2d.boost").path))
          self.assertEqual(
            resolved[block]["val_base"],
            str(mps_val.active.member("base.grid2d.boost").path))

  def test_train_and_validation_scientific_meaning_must_match(self):
    cases = (
      ("probe", {"probe": "ggl"}, "different probes"),
      ("parameter-order",
       {"ordered_names": ["H0", "omegabh2", "ns"]},
       "different ordered parameter names"),
      ("scientific-contract",
       {"scientific_contract": "b" * 64},
       "different scientific contracts"),
    )
    for label, validation_change, message in cases:
      with self.subTest(case=label):
        train = self._publish(label + "_train", "cosmolike")
        validation = self._publish(
          label + "_validation",
          "cosmolike",
          seed=18,
          **validation_change,
        )
        data = self._common_data(train, validation)
        data.update({
          "train_dv": train.members["payload.cosmolike.vector"],
          "val_dv": validation.members["payload.cosmolike.vector"],
        })
        self._write_config(data)
        with self.assertRaisesRegex(ValueError, message):
          self._resolved()

  def test_train_and_validation_may_use_different_sampling_procedures(self):
    train = self._publish("sampling_train", "cosmolike")
    validation = self._publish(
      "sampling_validation",
      "cosmolike",
      seed=18,
      sampling_mode="gaussian-mcmc",
    )
    data = self._common_data(train, validation)
    data.update({
      "train_dv": train.members["payload.cosmolike.vector"],
      "val_dv": validation.members["payload.cosmolike.vector"],
    })
    self._write_config(data)

    resolved = self._resolved()["data"]
    sources = resolved["_dataset_sources"]
    self.assertEqual(
      sources["train"]["identity"]["sampling"]["mode"], "uniform")
    self.assertEqual(
      sources["validation"]["identity"]["sampling"]["mode"],
      "gaussian-mcmc")

  def test_axis_mismatch_and_cross_dataset_payload_are_refused(self):
    train = self._publish("axis_train", "grid", axis_payload=b"axis-a\n")
    validation = self._publish(
      "axis_validation", "grid", seed=18, axis_payload=b"axis-b\n")
    data = self._common_data(train, validation)
    data.update({
      "train_dv": train.members["payload.grid.h"],
      "val_dv": validation.members["payload.grid.h"],
      "grid": {
        "quantity": "Hubble",
        "units": "km/s/Mpc",
        "law": "none",
        "z_file": train.members["axis.grid.h.redshift"],
      },
    })
    self._write_config(data)
    with self.assertRaisesRegex(ValueError, "different axis.grid.h.redshift"):
      self._resolved()

    matching = self._publish("matching_validation", "cosmolike", seed=18)
    train_cosmo = self._publish("matching_train", "cosmolike")
    mixed = self._common_data(train_cosmo, matching)
    mixed.update({
      "train_dv": matching.members["payload.cosmolike.vector"],
      "val_dv": matching.members["payload.cosmolike.vector"],
    })
    self._write_config(mixed)
    with self.assertRaisesRegex(ValueError, "assigns role"):
      self._resolved()


if __name__ == "__main__":
  unittest.main()

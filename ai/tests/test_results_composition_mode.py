"""Required artifact-composition facts and pre-construction validation.

An artifact's network state has the same shape whether its output is decoded
plainly, recombined with an NPCE base, or composed with a transfer base.  The
mode therefore has to be a required persisted fact; optional-group presence is
not a safe discriminator.  These tests pin the writer-side executed matrix and
the HDF5 reader's exact required/forbidden-group matrix without constructing a
model.  One integration witness additionally proves an old or damaged schema-v3
file is refused before a geometry constructor or ``torch.load`` can run.
"""

import contextlib
import os
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch
import yaml

from emulator import fixed_facts
from emulator import inference
from emulator import results


_MISSING = object()
_AUTO = object()


def _resolved_record(mode, refined):
  """Return the consumed top-level composition record for one valid mode."""
  pce = {"form": "residual"} if mode == "npce" else None
  transfer = None
  if mode == "transfer":
    transfer = {"from": "base", "form": "gain", "space": "physical"}
    if refined:
      transfer["refine"] = {"epochs": 1,
                            "base_lr_scale": 0.1,
                            "anchor": 1.0}
  return {"data": {},
          "train_args": {},
          "composition_mode": mode,
          "transfer_refined": refined,
          "pce": pce,
          "transfer": transfer}


@contextlib.contextmanager
def _composition_file(*,
                      mode="plain",
                      refined=False,
                      pce_group=False,
                      transfer_group=False,
                      drifted_group=False,
                      resolved=_AUTO):
  """Yield one in-memory HDF5 composition surface.

  The fixture deliberately writes no model or geometry payload.  It exercises
  the composition reader as a pure HDF5 boundary, so all corruptions are cheap
  and each matrix row has only the fact under test changed.
  """
  with h5py.File("composition-mode.h5", "w", driver="core",
                 backing_store=False) as artifact:
    if mode is not _MISSING:
      artifact.attrs["composition_mode"] = mode
    if refined is not _MISSING:
      artifact.attrs["transfer_refined"] = refined
    if pce_group:
      pce = artifact.create_group("pce")
      pce.attrs["form"] = "residual"
    if transfer_group:
      transfer = artifact.create_group("transfer_base")
      transfer.attrs["form"] = "gain"
      transfer.attrs["space"] = "physical"
      if drifted_group:
        transfer.create_group("drifted_state")
    if resolved is not _MISSING:
      if resolved is _AUTO:
        resolved_mode = mode if isinstance(mode, str) \
            and mode in ("plain", "npce", "transfer") else "plain"
        resolved_refined = refined if isinstance(refined, (bool, np.bool_)) \
            else False
        resolved = _resolved_record(resolved_mode, bool(resolved_refined))
      artifact.create_dataset(
        "config_resolved_yaml",
        data=yaml.safe_dump(resolved, sort_keys=False),
        dtype=h5py.string_dtype(encoding="utf-8"))
    yield artifact


class ArtifactCompositionReaderTest(unittest.TestCase):
  """Pin the authoritative enum, refined fact, groups, and corroboration."""

  def test_four_valid_rows(self):
    """Plain, NPCE, frozen transfer, and refined transfer are complete."""
    rows = (
      ("plain", False, False, False, False),
      ("npce", False, True, False, False),
      ("transfer", False, False, True, False),
      ("transfer", True, False, True, True),
    )
    for mode, refined, have_pce, have_transfer, have_drifted in rows:
      with self.subTest(mode=mode, refined=refined):
        with _composition_file(
            mode=mode,
            refined=refined,
            pce_group=have_pce,
            transfer_group=have_transfer,
            drifted_group=have_drifted) as artifact:
          self.assertEqual(
            results._read_artifact_composition(artifact, "fixture.h5"),
            (mode, refined))

  def test_required_and_forbidden_group_matrix(self):
    """Every group is checked in both directions, including mutual exclusion."""
    rows = (
      ("plain-pce", "plain", False, True, False, False),
      ("plain-transfer", "plain", False, False, True, False),
      ("npce-missing-pce", "npce", False, False, False, False),
      ("npce-second-group", "npce", False, True, True, False),
      ("transfer-missing-base", "transfer", False, False, False, False),
      ("transfer-second-group", "transfer", False, True, True, False),
      ("frozen-has-drift", "transfer", False, False, True, True),
      ("refined-missing-drift", "transfer", True, False, True, False),
      ("plain-marked-refined", "plain", True, False, False, False),
      ("npce-marked-refined", "npce", True, True, False, False),
    )
    for label, mode, refined, have_pce, have_transfer, have_drifted in rows:
      with self.subTest(label=label):
        with _composition_file(
            mode=mode,
            refined=refined,
            pce_group=have_pce,
            transfer_group=have_transfer,
            drifted_group=have_drifted) as artifact:
          with self.assertRaisesRegex(
              (KeyError, ValueError),
              "composition|plain|npce|transfer|pce|drifted|refined"):
            results._read_artifact_composition(artifact, "fixture.h5")

  def test_attributes_are_required_native_values(self):
    """Missing, unknown, byte, truthy-string, and integer facts are refused."""
    rows = (
      ("missing-mode", _MISSING, False, "composition_mode"),
      ("unknown-mode", "hybrid", False, "composition_mode"),
      ("byte-mode", b"plain", False, "composition_mode"),
      ("integer-mode", 1, False, "composition_mode"),
      ("boolean-mode", True, False, "composition_mode"),
      ("missing-refined", "plain", _MISSING, "transfer_refined"),
      ("string-refined", "plain", "False", "transfer_refined"),
      ("integer-refined-zero", "plain", 0, "transfer_refined"),
      ("integer-refined-one", "transfer", 1, "transfer_refined"),
    )
    for label, mode, refined, needle in rows:
      with self.subTest(label=label):
        with _composition_file(mode=mode, refined=refined) as artifact:
          with self.assertRaisesRegex((KeyError, ValueError), needle):
            results._read_artifact_composition(artifact, "fixture.h5")

  def test_resolved_yaml_fields_are_required_and_corroborate(self):
    """The consumed record corroborates both facts and both mode blocks."""
    base = _resolved_record("plain", False)
    rows = [("missing-dataset", _MISSING, "config_resolved_yaml")]
    for key in ("composition_mode", "transfer_refined", "pce", "transfer"):
      missing = dict(base)
      del missing[key]
      rows.append(("missing-" + key, missing, key))
    mode_mismatch = dict(base)
    mode_mismatch["composition_mode"] = "npce"
    rows.append(("mode-mismatch", mode_mismatch, "composition_mode"))
    refined_mismatch = dict(base)
    refined_mismatch["transfer_refined"] = True
    rows.append(("refined-mismatch", refined_mismatch, "transfer_refined"))
    pce_mismatch = dict(base)
    pce_mismatch["pce"] = {"form": "residual"}
    rows.append(("plain-pce-record", pce_mismatch, "pce"))
    transfer_mismatch = dict(base)
    transfer_mismatch["transfer"] = {"from": "base"}
    rows.append(("plain-transfer-record", transfer_mismatch, "transfer"))

    for label, resolved, needle in rows:
      with self.subTest(label=label):
        with _composition_file(mode="plain", refined=False,
                               resolved=resolved) as artifact:
          with self.assertRaisesRegex((KeyError, ValueError), needle):
            results._read_artifact_composition(artifact, "fixture.h5")

  def test_payload_submodes_match_the_consumed_record(self):
    """Nested attrs cannot select semantics different from resolved YAML."""
    rows = (
      ("npce-form", "npce", "form", "ratio"),
      ("transfer-form", "transfer", "form", "sum"),
      ("transfer-space", "transfer", "space", "whitened"),
      ("npce-byte-form", "npce", "form", np.bytes_(b"residual")),
    )
    for label, mode, key, value in rows:
      with self.subTest(label=label):
        with _composition_file(
            mode=mode,
            refined=False,
            pce_group=mode == "npce",
            transfer_group=mode == "transfer") as artifact:
          group = artifact["pce" if mode == "npce" else "transfer_base"]
          del group.attrs[key]
          group.attrs[key] = value
          with self.assertRaisesRegex(
              (KeyError, ValueError), "form|space|native|disagrees"):
            results._read_artifact_composition(artifact, "fixture.h5")

  def test_refine_blocks_are_mappings_when_present(self):
    """False-like scalars cannot masquerade as an executed refine block."""
    resolved = _resolved_record("transfer", True)
    resolved["transfer"]["refine"] = False
    with _composition_file(
        mode="transfer", refined=True, transfer_group=True,
        drifted_group=True, resolved=resolved) as artifact:
      with self.assertRaisesRegex(ValueError, "refine.*mapping"):
        results._read_artifact_composition(artifact, "fixture.h5")

    with _composition_file(
        mode="transfer", refined=True, transfer_group=True,
        drifted_group=True) as artifact:
      artifact.create_dataset(
        "config_yaml",
        data=yaml.safe_dump(
          {"transfer": {"form": "gain", "space": "physical",
                        "refine": False}},
          sort_keys=False),
        dtype=h5py.string_dtype(encoding="utf-8"))
      with self.assertRaisesRegex(ValueError, "refine.*mapping"):
        results._read_artifact_composition(artifact, "fixture.h5")

  def test_raw_config_submodes_corroborate_when_declared(self):
    """A non-null provenance block cannot contradict executed submodes."""
    rows = (
      ("npce-form", "npce", False, {"pce": {"form": "ratio"}}),
      ("transfer-form", "transfer", False,
       {"transfer": {"form": "sum", "space": "physical"}}),
      ("transfer-space", "transfer", False,
       {"transfer": {"form": "gain", "space": "whitened"}}),
      ("transfer-refine", "transfer", True,
       {"transfer": {"form": "gain", "space": "physical"}}),
    )
    for label, mode, refined, raw in rows:
      with self.subTest(label=label):
        with _composition_file(
            mode=mode,
            refined=refined,
            pce_group=mode == "npce",
            transfer_group=mode == "transfer",
            drifted_group=refined) as artifact:
          artifact.create_dataset(
            "config_yaml",
            data=yaml.safe_dump(raw, sort_keys=False),
            dtype=h5py.string_dtype(encoding="utf-8"))
          with self.assertRaisesRegex(
              (KeyError, ValueError), "form|space|refine|disagrees"):
            results._read_artifact_composition(artifact, "fixture.h5")


class ExecutedCompositionTest(unittest.TestCase):
  """Pin the writer-side facts against the objects the run actually used."""

  def _validate(self, mode, refined, pce, transfer, *,
                pce_form=_AUTO, resolved_pce=_AUTO,
                resolved_transfer=_AUTO):
    if pce_form is _AUTO:
      pce_form = "residual" if mode == "npce" else None
    if resolved_pce is _AUTO:
      resolved_pce = {"form": "residual"} if mode == "npce" else None
    if resolved_transfer is _AUTO:
      resolved_transfer = None
      if mode == "transfer":
        resolved_transfer = {"from": "base",
                             "form": "gain",
                             "space": "physical"}
        if refined is True:
          resolved_transfer["refine"] = {"epochs": 1}
    return results._validate_executed_composition(
      composition_mode=mode,
      transfer_refined=refined,
      pce=pce,
      pce_form=pce_form,
      transfer_base=transfer,
      resolved_pce=resolved_pce,
      resolved_transfer=resolved_transfer,
      where="executed fixture")

  def test_four_valid_executed_rows(self):
    """The writer accepts exactly the same four rows as the HDF5 reader."""
    rows = (
      ("plain", False, None, None),
      ("npce", False, object(), None),
      ("transfer", False, None,
       {"form": "gain", "space": "physical"}),
      ("transfer", True, None,
       {"form": "gain", "space": "physical",
        "drifted_state": object()}),
    )
    for mode, refined, pce, transfer in rows:
      with self.subTest(mode=mode, refined=refined):
        self._validate(mode, refined, pce, transfer)

  def test_invalid_executed_rows(self):
    """An enum cannot contradict the executed bases or refined state."""
    rows = (
      ("plain-pce", "plain", False, object(), None),
      ("plain-transfer", "plain", False, None,
       {"form": "gain", "space": "physical"}),
      ("plain-refined", "plain", True, None, None),
      ("npce-missing-pce", "npce", False, None, None),
      ("npce-transfer", "npce", False, object(),
       {"form": "gain", "space": "physical"}),
      ("npce-refined", "npce", True, object(), None),
      ("transfer-missing-base", "transfer", False, None, None),
      ("transfer-pce", "transfer", False, object(),
       {"form": "gain", "space": "physical"}),
      ("frozen-has-drift", "transfer", False, None,
       {"form": "gain", "space": "physical",
        "drifted_state": object()}),
      ("refined-missing-drift", "transfer", True, None,
       {"form": "gain", "space": "physical"}),
      ("unknown-mode", "hybrid", False, None, None),
      ("missing-mode", None, False, None, None),
      ("byte-mode", b"plain", False, None, None),
      ("string-refined", "plain", "False", None, None),
      ("integer-refined", "plain", 0, None, None),
    )
    for label, mode, refined, pce, transfer in rows:
      with self.subTest(label=label):
        with self.assertRaisesRegex(
            (KeyError, TypeError, ValueError),
            "composition|mode|plain|npce|transfer|pce|drifted|refined"):
          self._validate(mode, refined, pce, transfer)

  def test_writer_refuses_unknown_or_mistyped_submodes(self):
    """Writer and reader share the same native form/space grammar."""
    cases = (
      ("npce-unknown", lambda: self._validate(
        "npce", False, object(), None,
        pce_form="hybrid", resolved_pce={"form": "hybrid"})),
      ("npce-bytes", lambda: self._validate(
        "npce", False, object(), None,
        pce_form=b"residual", resolved_pce={"form": b"residual"})),
      ("transfer-unknown", lambda: self._validate(
        "transfer", False, None,
        {"form": "delta", "space": "law"},
        resolved_transfer={"form": "delta", "space": "law"})),
      ("transfer-numeric", lambda: self._validate(
        "transfer", False, None,
        {"form": 1, "space": 2},
        resolved_transfer={"form": 1, "space": 2})),
    )
    for label, invoke in cases:
      with self.subTest(label=label):
        with self.assertRaisesRegex(ValueError, "form|space|native"):
          invoke()

  def test_writer_refuses_nonmapping_refine_block(self):
    """Writer applies the same mapping grammar as both YAML readers."""
    with self.assertRaisesRegex(ValueError, "refine.*mapping"):
      self._validate(
        "transfer", True, None,
        {"form": "gain", "space": "physical",
         "drifted_state": object()},
        resolved_transfer={"form": "gain", "space": "physical",
                           "refine": False})


class InferenceCompositionRoutingTest(unittest.TestCase):
  """Inference selects payloads from the enum, never from their presence."""

  def test_three_valid_runtime_modes(self):
    pce = object()
    transfer = object()
    rows = (
      ("plain", None, None),
      ("npce", pce, None),
      ("transfer", None, transfer),
    )
    for mode, pce_base, transfer_base in rows:
      with self.subTest(mode=mode):
        self.assertEqual(
          inference._select_composition(mode, pce_base, transfer_base),
          (pce_base, transfer_base))

  def test_presence_cannot_override_runtime_mode(self):
    pce = object()
    transfer = object()
    rows = (
      ("plain", pce, None),
      ("plain", None, transfer),
      ("npce", None, None),
      ("npce", pce, transfer),
      ("transfer", None, None),
      ("transfer", pce, transfer),
      ("hybrid", None, None),
      (None, None, None),
    )
    for mode, pce_base, transfer_base in rows:
      with self.subTest(mode=mode, pce=pce_base is not None,
                        transfer=transfer_base is not None):
        with self.assertRaisesRegex(ValueError, "composition_mode"):
          inference._select_composition(mode, pce_base, transfer_base)


class _OrderingGeometry:
  """Importable geometry whose constructor must not run in ordering tests."""

  @classmethod
  def from_state(cls, device, state):
    del device, state
    raise AssertionError("geometry construction was reached")


class CompositionValidationOrderingTest(unittest.TestCase):
  """Prove corrupt composition is refused before construction or weight I/O."""

  def _write_geometry(self, parent, name, names):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group = parent.create_group(name)
    group.attrs["cls"] = __name__ + "._OrderingGeometry"
    group.create_dataset("names",
                         data=np.asarray(names, dtype=object),
                         dtype=string_dtype)

  def _write_artifact(self, path_root, *, legacy=False,
                      mode="plain", refined=False):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path_root + ".h5", "w") as artifact:
      artifact.attrs["schema_version"] = fixed_facts.SCHEMA_VERSION
      if not legacy:
        artifact.attrs["composition_mode"] = mode
        artifact.attrs["transfer_refined"] = refined
      artifact.create_dataset("model_recipe", data="{}", dtype=string_dtype)
      artifact.create_dataset(
        "config_yaml",
        data=yaml.safe_dump({"data": {}, "train_args": {},
                             "pce": None, "transfer": None},
                            sort_keys=False),
        dtype=string_dtype)
      resolved = {"data": {}, "train_args": {}} if legacy \
          else _resolved_record(mode, refined)
      artifact.create_dataset(
        "config_resolved_yaml",
        data=yaml.safe_dump(resolved, sort_keys=False),
        dtype=string_dtype)
      self._write_geometry(artifact, "param_geometry", ["alpha", "beta"])
      self._write_geometry(artifact, "dv_geometry", ["observable"])
      fixed_facts.write_h5(
        f=artifact,
        sidecar_text=fixed_facts.synthetic_sidecar(
          names=["alpha", "beta"],
          label="composition-ordering-witness"))

  def test_legacy_and_missing_required_group_refuse_before_side_effects(self):
    """Old schema-v3 and deleted-NPCE artifacts never reach construction."""
    cases = (("legacy", True, "plain", "composition_mode|re-save|migration"),
             ("npce-missing", False, "npce", "pce"))
    with tempfile.TemporaryDirectory() as temp_dir:
      for label, legacy, mode, error in cases:
        with self.subTest(label=label):
          path_root = os.path.join(temp_dir, label)
          self._write_artifact(path_root, legacy=legacy, mode=mode)
          with mock.patch.object(
              _OrderingGeometry,
              "from_state",
              side_effect=AssertionError("geometry construction reached")) \
              as construct_geometry, mock.patch.object(
                results.torch,
                "load",
                side_effect=AssertionError("torch.load reached")) \
              as load_weights:
            with self.assertRaisesRegex((KeyError, ValueError), error):
              results.rebuild_emulator(
                path_root=path_root,
                device=torch.device("cpu"),
                compile_model=False)
            construct_geometry.assert_not_called()
            load_weights.assert_not_called()


if __name__ == "__main__":
  unittest.main()

"""Focused checks for the five Cobaya emulator adapters.

The adapters translate saved-emulator results into objects that Cobaya can
request.  These tests replace Cobaya's base classes with small stand-ins and
replace saved emulators with predictors whose facts are written directly in
the test.  A failure therefore points to the adapter boundary rather than to
model training, HDF5 reading, or a configured CoCoA project.

Concrete examples make the boundaries visible.  A quoted ``"false"`` must
not become true by Python coercion.  Two spellings of one saved-emulator path
must not load the same artifact twice.  Two cosmic-shear sections must follow
their physical block order, even when the YAML lists them in reverse.  A
caller that changes a returned array must not change the result cached for the
next caller.
"""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import scipy.interpolate  # Load SciPy before PyTorch in the NumPy-2 test env.
import torch

from cobaya_theory import _adapter_contract


ROOT = Path(__file__).resolve().parents[2]


class _Theory:
  """Small replacement for the Cobaya lifecycle used by these tests."""

  extra_args = {}
  output_params = []

  def initialize(self):
    return None

  def initialize_with_provider(self, provider):
    self.provider = provider

  def must_provide(self, **requirements):
    del requirements
    return None


class _LoggedError(Exception):
  """Accept Cobaya's ``(logger, message)`` construction convention."""

  def __init__(self, logger, message):
    del logger
    super().__init__(message)


class _Logger:
  """Provide the debug method used by the matter-power adapter."""

  def debug(self, *args, **kwargs):
    del args, kwargs


def _load_adapter(filename, module_name):
  """Load one shipped adapter while replacing the unavailable Cobaya shell."""
  cobaya = types.ModuleType("cobaya")
  theory_module = types.ModuleType("cobaya.theory")
  log_module = types.ModuleType("cobaya.log")
  theory_module.Theory = _Theory
  log_module.LoggedError = _LoggedError
  log_module.get_logger = lambda name: _Logger()
  replacements = {
    "cobaya": cobaya,
    "cobaya.theory": theory_module,
    "cobaya.log": log_module,
  }
  path = ROOT / "cobaya_theory" / filename
  spec = importlib.util.spec_from_file_location(module_name, path)
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, replacements):
    spec.loader.exec_module(module)
  return module


class SharedOptionAndPathTests(unittest.TestCase):
  """Prove that all adapters use one strict interpretation of YAML values."""

  @classmethod
  def setUpClass(cls):
    cls.adapters = (
      (_load_adapter("emul_cosmic_shear.py", "adapter_contract_cosmic"),
       "emul_cosmic_shear"),
      (_load_adapter("emul_scalars.py", "adapter_contract_scalars"),
       "emul_scalars"),
      (_load_adapter("emul_cmb.py", "adapter_contract_cmb"), "emul_cmb"),
      (_load_adapter("emul_baosn.py", "adapter_contract_baosn"),
       "emul_baosn"),
      (_load_adapter("emul_mps.py", "adapter_contract_mps"), "emul_mps"),
    )

  def test_boolean_and_device_options_do_not_use_python_truthiness(self):
    """Quoted booleans and invented device names stop before model loading."""
    for value in (0, 1, "false", "true", None, np.bool_(False)):
      with self.subTest(option="compile", value=repr(value)):
        with self.assertRaisesRegex(ValueError, "actual Boolean"):
          _adapter_contract.exact_bool(
            {"compile": value}, "compile", adapter="example")

    for value in ("CPU", "gpu", "cuda:0", 0, True):
      with self.subTest(option="device", value=repr(value)):
        with self.assertRaisesRegex(ValueError, "must be one of"):
          _adapter_contract.pick_device({"device": value}, adapter="example")

  def test_option_container_and_unknown_names_refuse(self):
    """A YAML mapping may contain documented names only."""
    with self.assertRaisesRegex(ValueError, "must be a mapping"):
      _adapter_contract.validate_extra_args(
        ["device", "cpu"], adapter="example",
        allowed=("device",), retired="old names are retired")
    with self.assertRaisesRegex(ValueError, "unrecognized extra_args"):
      _adapter_contract.validate_extra_args(
        {"device": "cpu", "devcie": "cpu"}, adapter="example",
        allowed=("device",), retired="old names are retired")

  def test_relative_paths_need_rootdir_and_aliases_cannot_repeat(self):
    """A relative root needs CoCoA; member aliases are not a second model."""
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertRaisesRegex(ValueError, "requires ROOTDIR"):
        _adapter_contract.resolve_emulator_roots(
          {"emulators": ["models/shear"]}, adapter="example")

    with tempfile.TemporaryDirectory() as temporary:
      rootdir = Path(temporary)
      model = rootdir / "models" / "shear"
      model.parent.mkdir()
      alias = rootdir / "models" / "same-shear"
      for extension in (".h5", ".emul"):
        member = Path(str(model) + extension)
        member.touch()
        Path(str(alias) + extension).symlink_to(member)
      with mock.patch.dict(os.environ, {"ROOTDIR": str(rootdir)}):
        resolved = _adapter_contract.resolve_emulator_roots(
          {"emulators": ["models/shear"]}, adapter="example")
        self.assertEqual(resolved, [str(model.resolve())])
        with self.assertRaisesRegex(ValueError, "same saved .h5/.emul"):
          _adapter_contract.resolve_emulator_roots(
            {"emulators": ["models/shear", "models/same-shear"]},
            adapter="example")

  def test_root_container_entries_and_required_pair_size_are_exact(self):
    """Strings are entries, not a list; background and P(k) need two roots."""
    bad_sequences = ("one-root", [], [True], [""], ["   "])
    for roots in bad_sequences:
      with self.subTest(roots=repr(roots)):
        with self.assertRaises(ValueError):
          _adapter_contract.resolve_emulator_roots(
            {"emulators": roots}, adapter="example")
    with self.assertRaisesRegex(ValueError, "exactly 2"):
      _adapter_contract.resolve_emulator_roots(
        {"emulators": ["/tmp/only-one"]},
        adapter="example", exact_count=2)

  def test_each_adapter_rejects_a_quoted_compile_value(self):
    """The common rule is reached by all five public adapter classes."""
    for index, (module, class_name) in enumerate(self.adapters):
      adapter = getattr(module, class_name)()
      count = 2 if class_name in ("emul_baosn", "emul_mps") else 1
      adapter.extra_args = {
        "device": "cpu",
        "emulators": [f"/tmp/adapter-{index}-{item}"
                      for item in range(count)],
        "compile": "false",
      }
      with self.subTest(adapter=class_name):
        with self.assertRaisesRegex(ValueError, "actual Boolean"):
          adapter.initialize()

  def test_explicit_null_is_not_an_omitted_parameter_name_list(self):
    """YAML null cannot silently disable a supplied list-shaped option."""
    cosmic = self.adapters[0][0].emul_cosmic_shear()
    cosmic.extra_args = {
      "device": "cpu",
      "emulators": ["/tmp/cosmic-null-fast-params"],
      "fast_params": None,
      "compile": False,
    }
    with self.assertRaisesRegex(ValueError, "exactly one inner name list"):
      cosmic.initialize()

    scalars = self.adapters[1][0].emul_scalars()
    scalars.extra_args = {
      "device": "cpu",
      "emulators": ["/tmp/scalar-null-provides"],
      "provides": None,
      "compile": False,
    }
    with self.assertRaisesRegex(ValueError, "sequence of parameter-name"):
      scalars.initialize()

  def test_each_adapter_rejects_two_names_for_one_canonical_root(self):
    """No adapter can load one artifact twice through duplicate path text."""
    for module, class_name in self.adapters:
      adapter = getattr(module, class_name)()
      adapter.extra_args = {
        "device": "cpu",
        "emulators": ["/tmp/same-adapter-root", "/tmp/same-adapter-root"],
        "compile": False,
      }
      with self.subTest(adapter=class_name):
        with self.assertRaisesRegex(ValueError, "same canonical path"):
          adapter.initialize()

  def test_each_adapter_check_calls_the_shared_validator(self):
    """A future adapter edit cannot restore five drifting key-check copies."""
    filenames = (
      "emul_cosmic_shear.py",
      "emul_scalars.py",
      "emul_cmb.py",
      "emul_baosn.py",
      "emul_mps.py",
    )
    for filename in filenames:
      tree = ast.parse((ROOT / "cobaya_theory" / filename).read_text())
      checks = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "_check_extra_args"]
      calls = [node for node in ast.walk(checks[0])
               if isinstance(node, ast.Call)] if len(checks) == 1 else []
      with self.subTest(adapter=filename):
        self.assertEqual(len(checks), 1)
        self.assertTrue(any(
          isinstance(call.func, ast.Name)
          and call.func.id == "validate_extra_args"
          for call in calls))


class _SectionGeometry:
  """Three physical blocks used to test cosmic-shear assembly order."""

  PROBE_BLOCKS = {
    "early": (0, 1),
    "late": (2,),
    "overlap": (1,),
  }


class _SectionPredictor:
  """Return one fixed section and expose its saved layout facts."""

  def __init__(self, probe, values, *, section_sizes=(2, 3, 4), total=9):
    self.probe = probe
    self.section_sizes = section_sizes
    self.total_size = total
    self.geom = _SectionGeometry()
    self.values = np.asarray(values)

  def predict(self, params):
    del params
    return self.values


class CosmicSectionCompositionTests(unittest.TestCase):
  """Check physical ordering and every unsafe multi-section arrangement."""

  @classmethod
  def setUpClass(cls):
    cls.module = _load_adapter(
      "emul_cosmic_shear.py", "adapter_contract_cosmic_composition")

  def test_sections_follow_physical_blocks_not_yaml_order(self):
    """A late probe listed first is still placed after blocks zero and one."""
    late = _SectionPredictor("late", [20.0, 21.0, 22.0, 23.0])
    early = _SectionPredictor("early", [0.0, 1.0, 2.0, 3.0, 4.0])
    adapter = self.module.emul_cosmic_shear()
    adapter.predictors = [late, early]
    adapter._composition = adapter._build_composition("section")
    self.assertIs(adapter._composition[0][0], early)
    self.assertIs(adapter._composition[1][0], late)

    state = {}
    self.assertTrue(adapter.calculate(state))
    np.testing.assert_array_equal(
      state["cosmic_shear"],
      np.array([0.0, 1.0, 2.0, 3.0, 4.0, 20.0, 21.0, 22.0, 23.0]))

  def test_overlap_and_incompatible_global_layouts_refuse(self):
    """Two sections cannot claim one block or describe different datasets."""
    adapter = self.module.emul_cosmic_shear()
    adapter.predictors = [
      _SectionPredictor("early", np.arange(5.0)),
      _SectionPredictor("overlap", np.arange(3.0)),
    ]
    with self.assertRaisesRegex(ValueError, "both serve global block"):
      adapter._build_composition("section")

    adapter.predictors = [
      _SectionPredictor("early", np.arange(5.0)),
      _SectionPredictor(
        "late", np.arange(4.0), section_sizes=(2, 3, 5), total=10),
    ]
    with self.assertRaisesRegex(ValueError, "same global layout"):
      adapter._build_composition("section")

  def test_multiple_full_vectors_and_wrong_section_width_refuse(self):
    """One global vector is enough, and every section must have saved width."""
    adapter = self.module.emul_cosmic_shear()
    adapter.predictors = [
      _SectionPredictor("early", np.arange(5.0)),
      _SectionPredictor("late", np.arange(4.0)),
    ]
    with self.assertRaisesRegex(ValueError, "cannot combine multiple"):
      adapter._build_composition("3x2pt")

    wrong = _SectionPredictor("late", np.arange(3.0))
    adapter.predictors = [wrong]
    adapter._composition = adapter._build_composition("section")
    with self.assertRaisesRegex(ValueError, "returned shape"):
      adapter.calculate({})


class ScalarAndCmbContractTests(unittest.TestCase):
  """Check scalar publication and exact CMB request interpretation."""

  @classmethod
  def setUpClass(cls):
    cls.scalars = _load_adapter(
      "emul_scalars.py", "adapter_contract_scalar_publication")
    cls.cmb = _load_adapter("emul_cmb.py", "adapter_contract_cmb_requests")

  def _cmb_adapter(self, spectra, ell_arrays=None):
    """Build the public CMB calculation around two-point spectrum fixtures."""
    predictors = []
    stored_ells = {}
    for spectrum, values in spectra.items():
      output = np.asarray(values)
      predictors.append(types.SimpleNamespace(
        spectrum=spectrum,
        predict=lambda params, value=output: np.array(value, copy=True)))
      stored_ells[spectrum] = np.array(
        [2, 3] if ell_arrays is None else ell_arrays[spectrum])
    adapter = self.cmb.emul_cmb()
    adapter.predictors = predictors
    adapter._ell_arrays = stored_ells
    adapter._lmax_global = max(int(ells[-1]) for ells in stored_ells.values())
    return adapter

  def _assert_cmb_refusal_keeps_state(self, spectra):
    """Require one invalid spectrum set to leave the caller's state alone."""
    adapter = self._cmb_adapter(spectra)
    state = {"kept": "unchanged"}
    with self.assertRaises(ValueError):
      adapter.calculate(state, want_derived=True, p0=0.1)
    self.assertEqual(state, {"kept": "unchanged"})

  def test_cmb_signed_te_on_the_psd_boundary_is_published(self):
    """Both TE signs may reach the covariance boundary after float32 rounding."""
    tt = np.float32(0.1)
    ee = np.float32(0.2)
    rounded_boundary = np.float32(np.sqrt(tt * ee))
    direct_boundary = np.float32(np.sqrt(tt) * np.sqrt(ee))
    self.assertGreater(rounded_boundary, direct_boundary)
    self.assertEqual(
      rounded_boundary, np.nextafter(direct_boundary, np.float32(np.inf)))
    spectra = {
      "tt": np.array([tt, 4.0], dtype=np.float32),
      "ee": np.array([ee, 9.0], dtype=np.float32),
      "te": np.array([-rounded_boundary, 6.0], dtype=np.float32),
      "pp": np.array([0.0, 1.0], dtype=np.float32),
    }
    state = {}
    self.assertTrue(self._cmb_adapter(spectra).calculate(state, p0=0.1))
    np.testing.assert_array_equal(state["Cl"]["te"][2:], spectra["te"])

  def test_cmb_nonfinite_spectrum_leaves_state_unchanged(self):
    """NaN and infinity in any served spectrum refuse before publication."""
    good = {"tt": [1.0, 1.0], "ee": [1.0, 1.0],
            "te": [0.0, 0.0], "pp": [0.0, 1.0]}
    for spectrum in ("tt", "ee", "te", "pp"):
      for value in (np.nan, np.inf, -np.inf):
        with self.subTest(spectrum=spectrum, value=value):
          spectra = {name: list(values) for name, values in good.items()}
          spectra[spectrum][1] = value
          self._assert_cmb_refusal_keeps_state(spectra)

  def test_cmb_negative_tt_ee_or_pp_leaves_state_unchanged(self):
    """Negative TT, EE, or PP cannot become a public CMB prediction."""
    for spectrum in ("tt", "ee", "pp"):
      with self.subTest(spectrum=spectrum):
        spectra = {
          "tt": [1.0, 1.0], "ee": [1.0, 1.0],
          "te": [0.0, 0.0], "pp": [0.0, 1.0],
        }
        spectra[spectrum][1] = -1.0
        self._assert_cmb_refusal_keeps_state(spectra)

  def test_cmb_te_beyond_the_psd_bound_leaves_state_unchanged(self):
    """A cross-spectrum larger than sqrt(TT*EE) refuses atomically."""
    cases = (
      {"tt": [1.0, 1.0], "ee": [1.0, 1.0], "te": [0.0, 2.0]},
      {"tt": [1e-300, 1e-300], "ee": [1e-300, 1e-300],
       "te": [0.0, 1e-200]},
      {"tt": [1e308, 1e308], "ee": [1e308, 1e308],
       "te": [0.0, 1.7e308]},
    )
    for spectra in cases:
      with self.subTest(spectra=spectra):
        self._assert_cmb_refusal_keeps_state(spectra)

  def test_cmb_psd_check_uses_only_shared_stored_multipoles(self):
    """A missing auto-spectrum does not turn padding into physical data."""
    spectra = {
      "tt": [1.0, 1.0],
      "te": [2.0, 1.0, 2.0],
      "ee": [1.0, 1.0],
    }
    ell_arrays = {
      "tt": [2, 3],
      "te": [2, 3, 4],
      "ee": [3, 4],
    }
    state = {}
    adapter = self._cmb_adapter(spectra, ell_arrays=ell_arrays)
    self.assertTrue(adapter.calculate(state, p0=0.1))
    self.assertEqual(
      state["Cl"]["te"].tolist(), [0.0, 0.0, 2.0, 1.0, 2.0])

  def test_scalar_calculate_uses_its_validated_publication_names(self):
    """Only the adapter's validated names enter Cobaya's derived mapping."""
    first = types.SimpleNamespace(
      predict=lambda params: {"H0": 70.0, "rdrag": 147.0})
    second = types.SimpleNamespace(
      predict=lambda params: {"omegam": 0.3})
    adapter = self.scalars.emul_scalars()
    adapter.predictors = [first, second]
    adapter._provides = ["rdrag"]

    state = {}
    self.assertTrue(adapter.calculate(state, want_derived=True, x=1.0))
    self.assertEqual(state, {"derived": {"rdrag": 147.0}})
    self.assertNotIn("rdrag", state)

    state = {"derived": {"existing": 2.0}}
    self.assertTrue(adapter.calculate(state, want_derived=True, x=1.0))
    self.assertEqual(
      state, {"derived": {"existing": 2.0, "rdrag": 147.0}})

  def test_scalar_declared_subset_checks_without_filtering_metadata(self):
    """A check-only list cannot hide other outputs saved in the artifacts."""
    class _Predictor:
      def __init__(self, root, device, compile_model=False):
        del device, compile_model
        self._scalar = True
        self._cmb = self._grid = self._grid2d = False
        self.names = ["x"]
        self.output_names = (["H0", "rdrag"] if root.endswith("one")
                             else ["omegam"])

    adapter = self.scalars.emul_scalars()
    adapter.extra_args = {
      "device": "cpu",
      "emulators": ["/tmp/scalar-adapter-one", "/tmp/scalar-adapter-two"],
      "provides": ["rdrag"],
      "compile": False,
    }
    with mock.patch.object(self.scalars, "EmulatorPredictor", _Predictor), \
         mock.patch.object(self.scalars, "check_artifacts_pair_up",
                           lambda **kwargs: None):
      adapter.initialize()
    self.assertEqual(
      adapter.get_can_provide_params(), ["H0", "rdrag", "omegam"])

    adapter.extra_args["provides"] = []
    with mock.patch.object(self.scalars, "EmulatorPredictor", _Predictor), \
         mock.patch.object(self.scalars, "check_artifacts_pair_up",
                           lambda **kwargs: None):
      adapter.initialize()
    self.assertEqual(
      adapter.get_can_provide_params(), ["H0", "rdrag", "omegam"])

  def test_scalar_artifact_output_names_are_nonempty_unique_strings(self):
    """A rebuilt artifact cannot advertise an unusable derived name."""
    class _Predictor:
      output_names = []

      def __init__(self, root, device, compile_model=False):
        del root, device, compile_model
        self._scalar = True
        self._cmb = self._grid = self._grid2d = False
        self.names = ["x"]

    for bad_names in ([], [""], [1], ["rdrag", "rdrag"]):
      with self.subTest(output_names=bad_names):
        _Predictor.output_names = bad_names
        adapter = self.scalars.emul_scalars()
        adapter.extra_args = {
          "device": "cpu",
          "emulators": ["/tmp/scalar-bad-output-name"],
          "compile": False,
        }
        with mock.patch.object(self.scalars, "EmulatorPredictor", _Predictor):
          with self.assertRaises(ValueError):
            adapter.initialize()

  def test_scalar_calculate_leaves_state_alone_when_derived_not_requested(self):
    """Cobaya's false flag does not leak a top-level or derived result."""
    adapter = self.scalars.emul_scalars()
    adapter.predictors = [types.SimpleNamespace(
      predict=lambda params: {"rdrag": 147.0})]
    adapter._provides = ["rdrag"]
    state = {"kept": "unchanged"}
    self.assertTrue(adapter.calculate(state, want_derived=False))
    self.assertEqual(state, {"kept": "unchanged"})

  def test_cmb_requirement_is_a_mapping_with_exact_integer_lmax(self):
    """Lists, quoted numbers, booleans, and out-of-range values all stop."""
    adapter = self.cmb.emul_cmb()
    adapter._lmax_of = {"tt": 10}
    adapter.must_provide(Cl={"TT": np.int64(10)})

    with self.assertRaisesRegex(ValueError, "must be a mapping"):
      adapter.must_provide(Cl=[("tt", 10)])
    with self.assertRaisesRegex(ValueError, "spectrum name must be a string"):
      adapter.must_provide(Cl={1: 10})
    for value in (True, np.bool_(False), 10.0, "10", np.array(10)):
      with self.subTest(lmax=repr(value)):
        with self.assertRaisesRegex(ValueError, "non-Boolean integer"):
          adapter.must_provide(Cl={"tt": value})
    for value in (-1, 11):
      with self.subTest(lmax=value):
        with self.assertRaisesRegex(ValueError, "inclusive range"):
          adapter.must_provide(Cl={"tt": value})


class LiveCobayaScalarRoutingTests(unittest.TestCase):
  """Use Cobaya's real model builder to check one derived scalar route."""

  def test_advertised_scalar_reaches_cobaya_as_a_derived_result(self):
    """A scalar adapter creates its mapping and Cobaya receives that value."""
    try:
      from cobaya.likelihood import Likelihood
      from cobaya.model import get_model
      from cobaya.theory import Theory
      from cobaya_theory.emul_scalars import (
        emul_scalars as LiveEmulScalars,
      )
    except ImportError as error:
      self.skipTest("real Cobaya and its adapter imports are unavailable: "
                    + str(error))

    class _Predictor:
      """Return one transparent value without reading a saved artifact."""

      def predict(self, params):
        return {"rdrag": 146.0 + float(params["x"])}

    class RoutingScalars(LiveEmulScalars):
      """Keep the shipped routing code but replace artifact initialization."""

      def initialize(self):
        self.predictors = [_Predictor()]
        self._req = {"x": None}
        self._provides = ["rdrag"]

      def initialize_with_provider(self, provider):
        Theory.initialize_with_provider(self, provider)

    class ScalarLike(Likelihood):
      """Ask Cobaya for the one scalar advertised by the test theory."""

      seen = []

      def get_requirements(self):
        return {"rdrag": None}

      def calculate(self, state, want_derived=True, **params):
        del want_derived, params
        type(self).seen.append(self.provider.get_param("rdrag"))
        state["logp"] = 0.0

    model = get_model({
      "params": {"x": 1.0, "rdrag": {"derived": True}},
      "theory": {"scalars": RoutingScalars},
      "likelihood": {"scalar_like": ScalarLike},
    })
    theory = model.theory["scalars"]

    # This direct caller deliberately supplies an empty dictionary.  It
    # demonstrates why the adapter, rather than every caller, owns creation
    # of the conventional derived-result mapping.
    state = {}
    self.assertTrue(theory.calculate(
      state, want_derived=True, x=2.0))
    self.assertEqual(state, {"derived": {"rdrag": 148.0}})

    ScalarLike.seen = []
    result = model.logposterior({})
    self.assertEqual(ScalarLike.seen, [147.0])
    self.assertEqual(result.derived, [147.0])


class BackgroundDependencyTests(unittest.TestCase):
  """Check the background adapter's exact union of saved input names."""

  @classmethod
  def setUpClass(cls):
    cls.module = _load_adapter(
      "emul_baosn.py", "adapter_contract_background_dependencies")

  def test_background_requirements_are_the_two_artifact_name_union(self):
    """Cobaya receives every saved input once, with no invented names."""
    class _Predictor:
      def __init__(self, root, device, compile_model=False):
        del device, compile_model
        self._grid = True
        self._scalar = self._cmb = self._grid2d = False
        if "hubble" in root:
          self.quantity = "Hubble"
          self.units = "km/s/Mpc"
          self.names = ["H0", "omegab"]
          self.z = torch.tensor([0.0, 0.5, 1.0])
        else:
          self.quantity = "D_M"
          self.units = "Mpc"
          self.names = ["omegab", "omegam"]
          self.z = torch.tensor([1000.0, 1100.0])
        self.fixed_facts = {"flat_only": True}

    adapter = self.module.emul_baosn()
    adapter.extra_args = {
      "device": "cpu",
      "emulators": ["/tmp/hubble-adapter-contract",
                    "/tmp/distance-adapter-contract"],
      "compile": False,
    }
    with mock.patch.object(self.module, "EmulatorPredictor", _Predictor), \
         mock.patch.object(self.module, "check_artifacts_pair_up",
                           lambda **kwargs: None):
      adapter.initialize()
    self.assertEqual(
      adapter.get_requirements(),
      {"H0": None, "omegab": None, "omegam": None})


class ReturnedOwnershipTests(unittest.TestCase):
  """Mutating one public result must not alter the adapter's cached state."""

  @classmethod
  def setUpClass(cls):
    cls.cosmic = _load_adapter(
      "emul_cosmic_shear.py", "adapter_contract_cosmic_ownership")
    cls.scalars = _load_adapter(
      "emul_scalars.py", "adapter_contract_scalar_ownership")
    cls.cmb = _load_adapter("emul_cmb.py", "adapter_contract_cmb_ownership")
    cls.baosn = _load_adapter(
      "emul_baosn.py", "adapter_contract_baosn_ownership")
    cls.mps = _load_adapter("emul_mps.py", "adapter_contract_mps_ownership")

  def test_requirement_and_product_metadata_are_fresh_containers(self):
    """Changing discovered names cannot rewrite the adapter's declarations."""
    classes = (
      self.cosmic.emul_cosmic_shear,
      self.scalars.emul_scalars,
      self.cmb.emul_cmb,
      self.baosn.emul_baosn,
      self.mps.emul_mps,
    )
    for adapter_class in classes:
      adapter = adapter_class()
      adapter._req = {"x": None}
      returned = adapter.get_requirements()
      returned["invented"] = None
      with self.subTest(adapter=adapter_class.__name__):
        self.assertEqual(adapter.get_requirements(), {"x": None})

    scalar = self.scalars.emul_scalars()
    scalar._provides = ["rdrag"]
    names = scalar.get_can_provide_params()
    names.append("invented")
    self.assertEqual(scalar.get_can_provide_params(), ["rdrag"])

    background = self.baosn.emul_baosn()
    products = background.get_can_provide()
    products.append("invented")
    self.assertNotIn("invented", background.get_can_provide())

  def test_cosmic_and_cmb_getters_copy_cached_arrays(self):
    """Likelihood code cannot corrupt a data vector or a CMB spectrum."""
    cosmic = self.cosmic.emul_cosmic_shear()
    cosmic.current_state = {"cosmic_shear": np.array([1.0, 2.0])}
    first = cosmic.get_cosmic_shear()
    first[0] = -100.0
    np.testing.assert_array_equal(cosmic.get_cosmic_shear(), [1.0, 2.0])

    cmb = self.cmb.emul_cmb()
    cmb._cl_units = "muK2"
    cmb.current_state = {
      "Cl": {"ell": np.arange(4), "tt": np.array([0.0, 0.0, 3.0, 4.0])}}
    first_cl = cmb.get_Cl()
    first_cl["tt"][2] = -100.0
    first_cl["invented"] = np.ones(4)
    second_cl = cmb.get_Cl()
    np.testing.assert_array_equal(second_cl["tt"], [0.0, 0.0, 3.0, 4.0])
    self.assertNotIn("invented", second_cl)

  def test_background_hubble_getter_copies_even_a_reused_interpolator_buffer(self):
    """The getter owns its result even when an interpolator reuses memory."""
    class _ReusedBuffer:
      def __init__(self):
        self.values = np.array([70.0, 71.0])

      def __call__(self, z):
        del z
        return self.values

    adapter = self.baosn.emul_baosn()
    adapter._sn_max = 1.0
    adapter.current_state = {"baosn": {"itp": {"H": _ReusedBuffer()}}}
    first = adapter.get_Hubble([0.1, 0.2])
    first[0] = -100.0
    np.testing.assert_array_equal(
      adapter.get_Hubble([0.1, 0.2]), [70.0, 71.0])

  def test_matter_power_grid_getter_copies_all_three_arrays(self):
    """Changing k, z, or P(k,z) cannot alter a later provider call."""
    adapter = self.mps.emul_mps()
    adapter.log = _Logger()
    key = ("Pk_grid", False, "delta_tot", "delta_tot")
    adapter.current_state = {
      key: (
        np.array([0.1, 0.2]),
        np.array([0.0, 1.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
      )
    }
    k, z, power = adapter.get_Pk_grid(nonlinear=False)
    k[0], z[0], power[0, 0] = -1.0, -1.0, -1.0
    second_k, second_z, second_power = adapter.get_Pk_grid(nonlinear=False)
    np.testing.assert_array_equal(second_k, [0.1, 0.2])
    np.testing.assert_array_equal(second_z, [0.0, 1.0])
    np.testing.assert_array_equal(second_power, [[1.0, 2.0], [3.0, 4.0]])

  def test_matter_power_interpolator_is_fresh_for_each_reader(self):
    """One consumer cannot alter the interpolation metadata seen by another."""
    adapter = self.mps.emul_mps()
    adapter.log = _Logger()
    key = ("Pk_grid", False, "delta_tot", "delta_tot")
    k = np.geomspace(0.01, 10.0, 4)
    z = np.array([0.0, 0.5, 1.0, 2.0])
    power = (1.0 + z[:, None]) * (1.0 + k[None, :])
    adapter.current_state = {key: (k, z, power)}

    first = adapter.get_Pk_interpolator(nonlinear=False)
    original_z = np.array(first.z, copy=True)
    original_k = np.array(first.k, copy=True)
    first.z[0] = -100.0
    first.k[0] = -100.0

    second = adapter.get_Pk_interpolator(nonlinear=False)
    self.assertIsNot(first, second)
    np.testing.assert_array_equal(second.z, original_z)
    np.testing.assert_array_equal(second.k, original_k)


class MatterPowerDependencyTests(unittest.TestCase):
  """Check that a stored amplitude name is not duplicated for Syren."""

  @classmethod
  def setUpClass(cls):
    cls.module = _load_adapter(
      "emul_mps.py", "adapter_contract_mps_dependencies")

  def test_as_1e9_input_does_not_add_a_redundant_as_requirement(self):
    """Cobaya receives one amplitude spelling, exactly as the artifacts use."""
    class _Predictor:
      def __init__(self, root, device, compile_model=False):
        del device, compile_model
        self._grid2d = True
        self._scalar = self._cmb = self._grid = False
        self.quantity = "pklin" if "pklin" in root else "boost"
        self.names = ["As_1e9"] if self.quantity == "pklin" else []
        self.z = torch.tensor([0.0, 0.5, 1.0, 2.0])
        self.k = torch.tensor([0.01, 0.1, 1.0, 10.0])
        self.units = ("Mpc3" if self.quantity == "pklin"
                      else "dimensionless")
        self.law = ("syren_linear" if self.quantity == "pklin"
                    else "syren_halofit")
        self.fixed_facts = {
          "dark_energy_law": "cosmological-constant",
          "dark_energy_inputs": [],
          "cosmology_fixed": {"w": -1.0, "wa": 0.0},
        }
        if "bad-law" in root:
          self.law = ("syren_halofit" if self.quantity == "pklin"
                      else "syren_linear")
        if "bad-units" in root:
          self.units = ("dimensionless" if self.quantity == "pklin"
                        else "Mpc3")

    adapter = self.module.emul_mps()
    adapter.extra_args = {
      "device": "cpu",
      "emulators": ["/tmp/pklin-adapter-contract",
                    "/tmp/boost-adapter-contract"],
      "compile": False,
    }
    with mock.patch.object(self.module, "EmulatorPredictor", _Predictor), \
         mock.patch.object(self.module, "check_artifacts_pair_up",
                           lambda **kwargs: None):
      adapter.initialize()
    requirements = adapter.get_requirements()
    self.assertIn("As_1e9", requirements)
    self.assertNotIn("As", requirements)
    for name in ("ns", "H0", "omegab", "omegam"):
      self.assertIn(name, requirements)

  def test_matter_power_artifact_tuple_must_match_its_quantity(self):
    """A rebuilt artifact cannot give one surface another surface's law."""
    class _Predictor:
      def __init__(self, root, device, compile_model=False):
        del device, compile_model
        self._grid2d = True
        self._scalar = self._cmb = self._grid = False
        self.quantity = "pklin" if "pklin" in root else "boost"
        self.names = []
        self.z = torch.tensor([0.0, 0.5, 1.0, 2.0])
        self.k = torch.tensor([0.01, 0.1, 1.0, 10.0])
        self.units = ("Mpc3" if self.quantity == "pklin"
                      else "dimensionless")
        self.law = ("syren_linear" if self.quantity == "pklin"
                    else "syren_halofit")
        self.fixed_facts = {
          "dark_energy_law": "cosmological-constant",
          "dark_energy_inputs": [],
          "cosmology_fixed": {"w": -1.0, "wa": 0.0},
        }
        if "bad-law" in root:
          self.law = ("syren_halofit" if self.quantity == "pklin"
                      else "syren_linear")
        if "bad-units" in root:
          self.units = ("dimensionless" if self.quantity == "pklin"
                        else "Mpc3")

    cases = (
      ("/tmp/pklin-bad-law", "/tmp/boost-valid"),
      ("/tmp/pklin-bad-units", "/tmp/boost-valid"),
      ("/tmp/pklin-valid", "/tmp/boost-bad-law"),
      ("/tmp/pklin-valid", "/tmp/boost-bad-units"),
    )
    for pklin_root, boost_root in cases:
      adapter = self.module.emul_mps()
      adapter.extra_args = {
        "device": "cpu",
        "emulators": [pklin_root, boost_root],
        "compile": False,
      }
      with self.subTest(roots=(pklin_root, boost_root)), \
           mock.patch.object(self.module, "EmulatorPredictor", _Predictor), \
           mock.patch.object(self.module, "check_artifacts_pair_up",
                             lambda **kwargs: None):
        with self.assertRaisesRegex(ValueError, r"unsupported \(units, law\)"):
          adapter.initialize()


if __name__ == "__main__":
  unittest.main()

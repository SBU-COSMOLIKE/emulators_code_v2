"""Focused checks for the five Cobaya emulator adapters.

The adapters translate saved-emulator results into objects that Cobaya can
request.  These tests replace Cobaya's base classes with small stand-ins and
replace saved emulators with predictors whose facts are written directly in
the test.  A failure therefore points to the adapter boundary rather than to
model training, HDF5 reading, or a configured CoCoA project.

Concrete examples make the boundaries visible.  An unknown extra_args key
must stop the run instead of being ignored.  Two cosmic-shear sections must
follow their physical block order, even when the YAML lists them in reverse.
A caller that changes a returned array must not change the result cached for
the next caller.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import scipy.interpolate  # Load SciPy before PyTorch in the NumPy-2 test env.
import torch


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
  """Prove that every adapter refuses a malformed extra_args block loudly."""

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

  def test_each_adapter_rejects_an_unknown_extra_args_key(self):
    """A misspelled option stops the run instead of being ignored."""
    for module, class_name in self.adapters:
      adapter = getattr(module, class_name)()
      adapter.extra_args = {"devcie": "cpu"}
      with self.subTest(adapter=class_name):
        with self.assertRaisesRegex(ValueError, "unrecognized extra_args"):
          adapter.initialize()

  def test_each_adapter_requires_a_nonempty_emulators_list(self):
    """A missing or empty saved-root list stops before any file is opened."""
    for module, class_name in self.adapters:
      for roots in (None, []):
        adapter = getattr(module, class_name)()
        adapter.extra_args = ({"device": "cpu"} if roots is None
                              else {"device": "cpu", "emulators": roots})
        with self.subTest(adapter=class_name, roots=repr(roots)):
          with self.assertRaisesRegex(ValueError, "emulators"):
            adapter.initialize()

  def test_the_two_pair_adapters_require_exactly_two_roots(self):
    """The background and matter-power theories serve one pair each."""
    for index in (3, 4):
      module, class_name = self.adapters[index]
      adapter = getattr(module, class_name)()
      adapter.extra_args = {"device": "cpu",
                            "emulators": ["/tmp/only-one-root"]}
      with self.subTest(adapter=class_name):
        with self.assertRaisesRegex(ValueError, "exactly"):
          adapter.initialize()


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
    """Two sections cannot claim one block or two different layouts."""
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

  def test_scalar_duplicate_output_across_artifacts_is_refused(self):
    """Each derived parameter must come from exactly one emulator."""
    class _Predictor:
      def __init__(self, root, device, compile_model=False):
        del root, device, compile_model
        self._scalar = True
        self._cmb = self._grid = self._grid2d = False
        self.names = ["x"]
        self.output_names = ["rdrag"]

    adapter = self.scalars.emul_scalars()
    adapter.extra_args = {
      "device": "cpu",
      "emulators": ["/tmp/scalar-dup-one", "/tmp/scalar-dup-two"],
      "compile": False,
    }
    with mock.patch.object(self.scalars, "EmulatorPredictor", _Predictor):
      with self.assertRaisesRegex(ValueError, "two emulators provide"):
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

  def test_cmb_requirement_is_a_mapping_within_the_stored_range(self):
    """A wrong container, an unknown spectrum, or a wild lmax all stop."""
    adapter = self.cmb.emul_cmb()
    adapter._lmax_of = {"tt": 10}
    adapter.must_provide(Cl={"TT": np.int64(10)})

    with self.assertRaisesRegex(ValueError, "must be a mapping"):
      adapter.must_provide(Cl=[("tt", 10)])
    with self.assertRaisesRegex(ValueError, "no loaded artifact provides"):
      adapter.must_provide(Cl={"bb": 10})
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


class MatterPowerServingDomainTests(unittest.TestCase):
  """Check the public k and z limits before a spectrum is interpolated."""

  @classmethod
  def setUpClass(cls):
    cls.module = _load_adapter(
      "emul_mps.py", "adapter_contract_mps_serving_domain")
    cls.z = np.array([0.0, 0.5, 1.0, 2.0])
    cls.k = np.array([1.0, 2.0, 4.0, 8.0])
    cls.power = (1.0 + cls.z[:, None]) * cls.k[None, :] ** 2

  def _adapter(self, allow=None):
    """Return an adapter whose exact P(k,z) is known at both k tails."""
    adapter = self.module.emul_mps()
    adapter.log = _Logger()
    adapter._z = np.array(self.z, copy=True)
    adapter._k = np.array(self.k, copy=True)
    if allow is not None:
      adapter._allow_k_extrapolation = allow
    adapter.current_state = {
      ("Pk_grid", False, "delta_tot", "delta_tot"):
        (np.array(self.k, copy=True), np.array(self.z, copy=True),
         np.array(self.power, copy=True)),
    }
    return adapter

  def test_malformed_axes_and_surface_refuse_before_scipy(self):
    """Bad saved coordinates stop before FITPACK or a logarithm is called."""
    good_log_power = np.log(self.power)
    cases = (
      (self.z[:3], self.k, good_log_power[:3], "axes|points"),
      (self.z, self.k[:3], good_log_power[:, :3], "axes|points"),
      (np.array([[0.0, 0.5, 1.0, 2.0]]), self.k,
       good_log_power, "axes|points"),
      (np.array([0.0, 1.0, 0.5, 2.0]), self.k,
       good_log_power, "axes"),
      (np.array([0.0, 0.5, 0.5, 2.0]), self.k,
       good_log_power, "axes"),
      (np.array([0.0, 0.5, np.nan, 2.0]), self.k,
       good_log_power, "axes|surface"),
      (self.z, np.array([[1.0, 2.0, 4.0, 8.0]]),
       good_log_power, "axes|points"),
      (self.z, np.array([1.0, 4.0, 2.0, 8.0]),
       good_log_power, "axes"),
      (self.z, np.array([0.0, 2.0, 4.0, 8.0]),
       good_log_power, "axes|surface"),
      (self.z, np.array([1.0, 2.0, 2.0, 8.0]),
       good_log_power, "axes"),
      (self.z, np.array([1.0, 2.0, 4.0, np.inf]),
       good_log_power, "axes|surface"),
      (self.z, self.k, good_log_power[:, :3], "surface"),
      (self.z, self.k,
       np.where(np.indices(good_log_power.shape)[0] == 0,
                np.nan, good_log_power), "surface"),
    )
    for z, k, surface, message in cases:
      with self.subTest(message=message, z=repr(z), k=repr(k)):
        with self.assertRaisesRegex(ValueError, message):
          self.module.PowerSpectrumInterpolator(z, k, surface, logP=True)

  def test_saved_axis_rule_refuses_unsorted_values(self):
    """The rule used for both saved grids rejects invalid coordinates."""
    self.assertFalse(self.module._valid_mps_axis(
      np.array([0.0, 1.0, 0.5, 2.0])))
    self.assertFalse(self.module._valid_mps_axis(
      np.array([1.0, 0.0, 4.0, 8.0]), positive=True))

  def test_malformed_queries_refuse_before_log_and_z_never_extrapolates(self):
    """NaN, empty, or nonpositive queries cannot slip into spline calls."""
    interpolators = (
      self._adapter(allow=False).get_Pk_interpolator(nonlinear=False),
      self._adapter().get_Pk_interpolator(
        nonlinear=False, extrap_kmin=0.5, extrap_kmax=16.0),
    )
    malformed = (
      (np.array([]), 2.0),
      (np.nan, 2.0),
      (0.5, np.array([])),
      (0.5, np.nan),
      (0.5, 0.0),
      (0.5, -1.0),
    )
    for interpolator in interpolators:
      for z, k in malformed:
        with self.subTest(z=repr(z), k=repr(k)):
          with self.assertRaisesRegex(
              (ValueError, _LoggedError), "nonempty|finite|positive"):
            interpolator.P(z, k)
      for z in (-1.0e-9, 2.000000001, -0.1, 2.1):
        with self.subTest(z=z):
          with self.assertRaisesRegex(_LoggedError, "z="):
            interpolator.P(z, 2.0)

  def test_k_extrapolation_option_and_known_answer(self):
    """The default permits explicit log-log tails; false refuses them."""
    for option, expected in (({}, True),
                             ({"allow_k_extrapolation": False}, False)):
      adapter = self.module.emul_mps()
      adapter.extra_args = {
        "device": "cpu", "emulators": ["/missing/linear", "/missing/boost"],
        "compile": False, **option}
      with self.assertRaisesRegex(ValueError, "cannot be opened"):
        adapter.initialize()
      self.assertIs(adapter._allow_k_extrapolation, expected)

    default = self._adapter()
    wide = default.get_Pk_interpolator(
      nonlinear=False, extrap_kmin=0.5, extrap_kmax=16.0)
    self.assertAlmostEqual(float(wide.P(1.0, 0.5)), 0.5, places=11)
    self.assertAlmostEqual(float(wide.P(1.0, 16.0)), 512.0, places=9)

    closed = self._adapter(allow=False)
    with self.assertRaisesRegex(_LoggedError, "allow_k_extrapolation"):
      closed.get_Pk_interpolator(
        nonlinear=False, extrap_kmin=0.5, extrap_kmax=16.0)
    closed_range = closed.get_Pk_interpolator(nonlinear=False)
    for k in (0.999999999, 8.000000001):
      with self.assertRaisesRegex(_LoggedError, "k="):
        closed_range.P(1.0, k)

    for bounds in ({"extrap_kmin": 2.0}, {"extrap_kmax": 4.0}):
      with self.assertRaisesRegex(ValueError, "saved k range"):
        default.get_Pk_interpolator(nonlinear=False, **bounds)

    open_inside = default.get_Pk_interpolator(nonlinear=False)
    closed_inside = closed.get_Pk_interpolator(nonlinear=False)
    for z, k in ((1.0, 4.0), (0.5, np.sqrt(8.0))):
      with self.subTest(z=z, k=k):
        expected = (1.0 + z) * k ** 2
        self.assertAlmostEqual(float(open_inside.P(z, k)), expected,
                               places=10)
        self.assertAlmostEqual(float(closed_inside.P(z, k)), expected,
                               places=10)

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

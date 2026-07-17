"""CPU tests for generator inputs that must refuse before output starts."""

import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compute_data_vectors.generator_ingress import direct_child_filename
from compute_data_vectors.generator_ingress import convert_prior_bounds
from compute_data_vectors.generator_ingress import finite_number
from compute_data_vectors.generator_ingress import load_parameter_covariance
from compute_data_vectors.generator_ingress import native_boolean
from compute_data_vectors.generator_ingress import native_integer
from compute_data_vectors.generator_ingress import parameter_labels
from compute_data_vectors.generator_ingress import select_unique_rows
from compute_data_vectors.generator_ingress import validate_fiducial
from compute_data_vectors.generator_ingress import validate_train_args


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _parse_repository_file(relative_path):
  """Return one repository Python file as an abstract syntax tree."""
  path = _REPOSITORY_ROOT / relative_path
  return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_method(tree, class_name, method_name):
  """Find one named method without depending on line numbers or spacing."""
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == class_name:
      for member in node.body:
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) \
            and member.name == method_name:
          return member
  raise AssertionError(f"could not find {class_name}.{method_name}")


def _called_name(call):
  """Return the final name in a function call, such as ``save`` in np.save."""
  if not isinstance(call, ast.Call):
    return None
  function = call.func
  if isinstance(function, ast.Name):
    return function.id
  if isinstance(function, ast.Attribute):
    return function.attr
  return None


def _calls(node, name=None):
  """Return calls below *node*, optionally restricted to one final name."""
  found = [child for child in ast.walk(node) if isinstance(child, ast.Call)]
  if name is not None:
    found = [call for call in found if _called_name(call) == name]
  return sorted(found, key=lambda call: (call.lineno, call.col_offset))


def _slice_is_text(subscript, text):
  """Recognize a string key in a subscript on supported Python versions."""
  value = subscript.slice
  if isinstance(value, ast.Constant):
    return value.value == text
  return False


def _open_writes(call):
  """Say whether an ``open`` call requests a mode that can change a file."""
  if _called_name(call) != "open":
    return False
  mode = None
  if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
    mode = call.args[1].value
  for keyword in call.keywords:
    if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
      mode = keyword.value.value
  if mode is None:
    return False
  return isinstance(mode, str) and any(letter in mode for letter in "wax+")


def _output_mutations(node):
  """Find ordinary file-creation and file-writing calls below *node*."""
  mutation_names = {
    "mkdir", "makedirs", "open_memmap", "save", "savetxt", "savez",
    "savez_compressed", "touch", "write_bytes", "write_text",
  }
  return [call for call in _calls(node)
          if _called_name(call) in mutation_names or _open_writes(call)]


def _guard_excludes_fresh(test):
  """Recognize a branch whose body cannot run for a fresh operation."""
  if isinstance(test, ast.Compare) and len(test.ops) == 1 \
      and len(test.comparators) == 1:
    comparator = test.comparators[0]
    if isinstance(comparator, ast.Constant) and comparator.value == "fresh":
      return isinstance(test.ops[0], ast.NotEq)
    if isinstance(test.ops[0], ast.NotIn) and isinstance(
        comparator, (ast.Tuple, ast.List, ast.Set)):
      values = [element.value for element in comparator.elts
                if isinstance(element, ast.Constant)]
      return "fresh" in values
    if isinstance(test.ops[0], ast.In) and isinstance(
        comparator, (ast.Tuple, ast.List, ast.Set)):
      values = [element.value for element in comparator.elts
                if isinstance(element, ast.Constant)]
      return "fresh" not in values
  if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
    operand = test.operand
    if isinstance(operand, ast.Compare) and len(operand.ops) == 1 \
        and isinstance(operand.ops[0], ast.Eq) \
        and len(operand.comparators) == 1 \
        and isinstance(operand.comparators[0], ast.Constant):
      return operand.comparators[0].value == "fresh"
  return False


def _guard_requires_fresh(test):
  """Recognize a branch that runs only for a fresh operation."""
  if isinstance(test, ast.Compare) and len(test.ops) == 1 \
      and len(test.comparators) == 1:
    comparator = test.comparators[0]
    if isinstance(comparator, ast.Constant) and comparator.value == "fresh":
      return isinstance(test.ops[0], ast.Eq)
    if isinstance(test.ops[0], ast.In) and isinstance(
        comparator, (ast.Tuple, ast.List, ast.Set)):
      values = [element.value for element in comparator.elts
                if isinstance(element, ast.Constant)]
      return values == ["fresh"]
  return False


def _enclosing_if(function, target):
  """Return the narrowest ``if`` whose body contains *target*."""
  parents = {}
  for parent in ast.walk(function):
    for child in ast.iter_child_nodes(parent):
      parents[child] = parent
  current = target
  while current in parents:
    current = parents[current]
    if isinstance(current, ast.If):
      return current
  return None


def _names_read(node):
  """Return local names read by an expression."""
  return {child.id for child in ast.walk(node)
          if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def _attribute_chain(node):
  """Return a dotted attribute name such as ``self.args.seed``."""
  parts = []
  while isinstance(node, ast.Attribute):
    parts.append(node.attr)
    node = node.value
  if isinstance(node, ast.Name):
    parts.append(node.id)
    return ".".join(reversed(parts))
  return None


def _bound_local_names(target):
  """Return names bound by an assignment target, excluding ``self``."""
  if isinstance(target, ast.Name):
    return {target.id}
  if isinstance(target, (ast.Tuple, ast.List)):
    names = set()
    for element in target.elts:
      names.update(_bound_local_names(element))
    return names
  return set()


class PrimitiveSettingTests(unittest.TestCase):

  def test_native_integer_does_not_treat_boolean_or_float_as_an_integer(self):
    self.assertEqual(native_integer(3, "grid count", minimum=1), 3)
    self.assertEqual(native_integer(2, "mode", allowed=(1, 2)), 2)
    for value in (True, False, 3.0, np.int64(3), "3"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "native Python integer"):
          native_integer(value, "grid count")
    with self.assertRaisesRegex(ValueError, "at least 1"):
      native_integer(0, "grid count", minimum=1)
    with self.assertRaisesRegex(ValueError, "one of"):
      native_integer(3, "mode", allowed=(1, 2))

  def test_finite_number_accepts_native_numbers_without_accepting_coercions(self):
    self.assertEqual(finite_number(4, "edge"), 4.0)
    self.assertEqual(finite_number(-1.25, "edge"), -1.25)
    for value in (True, np.float64(2.0), "2", 10 ** 10000,
                  float("nan"), float("inf")):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "native Python|finite"):
          finite_number(value, "edge")

  def test_native_boolean_does_not_accept_zero_or_one(self):
    self.assertIs(native_boolean(True, "endpoint switch"), True)
    self.assertIs(native_boolean(False, "endpoint switch"), False)
    for value in (0, 1, "true", np.bool_(True)):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "native Python boolean"):
          native_boolean(value, "endpoint switch")

  def test_supporting_filename_cannot_leave_the_yaml_folder(self):
    self.assertEqual(
      direct_child_filename("proposal.covmat", "covariance"),
      "proposal.covmat")
    invalid = (
      "../proposal.covmat", "sub/proposal.covmat", "sub\\proposal.covmat",
      "/tmp/proposal.covmat", ".", "..", " proposal.covmat", "bad\x00name",
      "C:outside.covmat", "bad?.covmat", "NUL.txt", "trailing.",
    )
    for value in invalid:
      with self.subTest(value=value):
        with self.assertRaisesRegex(
            ValueError, "direct child|one nonempty filename|control|portable"):
          direct_child_filename(value, "covariance")


class TrainArgsTests(unittest.TestCase):

  def test_uniform_and_gaussian_modes_require_their_exact_fields(self):
    uniform = {
      "probe": "background",
      "ord": [["H0", "omegam"]],
      "grid": [0.0, 1.0, 20],
    }
    self.assertEqual(
      validate_train_args(uniform, ["grid"], True), ["H0", "omegam"])

    gaussian = {
      "probe": "background",
      "ord": [["H0", "omegam"]],
      "grid": [0.0, 1.0, 20],
      "fiducial": {"H0": 70.0, "omegam": 0.3},
      "params_covmat_file": "proposal.covmat",
    }
    self.assertEqual(
      validate_train_args(gaussian, ("grid",), False), ["H0", "omegam"])

    with self.assertRaisesRegex(ValueError, "unknown"):
      validate_train_args({**uniform, "typo": 7}, ["grid"], True)
    with self.assertRaisesRegex(ValueError, "missing"):
      validate_train_args(uniform, ["grid"], False)
    with self.assertRaisesRegex(ValueError, "unknown"):
      validate_train_args(gaussian, ["grid"], True)

  def test_parameter_order_has_one_nonempty_unique_native_string_list(self):
    base = {"probe": "cs", "ord": [["H0", "omegam"]]}
    invalid_orders = (
      [],
      ["H0", "omegam"],
      [["H0"], ["omegam"]],
      [[]],
      [["H0", "H0"]],
      [["H0", ""]],
      [["H0", 7]],
      (("H0", "omegam"),),
    )
    for order in invalid_orders:
      with self.subTest(order=order):
        with self.assertRaisesRegex(ValueError, "train_args.ord"):
          validate_train_args({**base, "ord": order}, [], True)

    for unsafe in ("two words", "line\nbreak", "tab\tname", "nul\x00name"):
      with self.subTest(unsafe=unsafe):
        with self.assertRaisesRegex(ValueError, "one visible token"):
          validate_train_args(
            {**base, "ord": [[unsafe]]}, [], True)

  def test_mapping_probe_and_family_key_types_are_not_guessed(self):
    with self.assertRaisesRegex(ValueError, "YAML mapping"):
      validate_train_args([], [], True)
    with self.assertRaisesRegex(ValueError, "probe"):
      validate_train_args({"probe": 7, "ord": [["H0"]]}, [], True)
    with self.assertRaisesRegex(ValueError, "repeated or reserved"):
      validate_train_args(
        {"probe": "cs", "ord": [["H0"]]}, ["probe"], True)
    with self.assertRaisesRegex(ValueError, "native list or tuple"):
      validate_train_args({"probe": "cs", "ord": [["H0"]]}, set(), True)

  def test_gaussian_support_fields_have_unambiguous_container_types(self):
    base = {
      "probe": "cs",
      "ord": [["H0"]],
      "fiducial": {"H0": 70.0},
      "params_covmat_file": "proposal.covmat",
    }
    with self.assertRaisesRegex(ValueError, "fiducial"):
      validate_train_args({**base, "fiducial": []}, [], False)
    with self.assertRaisesRegex(ValueError, "params_covmat_file"):
      validate_train_args({**base, "params_covmat_file": "  "}, [], False)
    for path in ("../cov.txt", "support/cov.txt", "/tmp/cov.txt"):
      with self.subTest(path=path):
        with self.assertRaisesRegex(ValueError, "direct child"):
          validate_train_args(
            {**base, "params_covmat_file": path}, [], False)


class CovarianceFileTests(unittest.TestCase):

  def setUp(self):
    self._temporary = tempfile.TemporaryDirectory()
    self.path = Path(self._temporary.name) / "proposal.covmat"

  def tearDown(self):
    self._temporary.cleanup()

  def write(self, text):
    self.path.write_text(text, encoding="utf-8")

  def test_header_superset_is_selected_in_requested_order(self):
    self.write(
      "# nuisance H0 omegam\n"
      "9.0 0.3 0.4\n"
      "0.3 4.0 0.2\n"
      "0.4 0.2 1.0\n")
    selected = load_parameter_covariance(self.path, ["omegam", "H0"])
    np.testing.assert_array_equal(selected, [[1.0, 0.2], [0.2, 4.0]])
    self.assertEqual(selected.dtype, np.dtype(np.float64))

  def test_one_by_one_covariance_remains_two_dimensional(self):
    self.write("# H0\n4.0\n")
    selected = load_parameter_covariance(self.path, ["H0"])
    self.assertEqual(selected.shape, (1, 1))
    self.assertEqual(selected[0, 0], 4.0)

  def test_header_must_be_present_unique_and_cover_sampled_names(self):
    cases = (
      ("H0 omegam\n1 0\n0 1\n", "first line"),
      ("# H0 H0\n1 0\n0 1\n", "repeats name"),
      ("# H0\n1\n", "missing sampled names"),
    )
    for text, message in cases:
      with self.subTest(message=message):
        self.write(text)
        with self.assertRaisesRegex(ValueError, message):
          load_parameter_covariance(self.path, ["H0", "omegam"])

  def test_body_must_match_header_and_be_a_finite_symmetric_square(self):
    cases = (
      ("# H0 omegam\n1 0 2\n0 1 2\n", "square"),
      ("# H0 omegam x\n1 0\n0 1\n", "header names"),
      ("# H0 omegam\n1 nan\nnan 1\n", "finite"),
      ("# H0 omegam\n1 0.1\n0.2 1\n", "symmetric"),
      ("# H0 omegam\n0 0\n0 1\n", "positive"),
    )
    for text, message in cases:
      with self.subTest(message=message):
        self.write(text)
        with self.assertRaisesRegex(ValueError, message):
          load_parameter_covariance(self.path, ["H0", "omegam"])

  def test_roundoff_asymmetry_is_normalized_but_real_asymmetry_refuses(self):
    self.write("# a b\n1.0 0.1\n0.10000000000000002 2.0\n")
    matrix = load_parameter_covariance(self.path, ["a", "b"])
    self.assertTrue(np.array_equal(matrix, matrix.T))
    self.assertAlmostEqual(matrix[0, 1], 0.1)

    self.write("# a b\n1.0 0.1\n0.100001 2.0\n")
    with self.assertRaisesRegex(ValueError, "symmetric"):
      load_parameter_covariance(self.path, ["a", "b"])

    self.write("# a b\n1e20 0.1\n1.0 1e20\n")
    with self.assertRaisesRegex(ValueError, "symmetric"):
      load_parameter_covariance(self.path, ["a", "b"])

    self.write("# a b\n1e308 1e308\n1e308 1e308\n")
    matrix = load_parameter_covariance(self.path, ["a", "b"])
    self.assertTrue(np.isfinite(matrix).all())
    self.assertTrue(np.array_equal(matrix, matrix.T))

    tiny = float(np.finfo(np.float64).tiny)
    next_tiny = float(np.nextafter(tiny, np.inf))
    self.write(
      "# a b\n"
      + f"1 {tiny:.17e}\n"
      + f"{next_tiny:.17e} 1\n")
    matrix = load_parameter_covariance(self.path, ["a", "b"])
    self.assertTrue(np.array_equal(matrix, matrix.T))

  def test_non_numeric_body_and_unreadable_path_have_contextual_errors(self):
    self.write("# H0\nhello\n")
    with self.assertRaisesRegex(ValueError, "numeric table"):
      load_parameter_covariance(self.path, ["H0"])
    with self.assertRaisesRegex(ValueError, "cannot read"):
      load_parameter_covariance(self.path.with_name("missing"), ["H0"])
    with self.assertRaisesRegex(ValueError, "path or string"):
      load_parameter_covariance([], ["H0"])


class FiducialAndLabelTests(unittest.TestCase):

  def test_prior_bound_conversion_preserves_open_and_finite_endpoints(self):
    hard, finite = convert_prior_bounds(
      [[-np.inf, np.inf], [0.0, 10.0]],
      [[-4.0, 4.0], [1.0, 9.0]],
      dtype=np.float32)
    self.assertTrue(np.isneginf(hard[0, 0]))
    self.assertTrue(np.isposinf(hard[0, 1]))
    self.assertTrue(np.isfinite(hard[1]).all())
    self.assertTrue(np.isfinite(finite).all())

    with self.assertRaisesRegex(ValueError, "finite.*became nonfinite"):
      convert_prior_bounds(
        [[0.0, 1.0e100]], [[1.0, 2.0]], dtype=np.float32)

  def test_fiducial_vector_follows_parameter_order_and_storage_dtype(self):
    result = validate_fiducial(
      {"unused": 5.0, "H0": 70, "omegam": 0.3},
      ["omegam", "H0"],
      dtype=np.float32)
    np.testing.assert_allclose(result, [0.3, 70.0])
    self.assertEqual(result.dtype, np.dtype(np.float32))

  def test_fiducial_refuses_missing_non_native_and_converted_nonfinite_values(self):
    with self.assertRaisesRegex(ValueError, "missing"):
      validate_fiducial({}, ["H0"])
    for value in (True, "70", np.float64(70), float("nan")):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "native Python|finite"):
          validate_fiducial({"H0": value}, ["H0"])
    with self.assertRaisesRegex(ValueError, "remain finite"):
      validate_fiducial({"H0": 1.0e300}, ["H0"], dtype=np.float32)
    with self.assertRaisesRegex(ValueError, "real floating dtype"):
      validate_fiducial({"H0": 70.0}, ["H0"], dtype=np.complex128)

  def test_missing_null_and_blank_latex_use_parameter_name(self):
    info = {
      "H0": {},
      "omegam": {"latex": None},
      "sigma8": {"latex": "  "},
      "ns": {"latex": " n_s "},
    }
    self.assertEqual(
      parameter_labels(info, ["H0", "omegam", "sigma8", "ns"]),
      ["H0", "omegam", "sigma8", "n_s"])

  def test_malformed_parameter_information_is_refused(self):
    with self.assertRaisesRegex(ValueError, "missing"):
      parameter_labels({}, ["H0"])
    with self.assertRaisesRegex(ValueError, "must be a mapping"):
      parameter_labels({"H0": None}, ["H0"])
    with self.assertRaisesRegex(ValueError, "string, null, or absent"):
      parameter_labels({"H0": {"latex": 7}}, ["H0"])

  def test_labels_allow_readable_latex_but_refuse_table_controls(self):
    self.assertEqual(
      parameter_labels(
        {"a": {"latex": r"A_{\rm visible} value"}}, ["a"]),
      [r"A_{\rm visible} value"])
    for label in (
        "two\nlines", "carriage\rreturn", "tab\tlabel", "nul\x00label"):
      with self.subTest(label=label):
        with self.assertRaisesRegex(ValueError, "control characters"):
          parameter_labels({"a": {"latex": label}}, ["a"])


class UniqueRowSelectionTests(unittest.TestCase):

  def test_selector_returns_exact_requested_unique_rows_and_matching_scores(self):
    samples = np.array([
      [1.0, 2.0],
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ])
    log_prob = np.array([-1.0, -99.0, -2.0, -3.0])
    rows, scores = select_unique_rows(
      samples, log_prob, requested=3, ndim=2,
      rng=np.random.default_rng(11))
    self.assertEqual(rows.shape, (3, 2))
    self.assertEqual(scores.shape, (3,))
    self.assertEqual(len(np.unique(rows, axis=0)), 3)
    association = {tuple(row): score for row, score in zip(rows, scores)}
    self.assertEqual(association[(1.0, 2.0)], -1.0)
    self.assertEqual(association[(3.0, 4.0)], -2.0)
    self.assertEqual(association[(5.0, 6.0)], -3.0)

  def test_selector_preserves_the_previous_seeded_valid_row_choice(self):
    samples = np.array([
      [5.0, 6.0],
      [1.0, 2.0],
      [3.0, 4.0],
      [1.0, 2.0],
    ])
    log_prob = np.array([-3.0, -1.0, -2.0, -99.0])
    unique_rows, first_indices = np.unique(
      samples, axis=0, return_index=True)
    expected_rng = np.random.default_rng(19)
    expected_indices = expected_rng.choice(
      np.arange(len(unique_rows)), size=2, replace=False)
    expected_rows = unique_rows[expected_indices]
    expected_scores = log_prob[first_indices][expected_indices]

    rows, scores = select_unique_rows(
      samples, log_prob, requested=2, ndim=2,
      rng=np.random.default_rng(19))
    np.testing.assert_array_equal(rows, expected_rows)
    np.testing.assert_array_equal(scores, expected_scores)

  def test_unique_row_shortfall_is_a_hard_refusal(self):
    samples = [[1.0], [1.0], [2.0]]
    with self.assertRaisesRegex(ValueError, "only 2 unique rows.*requested 3"):
      select_unique_rows(
        samples, [-1.0, -1.0, -2.0], requested=3, ndim=1,
        rng=np.random.default_rng(4))

  def test_rows_must_be_unique_at_the_published_float32_precision(self):
    samples = np.array([[1.00000000001], [1.00000000002]])
    probabilities = np.array([0.0, 1.0])
    with self.assertRaisesRegex(ValueError, "only 1 unique row.*requested 2"):
      select_unique_rows(
        samples, probabilities, requested=2, ndim=1,
        rng=np.random.default_rng(11))

  def test_shape_and_finiteness_are_checked_before_selection(self):
    cases = (
      ([[1.0, 2.0]], [-1.0], 1, "shape"),
      ([[1.0]], [[-1.0]], 1, "one-dimensional"),
      ([[float("nan")]], [-1.0], 1, "sample rows"),
      ([[1.0]], [float("inf")], 1, "log probabilities"),
    )
    for samples, scores, ndim, message in cases:
      with self.subTest(message=message):
        with self.assertRaisesRegex(ValueError, message):
          select_unique_rows(
            samples, scores, requested=1, ndim=ndim,
            rng=np.random.default_rng(4))

  def test_random_generator_must_return_one_valid_unique_row_selection(self):
    class MissingChoice:
      pass

    class InvalidChoice:
      def choice(self, values, size, replace):
        return np.full(size, len(values), dtype=int)

    class FloatingChoice:
      def choice(self, values, size, replace):
        return np.arange(size, dtype=float)

    for rng, message in ((MissingChoice(), "with choice"),
                         (InvalidChoice(), "invalid unique-row selection"),
                         (FloatingChoice(), "invalid unique-row selection")):
      with self.subTest(message=message):
        with self.assertRaisesRegex(ValueError, message):
          select_unique_rows([[1.0], [2.0]], [-1.0, -2.0], 1, 1, rng)


class ProductionBindingTests(unittest.TestCase):
  """Prove that the checked helpers remain on the live generator path."""

  @classmethod
  def setUpClass(cls):
    cls.core_tree = _parse_repository_file(
      "compute_data_vectors/generator_core.py")

  def test_setup_validates_inputs_without_creating_an_output_directory(self):
    """Malformed settings must stop before a chain folder or file is made."""
    setup = _class_method(self.core_tree, "GeneratorCore", "__setup_flags")
    helper_names = {
      "validate_train_args",
      "validate_fiducial",
      "load_parameter_covariance",
      "parameter_labels",
    }
    observed = {_called_name(call) for call in _calls(setup)}
    self.assertTrue(
      helper_names.issubset(observed),
      "GeneratorCore.__setup_flags must call every common ingress helper "
      "before a publication draft can exist")
    self.assertFalse(
      _calls(setup, "_prepare_dataset_publication"),
      "setup must validate only; it must not prepare an output draft")

    mutations = _output_mutations(setup)
    self.assertEqual(
      mutations, [],
      "setup may read the YAML and covariance, but may not create a "
      "directory or write a file; found: "
      + repr([(_called_name(call), call.lineno) for call in mutations]))

  def test_each_shared_scalar_control_keeps_its_production_validator(self):
    """Every documented strict CLI scalar must remain bound to its flag."""
    setup = _class_method(self.core_tree, "GeneratorCore", "__setup_flags")
    observed = {}
    for call in _calls(setup, "native_integer"):
      if not call.args:
        continue
      target = _attribute_chain(call.args[0])
      if target not in ("self.freqchk", "self.nparams",
                        "self.temp", "self.unif"):
        continue
      observed[target] = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in call.keywords
      }
    self.assertEqual(observed, {
      "self.freqchk": {"minimum": 1000},
      "self.nparams": {"minimum": 200},
      "self.temp": {"minimum": 1},
      "self.unif": {"allowed": (0, 1)},
    })

    refusal_conditions = {}
    for branch in ast.walk(setup):
      if not isinstance(branch, ast.If):
        continue
      messages = [constant.value for constant in ast.walk(branch)
                  if isinstance(constant, ast.Constant)
                  and isinstance(constant.value, str)]
      for flag in ("--boundary", "--maxcorr"):
        if any(flag in message for message in messages):
          refusal_conditions[flag] = ast.unparse(branch.test)
    self.assertEqual(set(refusal_conditions), {"--boundary", "--maxcorr"})
    boundary = refusal_conditions["--boundary"]
    self.assertIn("type(boundary) is not float", boundary)
    self.assertIn("math.isfinite(boundary)", boundary)
    self.assertIn("0.0 < boundary <= 1.0", boundary)
    maxcorr = refusal_conditions["--maxcorr"]
    self.assertIn("type(self.maxcorr) is not float", maxcorr)
    self.assertIn("math.isfinite(self.maxcorr)", maxcorr)
    self.assertIn("0.01 < self.maxcorr <= 1.0", maxcorr)

  def test_covariance_file_resolves_beside_the_actual_yaml(self):
    """A nested YAML must not read a same-named file beside ``fileroot``."""
    setup = _class_method(self.core_tree, "GeneratorCore", "__setup_flags")
    calls = _calls(setup, "load_parameter_covariance")
    self.assertEqual(len(calls), 1)
    self.assertGreaterEqual(len(calls[0].args), 1)
    self.assertEqual(
      ast.unparse(calls[0].args[0]),
      "Path(self.yaml).parent / raw_covmat_file")

  def test_fresh_constructor_samples_before_it_prepares_a_private_draft(self):
    """Only resume/append may need an output draft before sampling begins."""
    initializer = _class_method(self.core_tree, "GeneratorCore", "__init__")
    setup_calls = _calls(initializer, "__setup_flags")
    prepare_calls = _calls(initializer, "_prepare_dataset_publication")
    run_calls = _calls(initializer, "__run_mcmc")
    self.assertEqual(len(setup_calls), 1)
    self.assertEqual(len(prepare_calls), 1)
    self.assertEqual(len(run_calls), 1)
    self.assertLess(setup_calls[0].lineno, prepare_calls[0].lineno)
    self.assertLess(prepare_calls[0].lineno, run_calls[0].lineno)

    guard = _enclosing_if(initializer, prepare_calls[0])
    self.assertIsNotNone(
      guard, "early publication preparation must have a non-fresh guard")
    self.assertTrue(
      _guard_excludes_fresh(guard.test),
      "a fresh constructor must reach __run_mcmc without preparing files")

  def test_mcmc_refuses_duplicate_rows_before_the_first_fresh_output(self):
    """The exact row count is established before chain sidecars are written."""
    runner = _class_method(self.core_tree, "GeneratorCore", "__run_mcmc")
    select_calls = _calls(runner, "select_unique_rows")
    prepare_calls = _calls(runner, "_prepare_dataset_publication")
    mutations = _output_mutations(runner)
    self.assertEqual(
      len(select_calls), 1,
      "the Gaussian path must use the refusing unique-row selector")
    self.assertEqual(
      len(prepare_calls), 1,
      "fresh sampling must have one publication boundary")
    self.assertTrue(mutations, "the production writer calls were not found")
    first_write = min(mutations, key=lambda call: call.lineno)
    self.assertLess(select_calls[0].lineno, prepare_calls[0].lineno)
    self.assertLess(
      prepare_calls[0].lineno, first_write.lineno,
      "fresh publication preparation must remain before the first write")

    guard = _enclosing_if(runner, prepare_calls[0])
    self.assertIsNotNone(guard)
    self.assertTrue(
      _guard_requires_fresh(guard.test),
      "the in-method publication boundary must belong only to a fresh run")

  def test_paramnames_writer_uses_the_labels_validated_during_setup(self):
    """A missing optional ``latex`` key must not fail after chain output."""
    runner = _class_method(self.core_tree, "GeneratorCore", "__run_mcmc")
    stored_label_lists = []
    for call in _calls(runner, "list"):
      if len(call.args) != 1:
        continue
      argument = call.args[0]
      if isinstance(argument, ast.Attribute) \
          and isinstance(argument.value, ast.Name) \
          and argument.value.id == "self" \
          and argument.attr == "parameter_labels":
        stored_label_lists.append(call)
    self.assertEqual(
      len(stored_label_lists), 1,
      "the paramnames writer must copy self.parameter_labels prepared by "
      "parameter_labels() during setup")

    direct_latex_reads = [
      node for node in ast.walk(runner)
      if isinstance(node, ast.Subscript) and _slice_is_text(node, "latex")
    ]
    self.assertEqual(
      direct_latex_reads, [],
      "__run_mcmc must not read model_info[name]['latex'] after output began")

  def test_family_grid_settings_use_refusing_helpers_not_type_coercion(self):
    """Strings, booleans, and NumPy scalars must not be guessed into grids."""
    expectations = {
      "compute_data_vectors/dataset_generator_background.py": {
        "finite_number": 2,
        "native_integer": 1,
      },
      "compute_data_vectors/dataset_generator_cmb.py": {
        "native_integer": 2,
      },
      "compute_data_vectors/dataset_generator_mps.py": {
        "finite_number": 5,
        "native_integer": 2,
        "native_boolean": 1,
      },
    }
    for relative_path, expected_calls in expectations.items():
      with self.subTest(driver=relative_path):
        tree = _parse_repository_file(relative_path)
        reader = _class_method(tree, "dataset", "_read_train_args")
        names = [_called_name(call) for call in _calls(reader)]
        for helper_name, minimum in expected_calls.items():
          self.assertGreaterEqual(
            names.count(helper_name), minimum,
            f"{relative_path} must validate raw YAML values with "
            f"{helper_name} instead of converting them")

        # Follow simple local aliases such as
        # ``spec = train_args['z_sn']`` and ``for seg in z_segments``. This
        # catches a future return to float(spec[0]), int(seg[2]), or
        # bool(setting) while allowing conversions of arrays that have already
        # passed a refusing helper.
        raw_names = {"train_args"}
        changed = True
        while changed:
          changed = False
          for node in ast.walk(reader):
            targets = []
            source = None
            if isinstance(node, ast.Assign):
              targets = node.targets
              source = node.value
            elif isinstance(node, ast.AnnAssign):
              targets = [node.target]
              source = node.value
            elif isinstance(node, ast.For):
              targets = [node.target]
              source = node.iter
            if source is None or not (_names_read(source) & raw_names):
              continue
            for target in targets:
              for name in _bound_local_names(target):
                if name not in raw_names:
                  raw_names.add(name)
                  changed = True

        lossy_calls = []
        for call in _calls(reader):
          if _called_name(call) not in {"float", "int", "bool"}:
            continue
          if any(_names_read(argument) & raw_names for argument in call.args):
            lossy_calls.append(call)
        self.assertEqual(
          lossy_calls, [],
          f"{relative_path} must not coerce raw YAML values with float(), "
          "int(), or bool(); found lines "
          + repr([call.lineno for call in lossy_calls]))


if __name__ == "__main__":
  unittest.main()

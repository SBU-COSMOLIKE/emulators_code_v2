"""Tests for the GetDist posterior column meaning in the generator chain table.

The generator's parameter table reserves column two for GetDist's
``minuslogpost`` column. GetDist ranks a smaller value as the better sample,
so the generator must store minus the log posterior, not the sampler's own
log probability. A uniform run evaluates no posterior, so it must write a
neutral value that prevents any row from outranking another.
"""

import ast
import contextlib
import io
import os
import re
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

try:
  from getdist import loadMCSamples
except ImportError:
  loadMCSamples = None

_GETDIST_AVAILABLE = loadMCSamples is not None


# -----------------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------------

def _run_mcmc_tree(source_text):
  """Return the ``__run_mcmc`` method node from the parsed generator source.

  Arguments:
    source_text (str): the full text of ``generator_core.py``.

  Returns:
    ast.FunctionDef: the one ``__run_mcmc`` method inside ``GeneratorCore``.

  Raises:
    AssertionError: when ``GeneratorCore`` or ``__run_mcmc`` is not unique.
  """
  tree = ast.parse(source_text)
  classes = []
  methods = []
  for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "GeneratorCore":
      classes.append(node)
    if isinstance(node, ast.FunctionDef) and node.name == "__run_mcmc":
      methods.append(node)
  assert len(classes) == 1, f"expected one GeneratorCore class, found {len(classes)}"
  assert len(methods) == 1, f"expected one __run_mcmc method, found {len(methods)}"
  return methods[0]


def _assignments_to(function_node, name):
  """Return every assignment inside ``function_node`` whose target is ``name``.

  Arguments:
    function_node (ast.FunctionDef): the method to search.
    name (str): the left-hand side identifier to match.

  Returns:
    list[ast.Assign]: matching assignment statements.
  """
  found = []
  for node in ast.walk(function_node):
    if not isinstance(node, ast.Assign):
      continue
    if len(node.targets) != 1:
      continue
    target = node.targets[0]
    if isinstance(target, ast.Name) and target.id == name:
      found.append(node)
  return found


def _only(statements, description):
  """Return the single element of ``statements``.

  Arguments:
    statements (list): a list that must contain exactly one element.
    description (str): name of the expected statement for the error message.

  Returns:
    object: the one element of ``statements``.

  Raises:
    AssertionError: when the list does not hold exactly one element.
  """
  assert len(statements) == 1, (
    f"expected exactly one {description}, found {len(statements)}"
  )
  return statements[0]


def _execute(statement, namespace):
  """Compile and execute one AST statement in ``namespace``.

  Arguments:
    statement (ast.AST): the statement to run.
    namespace (dict): the execution namespace.

  Returns:
    dict: the updated namespace.
  """
  module = ast.Module(body=[statement], type_ignores=[])
  ast.fix_missing_locations(module)
  code = compile(module, "<test>", "exec")
  exec(code, namespace)
  return namespace


def _ranking_ok(minus_logpost):
  """Return the predicate GetDist uses to prefer row 0 over row 1.

  Arguments:
    minus_logpost (np.ndarray): two-row column vector of ``minuslogpost`` values.

  Returns:
    bool: True when row 0 has the smaller (better) value.
  """
  return bool(minus_logpost[0, 0] < minus_logpost[1, 0])


# -----------------------------------------------------------------------------
# Posterior column statement tests
# -----------------------------------------------------------------------------

class PosteriorColumnStatementTests(unittest.TestCase):
  """Extract production statements from ``generator_core.py`` and run them."""

  @classmethod
  def setUpClass(cls):
    with open("compute_data_vectors/generator_core.py", encoding="utf-8") as handle:
      cls.source_text = handle.read()
    cls.run_mcmc = _run_mcmc_tree(cls.source_text)

  def test_sampler_log_probability_is_stored_with_flipped_sign(self):
    """The tempered branch stores minus the sampler's own log probability."""
    negated_candidates = []
    for node in _assignments_to(self.run_mcmc, "minus_logpost"):
      if isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
        negated_candidates.append(node)
    negated = _only(negated_candidates, "negated minus_logpost assignment")

    sampler_candidates = []
    for node in _assignments_to(self.run_mcmc, "sampler_log_prob"):
      if isinstance(node.value, ast.Call):
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == "get_log_prob":
          sampler_candidates.append(node)
    sampler_assign = _only(sampler_candidates, "sampler_log_prob assignment")
    self.assertIsNotNone(sampler_assign)
    namespace = {
      "np": np,
      "sampler_log_prob": np.array([-4.0, -9.0]),
      "keep": np.array([0, 1]),
    }
    _execute(negated, namespace)
    result = namespace["minus_logpost"]
    expected = np.array([[4.0], [9.0]])
    self.assertTrue(np.array_equal(result, expected))
    self.assertEqual(result.shape, (2, 1))
    self.assertTrue(_ranking_ok(result))

  def test_derived_chi2_column_is_twice_the_posterior_column(self):
    """The trailing column is twice the negative log posterior."""
    chi2_assign = _only(
      _assignments_to(self.run_mcmc, "chi2"),
      "chi2 assignment",
    )
    namespace = {
      "np": np,
      "minus_logpost": np.array([[4.0], [9.0]]),
    }
    _execute(chi2_assign, namespace)
    result = namespace["chi2"]
    expected = np.array([[8.0], [18.0]])
    self.assertTrue(np.array_equal(result, expected))
    self.assertTrue(np.array_equal(result / 2.0, namespace["minus_logpost"]))

  def test_uniform_rows_carry_the_neutral_zero_value(self):
    """A uniform run writes zero in the posterior column for every row."""
    zero_candidates = []
    for node in _assignments_to(self.run_mcmc, "minus_logpost"):
      if isinstance(node.value, ast.Call):
        zero_candidates.append(node)
    zero_assign = _only(zero_candidates, "zero-filled minus_logpost assignment")
    namespace = {
      "np": np,
      "nparams": 3,
      "self": SimpleNamespace(dtype=np.float32),
    }
    _execute(zero_assign, namespace)
    result = namespace["minus_logpost"]
    self.assertEqual(result.shape, (3, 1))
    self.assertEqual(result.dtype, np.float32)
    self.assertTrue(np.all(result == 0.0))
    self.assertFalse(_ranking_ok(result))

  def test_chain_header_names_the_getdist_posterior_column(self):
    """Both chain writers name the second column ``minuslogpost``."""
    lists = []
    for node in ast.walk(ast.parse(self.source_text)):
      if not isinstance(node, ast.List):
        continue
      if not node.elts:
        continue
      first = node.elts[0]
      if isinstance(first, ast.Constant) and first.value == "weights":
        lists.append(node)
    self.assertEqual(len(lists), 2)
    for header_list in lists:
      self.assertEqual(header_list.elts[1].value, "minuslogpost")
    self.assertEqual(re.findall(r"\blnp\b", self.source_text), [])

  def test_reversing_the_sign_breaks_the_ranking_assertion(self):
    """Storing the sampler's sign unchanged must reverse the ranking.

    This test mutates the production source by removing the negation and
    proves that the ranking assertion in the first test fails, so that
    assertion is load-bearing rather than incidentally true.
    """
    original = "minus_logpost = -sampler_log_prob[keep, None]"
    mutated = "minus_logpost = sampler_log_prob[keep, None]"
    count_before = self.source_text.count(original)
    self.assertEqual(count_before, 1)
    mutated_source = self.source_text.replace(original, mutated)
    mutated_tree = _run_mcmc_tree(mutated_source)
    subscript_candidates = []
    for node in _assignments_to(mutated_tree, "minus_logpost"):
      if isinstance(node.value, ast.Subscript):
        value = node.value.value
        if isinstance(value, ast.Name) and value.id == "sampler_log_prob":
          subscript_candidates.append(node)
    subscript_assign = _only(
      subscript_candidates, "mutated minus_logpost assignment"
    )
    namespace = {
      "np": np,
      "sampler_log_prob": np.array([-4.0, -9.0]),
      "keep": np.array([0, 1]),
    }
    _execute(subscript_assign, namespace)
    result = namespace["minus_logpost"]
    self.assertFalse(_ranking_ok(result))


# -----------------------------------------------------------------------------
# GetDist ranking tests
# -----------------------------------------------------------------------------

class GetDistRankingTests(unittest.TestCase):
  """Execute the production chain writer and let GetDist rank the result."""

  @classmethod
  def setUpClass(cls):
    with open("compute_data_vectors/generator_core.py", encoding="utf-8") as handle:
      cls.source_text = handle.read()
    cls.run_mcmc = _run_mcmc_tree(cls.source_text)

  def _write_two_row_chain(self, directory, minus_logpost, chi2):
    """Write a two-row chain using the production ``np.savetxt`` writer.

    Arguments:
      directory (str): temporary folder for the chain and sidecar.
      minus_logpost (np.ndarray): column two values.
      chi2 (np.ndarray): trailing column values.

    Returns:
      str: the chain root path (without ``.1.txt``).
    """
    # The production writer is np.savetxt. Find the call whose first
    # positional argument is the bare name ``fname``.
    writer = None
    for node in ast.walk(self.run_mcmc):
      if not isinstance(node, ast.Call):
        continue
      if not (isinstance(node.func, ast.Attribute)
              and node.func.attr == "savetxt"):
        continue
      if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "fname":
        writer = node
        break
    self.assertIsNotNone(writer)
    root = os.path.join(directory, "probe")
    fname = root + ".1.txt"
    namespace = {
      "np": np,
      "fname": fname,
      "w": np.ones((2, 1)),
      "minus_logpost": minus_logpost,
      "xf": np.array([[67.0], [68.0]]),
      "chi2": chi2,
      "names": ["H0"],
      "hd": "Uniform Sampling seed=1 rng=numpy.default_rng\n",
    }
    _execute(ast.Expr(value=writer), namespace)
    with open(root + ".paramnames", "w") as handle:
      handle.write("H0 H_0\n")
      handle.write("chi2* chi-square\n")
    return root

  @unittest.skipUnless(_GETDIST_AVAILABLE, "GetDist is required to rank a chain")
  def test_repaired_sign_makes_getdist_prefer_the_larger_posterior(self):
    """The repaired sign ranks the row with log p = -4 above log p = -9."""
    minus_logpost = np.array([[4.0], [9.0]])
    chi2 = np.array([[8.0], [18.0]])
    with tempfile.TemporaryDirectory() as tmp:
      root = self._write_two_row_chain(tmp, minus_logpost, chi2)
      with contextlib.redirect_stdout(io.StringIO()):
        samples = loadMCSamples(
          root, settings=dict(ignore_rows=0.0)
        )
      self.assertTrue(np.array_equal(samples.loglikes, np.array([4.0, 9.0])))
      self.assertEqual(int(np.argmin(samples.loglikes)), 0)

  @unittest.skipUnless(_GETDIST_AVAILABLE, "GetDist is required to rank a chain")
  def test_retired_sign_makes_getdist_prefer_the_smaller_posterior(self):
    """The retired sign reverses the ranking and makes GetDist prefer the worse row.

    This arm exists to prove that the previous method's assertion is
    load-bearing: GetDist itself supplies the ranking, and storing the
    sampler's sign unchanged flips the preferred row.
    """
    minus_logpost = np.array([[-4.0], [-9.0]])
    chi2 = np.array([[-8.0], [-18.0]])
    with tempfile.TemporaryDirectory() as tmp:
      root = self._write_two_row_chain(tmp, minus_logpost, chi2)
      with contextlib.redirect_stdout(io.StringIO()):
        samples = loadMCSamples(
          root, settings=dict(ignore_rows=0.0)
        )
      self.assertEqual(int(np.argmin(samples.loglikes)), 1)


if __name__ == "__main__":
  unittest.main()

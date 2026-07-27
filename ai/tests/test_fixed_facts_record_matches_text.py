"""CPU tests that a saved record must still match the text it was copied from.

A saved emulator stores the scientific record twice: as structured sections,
and as the generator's own text beside them. ``read_h5`` reads both and refuses
when they disagree, which is what makes "copied unchanged, never re-derived" a
checkable statement rather than a promise.

The existing rebuild test rewrites both halves together on purpose, so that
``read_h5`` accepts and a later geometry comparison is the witness. That leaves
the uncoordinated rewrite — one half edited, the other left alone — untested,
and mutation testing confirmed it: disabling the two-way comparison left the
whole suite green.
"""

import os
from pathlib import Path
import tempfile
import unittest

import h5py

from ai.gates.checks.compile_recipe import save_fixture
from emulator import fixed_facts, results


class RecordMatchesProducerTextTests(unittest.TestCase):
  """Require the stored sections and the stored text to agree both ways."""

  def _artifact(self, directory, label):
    """Write one valid saved emulator and return its .h5 path.

    Arguments:
      directory = the temporary folder that owns the files.
      label     = the fixture's case label.

    Returns:
      The written .h5 path.
    """
    root = os.path.join(directory, label)
    save_fixture(path_root=Path(root), compile_mode="default",
                 case_label=label)
    return root + ".h5"

  def _schema_version(self, artifact):
    """Read the schema version the file announces."""
    return int(artifact.attrs["schema_version"])

  def test_an_untouched_record_reads_back(self):
    """The baseline must pass, or the refusals below prove nothing."""
    with tempfile.TemporaryDirectory() as temp:
      path = self._artifact(temp, "record-matches")
      with h5py.File(path, "r") as artifact:
        blocks = fixed_facts.read_h5(
          f=artifact,
          schema_version=self._schema_version(artifact),
          where="untouched fixture")
      self.assertIn(fixed_facts.FIXED_FACTS_GROUP, blocks)
      self.assertIn(fixed_facts.INPUT_DOMAIN_GROUP, blocks)

  def test_a_stored_fact_edited_after_writing_is_refused(self):
    """One section edited while the producer text is left alone.

    This is the tampering the two-way check exists for: the file still
    parses, every section is still present, and the record still satisfies
    every schema law on its own. Only the comparison against the text the
    generator actually wrote can tell that a value changed.
    """
    with tempfile.TemporaryDirectory() as temp:
      path = self._artifact(temp, "record-edited")
      with h5py.File(path, "r+") as artifact:
        group = artifact[fixed_facts.FIXED_FACTS_GROUP]
        group.attrs["generator"] = "some-other-generator.py"
      with h5py.File(path, "r") as artifact:
        with self.assertRaisesRegex(
            ValueError, "does not match the producer text"):
          fixed_facts.read_h5(
            f=artifact,
            schema_version=self._schema_version(artifact),
            where="edited fixture")

  def test_a_dropped_sampled_name_is_refused(self):
    """A coordinate removed from the stored domain but left in the text.

    This one is refused before the two-way comparison runs, by the schema
    laws that require the domain's own parts to agree in length. It is kept
    because the refusal is worth holding, but the test above is the one that
    stands behind the comparison against the producer text.
    """
    with tempfile.TemporaryDirectory() as temp:
      path = self._artifact(temp, "record-dropped")
      with h5py.File(path, "r+") as artifact:
        group = artifact[fixed_facts.INPUT_DOMAIN_GROUP]
        names = [fixed_facts._plain(value) for value in group["names"][()]]
        if len(names) < 2:
          self.skipTest("fixture has too few sampled names to drop one")
        del group["names"]
        group.create_dataset(
          "names", data=names[:-1],
          dtype=h5py.string_dtype(encoding="utf-8"))
      with h5py.File(path, "r") as artifact:
        with self.assertRaises(ValueError):
          fixed_facts.read_h5(
            f=artifact,
            schema_version=self._schema_version(artifact),
            where="dropped fixture")

  def test_a_file_without_the_producer_text_is_refused(self):
    """Without the text there is nothing to check the record against."""
    with tempfile.TemporaryDirectory() as temp:
      path = self._artifact(temp, "record-no-text")
      with h5py.File(path, "r+") as artifact:
        del artifact[fixed_facts.SIDECAR_DATASET]
      with h5py.File(path, "r") as artifact:
        with self.assertRaisesRegex(
            ValueError, "not the producer's own text"):
          fixed_facts.read_h5(
            f=artifact,
            schema_version=self._schema_version(artifact),
            where="textless fixture")


if __name__ == "__main__":
  unittest.main()

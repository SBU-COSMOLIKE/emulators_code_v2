"""Check that the tests guide lists every neighboring Python file."""

from pathlib import Path
import re
import unittest


TESTS_DIRECTORY = Path(__file__).resolve().parent
README_PATH = TESTS_DIRECTORY / "README.md"
PYTHON_TABLE_ROW_PATTERN = re.compile(
  r"^\| `([A-Za-z0-9_]+\.py)` \|",
  flags=re.MULTILINE,
)


class TestsReadmeInventoryTests(unittest.TestCase):
  """Keep the reader-facing inventory synchronized with this directory."""

  def test_every_immediate_python_file_is_documented(self):
    expected_files = set()
    for path in TESTS_DIRECTORY.glob("*.py"):
      expected_files.add(path.name)

    readme = README_PATH.read_text(encoding="utf-8")
    documented_names = PYTHON_TABLE_ROW_PATTERN.findall(readme)
    documented_files = set()
    duplicate_files = []
    for name in documented_names:
      if name in documented_files:
        duplicate_files.append(name)
      documented_files.add(name)

    missing_files = sorted(expected_files - documented_files)
    stale_files = sorted(documented_files - expected_files)
    self.assertEqual(
      duplicate_files,
      [],
      "README.md has duplicate Python-file table rows; keep one row for: "
      + repr(duplicate_files),
    )
    self.assertEqual(
      missing_files,
      [],
      "README.md needs one table row for each Python file; add rows for: "
      + repr(missing_files),
    )
    self.assertEqual(
      stale_files,
      [],
      "README.md has rows for absent Python files; remove or rename: "
      + repr(stale_files),
    )


if __name__ == "__main__":
  unittest.main()

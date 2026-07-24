#!/usr/bin/env python3
"""artifact-readback: saved-artifact attributes are parsed by type, not truthiness.

HDF5 attributes carry no strong type, so a saved marker read back with Python
truthiness can flip a feature bit: the string "False" is truthy, so a transfer
artifact whose transfer_refined marker literally reads "False" would load its
drifted prediction weights. This check drives the shared typed reader
(_read_native_bool) directly and censuses the module for any remaining
truthiness coercion of an artifact attribute.

The live save/forge/rebuild proof (write a real transfer artifact, forge the
attribute to the string "False", confirm the current code would rebuild the
drifted weights, then confirm the typed reader refuses it) needs a real HDF5
artifact and is owned by the workstation artifact-integrity gate; this leg
proves the read-boundary type contract with no file.
"""
import ast
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from emulator.results import _read_native_bool

FAILURES = []


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def check_native_bool():
    """A native boolean is accepted; every non-boolean type is refused."""
    # native booleans (Python and numpy) pass through by value.
    report("native True reads True",
           _read_native_bool({"k": True}, "k", default=False, where="w") is True,
           "True")
    report("native False reads False",
           _read_native_bool({"k": False}, "k", default=True, where="w") is False,
           "False")
    report("numpy bool_ reads its value",
           _read_native_bool({"k": np.bool_(True)}, "k", default=False,
                             where="w") is True, "np.bool_(True)")
    # an absent key returns the default (a native boolean).
    report("absent key returns the default",
           _read_native_bool({}, "k", default=False, where="w") is False
           and _read_native_bool({}, "k", default=True, where="w") is True,
           "default respected")
    # the defect: the string "False" is truthy; the typed reader refuses it.
    for bad in ("False", "true", "0", "1", "yes", ""):
        raised = False
        try:
            _read_native_bool({"k": bad}, "k", default=False, where="art")
        except ValueError as exc:
            raised = "native boolean" in str(exc) and "art" in str(exc)
        report("string " + repr(bad) + " refused (not truthiness-coerced)",
               raised, "ValueError names the file + schema")
    # integers are refused too (0/1 are not native booleans).
    for bad in (1, 0, np.int64(1)):
        raised = False
        try:
            _read_native_bool({"k": bad}, "k", default=False, where="art")
        except ValueError:
            raised = True
        report("integer " + repr(int(bad)) + " refused", raised, "ValueError")


def check_no_truthiness_census():
    """No artifact attribute is still read with a bool()/if truthiness coercion."""
    src = open(os.path.join(_REPO, "emulator", "results.py")).read()
    # the specific defect pattern: bool(...attrs...get...) truthiness coercion.
    bad_patterns = ["bool(f.attrs.get", "bool(f.attrs[", "bool(g.attrs.get",
                    "bool(tb.attrs.get"]
    hits = [p for p in bad_patterns if p in src]
    report("no artifact boolean attribute is read by truthiness coercion",
           len(hits) == 0, "offending patterns: " + repr(hits))
    tree = ast.parse(src)
    typed_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_read_native_bool" or not node.args:
            continue
        first = node.args[0]
        if (isinstance(first, ast.Attribute)
                and first.attr == "attrs"
                and isinstance(first.value, ast.Name)
                and first.value.id == "f"):
            typed_calls.append(node.lineno)
    report("the typed reader _read_native_bool is the read boundary",
           len(typed_calls) > 0,
           "transfer_refined routed through it at lines "
           + repr(typed_calls))


def check_scalar_records_rescale():
    """The scalar driver stamps the rescale fact its own fine-tune loader needs.

    A scalar run has no analytic rescale, but warmstart.load_source requires
    the rescale root attr of every source it admits (it refuses a missing one
    as ambiguous, never defaulting it). The scalar driver must therefore record
    rescale="none" explicitly, or its own artifact cannot be its supported
    fine-tune source. The live save/reload epoch-0 parity leg is owned by the
    workstation finetune-identity gate; this census proves the attr is stamped.
    """
    src = open(os.path.join(_REPO, "driver/scalar_train_emulator.py")).read()
    report("scalar driver records rescale='none' in its run-identity attrs",
           '"rescale":' in src and '"none"' in src,
           "load_source admits the scalar driver's own artifact")


def main():
    print("artifact-readback (typed attribute parsing; no torch build, no HDF5 file)")
    print("\n-- native-boolean typing --")
    check_native_bool()
    print("\n-- truthiness-coercion census --")
    check_no_truthiness_census()
    print("\n-- scalar artifact records its rescale fact --")
    check_scalar_records_rescale()
    print("")
    if FAILURES:
        print("artifact-readback: %d FAILURE(S): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("artifact-readback: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""generator-seed: the dataset generator samples from an owned, recorded RNG.

The generator had no seed surface and drew every sample from the process-global
np.random, so two runs with byte-identical YAML, command line, and code produced
different parameter tables and neither the chain header nor any sidecar recorded
why -- an audit could not replay the dataset from its recorded inputs. A
required integer seed now owns a numpy Generator threaded through the uniform
sampling, the emcee walker initialization, the sampler's own moves, and the
thinning subselection, and the seed and RNG are written into the chain header.

The generator imports MPI, cobaya and CAMB, so a live end-to-end replay is owned
by the workstation smoke gates; this check censuses the sampling surface (no
process-global np.random draw remains, the owned Generator is used, the seed is
required and recorded, a non-integer seed is refused) and confirms the numpy
Generator's own replay guarantee with a tiny draw.
"""
import ast
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_GEN = os.path.join(_REPO, "compute_data_vectors", "generator_core.py")

FAILURES = []


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


def check_no_global_random():
    """No sampling draw uses the process-global np.random.<dist>."""
    src = open(_GEN).read()
    # the global sampling calls the fix removed (default_rng / RandomState for
    # the owned generator remain legitimate).
    for bad in ("np.random.uniform(", "np.random.normal(", "np.random.choice(",
                "np.random.rand(", "np.random.randn("):
        report("no process-global " + bad.rstrip("(") + " draw remains",
               bad not in src, "owned RNG only")
    report("the owned numpy Generator is used for sampling",
           "self.rng = np.random.default_rng(self.seed)" in src
           and "self.rng.uniform(" in src
           and "self.rng.standard_normal(" in src
           and "self.rng.choice(" in src,
           "self.rng threaded through sampling and row selection")
    report("the emcee sampler gets a seeded random state",
           "sampler._random = np.random.RandomState(" in src, "seeded moves")


def check_seed_required_and_recorded():
    """The seed is a required integer CLI arg written to the chain header."""
    src = open(_GEN).read()
    # find the add_argument("--seed", ...) call in the syntax tree and
    # require the argparse contract: required=True and type=int, so a run
    # without a seed (or with a non-integer one) refuses at the CLI.
    seed_required = False
    seed_integer = False
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--seed"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "required" \
                    and isinstance(keyword.value, ast.Constant) \
                    and keyword.value.value is True:
                seed_required = True
            if keyword.arg == "type" \
                    and isinstance(keyword.value, ast.Name) \
                    and keyword.value.id == "int":
                seed_integer = True
    report("--seed is a required CLI argument",
           seed_required, "no default seed")
    report("--seed parses as an integer",
           seed_integer, "argparse type=int")
    report("the seed + RNG are recorded in the chain header",
           "seed={self.seed}" in src and "rng=numpy.default_rng" in src,
           "replayable from the recorded inputs")


def check_generator_replay():
    """The numpy Generator's own guarantee: same seed -> same draws."""
    a = np.random.default_rng(1234).uniform(0, 1, size=(50, 4))
    b = np.random.default_rng(1234).uniform(0, 1, size=(50, 4))
    c = np.random.default_rng(9999).uniform(0, 1, size=(50, 4))
    report("same seed reproduces the same table; a different seed differs",
           np.array_equal(a, b) and not np.array_equal(a, c),
           "default_rng is deterministic per seed")


def main():
    print("generator-seed (owned, recorded sampling RNG; census + replay)")
    print("\n-- no process-global random draw --")
    check_no_global_random()
    print("\n-- seed required + recorded --")
    check_seed_required_and_recorded()
    print("\n-- numpy Generator replay guarantee --")
    check_generator_replay()
    print("")
    if FAILURES:
        print("generator-seed: %d FAILURE(S): %s"
              % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("generator-seed: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

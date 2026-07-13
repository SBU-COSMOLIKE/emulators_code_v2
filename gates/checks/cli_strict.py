#!/usr/bin/env python3
"""cli-strict: every public executable rejects a misspelled flag before it acts.

The training / tuning drivers and the two data producers parsed their command
line with parse_known_args and discarded the unknown tokens, so a misspelled
flag (--activaton, --quieet, --diagnostc, --sav) was silently ignored and the
run proceeded at the YAML or default value -- most dangerously publishing to
the default --save root. Strict parse_args rejects an unrecognized token with a
usage error and a nonzero exit before any data is read, any artifact is loaded,
CAMB is started, workers are spawned, or an output root is chosen.

This check censuses all eight entry points for parse_args (no parse_known_args)
and drives two representative driver mains with the expensive boundary
monkeypatched to a sentinel: a misspelled flag exits nonzero WITHOUT reaching
the boundary, while a valid command line reaches it (parsing succeeded). Strict
parsing before the boundary is a Python guarantee once parse_args is used, so
the census generalizes it to the remaining entry points.
"""
import importlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

FAILURES = []

ENTRY_POINTS = [
    "cosmic_shear_train_emulator.py",
    "cosmic_shear_sweep_ntrain_emulator.py",
    "cosmic_shear_sweep_hyperparam_emulator.py",
    "cosmic_shear_bakeoff_activation_emulator.py",
    "cosmic_shear_tune_emulator.py",
    "scalar_train_emulator.py",
    "compute_data_vectors/generator_core.py",
    "compute_data_vectors/compute_cmb_covariance.py",
]


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


class _Reached(Exception):
    """Raised by the monkeypatched boundary: the CLI passed parsing."""


def drive_main(module_name, argv, boundary="resolve_cocoa_config"):
    """Call a driver's main() with argv, the first expensive boundary replaced.

    Returns "reached-boundary" (parsing succeeded and the boundary was hit),
    ("exit", code) (parse_args rejected the command line), or "returned".
    """
    mod = importlib.import_module(module_name)
    saved_boundary = getattr(mod, boundary)
    saved_argv = sys.argv

    def _boom(*args, **kwargs):
        raise _Reached()

    setattr(mod, boundary, _boom)
    sys.argv = ["prog"] + argv
    try:
        mod.main()
        return "returned"
    except _Reached:
        return "reached-boundary"
    except SystemExit as exc:
        return ("exit", exc.code)
    finally:
        setattr(mod, boundary, saved_boundary)
        sys.argv = saved_argv


def check_census():
    """Every entry point uses strict parse_args, none parse_known_args."""
    for rel in ENTRY_POINTS:
        src = open(os.path.join(_REPO, rel)).read()
        report(rel + " parses strictly (parse_args, no parse_known_args)",
               "parse_known_args" not in src and ".parse_args()" in src,
               "strict parse")


def check_live_drivers():
    """Two driver mains: a misspelled flag exits nonzero before the boundary."""
    valid = ["--root", "x", "--fileroot", "y"]
    for module in ("cosmic_shear_train_emulator", "scalar_train_emulator"):
        # a valid command line reaches the (monkeypatched) boundary.
        result = drive_main(module, valid)
        report(module + ": a valid command line reaches the boundary",
               result == "reached-boundary", str(result))
        # a misspelled flag is rejected by parse_args before the boundary.
        result = drive_main(module, valid + ["--definitely-not-a-flag", "z"])
        rejected = isinstance(result, tuple) and result[0] == "exit" \
            and result[1] != 0
        report(module + ": a misspelled flag exits nonzero, boundary not "
               "reached", rejected, str(result))
        # a misspelled valued flag (--activaton) is likewise rejected, not
        # silently ignored and run at the default.
        result = drive_main(module, valid + ["--activaton", "relu"])
        rejected = isinstance(result, tuple) and result[0] == "exit" \
            and result[1] != 0
        report(module + ": --activaton (typo) exits nonzero, never ignored",
               rejected, str(result))


def main():
    print("cli-strict (misspelled flags are usage errors, not silent ignores)")
    print("\n-- census: all eight entry points parse strictly --")
    check_census()
    print("\n-- live: representative driver mains reject a typo --")
    check_live_drivers()
    print("")
    if FAILURES:
        print("cli-strict: %d FAILURE(S): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("cli-strict: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

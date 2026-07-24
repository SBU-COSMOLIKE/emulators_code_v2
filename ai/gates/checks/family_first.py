#!/usr/bin/env python3
"""family-first: every train / tune / sweep driver owns exactly one family.

A wrong-family YAML must fail at startup naming the right driver, never train
under the wrong public identity (and, for a scalar YAML, never reach the
run_tag KeyError deep inside the run). The direct cosmic_shear drivers used to
pass family=None, which skipped the family check entirely, so a CMB / grid /
grid2d / scalar YAML launched through driver/cosmic_shear_train_emulator.py passed
straight into training. This check drives require_family_block directly and
censuses the four cosmic_shear drivers for the "cosmolike" default plus the
unconditional call.

Importing the driver imports torch (through emulator.results), so this runs in
the torch environment; the family check itself is pure Python.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# the drivers live in driver/; put it on the path so the bare-name
# import below resolves.
_DRIVER_DIR = os.path.join(_REPO, "driver")
if _DRIVER_DIR not in sys.path:
    sys.path.insert(0, _DRIVER_DIR)

from cosmic_shear_train_emulator import require_family_block

FAILURES = []


def report(label, ok, detail=""):
    print("  [" + ("PASS" if ok else "FAIL") + "] " + label + "  (" + detail + ")")
    if not ok:
        FAILURES.append(label)


# the wrong-family block -> the driver it belongs to.
OTHER_FAMILIES = {"cmb": "driver/cmb_train_emulator.py",
                  "grid": "driver/baosn_train_emulator.py",
                  "grid2d": "driver/mps_train_emulator.py",
                  "outputs": "driver/scalar_train_emulator.py"}


def check_cosmolike_identity():
    """A direct cosmic-shear run owns cosmolike and rejects every other family."""
    for block, driver in OTHER_FAMILIES.items():
        raised = False
        try:
            require_family_block({block: {}}, "cosmolike",
                                 "cosmic_shear_train_emulator")
        except SystemExit as exc:
            raised = driver in str(exc)
        report("cosmolike identity rejects a data." + block + " YAML naming "
               + driver, raised, "SystemExit names the right driver")
    # a clean cosmolike data-vector YAML (no other family's block) is accepted.
    accepted = True
    try:
        require_family_block({"cosmolike_data_dir": "x"}, "cosmolike",
                             "cosmic_shear_train_emulator")
    except SystemExit:
        accepted = False
    report("cosmolike identity accepts a clean cosmic-shear YAML", accepted,
           "no other family block -> trains")


def check_wrapper_families():
    """A per-family wrapper accepts its own block and rejects another's."""
    accepted = True
    try:
        require_family_block({"cmb": {}}, "cmb", "cmb_train_emulator")
    except SystemExit:
        accepted = False
    report("cmb wrapper accepts a data.cmb YAML", accepted, "trains")
    raised = False
    try:
        require_family_block({"grid": {}}, "cmb", "cmb_train_emulator")
    except SystemExit as exc:
        raised = "driver/baosn_train_emulator.py" in str(exc)
    report("cmb wrapper rejects a data.grid YAML naming the grid driver",
           raised, "SystemExit names baosn")


def check_driver_census():
    """The four cosmic_shear drivers default to cosmolike and always check."""
    drivers = ["driver/cosmic_shear_train_emulator.py",
               "driver/cosmic_shear_sweep_hyperparam_emulator.py",
               "driver/cosmic_shear_sweep_ntrain_emulator.py",
               "driver/cosmic_shear_tune_emulator.py"]
    for name in drivers:
        src = open(os.path.join(_REPO, name)).read()
        default_ok = 'family="cosmolike"' in src
        # the family check must not be guarded by an is-not-None branch that a
        # direct (family=None) run would skip -- family is always an identity.
        no_guard = "if family is not None:" not in src
        calls = "require_family_block(data=cfg[\"data\"], family=family" in src
        report(name + " defaults family=cosmolike and always checks",
               default_ok and no_guard and calls,
               "default cosmolike, unconditional require_family_block")
    # the misleading dispatcher prose is gone.
    for name in drivers:
        src = open(os.path.join(_REPO, name)).read()
        report(name + " drops the 'dispatching driver trains whatever' prose",
               "dispatching driver" not in src and "trains whatever" not in src,
               "no unrestricted-dispatcher claim")


def main():
    print("family-first (driver-family enforcement; pure Python family check)")
    print("\n-- cosmolike identity --")
    check_cosmolike_identity()
    print("\n-- per-family wrappers --")
    check_wrapper_families()
    print("\n-- driver census --")
    check_driver_census()
    print("")
    if FAILURES:
        print("family-first: %d FAILURE(S): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("family-first: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

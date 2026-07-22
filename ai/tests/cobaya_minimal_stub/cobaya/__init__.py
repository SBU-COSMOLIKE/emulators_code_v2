"""A minimal on-disk stand-in for the cobaya package, for child tests.

This package exists so ai/tests/mps_sigma8_child_checks.py can import
the matter-power adapter without a real cobaya installation and without
editing any process's import table: the launching test places this
directory ahead of every installed package on PYTHONPATH before the
child process starts. The stand-in provides only the two modules the
adapter imports at load time (cobaya.theory and cobaya.log), each
reduced to the smallest surface those imports touch.

COBAYA_MINIMAL_STUB marks the package so the child checks can prove
they imported this stand-in and not a real cobaya installation.
"""

COBAYA_MINIMAL_STUB = True

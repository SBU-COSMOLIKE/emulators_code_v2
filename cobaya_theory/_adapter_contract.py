"""Shared input checks for the five Cobaya emulator adapters.

Each adapter remains a separate, explicit Cobaya ``Theory`` class.  This
module only owns configuration rules that must mean the same thing in every
class: option types, device names, and saved-emulator paths.  Keeping those
small rules here prevents one adapter from silently accepting a value that
the other four refuse.
"""

from collections.abc import Mapping, Sequence
import os

import torch


_DEVICE_NAMES = ("cpu", "cuda", "mps")


def validate_extra_args(extra_args, *, adapter, allowed, retired):
    """Require a mapping containing only the adapter's documented options."""
    if not isinstance(extra_args, Mapping):
        raise ValueError(
            adapter + ": extra_args must be a mapping, got "
            + repr(type(extra_args).__name__))
    unknown = [key for key in extra_args if key not in allowed]
    if unknown:
        raise ValueError(
            adapter + ": unrecognized extra_args key(s) "
            + repr(sorted(unknown, key=repr))
            + ". Accepted keys are " + repr(list(allowed)) + "; " + retired)


def exact_bool(extra_args, name, *, adapter, default=False):
    """Read one Boolean option without accepting truthy strings or numbers."""
    value = extra_args.get(name, default)
    if type(value) is not bool:
        raise ValueError(
            adapter + ": extra_args " + repr(name)
            + " must be an actual Boolean (true or false), got "
            + repr(value))
    return value


def exact_choice(extra_args, name, *, adapter, choices, default):
    """Read one native string chosen from a closed, case-sensitive set."""
    value = extra_args.get(name, default)
    if type(value) is not str or value not in choices:
        raise ValueError(
            adapter + ": extra_args " + repr(name)
            + " must be one of " + repr(tuple(choices))
            + " as written, got " + repr(value))
    return value


def pick_device(extra_args, *, adapter):
    """Resolve a registered device, retaining the documented safe fallback.

    An unknown name is a configuration error.  A known accelerator that is
    unavailable on this machine falls back to the best available supported
    device, preserving the adapters' established portable behavior.
    """
    requested = exact_choice(
        extra_args, "device", adapter=adapter,
        choices=_DEVICE_NAMES, default="cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if (requested in ("cuda", "mps")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def resolve_emulator_roots(extra_args, *, adapter, exact_count=None):
    """Validate saved roots and refuse two aliases for one artifact pair."""
    roots = extra_args.get("emulators")
    if (not isinstance(roots, Sequence)
            or isinstance(roots, (str, bytes, bytearray))):
        raise ValueError(
            adapter + ": extra_args 'emulators' must be a sequence of "
            "saved-emulator path strings, got " + repr(roots))
    if not roots:
        raise ValueError(
            adapter + ": extra_args 'emulators' must contain at least one "
            "saved-emulator path root")
    if exact_count is not None and len(roots) != exact_count:
        raise ValueError(
            adapter + ": extra_args 'emulators' must contain exactly "
            + repr(exact_count) + " path roots, got " + repr(len(roots)))

    rootdir = os.environ.get("ROOTDIR")
    resolved = []
    shown_by_path = {}
    shown_by_members = {}
    for index, root in enumerate(roots):
        if type(root) is not str or not root.strip():
            raise ValueError(
                adapter + ": emulators[" + repr(index)
                + "] must be a nonempty path string, got " + repr(root))
        if os.path.isabs(root):
            path = root
        else:
            if (type(rootdir) is not str or not rootdir
                    or not os.path.isabs(rootdir)
                    or not os.path.isdir(rootdir)):
                raise ValueError(
                    adapter + ": relative emulator root " + repr(root)
                    + " requires ROOTDIR to name an existing absolute "
                    "CoCoA folder")
            path = os.path.join(rootdir, root)
        absolute = os.path.abspath(path)
        canonical = os.path.realpath(absolute)
        if canonical in shown_by_path:
            raise ValueError(
                adapter + ": emulator roots " + repr(shown_by_path[canonical])
                + " and " + repr(root) + " resolve to the same canonical "
                "path " + repr(canonical))
        members = tuple(
            os.path.realpath(absolute + extension)
            for extension in (".h5", ".emul"))
        if members in shown_by_members:
            raise ValueError(
                adapter + ": emulator roots "
                + repr(shown_by_members[members]) + " and " + repr(root)
                + " resolve to the same saved .h5/.emul artifact pair "
                + repr(members))
        shown_by_path[canonical] = root
        shown_by_members[members] = root
        resolved.append(canonical)
    return resolved


def name_sequence(value, *, adapter, option, allow_empty=True, label=None):
    """Return one checked list of unique, nonempty native string names."""
    subject = ("extra_args " + repr(option)) if label is None else label
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))):
        raise ValueError(
            adapter + ": " + subject
            + " must be a sequence of parameter-name strings, got "
            + repr(value))
    if not allow_empty and not value:
        raise ValueError(
            adapter + ": " + subject + " must not be empty")
    names = []
    seen = set()
    for index, name in enumerate(value):
        if type(name) is not str or not name.strip():
            raise ValueError(
                adapter + ": " + subject + "[" + repr(index)
                + "] must be a nonempty parameter-name string, got "
                + repr(name))
        if name in seen:
            raise ValueError(
                adapter + ": " + subject
                + " repeats parameter name " + repr(name))
        seen.add(name)
        names.append(name)
    return names


def fast_parameter_groups(value, *, adapter, emulator_count):
    """Validate one inner parameter-name list for each cosmic-shear root."""
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))
            or len(value) != emulator_count):
        raise ValueError(
            adapter + ": extra_args 'fast_params' must contain exactly one "
            "inner name list per emulator (" + repr(emulator_count)
            + " here), got " + repr(value))
    return [
        name_sequence(group, adapter=adapter,
                      option="fast_params[" + repr(index) + "]")
        for index, group in enumerate(value)
    ]

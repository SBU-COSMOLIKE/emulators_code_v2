#!/usr/bin/env python3
"""Classify candidate paths without reading Git or changing workflow state.

The mailbox daemon supplies paths already read from Git and the protected
path sets already read from the role contract.  This module answers one
question: may the candidate proceed normally, does it need an Architect scope
decision, or did it touch a globally protected path?
"""


def forbidden_paths(
        changed_paths, *, forbidden_files, control_plane_files,
        forbidden_prefixes, protected_control_plane):
    """Return candidate paths that the selected ticket class may not change.

    Ordinary tickets cannot change any control-plane file. A protected
    control-plane ticket may change the control files named by its directive,
    but it still cannot change other globally forbidden files or paths.
    """
    return {
        path for path in changed_paths
        if (path in forbidden_files
            or (path in control_plane_files and not protected_control_plane)
            or (any(path.startswith(prefix) for prefix in forbidden_prefixes)
                and not (protected_control_plane
                         and path in control_plane_files)))}


def classify(changed_paths, path_scope, protected_paths):
    """Return the candidate's path result and the paths that caused it.

    ``protected_paths`` always wins over the ticket's planned file list.
    Other undeclared paths produce ``SCOPE_EXCEEDED`` so the Architect can
    accept or reject a legitimate implementation discovery. An exact match
    produces ``IN_SCOPE``.
    """
    protected = set(protected_paths)
    if protected:
        return "PROTECTED_PATH_VIOLATION", protected
    exceeded = set(changed_paths) - set(path_scope or ())
    if exceeded:
        return "SCOPE_EXCEEDED", exceeded
    return "IN_SCOPE", set()

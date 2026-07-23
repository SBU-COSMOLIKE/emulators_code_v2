#!/usr/bin/env python3
"""Classify candidate paths without reading Git or changing workflow state.

A candidate is an Implementer's proposed change waiting for review, and
its candidate paths are the repository files that change in it. The
mailbox daemon (the watcher program that runs the development roles in
turn) supplies those paths already read from Git and the protected path
sets already read from the role contract, the settings file
``ai/notes/role-contract.yaml`` that records what each role may touch.
This module answers one question: may the candidate proceed normally,
does it need an Architect scope decision, or did it touch a globally
protected path? It holds no state of its own, so the answer depends only
on the arguments.
"""


def forbidden_paths(
        changed_paths, *, forbidden_files, control_plane_files,
        forbidden_prefixes, protected_control_plane):
    """Return candidate paths that the selected ticket class may not change.

    A ticket class is the declared kind of work a ticket performs; the
    class decides which protected files its candidate may touch. The
    control plane is the set of files that run the AI workflow itself
    (the daemon, the guards, and the contracts). A forbidden prefix wins
    over everything: even the protected-control-plane class, which is
    reserved for Architect-owned administration and never reaches an
    Implementer candidate, cannot open ``ai/tools/`` or another prefix
    that the role contract reserves for maintenance outside the mailbox
    workflow.

    Arguments:
      changed_paths           = iterable of repository-relative paths that
                                the candidate changes.
      forbidden_files         = exact paths that no ticket class may change.
      control_plane_files     = exact paths that only the
                                protected-control-plane class may change.
      forbidden_prefixes      = path prefixes (leading folder strings) that
                                are refused for every class.
      protected_control_plane = True when the selected class is
                                protected-control-plane, which unlocks the
                                ``control_plane_files`` set and nothing else.

    Returns:
      The set of changed paths the class may not touch; an empty set means
      every path is admissible for this class.
    """
    return {
        path for path in changed_paths
        if (path in forbidden_files
            or (path in control_plane_files and not protected_control_plane)
            or any(path.startswith(prefix) for prefix in forbidden_prefixes))}


def classify(changed_paths, path_scope, protected_paths):
    """Return the candidate's path verdict and the paths that caused it.

    ``protected_paths`` always wins over the ticket's planned file list:
    touching even one protected file refuses the whole candidate. A path
    that is not protected but also not planned produces ``SCOPE_EXCEEDED``
    instead of a refusal, so the Architect can accept or reject a
    legitimate implementation discovery (a file the work turned out to
    need that the plan did not name). When every changed path was planned,
    the verdict is ``IN_SCOPE``.

    Arguments:
      changed_paths   = iterable of repository-relative paths that the
                        candidate changes.
      path_scope      = the ticket's planned file list; ``None`` means no
                        list, so every changed path counts as undeclared.
      protected_paths = paths from the candidate that a prior check found
                        protected, usually from :func:`forbidden_paths`.

    Returns:
      A pair ``(verdict, paths)``. ``verdict`` is one of
      ``"PROTECTED_PATH_VIOLATION"``, ``"SCOPE_EXCEEDED"``, or
      ``"IN_SCOPE"``; ``paths`` is the set of paths that produced the
      verdict, empty for ``IN_SCOPE``.
    """
    protected = set(protected_paths)
    if protected:
        return "PROTECTED_PATH_VIOLATION", protected
    exceeded = set(changed_paths) - set(path_scope or ())
    if exceeded:
        return "SCOPE_EXCEEDED", exceeded
    return "IN_SCOPE", set()

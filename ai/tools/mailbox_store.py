"""Mailbox file claims, watch locks, and dispatch topology validation.

Only one watcher may claim a message, and two watchers must never
share one mailbox. This file owns the filesystem rules that keep that
true: the atomic claim that moves a message into the work-in-progress
folder without overwriting, the advisory file locks (locks that die
with their process) marking a live watch and its fix-only and
skip-redteam policies, and the checks that every directory a dispatch
is about to use is the exact saved, unredirected path the daemon
recorded.

This file is one part of the mailbox daemon and holds definitions only.
``mailbox_daemon.py`` loads it from its own directory, binds the name
``daemon`` below to a live view of its own namespace, and adopts every name
in PART_EXPORTS into that namespace. Code here reaches every constant,
standard-library module, and collaborator through ``daemon.<name>``, so the
daemon keeps one shared namespace no matter how many files store its source.
"""


# Bound by mailbox_daemon.py to a live view of the daemon namespace
# before any of these definitions can run.
daemon = None

PART_EXPORTS = (
    "move_without_overwrite",
    "claim_message",
    "mailbox_path_is_unredirected",
    "held_lock_probe",
    "held_lock_owner",
    "dispatch_lock_is_live_watch",
    "fix_only_watch_is_active",
    "skip_redteam_watch_is_active",
    "skip_redteam_policy_active",
    "mailbox_candidates",
    "warn_if_mailbox_unwatched",
    "live_action_topology_is_current",
    "acquire_dispatch_lock",
    "release_dispatch_lock",
    "acquire_fix_only_lock_while_sequence_locked",
    "acquire_fix_only_lock",
    "release_fix_only_lock",
    "acquire_skip_redteam_lock_while_sequence_locked",
    "acquire_skip_redteam_lock",
    "release_skip_redteam_lock",
    "main_checkout_turn_lock_path",
    "acquire_main_checkout_turn_lock",
    "release_main_checkout_turn_lock",
    "validate_live_agent_dispatch_topology",
    "recheck_agent_dispatch_directories",
    "revalidate_agent_dispatch_topology",
    "revalidate_protected_policy_admin_topology",
    "_architect_ordinary_tracked_state",
    "_ordinary_untracked_worktree_state",
    "_git_path_bytes",
    "_top_level_tracked_markdown",
    "_require_exact_permanent_note_set",
    "_validate_protected_tracked_state",
    "_validate_sealed_backlog",
    "require_closed_backlog_ticket",
    "_bridge_local_sealed_backlog",
    "_validate_current_protected_primary_state",
    "_capture_shared_protected_state",
    "_recheck_shared_protected_state",
    "capture_persistent_role_state",
    "implementer_checkpoint_delivered",
    "recheck_persistent_role_state",
    "validate_live_sol_dispatch_topology",
    "recheck_sol_dispatch_directories",
    "revalidate_sol_dispatch_topology",
)


def move_without_overwrite(path, directory):
    """Move a message into a state directory without replacing history.

    Arguments:
      path      = the current message path.
      directory = the destination directory.

    Returns:
      The destination path, or None when that name is already present or the
      source was claimed first.
    """
    daemon.os.makedirs(directory, exist_ok=True)
    destination = daemon.os.path.join(directory, daemon.os.path.basename(path))
    try:
        daemon.os.link(path, destination)
    except FileExistsError:
        print("  !! refusing to overwrite existing message state: "
              + destination)
        return None
    except FileNotFoundError:
        return None
    daemon.fsync_directory(directory=directory)
    daemon.os.unlink(path)
    return destination


def claim_message(path):
    """Atomically remove a message from the pending queue before dispatch.

    A claimed message remains in ``inflight/`` if the daemon is interrupted.
    That ambiguous state requires a human decision and is never dispatched a
    second time automatically.

    Arguments:
      path = the pending mailbox path.

    Returns:
      The inflight path, or None when another process claimed it first.
    """
    claimed = daemon.move_without_overwrite(
        path=path,
        directory=daemon.os.path.join(daemon.MAILBOX, "inflight"))
    if claimed is None:
        print("  note: " + daemon.os.path.basename(path)
              + " was already claimed; skipping duplicate dispatch.")
    return claimed


def mailbox_path_is_unredirected(mailbox):
    """Return whether ``mailbox`` stays inside its lexical repository path.

    ``O_NOFOLLOW`` protects the final lock file, but would still follow a
    symlink used as an earlier ``notes`` or ``mailbox`` component.  Compare
    real paths relative to the repository's own real path so symlinks *above*
    the checkout remain harmless while redirects *inside* it are rejected.

    Arguments:
      mailbox = the mailbox directory path to test.

    Returns:
      True when every path component inside the repository is a real
      directory rather than a symlink redirect; False otherwise, and
      False for any path-resolution error.
    """
    repository = daemon.os.path.abspath(daemon.REPO_ROOT)
    candidate = daemon.os.path.abspath(mailbox)
    try:
        if daemon.os.path.commonpath([repository, candidate]) != repository:
            return False
        relative = daemon.os.path.relpath(candidate, repository)
    except (OSError, ValueError):
        return False
    expected = daemon.os.path.normpath(daemon.os.path.join(
        daemon.os.path.realpath(repository), relative))
    return daemon.os.path.realpath(candidate) == expected


def held_lock_probe(mailbox, lock_name):
    """Probe a regular exact-path lock and its bounded owner metadata.

    The probe is deliberately read-only.  Opening a missing lock must never
    create it because both ``--send --dry-run`` and a refused discovery promise
    zero filesystem mutation.  A shared nonblocking probe coexists with other
    diagnostics but is refused by the exclusive lock held by the real owner.

    Arguments:
      mailbox   = the mailbox directory holding the lock.
      lock_name = the lock's filename inside that directory.

    Returns:
      ``(held, owner)``. ``held`` is true only when the exact regular inode is
      actively locked. ``owner`` is its bounded ASCII text, or ``None`` when
      held metadata is malformed. Symlinks, redirected parents, stale files,
      replacements, and devices never count as held.
    """
    lock_path = daemon.os.path.join(mailbox, lock_name)
    descriptor = None
    probe_acquired = False
    try:
        if not daemon.mailbox_path_is_unredirected(mailbox=mailbox):
            return False, None
        before = daemon.os.lstat(lock_path)
        if not daemon.stat.S_ISREG(before.st_mode):
            return False, None
        flags = daemon.os.O_RDONLY | daemon.os.O_NONBLOCK
        flags = flags | getattr(daemon.os, "O_CLOEXEC", 0)
        flags = flags | getattr(daemon.os, "O_NOFOLLOW", 0)
        descriptor = daemon.os.open(lock_path, flags)
        opened = daemon.os.fstat(descriptor)
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (before.st_dev, before.st_ino)):
            return False, None
        try:
            # A watch/once loop owns an exclusive flock.  SH is intentional:
            # simultaneous send diagnostics can all acquire it, so they can
            # never mistake one another for a live watcher.
            daemon.fcntl.flock(descriptor, daemon.fcntl.LOCK_SH | daemon.fcntl.LOCK_NB)
            probe_acquired = True
            return False, None
        except BlockingIOError:
            pass
        # The path may have been replaced after open().  A lock on an
        # unlinked/orphaned inode does not protect the filename a future watch
        # would use, so it cannot suppress the warning.
        current = daemon.os.lstat(lock_path)
        if (not daemon.stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino)
                != (opened.st_dev, opened.st_ino)):
            return False, None
        # Bound the read so a corrupt/sparse lock cannot consume unbounded
        # memory.  os.pread leaves the descriptor offset untouched.
        owner_bytes = daemon.os.pread(descriptor, 129, 0)
        if len(owner_bytes) > 128:
            return True, None
        try:
            owner = owner_bytes.decode("ascii")
        except UnicodeError:
            return True, None
        return True, owner
    except OSError:
        return False, None
    finally:
        if descriptor is not None:
            if probe_acquired:
                try:
                    daemon.fcntl.flock(descriptor, daemon.fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                daemon.os.close(descriptor)
            except OSError:
                pass


def held_lock_owner(mailbox, lock_name):
    """Return valid owner text for an actively held exact-path lock.

    Arguments:
      mailbox   = the mailbox folder holding the lock.
      lock_name = the lock's filename.

    Returns:
      The owner text, or ``None`` when the lock is not held or its
      metadata is unusable.
    """
    held, owner = daemon.held_lock_probe(mailbox=mailbox, lock_name=lock_name)
    if not held:
        return None
    return owner


def dispatch_lock_is_live_watch(mailbox):
    """Return whether ``mailbox`` has an exact held ``watch pid N`` lock.

    Arguments:
      mailbox = the mailbox folder to probe.

    Returns:
      True only when the dispatch lock is actively held and its owner
      text names a watch loop; a once loop or malformed owner reads
      False.
    """
    owner = daemon.held_lock_owner(mailbox=mailbox, lock_name=".dispatch.lock")
    if owner is None:
        return False
    return daemon.WATCH_LOCK_OWNER_RE.fullmatch(owner) is not None


def fix_only_watch_is_active(mailbox=None):
    """Return whether this mailbox's reserved mode lock is actively held.

    Owner text is diagnostic, not authority: once the exact-path regular lock
    is held, malformed or concurrently damaged metadata must fail closed as
    fix-only.  Unlocked stale files still read inactive.

    Arguments:
      mailbox = the mailbox to probe, or ``None`` for this daemon's
                own mailbox.

    Returns:
      True when the fix-only mode lock is actively held.
    """
    if mailbox is None:
        mailbox = daemon.MAILBOX
    held, _ = daemon.held_lock_probe(
        mailbox=mailbox, lock_name=daemon.FIX_ONLY_LOCK_NAME)
    return held


def skip_redteam_watch_is_active(mailbox=None):
    """Return whether this mailbox has a live two-role watch marker.

    Arguments:
      mailbox = the mailbox to probe, or ``None`` for this daemon's
                own mailbox.

    Returns:
      True when the two-role mode lock is actively held.
    """
    if mailbox is None:
        mailbox = daemon.MAILBOX
    held, _ = daemon.held_lock_probe(
        mailbox=mailbox, lock_name=daemon.SKIP_REDTEAM_LOCK_NAME)
    return held


def skip_redteam_policy_active():
    """Return whether this process or its mailbox is in two-role mode.

    Either signal is binding: the environment setting inherited from
    a two-role watch, or the live mode lock on the mailbox.

    Returns:
      True when either signal is present; a Red Team send must then
      be refused.
    """
    return (daemon.skip_redteam_environment_active()
            or daemon.skip_redteam_watch_is_active())


def mailbox_candidates():
    """Return every mailbox whose watcher could serve this repository.

    The current mailbox and the main checkout are always included.  Worktree
    discovery uses scandir instead of ``glob('*')`` so a legal hidden
    worktree name is not silently missed.  Paths are absolute, de-duplicated,
    and sorted to keep warning output deterministic.

    Returns:
      The sorted list of candidate mailbox directory paths.
    """
    candidates = {
        daemon.os.path.abspath(daemon.MAILBOX),
        daemon.os.path.abspath(daemon.os.path.join(
            daemon.REPO_ROOT, "ai", "notes", "mailbox")),
    }
    worktrees = daemon.os.path.join(daemon.REPO_ROOT, ".claude", "worktrees")
    try:
        if not daemon.mailbox_path_is_unredirected(mailbox=worktrees):
            return sorted(candidates)
        worktrees_state = daemon.os.lstat(worktrees)
        if not daemon.stat.S_ISDIR(worktrees_state.st_mode):
            return sorted(candidates)
        with daemon.os.scandir(worktrees) as entries:
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                candidates.add(daemon.os.path.abspath(daemon.os.path.join(
                    entry.path, "ai", "notes", "mailbox")))
    except OSError:
        pass
    return sorted(candidates)


def warn_if_mailbox_unwatched():
    """Warn when a send targets a mailbox with no live watch loop.

    The warning is advisory: callers continue to publish (or rehearse) the
    message.  Other watched mailboxes are reported as recovery clues, not as
    alternative destinations; the daemon never silently reroutes a send.
    """
    own_mailbox = daemon.os.path.abspath(daemon.MAILBOX)
    if daemon.dispatch_lock_is_live_watch(mailbox=own_mailbox):
        return
    print("  !! warning: no active watch is polling this mailbox: "
          + own_mailbox)
    for candidate in daemon.mailbox_candidates():
        if candidate == own_mailbox:
            continue
        if daemon.dispatch_lock_is_live_watch(mailbox=candidate):
            print("  !! warning: another mailbox under this repository has "
                  "a live watch: " + candidate)


def live_action_topology_is_current(agent, action):
    """Refuse an action whose saved worktrees were removed by cleanup.

    Arguments:
      agent  = the role whose topology the action needs.
      action = short action name for the refusal message.

    Returns:
      True when the saved topology still validates, or no topology is
      active; False after printing the refusal.
    """
    if daemon.ACTIVE_TOPOLOGY is None:
        return True
    try:
        daemon.validate_live_agent_dispatch_topology(agent=agent)
    except (OSError, daemon.PrimaryWorktreeError) as exc:
        print(action + " refused: saved worktree topology changed ("
              + str(exc) + ").")
        return False
    return True


def acquire_dispatch_lock(mode="unknown"):
    """Acquire the process-wide dispatch-loop lock without a PID race.

    Arguments:
      mode = ``watch`` or ``once`` for command-line loops.  The default keeps
             older direct callers compatible but is deliberately not treated
             as proof of an active watcher by send diagnostics.

    Returns:
      An open locked file, or None when another loop owns the lock.
    """
    if mode not in ("watch", "once"):
        mode = "unknown"
    if not daemon.live_action_topology_is_current("fable", "dispatch"):
        return None
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    lock_path = daemon.os.path.join(daemon.MAILBOX, ".dispatch.lock")
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        daemon.fcntl.flock(
            lock_file.fileno(),
            daemon.fcntl.LOCK_EX | daemon.fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        owner = lock_file.read().strip()
        lock_file.close()
        print("another dispatch loop is already running ("
              + (owner or "owner unknown") + "); refusing to overlap it.")
        return None
    if not daemon.live_action_topology_is_current("fable", "dispatch"):
        daemon.release_dispatch_lock(lock_file=lock_file)
        return None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(mode + " pid " + str(daemon.os.getpid()))
    lock_file.flush()
    return lock_file


def release_dispatch_lock(lock_file):
    """Release a lock returned by acquire_dispatch_lock().

    Arguments:
      lock_file = the open locked file.

    Returns:
      None.
    """
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def acquire_fix_only_lock_while_sequence_locked():
    """Create the mode marker after the caller serializes publishers.

    The lock file is created and locked only after proving the
    mailbox path is unredirected, and the path is rechecked after
    every step — locking, then publishing the owner — so a swapped
    path can never leave a lock on an orphaned file while the public
    name points elsewhere.

    Returns:
      The open locked mode file, or ``None`` after printing why
      activation failed.
    """
    if not daemon.mailbox_path_is_unredirected(mailbox=daemon.MAILBOX):
        print("cannot activate fix-only mode on a redirected mailbox path")
        return None
    lock_path = daemon.os.path.join(daemon.MAILBOX, daemon.FIX_ONLY_LOCK_NAME)
    flags = daemon.os.O_RDWR | daemon.os.O_CREAT | daemon.os.O_NONBLOCK
    flags = flags | getattr(daemon.os, "O_CLOEXEC", 0)
    flags = flags | getattr(daemon.os, "O_NOFOLLOW", 0)
    try:
        descriptor = daemon.os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("cannot activate fix-only mode: " + str(exc))
        return None
    lock_file = daemon.os.fdopen(descriptor, "r+", encoding="utf-8")

    def path_still_names_opened_inode(opened):
        """Return whether the public mode path still names this descriptor.

        Arguments:
          opened = the fstat result of the descriptor this process
                   opened.

        Returns:
          True when the mode-lock path still points at that exact
          regular file; False when it vanished or was replaced.
        """
        try:
            current = daemon.os.lstat(lock_path)
        except OSError:
            return False
        return (daemon.stat.S_ISREG(current.st_mode)
                and (opened.st_dev, opened.st_ino)
                == (current.st_dev, current.st_ino))

    try:
        opened = daemon.os.fstat(lock_file.fileno())
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or not path_still_names_opened_inode(opened=opened)):
            print("cannot activate fix-only mode: mode lock is not an "
                  "unchanged regular file")
            lock_file.close()
            return None
        daemon.fcntl.flock(
            lock_file.fileno(),
            daemon.fcntl.LOCK_EX | daemon.fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        print("cannot activate fix-only mode: its mode lock is already held "
              "or unreadable (" + str(exc) + ")")
        lock_file.close()
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot activate fix-only mode: mode lock path changed while "
              "its lock was acquired")
        daemon.release_fix_only_lock(lock_file=lock_file)
        return None
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write("fix-only watch pid " + str(daemon.os.getpid()))
        lock_file.flush()
        daemon.os.fsync(lock_file.fileno())
    except OSError as exc:
        print("cannot activate fix-only mode: could not publish its owner ("
              + str(exc) + ")")
        daemon.release_fix_only_lock(lock_file=lock_file)
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot activate fix-only mode: mode lock path changed while "
              "its owner was published")
        daemon.release_fix_only_lock(lock_file=lock_file)
        return None
    return lock_file


def acquire_fix_only_lock():
    """Atomically activate fix-only mode relative to message publication.

    Sol senders perform their final policy check while holding the same
    sequence lock.  Therefore a concurrent sender either publishes wholly
    before activation or observes the held mode marker and refuses; it cannot
    publish after the watch has become fix-only.

    Returns:
      The open locked mode file, or ``None`` after printing the
      failure.
    """
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    sequence_path = daemon.os.path.join(daemon.MAILBOX, ".sequence.lock")
    try:
        with open(sequence_path, "a+", encoding="utf-8") as sequence_file:
            daemon.fcntl.flock(sequence_file.fileno(), daemon.fcntl.LOCK_EX)
            try:
                return daemon.acquire_fix_only_lock_while_sequence_locked()
            finally:
                daemon.fcntl.flock(sequence_file.fileno(), daemon.fcntl.LOCK_UN)
    except OSError as exc:
        print("cannot activate fix-only mode: sequence lock failed ("
              + str(exc) + ")")
        return None


def release_fix_only_lock(lock_file):
    """Release a lock returned by ``acquire_fix_only_lock``.

    Arguments:
      lock_file = the open locked mode file from the acquire call; it
                  is unlocked and closed.
    """
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def acquire_skip_redteam_lock_while_sequence_locked():
    """Create the two-role mode marker after publishers are serialized.

    Mirrors the fix-only activation: the mailbox path must be
    unredirected, and it is rechecked after locking and again after
    publishing the owner text.

    Returns:
      The open locked mode file, or ``None`` after printing why
      activation failed.
    """
    if not daemon.mailbox_path_is_unredirected(mailbox=daemon.MAILBOX):
        print("cannot disable the red-team route on a redirected mailbox "
              "path")
        return None
    lock_path = daemon.os.path.join(daemon.MAILBOX, daemon.SKIP_REDTEAM_LOCK_NAME)
    flags = daemon.os.O_RDWR | daemon.os.O_CREAT | daemon.os.O_NONBLOCK
    flags = flags | getattr(daemon.os, "O_CLOEXEC", 0)
    flags = flags | getattr(daemon.os, "O_NOFOLLOW", 0)
    try:
        descriptor = daemon.os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("cannot disable the red-team route: " + str(exc))
        return None
    lock_file = daemon.os.fdopen(descriptor, "r+", encoding="utf-8")

    def path_still_names_opened_inode(opened):
        """Return whether the public mode path still names this descriptor.

        Arguments:
          opened = the fstat result of the descriptor this process
                   opened.

        Returns:
          True when the mode-lock path still points at that exact
          regular file; False when it vanished or was replaced.
        """
        try:
            current = daemon.os.lstat(lock_path)
        except OSError:
            return False
        return (daemon.stat.S_ISREG(current.st_mode)
                and (opened.st_dev, opened.st_ino)
                == (current.st_dev, current.st_ino))

    try:
        opened = daemon.os.fstat(lock_file.fileno())
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or not path_still_names_opened_inode(opened=opened)):
            print("cannot disable the red-team route: mode lock is not an "
                  "unchanged regular file")
            lock_file.close()
            return None
        daemon.fcntl.flock(
            lock_file.fileno(),
            daemon.fcntl.LOCK_EX | daemon.fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        print("cannot disable the red-team route: its mode lock is already "
              "held or unreadable (" + str(exc) + ")")
        lock_file.close()
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot disable the red-team route: mode lock path changed "
              "while its lock was acquired")
        daemon.release_skip_redteam_lock(lock_file=lock_file)
        return None
    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write("two-role watch pid " + str(daemon.os.getpid()))
        lock_file.flush()
        daemon.os.fsync(lock_file.fileno())
    except OSError as exc:
        print("cannot disable the red-team route: could not publish its "
              "owner (" + str(exc) + ")")
        daemon.release_skip_redteam_lock(lock_file=lock_file)
        return None
    if not path_still_names_opened_inode(opened=opened):
        print("cannot disable the red-team route: mode lock path changed "
              "while its owner was published")
        daemon.release_skip_redteam_lock(lock_file=lock_file)
        return None
    return lock_file


def acquire_skip_redteam_lock():
    """Atomically disable Sol dispatch relative to daemon message sends.

    Returns:
      The open locked mode file, or ``None`` after printing the
      failure.
    """
    # Refuse a redirected mailbox before creating even its sequence-lock
    # file. The inner check stays binding because the path can still change
    # between this preflight and publication of the mode marker.
    if not daemon.mailbox_path_is_unredirected(mailbox=daemon.MAILBOX):
        print("cannot disable the red-team route on a redirected mailbox "
              "path")
        return None
    daemon.os.makedirs(daemon.MAILBOX, exist_ok=True)
    sequence_path = daemon.os.path.join(daemon.MAILBOX, ".sequence.lock")
    try:
        with open(sequence_path, "a+", encoding="utf-8") as sequence_file:
            daemon.fcntl.flock(sequence_file.fileno(), daemon.fcntl.LOCK_EX)
            try:
                return daemon.acquire_skip_redteam_lock_while_sequence_locked()
            finally:
                daemon.fcntl.flock(sequence_file.fileno(), daemon.fcntl.LOCK_UN)
    except OSError as exc:
        print("cannot disable the red-team route: sequence lock failed ("
              + str(exc) + ")")
        return None


def release_skip_redteam_lock(lock_file):
    """Release a lock returned by ``acquire_skip_redteam_lock``.

    Arguments:
      lock_file = the open locked mode file from the acquire call; it
                  is unlocked and closed.
    """
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def main_checkout_turn_lock_path():
    """Return the one ignored lock shared by every repository worktree.

    The lock lives under ``.claude/worktrees`` in the main checkout —
    a Git-ignored location every linked worktree can reach — so one
    lock serializes all of them.

    Returns:
      The lock path; the folders on the way are created as plain
      directories.

    Raises:
      daemon.PrimaryWorktreeError: when a component is not a plain
        directory.
    """
    repository = daemon.os.path.abspath(daemon.REPO_ROOT)
    daemon._plain_directory(path=repository, label="repository root")
    claude_root = daemon.os.path.join(repository, ".claude")
    daemon._plain_directory(path=claude_root, label=".claude directory", create=True)
    managed_root = daemon.os.path.join(claude_root, "worktrees")
    daemon._plain_directory(
        path=managed_root, label="managed worktree directory", create=True)
    return daemon.os.path.join(managed_root, daemon.MAIN_CHECKOUT_TURN_LOCK_NAME)


def acquire_main_checkout_turn_lock():
    """Serialize Architect decisions that the parent daemon may land.

    Returns:
      The open locked file, or ``None`` after printing why the main
      checkout cannot be serialized. The wait is blocking: the caller
      queues behind the current holder rather than failing.
    """
    try:
        lock_path = daemon.main_checkout_turn_lock_path()
    except (OSError, daemon.PrimaryWorktreeError) as exc:
        print("cannot serialize the main checkout: " + str(exc))
        return None
    flags = daemon.os.O_RDWR | daemon.os.O_CREAT | daemon.os.O_NONBLOCK
    flags |= getattr(daemon.os, "O_CLOEXEC", 0)
    flags |= getattr(daemon.os, "O_NOFOLLOW", 0)
    try:
        descriptor = daemon.os.open(lock_path, flags, 0o600)
    except OSError as exc:
        print("cannot serialize the main checkout: " + str(exc))
        return None
    lock_file = daemon.os.fdopen(descriptor, "r+", encoding="utf-8")
    try:
        opened = daemon.os.fstat(lock_file.fileno())
        if not daemon.stat.S_ISREG(opened.st_mode):
            raise OSError("main-checkout turn lock is not a regular file")
        daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_EX)
        current = daemon.os.lstat(lock_path)
        if (not daemon.stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino)
                != (current.st_dev, current.st_ino)):
            raise OSError("main-checkout turn lock path changed")
    except OSError as exc:
        print("cannot serialize the main checkout: " + str(exc))
        try:
            daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()
        return None
    return lock_file


def release_main_checkout_turn_lock(lock_file):
    """Release an Architect main-checkout turn lock.

    Arguments:
      lock_file = the open locked file from the acquire call; it is
                  unlocked and closed.
    """
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def validate_live_agent_dispatch_topology(agent):
    """Re-prove one mutable agent's saved checkout before launch.

    Topology is the saved map of who works where: the Architect's
    primary worktree, the Implementer's worktree, the Sol worktree,
    and the shared notes directory. Under the primary lock, every
    saved state file is reloaded and revalidated, the live paths must
    equal the saved ones, and Sol's command must still carry its
    exact working-folder and notes options.

    Arguments:
      agent = ``"fable"``, ``"opus"``, or ``"sol"``.

    Returns:
      A proof mapping with the role path and its directory identity,
      the notes path and identity, and the authoritative role files —
      the facts a later recheck compares.

    Raises:
      ValueError: for an unknown agent.
      daemon.PrimaryWorktreeError: when no topology is active or any
        saved binding no longer holds.
    """
    if agent not in {"fable", "opus", "sol"}:
        raise ValueError(
            "topology proof is defined only for Fable, Opus, and Sol")
    if daemon.ACTIVE_TOPOLOGY is None:
        raise daemon.PrimaryWorktreeError(
            "live " + agent + " dispatch has no validated topology")
    lock_file = daemon._open_primary_lock(repository_root=daemon.REPO_ROOT)
    try:
        primary = daemon.load_primary_state(path=daemon.ACTIVE_TOPOLOGY["primary_state"])
        if primary["schema"] != daemon.PRIMARY_STATE_SCHEMA:
            raise daemon.PrimaryWorktreeError(
                "live dispatch requires topology-aware primary state")
        primary = daemon.validate_primary_state(
            state=primary, repository_root=daemon.REPO_ROOT, allow_move=False)
        implementer = daemon.load_primary_state(
            path=daemon.ACTIVE_TOPOLOGY["implementer_state"])
        implementer = daemon.validate_implementer_state(
            state=implementer, repository_root=daemon.REPO_ROOT,
            primary_state=primary, allow_move=False)
        sol = daemon.load_primary_state(path=daemon.ACTIVE_TOPOLOGY["sol_state"])
        sol = daemon.validate_sol_state(
            state=sol, repository_root=daemon.REPO_ROOT, primary_state=primary,
            allow_move=False)
        notes = daemon.validated_primary_notes(primary_path=primary["path"])
        authoritative_files = daemon.validate_authoritative_role_files(
            primary_path=primary["path"])
        daemon.validate_distinct_agent_states(
            primary_state=primary, implementer_state=implementer,
            sol_state=sol)
        expected = daemon.ACTIVE_TOPOLOGY
        if (daemon.os.path.abspath(primary["path"]) != expected["primary_path"]
                or primary["branch"] != expected["primary_branch"]
                or daemon.os.path.abspath(implementer["path"])
                != expected["implementer_path"]
                or daemon.os.path.abspath(sol["path"]) != expected["sol_path"]
                or notes != expected["shared_notes"]
                or daemon.AGENT_CWD["fable"] != expected["primary_path"]
                or daemon.AGENT_CWD["opus"] != expected["implementer_path"]
                or daemon.AGENT_CWD["sol"] != expected["sol_path"]
                or daemon.os.path.realpath(daemon.os.path.join(daemon.AI_ROOT, "notes"))
                != expected["shared_notes"]):
            raise daemon.PrimaryWorktreeError(
                "saved agent topology changed after this process started")
        if agent == "sol":
            command = daemon.AGENT_COMMANDS["sol"]
            if command.count("--cd") != 1 or command.count("--add-dir") != 1:
                raise daemon.PrimaryWorktreeError(
                    "Sol command must carry one exact cwd and notes grant")
            cd_index = command.index("--cd")
            add_index = command.index("--add-dir")
            if cd_index + 1 >= len(command) or add_index + 1 >= len(command):
                raise daemon.PrimaryWorktreeError(
                    "Sol command is missing its cwd or notes value")
            if (daemon.os.path.abspath(command[cd_index + 1])
                    != expected["sol_path"]
                    or daemon.os.path.realpath(command[add_index + 1])
                    != expected["shared_notes"]):
                raise daemon.PrimaryWorktreeError(
                    "Sol command no longer matches saved worktree state")
        if agent == "fable":
            role_path = expected["primary_path"]
            role_label = "saved Architect primary worktree"
        elif agent == "opus":
            role_path = expected["implementer_path"]
            role_label = "saved Implementer worktree"
        else:
            role_path = expected["sol_path"]
            role_label = "saved Sol worktree"
        role_identity = daemon._plain_directory(
            path=role_path, label=role_label)
        notes_identity = daemon._plain_directory(
            path=expected["shared_notes"], label="shared notes directory")
        proof = {
            "agent": agent,
            "role_path": role_path,
            "role_identity": role_identity,
            "notes_path": expected["shared_notes"],
            "notes_identity": notes_identity,
            "authoritative_files": authoritative_files,
        }
        if agent == "sol":
            proof["sol_path"] = role_path
            proof["sol_identity"] = role_identity
        return proof
    finally:
        daemon._release_primary_lock(lock_file=lock_file)


def recheck_agent_dispatch_directories(proof, mutable_paths=()):
    """Prove launch pathnames still name the pre-claim directories.

    Arguments:
      proof         = the mapping from
                      validate_live_agent_dispatch_topology.
      mutable_paths = role files that may legitimately differ during
                      this recheck.

    Raises:
      daemon.PrimaryWorktreeError: for a missing proof or a changed
        directory or role file.
    """
    if proof is None:
        raise daemon.PrimaryWorktreeError(
            "live dispatch is missing its topology proof")
    daemon._require_directory_identity(
        path=proof["role_path"], identity=proof["role_identity"],
        label="saved " + proof["agent"] + " worktree")
    daemon._require_directory_identity(
        path=proof["notes_path"], identity=proof["notes_identity"],
        label="shared notes directory")
    daemon.recheck_authoritative_role_files(
        proof=proof["authoritative_files"], mutable_paths=mutable_paths)


def revalidate_agent_dispatch_topology(proof):
    """Re-prove all Git and command bindings without accepting a new inode.

    Arguments:
      proof = the pre-claim topology proof.

    Returns:
      The fresh proof, which must equal the old one exactly.

    Raises:
      daemon.PrimaryWorktreeError: when anything moved between claim
        and launch.
    """
    daemon.recheck_agent_dispatch_directories(proof=proof)
    current = daemon.validate_live_agent_dispatch_topology(agent=proof["agent"])
    if current != proof:
        raise daemon.PrimaryWorktreeError(
            "saved agent worktree topology changed after message claim")
    daemon.recheck_agent_dispatch_directories(proof=current)
    return current


def revalidate_protected_policy_admin_topology(proof):
    """Allow only Architect-owned policy files to change in an admin turn.

    Arguments:
      proof = the pre-claim topology proof.

    Returns:
      The fresh proof; the Architect role files and the role contract
      are the only files allowed to differ.

    Raises:
      daemon.PrimaryWorktreeError: for any other change.
    """
    if not isinstance(proof, dict):
        return daemon.revalidate_agent_dispatch_topology(proof=proof)
    mutable = daemon.ARCHITECT_ROLE_PATHS + (daemon.ROLE_CONTRACT_RELATIVE_PATH,)
    daemon.recheck_agent_dispatch_directories(proof=proof, mutable_paths=mutable)
    current = daemon.validate_live_agent_dispatch_topology(agent=proof["agent"])
    daemon.recheck_agent_dispatch_directories(
        proof=current, mutable_paths=mutable)
    return current


def _architect_ordinary_tracked_state(worktree, base_commit, cached):
    """Return exact non-note tracked state relative to one frozen base.

    Arguments:
      worktree    = the Architect worktree to capture.
      base_commit = the frozen base to diff against.
      cached      = True for the staged (index) state, False for the
                    working files.

    Returns:
      Raw diff bytes excluding the permanent notes and the backlog,
      which have their own guards.

    Raises:
      daemon.PrimaryWorktreeError: when Git cannot produce the state.
    """
    arguments = [
        "diff", "--no-ext-diff", "--no-renames", "--binary",
        "--full-index", "--ignore-submodules=none"]
    if cached:
        arguments.append("--cached")
    arguments.extend([base_commit, "--", "."])
    arguments.extend(
        ":(top,exclude)" + path
        for path in daemon.ARCHITECT_PERMANENT_NOTE_PATHS
        + (daemon.BACKLOG_RELATIVE_PATH,))
    try:
        result = daemon._run_git(
            repository_root=worktree, arguments=arguments, check=False)
    except daemon.PrimaryWorktreeError:
        raise
    if result.returncode != 0:
        raise daemon.PrimaryWorktreeError(
            "cannot capture Architect ordinary tracked state")
    return result.stdout


def _ordinary_untracked_worktree_state(worktree):
    """Return every nonignored untracked path in one persistent worktree.

    Mailbox transport, relay logs, and temporary note evidence are ignored by
    Git and therefore do not appear in this proof. The tracked backlog has its
    own Architect seal. A newly created source, test, README, or tool appears.

    Arguments:
      worktree = the worktree to list.

    Returns:
      Raw NUL-separated path bytes.

    Raises:
      daemon.PrimaryWorktreeError: when Git cannot list the paths.
    """
    try:
        result = daemon._run_git(
            repository_root=worktree,
            arguments=["ls-files", "--others", "--exclude-standard", "-z",
                       "--", "."],
            check=False)
    except daemon.PrimaryWorktreeError:
        raise
    if result.returncode != 0:
        raise daemon.PrimaryWorktreeError(
            "cannot capture nonignored untracked worktree paths")
    return result.stdout


def _git_path_bytes(worktree, object_name, relative_path, maximum_bytes):
    """Read one bounded tracked blob from an exact commit or the index.

    Arguments:
      worktree      = the worktree whose repository is read.
      object_name   = commit name, or the empty string for the index.
      relative_path = the tracked path to read.
      maximum_bytes = size bound for the protected file.

    Returns:
      The file bytes.

    Raises:
      daemon.PrimaryWorktreeError: for a missing or oversized file.
    """
    result = daemon._run_git(
        repository_root=worktree,
        arguments=["show", object_name + ":" + relative_path],
        check=False)
    if result.returncode != 0:
        raise daemon.PrimaryWorktreeError(
            relative_path + " is missing from " + object_name)
    if len(result.stdout) > maximum_bytes:
        raise daemon.PrimaryWorktreeError(
            relative_path + " exceeds its protected size limit")
    return result.stdout


def _top_level_tracked_markdown(raw_paths, label):
    """Decode one NUL path list and select top-level ai/notes Markdown.

    Arguments:
      raw_paths = NUL-separated path bytes from Git.
      label     = source name for error messages.

    Returns:
      The set of paths directly under ``ai/notes`` ending in ``.md``.

    Raises:
      daemon.PrimaryWorktreeError: for a non-UTF-8 path.
    """
    selected = set()
    try:
        values = [raw.decode("utf-8", errors="strict")
                  for raw in raw_paths.split(b"\0") if raw]
    except UnicodeDecodeError as exc:
        raise daemon.PrimaryWorktreeError(
            label + " contains a non-UTF-8 tracked path") from exc
    for relative_path in values:
        parent, separator, name = relative_path.rpartition("/")
        if (separator and parent == "ai/notes"
                and name.casefold().endswith(".md")):
            selected.add(relative_path)
    return selected


def _require_exact_permanent_note_set(primary_worktree, head):
    """Require exactly eleven tracked top-level notes in HEAD and the index.

    The backlog is discarded from both sets first: it is tracked in
    the same folder but sealed by its own guard rather than counted
    as a permanent note.

    Arguments:
      primary_worktree = the Architect worktree.
      head             = the commit to compare against the index.

    Raises:
      daemon.PrimaryWorktreeError: when either set differs from the
        eleven permanent notes.
    """
    expected = set(daemon.ARCHITECT_PERMANENT_NOTE_PATHS)
    head_result = daemon._run_git(
        repository_root=primary_worktree,
        arguments=["ls-tree", "-r", "--name-only", "-z", head,
                   "--", "ai/notes"])
    index_result = daemon._run_git(
        repository_root=primary_worktree,
        arguments=["ls-files", "-z", "--", "ai/notes"])
    head_notes = daemon._top_level_tracked_markdown(
        raw_paths=head_result.stdout, label="Architect HEAD")
    index_notes = daemon._top_level_tracked_markdown(
        raw_paths=index_result.stdout, label="Architect index")
    head_notes.discard(daemon.BACKLOG_RELATIVE_PATH)
    index_notes.discard(daemon.BACKLOG_RELATIVE_PATH)
    if head_notes != expected or index_notes != expected:
        raise daemon.PrimaryWorktreeError(
            "Architect HEAD and index must contain exactly the eleven "
            "permanent top-level Markdown notes")


def _validate_protected_tracked_state(primary_worktree):
    """Require protected policy files and their guard to match primary HEAD.

    For every protected tracked path, the committed bytes, the staged
    bytes, and the working file must be identical, and HEAD must not
    move while the check runs.

    Arguments:
      primary_worktree = the Architect worktree.

    Raises:
      daemon.PrimaryWorktreeError: for any mismatch or a moving HEAD.
    """
    try:
        head = daemon.worktree_head(worktree=primary_worktree)
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot identify the Architect commit protecting permanent "
            "notes: " + str(exc)) from exc
    daemon._require_exact_permanent_note_set(
        primary_worktree=primary_worktree, head=head)
    for relative_path in daemon.ARCHITECT_PROTECTED_TRACKED_PATHS:
        expected = daemon._git_path_bytes(
            worktree=primary_worktree, object_name=head,
            relative_path=relative_path,
            maximum_bytes=daemon.MAX_PROTECTED_NOTE_BYTES)
        staged = daemon._git_path_bytes(
            worktree=primary_worktree, object_name="",
            relative_path=relative_path,
            maximum_bytes=daemon.MAX_PROTECTED_NOTE_BYTES)
        try:
            working = daemon.stable_regular_bytes(
                path=daemon.os.path.join(primary_worktree, relative_path),
                maximum_bytes=daemon.MAX_PROTECTED_NOTE_BYTES,
                label="protected " + relative_path)
        except (OSError, ValueError) as exc:
            raise daemon.PrimaryWorktreeError(str(exc)) from exc
        if staged != expected or working != expected:
            raise daemon.PrimaryWorktreeError(
                relative_path + " does not match the current Architect "
                "commit and index")
    try:
        ending_head = daemon.worktree_head(worktree=primary_worktree)
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot recheck the Architect commit protecting permanent "
            "notes: " + str(exc)) from exc
    if ending_head != head:
        raise daemon.PrimaryWorktreeError(
            "Architect HEAD changed while permanent notes were checked")


def _validate_sealed_backlog(primary_worktree):
    """Return the backlog bytes after matching the Architect-sealed SHA.

    The backlog and its guard state must both exist or both be
    absent. The guard is JSON naming the backlog path and its SHA-256
    digest (version 2 also records the previous digest); the backlog
    bytes must hash to the sealed digest exactly.

    Arguments:
      primary_worktree = the worktree holding ai/notes.

    Returns:
      The verified backlog bytes, or empty bytes when both files are
      absent.

    Raises:
      daemon.PrimaryWorktreeError: for a lone file, a malformed
        guard, or a digest mismatch.
    """
    notes = daemon.os.path.join(primary_worktree, "ai", "notes")
    backlog_path = daemon.os.path.join(notes, "backlog.md")
    state_path = daemon.os.path.join(notes, daemon.BACKLOG_GUARD_STATE_NAME)
    backlog_exists = daemon.os.path.lexists(backlog_path)
    state_exists = daemon.os.path.lexists(state_path)
    if not backlog_exists and not state_exists:
        if (daemon.os.path.lexists(backlog_path)
                or daemon.os.path.lexists(state_path)):
            raise daemon.PrimaryWorktreeError(
                "backlog or its guard appeared while absence was checked")
        return b""
    if backlog_exists != state_exists:
        raise daemon.PrimaryWorktreeError(
            "backlog and its Architect-sealed guard must either both exist "
            "or both be absent")
    try:
        state_before = daemon.stable_regular_bytes(
            path=state_path, maximum_bytes=daemon.MAX_BACKLOG_GUARD_STATE_BYTES,
            label="backlog guard state")
        backlog = daemon.stable_regular_bytes(
            path=backlog_path, maximum_bytes=daemon.MAX_BACKLOG_LEDGER_BYTES,
            label="Architect backlog")
        state_after = daemon.stable_regular_bytes(
            path=state_path, maximum_bytes=daemon.MAX_BACKLOG_GUARD_STATE_BYTES,
            label="backlog guard state")
    except (OSError, ValueError) as exc:
        raise daemon.PrimaryWorktreeError(str(exc)) from exc
    if state_after != state_before:
        raise daemon.PrimaryWorktreeError(
            "backlog guard state changed while the backlog was checked")
    try:
        state = daemon.json.loads(
            state_before.decode("utf-8", errors="strict"),
            object_pairs_hook=daemon._duplicate_key_refusal)
    except (UnicodeDecodeError, daemon.json.JSONDecodeError) as exc:
        raise daemon.PrimaryWorktreeError(
            "backlog guard state is not exact UTF-8 JSON: " + str(exc)) \
            from exc
    version = state.get("version") if isinstance(state, dict) else None
    expected_fields = {"backlog", "sha256", "version"}
    if version == 2:
        expected_fields.add("previous_sha256")
    if (not isinstance(state, dict)
            or type(version) is not int or version not in {1, 2}
            or set(state) != expected_fields
            or state.get("backlog") != "ai/notes/backlog.md"
            or not isinstance(state.get("sha256"), str)
            or daemon.re.fullmatch(r"[0-9a-f]{64}", state["sha256"]) is None
            or (version == 2 and (
                not isinstance(state.get("previous_sha256"), str)
                or daemon.re.fullmatch(
                    r"[0-9a-f]{64}", state["previous_sha256"]) is None))):
        raise daemon.PrimaryWorktreeError(
            "backlog guard state has missing, extra, or invalid fields")
    observed = daemon.hashlib.sha256(backlog).hexdigest()
    if observed != state["sha256"]:
        raise daemon.PrimaryWorktreeError(
            "backlog differs from the SHA-256 last sealed by the Architect")
    return backlog


def require_closed_backlog_ticket(ticket_anchor, sealed_backlog):
    """Prove one ticket is Closed before landing.

    The ticket must sit below the Closed-tickets heading, must not
    appear in the Open index, and its section must carry the four
    standard headings in order with exactly one CLOSED status line
    and a "What is missing" section saying only "Nothing for this
    ticket."

    Arguments:
      ticket_anchor  = the ticket's anchor name.
      sealed_backlog = verified backlog bytes from the seal check.

    Raises:
      daemon.BacklogTicketOpenError: when the ticket is still open.
      daemon.TicketCycleStateError: for undecodable bytes or a
        malformed ticket section.
    """
    try:
        lines = sealed_backlog.decode("utf-8", errors="strict").splitlines()
    except (AttributeError, UnicodeDecodeError) as exc:
        raise daemon.TicketCycleStateError("backlog is not UTF-8") from exc
    marker = '<a id="' + ticket_anchor + '"></a>'
    if (lines.count("# Closed tickets") != 1 or lines.count(marker) != 1
            or any(daemon.OPEN_BACKLOG_CANDIDATE_RE.match(line)
                   and "(#" + ticket_anchor + ")" in line for line in lines)):
        raise daemon.BacklogTicketOpenError("ticket is Open: " + ticket_anchor)
    start = lines.index(marker) + 1
    if start <= lines.index("# Closed tickets"):
        raise daemon.BacklogTicketOpenError("ticket is Open: " + ticket_anchor)
    end = next((index for index in range(start + 1, len(lines))
                if lines[index].startswith(("## ", '<a id="'))), len(lines))
    section = lines[start:end]
    headings = ["### High-level summary", "### Current status",
                "### What is already fixed", "### What is missing"]
    if (not section or not section[0].startswith("## ")
            or any(section.count(heading) != 1 for heading in headings)):
        raise daemon.TicketCycleStateError("invalid ticket: " + ticket_anchor)
    positions = [section.index(heading) for heading in headings]
    if positions != sorted(positions):
        raise daemon.TicketCycleStateError("invalid ticket: " + ticket_anchor)
    status = [line for line in section[positions[1] + 1:positions[2]]
              if line.startswith("**CLOSED.**")]
    missing = section[positions[3] + 1:]
    missing = missing[:next((index for index, line in enumerate(missing)
                            if line.startswith(("### ", "<details>"))),
                           len(missing))]
    if len(status) != 1 or [line for line in missing if line] != [
            "Nothing for this ticket."]:
        raise daemon.TicketCycleStateError("invalid ticket: " + ticket_anchor)


def _bridge_local_sealed_backlog(primary_worktree):
    """Adopt legacy local state or initialize the tracked backlog seal.

    Startup bridge for three situations: a sync-recovery file left by
    an interrupted run is restored or discarded against the saved
    digest; a tracked backlog without a guard gains a fresh seal when
    it matches HEAD; and files beside the main checkout are copied in
    when the primary has none. Every path ends by revalidating the
    seal.

    Arguments:
      primary_worktree = the Architect worktree receiving the seal.

    Raises:
      daemon.PrimaryWorktreeError: for conflicting copies or a seal
        that cannot be validated.
    """
    names = ("backlog.md", daemon.BACKLOG_GUARD_STATE_NAME)
    source_notes = daemon.os.path.join(daemon.REPO_ROOT, "ai", "notes")
    target_notes = daemon.os.path.join(primary_worktree, "ai", "notes")
    targets = [daemon.os.path.join(target_notes, name) for name in names]
    recovery = daemon.os.path.join(target_notes, daemon.BACKLOG_SYNC_RECOVERY_NAME)
    if daemon.os.path.lexists(recovery):
        previous_digest = daemon._saved_backlog_digest(
            primary_path=primary_worktree)
        restored_recovery = False
        saved = daemon.stable_regular_bytes(
            path=recovery, maximum_bytes=daemon.MAX_BACKLOG_LEDGER_BYTES,
            label="backlog sync recovery")
        if not daemon.os.path.lexists(targets[0]):
            daemon.os.replace(recovery, targets[0])
            restored_recovery = True
        else:
            working = daemon.stable_regular_bytes(
                path=targets[0], maximum_bytes=daemon.MAX_BACKLOG_LEDGER_BYTES,
                label="Architect backlog")
            if working == saved:
                daemon.os.unlink(recovery)
                restored_recovery = (
                    daemon.hashlib.sha256(saved).hexdigest() != previous_digest)
            else:
                head = daemon._run_git(
                    repository_root=primary_worktree,
                    arguments=["show", "HEAD:" + daemon.BACKLOG_RELATIVE_PATH],
                    check=False)
                if head.returncode != 0 or head.stdout != working:
                    raise daemon.PrimaryWorktreeError(
                        "backlog sync recovery conflicts with visible work")
                daemon.os.replace(recovery, targets[0])
                restored_recovery = True
        if restored_recovery:
            daemon._reseal_recovered_backlog(
                primary_path=primary_worktree,
                previous_digest=previous_digest, backlog=saved)
    if daemon.os.path.lexists(targets[0]) and not daemon.os.path.lexists(targets[1]):
        committed = daemon._run_git(
            repository_root=primary_worktree,
            arguments=["show", "HEAD:" + daemon.BACKLOG_RELATIVE_PATH],
            check=False)
        if (committed.returncode == 0
                and len(committed.stdout) <= daemon.MAX_BACKLOG_LEDGER_BYTES):
            working = daemon.stable_regular_bytes(
                path=targets[0], maximum_bytes=daemon.MAX_BACKLOG_LEDGER_BYTES,
                label="tracked Architect backlog")
        else:
            working = None
        if working is not None and committed.stdout == working:
            daemon._atomic_write_primary_state(
                state={"backlog": daemon.BACKLOG_RELATIVE_PATH,
                       "sha256": daemon.hashlib.sha256(working).hexdigest(),
                       "version": 1},
                path=targets[1])
    if all(daemon.os.path.lexists(path) for path in targets):
        try:
            daemon._validate_sealed_backlog(primary_worktree=primary_worktree)
        except daemon.PrimaryWorktreeError:
            if daemon._clean_worktree_status(worktree=primary_worktree):
                raise
            committed = daemon._run_git(
                repository_root=primary_worktree,
                arguments=["show", "HEAD:" + daemon.BACKLOG_RELATIVE_PATH],
                check=False)
            working = daemon.stable_regular_bytes(
                path=targets[0], maximum_bytes=daemon.MAX_BACKLOG_LEDGER_BYTES,
                label="tracked Architect backlog")
            if committed.returncode != 0 or committed.stdout != working:
                raise
            daemon._atomic_write_primary_state(
                state={"backlog": daemon.BACKLOG_RELATIVE_PATH,
                       "sha256": daemon.hashlib.sha256(working).hexdigest(),
                       "version": 1},
                path=targets[1])
            daemon._validate_sealed_backlog(primary_worktree=primary_worktree)
        return

    sources = [daemon.os.path.join(source_notes, name) for name in names]
    if not any(daemon.os.path.lexists(path) for path in sources):
        daemon._validate_sealed_backlog(primary_worktree=primary_worktree)
        return
    daemon._validate_sealed_backlog(primary_worktree=daemon.REPO_ROOT)

    for source, target in zip(sources, targets):
        if (daemon.os.path.lexists(target)
                and not daemon._regular_files_equal(source, target)):
            raise daemon.PrimaryWorktreeError(
                "primary backlog conflicts: " + target)
    for source, target in zip(sources, targets):
        if not daemon.os.path.lexists(target):
            daemon._copy_regular_archive_file(
                source=source, destination=target,
                expected_size=daemon.os.lstat(source).st_size)
    daemon._validate_sealed_backlog(primary_worktree=primary_worktree)


def _validate_current_protected_primary_state(primary_worktree):
    """Accept current Architect authority, including a concurrent seal/commit.

    The protected-state check can race a legitimate Architect turn
    that is committing notes or resealing the backlog, so a failure
    is retried a few times before it becomes an error.

    Arguments:
      primary_worktree = the Architect worktree.

    Raises:
      daemon.PrimaryWorktreeError: when the state never settles into
        an accepted form.
    """
    last_error = None
    for attempt in range(daemon.PROTECTED_STATE_RECHECK_ATTEMPTS):
        try:
            daemon._validate_protected_tracked_state(
                primary_worktree=primary_worktree)
            daemon._validate_sealed_backlog(primary_worktree=primary_worktree)
            return
        except daemon.PrimaryWorktreeError as exc:
            last_error = exc
            if attempt + 1 < daemon.PROTECTED_STATE_RECHECK_ATTEMPTS:
                daemon.time.sleep(daemon.PROTECTED_STATE_RECHECK_SECONDS)
    raise daemon.PrimaryWorktreeError(
        "shared Architect-owned notes are not in an accepted state: "
        + str(last_error))


def _capture_shared_protected_state():
    """Return a proof that can revalidate shared protected notes by authority.

    Returns:
      Mapping with the primary worktree path and its directory
      identity, taken after the protected state validated.
    """
    primary = daemon.AGENT_CWD["fable"]
    identity = daemon._plain_directory(
        path=primary, label="saved Architect primary worktree")
    daemon._validate_current_protected_primary_state(
        primary_worktree=primary)
    return {"primary": primary, "identity": identity}


def _recheck_shared_protected_state(proof):
    """Allow only an Architect-sealed backlog or committed permanent notes.

    Arguments:
      proof = the mapping from _capture_shared_protected_state.

    Raises:
      daemon.PrimaryWorktreeError: for a malformed proof, a moved
        primary directory, or an unaccepted protected state.
    """
    if (not isinstance(proof, dict)
            or set(proof) != {"identity", "primary"}):
        raise daemon.PrimaryWorktreeError(
            "shared protected-note proof is missing or malformed")
    daemon._require_directory_identity(
        path=proof["primary"], identity=proof["identity"],
        label="saved Architect primary worktree")
    daemon._validate_current_protected_primary_state(
        primary_worktree=proof["primary"])


def capture_persistent_role_state(agent):
    """Freeze tracked-state authority for persistent non-Implementer roles.

    What is frozen depends on the role. Sol must start clean, so its
    HEAD is recorded and any tracked change refuses dispatch. The
    Implementer's own worktree is not frozen — implementing is its
    job — but the shared protected notes are. The Architect's
    ordinary tracked, staged, and untracked state is captured exactly
    so an audit turn can prove it changed nothing outside its
    authority.

    Arguments:
      agent = the role; anything but the three roles returns None.

    Returns:
      The role's proof mapping, or ``None``.

    Raises:
      daemon.PrimaryWorktreeError: when the state cannot be captured
        or Sol starts dirty.
    """
    if agent not in {"fable", "opus", "sol"}:
        return None
    worktree = daemon.AGENT_CWD[agent]
    shared_proof = (daemon._capture_shared_protected_state()
                    if agent in {"opus", "sol"} else None)
    try:
        head = daemon.worktree_head(worktree=worktree)
    except daemon.TicketCycleStateError as exc:
        raise daemon.PrimaryWorktreeError(
            "cannot capture persistent " + agent + " HEAD: " + str(exc)) \
            from exc
    if agent == "sol":
        if daemon._tracked_worktree_changes(worktree=worktree):
            raise daemon.PrimaryWorktreeError(
                "saved Sol worktree has tracked or nonignored untracked "
                "changes; preserve them manually before Red Team dispatch")
        return {"agent": agent, "worktree": worktree, "head": head,
                "shared_proof": shared_proof}
    if agent == "opus":
        return {"agent": agent, "worktree": worktree,
                "shared_proof": shared_proof}
    return {
        "agent": agent,
        "worktree": worktree,
        "base": head,
        "worktree_state": daemon._architect_ordinary_tracked_state(
            worktree=worktree, base_commit=head, cached=False),
        "index_state": daemon._architect_ordinary_tracked_state(
            worktree=worktree, base_commit=head, cached=True),
        "untracked_state": daemon._ordinary_untracked_worktree_state(
            worktree=worktree),
    }


def implementer_checkpoint_delivered(state_path):
    """Return true only after the hook records its complete instruction.

    Arguments:
      state_path = the checkpoint state file, or empty for none.

    Returns:
      True only when the file holds the exact triggered marker.
    """
    if not state_path:
        return False
    marker = daemon.stable_regular_bytes(
        path=state_path, maximum_bytes=32,
        label="Implementer checkpoint marker", missing_ok=True)
    return marker == b"triggered\n"


def recheck_persistent_role_state(proof):
    """Refuse tracked edits outside the authority of Fable or Sol.

    Arguments:
      proof = the mapping from capture_persistent_role_state, or
              ``None`` to check nothing.

    Raises:
      daemon.PrimaryWorktreeError: when Sol's worktree changed at
        all, or the Architect's ordinary tracked, staged, or
        untracked state differs from the captured proof. Changes are
        preserved on disk for inspection, never reverted.
    """
    if proof is None:
        return
    agent = proof["agent"]
    worktree = proof["worktree"]
    if agent in {"opus", "sol"}:
        daemon._recheck_shared_protected_state(proof=proof["shared_proof"])
    if agent == "opus":
        return
    if agent == "sol":
        try:
            current_head = daemon.worktree_head(worktree=worktree)
        except daemon.TicketCycleStateError as exc:
            raise daemon.PrimaryWorktreeError(
                "cannot recheck persistent Sol HEAD: " + str(exc)) from exc
        if (current_head != proof["head"]
                or daemon._tracked_worktree_changes(worktree=worktree)):
            raise daemon.PrimaryWorktreeError(
                "Red Team changed tracked or nonignored untracked files in "
                "its persistent worktree; the changes were preserved for "
                "inspection")
        return
    if agent != "fable":
        raise daemon.PrimaryWorktreeError("unknown persistent role-state proof")
    current_worktree = daemon._architect_ordinary_tracked_state(
        worktree=worktree, base_commit=proof["base"], cached=False)
    current_index = daemon._architect_ordinary_tracked_state(
        worktree=worktree, base_commit=proof["base"], cached=True)
    current_untracked = daemon._ordinary_untracked_worktree_state(
        worktree=worktree)
    if (current_worktree != proof["worktree_state"]
            or current_index != proof["index_state"]
            or current_untracked != proof["untracked_state"]):
        raise daemon.PrimaryWorktreeError(
            "Architect changed ordinary tracked or nonignored untracked "
            "source, tests, README, or tools in the coordination worktree; "
            "the changes were preserved for inspection")


def validate_live_sol_dispatch_topology():
    """Run the general topology proof for the Red Team checkout.

    Returns:
      The proof mapping from validate_live_agent_dispatch_topology
      for ``agent="sol"``; kept as a named entry point so focused
      witnesses can exercise the Sol case alone.
    """
    return daemon.validate_live_agent_dispatch_topology(agent="sol")


def recheck_sol_dispatch_directories(proof):
    """Run the general directory recheck for a Red Team proof.

    Arguments:
      proof = the mapping from validate_live_sol_dispatch_topology.

    Returns:
      The delegate's result; kept as a named entry point so focused
      witnesses can exercise the Sol case alone.
    """
    return daemon.recheck_agent_dispatch_directories(proof=proof)


def revalidate_sol_dispatch_topology(proof):
    """Run the general topology revalidation for a Red Team proof.

    Arguments:
      proof = the pre-claim proof from
              validate_live_sol_dispatch_topology.

    Returns:
      The fresh proof, which must equal the old one exactly; kept as
      a named entry point so focused witnesses can exercise the Sol
      case alone.
    """
    return daemon.revalidate_agent_dispatch_topology(proof=proof)

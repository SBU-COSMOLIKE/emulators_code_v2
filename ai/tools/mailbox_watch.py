"""Watch-loop rendezvous, safe-kill windows, and cycle barriers.

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
    "report_in_flight_status",
    "report_admitted_status",
    "report_safe_interval_closed",
    "_RendezvousPermit",
    "SafeKillRendezvous",
    "_rendezvous_turn_started",
    "_rendezvous_turn_finished",
    "_ticket_cycle_completed",
    "waiting_messages_text",
    "run_safe_kill_countdown",
    "report_ordinary_safe_poll",
    "positive_int",
    "nonnegative_cycle_count",
    "nonnegative_max_characters",
    "strict_cycle_ledger_count",
    "acquire_cycle_completion_barrier",
    "release_cycle_completion_barrier",
    "acquire_positive_cycle_exit_barrier",
    "report_cycle_limit_exit",
    "report_cycle_work_complete",
    "report_cycle_completion_unverified",
    "truthy_fix_only",
    "validate_model_name",
)


def report_in_flight_status(count):
    """Print the truthful unsafe status for one or more live children."""
    noun = "turn" if count == 1 else "turns"
    print(str(count) + " " + noun
          + " in flight; not safe to stop.", flush=True)


def report_admitted_status():
    """Expire any earlier safe line before an attempt can claim its file."""
    print("dispatch preparation admitted; not safe to stop.", flush=True)


def report_safe_interval_closed():
    """Invalidate a completed safe interval before admissions can reopen."""
    print("safe interval ended; not safe to stop.", flush=True)


class _RendezvousPermit:
    """One watch-global release from before claim through state publication."""

    def __init__(self):
        self.launched = False
        self.reaped = False
        self.released = False


class SafeKillRendezvous:
    """Close watch admissions periodically and prove every lane is idle.

    ``active_attempts`` deliberately covers more than live children.  A turn
    that passed the admission gate but has not reached Popen can already have
    claimed its mailbox file, so an advertised safe window must wait for that
    whole attempt as well as for every launched child.
    """

    def __init__(self, source_path=None, source_stamp=None,
                 ticket_cycle_limit=None, ticket_cycle_topology=None,
                 companion_sources=None):
        if (ticket_cycle_limit is not None
                and ticket_cycle_topology not in daemon.ARCHITECT_COMMIT_MODES):
            raise ValueError(
                "a finite ticket-cycle controller needs a valid topology")
        self._lock = daemon.threading.Condition()
        self._active_attempts = 0
        self._in_flight = 0
        self._completed = 0
        self._ticket_cycles_completed = 0
        self._ticket_cycle_limit = ticket_cycle_limit
        self._ticket_cycle_topology = ticket_cycle_topology
        self._draining = False
        self._deadline = self._next_deadline()
        self._source_path = source_path
        self._source_stamp = source_stamp
        # The daemon's source spans mailbox_daemon.py plus its part files;
        # an edit to ANY of them must stop stale code from dispatching.
        # ``companion_sources`` holds (path, stamp) pairs for the parts.
        self._companion_sources = (tuple(companion_sources)
                                   if companion_sources is not None else ())
        self._source_changed = False

    @staticmethod
    def _next_deadline():
        return (daemon.time.monotonic()
                + float(daemon.RENDEZVOUS_MINUTE_INTERVAL) * 60.0)

    def _arm_if_due_locked(self):
        if (self._completed >= daemon.RENDEZVOUS_DISPATCH_INTERVAL
                or daemon.time.monotonic() >= self._deadline):
            self._draining = True

    def _stop_for_source_change_locked(self):
        if self._source_path is None:
            return
        watched = ((self._source_path, self._source_stamp),)
        watched = watched + self._companion_sources
        changed = False
        for path, stamp in watched:
            try:
                if daemon.os.path.getmtime(path) != stamp:
                    changed = True
            except OSError:
                changed = True
        if changed:
            self._source_changed = True
            self._draining = True

    def begin_attempt(self, ignore_ticket_limit=False):
        """Return a permit, optionally for cycle-free administration."""
        while True:
            with self._lock:
                self._stop_for_source_change_locked()
                if (not ignore_ticket_limit
                        and self._ticket_cycle_limit is not None
                        and self._ticket_cycles_completed
                        >= self._ticket_cycle_limit):
                    return None
                self._arm_if_due_locked()
                if self._draining:
                    return None
                # Reserve cadence capacity across all cwd lanes.  A refusal
                # or Popen failure later frees the reservation; a reaped child
                # converts it into one completed turn.  This prevents a fast
                # lane from starting turn K+1 while turn K is still live.
                if (self._completed + self._active_attempts
                        < daemon.RENDEZVOUS_DISPATCH_INTERVAL):
                    permit = daemon._RendezvousPermit()
                    self._active_attempts = self._active_attempts + 1
                else:
                    self._lock.wait()
                    continue
            # This flushed transition happens before begin_attempt returns,
            # so dispatch cannot claim the root message while an expired
            # ordinary-poll or countdown line is still the visible status.
            try:
                daemon.report_admitted_status()
            except BaseException:
                # A broken output stream must not strand an unreturned permit
                # and make the global gate appear permanently busy.
                with self._lock:
                    self._active_attempts = self._active_attempts - 1
                    self._lock.notify_all()
                raise
            return permit

    def source_changed(self):
        """Return whether an admission observed a stale daemon source."""
        with self._lock:
            return self._source_changed

    def turn_started(self, permit):
        """Record a successful Popen and print the exact unsafe status."""
        with self._lock:
            if permit.launched:
                raise RuntimeError("rendezvous permit launched twice")
            permit.launched = True
            self._in_flight = self._in_flight + 1
            count = self._in_flight
            daemon.report_in_flight_status(count=count)

    def turn_finished(self, permit):
        """Count one reaped child regardless of its exit or archive result."""
        with self._lock:
            if not permit.launched or permit.reaped:
                raise RuntimeError("invalid rendezvous child completion")
            permit.reaped = True
            self._in_flight = self._in_flight - 1
            self._completed = self._completed + 1
            self._arm_if_due_locked()
            count = self._in_flight
            if count:
                daemon.report_in_flight_status(count=count)
            self._lock.notify_all()

    def finish_attempt(self, permit):
        """Release post-child state work and freeze on an unreaped child."""
        with self._lock:
            if permit.released:
                raise RuntimeError("rendezvous permit released twice")
            permit.released = True
            self._active_attempts = self._active_attempts - 1
            if permit.launched and not permit.reaped:
                # Never advertise safety, or release later work, after losing
                # truthful custody of a child process.
                self._draining = True
            self._arm_if_due_locked()
            self._lock.notify_all()

    def window_ready(self):
        """Return True only for a due drain with no child or preparation."""
        with self._lock:
            self._arm_if_due_locked()
            return (self._draining and self._active_attempts == 0
                    and self._in_flight == 0)

    def all_idle(self):
        """Return whether no admitted attempt or launched child remains."""
        with self._lock:
            return self._active_attempts == 0 and self._in_flight == 0

    def ticket_cycle_returned(self):
        """Record one completed ticket.

        A normal ticket reaches this method after its Red Team return. A
        ticket in a no-Red-Team mode reaches it after the daemon records local
        landing L. Child-turn cadence and manual safe-stop windows never call
        it.
        """
        with self._lock:
            self._ticket_cycles_completed = (
                self._ticket_cycles_completed + 1)
            self._lock.notify_all()

    def completed_ticket_cycles(self):
        """Return the completed ticket-cycle count for this watch."""
        with self._lock:
            return self._ticket_cycles_completed

    def ticket_cycle_limit_reached(self):
        """Return whether a positive cycle limit has already been met."""
        with self._lock:
            return (self._ticket_cycle_limit is not None
                    and self._ticket_cycles_completed
                    >= self._ticket_cycle_limit)

    def ticket_cycle_limit_value(self):
        """Return the positive ticket limit, or ``None`` when unbounded."""
        with self._lock:
            return self._ticket_cycle_limit

    def ticket_cycle_topology_value(self):
        """Return the topology bound to this finite watch, if any."""
        with self._lock:
            return self._ticket_cycle_topology

    def restore_completed_ticket_cycles(self, count):
        """Restore durable progress for an interrupted finite watch."""
        if (isinstance(count, bool) or not isinstance(count, int)
                or count < 0):
            raise ValueError("restored ticket-cycle count must be nonnegative")
        with self._lock:
            if self._ticket_cycles_completed != 0:
                raise ValueError("ticket-cycle progress was already restored")
            if (self._ticket_cycle_limit is not None
                    and count > self._ticket_cycle_limit):
                raise ValueError("restored progress exceeds the cycle limit")
            self._ticket_cycles_completed = count

    def reset_after_safe_opportunity(self):
        """Start a fresh cadence epoch after a proven all-idle interval."""
        with self._lock:
            if self._active_attempts != 0 or self._in_flight != 0:
                raise RuntimeError("cannot reset a non-idle rendezvous")
            self._completed = 0
            self._draining = False
            self._deadline = self._next_deadline()
            self._lock.notify_all()


def _rendezvous_turn_started():
    """Bind a successful Popen to this worker's active watch permit."""
    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
    permit = getattr(daemon._RENDEZVOUS_LOCAL, "permit", None)
    if controller is not None and permit is not None:
        controller.turn_started(permit=permit)


def _rendezvous_turn_finished():
    """Bind a reaped child to this worker's active watch permit."""
    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
    permit = getattr(daemon._RENDEZVOUS_LOCAL, "permit", None)
    if controller is not None and permit is not None:
        controller.turn_finished(permit=permit)


def _ticket_cycle_completed():
    """Count one verified ticket completion for the active watch."""
    controller = daemon._ACTIVE_WATCH_RENDEZVOUS
    if controller is not None:
        controller.ticket_cycle_returned()


def waiting_messages_text(count):
    """Return a grammatically exact waiting-message count."""
    if count == 0:
        return "no messages waiting"
    noun = "message" if count == 1 else "messages"
    return str(count) + " " + noun + " waiting"


def run_safe_kill_countdown(controller):
    """Print 20 safe seconds when no role is starting or running."""
    if not controller.window_ready():
        raise RuntimeError(
            "safe Ctrl-C countdown requested while a role is still active")
    for seconds_more in range(daemon.SAFE_KILL_COUNTDOWN_SECONDS - 1, -1, -1):
        waiting = len(daemon.pending_messages())
        print("every enabled role is idle; safe to Ctrl-C for "
              + str(seconds_more)
              + "s more; " + daemon.waiting_messages_text(count=waiting) + ".",
              flush=True)
        daemon.time.sleep(1)
    daemon.report_safe_interval_closed()
    controller.reset_after_safe_opportunity()


def report_ordinary_safe_poll(controller, reset_cadence=True):
    """Report a safe Ctrl-C wait when every role job is idle.

    This ordinary mailbox check never completes a ticket cycle. Only a
    correlated Red Team return for one daemon-recorded local landing L does
    that in the default three-role mode.
    """
    if not controller.all_idle():
        return False
    waiting = len(daemon.pending_messages())
    print("every enabled role is idle; safe to Ctrl-C for this "
          + str(daemon.WATCH_POLL_SECONDS) + "s poll; "
          + daemon.waiting_messages_text(count=waiting) + ".", flush=True)
    if reset_cadence:
        controller.reset_after_safe_opportunity()
    return True


def positive_int(value):
    """Parse an argparse integer that must be strictly positive."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise daemon.argparse.ArgumentTypeError(
            "value must be a positive integer") from exc
    if parsed <= 0 or parsed > daemon.MAX_DISPATCH_TIMEOUT_MINUTES:
        raise daemon.argparse.ArgumentTypeError(
            "value must be a positive integer no larger than "
            + str(daemon.MAX_DISPATCH_TIMEOUT_MINUTES))
    return parsed


def nonnegative_cycle_count(value):
    """Parse an argparse cycle count, including zero's drain-all meaning."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise daemon.argparse.ArgumentTypeError(
            "cycle count must be a nonnegative integer") from exc
    if parsed < 0 or parsed > daemon.MAX_CYCLE_COUNT:
        raise daemon.argparse.ArgumentTypeError(
            "cycle count must be a nonnegative integer no larger than "
            + str(daemon.MAX_CYCLE_COUNT))
    return parsed


def nonnegative_max_characters(value):
    """Parse a ticket character limit, where zero means unlimited."""
    if not isinstance(value, str) or daemon.re.fullmatch(r"[0-9]+", value) is None:
        raise daemon.argparse.ArgumentTypeError(
            "max characters must use only decimal digits 0 through 9")
    return int(value)


def strict_cycle_ledger_count():
    """Read the cycle-zero ledger fail-closed from one verified regular file."""
    try:
        before = daemon.os.lstat(daemon.BACKLOG_LEDGER)
    except OSError as exc:
        return None, "cannot stat backlog ledger: " + str(exc)
    if not daemon.stat.S_ISREG(before.st_mode):
        return None, "backlog ledger is not a regular file"
    if before.st_size > daemon.MAX_BACKLOG_LEDGER_BYTES:
        return None, "backlog ledger is too large to verify"
    flags = daemon.os.O_RDONLY
    if hasattr(daemon.os, "O_CLOEXEC"):
        flags = flags | daemon.os.O_CLOEXEC
    if hasattr(daemon.os, "O_NONBLOCK"):
        flags = flags | daemon.os.O_NONBLOCK
    if hasattr(daemon.os, "O_NOFOLLOW"):
        flags = flags | daemon.os.O_NOFOLLOW
    try:
        descriptor = daemon.os.open(daemon.BACKLOG_LEDGER, flags)
    except OSError as exc:
        return None, "cannot open backlog ledger: " + str(exc)
    try:
        opened = daemon.os.fstat(descriptor)
        if (not daemon.stat.S_ISREG(opened.st_mode)
                or opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino):
            return None, "backlog ledger changed identity while opening"
        chunks = []
        size = 0
        while True:
            chunk = daemon.os.read(descriptor, 65536)
            if not chunk:
                break
            size = size + len(chunk)
            if size > daemon.MAX_BACKLOG_LEDGER_BYTES:
                return None, "backlog ledger grew too large while reading"
            chunks.append(chunk)
        try:
            text = b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, "backlog ledger is not valid UTF-8: " + str(exc)
        metadata_before = (
            opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        try:
            current = daemon.os.lstat(daemon.BACKLOG_LEDGER)
        except OSError as exc:
            return None, "cannot restat backlog ledger: " + str(exc)
        if (not daemon.stat.S_ISREG(current.st_mode)
                or current.st_dev != opened.st_dev
                or current.st_ino != opened.st_ino):
            return None, "backlog ledger changed identity while reading"
        after_identity = daemon.os.fstat(descriptor)
        metadata_after_identity = (
            after_identity.st_size, after_identity.st_mtime_ns,
            after_identity.st_ctime_ns)
        if metadata_after_identity != metadata_before:
            return None, "backlog ledger changed while verifying identity"
    except OSError as exc:
        return None, "cannot verify backlog ledger: " + str(exc)
    finally:
        daemon.os.close(descriptor)
    count = sum(1 for line in text.splitlines()
                if line.startswith("- OPEN"))
    return count, None


def acquire_cycle_completion_barrier(backlog_outcome,
                                     skip_redteam=False):
    """Return a held send barrier only when cycle-zero work is verified done.

    Daemon sends serialize publication through ``.sequence.lock``. Holding
    that lock from the final queue/ledger scan until the watch lock is released
    gives zero mode a real cutoff: a racing send either lands before the scan
    and prevents exit, or lands after the watcher is no longer advertised.
    """
    failed_debt = daemon.architect_notes_failed_debt_error()
    if failed_debt is not None:
        return None, failed_debt
    if backlog_outcome is False:
        return None, None
    lock_path = daemon.os.path.join(daemon.MAILBOX, ".sequence.lock")
    lock_file = None
    try:
        lock_file = open(lock_path, "a+", encoding="utf-8")
        daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_EX)
    except OSError as exc:
        if lock_file is not None:
            try:
                lock_file.close()
            except OSError:
                pass
        return None, "cannot lock mailbox publication: " + str(exc)
    try:
        ledger, error = daemon.strict_cycle_ledger_count()
        try:
            active_cycles = daemon.active_ticket_cycle_count(
                skip_redteam=skip_redteam)
        except daemon.TicketCycleStateError as exc:
            active_cycles = None
            error = "cannot verify ticket-cycle state: " + str(exc)
        waiting_before = daemon.enabled_pending_messages(
            skip_redteam=skip_redteam)
        waiting_after = daemon.enabled_pending_messages(
            skip_redteam=skip_redteam)
    except OSError as exc:
        ledger = None
        error = "cannot verify pending mailbox messages: " + str(exc)
        waiting_before = []
        waiting_after = []
    notes_pending = daemon.architect_notes_transition_pending()
    if (error is None and ledger == 0 and active_cycles == 0
            and not notes_pending
            and not waiting_before and not waiting_after):
        return lock_file, None
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()
    return None, error


def release_cycle_completion_barrier(lock_file):
    """Release the final cycle-zero send barrier after watch-lock release."""
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()


def acquire_positive_cycle_exit_barrier(backlog_outcome,
                                        skip_redteam=False):
    """Fence sends and refuse finite exit while note administration waits."""
    failed_debt = daemon.architect_notes_failed_debt_error()
    if failed_debt is not None:
        return None, failed_debt
    if backlog_outcome is False:
        return None, None
    lock_path = daemon.os.path.join(daemon.MAILBOX, ".sequence.lock")
    lock_file = None
    try:
        lock_file = open(lock_path, "a+", encoding="utf-8")
        daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_EX)
        active = daemon.active_ticket_cycle_count(skip_redteam=skip_redteam)
        notes_pending = daemon.architect_notes_transition_pending()
    except (OSError, daemon.TicketCycleStateError) as exc:
        if lock_file is not None:
            try:
                daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
                lock_file.close()
            except OSError:
                pass
        return None, "cannot verify finite-cycle exit: " + str(exc)
    if active == 0 and not notes_pending:
        return lock_file, None
    daemon.fcntl.flock(lock_file.fileno(), daemon.fcntl.LOCK_UN)
    lock_file.close()
    return None, None


def report_cycle_limit_exit(completed_cycles, cycle_limit,
                            skip_redteam=False):
    """Report a positive cycle limit after every active role job ends."""
    waiting = len(daemon.pending_messages())
    ledger = daemon.backlog_ledger_count()
    cycle_noun = "cycle" if completed_cycles == 1 else "cycles"
    ledger_noun = "item" if ledger == 1 else "items"
    print("cycle limit reached (" + str(completed_cycles) + "/"
          + str(cycle_limit) + " " + cycle_noun
          + "); every enabled role is idle; watcher stopped; "
          + daemon.waiting_messages_text(count=waiting) + "; " + str(ledger)
          + " backlog " + ledger_noun
          + " still begin with '- OPEN'.", flush=True)
    if skip_redteam:
        daemon.report_deferred_sol_messages()


def report_cycle_work_complete(completed_cycles, skip_redteam=False):
    """Report cycle zero after its waiting-work checks all pass."""
    noun = "cycle" if completed_cycles == 1 else "cycles"
    if skip_redteam:
        print("implementation drain complete after "
              + str(completed_cycles) + " " + noun
              + "; no enabled Architect or Implementer message is waiting "
              "or running; ai/notes/backlog.md has no '- OPEN' item; "
              "disabled Red Team work remains untouched; watcher stopped.",
              flush=True)
        return
    print("cycle work complete after " + str(completed_cycles) + " " + noun
          + "; no role message is waiting or running; "
          "ai/notes/backlog.md has no '- OPEN' item; watcher stopped.",
          flush=True)


def report_cycle_completion_unverified(error):
    """Explain why zero mode stayed live instead of claiming completion."""
    print("cycle zero cannot verify completion: " + error
          + "; watcher remains active.", flush=True)


def truthy_fix_only(value):
    """Parse the deliberately forgiving truthy value for ``--fix-only``.

    The user explicitly allowed capitalization mistakes and surrounding
    whitespace.  Other supplied values are errors rather than silently
    disabling a safety mode because of a typo.
    """
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    raise daemon.argparse.ArgumentTypeError(
        "value must be 1, true, or yes (capitalization is ignored)")


def validate_model_name(value):
    """Accept one provider model name without shell ambiguity."""
    if (not isinstance(value, str) or not value or "\x00" in value
            or any(character.isspace() for character in value)):
        raise daemon.argparse.ArgumentTypeError(
            "model must be one non-whitespace alias or full name")
    return value

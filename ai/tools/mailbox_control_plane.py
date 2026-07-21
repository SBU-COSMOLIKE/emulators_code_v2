"""Protected control-plane two-key state and landing execution.

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
    "control_plane_ticket_state",
    "require_validated_architect_go_receipt",
    "record_control_plane_architect_go",
    "record_control_plane_redteam_decision",
    "record_control_plane_integration_stale",
    "record_control_plane_integration_go",
    "prepare_revalidated_control_plane_landing",
    "protected_landing_ready",
    "control_plane_keys_ready",
    "control_plane_redteam_key_matches",
    "_live_control_plane_fingerprint",
    "trusted_control_plane_check",
    "record_control_plane_check",
    "execute_architect_go_locked",
    "require_architect_landing_locked",
    "_require_retirement_landing_locked",
    "retire_cycle_candidate_locked",
    "retire_superseded_failed_architect_go",
    "retire_cycle_candidate",
    "_symbolic_worktree_branch",
    "_architect_only_sealed_backlog",
    "_architect_backlog_matches_target",
    "_clear_landed_architect_backlog",
    "sync_clean_role_baseline",
    "_role_baseline_plan_locked",
    "preflight_role_baseline_sync",
    "sync_all_clean_role_baselines",
    "_permanent_note_commit_paths",
    "require_architect_notes_commit_object",
    "require_architect_notes_commit",
    "_require_no_ordinary_landing_transition_locked",
    "require_no_ordinary_landing_transition",
    "architect_notes_transition_pending",
    "failed_architect_notes_transition_paths",
    "architect_notes_failed_debt_error",
    "message_belongs_to_active_cycle",
    "requeue_retryable_daemon_message",
    "publish_backlog_close_request",
    "defer_protected_stale_integration",
    "prepared_landing_reached_main",
    "finish_claimed_architect_go",
    "finish_claimed_architect_notes_go",
)


def control_plane_ticket_state(cycle_id, candidate_commit=None):
    """Return a copy of one protected ticket's durable state."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        if active is None or active.get("ticket_class", "ordinary") != (
                "protected-control-plane"):
            return None
        if candidate_commit is not None:
            saved = daemon.read_candidate_state()["cycles"].get(cycle_id)
            if saved is None or saved["commit"] != candidate_commit:
                raise daemon.TicketCycleStateError(
                    "protected decision does not name saved candidate C")
        return dict(active["control_plane"])
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def require_validated_architect_go_receipt(cycle_id, candidate_commit):
    """Require D0's saved proof that one Architect turn produced GO(C)."""
    matches = []
    pattern = daemon.os.path.join(
        daemon.MAILBOX, daemon.IMPLEMENTER_DELIVERY_PREFIX + "*")
    for path in daemon.glob.glob(pattern):
        fields = daemon.os.path.basename(path)[len(daemon.IMPLEMENTER_DELIVERY_PREFIX):] \
            .split("@")
        if len(fields) != 4:
            continue
        request_name, request_digest, return_name, return_digest = fields
        request_match = daemon.PENDING_MESSAGE_RE.fullmatch(request_name)
        return_match = daemon.PENDING_MESSAGE_RE.fullmatch(return_name)
        if (request_match is None or request_match.group(1) != "fable"
                or return_match is None
                or return_match.group(1) != "daemon"):
            continue
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="validated Architect GO receipt")
            message = raw.decode("utf-8", errors="strict")
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        found_cycle, found_candidate, _mode, problem = (
            daemon._architect_go_request(message=message))
        if (problem is None and found_cycle == cycle_id
                and found_candidate == candidate_commit
                and daemon.hashlib.sha256(raw).hexdigest() == return_digest
                and daemon.re.fullmatch(r"[0-9a-f]{64}", request_digest)):
            matches.append(path)
    if len(matches) != 1:
        raise daemon.TicketCycleStateError(
            "protected Architect GO lacks exactly one D0-validated "
            "Architect-turn delivery receipt")


def record_control_plane_architect_go(cycle_id, candidate_commit):
    """Persist the first key before publishing mandatory Red Team work."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        saved = daemon.read_candidate_state()["cycles"].get(cycle_id)
        if (active is None or active["phase"] != "implementation"
                or active.get("ticket_class") != "protected-control-plane"
                or saved is None or saved["commit"] != candidate_commit):
            raise daemon.TicketCycleStateError(
                "Architect protected GO does not name active candidate C")
        control = dict(active["control_plane"])
        prior = control["architect_candidate"]
        if prior == candidate_commit:
            return
        if prior is not None and prior != candidate_commit:
            raise daemon.TicketCycleStateError(
                "protected Architect decision changed candidate C")
        # The receipt is created by D0 only after it validates the fresh
        # Architect outcome. Check it while holding the state lock, then save
        # the decision so later recovery no longer depends on the short-lived
        # delivery hard link.
        daemon.require_validated_architect_go_receipt(
            cycle_id=cycle_id, candidate_commit=candidate_commit)
        control["architect_candidate"] = candidate_commit
        state["active"][cycle_id] = dict(active, control_plane=control)
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def record_control_plane_redteam_decision(cycle_id, candidate_commit,
                                          decision):
    """Persist the second exact key; it grants no landing by itself."""
    if decision not in daemon.CONTROL_PLANE_REVIEW_RESULTS:
        raise daemon.TicketCycleStateError("invalid protected Red Team decision")
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        saved = daemon.read_candidate_state()["cycles"].get(cycle_id)
        if (active is None or active["phase"] != "implementation"
                or active.get("ticket_class") != "protected-control-plane"
                or saved is None or saved["commit"] != candidate_commit):
            raise daemon.TicketCycleStateError(
                "Red Team protected decision does not name active C")
        control = dict(active["control_plane"])
        prior = control["redteam_result"]
        if (prior is not None
                and (prior != decision
                     or control["redteam_candidate"] != candidate_commit)):
            raise daemon.TicketCycleStateError(
                "protected Red Team decision changed identity")
        control["redteam_result"] = decision
        control["redteam_candidate"] = candidate_commit
        state["active"][cycle_id] = dict(active, control_plane=control)
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def record_control_plane_integration_stale(
        cycle_id, candidate_commit, stale_landing, old_main, new_main):
    """Preserve both C approvals while recording that prepared L is stale."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or active["phase"] != "implementation"):
            raise daemon.TicketCycleStateError(
                "stale integration has no active protected ticket")
        control = dict(active["control_plane"])
        if not daemon.control_plane_keys_ready(
                control=control, candidate_commit=candidate_commit):
            raise daemon.TicketCycleStateError(
                "stale integration did not preserve both exact-C approvals")
        control.update({
            "integration_status": "STALE",
            "integration_main": new_main,
            "stale_landing": stale_landing,
            "stale_parent": old_main,
            "integration_evidence": None,
            # C passed the first shadow, but the replacement landing must be
            # checked as the exact combined tree on M1.
            "shadow_status": None,
            "shadow_evidence": None,
        })
        state["active"][cycle_id] = dict(active, control_plane=control)
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def record_control_plane_integration_go(
        cycle_id, candidate_commit, new_main, evidence):
    """Record a fresh Architect GO for the exact C-on-M1 interaction."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or active["phase"] != "implementation"):
            raise daemon.TicketCycleStateError(
                "integration GO has no active protected ticket")
        control = dict(active["control_plane"])
        if (not daemon.control_plane_keys_ready(
                control=control, candidate_commit=candidate_commit)
                or control["integration_status"] != "STALE"
                or control["integration_main"] != new_main):
            raise daemon.TicketCycleStateError(
                "integration GO changed C, M1, or either approval")
        current_main = daemon._exact_git_object(
            arguments=["rev-parse", "--verify",
                       "refs/heads/main^{commit}"],
            label="main at integration revalidation")
        if current_main != new_main:
            raise daemon.TicketCycleStateError(
                "main advanced again before integration GO was recorded")
        control["integration_status"] = "REVALIDATED"
        control["integration_evidence"] = evidence
        state["active"][cycle_id] = dict(active, control_plane=control)
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def prepare_revalidated_control_plane_landing(cycle_id, candidate_commit):
    """Retire only stale L after proving main still equals approved M1."""
    control = daemon.control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit)
    if control is None or control["integration_status"] != "REVALIDATED":
        return
    current_main = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="main before revalidated protected landing")
    approved_main = control["integration_main"]
    stale_landing = control["stale_landing"]
    old_main = control["stale_parent"]
    reference = daemon.cycle_landing_ref(cycle_id=cycle_id)
    journaled = daemon.git_ref_commit(reference=reference)
    if current_main != approved_main:
        # A crash may have occurred after D0 replaced old L with a new
        # provisional landing on the last revalidated main. Bind the next
        # stale event to the landing actually in the private journal, not to
        # an older L that is no longer retryable.
        if journaled is not None and journaled != stale_landing:
            stale_landing = journaled
            old_main = daemon._verify_prepared_landing(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing_commit=journaled)
        problem = daemon._prepared_landing_main_problem(
            candidate_commit=candidate_commit,
            landing_commit=stale_landing, parent_commit=old_main,
            current_main=current_main)
        raise daemon.RetryableArchitectLandingError(
            problem or "main changed after integration revalidation")
    if journaled is None:
        return
    if journaled != stale_landing:
        parent = daemon._verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=journaled)
        if parent != approved_main:
            raise daemon.TicketCycleStateError(
                "replacement landing journal has an unapproved parent")
        return
    daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["update-ref", "-d", reference, stale_landing])
    if daemon.git_ref_commit(reference=reference) is not None:
        raise daemon.TicketCycleStateError("stale landing journal was not retired")


def protected_landing_ready(cycle_id, candidate_commit):
    """Require both independently persisted decisions for exact C."""
    control = daemon.control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit)
    if control is None:
        return True
    return daemon.control_plane_keys_ready(
        control=control, candidate_commit=candidate_commit)


def control_plane_keys_ready(control, candidate_commit):
    """Pure exact-C two-key decision used by D0 and focused tests."""
    return (isinstance(control, dict)
            and daemon.FULL_COMMIT_RE.fullmatch(candidate_commit) is not None
            and control.get("architect_candidate") == candidate_commit
            and control.get("redteam_candidate") == candidate_commit
            and control.get("redteam_result") == "ACCEPT-CONTROL-PLANE")


def control_plane_redteam_key_matches(control, candidate_commit, decision):
    """Return whether D0 already saved this exact Sol decision."""
    return (isinstance(control, dict)
            and decision in daemon.CONTROL_PLANE_REVIEW_RESULTS
            and control.get("redteam_candidate") == candidate_commit
            and control.get("redteam_result") == decision)


def _live_control_plane_fingerprint():
    """Hash D0's live state and trusted refs around a shadow run."""
    digest = daemon.hashlib.sha256()
    for name in (daemon.TICKET_CYCLE_STATE_NAME, daemon.CANDIDATE_STATE_NAME):
        path = daemon.os.path.join(daemon.MAILBOX, name)
        digest.update(name.encode("utf-8") + b"\0")
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_TICKET_CYCLE_STATE_BYTES,
                label="live control-plane state", missing_ok=True)
        except (OSError, ValueError) as exc:
            raise daemon.TicketCycleStateError(str(exc)) from exc
        digest.update(b"<missing>" if raw is None else raw)
    notes = daemon.os.path.join(daemon.WORKTREE, "ai", "notes")
    for name, maximum in (
            ("backlog.md", daemon.MAX_BACKLOG_LEDGER_BYTES),
            (daemon.BACKLOG_GUARD_STATE_NAME, daemon.MAX_BACKLOG_GUARD_STATE_BYTES),
            (daemon.BACKLOG_SYNC_RECOVERY_NAME, daemon.MAX_BACKLOG_LEDGER_BYTES)):
        path = daemon.os.path.join(notes, name)
        digest.update(path.encode("utf-8") + b"\0")
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=maximum,
                label="live control-plane recovery", missing_ok=True)
        except (OSError, ValueError) as exc:
            raise daemon.TicketCycleStateError(str(exc)) from exc
        digest.update(b"<missing>" if raw is None else raw)
    relay_state = []
    for pattern in (".pending-notes-admin-*.json",
                    "pending-main-push-*.txt"):
        relay_state.extend(daemon.glob.glob(
            daemon.os.path.join(daemon.RELAY_DIR, pattern)))
    for path in sorted(relay_state):
        digest.update(path.encode("utf-8") + b"\0")
        try:
            raw = daemon.stable_regular_bytes(
                path=path, maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
                label="live relay recovery state")
        except (OSError, ValueError) as exc:
            raise daemon.TicketCycleStateError(str(exc)) from exc
        digest.update(raw)
    if daemon.ACTIVE_TOPOLOGY is not None:
        for name in ("primary_state", "implementer_state", "sol_state"):
            path = daemon.ACTIVE_TOPOLOGY[name]
            digest.update(path.encode("utf-8") + b"\0")
            try:
                raw = daemon.stable_regular_bytes(
                    path=path, maximum_bytes=daemon.MAX_PRIMARY_STATE_BYTES,
                    label="live control-plane topology")
            except (OSError, ValueError) as exc:
                raise daemon.TicketCycleStateError(str(exc)) from exc
            digest.update(raw)
    refs = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["for-each-ref", "--format=%(refname) %(objectname)",
                   daemon.CANDIDATE_REF_ROOT, "refs/heads/main"])
    digest.update(refs.stdout)
    return digest.hexdigest()


def trusted_control_plane_check(commit, label):
    """Run D1 in a standalone temporary repository under D0's driver.

    This is protocol isolation, not a hostile-process sandbox.  D1 receives
    no path to the live mailbox or Git common directory. D0 verifies that its
    own state and refs are byte-identical after the bounded checks.
    """
    if daemon.FULL_COMMIT_RE.fullmatch(commit) is None:
        raise daemon.TicketCycleStateError("control-plane check needs a full commit")
    before = daemon._live_control_plane_fingerprint()
    daemon.os.makedirs(daemon.RELAY_DIR, exist_ok=True)
    stamp = daemon.datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = daemon.os.path.join(
        daemon.RELAY_DIR, stamp + "-control-plane-" + label + ".log")
    commands = []
    with daemon.tempfile.TemporaryDirectory(prefix="mailbox-control-plane-") as root:
        handoff_repository = daemon.os.path.join(root, "handoff")
        repository = daemon.os.path.join(root, "repo")
        setup = (
            ["git", "init", "--quiet"],
            ["git", "fetch", "--quiet", "--no-tags",
             daemon.AGENT_CWD["fable"], commit],
            ["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        )
        for checkout in (handoff_repository, repository):
            daemon.os.makedirs(checkout, exist_ok=True)
            for command in setup:
                result = daemon.subprocess.run(
                    command, cwd=checkout, stdout=daemon.subprocess.PIPE,
                    stderr=daemon.subprocess.PIPE, check=False)
                if result.returncode != 0:
                    raise daemon.TicketCycleStateError(
                        "cannot create disposable control-plane checkout: "
                        + result.stderr.decode(
                            "utf-8", errors="replace").strip()[:500])

        expected = daemon._CONTROL_PLANE_HANDOFF.copy_d0_state(
            controller=daemon,
            repository=handoff_repository)
        expected_path = daemon.os.path.join(root, "d0-expected.json")
        with open(expected_path, "w", encoding="utf-8") as stream:
            daemon.json.dump(expected, stream, sort_keys=True, indent=2)
            stream.write("\n")
        for reference, inherited_commit in expected["refs"].items():
            fetch = daemon.subprocess.run(
                ["git", "fetch", "--quiet", "--no-tags",
                 daemon.AGENT_CWD["fable"], inherited_commit],
                cwd=handoff_repository, stdout=daemon.subprocess.PIPE,
                stderr=daemon.subprocess.PIPE, check=False)
            if fetch.returncode != 0:
                raise daemon.TicketCycleStateError(
                    "cannot copy D0 Git identity into shadow checkout")
            update = daemon.subprocess.run(
                ["git", "update-ref", reference, inherited_commit],
                cwd=handoff_repository, stdout=daemon.subprocess.PIPE,
                stderr=daemon.subprocess.PIPE, check=False)
            if update.returncode != 0:
                raise daemon.TicketCycleStateError(
                    "cannot publish copied D0 Git identity in shadow checkout")

        commands.append((
            [daemon.sys.executable, "-c", daemon._CONTROL_PLANE_HANDOFF.TAKEOVER_PROBE,
             expected_path],
            handoff_repository))
        # This program is D0's harness. It is generated outside the candidate
        # checkout, while every imported function below comes from D1 at C.
        # Candidate tests are intentionally not imported or trusted here.
        probe = """
import os
import subprocess
import sys
shadow_repository = os.path.realpath(os.getcwd())
from ai.tools import handoff_contract as h
from ai.tools import mailbox_daemon as d
from ai.tools.role_contract import ROLE_CONTRACT

assert os.path.realpath(d.REPO_ROOT) == shadow_repository

base = (
    '- Roles: `Architect + Implementer + Red Team`\\n'
    '- Discovery severity: `medium`\\n'
    '- Review scope: `bounded`\\n')
assert h._require_architect_role_plan(
    base + '- Ticket class: `ordinary`')['ticket_class'] == 'ordinary'
try:
    h._require_architect_role_plan(
        base + '- Ticket class: `protected-control-plane`')
except h.DirectiveError as exc:
    assert 'reserved for Architect-owned ai/notes administration' in str(exc)
else:
    raise AssertionError('protected-control-plane plan was accepted')
assert d.ticket_class_configuration_problem('ordinary', True) is None
for skip_redteam in (False, True):
    problem = d.ticket_class_configuration_problem(
        'protected-control-plane', skip_redteam)
    assert 'Architect-owned ai/notes administration' in problem
    assert 'ticket Open' in problem

tool = ROLE_CONTRACT['protected_paths']['trusted_tools']['mailbox_daemon']
result, paths = d.classify_candidate_scope(
    {tool}, {tool}, ticket_class='ordinary')
assert result == 'PROTECTED_PATH_VIOLATION' and paths == {tool}
result, paths = d.classify_candidate_scope(
    {tool}, {tool}, ticket_class='protected-control-plane')
assert result == 'PROTECTED_PATH_VIOLATION' and paths == {tool}
other = 'emulator/unplanned.py'
result, paths = d.classify_candidate_scope(
    {other}, {tool}, ticket_class='protected-control-plane')
assert result == 'SCOPE_EXCEEDED' and paths == {other}

cycle = 'protected-shadow@' + '1' * 40
c1, c2 = '2' * 40, '3' * 40
request = d.control_plane_review_request_payload(cycle, c1)
found_cycle, found_candidate, _body, problem = (
    d._redteam_control_plane_envelope(request))
assert problem is None and (found_cycle, found_candidate) == (cycle, c1)
receipt = d.control_plane_review_receipt_payload(
    cycle, c1, 'ACCEPT-CONTROL-PLANE', 'accepted')
found_cycle, found_candidate, result, _body, problem = (
    d._control_plane_review_receipt(receipt))
assert problem is None
assert (found_cycle, found_candidate, result) == (
    cycle, c1, 'ACCEPT-CONTROL-PLANE')

control = d.empty_control_plane_state()
assert not d.control_plane_keys_ready(control, c1)
control['architect_candidate'] = c1
assert not d.control_plane_keys_ready(control, c1)
control['redteam_candidate'] = c2
control['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
assert not d.control_plane_keys_ready(control, c1)
control['redteam_candidate'] = c1
control['redteam_result'] = 'REJECT-CONTROL-PLANE'
assert not d.control_plane_keys_ready(control, c1)
control['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
assert d.control_plane_keys_ready(control, c1)
assert not d.control_plane_keys_ready(control, c2)

state = d.empty_ticket_cycle_state()
state['active'][cycle] = {
    'phase': 'implementation', 'commit': None, 'mode': 'normal',
    'route': 'primary', 'ticket_class': 'protected-control-plane',
    'path_scope': [tool], 'control_plane': control}
normalized = d.validate_ticket_cycle_state(state)
assert normalized['active'][cycle]['control_plane'] == control

os.makedirs(d.MAILBOX, exist_ok=True)
d._bridge_local_sealed_backlog(shadow_repository)
owner = d.acquire_dispatch_lock(mode='once')
assert owner is not None
try:
    assert d.acquire_dispatch_lock(mode='once') is None
finally:
    d.release_dispatch_lock(owner)

# Drive the real D1 state and landing functions from this D0-owned program.
# Every Git object, state file, and journal below belongs to the disposable
# candidate checkout. The outer D0 process separately fingerprints the live
# state and refs before and after this child exits.
os.environ['GIT_AUTHOR_NAME'] = 'D0 shadow harness'
os.environ['GIT_AUTHOR_EMAIL'] = 'shadow@example.invalid'
os.environ['GIT_COMMITTER_NAME'] = 'D0 shadow harness'
os.environ['GIT_COMMITTER_EMAIL'] = 'shadow@example.invalid'

def git_result(arguments, stdin=None):
    return subprocess.run(
        ['git', '-C', shadow_repository] + list(arguments),
        input=stdin, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)

def git(arguments, stdin=None):
    result = git_result(arguments, stdin)
    assert result.returncode == 0, (arguments, result.stderr)
    return result.stdout.strip()

def ref_or_none(reference):
    result = git_result(['rev-parse', '--verify', '--quiet',
                         reference + '^{commit}'])
    assert result.returncode in (0, 1), result.stderr
    return result.stdout.strip() if result.returncode == 0 else None

def child_commit(parent, path, content, message):
    git(['read-tree', parent])
    blob = git(['hash-object', '-w', '--stdin'], content)
    git(['update-index', '--add', '--cacheinfo', '100644', blob, path])
    tree = git(['write-tree'])
    commit = git(['commit-tree', tree, '-p', parent], message + '\\n')
    git(['read-tree', 'HEAD'])
    return commit

base_commit = git(['rev-parse', 'HEAD'])
git(['update-ref', 'refs/heads/main', base_commit])
candidate = child_commit(
    base_commit, 'shadow-candidate.txt', 'candidate\\n',
    'shadow candidate')
new_main = child_commit(
    base_commit, 'shadow-main.txt', 'new main\\n',
    'concurrent main')
other_candidate = new_main
cycle = 'protected-shadow-landing@' + base_commit
other_cycle = 'protected-shadow-other@' + base_commit
candidate_ref = d.cycle_candidate_ref(cycle)
landing_ref = d.cycle_landing_ref(cycle)
git(['update-ref', candidate_ref, candidate, '0' * 40])

candidate_state = d.empty_candidate_state()
candidate_state['cycles'][cycle] = {
    'ref': candidate_ref, 'commit': candidate}
d.write_candidate_state(candidate_state)

def save_control(control):
    state = d.empty_ticket_cycle_state()
    state['active'][cycle] = {
        'phase': 'implementation', 'commit': None, 'mode': 'normal',
        'route': 'primary', 'ticket_class': 'protected-control-plane',
        'path_scope': [tool], 'control_plane': control}
    d.write_ticket_cycle_state(state)

def landing_must_be_blocked(label):
    assert ref_or_none(landing_ref) is None, label
    try:
        d.execute_architect_go_locked(cycle, candidate, 'normal')
    except d.TicketCycleStateError:
        pass
    else:
        raise AssertionError(label + ' unexpectedly created a landing')
    assert ref_or_none(landing_ref) is None, label

# No Architect decision is a NO-GO. Neither no keys nor Architect alone can
# reach the landing primitive. Red Team acceptance alone is also insufficient.
control = d.empty_control_plane_state()
save_control(control)
landing_must_be_blocked('missing Architect and Red Team decisions')
control['architect_candidate'] = candidate
save_control(control)
landing_must_be_blocked('missing Red Team decision')
redteam_only = d.empty_control_plane_state()
redteam_only['redteam_candidate'] = candidate
redteam_only['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
assert not d.control_plane_keys_ready(redteam_only, candidate)
redteam_only_state = d.empty_ticket_cycle_state()
redteam_only_state['active'][cycle] = {
    'phase': 'implementation', 'commit': None, 'mode': 'normal',
    'route': 'primary', 'ticket_class': 'protected-control-plane',
    'path_scope': [tool], 'control_plane': redteam_only}
try:
    d.validate_ticket_cycle_state(redteam_only_state)
except d.TicketCycleStateError:
    pass
else:
    raise AssertionError('Red Team acceptance survived without Architect GO')
assert ref_or_none(landing_ref) is None
no_go = d.architect_go_request_payload(cycle, candidate, 'normal').replace(
    d.MAILBOX_DECISION_HEADER + 'GO',
    d.MAILBOX_DECISION_HEADER + 'NO-GO')
assert d._architect_go_request(no_go)[3] is not None

# Exact cycle and candidate identity are checked by D1's real state writers.
for wrong_cycle, wrong_candidate in (
        (other_cycle, candidate), (cycle, other_candidate)):
    try:
        d.record_control_plane_redteam_decision(
            wrong_cycle, wrong_candidate, 'ACCEPT-CONTROL-PLANE')
    except d.TicketCycleStateError:
        pass
    else:
        raise AssertionError('wrong Red Team identity was accepted')
try:
    d.prepare_exact_squash_landing(cycle, other_candidate, 'normal')
except d.TicketCycleStateError:
    pass
else:
    raise AssertionError('wrong candidate reached landing preparation')

# A rejection names exact C but still cannot create L.
control['redteam_candidate'] = candidate
control['redteam_result'] = 'REJECT-CONTROL-PLANE'
save_control(control)
landing_must_be_blocked('Red Team rejection')

# Both exact-C decisions survive a fresh import, which represents a daemon
# restart reading only the serialized files and private candidate ref.
control['redteam_result'] = 'ACCEPT-CONTROL-PLANE'
save_control(control)
restart_probe = '''
from ai.tools import mailbox_daemon as restarted
cycle, candidate, candidate_ref = __import__('sys').argv[1:]
control = restarted.control_plane_ticket_state(cycle, candidate)
assert restarted.control_plane_keys_ready(control, candidate)
saved = restarted.read_candidate_state()['cycles'][cycle]
assert saved == {'ref': candidate_ref, 'commit': candidate}
assert restarted.git_ref_commit(candidate_ref) == candidate
'''
restart = subprocess.run(
    [sys.executable, '-c', restart_probe, cycle, candidate, candidate_ref],
    cwd=shadow_repository, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, check=False)
assert restart.returncode == 0, restart.stdout
assert d.protected_landing_ready(cycle, candidate)

# D1 may calculate inside this disposable checkout, but only the D0-owned
# harness decides whether its real landing primitive behaved correctly.
landing, parent, returned_ref = d.prepare_exact_squash_landing(
    cycle, candidate, 'normal')
assert returned_ref == landing_ref
assert parent == base_commit
parent_row = git(['rev-list', '--parents', '-n', '1', landing]).split()
assert parent_row == [landing, base_commit]
assert ref_or_none(landing_ref) == landing
landing_tree = git(['rev-parse', landing + '^{tree}'])
expected_tree = git(['merge-tree', '--write-tree', base_commit, candidate])
assert landing_tree == expected_tree
assert git(['rev-parse', 'refs/heads/main']) == base_commit

# If main changes after L was prepared, the existing L and new main remain
# untouched and the real D1 path must request integration revalidation.
git(['update-ref', 'refs/heads/main', new_main, base_commit])
try:
    d.prepare_exact_squash_landing(cycle, candidate, 'normal')
except d.RetryableArchitectLandingError as exc:
    assert d.STALE_INTEGRATION_REVALIDATION in str(exc)
else:
    raise AssertionError('changed main reused a stale protected landing')
assert git(['rev-parse', 'refs/heads/main']) == new_main
assert ref_or_none(landing_ref) == landing
print('D0_SHADOW_SCENARIOS_PASSED')
print('CONTROL_PLANE_HEALTHY', ROLE_CONTRACT['schema_version'])
"""
        commands.append(([
            daemon.sys.executable, "-m", "py_compile",
            daemon.os.path.join(repository, "ai", "tools", "role_contract.py"),
            daemon.os.path.join(repository, "ai", "tools", "handoff_contract.py"),
            daemon.os.path.join(repository, "ai", "tools", "mailbox_daemon.py")],
            repository))
        commands.append(([daemon.sys.executable, "-c", probe], repository))
        environment = daemon.os.environ.copy()
        for name in tuple(environment):
            if name.startswith("MAILBOX_"):
                del environment[name]
        ok = True
        with open(log_path, "w", encoding="utf-8") as stream:
            for command, command_cwd in commands:
                stream.write("$ " + " ".join(command) + "\n")
                try:
                    result = daemon.subprocess.run(
                        command, cwd=command_cwd,
                        env=environment, stdout=daemon.subprocess.PIPE,
                        stderr=daemon.subprocess.STDOUT, text=True, check=False,
                        timeout=120)
                except (OSError, daemon.subprocess.TimeoutExpired) as exc:
                    stream.write(type(exc).__name__ + ": " + str(exc)
                                 + "\n")
                    stream.write("rc=not-completed\n")
                    ok = False
                    break
                stream.write(result.stdout)
                stream.write("rc=" + str(result.returncode) + "\n")
                if result.returncode != 0:
                    ok = False
                    break
            stream.flush()
            daemon.os.fsync(stream.fileno())
    after = daemon._live_control_plane_fingerprint()
    if after != before:
        raise daemon.TicketCycleStateError(
            "shadow D1 changed D0 live state or private refs")
    return ok, log_path


def record_control_plane_check(cycle_id, candidate_commit, kind, ok,
                               evidence):
    """Persist one D0 shadow or post-landing health result."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        state = daemon.read_ticket_cycle_state()
        active = state["active"].get(cycle_id)
        saved = daemon.read_candidate_state()["cycles"].get(cycle_id)
        if (active is None
                or active.get("ticket_class") != "protected-control-plane"
                or saved is None
                or saved["commit"] != candidate_commit):
            raise daemon.TicketCycleStateError(
                "control-plane check does not name exact saved candidate C")
        control = dict(active["control_plane"])
        if kind == "shadow":
            control["shadow_status"] = "PASSED" if ok else "FAILED"
            control["shadow_evidence"] = evidence
        elif kind == "health":
            control["health_status"] = (
                "HEALTHY" if ok else "CONTROL_PLANE_HEALTH_FAILED")
            control["health_evidence"] = evidence
        else:
            raise daemon.TicketCycleStateError("unknown control-plane check kind")
        state["active"][cycle_id] = dict(active, control_plane=control)
        daemon.write_ticket_cycle_state(state=state)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def execute_architect_go_locked(cycle_id, candidate_commit, mode,
                                sealed_backlog=None):
    """Land C as exact L and durably advance state before any push."""
    protected = daemon.control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit) is not None
    if protected and not daemon.protected_landing_ready(
            cycle_id=cycle_id, candidate_commit=candidate_commit):
        raise daemon.TicketCycleStateError(
            "protected landing lacks exact Architect and Red Team keys")
    if protected:
        daemon.prepare_revalidated_control_plane_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit)
    landing = daemon.recorded_landing_for_architect_go(
        cycle_id=cycle_id, mode=mode)
    if landing is None:
        landing, parent, _reference = daemon.prepare_exact_squash_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            mode=mode, sealed_backlog=sealed_backlog)
    else:
        parent = daemon._verify_prepared_landing(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing, expected_backlog=sealed_backlog)
        journaled = daemon.git_ref_commit(
            reference=daemon.cycle_landing_ref(cycle_id=cycle_id))
        if journaled is not None and journaled != landing:
            raise daemon.TicketCycleStateError(
                "durable cycle state and landing crash journal disagree")
    if protected:
        control = daemon.control_plane_ticket_state(
            cycle_id=cycle_id, candidate_commit=candidate_commit)
        if (control["integration_status"] == "REVALIDATED"
                and control["shadow_status"] != "PASSED"):
            shadow_ok, shadow_log = daemon.trusted_control_plane_check(
                commit=landing, label="integration-shadow")
            daemon.record_control_plane_check(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                kind="shadow", ok=shadow_ok, evidence=shadow_log)
            if not shadow_ok:
                raise daemon.RetryableArchitectLandingError(
                    "SHADOW_VALIDATION_FAILED for exact revalidated "
                    "integration L=" + landing + "; evidence -> "
                    + shadow_log)
    daemon.preflight_role_baseline_sync(
        target=landing, retiring_candidate=candidate_commit)
    daemon.land_prepared_commit_in_clean_user_checkout(
        landing=landing, parent=parent,
        candidate_commit=candidate_commit)
    completed_now = daemon.record_architect_commit(
        cycle_id=cycle_id, accepted_commit=landing, mode=mode)
    if mode == "normal" and not protected:
        daemon.publish_redteam_closure_request(
            cycle_id=cycle_id, landing=landing)
    return landing, completed_now


def require_architect_landing_locked(cycle_id, landing_commit,
                                     ticket_state):
    """Bind candidate C to its exact, distinct squash landing L on main."""
    if daemon.ACTIVE_TOPOLOGY is None:
        # Pure function tests do not represent a live dispatch topology.
        return None
    candidate_state = daemon.read_candidate_state()
    record = daemon.candidate_record_locked(
        cycle_id=cycle_id, ticket_state=ticket_state,
        candidate_state=candidate_state)
    if record is None:
        raise daemon.TicketCycleStateError(
            "Architect landing has no saved candidate for this cycle")
    candidate_commit = record["commit"]
    if landing_commit == candidate_commit:
        raise daemon.TicketCycleStateError(
            "daemon landing record names the Implementer candidate, not its "
            "distinct squash landing")
    if not daemon.git_commit_exists(commit=landing_commit):
        raise daemon.TicketCycleStateError(
            "daemon landing record names a missing landing commit")
    current_main = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    if current_main != landing_commit:
        raise daemon.TicketCycleStateError(
            "daemon landing record does not name the current main landing")
    parent_commit = daemon._single_commit_parent(commit=landing_commit)
    daemon._require_ancestor_or_same(
        ancestor=daemon.cycle_starting_commit(cycle_id),
        descendant=parent_commit,
        label="landing parent does not preserve the cycle base")
    expected_tree = daemon._tree_with_backlog(
        tree=daemon._exact_squash_tree(
            parent_commit=parent_commit, candidate_commit=candidate_commit),
        backlog=daemon._landing_backlog(landing_commit=landing_commit))
    landing_tree = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", landing_commit + "^{tree}"],
        label="Architect landing tree")
    if landing_tree != expected_tree:
        raise daemon.TicketCycleStateError(
            "Architect landing tree is not the exact candidate plus its "
            "sealed backlog on the landing parent")
    return candidate_commit


def _require_retirement_landing_locked(cycle_id, landing_commit,
                                       ticket_state):
    """Prove one durable L authorizes retirement of this cycle's C."""
    active = ticket_state["active"].get(cycle_id)
    completed = ticket_state["completed"].get(cycle_id)
    recorded = completed
    if recorded is None and active is not None:
        if active["phase"] == "implementation":
            raise daemon.TicketCycleStateError(
                "candidate cannot retire before its daemon landing")
        recorded = active["commit"]
    if recorded != landing_commit:
        raise daemon.TicketCycleStateError(
            "candidate retirement does not name its durable landing")
    if not daemon.git_commit_exists(commit=landing_commit):
        raise daemon.TicketCycleStateError(
            "candidate retirement landing no longer exists")
    current_main = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    daemon._require_ancestor_or_same(
        ancestor=landing_commit, descendant=current_main,
        label="current main does not preserve the retired cycle landing")


def retire_cycle_candidate_locked(cycle_id, candidate_commit,
                                  landing_commit):
    """Hand off exact clean C to L, then delete only C's ownership."""
    if daemon.ACTIVE_TOPOLOGY is None:
        return True
    ticket_state = daemon.read_ticket_cycle_state()
    daemon._require_retirement_landing_locked(
        cycle_id=cycle_id, landing_commit=landing_commit,
        ticket_state=ticket_state)
    state = daemon.read_candidate_state()
    record = state["cycles"].get(cycle_id)
    reference = daemon.cycle_candidate_ref(cycle_id=cycle_id)
    current = daemon.git_ref_commit(reference=reference)
    if record is None:
        if current is not None:
            raise daemon.TicketCycleStateError(
                "accepted cycle has an unowned candidate ref")
        return True
    if record["commit"] != candidate_commit:
        raise daemon.TicketCycleStateError(
            "refusing to retire another candidate commit")
    worktree = daemon.AGENT_CWD["opus"]
    head = daemon.worktree_head(worktree=worktree)
    if head == record["commit"]:
        if daemon._clean_worktree_status(worktree=worktree):
            # Another ticket may already be editing from C. Keep both the
            # state row and ref until that work is saved or moved; deleting
            # them here would erase the only durable authority for C.
            return False
        daemon._run_git(
            repository_root=worktree,
            arguments=["reset", "--hard", landing_commit])
        if (daemon.worktree_head(worktree=worktree) != landing_commit
                or daemon._clean_worktree_status(worktree=worktree)):
            raise daemon.TicketCycleStateError(
                "Implementer checkout did not complete exact C-to-L handoff")
        head = landing_commit
    preserved_heads = {
        item["commit"] for other_cycle, item in state["cycles"].items()
        if other_cycle != cycle_id
    }
    preserved_heads.update(
        daemon.cycle_starting_commit(other_cycle)
        for other_cycle, item in ticket_state["active"].items()
        if other_cycle != cycle_id and item["phase"] == "implementation")
    if head != landing_commit and head not in preserved_heads:
        # A concurrent Implementer turn is allowed, but only durable cycle
        # state may prove that its HEAD is not abandoned work.
        return False
    if current is not None:
        if current != record["commit"]:
            raise daemon.TicketCycleStateError(
                "candidate ref changed before retirement")
        daemon._run_git(
            repository_root=daemon.AGENT_CWD["fable"],
            arguments=["update-ref", "-d", reference, record["commit"]])
    del state["cycles"][cycle_id]
    daemon.write_candidate_state(state=state)
    return True


def retire_superseded_failed_architect_go(cycle_id, candidate_commit, mode):
    """Archive rejected GO after landing."""
    paths = daemon.glob.glob(
        daemon.os.path.join(daemon.MAILBOX, "failed", "*-to-daemon.md"))
    for path in sorted(paths, key=daemon.message_sequence):
        try:
            returned = daemon._architect_go_request(daemon.read_cycle_message(path=path))
        except (OSError, ValueError, daemon.TicketCycleStateError):
            continue
        if returned != (cycle_id, candidate_commit, mode, None):
            continue
        _destination, verified = daemon.verified_state_move(path, daemon.DONE)
        if not verified:
            print("  warning: rejected GO remains: " + daemon.os.path.basename(path))


def retire_cycle_candidate(cycle_id, candidate_commit, landing_commit, mode):
    """Retire exact C after durable GO state, preserving concurrent work."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        retired = daemon.retire_cycle_candidate_locked(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing_commit)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)
    daemon.retire_superseded_failed_architect_go(
        cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode)
    return retired


def _symbolic_worktree_branch(worktree, expected_branch, label):
    """Require one persistent role checkout to stay on its saved branch."""
    result = daemon._run_git(
        repository_root=worktree,
        arguments=["symbolic-ref", "-q", "HEAD"], check=False)
    try:
        branch = result.stdout.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(label + " branch is not UTF-8") from exc
    if result.returncode != 0 or branch != expected_branch:
        raise daemon.TicketCycleStateError(
            label + " checkout left its saved branch")


def _architect_only_sealed_backlog(worktree):
    """Return a sealed backlog when it is the Architect's only change."""
    changed = daemon._run_git(
        repository_root=worktree,
        arguments=["diff", "--name-only", "-z", "HEAD", "--", "."])
    try:
        paths = {item.decode("utf-8", errors="strict")
                 for item in changed.stdout.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "Architect changed a non-UTF-8 path") from exc
    untracked = daemon._run_git(
        repository_root=worktree,
        arguments=["ls-files", "--others", "--exclude-standard", "-z",
                   "--", "."])
    if paths != {daemon.BACKLOG_RELATIVE_PATH} or untracked.stdout:
        return None
    try:
        return daemon._validate_sealed_backlog(primary_worktree=worktree)
    except daemon.PrimaryWorktreeError as exc:
        raise daemon.TicketCycleStateError(str(exc)) from exc


def _architect_backlog_matches_target(worktree, target):
    """Return whether the sealed backlog is the only change and is in L."""
    working = daemon._architect_only_sealed_backlog(worktree=worktree)
    if working is None:
        return False
    return working == daemon._landing_backlog(landing_commit=target)


def _clear_landed_architect_backlog(worktree, target):
    """Restore old bytes before a fast-forward that contains the same edit."""
    if not daemon._architect_backlog_matches_target(
            worktree=worktree, target=target):
        raise daemon.TicketCycleStateError(
            "Architect checkout has work beyond the backlog in this landing")
    backlog = daemon.os.path.join(worktree, daemon.BACKLOG_RELATIVE_PATH)
    recovery = daemon.os.path.join(
        worktree, "ai", "notes", daemon.BACKLOG_SYNC_RECOVERY_NAME)
    if daemon.os.path.lexists(recovery):
        raise daemon.TicketCycleStateError("backlog sync recovery already exists")
    daemon.os.replace(backlog, recovery)
    result = daemon._run_git(
        repository_root=worktree,
        arguments=["restore", "--source=HEAD", "--staged", "--worktree",
                   "--", daemon.BACKLOG_RELATIVE_PATH],
        check=False)
    if result.returncode != 0 or daemon._clean_worktree_status(worktree=worktree):
        daemon.os.replace(recovery, backlog)
        raise daemon.TicketCycleStateError(
            "Architect backlog could not be prepared for baseline sync")
    return recovery


def sync_clean_role_baseline(worktree, expected_branch, target, label):
    """Fast-forward one clean role baseline to an exact landed commit."""
    daemon._symbolic_worktree_branch(
        worktree=worktree, expected_branch=expected_branch, label=label)
    recovery = None
    if daemon._clean_worktree_status(worktree=worktree):
        if label != "Architect":
            raise daemon.TicketCycleStateError(
                label + " checkout has staged, unstaged, or untracked work")
        recovery = daemon._clear_landed_architect_backlog(
            worktree=worktree, target=target)
    current = daemon.worktree_head(worktree=worktree)
    if current == target:
        return False
    daemon._require_ancestor_or_same(
        ancestor=current, descendant=target,
        label=label + " baseline is not an ancestor of the landing")
    result = daemon._run_git(
        repository_root=worktree,
        arguments=["merge", "--ff-only", target], check=False)
    if result.returncode != 0:
        if recovery is not None:
            daemon.os.replace(recovery, daemon.os.path.join(
                worktree, daemon.BACKLOG_RELATIVE_PATH))
        raise daemon.TicketCycleStateError(
            label + " baseline could not fast-forward to the landing")
    if (daemon.worktree_head(worktree=worktree) != target
            or daemon._clean_worktree_status(worktree=worktree)):
        raise daemon.TicketCycleStateError(
            label + " baseline did not advance cleanly to the landing")
    if recovery is not None:
        daemon.os.unlink(recovery)
    return True


def _role_baseline_plan_locked(target, retiring_candidate=None):
    """Preflight all role baselines without changing a checkout."""
    candidate_state = daemon.read_candidate_state()
    ticket_state = daemon.read_ticket_cycle_state()
    plan = []
    for worktree, branch, label in (
            (daemon.AGENT_CWD["fable"], daemon.AGENT_BRANCH["fable"], "Architect"),
            (daemon.AGENT_CWD["sol"], daemon.SOL_BRANCH, "Red Team")):
        daemon._symbolic_worktree_branch(
            worktree=worktree, expected_branch=branch, label=label)
        current = daemon.worktree_head(worktree=worktree)
        sealed_overlay = (
            daemon._architect_only_sealed_backlog(worktree=worktree)
            if label == "Architect" and current == target else None)
        if (daemon._clean_worktree_status(worktree=worktree)
                and sealed_overlay is None
                and not (label == "Architect"
                         and daemon._architect_backlog_matches_target(
                             worktree=worktree, target=target))):
            raise daemon.TicketCycleStateError(
                label + " checkout has work that baseline sync would touch")
        daemon._require_ancestor_or_same(
            ancestor=current, descendant=target,
            label=label + " baseline is not an ancestor of the landing")
        plan.append((worktree, branch, label, current != target))
    opus_head = daemon.worktree_head(worktree=daemon.AGENT_CWD["opus"])
    preserved = {record["commit"]
                 for record in candidate_state["cycles"].values()}
    active_bases = {
        daemon.cycle_starting_commit(cycle_id)
        for cycle_id, record in ticket_state["active"].items()
        if record["phase"] == "implementation"}
    preserved.update(active_bases)
    daemon._symbolic_worktree_branch(
        worktree=daemon.AGENT_CWD["opus"], expected_branch=daemon.IMPLEMENTER_BRANCH,
        label="Implementer")
    if opus_head == retiring_candidate:
        if daemon._clean_worktree_status(worktree=daemon.AGENT_CWD["opus"]):
            raise daemon.TicketCycleStateError(
                "Implementer candidate C has unsaved work")
        plan.append((daemon.AGENT_CWD["opus"], daemon.IMPLEMENTER_BRANCH,
                     "Implementer candidate", False))
    elif opus_head in preserved:
        plan.append((daemon.AGENT_CWD["opus"], daemon.IMPLEMENTER_BRANCH,
                     "Implementer preserved work", False))
    elif any(daemon.git_commit_descends_from(
            starting_commit=base, accepted_commit=opus_head)
            for base in active_bases):
        plan.append((daemon.AGENT_CWD["opus"], daemon.IMPLEMENTER_BRANCH,
                     "Implementer active work", False))
    elif any(daemon.git_commit_descends_from(
            starting_commit=opus_head, accepted_commit=base)
            for base in active_bases):
        plan.append((daemon.AGENT_CWD["opus"], daemon.IMPLEMENTER_BRANCH,
                     "Implementer older active base", False))
    else:
        if daemon._clean_worktree_status(worktree=daemon.AGENT_CWD["opus"]):
            raise daemon.TicketCycleStateError(
                "Implementer checkout has work that baseline sync would "
                "touch")
        daemon._require_ancestor_or_same(
            ancestor=opus_head, descendant=target,
            label="Implementer baseline is not an ancestor of the landing")
        plan.append((daemon.AGENT_CWD["opus"], daemon.IMPLEMENTER_BRANCH,
                     "Implementer", opus_head != target))
    return tuple(plan)


def preflight_role_baseline_sync(target, retiring_candidate=None):
    """Prove every role can preserve or fast-forward before main changes."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        return daemon._role_baseline_plan_locked(
            target=target, retiring_candidate=retiring_candidate)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def sync_all_clean_role_baselines(target):
    """Advance clean idle role baselines under exact ticket-state authority."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        plan = daemon._role_baseline_plan_locked(target=target)
        changed = False
        for worktree, branch, label, should_sync in plan:
            if should_sync:
                changed = (daemon.sync_clean_role_baseline(
                    worktree=worktree, expected_branch=branch,
                    target=target, label=label) or changed)
        return changed
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def _permanent_note_commit_paths(base_commit, notes_commit):
    """Return exact modified paths while refusing structural Git changes."""
    summary = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["diff", "--summary", base_commit, notes_commit,
                   "--", "."])
    if summary.stdout:
        raise daemon.TicketCycleStateError(
            "permanent-note commit changes a path mode, type, name, or "
            "existence")
    result = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["diff", "--name-only", "-z", "--diff-filter=M",
                   base_commit, notes_commit, "--", "."])
    try:
        paths = [item.decode("utf-8", errors="strict")
                 for item in result.stdout.split(b"\0") if item]
    except UnicodeDecodeError as exc:
        raise daemon.TicketCycleStateError(
            "permanent-note commit path is not UTF-8") from exc
    if (not paths
            or len(paths) != len(set(paths))
            or not set(paths).issubset(set(
                daemon.ARCHITECT_PROTECTED_POLICY_PATHS))):
        raise daemon.TicketCycleStateError(
            "protected-policy commit must modify only a role file, the "
            "protected YAML contract, or one of the exact eleven permanent "
            "notes")
    return tuple(paths)


def require_architect_notes_commit_object(base_commit, notes_commit):
    """Prove immutable B-to-P history and its exact note-only path set."""
    if (not daemon.git_commit_exists(commit=base_commit)
            or not daemon.git_commit_exists(commit=notes_commit)):
        raise daemon.TicketCycleStateError(
            "Architect notes request names a missing B or P commit")
    if daemon._single_commit_parent(commit=notes_commit) != base_commit:
        raise daemon.TicketCycleStateError(
            "Architect notes P must be exactly one commit directly on B")
    daemon._permanent_note_commit_paths(
        base_commit=base_commit, notes_commit=notes_commit)


def require_architect_notes_commit(base_commit, notes_commit,
                                   allow_landed_replay=False):
    """Prove clean one-parent B-to-P authority for a note-only landing."""
    primary = daemon.AGENT_CWD["fable"]
    daemon._symbolic_worktree_branch(
        worktree=primary, expected_branch=daemon.AGENT_BRANCH["fable"],
        label="Architect")
    if daemon._clean_worktree_status(worktree=primary):
        raise daemon.TicketCycleStateError(
            "Architect note checkout is not clean at commit P")
    if daemon.worktree_head(worktree=primary) != notes_commit:
        raise daemon.TicketCycleStateError(
            "Architect notes GO does not name primary HEAD P")
    daemon.require_architect_notes_commit_object(
        base_commit=base_commit, notes_commit=notes_commit)
    try:
        daemon._validate_protected_tracked_state(primary_worktree=primary)
    except daemon.PrimaryWorktreeError as exc:
        raise daemon.TicketCycleStateError(str(exc)) from exc
    try:
        proposed_contract = daemon._local_role_contract_tool().load_role_contract(
            daemon.os.path.join(primary, daemon.ROLE_CONTRACT_RELATIVE_PATH))
        daemon.validate_role_contract_bindings(contract=proposed_contract)
    except (OSError, RuntimeError, ValueError) as exc:
        raise daemon.TicketCycleStateError(
            "proposed role contract is invalid: " + str(exc)) from exc
    current_main = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit")
    allowed = {base_commit, notes_commit} if allow_landed_replay \
        else {base_commit}
    if current_main not in allowed:
        raise daemon.TicketCycleStateError(
            "Architect notes B is not the exact current main baseline")
    return current_main


def _require_no_ordinary_landing_transition_locked(current_dispatch_path):
    """Refuse P while ordinary durable work exists; caller holds state lock."""
    ticket_state = daemon.read_ticket_cycle_state()
    candidate_state = daemon.read_candidate_state()
    if ticket_state["active"] or candidate_state["cycles"]:
        raise daemon.TicketCycleStateError(
            "permanent notes wait until every active ticket and candidate "
            "is retired")
    refs = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["for-each-ref", "--format=%(refname)",
                   daemon.CANDIDATE_REF_ROOT])
    if refs.stdout.strip():
        raise daemon.TicketCycleStateError(
            "permanent notes wait until every candidate/landing ref is "
            "retired")
    current_key = daemon._path_key(current_dispatch_path)
    for directory in (daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight"),
                      daemon.os.path.join(daemon.MAILBOX, "failed")):
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-daemon.md")):
            if daemon._path_key(path) == current_key:
                continue
            try:
                message = daemon.read_cycle_message(path=path)
            except (OSError, ValueError, daemon.TicketCycleStateError) as exc:
                raise daemon.TicketCycleStateError(
                    "cannot verify another daemon request: " + str(exc)) \
                    from exc
            if message.startswith(
                    daemon.MAILBOX_RETURN_HEADER + "architect-go"):
                raise daemon.TicketCycleStateError(
                    "permanent notes wait for the ordinary Architect GO")


def require_no_ordinary_landing_transition(current_dispatch_path):
    """Refuse P while any ordinary ticket, C/ref, landing ref, or GO remains."""
    lock_file = daemon.acquire_ticket_cycle_lock()
    try:
        daemon._require_no_ordinary_landing_transition_locked(
            current_dispatch_path=current_dispatch_path)
    finally:
        daemon.release_ticket_cycle_lock(lock_file=lock_file)


def architect_notes_transition_pending():
    """Return whether a durable note admin turn or P landing is unresolved."""
    for directory in (daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight"),
                      daemon.os.path.join(daemon.MAILBOX, "failed")):
        for suffix, header in (
                ("*-to-fable.md", daemon.MAILBOX_ADMIN_HEADER),
                ("*-to-daemon.md",
                 daemon.MAILBOX_RETURN_HEADER + "architect-notes-go")):
            for path in daemon.glob.glob(daemon.os.path.join(directory, suffix)):
                try:
                    matches = daemon.regular_file_has_prefix(
                        path=path, prefix=header.encode("ascii"))
                except (OSError, ValueError):
                    continue
                if matches:
                    return True
    return False


def failed_architect_notes_transition_paths():
    """Return exact failed admin/P files that no watcher may retry itself."""
    failed = daemon.os.path.join(daemon.MAILBOX, "failed")
    found = []
    for suffix, header in (
            ("*-to-fable.md", daemon.MAILBOX_ADMIN_HEADER),
            ("*-to-daemon.md",
             daemon.MAILBOX_RETURN_HEADER + "architect-notes-go")):
        for path in daemon.glob.glob(daemon.os.path.join(failed, suffix)):
            try:
                matches = daemon.regular_file_has_prefix(
                    path=path, prefix=header.encode("ascii"))
            except (OSError, ValueError):
                continue
            if matches:
                found.append(path)
    return sorted(found, key=daemon.message_sequence)


def architect_notes_failed_debt_error():
    """Explain failed-only note debt as a finite user-action stop."""
    paths = daemon.failed_architect_notes_transition_paths()
    if not paths:
        return None
    relative = [daemon.os.path.relpath(path, daemon.MAILBOX) for path in paths]
    return (daemon.ARCHITECT_NOTES_DEBT_PREFIX
            + ", ".join(relative)
            + "; inspect the saved failure, correct its cause, then move "
              "only the verified exact request back to the mailbox root "
              "before restarting the watcher")


def message_belongs_to_active_cycle(path, active_cycles):
    """Return whether one root agent message advances an admitted ticket."""
    match = daemon.PENDING_MESSAGE_RE.match(daemon.os.path.basename(path))
    if match is None:
        return False
    try:
        message = daemon.read_cycle_message(path=path)
    except (OSError, ValueError, daemon.TicketCycleStateError):
        return False
    agent = match.group(1)
    if agent in {"fable", "opus"} and message.startswith(
            daemon.MAILBOX_FLOW_HEADER):
        cycle_id, _mode, _body, problem = daemon._ticket_flow_envelope(
            message=message)
        return problem is None and cycle_id in active_cycles
    if agent == "fable" and message.startswith(daemon.MAILBOX_RETURN_HEADER):
        cycle_id, _commit, result, _body, problem = (
            daemon._redteam_review_receipt(message=message))
        return (problem is None and result == "REOPEN"
                and cycle_id in active_cycles)
    if agent == "sol" and daemon.sol_ticket_kind(message=message) == "closure":
        cycle_id = daemon.redteam_closure_ticket(message=message)
        return cycle_id in active_cycles
    if agent == "sol" and daemon.sol_ticket_kind(message=message) == "control-plane":
        cycle_id, _candidate, _body, problem = (
            daemon._redteam_control_plane_envelope(message=message))
        return problem is None and cycle_id in active_cycles
    return False


def requeue_retryable_daemon_message(dispatch_path):
    """Return one valid inflight GO to root without calling it malformed."""
    _path, verified = daemon.verified_state_move(
        dispatch_path=dispatch_path, directory=daemon.MAILBOX)
    return verified


def publish_backlog_close_request(cycle_id, candidate_commit, mode):
    """Queue one exact Architect correction while preserving accepted C."""
    payload = daemon.backlog_close_request_payload(
        cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode)
    for directory in (daemon.MAILBOX, daemon.os.path.join(daemon.MAILBOX, "inflight"),
                      daemon.os.path.join(daemon.MAILBOX, "prelaunch")):
        for path in daemon.glob.glob(daemon.os.path.join(directory, "*-to-fable.md")):
            try:
                if daemon.read_cycle_message(path=path) == payload:
                    return path
            except (OSError, ValueError, daemon.TicketCycleStateError):
                continue
    if not daemon.send(agent="fable", text=payload, dry_run=False):
        raise daemon.RetryableArchitectLandingError(
            "could not publish backlog-close recovery")
    matches = [path for path in daemon.glob.glob(
        daemon.os.path.join(daemon.MAILBOX, "*-to-fable.md"))
        if daemon.read_cycle_message(path=path) == payload]
    if len(matches) != 1:
        raise daemon.RetryableArchitectLandingError(
            "backlog-close recovery was not published exactly once")
    return matches[0]


def defer_protected_stale_integration(
        dispatch_path, cycle_id, candidate_commit, mode, problem):
    """Save a moved-main event and queue its same-cycle Architect audit."""
    details = daemon.stale_integration_details(problem=problem)
    if details is None or details["candidate"] != candidate_commit:
        raise daemon.TicketCycleStateError(
            "protected stale diagnosis changed exact candidate C")
    daemon.record_control_plane_integration_stale(
        cycle_id=cycle_id, candidate_commit=candidate_commit,
        stale_landing=details["stale_landing"],
        old_main=details["old_main"], new_main=details["new_main"])
    request = daemon.publish_control_plane_integration_request(
        cycle_id=cycle_id, candidate=candidate_commit,
        stale_landing=details["stale_landing"],
        old_main=details["old_main"], new_main=details["new_main"],
        mode=mode)
    deferred = daemon.move_without_overwrite(
        path=dispatch_path,
        directory=daemon.os.path.join(daemon.MAILBOX, "integration-stale"))
    if deferred is None and daemon.os.path.lexists(dispatch_path):
        raise daemon.TicketCycleStateError(
            "stale Architect GO could not enter its durable waiting state")
    return request


def prepared_landing_reached_main(cycle_id):
    """Return whether main contains this cycle's journaled landing."""
    landing = daemon.git_ref_commit(reference=daemon.cycle_landing_ref(cycle_id=cycle_id))
    if landing is None:
        return False
    current = daemon._exact_git_object(
        arguments=["rev-parse", "--verify", "refs/heads/main^{commit}"],
        label="current main commit after landing error")
    if current == landing:
        return True
    result = daemon._run_git(
        repository_root=daemon.AGENT_CWD["fable"],
        arguments=["merge-base", "--is-ancestor", landing, current],
        check=False)
    if result.returncode not in {0, 1}:
        raise daemon.TicketCycleStateError(
            "cannot determine whether main contains the prepared landing")
    return result.returncode == 0


def finish_claimed_architect_go(dispatch_path, cycle_id,
                                candidate_commit, mode):
    """Finish or replay one already-claimed, well-formed Architect GO."""
    name = daemon.os.path.basename(dispatch_path)
    try:
        active = daemon.read_ticket_cycle_state()["active"].get(cycle_id)
        if (active is None or active["mode"] != mode
                or daemon.candidate_commit_for_cycle(cycle_id) != candidate_commit):
            raise daemon.TicketCycleStateError(
                "Architect GO changed the active cycle, mode, or candidate")
        sealed_backlog = daemon._validate_sealed_backlog(
            primary_worktree=daemon.AGENT_CWD["fable"])
        daemon.require_closed_backlog_ticket(
            ticket_anchor=daemon.cycle_ticket_anchor(cycle_id),
            sealed_backlog=sealed_backlog)
    except daemon.BacklogTicketOpenError:
        request = daemon.publish_backlog_close_request(
            cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode)
        if not daemon.archive_consumed_message(dispatch_path=dispatch_path):
            raise daemon.RetryableArchitectLandingError(
                "accepted GO could not enter backlog-close recovery")
        print("backlog closure required before landing " + candidate_commit
              + "; preserved C and the prior audit; queued "
              + daemon.os.path.basename(request) + " for bookkeeping and one "
                "fresh exact GO.")
        return True, 0, None
    except (daemon.PrimaryWorktreeError, daemon.TicketCycleStateError) as exc:
        parked = daemon.park_failed_message(dispatch_path=dispatch_path)
        state = "parked." if parked else "move failed."
        print("refused " + name + ": " + str(exc)
              + "; C and its cycle remain. Close and seal the ticket, then "
              "send a fresh GO; " + state)
        return False, 0, None
    protected = daemon.control_plane_ticket_state(
        cycle_id=cycle_id, candidate_commit=candidate_commit)
    if protected is not None:
        try:
            daemon.record_control_plane_architect_go(
                cycle_id=cycle_id, candidate_commit=candidate_commit)
            protected = daemon.control_plane_ticket_state(
                cycle_id=cycle_id, candidate_commit=candidate_commit)
            if protected["redteam_result"] is None:
                request = daemon.publish_control_plane_review_request(
                    cycle_id=cycle_id, candidate=candidate_commit)
                print("protected Architect GO(C) recorded; waiting for the "
                      "mandatory Red Team decision on exact C; request "
                      + daemon.os.path.basename(request) + ".")
                return None, 0, None
            if protected["redteam_result"] == "REJECT-CONTROL-PLANE":
                daemon.publish_control_plane_repair_request(
                    cycle_id=cycle_id, candidate=candidate_commit,
                    mode=mode)
                rejected_path = daemon.move_without_overwrite(
                    path=dispatch_path,
                    directory=daemon.os.path.join(daemon.MAILBOX, "redteam-rejected"))
                if rejected_path is None:
                    return False, 0, None
                print("protected candidate C was rejected by Red Team; C "
                      "was preserved and an Architect repair turn was "
                      "queued.")
                return True, 0, None
            if (protected["health_status"]
                    == "CONTROL_PLANE_HEALTH_FAILED"):
                raise daemon.FatalArchitectLandingError(
                    "CONTROL_PLANE_HEALTH_FAILED: D0 is recovery-only; "
                    "inspect " + protected["health_evidence"]
                    + " and preserve the recorded landing")
            if protected["shadow_status"] == "FAILED":
                print("SHADOW_VALIDATION_FAILED: D0 preserved C, both "
                      "decisions, and the evidence at "
                      + protected["shadow_evidence"] + ".")
                return None, 0, None
            if (protected["shadow_status"] != "PASSED"
                    and protected["integration_status"]
                    != "REVALIDATED"):
                shadow_ok, shadow_log = daemon.trusted_control_plane_check(
                    commit=candidate_commit, label="shadow")
                daemon.record_control_plane_check(
                    cycle_id=cycle_id, candidate_commit=candidate_commit,
                    kind="shadow", ok=shadow_ok, evidence=shadow_log)
                if not shadow_ok:
                    print("SHADOW_VALIDATION_FAILED: D0 did not create L; "
                          "evidence -> " + shadow_log)
                    return None, 0, None
        except daemon.FatalArchitectLandingError:
            raise
        except (OSError, daemon.TicketCycleStateError) as exc:
            print("protected control-plane gate stopped before landing: "
                  + str(exc) + "; C and GO remain preserved.")
            return None, 0, None
    main_lock = daemon.acquire_main_checkout_turn_lock()
    if main_lock is None:
        requeued = daemon.requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        raise daemon.FatalArchitectLandingError(
            "daemon landing lock was unavailable; "
            + ("the exact GO was returned to the mailbox root"
               if requeued else
               "the inflight GO remains preserved for recovery")
            + ". Stop the other landing process and restart.")
    try:
        landing, completed = daemon.execute_architect_go_locked(
            cycle_id=cycle_id, candidate_commit=candidate_commit, mode=mode,
            sealed_backlog=sealed_backlog)
    except daemon.RetryableArchitectLandingError as exc:
        daemon.release_main_checkout_turn_lock(lock_file=main_lock)
        if (protected is not None
                and daemon.stale_integration_details(problem=exc) is not None):
            try:
                request = daemon.defer_protected_stale_integration(
                    dispatch_path=dispatch_path, cycle_id=cycle_id,
                    candidate_commit=candidate_commit, mode=mode,
                    problem=exc)
            except (OSError, daemon.TicketCycleStateError) as recovery_exc:
                raise daemon.FatalArchitectLandingError(
                    str(exc) + "; C and both approvals remain preserved, "
                    "but D0 could not queue integration revalidation: "
                    + str(recovery_exc)) from exc
            print(daemon.STALE_INTEGRATION_REVALIDATION + ": C and both approvals "
                  "were preserved; Architect integration audit queued as "
                  + daemon.os.path.basename(request) + ".")
            return None, 0, None
        requeued = daemon.requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        preserved = ("the exact GO was returned to the mailbox root"
                     if requeued else
                     "the inflight GO remains preserved for recovery")
        if daemon.STALE_INTEGRATION_REVALIDATION in str(exc):
            remedy = (
                "Automated integration revalidation is not supported yet; "
                "keep "
                "C, L, GO, and the user's work preserved, and do not restart "
                "this cycle until that recovery is handled explicitly")
        elif "SHADOW_VALIDATION_FAILED" in str(exc):
            remedy = (
                "Keep C, both approvals, the revalidated landing, and its "
                "named evidence preserved; do not install that landing")
        elif ("durable-state recovery" in str(exc)
              or "history requires user reconciliation" in str(exc)):
            remedy = (
                "Keep C, L, GO, and the user's work preserved and inspect "
                "the named Git history before retrying")
        else:
            remedy = (
                "Make the user's unchanged-parent main checkout clean, then "
                "restart this watcher")
        raise daemon.FatalArchitectLandingError(
            str(exc) + "; " + preserved + ". " + remedy + ".") from exc
    except (OSError, daemon.PrimaryWorktreeError, daemon.TicketCycleStateError) as exc:
        try:
            landing_reached_main = daemon.prepared_landing_reached_main(
                cycle_id=cycle_id)
        except (OSError, daemon.PrimaryWorktreeError,
                daemon.TicketCycleStateError) as proof_exc:
            daemon.release_main_checkout_turn_lock(lock_file=main_lock)
            requeued = daemon.requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            raise daemon.FatalArchitectLandingError(
                "landing recovery could not determine whether main already "
                "advanced: " + str(proof_exc) + "; "
                + ("the exact GO was returned to the mailbox root"
                   if requeued else
                   "the inflight GO remains preserved for recovery")
                + ". Repair Git access, then restart.") from exc
        if landing_reached_main:
            daemon.release_main_checkout_turn_lock(lock_file=main_lock)
            requeued = daemon.requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            raise daemon.FatalArchitectLandingError(
                "main already contains the prepared landing, but its "
                "final checks did not finish: " + str(exc) + "; "
                + ("the exact GO was returned to the mailbox root"
                   if requeued else
                   "the inflight GO remains preserved for recovery")
                + ". Restart to finish the same landing.") from exc
        daemon.release_main_checkout_turn_lock(lock_file=main_lock)
        parked = daemon.park_failed_message(dispatch_path=dispatch_path)
        print("refused " + name + ": exact local landing was not accepted: "
              + str(exc) + "; "
              + ("parked in failed/." if parked else
                 "failed-state move was not verified."))
        return False, 0, None
    if protected is not None:
        try:
            health_ok, health_log = daemon.trusted_control_plane_check(
                commit=landing, label="health")
            daemon.record_control_plane_check(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                kind="health", ok=health_ok, evidence=health_log)
            if not health_ok:
                daemon.release_main_checkout_turn_lock(lock_file=main_lock)
                raise daemon.FatalArchitectLandingError(
                    "CONTROL_PLANE_HEALTH_FAILED: L is preserved at "
                    + landing + " and D0 is stopping before new work; "
                      "inspect " + health_log
                      + ". Repair with the preserved trusted controller; "
                        "do not rewrite history")
            completed = int(daemon.complete_protected_ticket_cycle(
                cycle_id=cycle_id, candidate_commit=candidate_commit,
                landing=landing))
        except daemon.FatalArchitectLandingError:
            raise
        except (OSError, daemon.TicketCycleStateError) as exc:
            daemon.release_main_checkout_turn_lock(lock_file=main_lock)
            raise daemon.FatalArchitectLandingError(
                "CONTROL_PLANE_HEALTH_FAILED: L is preserved at "
                + landing + "; D0 health state could not finish: "
                + str(exc)) from exc
    # State first, archive second. A crash in between leaves one inflight GO
    # whose exact landing, state, and closure publication replay idempotently.
    try:
        daemon.write_push_debt(
            landing=landing,
            detail="local landing recorded; remote push not yet attempted")
    except OSError as exc:
        daemon.release_main_checkout_turn_lock(lock_file=main_lock)
        requeued = daemon.requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        raise daemon.FatalArchitectLandingError(
            "local landing state is durable, but its required push-debt "
            "note could not be written: " + str(exc) + "; "
            + ("the exact GO was returned to the mailbox root"
               if requeued else
               "the inflight GO remains preserved for recovery")
            + ". Repair relay-directory writes, then restart.") from exc
    if not daemon.archive_consumed_message(dispatch_path=dispatch_path):
        daemon.release_main_checkout_turn_lock(lock_file=main_lock)
        return False, 0, landing
    try:
        daemon.retire_cycle_landing_ref(cycle_id=cycle_id, landing=landing)
    except (OSError, daemon.TicketCycleStateError) as exc:
        print("  warning: durable state and GO archive are complete, but "
              "the private landing journal remains for recovery: "
              + str(exc))
    try:
        retired = daemon.retire_cycle_candidate(
            cycle_id=cycle_id, candidate_commit=candidate_commit,
            landing_commit=landing, mode=mode)
    except (OSError, daemon.TicketCycleStateError) as exc:
        print("  warning: durable state and GO archive are complete, but "
              "the private candidate journal remains for recovery: "
              + str(exc))
        retired = False
    try:
        daemon.sync_all_clean_role_baselines(target=landing)
    except (OSError, daemon.TicketCycleStateError) as exc:
        daemon.release_main_checkout_turn_lock(lock_file=main_lock)
        raise daemon.FatalArchitectLandingError(
            "local landing is durable, but clean role baselines did not "
            "finish advancing to it: " + str(exc)
            + "; restart to replay the archived GO recovery") from exc
    daemon.release_main_checkout_turn_lock(lock_file=main_lock)
    daemon.deliver_pending_ticket_cycle_returns()
    if protected is not None:
        print("protected ticket cycle complete after exact Architect GO(C), "
              "Red Team ACCEPT(C), D0 shadow validation, and healthy L: "
              + cycle_id + ".")
    elif mode == "two-role":
        if completed:
            print("ticket cycle complete at the exact local landing: "
                  + cycle_id + ".")
        else:
            print("ticket cycle was already complete at the exact local "
                  "landing: " + cycle_id + ".")
    else:
        print("recorded exact local landing " + landing
              + " for ticket cycle " + cycle_id
              + "; its advisory Red Team review is queued.")
    try:
        pushed, detail = daemon.push_exact_landing_or_record_debt(landing=landing)
    except (OSError, ValueError) as exc:
        pushed = False
        detail = str(exc)
    if pushed:
        print("verified remote main at exact landing " + landing + ".")
    else:
        print("local landing is complete; remote push remains follow-up "
              "debt for " + landing + (": " + detail if detail else "."))
    return True, completed, landing


def finish_claimed_architect_notes_go(dispatch_path, base_commit,
                                      notes_commit, return_outcome=False):
    """Fast-forward exact note-only P, sync clean roles, archive, and push."""
    def result(consumed, outcome):
        ordinary = (consumed, notes_commit)
        return ordinary + (outcome,) if return_outcome else ordinary

    name = daemon.os.path.basename(dispatch_path)
    main_lock = daemon.acquire_main_checkout_turn_lock()
    if main_lock is None:
        daemon.requeue_retryable_daemon_message(dispatch_path=dispatch_path)
        raise daemon.FatalArchitectLandingError(
            "Architect note landing lock was unavailable; exact request "
            "was preserved for restart")
    main_advanced = False
    try:
        try:
            current_main = daemon.require_architect_notes_commit(
                base_commit=base_commit, notes_commit=notes_commit,
                allow_landed_replay=True)
        except daemon.TicketCycleStateError as exc:
            parked = daemon.park_failed_message(dispatch_path=dispatch_path)
            print("refused " + name + ": note-only landing was invalid: "
                  + str(exc) + "; "
                  + ("parked in failed/." if parked else
                     "failed-state move was not verified."))
            return result(False, daemon.DAEMON_MESSAGE_HARD_STOP)
        landed_replay = current_main == notes_commit
        if not landed_replay:
            try:
                daemon.require_no_ordinary_landing_transition(
                    current_dispatch_path=dispatch_path)
            except daemon.TicketCycleStateError as exc:
                requeued = daemon.requeue_retryable_daemon_message(
                    dispatch_path=dispatch_path)
                print("deferred " + name + ": " + str(exc) + "; "
                      + ("request returned to mailbox root"
                         if requeued else
                         "request remains preserved in inflight")
                      + ". Older admitted ticket work may continue.")
                return result(False, daemon.DAEMON_NOTE_DEFERRED)
            # Re-prove the no-ticket barrier and exact B/P immediately before
            # changing the user checkout. The main-turn lock prevents a
            # landing between these checks and the ff-only operation.
            daemon.require_no_ordinary_landing_transition(
                current_dispatch_path=dispatch_path)
            current_main = daemon.require_architect_notes_commit(
                base_commit=base_commit, notes_commit=notes_commit,
                allow_landed_replay=True)
            if current_main != base_commit:
                raise daemon.TicketCycleStateError(
                    "permanent-note B changed before its exact landing")
        daemon.preflight_role_baseline_sync(target=notes_commit)
        daemon.land_prepared_commit_in_clean_user_checkout(
            landing=notes_commit, parent=base_commit)
        main_advanced = True
        try:
            daemon.write_push_debt(
                landing=notes_commit,
                detail="local permanent-note landing recorded; remote push "
                       "not yet attempted")
        except OSError as exc:
            requeued = daemon.requeue_retryable_daemon_message(
                dispatch_path=dispatch_path)
            raise daemon.FatalArchitectLandingError(
                "permanent-note P reached main, but push debt could not be "
                "saved: " + str(exc) + "; "
                + ("request returned to mailbox root" if requeued else
                   "request remains preserved in inflight")) from exc
        daemon.sync_all_clean_role_baselines(target=notes_commit)
    except daemon.RetryableArchitectLandingError as exc:
        requeued = daemon.requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        raise daemon.FatalArchitectLandingError(
            str(exc) + "; permanent-note request "
            + ("returned to mailbox root" if requeued
               else "remains preserved in inflight")) from exc
    except daemon.FatalArchitectLandingError:
        raise
    except (OSError, daemon.TicketCycleStateError) as exc:
        requeued = daemon.requeue_retryable_daemon_message(
            dispatch_path=dispatch_path)
        phase = ("after P reached main" if main_advanced else
                 "before P changed main")
        raise daemon.FatalArchitectLandingError(
            "permanent-note landing stopped " + phase + ": " + str(exc)
            + "; " + ("request returned to mailbox root" if requeued else
                       "request remains preserved in inflight")
            + "; restart after correcting the named role baseline") from exc
    finally:
        daemon.release_main_checkout_turn_lock(lock_file=main_lock)
    try:
        receipt_raw = daemon.stable_regular_bytes(
            path=dispatch_path,
            maximum_bytes=daemon.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
            label="consumed permanent-note GO receipt")
    except (OSError, ValueError) as exc:
        raise daemon.FatalArchitectLandingError(
            "permanent-note P and role baselines are ready, but the exact "
            "GO receipt could not be reread before archive: " + str(exc)) \
            from exc
    receipt_sha256 = daemon.hashlib.sha256(receipt_raw).hexdigest()
    if not daemon.archive_consumed_message(dispatch_path=dispatch_path):
        return result(False, daemon.DAEMON_MESSAGE_HARD_STOP)
    try:
        retired_journal = daemon.retire_validated_commit_admin_journal(
            base_commit=base_commit, notes_commit=notes_commit,
            receipt_sha256=receipt_sha256)
    except (OSError, daemon.TicketCycleStateError) as exc:
        raise daemon.FatalArchitectLandingError(
            "permanent-note GO is archived and P baselines are verified, "
            "but its validated admin journal remains: " + str(exc)) \
            from exc
    if retired_journal:
        print("retired validated permanent-note admin journal after exact "
              "P receipt consumption.")
    try:
        pushed, detail = daemon.push_exact_landing_or_record_debt(
            landing=notes_commit)
    except (OSError, ValueError) as exc:
        pushed, detail = False, str(exc)
    if pushed:
        print("verified remote main at permanent-note commit "
              + notes_commit + ".")
    else:
        print("permanent-note landing is complete; remote push remains "
              "follow-up debt for " + notes_commit
              + (": " + detail if detail else "."))
    return result(True, daemon.DAEMON_MESSAGE_CONSUMED)

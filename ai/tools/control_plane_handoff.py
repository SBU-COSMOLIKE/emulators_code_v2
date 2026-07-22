"""Copy D0 state and test whether a proposed D1 can inherit it.

D0 is the watcher program currently in charge; D1 is a proposed
replacement watcher. The trusted controller imports this module from its
own checkout.  The module copies live records into a temporary Git
checkout and supplies a D0-owned Python probe.  Candidate tests never
decide whether the takeover succeeds.
"""

import glob
import hashlib
import json
import os


def _copy_regular(controller, source, destination, repository, maximum,
                  label):
  """Copy one bounded regular file without following a D1 redirect."""
  try:
    raw = controller.stable_regular_bytes(
        path=source, maximum_bytes=maximum, label=label, missing_ok=True)
  except (OSError, ValueError) as exc:
    raise controller.TicketCycleStateError(str(exc)) from exc
  if raw is None:
    return None
  repository = os.path.realpath(repository)
  parent = os.path.dirname(destination)
  if os.path.commonpath((repository, os.path.realpath(parent))) != repository:
    raise controller.TicketCycleStateError(
        "D1 redirected a disposable saved-state destination")
  os.makedirs(parent, exist_ok=True)
  if (os.path.commonpath((repository, os.path.realpath(parent)))
      != repository
      or (os.path.lexists(destination)
          and (os.path.islink(destination)
               or not os.path.isfile(destination)))):
    raise controller.TicketCycleStateError(
        "D1 redirected a disposable saved-state destination")
  with open(destination, "wb") as stream:
    stream.write(raw)
  return hashlib.sha256(raw).hexdigest()


def copy_d0_state(controller, repository):
  """Copy inheritable D0 records and return their trusted interpretation.

  Arguments:
    controller = the already imported, currently trusted daemon module.
    repository = the disposable D1 checkout that receives exact copies.

  Returns:
    A JSON-safe description of D0's schemas, workflow meaning, file digests,
    saved worktree identities, and private Git refs.
  """
  files = {}

  def copy(source, relative, maximum, label):
    files[relative] = _copy_regular(
        controller=controller, source=source,
        destination=os.path.join(repository, relative),
        repository=repository, maximum=maximum, label=label)

  copy(controller.ticket_cycle_state_path(),
       "ai/notes/mailbox/" + controller.TICKET_CYCLE_STATE_NAME,
       controller.MAX_TICKET_CYCLE_STATE_BYTES, "D0 ticket-cycle state")
  copy(controller.candidate_state_path(),
       "ai/notes/mailbox/" + controller.CANDIDATE_STATE_NAME,
       controller.MAX_CANDIDATE_STATE_BYTES, "D0 candidate state")
  notes = os.path.join(controller.WORKTREE, "ai", "notes")
  copy(os.path.join(notes, "backlog.md"), "ai/notes/backlog.md",
       controller.MAX_BACKLOG_LEDGER_BYTES, "D0 backlog")
  copy(os.path.join(notes, controller.BACKLOG_GUARD_STATE_NAME),
       "ai/notes/" + controller.BACKLOG_GUARD_STATE_NAME,
       controller.MAX_BACKLOG_GUARD_STATE_BYTES, "D0 backlog guard")
  copy(os.path.join(notes, controller.BACKLOG_SYNC_RECOVERY_NAME),
       "ai/notes/" + controller.BACKLOG_SYNC_RECOVERY_NAME,
       controller.MAX_BACKLOG_LEDGER_BYTES, "D0 backlog recovery")

  topology = {}
  if controller.ACTIVE_TOPOLOGY is not None:
    for role, key in (("architect", "primary_state"),
                      ("implementer", "implementer_state"),
                      ("red_team", "sol_state")):
      relative = ".d0-state/topology/" + role + ".json"
      source = controller.ACTIVE_TOPOLOGY[key]
      copy(source, relative, controller.MAX_PRIMARY_STATE_BYTES,
           "D0 " + role + " worktree state")
      topology[role] = controller.load_primary_state(path=source)

  for pattern in (".pending-notes-admin-*.json",
                  "pending-main-push-*.txt"):
    for source in sorted(glob.glob(
        os.path.join(controller.RELAY_DIR, pattern))):
      name = os.path.basename(source)
      if "/" in name or "\\" in name or not name.isprintable():
        raise controller.TicketCycleStateError(
            "D0 relay state has an unsafe filename")
      copy(source, "ai/notes/relay/" + name,
           controller.MAX_PRIMARY_ARCHIVE_FILE_BYTES,
           "D0 relay recovery state")

  refs = {}
  result = controller._run_git(
      repository_root=controller.AGENT_CWD["fable"],
      arguments=["for-each-ref", "--format=%(refname) %(objectname)",
                 controller.CANDIDATE_REF_ROOT, "refs/heads/main"])
  for line in result.stdout.decode("ascii", errors="strict").splitlines():
    reference, commit = line.split(" ", 1)
    if (reference != "refs/heads/main"
        and not reference.startswith(controller.CANDIDATE_REF_ROOT + "/")):
      raise controller.TicketCycleStateError(
          "D0 returned an unexpected Git ref")
    if controller.FULL_COMMIT_RE.fullmatch(commit) is None:
      raise controller.TicketCycleStateError(
          "D0 returned an invalid Git commit")
    refs[reference] = commit

  return {
      "schemas": {
          "ticket_cycle": controller.TICKET_CYCLE_STATE_SCHEMA,
          "candidate": controller.CANDIDATE_STATE_SCHEMA,
          "primary_worktree": controller.PRIMARY_STATE_SCHEMA,
          "implementer_worktree": controller.IMPLEMENTER_STATE_SCHEMA,
          "red_team_worktree": controller.SOL_STATE_SCHEMA,
          "notes_admin": controller.ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA,
      },
      "preserved_invariants": list(
          controller.CONTROL_PLANE_PRESERVED_INVARIANTS),
      "migration_path": controller.CONTROL_PLANE_MIGRATION_PATH,
      "ticket_state": controller.read_ticket_cycle_state(),
      "candidate_state": controller.read_candidate_state(),
      "topology": topology,
      "files": files,
      "refs": refs,
  }


# This program is created by D0 and executed with D1 on sys.path. It permits
# one declared migration only inside the disposable checkout, then launches a
# fresh D1 process and compares D1's interpretation with D0's saved meaning.
TAKEOVER_PROBE = r"""
import hashlib
import json
import os
import subprocess
import sys

expected_path = sys.argv[1]
with open(expected_path, encoding='utf-8') as stream:
    expected = json.load(stream)

from ai.tools import mailbox_daemon as d

candidate_schemas = {
    'ticket_cycle': d.TICKET_CYCLE_STATE_SCHEMA,
    'candidate': d.CANDIDATE_STATE_SCHEMA,
    'primary_worktree': d.PRIMARY_STATE_SCHEMA,
    'implementer_worktree': d.IMPLEMENTER_STATE_SCHEMA,
    'red_team_worktree': d.SOL_STATE_SCHEMA,
    'notes_admin': d.ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA,
}
migrated = candidate_schemas != expected['schemas']
if migrated:
    declaration_path = os.path.join(
        os.getcwd(), expected['migration_path'])
    try:
        with open(declaration_path, encoding='utf-8') as stream:
            declaration = json.load(stream)
    except (OSError, ValueError) as exc:
        raise AssertionError(
            'state schema changed without an explicit migration declaration') \
            from exc
    if (not isinstance(declaration, dict)
            or set(declaration) != {'state_migration'}):
        raise AssertionError('state migration declaration has wrong keys')
    migration = declaration['state_migration']
    if (not isinstance(migration, dict)
            or set(migration) != {
                'from_schema', 'to_schema', 'preserved_invariants',
                'function'}
            or migration['from_schema'] != expected['schemas']
            or migration['to_schema'] != candidate_schemas
            or migration['preserved_invariants']
            != expected['preserved_invariants']
            or not isinstance(migration['function'], str)
            or not migration['function'].isidentifier()):
        raise AssertionError('state migration declaration is not exact')
    migrate = getattr(d, migration['function'], None)
    if not callable(migrate):
        raise AssertionError('declared state migration function is missing')
    migrate(os.getcwd(), d.ticket_cycle_state_path(),
            d.candidate_state_path())

actual_path = os.path.join(os.path.dirname(expected_path), 'd1-actual.json')
reader = r'''
import hashlib
import json
import os
import sys
from ai.tools import mailbox_daemon as d

expected_path, actual_path = sys.argv[1:]
with open(expected_path, encoding='utf-8') as stream:
    expected = json.load(stream)

files = {}
for relative in expected['files']:
    path = os.path.join(os.getcwd(), relative)
    if not os.path.exists(path):
        files[relative] = None
    else:
        with open(path, 'rb') as stream:
            files[relative] = hashlib.sha256(stream.read()).hexdigest()

topology = {}
for role in expected['topology']:
    topology[role] = d.load_primary_state(
        os.path.join(os.getcwd(), '.d0-state', 'topology', role + '.json'))

actual = {
    'schemas': {
        'ticket_cycle': d.TICKET_CYCLE_STATE_SCHEMA,
        'candidate': d.CANDIDATE_STATE_SCHEMA,
        'primary_worktree': d.PRIMARY_STATE_SCHEMA,
        'implementer_worktree': d.IMPLEMENTER_STATE_SCHEMA,
        'red_team_worktree': d.SOL_STATE_SCHEMA,
        'notes_admin': d.ARCHITECT_NOTES_ADMIN_JOURNAL_SCHEMA,
    },
    'ticket_state': d.read_ticket_cycle_state(),
    'candidate_state': d.read_candidate_state(),
    'topology': topology,
    'files': files,
    'refs': {name: d.git_ref_commit(name) for name in expected['refs']},
}
with open(actual_path, 'w', encoding='utf-8') as stream:
    json.dump(actual, stream, sort_keys=True, indent=2)
    stream.write('\n')
'''
result = subprocess.run(
    [sys.executable, '-c', reader, expected_path, actual_path],
    cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, check=False)
assert result.returncode == 0, result.stdout
with open(actual_path, encoding='utf-8') as stream:
    actual = json.load(stream)

assert actual['schemas'] == candidate_schemas
assert actual['topology'] == expected['topology']
assert actual['refs'] == expected['refs']
assert actual['ticket_state']['schema'] == candidate_schemas['ticket_cycle']
assert actual['candidate_state']['schema'] == candidate_schemas['candidate']
assert {key: value for key, value in actual['ticket_state'].items()
        if key != 'schema'} == {
            key: value for key, value in expected['ticket_state'].items()
            if key != 'schema'}
assert actual['candidate_state']['cycles'] == \
    expected['candidate_state']['cycles']
for relative, digest in expected['files'].items():
    if relative.endswith(d.TICKET_CYCLE_STATE_NAME) \
            or relative.endswith(d.CANDIDATE_STATE_NAME):
        if not migrated:
            assert actual['files'][relative] == digest
    else:
        assert actual['files'][relative] == digest
print('D0_TO_D1_STATE_HANDOFF_PASSED')
"""

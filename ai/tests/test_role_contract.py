"""Keep the protected role contract strict and bound to live controls."""

import ast
import copy
from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

from ai.tools import mailbox_daemon
from ai.tools.role_contract import (
    ROLE_CONTRACT,
    ROLE_CONTRACT_PATH,
    RoleContractError,
    load_role_contract,
    validate_role_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_contract(value):
    """Write one JSON-compatible YAML fixture and return its path context."""
    directory = tempfile.TemporaryDirectory(prefix="role-contract-")
    path = Path(directory.name) / "role-contract.yaml"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return directory, path


class RoleContractTests(unittest.TestCase):
    """Reject schema drift before a watcher can use altered authority."""

    def test_canonical_file_is_valid_and_uses_one_policy_review(self):
        contract_path = Path(ROLE_CONTRACT_PATH)
        if not contract_path.is_absolute():
            contract_path = REPO_ROOT / contract_path
        self.assertEqual(
            contract_path.resolve(),
            (REPO_ROOT / "ai/notes/role-contract.yaml").resolve())
        loaded = load_role_contract()
        self.assertEqual(loaded, ROLE_CONTRACT)
        validate_role_contract(loaded)
        self.assertEqual(
            loaded["evidence"]["protected_policy_review_rounds"], 1)
        self.assertEqual(loaded["worktrees"], {
            "architect_branch": "refs/heads/claude/mailbox-primary",
            "architect_name": "mailbox-primary",
            "claude_branch_prefix": "claude/",
            "cleanup_action": "--clean-all",
            "implementer_branch": "refs/heads/claude/mailbox-implementer",
            "implementer_name": "mailbox-implementer",
            "legacy_cleanup_prefix": "worktree-agent-",
            "sol_branch": "refs/heads/codex/mailbox-sol",
            "sol_branch_prefix": "codex/",
            "sol_name": "mailbox-sol",
            "topology": "separate-role-worktrees-v1",
        })
        self.assertEqual(
            loaded["protected_paths"]["role_files"],
            [".claude/FABLE_ROLE.md", ".claude/OPUS_ROLE.md",
             ".codex/REDTEAM_ROLE.md"])
        self.assertEqual(
            loaded["protected_paths"]["protected_reference_files"],
            ["ai/notes/implementer-failure-modes.yaml"])
        self.assertEqual(loaded["limits"], {
            "protected_policy_file_bytes": 4 * 1024 * 1024,
            "role_contract_bytes": 64 * 1024,
        })
        protected = loaded["protected_paths"]
        named_tools = (set(protected["guard_files"].values())
                       | set(protected["trusted_tools"].values()))
        shipped_tools = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "ai/tools").glob("*.py")}
        self.assertEqual(named_tools, shipped_tools)
        mailbox_daemon.validate_role_contract_bindings(loaded)

    def test_failure_catalog_references_current_code_and_configuration(self):
        """Keep reference-only catalog links useful without granting authority."""
        catalog_path = REPO_ROOT / "ai/notes/implementer-failure-modes.yaml"
        catalog = catalog_path.read_text(encoding="utf-8")
        identifiers = re.findall(
            r"^  - id: ([a-z][a-z0-9_]*)$", catalog, re.MULTILINE)
        self.assertTrue(identifiers)
        self.assertEqual(len(identifiers), len(set(identifiers)))

        references = re.findall(
            r"^      - (ai/tools/[A-Za-z0-9_./-]+\.py)::"
            r"([A-Za-z_][A-Za-z0-9_]*)$", catalog, re.MULTILINE)
        self.assertTrue(references)
        for relative, symbol in references:
            with self.subTest(reference=relative + "::" + symbol):
                path = REPO_ROOT / relative
                self.assertTrue(path.is_file())
                tree = ast.parse(path.read_text(encoding="utf-8"))
                names = {node.name for node in tree.body
                         if isinstance(
                             node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))}
                self.assertIn(symbol, names)

        interval_reference = (
            "configuration: ai/notes/role-contract.yaml::"
            "runtime.implementer_review_minutes")
        self.assertIn(interval_reference, catalog)
        timed_section = catalog.split(
            "  - id: timed_complexity\n", 1)[1].split("\n  - id:", 1)[0]
        self.assertNotIn("90 minutes", timed_section)
        self.assertIsInstance(
            ROLE_CONTRACT["runtime"]["implementer_review_minutes"], int)

    def test_duplicate_key_refuses(self):
        key = next(iter(ROLE_CONTRACT))
        canonical = json.dumps(ROLE_CONTRACT)
        duplicate = (
            "{" + json.dumps(key) + ":" + json.dumps(ROLE_CONTRACT[key])
            + "," + canonical[1:])
        with tempfile.TemporaryDirectory(prefix="role-contract-") as tmp:
            path = Path(tmp) / "role-contract.yaml"
            path.write_text(duplicate + "\n", encoding="utf-8")
            with self.assertRaises(RoleContractError):
                load_role_contract(path)

    def test_unknown_missing_and_wrong_type_refuse(self):
        cases = {}

        unknown = copy.deepcopy(ROLE_CONTRACT)
        unknown["roles"]["architect"]["unexpected_policy"] = True
        cases["unknown key"] = unknown

        missing = copy.deepcopy(ROLE_CONTRACT)
        del missing["candidate"]["full_hash_required"]
        cases["missing key"] = missing

        wrong_type = copy.deepcopy(ROLE_CONTRACT)
        wrong_type["runtime"]["implementer_review_minutes"] = "90"
        cases["wrong type"] = wrong_type

        for label, value in cases.items():
            with self.subTest(case=label):
                directory, path = write_contract(value)
                try:
                    with self.assertRaises(RoleContractError):
                        load_role_contract(path)
                finally:
                    directory.cleanup()

    def test_yaml_may_tighten_but_not_expand_the_bootstrap_read_cap(self):
        too_small = copy.deepcopy(ROLE_CONTRACT)
        too_small["limits"]["role_contract_bytes"] = 1
        directory, path = write_contract(too_small)
        try:
            with self.assertRaisesRegex(RoleContractError, "configured limit"):
                load_role_contract(path)
        finally:
            directory.cleanup()

        too_large = copy.deepcopy(ROLE_CONTRACT)
        too_large["limits"]["role_contract_bytes"] = 64 * 1024 + 1
        with self.assertRaisesRegex(RoleContractError, "reader cap"):
            validate_role_contract(too_large)

    def test_daemon_reads_configurable_values_from_yaml(self):
        self.assertEqual(
            mailbox_daemon.IMPLEMENTER_REVIEW_MINUTES,
            ROLE_CONTRACT["runtime"]["implementer_review_minutes"])
        self.assertEqual(
            mailbox_daemon.DEFAULT_DISPATCH_TIMEOUT_MINUTES,
            ROLE_CONTRACT["runtime"]["dispatch_timeout_default_minutes"])
        self.assertEqual(
            mailbox_daemon.DEFAULT_REVIEW_EFFORT,
            ROLE_CONTRACT["runtime"]["routine_review_effort"])
        self.assertEqual(
            mailbox_daemon.ARCHITECT_CANDIDATE_FORBIDDEN_PREFIXES,
            tuple(ROLE_CONTRACT["protected_paths"][
                "candidate_forbidden_prefixes"]))
        self.assertEqual(
            mailbox_daemon.PRIMARY_WORKTREE_NAME,
            ROLE_CONTRACT["worktrees"]["architect_name"])

    def test_value_only_policy_updates_need_no_python_mirror(self):
        cases = (
            ("implementer_review_minutes", 91),
            ("dispatch_timeout_default_minutes", 121),
            ("routine_review_effort", "low"),
        )
        for name, value in cases:
            with self.subTest(control=name):
                drifted = copy.deepcopy(ROLE_CONTRACT)
                drifted["runtime"][name] = value
                self.assertIs(
                    mailbox_daemon.validate_role_contract_bindings(drifted),
                    drifted)

        extended = copy.deepcopy(ROLE_CONTRACT)
        extended["protected_paths"][
            "candidate_forbidden_prefixes"].append(".ai-secrets/")
        mailbox_daemon.validate_role_contract_bindings(extended)
        self.assertEqual(
            mailbox_daemon.candidate_forbidden_paths(
                {".ai-secrets/key", "emulator/model.py"},
                contract=extended),
            {".ai-secrets/key"})

    def test_fresh_daemon_import_uses_yaml_only_prefix_update(self):
        with tempfile.TemporaryDirectory(prefix="role-policy-import-") as tmp:
            root = Path(tmp)
            tools = root / "ai/tools"
            notes = root / "ai/notes"
            tools.mkdir(parents=True)
            notes.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "ai/tools/mailbox_daemon.py",
                tools / "mailbox_daemon.py")
            shutil.copy2(
                REPO_ROOT / "ai/tools/role_contract.py",
                tools / "role_contract.py")
            shutil.copy2(
                REPO_ROOT / "ai/tools/reopen_transition.py",
                tools / "reopen_transition.py")
            shutil.copy2(
                REPO_ROOT / "ai/tools/provider_health.py",
                tools / "provider_health.py")
            shutil.copy2(
                REPO_ROOT / "ai/tools/candidate_admission.py",
                tools / "candidate_admission.py")
            shutil.copy2(
                REPO_ROOT / "ai/tools/review_dispatch.py",
                tools / "review_dispatch.py")
            extended = copy.deepcopy(ROLE_CONTRACT)
            extended["protected_paths"][
                "candidate_forbidden_prefixes"].append(".ai-secrets/")
            (notes / "role-contract.yaml").write_text(
                json.dumps(extended, sort_keys=True, indent=2) + "\n",
                encoding="utf-8")

            spec = importlib.util.spec_from_file_location(
                "isolated_mailbox_daemon", tools / "mailbox_daemon.py")
            isolated = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(isolated)
            self.assertEqual(
                isolated.candidate_forbidden_paths(
                    {".ai-secrets/key", "emulator/model.py"}),
                {".ai-secrets/key"})

    def test_daemon_refuses_every_worktree_identity_drift(self):
        cases = {
            "name": ("architect_name", "another-primary"),
            "branch": ("implementer_branch",
                       "refs/heads/claude/another-implementer"),
            "topology": ("topology", "another-topology"),
            "prefix": ("sol_branch_prefix", "worktree-agent-"),
        }
        for label, (name, replacement) in cases.items():
            with self.subTest(control=label):
                drifted = copy.deepcopy(ROLE_CONTRACT)
                drifted["worktrees"][name] = replacement
                with self.assertRaisesRegex(
                        RoleContractError, "saved-state migration"):
                    mailbox_daemon.validate_role_contract_bindings(drifted)

    def test_contract_cannot_redirect_or_remove_bootstrap_protection(self):
        cases = []
        role = copy.deepcopy(ROLE_CONTRACT)
        role["protected_paths"]["role_files"][0] = ".claude/OTHER.md"
        cases.append(("role file", role))
        guard = copy.deepcopy(ROLE_CONTRACT)
        guard["protected_paths"]["guard_files"][
            "role_contract_reader"] = "ai/tools/other_reader.py"
        cases.append(("guard file", guard))
        trusted = copy.deepcopy(ROLE_CONTRACT)
        trusted["protected_paths"]["trusted_tools"][
            "mailbox_daemon"] = "ai/tools/other_daemon.py"
        cases.append(("trusted tool", trusted))
        note = copy.deepcopy(ROLE_CONTRACT)
        note["protected_paths"]["permanent_notes"].pop()
        cases.append(("permanent note", note))
        reference = copy.deepcopy(ROLE_CONTRACT)
        reference["protected_paths"]["protected_reference_files"].pop()
        cases.append(("protected reference", reference))
        forbidden_file = copy.deepcopy(ROLE_CONTRACT)
        forbidden_file["protected_paths"][
            "candidate_forbidden_files"].remove(".gitignore")
        cases.append(("Git-control file", forbidden_file))
        forbidden_prefix = copy.deepcopy(ROLE_CONTRACT)
        forbidden_prefix["protected_paths"][
            "candidate_forbidden_prefixes"].remove(".claude/")
        cases.append(("control prefix", forbidden_prefix))
        tools_prefix = copy.deepcopy(ROLE_CONTRACT)
        tools_prefix["protected_paths"][
            "candidate_forbidden_prefixes"].remove("ai/tools/")
        cases.append(("external tool prefix", tools_prefix))
        backlog = copy.deepcopy(ROLE_CONTRACT)
        backlog["backlog"]["path"] = "ai/notes/another-backlog.md"
        cases.append(("backlog path", backlog))

        for label, drifted in cases:
            with self.subTest(control=label):
                with self.assertRaises(RoleContractError):
                    validate_role_contract(drifted)

    def test_daemon_refuses_safety_floor_relaxations(self):
        cases = []
        force = copy.deepcopy(ROLE_CONTRACT)
        force["landing"]["force_push_allowed"] = True
        cases.append(("force push", force))
        mutable = copy.deepcopy(ROLE_CONTRACT)
        mutable["candidate"]["immutable"] = False
        cases.append(("mutable candidate", mutable))
        implementer_decides = copy.deepcopy(ROLE_CONTRACT)
        implementer_decides["roles"]["implementer"]["may_decide"] = True
        cases.append(("Implementer decides", implementer_decides))
        wrong_editor = copy.deepcopy(ROLE_CONTRACT)
        wrong_editor["backlog"]["editor"] = "implementer"
        cases.append(("Implementer edits backlog", wrong_editor))

        for label, drifted in cases:
            with self.subTest(control=label):
                with self.assertRaises(RoleContractError):
                    mailbox_daemon.validate_role_contract_bindings(drifted)

    def test_failure_mode_catalog_is_forbidden_to_implementer(self):
        self.assertIn(
            "ai/notes/implementer-failure-modes.yaml",
            mailbox_daemon.candidate_forbidden_files_from_contract(
                ROLE_CONTRACT))

    def test_changed_contract_requires_a_fresh_process(self):
        drifted = copy.deepcopy(ROLE_CONTRACT)
        drifted["runtime"]["implementer_review_minutes"] = 91
        with mock.patch.object(
                mailbox_daemon._ROLE_CONTRACT_TOOL, "load_role_contract",
                return_value=drifted):
            self.assertIn(
                "restart",
                mailbox_daemon.role_contract_snapshot_problem())
            with self.assertRaisesRegex(RoleContractError, "restart"):
                mailbox_daemon.validate_role_contract_bindings()

    def test_watcher_finishes_policy_landing_then_requests_restart(self):
        with mock.patch.object(
                mailbox_daemon, "role_contract_snapshot_problem",
                return_value="role contract changed; restart"), \
                mock.patch.object(
                    mailbox_daemon, "architect_notes_transition_pending",
                    side_effect=(True, False)):
            self.assertIsNone(mailbox_daemon.role_contract_exit_status())
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    mailbox_daemon.role_contract_exit_status(), 0)
        self.assertIn("role contract changed; restart", output.getvalue())

    def test_invalid_contract_stops_even_while_policy_landing_is_pending(self):
        with mock.patch.object(
                mailbox_daemon, "role_contract_snapshot_problem",
                return_value="role contract on disk is invalid: bad"), \
                mock.patch.object(
                    mailbox_daemon, "architect_notes_transition_pending",
                    return_value=True):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    mailbox_daemon.role_contract_exit_status(), 1)
        self.assertIn("invalid", output.getvalue())

    def test_policy_landing_does_not_release_a_role_in_the_same_pass(self):
        daemon_path = "/tmp/0001-to-daemon.md"
        second_daemon = "/tmp/0002-to-daemon.md"
        architect_path = "/tmp/0003-to-fable.md"
        with mock.patch.object(
                mailbox_daemon, "read_ticket_cycle_state",
                return_value=mailbox_daemon.empty_ticket_cycle_state()), \
                mock.patch.object(
                    mailbox_daemon, "pending_messages",
                    return_value=[daemon_path, second_daemon,
                                  architect_path]), \
                mock.patch.object(
                    mailbox_daemon, "message_is_enabled_for_topology",
                    return_value=True), \
                mock.patch.object(
                    mailbox_daemon, "regular_file_has_prefix",
                    return_value=True), \
                mock.patch.object(
                    mailbox_daemon, "consume_daemon_message",
                    return_value=mailbox_daemon.DAEMON_MESSAGE_CONSUMED) \
                as consume, \
                mock.patch.object(
                    mailbox_daemon, "role_contract_snapshot_problem",
                    return_value="role contract changed; restart"), \
                mock.patch.object(
                    mailbox_daemon, "architect_notes_transition_pending",
                    return_value=True), \
                mock.patch.object(mailbox_daemon, "drain_lane") as drain:
            self.assertEqual(
                mailbox_daemon.process_backlog(dry_run=False),
                mailbox_daemon.ROLE_CONTRACT_RESTART_REQUIRED)
        consume.assert_called_once()
        drain.assert_not_called()

    def test_new_policy_landing_stops_before_the_next_daemon_message(self):
        first_daemon = "/tmp/0001-to-daemon.md"
        second_daemon = "/tmp/0002-to-daemon.md"
        architect_path = "/tmp/0003-to-fable.md"
        with mock.patch.object(
                mailbox_daemon, "read_ticket_cycle_state",
                return_value=mailbox_daemon.empty_ticket_cycle_state()), \
                mock.patch.object(
                    mailbox_daemon, "pending_messages",
                    return_value=[first_daemon, second_daemon,
                                  architect_path]), \
                mock.patch.object(
                    mailbox_daemon, "message_is_enabled_for_topology",
                    return_value=True), \
                mock.patch.object(
                    mailbox_daemon, "consume_daemon_message",
                    return_value=mailbox_daemon.DAEMON_MESSAGE_CONSUMED) \
                as consume, \
                mock.patch.object(
                    mailbox_daemon, "role_contract_snapshot_problem",
                    side_effect=(None, "role contract changed; restart")), \
                mock.patch.object(mailbox_daemon, "drain_lane") as drain:
            self.assertEqual(
                mailbox_daemon.process_backlog(dry_run=False),
                mailbox_daemon.ROLE_CONTRACT_RESTART_REQUIRED)
        consume.assert_called_once_with(
            path=first_daemon, dry_run=False, return_outcome=True)
        drain.assert_not_called()

    def test_live_transport_must_remain_candidate_protected(self):
        with mock.patch.object(
                mailbox_daemon, "MAILBOX",
                str(REPO_ROOT / "unprotected-mailbox")):
            with self.assertRaises(RoleContractError):
                mailbox_daemon.validate_role_contract_bindings(
                    copy.deepcopy(ROLE_CONTRACT))

    def test_candidate_cannot_change_role_or_git_control_files(self):
        changed = {
            ".claude/settings.json", ".codex/new-rule.md", ".gitattributes",
            ".gitignore", ".gitmodules", "CLAUDE.md",
            "ai/notes/backlog.md", "ai/notes/.backlog-guard.json",
            "ai/notes/mailbox/0001-to-fable.md",
            "ai/notes/relay/dispatch.log", "emulator/model.py"}
        self.assertEqual(mailbox_daemon.candidate_forbidden_paths(changed),
                         changed - {"emulator/model.py"})


if __name__ == "__main__":
    unittest.main()

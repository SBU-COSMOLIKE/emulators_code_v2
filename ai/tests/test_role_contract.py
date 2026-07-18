"""Keep the protected role contract strict and bound to live controls."""

import copy
import json
from pathlib import Path
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

    def test_daemon_refuses_timing_drift(self):
        cases = (
            ("implementer_review_minutes", 91),
            ("dispatch_timeout_default_minutes", 121),
        )
        for name, value in cases:
            with self.subTest(control=name):
                drifted = copy.deepcopy(ROLE_CONTRACT)
                drifted["runtime"][name] = value
                with self.assertRaises(RoleContractError):
                    mailbox_daemon.validate_role_contract_bindings(drifted)

    def test_daemon_refuses_worktree_namespace_drift(self):
        drifted = copy.deepcopy(ROLE_CONTRACT)
        drifted["worktrees"]["sol_branch_prefix"] = "worktree-agent-"
        with self.assertRaises(RoleContractError):
            mailbox_daemon.validate_role_contract_bindings(drifted)

    def test_daemon_refuses_protected_path_drift(self):
        cases = (
            ("role_files", [".claude/FABLE_ROLE.md"]),
            ("candidate_forbidden_files", [".gitignore"]),
            ("candidate_forbidden_prefixes", [".claude/"]),
        )
        for name, value in cases:
            with self.subTest(control=name):
                drifted = copy.deepcopy(ROLE_CONTRACT)
                drifted["protected_paths"][name] = value
                with self.assertRaises(RoleContractError):
                    mailbox_daemon.validate_role_contract_bindings(drifted)

    def test_daemon_refuses_live_enforcement_drift(self):
        cases = (
            ("ARCHITECT_CANDIDATE_FORBIDDEN_FILES", frozenset()),
            ("ARCHITECT_CANDIDATE_FORBIDDEN_PREFIXES", (".claude/",)),
            ("ARCHITECT_TRUSTED_TOOL_PATHS", ("ai/tools/mailbox_daemon.py",)),
            ("AI_BRANCH_PREFIXES", ("refs/heads/claude/",)),
            ("CLEANUP_ACTION", "--erase-all"),
            ("BACKLOG_LEDGER", str(REPO_ROOT / "wrong-backlog.md")),
            ("MAILBOX", str(REPO_ROOT / "wrong-mailbox")),
            ("RELAY_DIR", str(REPO_ROOT / "wrong-relay")),
        )
        for name, value in cases:
            with self.subTest(control=name), mock.patch.object(
                    mailbox_daemon, name, value):
                with self.assertRaises(RoleContractError):
                    mailbox_daemon.validate_role_contract_bindings()

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

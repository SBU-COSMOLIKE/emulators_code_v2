"""Keep the protected role contract strict and bound to live controls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

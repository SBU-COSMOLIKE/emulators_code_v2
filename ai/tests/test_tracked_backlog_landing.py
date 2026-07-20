"""Prove one landing preserves both the audited fix and tracked backlog."""

from pathlib import Path
import unittest

from ai.tests.tools_mailbox_daemon_primary_worktree_repro import (
    close_backlog_ticket,
    default_implementer,
    default_primary,
    default_sol,
    git,
    invoke,
    load_scratch_daemon,
    scratch_repository,
    seal_backlog,
)


class TrackedBacklogLandingTests(unittest.TestCase):
    """Keep an Architect backlog edit in the ticket's single commit."""

    def test_landing_contains_candidate_and_sealed_backlog(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            primary = default_primary(root)
            implementer = default_implementer(root)
            daemon = load_scratch_daemon(primary)
            daemon.ensure_primary_execution(live_action=True, dry_run=False)

            base = git(implementer, "rev-parse", "HEAD").stdout.strip()
            anchor = "tracked-backlog-landing"
            cycle = anchor + "@" + base
            backlog = primary / "ai/notes/backlog.md"
            backlog.write_text(
                "- OPEN **HIGH** **BUG FIX** — [Tracked backlog](#"
                + anchor + ")\n\n<a id=\"" + anchor + "\"></a>\n"
                "**Red Team reopen count: 0.**\n"
                "**Red Team reopening: allowed.**\n",
                encoding="utf-8", newline="")
            seal_backlog(primary)
            daemon.register_ticket_cycle_message(
                agent="opus",
                message=("MAILBOX-FLOW: ticket\nMAILBOX-CYCLE: " + cycle
                         + "\nMAILBOX-MODE: two-role\n\nFix it.\n"),
                skip_redteam=True)
            daemon.prepare_implementer_cycle_checkout(cycle_id=cycle)
            changed = Path(implementer) / "accepted-fix.txt"
            changed.write_text("fixed\n", encoding="utf-8")
            git(implementer, "add", changed.name)
            git(implementer, "commit", "-m", "Add the accepted fix")
            candidate = daemon.record_implementer_candidate(
                cycle_id=cycle, starting_head=base)

            close_backlog_ticket(primary=primary, anchor=anchor)
            sealed = backlog.read_bytes()
            landing, parent, _reference = daemon.prepare_exact_squash_landing(
                cycle_id=cycle, candidate_commit=candidate, mode="two-role",
                sealed_backlog=sealed)

            self.assertEqual(
                git(root, "show", landing + ":ai/notes/backlog.md").stdout,
                sealed.decode("utf-8"))
            self.assertEqual(
                git(root, "show", landing + ":accepted-fix.txt").stdout,
                "fixed\n")
            daemon.preflight_role_baseline_sync(
                target=landing, retiring_candidate=candidate)
            daemon.land_prepared_commit_in_clean_user_checkout(
                landing=landing, parent=parent,
                candidate_commit=candidate)
            daemon.record_architect_commit(
                cycle_id=cycle, accepted_commit=landing, mode="two-role")
            daemon.sync_all_clean_role_baselines(target=landing)
            self.assertEqual(backlog.read_bytes(), sealed)
            self.assertEqual(git(primary, "status", "--porcelain").stdout, "")
            self.assertEqual(
                git(default_sol(root), "rev-parse", "HEAD").stdout.strip(),
                landing)

    def test_reopen_decision_gets_its_own_backlog_landing(self):
        with scratch_repository() as root:
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)
            backlog = root / "ai/notes/backlog.md"
            backlog.write_text(
                "# Closed tickets\n\n<a id=\"reopen-landing\"></a>\n"
                "## Reopen landing\n\n**Red Team reopen count: 0.**\n\n"
                "**Red Team reopening: allowed.**\n",
                encoding="utf-8", newline="")
            git(root, "add", "ai/notes/backlog.md")
            git(root, "commit", "-m", "Add closed ticket")
            rc, output, error = invoke(root, ["--once"])
            self.assertEqual(rc, 0, output + error)

            primary = default_primary(root)
            daemon = load_scratch_daemon(primary)
            daemon.ensure_primary_execution(live_action=True, dry_run=False)
            reviewed = git(root, "rev-parse", "HEAD").stdout.strip()
            cycle = "reopen-landing@" + reviewed
            state = daemon.read_ticket_cycle_state()
            state["active"][cycle] = {
                "phase": "awaiting-redteam", "commit": reviewed,
                "mode": "normal", "route": "primary",
                "ticket_class": "ordinary"}
            daemon.write_ticket_cycle_state(state=state)

            primary_backlog = primary / "ai/notes/backlog.md"
            primary_backlog.write_text(
                primary_backlog.read_text(encoding="utf-8").replace(
                    "reopen count: 0", "reopen count: 1").replace(
                    "reopening: allowed",
                    "reopening: barred by Architect NO-GO"),
                encoding="utf-8", newline="")
            seal_backlog(primary)
            inflight = primary / "ai/notes/mailbox/inflight/0001-to-fable.md"
            inflight.parent.mkdir(parents=True, exist_ok=True)
            inflight.write_text("reopen decision\n", encoding="utf-8")

            landing, completed = daemon.land_architect_reopen_decision(
                dispatch_path=str(inflight), cycle_id=cycle,
                reviewed_landing=reviewed, decision="NO-GO")
            changed = git(
                root, "diff", "--name-only", reviewed, landing).stdout
            final = daemon.read_ticket_cycle_state()
            self.assertTrue(completed)
            self.assertEqual(changed, "ai/notes/backlog.md\n")
            self.assertEqual(final["completed"][cycle], landing)
            self.assertEqual(git(root, "rev-parse", "HEAD").stdout.strip(),
                             landing)
            self.assertEqual(git(primary, "status", "--porcelain").stdout,
                             "")
            self.assertIn(
                "barred by Architect NO-GO",
                git(root, "show", landing + ":ai/notes/backlog.md").stdout)


if __name__ == "__main__":
    unittest.main()

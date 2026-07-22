"""Prove two same-second dispatches keep two separate relay logs.

Each role turn archives its complete terminal output under a relay-log
filename built from a one-second timestamp and the role name. Before the
exclusive reservation, two turns of the same role inside one clock second
chose the same name, and the later ``open(..., "w")`` truncated the earlier
turn's evidence. These tests hand the reservation function one frozen stamp
directly — no clock is involved — so the same-second collision is exercised
on every run.
"""

import os
import tempfile
import unittest

from ai.tools import mailbox_daemon as daemon


FROZEN_STAMP = "20990101-120000"


class RelayLogReservationTests(unittest.TestCase):
    """One frozen stamp must still yield one file per dispatched turn."""

    def test_equal_stamps_reserve_two_files_and_keep_both_logs(self):
        """The second same-second run must not truncate the first log."""
        with tempfile.TemporaryDirectory(prefix="relay-reserve-") as relay:
            first = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="opus", relay_directory=relay)
            second = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="opus", relay_directory=relay)
            self.assertNotEqual(first, second)
            with open(first, "w", encoding="utf-8") as stream:
                stream.write("first turn output\n")
            with open(second, "w", encoding="utf-8") as stream:
                stream.write("second turn output\n")
            with open(first, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "first turn output\n")
            with open(second, encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "second turn output\n")

    def test_reserved_names_keep_the_stamp_role_and_suffix_order(self):
        """The readable name survives; only the collision gains a suffix."""
        with tempfile.TemporaryDirectory(prefix="relay-reserve-") as relay:
            first = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="opus", relay_directory=relay)
            second = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="opus", relay_directory=relay)
            third = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="opus", relay_directory=relay)
            expected_first = FROZEN_STAMP + "-dispatch-opus.log"
            expected_second = FROZEN_STAMP + "-dispatch-opus-01.log"
            expected_third = FROZEN_STAMP + "-dispatch-opus-02.log"
            self.assertEqual(os.path.basename(first), expected_first)
            self.assertEqual(os.path.basename(second), expected_second)
            self.assertEqual(os.path.basename(third), expected_third)

    def test_two_roles_share_one_second_without_any_suffix(self):
        """Different roles never collide, so both keep the plain name."""
        with tempfile.TemporaryDirectory(prefix="relay-reserve-") as relay:
            implementer = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="opus", relay_directory=relay)
            architect = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="fable", relay_directory=relay)
            expected_implementer = FROZEN_STAMP + "-dispatch-opus.log"
            expected_architect = FROZEN_STAMP + "-dispatch-fable.log"
            self.assertEqual(
                os.path.basename(implementer), expected_implementer)
            self.assertEqual(
                os.path.basename(architect), expected_architect)

    def test_missing_relay_directory_is_created(self):
        """A first run on a fresh checkout must not fail on the folder."""
        with tempfile.TemporaryDirectory(prefix="relay-reserve-") as parent:
            relay = os.path.join(parent, "relay")
            reserved = daemon.reserve_dispatch_log_path(
                stamp=FROZEN_STAMP, agent="sol", relay_directory=relay)
            self.assertTrue(os.path.isfile(reserved))


if __name__ == "__main__":
    unittest.main()

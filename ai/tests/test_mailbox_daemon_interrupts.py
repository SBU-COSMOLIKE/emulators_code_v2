"""Focused tests for the daemon's Ctrl-C and process-group handling."""

import os
import signal
import threading
import unittest

from ai.tools import mailbox_daemon as daemon


class FakeAgentProcess:
    """Explicit stand-in for a launched agent CLI without a real pid.

    ``kill_agent_process`` must fall back to the direct kill when the
    group kill cannot address the process.
    """

    def __init__(self, running=True):
        self.running = running
        self.killed = False
        self.waited = False

    def poll(self):
        if self.running:
            return None
        return 0

    def kill(self):
        self.killed = True
        self.running = False

    def wait(self):
        self.waited = True
        return 0


class DeferredInterruptsTests(unittest.TestCase):
    """Pin the finish-the-transition-first Ctrl-C behavior."""

    def test_interrupt_inside_the_body_is_raised_at_the_exit(self):
        body_completed = False
        with self.assertRaises(KeyboardInterrupt):
            with daemon.DeferredInterrupts():
                os.kill(os.getpid(), signal.SIGINT)
                # The signal was recorded, not raised: the transition body
                # keeps running to its natural end.
                body_completed = True
        self.assertTrue(body_completed)

    def test_no_interrupt_means_no_effect(self):
        with daemon.DeferredInterrupts():
            observed = 1 + 1
        self.assertEqual(observed, 2)

    def test_previous_handler_returns_after_the_block(self):
        before = signal.getsignal(signal.SIGINT)
        with daemon.DeferredInterrupts():
            inside = signal.getsignal(signal.SIGINT)
            self.assertNotEqual(inside, before)
        self.assertEqual(signal.getsignal(signal.SIGINT), before)

    def test_worker_thread_leaves_signal_handling_untouched(self):
        before = signal.getsignal(signal.SIGINT)
        seen = {}

        def run_in_worker():
            with daemon.DeferredInterrupts():
                seen["inside"] = signal.getsignal(signal.SIGINT)

        worker = threading.Thread(target=run_in_worker)
        worker.start()
        worker.join()
        self.assertEqual(seen["inside"], before)
        self.assertEqual(signal.getsignal(signal.SIGINT), before)


class AgentLaunchShapeTests(unittest.TestCase):
    """Pin the session-isolating launch of agent CLI children."""

    def test_the_one_launch_site_starts_a_new_session(self):
        import pathlib
        dispatch_source = pathlib.Path(
            daemon.SCRIPT_DIR, "mailbox_dispatch.py").read_text(
                encoding="utf-8")
        launches = dispatch_source.count("daemon.subprocess.Popen(")
        self.assertEqual(launches, 1)
        self.assertEqual(
            dispatch_source.count("start_new_session=True"), 1)


class KillAgentProcessTests(unittest.TestCase):
    """Pin the group-kill fallback and the live-process registry sweep."""

    def test_process_without_a_pid_falls_back_to_direct_kill(self):
        proc = FakeAgentProcess()
        daemon.kill_agent_process(proc=proc)
        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)

    def test_live_sweep_kills_running_processes_only(self):
        running = FakeAgentProcess(running=True)
        finished = FakeAgentProcess(running=False)
        with daemon._LIVE_AGENT_PROCESSES_LOCK:
            daemon._LIVE_AGENT_PROCESSES[id(running)] = running
            daemon._LIVE_AGENT_PROCESSES[id(finished)] = finished
        try:
            daemon.kill_live_agent_processes()
        finally:
            with daemon._LIVE_AGENT_PROCESSES_LOCK:
                daemon._LIVE_AGENT_PROCESSES.pop(id(running), None)
                daemon._LIVE_AGENT_PROCESSES.pop(id(finished), None)
        self.assertTrue(running.killed)
        self.assertFalse(finished.killed)


if __name__ == "__main__":
    unittest.main()

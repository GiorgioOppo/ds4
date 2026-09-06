"""Offline regression tests for the embedded Linux bootstrap; never install tools."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SOURCE = (Path(__file__).resolve().parents[2] /
          "Sources/DwarfStar/Features/MiniToolBench/MiniToolBenchBootstrap.swift")
SCRIPT = SOURCE.read_text().split('static let script = #"""\n', 1)[1].rsplit('\n"""#', 1)[0]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.module = {"__name__": "bootstrap_test"}
        exec(compile(SCRIPT, str(SOURCE), "exec"), self.module)
        self.temporary = tempfile.TemporaryDirectory(prefix="dwarfstar-bootstrap-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.events = []
        self.module["emit"] = lambda event, **fields: self.events.append({"event": event, **fields})
        self.configuration = {
            "suite": "core19", "tier": "smoke", "endpoint": "http://host.containers.internal:8080/v1",
            "platform": "apple-silicon", "modelName": "Example's model $(false)",
            "modelID": "/models/flash.gguf", "engine": "DwarfStar", "backend": "metal",
            "contextLength": "", "attempts": 2,
        }

    def invoke_main(self, request):
        stream = io.TextIOWrapper(io.BytesIO(json.dumps(request).encode()))
        with patch.object(sys, "stdin", stream), patch.object(sys, "platform", "linux"), \
                patch.object(sys, "version_info", (3, 14, 6)):
            try:
                return self.module["main"]()
            finally:
                if self.module["_log"]:
                    self.module["_log"].close()

    def test_config_values_are_single_arguments_and_context_is_not_invented(self):
        args = self.module["run_arguments"](self.configuration, "run-1", self.root / "results")
        self.assertIn("--model-name=Example's model $(false)", args)
        self.assertIn("--model=/models/flash.gguf", args)
        self.assertIn("--attempts=2", args)
        self.assertIn("--concurrency=1", args)
        self.assertFalse(any(arg.startswith("--context-length") for arg in args))
        self.configuration["contextLength"] = "32768"
        self.assertIn("--context-length=32768", self.module["connection_arguments"](self.configuration))

    def test_invalid_attempts_and_endpoint_are_rejected(self):
        error = self.module["BootstrapError"]
        for value in (True, 1.0, "1", 0, 3):
            with self.subTest(attempts=value), self.assertRaises(error):
                self.module["run_arguments"]({**self.configuration, "attempts": value}, "run-1", self.root)
        for value in ("http://0.0.0.0:8080/v1", "http://user:secret@example.com/v1", "http://example.com/v1?q=x"):
            with self.subTest(endpoint=value), self.assertRaises(error):
                self.module["connection_arguments"]({**self.configuration, "endpoint": value})

    def test_cancel_before_worker_start_is_durable(self):
        root = self.module["owned_workspace"]({"workspace_root": str(self.root)})
        self.assertEqual(self.module["cancel"](root, "early-cancel"), 0)
        self.assertTrue((root / "runs/early-cancel/cancel-requested.json").is_file())
        self.module["prepare"] = lambda root: self.fail("Preparation started after cancellation")
        self.assertEqual(self.invoke_main({"action": "run", "run_id": "early-cancel",
                                         "workspace_root": str(root)}), 130)
        self.assertEqual(self.events[-1]["status"], "cancelled")

    def test_cancel_does_not_signal_reused_pid(self):
        directory = self.root / "runs/reused"
        directory.mkdir(parents=True)
        (directory / "control.json").write_text(json.dumps({
            "owner": self.module["OWNER"], "run_id": "reused", "status": "active",
            "pid": 999999, "start": "before", "child_pid": 999998, "child_start": "before",
        }))
        self.module["process_start"] = lambda pid: "after"
        with patch("os.kill") as kill, patch("os.killpg") as killpg:
            self.module["cancel"](self.root, "reused")
            kill.assert_not_called()
            killpg.assert_not_called()
        self.assertTrue((directory / "cancel-requested.json").is_file())

    def test_marker_interrupts_running_child(self):
        marker = self.root / "cancel.json"
        self.module["_cancel_marker"] = marker
        timer = threading.Timer(0.2, lambda: marker.write_text("{}"))
        timer.start()
        self.addCleanup(timer.join)
        started = time.monotonic()
        with self.assertRaises(self.module["Cancelled"]):
            self.module["run_process"]([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"])
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNone(self.module["_child"])

    def test_timeout_still_applies_after_child_closes_output(self):
        started = time.monotonic()
        with self.assertRaises(self.module["BootstrapError"]):
            self.module["run_process"]([sys.executable, "-c",
                "import os,time; os.close(1); os.close(2); time.sleep(30)"], timeout=0.2)
        self.assertLess(time.monotonic() - started, 5)

    def test_export_keeps_official_results_and_skips_symlink_and_config(self):
        run = self.root / "run"
        source = run / "results/core19/platform/model/profile"
        source.mkdir(parents=True)
        summary = {"passed_tasks": 1, "total_tasks": 1, "results": ["results-task.json"]}
        for name, value in (("summary.json", summary), ("run-meta.json", {"revision": "official"}),
                            ("results-task.json", {"passed": True}), ("transcript-task.json", {"steps": []}),
                            ("config.json", {"api_key": "do-not-export"})):
            (source / name).write_text(json.dumps(value))
        (source / "transcript-link.json").symlink_to(source / "config.json")
        (run / "bootstrap.jsonl").write_text('{"event":"log","message":"safe"}\n')
        export = self.root / "export"
        export.mkdir()
        destination, actual = self.module["export_artifacts"](run, export, "run-1")
        self.assertEqual(actual, summary)
        self.assertEqual({p.name for p in destination.iterdir()}, {
            "summary.json", "run-meta.json", "results-task.json", "transcript-task.json", "bootstrap.jsonl"})
        self.assertEqual(json.loads((destination / "summary.json").read_text()), summary)

    def test_run_automatically_prepares_then_checks_then_executes(self):
        export = self.root / "export"
        export.mkdir()
        workspace = self.root / "workspace"
        steps = []
        def prepare(root):
            steps.append("prepare")
            return Path("/fake/python"), root / "runner", {"TBENCH_API_KEY": self.module["_secret"]}
        def run(arguments, **kwargs):
            action = arguments[2]
            steps.append(action)
            self.assertEqual(kwargs["env"]["TBENCH_API_KEY"], "private-key")
            self.assertFalse(any("private-key" in str(arg) for arg in arguments))
            if action == "run":
                result = workspace / "runs/run-1/results/profile"
                result.mkdir(parents=True)
                (result / "summary.json").write_text('{"passed_tasks":1,"total_tasks":1}')
            return 0, ""
        self.module["prepare"] = prepare
        self.module["run_process"] = run
        self.module["container_preflight"] = lambda *args: steps.append("preflight")
        self.assertEqual(self.invoke_main({"action": "run", "run_id": "run-1", "workspace_root": str(workspace),
            "export_directory": str(export), "configuration": self.configuration, "api_key": "private-key"}), 0)
        self.assertEqual(steps, ["prepare", "doctor", "preflight", "run"])
        result = next(event for event in self.events if event["event"] == "result")
        self.assertEqual(result["directory"], str(export / "run-1"))
        self.assertFalse(result["partial"])
        self.assertEqual(self.events[-1]["status"], "success")
        self.assertNotIn("private-key", (workspace / "runs/run-1/control.json").read_text())

    def test_failed_doctor_does_not_start_benchmark(self):
        workspace = self.root / "workspace"
        self.module["prepare"] = lambda root: (Path("/fake/python"), root / "runner", {})
        actions = []
        def fail(arguments, **kwargs):
            actions.append(arguments[2])
            raise self.module["BootstrapError"]("Endpoint not available")
        self.module["run_process"] = fail
        export = self.root / "export"
        export.mkdir()
        with self.assertRaises(self.module["BootstrapError"]):
            self.invoke_main({"action": "run", "run_id": "run-1", "workspace_root": str(workspace),
                              "export_directory": str(export), "configuration": self.configuration})
        self.assertEqual(actions, ["doctor"])
        self.assertTrue((export / "run-1/bootstrap.jsonl").is_file())
        self.assertFalse(any(event["event"] == "result" for event in self.events))

    def test_doctor_only_prepares_and_checks_container_without_benchmark(self):
        workspace = self.root / "workspace"
        steps = []
        def prepare(root):
            steps.append("prepare")
            return Path("/fake/python"), root / "runner", {}
        def run(arguments, **kwargs):
            steps.append(arguments[2])
            return 0, ""
        self.module["prepare"] = prepare
        self.module["run_process"] = run
        self.module["container_preflight"] = lambda *args: steps.append("preflight")
        self.assertEqual(self.invoke_main({"action": "doctor", "run_id": "doctor-only", "workspace_root": str(workspace),
                                          "configuration": self.configuration}), 0)
        self.assertEqual(steps, ["prepare", "doctor", "preflight"])
        self.assertFalse((workspace / "runs/doctor-only/run-requested.json").exists())
        self.assertFalse(any(event["event"] == "result" for event in self.events))
        self.assertEqual(self.events[-1]["status"], "success")

    def test_podman_6_adapter_replaces_only_official_overlay_without_editing_checkout(self):
        repository = self.root / "runner"
        original = repository / "compat/podman/terminal-network-podman-loopback.json"
        original.parent.mkdir(parents=True)
        original.write_text('{"services":{"main":{"network_mode":"slirp4netns:allow_host_loopback=true"}}}')
        before = original.read_bytes()
        (self.root / "bin").mkdir()
        runtime = self.root / "fake-podman"
        runtime.write_text("#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
        runtime.chmod(0o700)
        self.module["run_process"] = lambda args, **kwargs: (0, "true" if "info" in args else "podman version 6.0.2")
        self.module["install_runtime_compatibility"](self.root, repository, Path(sys.executable), runtime, {})
        overlay = self.root / "compat/terminal-network-loopback.json"
        self.assertEqual(json.loads(overlay.read_text()), {"services": {"main": {"network_mode": "pasta:--map-gw"}}})
        self.assertEqual(original.read_bytes(), before)
        adapter = self.root / "compat/docker-runtime.py"
        unrelated = self.root / "different-overlay.json"
        arguments = ["compose", "-f", str(unrelated), "-f", str(original), "up", "--detach"]
        actual = json.loads(subprocess.check_output([sys.executable, str(adapter), *arguments]))
        self.assertEqual(actual, ["compose", "-f", str(unrelated), "-f", str(overlay), "up", "--detach"])
        actual = json.loads(subprocess.check_output([sys.executable, str(adapter), "compose", "--file=" + str(original), "config"]))
        self.assertEqual(actual, ["compose", "--file=" + str(overlay), "config"])
        actual = json.loads(subprocess.check_output([sys.executable, str(adapter), "cp", str(original), "container:/tmp/file"]))
        self.assertEqual(actual, ["cp", str(original), "container:/tmp/file"])

    def test_podman_5_retains_upstream_network(self):
        (self.root / "bin").mkdir()
        self.module["run_process"] = lambda *args, **kwargs: (0, "podman version 5.6.2")
        self.module["install_runtime_compatibility"](self.root, self.root / "runner", Path(sys.executable), "/usr/bin/podman", {})
        overlay = json.loads((self.root / "compat/terminal-network-loopback.json").read_text())
        self.assertEqual(overlay["services"]["main"]["network_mode"], "slirp4netns:allow_host_loopback=true")

    def test_podman_6_rootful_uses_bridge_and_unknown_mode_is_rejected(self):
        (self.root / "bin").mkdir()
        for rootless, expected in (("false", "bridge"), ("unknown", None)):
            self.module["run_process"] = lambda args, **kwargs: (0, rootless if "info" in args else "podman version 6.0.2")
            if expected:
                self.module["install_runtime_compatibility"](self.root, self.root / "runner", Path(sys.executable), "/usr/bin/podman", {})
                overlay = json.loads((self.root / "compat/terminal-network-loopback.json").read_text())
                self.assertEqual(overlay["services"]["main"]["network_mode"], expected)
            else:
                with self.assertRaises(self.module["BootstrapError"]):
                    self.module["install_runtime_compatibility"](self.root, self.root / "runner", Path(sys.executable), "/usr/bin/podman", {})

    def test_failed_container_preflight_prevents_all_benchmark_attempts(self):
        workspace = self.root / "workspace"
        self.module["prepare"] = lambda root: (Path("/fake/python"), root / "runner", {})
        actions = []
        projects = []
        def fail(arguments, **kwargs):
            if arguments[1] == "compose":
                action = "up" if "up" in arguments else "down"
                projects.append(arguments[arguments.index("--project-name") + 1])
            else:
                action = arguments[2]
            actions.append(action)
            if action == "up":
                raise self.module["BootstrapError"]("slirp4netns support has been removed")
            return 0, ""
        self.module["run_process"] = fail
        export = self.root / "export"
        export.mkdir()
        with self.assertRaisesRegex(self.module["BootstrapError"], "nessun tentativo"):
            self.invoke_main({"action": "run", "run_id": "run-1", "workspace_root": str(workspace),
                              "export_directory": str(export), "configuration": self.configuration})
        self.assertEqual(actions, ["doctor", "up", "down"])
        self.assertEqual(projects[0], projects[1])
        self.assertTrue(projects[0].startswith("dwarfstar-preflight-"))
        self.assertFalse((workspace / "runs/run-1/run-requested.json").exists())
        self.assertFalse(any(event["event"] == "result" for event in self.events))

    def test_preflight_cancellation_still_removes_only_its_project(self):
        calls = []
        def run(arguments, **kwargs):
            calls.append((arguments, kwargs))
            if "up" in arguments:
                self.module["_cancelled"] = True
                raise self.module["Cancelled"]()
            return 0, ""
        self.module["run_process"] = run
        with self.assertRaises(self.module["Cancelled"]):
            self.module["container_preflight"](self.root, self.root / "runner", {}, self.root, self.configuration)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0][-1], "down")
        self.assertFalse(calls[1][1]["cancellable"])
        self.assertEqual(calls[0][0][:-2], calls[1][0][:-1])
        config = json.loads((self.root / "preflight/compose.json").read_text())
        self.assertEqual(config["services"]["main"]["entrypoint"], ["/bin/sh", "-c", 'trap "exit 0" TERM INT; sleep 120 & wait'])
        self.assertNotIn("volumes", config["services"]["main"])
        self.assertNotIn("environment", config["services"]["main"])

    def test_preflight_accepts_ansi_acknowledgement_but_rejects_embedded_token(self):
        for output, valid in (("\x1b[0mDWARFSTAR_PREFLIGHT_OK\x1b[0m\n", True),
                              ("\x1b[0merror quoting DWARFSTAR_PREFLIGHT_OK\n", False)):
            with self.subTest(output=output):
                calls = []
                def run(arguments, **kwargs):
                    calls.append(arguments)
                    return 0, output if "exec" in arguments and "/bin/sh" in arguments else ""
                self.module["run_process"] = run
                if valid:
                    self.module["container_preflight"](self.root, self.root / "runner", {}, self.root, self.configuration)
                    self.assertTrue(any("ahosts" in args for args in calls))
                else:
                    with self.assertRaises(self.module["BootstrapError"]):
                        self.module["container_preflight"](self.root, self.root / "runner", {}, self.root, self.configuration)
                self.assertEqual(calls[-1][-1], "down")

    def test_ansi_normalizer_preserves_plain_text_and_exact_lines(self):
        source = "\x1b[0mDWARFSTAR_PREFLIGHT_OK\x1b[0m\n\x1b[31merror quoting DWARFSTAR_PREFLIGHT_OK\x1b[0m"
        expected = "DWARFSTAR_PREFLIGHT_OK\nerror quoting DWARFSTAR_PREFLIGHT_OK"
        self.assertEqual(self.module["strip_ansi"](source), expected)
        self.assertEqual(self.module["strip_ansi"](expected), expected)

    def test_process_strips_ansi_from_output_and_streamed_logs(self):
        script = "import sys; sys.stdout.write('\\x1b[0mDWARFSTAR_PREFLIGHT_OK\\n\\x1b[32mfinal fragment\\x1b[0m')"
        code, output = self.module["run_process"]([sys.executable, "-c", script])
        self.assertEqual(code, 0)
        self.assertEqual(output, "DWARFSTAR_PREFLIGHT_OK\nfinal fragment")
        logs = [event["message"] for event in self.events if event["event"] == "log"]
        self.assertEqual(logs, ["DWARFSTAR_PREFLIGHT_OK", "final fragment"])

    def test_cleanup_child_can_run_after_cancellation(self):
        self.module["_cancelled"] = True
        code, output = self.module["run_process"]([sys.executable, "-c", "print('cleanup')"], cancellable=False)
        self.assertEqual((code, output), (0, "cleanup"))

    def test_key_redacted_in_jsonl(self):
        module = {"__name__": "redaction_test"}
        exec(compile(SCRIPT, str(SOURCE), "exec"), module)
        module["_secret"] = "private-key"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module["emit"]("log", message="token=private-key", details={"value": "private-key"})
        self.assertNotIn("private-key", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["details"]["value"], "[redacted]")


if __name__ == "__main__":
    unittest.main()

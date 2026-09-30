"""CPU-only checks for the checkpoint-420 fork launch contract."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIRECTORY = ROOT / "scripts/experiment_plans/linear_square_ablation"
SPEC = importlib.util.spec_from_file_location("plan13_launcher", PLAN_DIRECTORY / "launch.py")
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


class Plan13LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / "repo"
        self.environment = patch.dict(os.environ, {"REPO_DIR": str(self.repo)}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def resolve(self, *args):
        return LAUNCHER.resolve_arguments(["13", *args])[2]

    def launch(self, *args):
        with patch.object(LAUNCHER.sys, "argv", ["launch.py", "13", "--layout", "h200_8gpu", *args]):
            LAUNCHER.main()

    def install_base_stub(self):
        base = Path(self.resolve("--layout", "h200_8gpu")["base_launcher"])
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_text("#!/bin/bash\nexit 99\n")

    def test_both_layouts_keep_source_and_fixed_training_settings(self):
        for layout, gpus, batches in (("a6000_16gpu", 16, 12), ("h200_8gpu", 8, 24)):
            with self.subTest(layout=layout):
                result = self.resolve("--layout", layout)
                self.assertEqual(result["run_name"], f"sd35_pickscore_{layout}_plan13_linear_fork420_scale5_kl1e-4")
                self.assertTrue(result["fork_from_checkpoint"].endswith(
                    "outputs/step1_1_linear_scale15/sd35_pickscore_a6000_16gpu_step1_1_linear_scale15_kl1e-4/checkpoints/checkpoint-420"))
                self.assertEqual(result["sheet_id"], "B.6")
                self.assertEqual(result["linear_target_scale"], 5)
                self.assertEqual(result["target_scale_schedule"], "constant")
                self.assertEqual(result["transform"], "none")
                self.assertEqual(result["config"]["train.learning_rate"], 0.0003)
                self.assertEqual(result["config"]["train.beta"], 0.0001)
                self.assertEqual(result["config"]["num_epochs"], 1000)
                self.assertEqual(result["config"]["sample.num_batches_per_epoch"], batches)
                self.assertEqual(result["num_gpus"], gpus)
                self.assertTrue(result["require_resume"])
                self.assertEqual(result["training_budget"]["maximum_additional_update_attempts"], 580)
                self.assertIn("--fork_from_checkpoint=" + result["fork_from_checkpoint"], result["torchrun_args"])
                self.assertNotIn("--config.resume_from=" + result["fork_from_checkpoint"], result["torchrun_args"])

    def test_source_override_and_independent_output_overrides(self):
        source = Path(self.temporary.name) / "parent/checkpoints/checkpoint-420"
        save = Path(self.temporary.name) / "child"
        logs = Path(self.temporary.name) / "logs"
        result = self.resolve(f"--fork_from_checkpoint={source}", f"--config.save_dir={save}", f"--config.logdir={logs}")
        self.assertEqual(result["fork_from_checkpoint"], str(source))
        self.assertEqual(result["save_dir"], str(save))
        self.assertEqual(result["logdir"], str(logs))

    def test_rejects_invalid_source_and_fixed_parameter_overrides(self):
        bad_arguments = (
            ("--fork_from_checkpoint=",),
            ("--fork_from_checkpoint=relative/checkpoint-420",),
            ("--fork_from_checkpoint=/parent/checkpoints/checkpoint-450",),
            ("--linear_target_scale=15",),
            ("--config.train.learning_rate=0.001",),
            ("--config.num_epochs=1420",),
        )
        for args in bad_arguments:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.resolve(*args)

    def test_existing_plans_neither_accept_nor_receive_fork_flag(self):
        for run_id in (f"{number:02d}" for number in range(1, 13)):
            with self.subTest(run_id=run_id):
                result = LAUNCHER.resolve_arguments([run_id])[2]
                self.assertFalse(any(arg.startswith("--fork_from_checkpoint") for arg in result["torchrun_args"]))
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    LAUNCHER.resolve_arguments([run_id, "--fork_from_checkpoint=/parent/checkpoints/checkpoint-420"])
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    LAUNCHER.resolve_arguments([run_id, "--fork_from_checkpoint="])

    def test_rejects_source_output_overlap_and_source_as_child_resume(self):
        source = Path(self.resolve()["fork_from_checkpoint"])
        source_run = source.parent.parent
        for option, value in (("--config.save_dir", source_run), ("--config.logdir", source_run / "logs"),
                              ("--config.save_dir", source_run.parent), ("--config.resume_from", source)):
            with self.subTest(option=option, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.resolve(f"{option}={value}")

    def test_rejects_overlapping_child_save_and_log_paths(self):
        save = Path(self.temporary.name) / "child"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.resolve(f"--config.save_dir={save}", f"--config.logdir={save}/logs")

    def test_rejects_source_output_overlap_through_symlink(self):
        source = Path(self.resolve()["fork_from_checkpoint"])
        source.parent.mkdir(parents=True)
        alias = Path(self.temporary.name) / "alias"
        alias.symlink_to(source.parent.parent, target_is_directory=True)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.resolve(f"--config.save_dir={alias}/child")

    def test_shell_dry_run_has_no_side_effects_even_if_source_missing(self):
        environment = dict(os.environ, PATH=os.defpath)
        command = ["bash", str(PLAN_DIRECTORY / "run_plan13_linear_fork420_scale5.sh"), "--dry-run"]
        process = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
        result = json.loads(process.stdout)
        self.assertFalse(result["fork_source_exists"])
        self.assertFalse(result["resume_directory_exists"])
        self.assertFalse(self.repo.exists())

    def test_missing_source_and_child_fail_before_base_launcher(self):
        with patch.object(LAUNCHER.os, "execvpe") as execute, contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit):
                self.launch()
        execute.assert_not_called()
        self.assertIn("No new training was started", errors.getvalue())
        self.assertFalse(self.repo.exists())

    def test_existing_source_allows_fresh_child_directory(self):
        self.install_base_stub()
        result = self.resolve("--layout", "h200_8gpu")
        Path(result["fork_from_checkpoint"]).mkdir(parents=True)
        with patch.object(LAUNCHER.os, "execvpe") as execute, contextlib.redirect_stderr(io.StringIO()):
            self.launch()
        execute.assert_called_once()
        self.assertFalse(Path(result["save_dir"]).exists())
        self.assertIn("--require_resume=true", execute.call_args.args[1])

    def test_existing_child_search_directory_allows_missing_source(self):
        self.install_base_stub()
        result = self.resolve("--layout", "h200_8gpu")
        Path(result["save_dir"]).mkdir(parents=True)
        with patch.object(LAUNCHER.os, "execvpe") as execute, contextlib.redirect_stderr(io.StringIO()):
            self.launch()
        execute.assert_called_once()
        self.assertFalse(Path(result["fork_from_checkpoint"]).exists())

    def test_explicit_child_resume_allows_missing_source(self):
        self.install_base_stub()
        child = Path(self.temporary.name) / "child/checkpoints/checkpoint-450"
        child.mkdir(parents=True)
        with patch.object(LAUNCHER.os, "execvpe") as execute, contextlib.redirect_stderr(io.StringIO()):
            self.launch(f"--config.resume_from={child}")
        self.assertIn(f"--config.resume_from={child}", execute.call_args.args[1])

    def test_missing_explicit_child_does_not_fall_back_to_source(self):
        result = self.resolve()
        Path(result["fork_from_checkpoint"]).mkdir(parents=True)
        with patch.object(LAUNCHER.os, "execvpe") as execute, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.launch(f"--config.resume_from={self.repo}/missing/checkpoint-450")
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()

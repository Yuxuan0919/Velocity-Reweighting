"""Check the handoff commands without loading a model or starting GPU workers."""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


DIRECTORY = Path(__file__).resolve().parent
MATRIX = [
    ("upper", "1", "2"), ("upper", "1", "1"),
    ("two_sided", "0.6", "2"), ("upper", "1", "3"),
    ("upper", "1", "0.6"), ("two_sided", "0.3", "2"),
    ("upper", "1", "5"), ("upper", "1", "0.3"),
]


class FixedLauncherTests(unittest.TestCase):
    def launch(self, row, *args, environment=None, hardware="a6000_16gpu"):
        script = DIRECTORY / f"run_sd3_{hardware}_shared_exp_clip{row}_kl1e-4.sh"
        env = {"PATH": os.defpath}
        env.update(environment or {})
        return subprocess.run(
            ["bash", str(script), *args],
            cwd="/tmp", env=env, capture_output=True, text=True, timeout=10,
        )

    def test_eight_handoff_commands_preserve_the_document_matrix(self):
        for row, (variant, negative, positive) in enumerate(MATRIX, 1):
            with self.subTest(row=row):
                result = self.launch(row, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout)
                self.assertEqual(args[0], "torchrun")
                expected = [
                    "--nnodes=2", "--nproc_per_node=8",
                    "--config.sample.train_batch_size=6",
                    "--config.sample.num_image_per_prompt=24",
                    "--config.sample.num_batches_per_epoch=12",
                    "--config.train.batch_size=6",
                    "--config.train.gradient_accumulation_steps=12",
                    "--config.train.num_inner_epochs=1",
                    "--config.train.timestep_fraction=1.0",
                    "--config.train.beta=0.0001",
                    "--awr_variance_gate=false", "--awr_gamma_floor=1e-6",
                    f"--shared_clip_variant={variant}",
                    f"--shared_clip_epsilon_minus={negative}",
                    f"--shared_clip_epsilon_plus={positive}",
                    f"--config.run_name=sd35_pickscore_a6000_16gpu_shared_exp_clip{row}_kl1e-4",
                ]
                for flag in expected:
                    self.assertIn(flag, args)
                self.assertTrue(any(arg.endswith("/train_nft_sd3_ours-singleloss-AWR.py") for arg in args))
                self.assertTrue(any(arg.startswith("--config=") and arg.endswith(":sd3_pickscore") for arg in args))

    def test_old_selection_variables_cannot_change_a_fixed_experiment(self):
        result = self.launch(3, "--dry-run", environment={
            "CLIP_RUN": "8", "TASK": "ocr", "NNODES": "1", "NPROC_PER_NODE": "1",
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertIn("--shared_clip_variant=two_sided", args)
        self.assertIn("--shared_clip_epsilon_minus=0.6", args)
        self.assertIn("--nnodes=2", args)
        self.assertIn("--nproc_per_node=8", args)
        self.assertIn("--config.train.gradient_accumulation_steps=12", args)

    def test_existing_platform_supplies_communication_details(self):
        for rank_key in ("SENSECORE_PYTORCH_NODE_RANK", "NODE_RANK", "RANK", "SLURM_NODEID"):
            with self.subTest(rank_key=rank_key):
                result = self.launch(1, "--dry-run", environment={
                    rank_key: "1", "MASTER_ADDR": "192.0.2.10", "MASTER_PORT": "29599",
                })
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout)
                self.assertIn("--node_rank=1", args)
                self.assertIn("--master_addr=192.0.2.10", args)
                self.assertIn("--master_port=29599", args)

    def test_actual_launch_requires_platform_metadata_before_loading_conda(self):
        result = self.launch(1)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("No such file", result.stderr)

    def test_h20_matrix_changes_only_hardware_batching_and_output_paths(self):
        for row in range(1, 9):
            with self.subTest(row=row):
                result = self.launch(row, "--dry-run", hardware="h20_8gpu")
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout)
                for flag in ("--standalone", "--nnodes=1", "--nproc_per_node=8"):
                    self.assertIn(flag, args)
                self.assertFalse(any(arg.startswith(("--node_rank=", "--master_addr=", "--master_port=")) for arg in args))
                self.assertIn("--config.sample.num_batches_per_epoch=24", args)
                self.assertIn("--config.train.gradient_accumulation_steps=24", args)
                self.assertIn(f"--config.run_name=sd35_pickscore_h20_8gpu_shared_exp_clip{row}_kl1e-4", args)
                # Compare the complete trainer argv: only batching and hardware
                # labels in output paths may differ from the checked A6000 run.
                baseline = shlex.split(self.launch(row, "--dry-run").stdout)
                baseline = baseline[next(i for i, arg in enumerate(baseline) if arg.startswith("--config=")):]
                normalized = [arg.replace("h20_8gpu", "a6000_16gpu") for arg in args]
                normalized = [
                    arg.replace("num_batches_per_epoch=24", "num_batches_per_epoch=12")
                       .replace("gradient_accumulation_steps=24", "gradient_accumulation_steps=12")
                    for arg in normalized
                ]
                normalized = normalized[next(i for i, arg in enumerate(normalized) if arg.startswith("--config=")):]
                self.assertEqual(normalized, baseline)

    def test_h20_is_standalone_despite_inherited_distributed_settings(self):
        result = self.launch(6, "--dry-run", hardware="h20_8gpu", environment={
            "NNODES": "2", "NODE_RANK": "1", "SENSECORE_PYTORCH_NODE_RANK": "1",
            "MASTER_ADDR": "192.0.2.10", "CLIP_RUN": "1", "TASK": "ocr",
        })
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertIn("--standalone", args)
        self.assertIn("--nnodes=1", args)
        self.assertIn("--shared_clip_epsilon_minus=0.3", args)
        self.assertIn("--config.train.gradient_accumulation_steps=24", args)

    def test_h20_actual_command_reaches_torchrun_without_platform_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / "repo").mkdir()
            binaries = temporary / "bin"
            binaries.mkdir()
            (binaries / "activate").write_text("# Test conda activation; no environment changes.\n")
            torchrun = binaries / "torchrun"
            torchrun.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            torchrun.chmod(0o755)
            result = self.launch(1, hardware="h20_8gpu", environment={
                "PATH": f"{binaries}:{os.defpath}",
                "CONDA_ROOT": str(temporary), "REPO_DIR": str(temporary / "repo"),
            })
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--standalone", result.stdout.splitlines())
            self.assertIn("--nnodes=1", result.stdout.splitlines())
            self.assertIn("--config.train.gradient_accumulation_steps=24", result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()

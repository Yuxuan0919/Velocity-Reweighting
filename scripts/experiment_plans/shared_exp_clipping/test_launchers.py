"""Check the handoff commands without loading a model or starting GPU workers."""

import os
from pathlib import Path
import shlex
import subprocess
import unittest


DIRECTORY = Path(__file__).resolve().parent


class FixedLauncherTests(unittest.TestCase):
    def launch(self, row, *args, environment=None):
        script = DIRECTORY / f"run_sd3_a6000_16gpu_shared_exp_clip{row}_kl1e-4.sh"
        env = {"PATH": os.defpath}
        env.update(environment or {})
        return subprocess.run(
            ["bash", str(script), *args],
            cwd="/tmp", env=env, capture_output=True, text=True, timeout=10,
        )

    def test_eight_handoff_commands_preserve_the_document_matrix(self):
        matrix = [
            ("upper", "1", "2"), ("upper", "1", "1"),
            ("two_sided", "0.6", "2"), ("upper", "1", "3"),
            ("upper", "1", "0.6"), ("two_sided", "0.3", "2"),
            ("upper", "1", "5"), ("upper", "1", "0.3"),
        ]
        for row, (variant, negative, positive) in enumerate(matrix, 1):
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


if __name__ == "__main__":
    unittest.main()

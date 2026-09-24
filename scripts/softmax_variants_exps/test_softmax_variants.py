"""CPU checks for the formulas, gathered group ordering and launch commands."""

import ast
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import unittest

import numpy as np

from weight_mappings import compute_variant_weights


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = json.loads((HERE / "experiments.json").read_text())


def weights(rewards, variant, c=1.0, p=1.0):
    return compute_variant_weights(
        ["prompt"] * len(rewards), [0] * len(rewards), rewards,
        variant, c=c, p=p, expected_group_size=len(rewards),
    )


class WeightFormulaTests(unittest.TestCase):
    def test_all_18_against_direct_formulas(self):
        rewards = np.array([
            -1.2, -0.2, -0.05, 0.1, 0.15, 0.2, 0.23, 0.25,
            0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42,
            0.44, 0.46, 0.48, 0.5, 0.55, 0.6, 0.7, 0.9,
        ])
        for entry in PLAN:
            group, c, p = entry["variant"], entry["c"], entry["p"]
            with self.subTest(run=entry["run"]):
                if group == "A":
                    f = (rewards >= (1 - c) * rewards.max()).astype(float)
                elif group == "B":
                    f = np.exp(rewards / (c * np.std(rewards, ddof=0)))
                elif group == "C":
                    f = np.maximum(rewards, 0)
                elif group == "D":
                    f = 1 / (1 + np.exp(-(rewards - 0.5) / c))
                elif group == "E":
                    f = np.maximum(rewards, 0) ** p
                else:
                    f = np.exp(rewards / (c * np.mean(np.abs(rewards - rewards.mean()))))
                actual = weights(rewards, group, c, p)
                np.testing.assert_allclose(actual, f / f.mean(), rtol=2e-13, atol=1e-14)
                self.assertAlmostEqual(actual.sum(), 24)
                self.assertTrue(np.all(actual >= 0))

    def test_near_max_ties_threshold_and_no_positive_group(self):
        np.testing.assert_array_equal(weights([0.9, 0.9, 0.8, -1], "A", 0), [2, 2, 0, 0])
        np.testing.assert_allclose(weights([1, 0.75, 0.74, -1], "A", 0.25), [2, 2, 0, 0])
        one_hot = weights(list(range(24)), "A", 0)
        self.assertEqual(one_hot.max(), 24)  # No inherited NFT clipping to [0, 2].
        for c in (0, 0.05, 0.1, 0.25):
            for rewards in ([-2, -1, -0.1], [-2, -1, 0], [0, 0, 0]):
                np.testing.assert_array_equal(weights(rewards, "A", c), np.ones(3))

    def test_zero_scales_and_zero_mapped_rewards(self):
        for group in ("B", "F"):
            for reward in (-3, 0, 0.5):
                np.testing.assert_array_equal(weights([reward] * 24, group), np.ones(24))
        for group in ("C", "E"):
            np.testing.assert_array_equal(weights([-3, -1, 0], group, p=0.5), np.ones(3))
        np.testing.assert_allclose(weights([-1, 0, 0.25, 1], "E", p=0.5), [0, 0, 4/3, 8/3])

    def test_sigmoid_underflow_preserves_relative_weights(self):
        rewards = np.array([-100.03, -100.02, -100.01, -100.0])
        relative = np.exp((rewards - rewards.max()) / 0.01)
        actual = weights(rewards, "D", c=0.01)
        np.testing.assert_allclose(actual, relative / relative.mean(), rtol=3e-12)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(weights([100] * 4, "D", c=0.01), np.ones(4))

    def test_softmax_scale_and_translation_invariance(self):
        rewards = np.array([-1.3, 0.1, 0.2, 0.4])
        for group in ("B", "F"):
            expected = weights(rewards, group, c=0.75)
            for transformed in (rewards + 20, rewards * 1e-200, rewards * 1e200):
                np.testing.assert_allclose(weights(transformed, group, c=0.75), expected, rtol=1e-12)

    def test_rank_major_gather_keeps_rollout_groups_separate(self):
        # Four ranks, two rollout batches, six samples/rank/batch, same prompt.
        reward0 = np.linspace(-0.4, 0.9, 24)
        reward1 = np.linspace(0.1, 0.5, 24) ** 2
        gathered = np.concatenate([
            np.concatenate([reward0[rank*6:(rank+1)*6], reward1[rank*6:(rank+1)*6]])
            for rank in range(4)
        ])
        batch_ids = np.tile(np.repeat([0, 1], 6), 4)
        actual = compute_variant_weights(
            ["same"] * 48, batch_ids, np.repeat(gathered[:, None], 10, axis=1),
            "B", c=0.5, expected_group_size=24,
        )
        expected0, expected1 = weights(reward0, "B", c=0.5), weights(reward1, "B", c=0.5)
        for rank, row in enumerate(actual.reshape(4, 12)):
            np.testing.assert_allclose(row[:6], expected0[rank*6:(rank+1)*6])
            np.testing.assert_allclose(row[6:], expected1[rank*6:(rank+1)*6])
        # Different prompts in the same rollout must also stay separate.
        np.testing.assert_allclose(
            compute_variant_weights(["a", "b", "a", "b"], [0]*4, [1, 4, 3, 4], "C"),
            [0.5, 1, 1.5, 1],
        )

    def test_invalid_data_is_rejected(self):
        for group, c, p in (("A", -1, 1), ("B", 0, 1), ("D", 0, 1), ("E", 1, 0), ("F", -1, 1)):
            with self.assertRaises(ValueError):
                weights([1, 2], group, c, p)
        for bad in ([1, np.nan], [1, np.inf]):
            with self.assertRaises(ValueError):
                weights(bad, "C")
        with self.assertRaisesRegex(ValueError, "expected 24"):
            compute_variant_weights(["p"]*2, [0]*2, [1, 2], "C", expected_group_size=24)
        with self.assertRaisesRegex(ValueError, "identical"):
            compute_variant_weights(["p"], [0], [[1, 2]], "C")


class VelocityLossTests(unittest.TestCase):
    def test_velocity_loss_is_x_prediction_equivalent(self):
        rng = np.random.default_rng(7)
        shape = (8, 3, 4)
        x0 = rng.normal(size=shape)
        noise = rng.normal(size=shape)
        forward_v = rng.normal(size=shape)
        old_v = rng.normal(size=shape)
        t = rng.uniform(0.05, 0.95, size=(shape[0], 1, 1))
        correction = rng.normal(size=(shape[0], 1, 1))
        reduce_dims = (1, 2)

        xt = (1.0 - t) * x0 + t * noise
        forward_x = xt - t * forward_v
        old_x = xt - t * old_v
        clean_x_discrepancy = x0 - old_x
        target_x = old_x + correction * clean_x_discrepancy
        alpha_x = 1.0 / np.mean(np.abs(clean_x_discrepancy), axis=reduce_dims, keepdims=True)
        x_loss = np.mean(alpha_x * (forward_x - target_x) ** 2, axis=reduce_dims)

        clean_v_discrepancy = (noise - x0) - old_v
        target_v = old_v + correction * clean_v_discrepancy
        alpha_v = 1.0 / np.mean(np.abs(clean_v_discrepancy), axis=reduce_dims, keepdims=True)
        v_loss = np.mean(t * alpha_v * (forward_v - target_v) ** 2, axis=reduce_dims)

        np.testing.assert_allclose(v_loss, x_loss, rtol=2e-13, atol=1e-14)

    def test_all_trainers_use_only_the_active_velocity_loss(self):
        expected_fragments = (
            "clean_v_discrepancy = (noise.float() - x0.float()) - old_v_prediction",
            "target_v_prediction = old_v_prediction + correction_coefficient_expanded * clean_v_discrepancy",
            "trajectory_alpha * t_expanded.float() * (forward_v_prediction - target_v_prediction) ** 2",
            "ori_policy_loss = target_v_prediction_loss",
        )
        forbidden_active_names = {
            "forward_x_prediction",
            "old_x_prediction",
            "clean_x_discrepancy",
            "trajectory_alpha_base_x_prediction",
            "target_x_prediction",
            "target_x_prediction_loss",
        }
        trainers = sorted(HERE.glob("train_nft_sd3_*.py"))
        self.assertEqual(len(trainers), 6)
        for path in trainers:
            text = path.read_text()
            tree = ast.parse(text)
            active_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            with self.subTest(trainer=path.name):
                for fragment in expected_fragments:
                    self.assertIn(fragment, text)
                self.assertTrue(forbidden_active_names.isdisjoint(active_names))


class IntegrationTests(unittest.TestCase):
    def test_plan_matches_latex_table(self):
        tex = (ROOT / "assets/softmax_variant/variants.tex").read_text()
        table = tex.split(r"\label{tab:weight-variant-planned-sweep}", 1)[1].split(r"\end{table}", 1)[0]
        rows = []
        group = None
        planned_runs = {entry["run"] for entry in PLAN}
        for line in table.splitlines():
            match = re.match(r"\s*(\d{2})\s*&", line)
            if not match:
                continue
            if match.group(1) not in planned_runs:
                continue
            _, experiment, configuration, _ = line.split("&")
            group_match = re.search(r"([A-F]):", experiment)
            if group_match:
                group = group_match.group(1)
            rows.append((match.group(1), group, configuration.strip()))
        self.assertEqual(len(rows), len(PLAN))
        for (run, group, config), entry in zip(rows, PLAN, strict=True):
            self.assertEqual((run, group), (entry["run"], entry["variant"]))
            if entry["parameter"] == "c":
                self.assertEqual(config, "$c=" + entry["value"] + "$")
            elif group == "E":
                self.assertEqual(config, "$f(R)=[R]_+^{" + entry["value"] + "}$")
            else:
                self.assertEqual(config, "$f(R)=[R]_+$")

    def test_original_training_lines_retained_and_python_parses(self):
        original = (
            ROOT / "scripts/alpha_exps/train_nft_sd3_ours-1.singleloss-alpha.py"
        ).read_text().splitlines()
        for path in HERE.glob("train_nft_sd3_*.py"):
            text = path.read_text()
            ast.parse(text)
            modified = iter(text.splitlines())
            for old in original:
                indent = old[:len(old)-len(old.lstrip())]
                commented = indent + "# " + old[len(indent):] if old.strip() else old
                self.assertTrue(
                    any(line in (old, commented) for line in modified),
                    f"{path.name} deleted or reordered original line: {old!r}",
                )
        for path in (HERE / "pickscore_4090").glob("*.sh"):
            subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True)

    def dry_run(self, entry, **settings):
        env = os.environ.copy()
        for key in ("WORLD_SIZE", "RANK", "NODE_RANK", "SENSECORE_PYTORCH_NNODES",
                    "SENSECORE_PYTORCH_NODE_RANK", "SLURM_NODEID", "SAVE_DIR",
                    "RUN_NAME", "EXPERIMENT_BETA", "MASTER_ADDR"):
            env.pop(key, None)
        env.update(DRY_RUN="1", NNODES="1", NPROC_PER_NODE="8", PER_DEVICE_BATCH="6")
        env.update(settings)
        return subprocess.run(
            ["bash", str(HERE / "pickscore_4090" / entry["launcher"])],
            env=env, cwd="/tmp", text=True, capture_output=True,
        )

    def test_all_launchers_route_to_correct_trainers(self):
        saves = set()
        for entry in PLAN:
            result = self.dry_run(entry)
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout.splitlines()[-1])
            self.assertIn("scripts/softmax_variants_exps/" + entry["trainer"], command)
            flags = dict(arg[2:].split("=", 1) for arg in command if arg.startswith("--") and "=" in arg)
            self.assertEqual(float(flags["weight_c"]), entry["c"])
            self.assertEqual(float(flags["weight_p"]), entry["p"])
            self.assertEqual(flags["config"], "config/nft.py:sd3_pickscore")
            self.assertEqual(flags["config.sample.num_image_per_prompt"], "24")
            self.assertEqual(flags["config.train.gradient_accumulation_steps"], "24")
            self.assertEqual(flags["config.train.trajectory_alpha_prediction"], "old_prediction")
            saves.add(flags["config.save_dir"])
        self.assertEqual(len(saves), 18)

    def test_multinode_and_invalid_batch_settings(self):
        for processes, accumulation in (("8", "12"), ("6", "16")):
            result = self.dry_run(
                PLAN[0], NNODES="2", NODE_RANK="1", NPROC_PER_NODE=processes,
                MASTER_ADDR="192.0.2.1", EXPERIMENT_BETA="5.0",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            command = shlex.split(result.stdout.splitlines()[-1])
            self.assertIn("--node_rank=1", command)
            self.assertIn("--config.train.gradient_accumulation_steps=" + accumulation, command)
            self.assertIn("--config.beta=5.0", command)
        invalid = self.dry_run(PLAN[0], PER_DEVICE_BATCH="1")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("24-image prompt groups", invalid.stderr)
        missing_host = self.dry_run(PLAN[0], NNODES="2", NODE_RANK="0")
        self.assertNotEqual(missing_host.returncode, 0)
        self.assertIn("Set MASTER_ADDR", missing_host.stderr)


if __name__ == "__main__":
    unittest.main()

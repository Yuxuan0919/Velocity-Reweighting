import unittest

import numpy as np

from flow_grpo.shared_coefficient_clipping import compute_shared_exp_shifts


def direct_formula(rewards, batches, prompts, variant, epsilon_plus, epsilon_minus):
    """Independent, uncentered evaluation of the document's equations."""
    rewards = np.asarray(rewards, dtype=np.float64)
    sigma = max(np.std(rewards, ddof=0), 1e-6)
    result = np.empty_like(rewards)
    keys = list(zip(batches, prompts))
    for key in set(keys):
        indices = [i for i, sample_key in enumerate(keys) if sample_key == key]
        unnormalized = np.exp(rewards[indices] / sigma)
        delta = len(indices) * unnormalized / unnormalized.sum() - 1
        positive = max(np.maximum(delta, 0))
        negative = max(np.maximum(-delta, 0))
        coefficient = 1.0
        if variant != "unclipped" and positive:
            coefficient = min(coefficient, epsilon_plus / positive)
        if variant == "two_sided" and negative:
            coefficient = min(coefficient, epsilon_minus / negative)
        result[indices] = coefficient * delta
    return result


class SharedCoefficientClippingTests(unittest.TestCase):
    def test_all_eight_configurations_match_independent_equations(self):
        rng = np.random.default_rng(34)
        rewards = np.concatenate([rng.normal(size=24), np.r_[8.0, np.zeros(23)], np.tile([-1.0, 1.0], 12)])
        batches = np.repeat([0, 1, 2], 24)
        prompts = np.full(72, 9)
        matrix = [
            ("upper", 2.0, 1.0),
            ("upper", 1.0, 1.0),
            ("two_sided", 2.0, 0.6),
            ("upper", 3.0, 1.0),
            ("upper", 0.6, 1.0),
            ("two_sided", 2.0, 0.3),
            ("upper", 5.0, 1.0),
            ("upper", 0.3, 1.0),
            ("unclipped", 2.0, 1.0),
        ]
        raw = direct_formula(rewards, batches, prompts, "unclipped", 2.0, 1.0)
        for variant, positive, negative in matrix:
            with self.subTest(variant=variant, positive=positive, negative=negative):
                actual, stats = compute_shared_exp_shifts(
                    rewards, batches, prompts, 24,
                    variant=variant, epsilon_plus=positive, epsilon_minus=negative,
                )
                expected = direct_formula(rewards, batches, prompts, variant, positive, negative)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(actual.dtype, np.float32)
                self.assertLess(stats["zero_sum_max_abs"], 2e-6)
                self.assertGreaterEqual(float(actual.min()), -negative - 1e-7)
                if variant != "unclipped":
                    self.assertLessEqual(float(actual.max()), positive + 1e-7)
                for group in actual.reshape(3, 24):
                    self.assertAlmostEqual(float((1.0 + group.astype(np.float64)).mean()), 1.0, places=6)
                # Each whole group keeps precisely one scale, including negative samples.
                ratios = actual.reshape(3, 24) / raw.reshape(3, 24)
                np.testing.assert_allclose(ratios, np.repeat(ratios[:, :1], 24, axis=1), rtol=1e-6)

    def test_global_population_std_and_full_scattered_prompt_groups(self):
        # Rank-major gathering scatters groups and repeats the prompt in another batch.
        rewards = np.array([0.0, 0.0, 1.0, 10.0])
        batches = np.array([0, 1, 0, 1])
        prompts = np.array([7, 7, 7, 7])
        actual, stats = compute_shared_exp_shifts(rewards, batches, prompts, 2, variant="unclipped")
        expected = direct_formula(rewards, batches, prompts, "unclipped", 2.0, 1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        self.assertAlmostEqual(stats["gamma"], np.sqrt(17.6875))
        self.assertEqual(stats["group_count"], 2)
        self.assertNotAlmostEqual(abs(float(actual[0])), abs(float(actual[1])))
        permutation = np.array([2, 0, 3, 1])
        permuted, _ = compute_shared_exp_shifts(
            rewards[permutation], batches[permutation], prompts[permutation], 2, variant="unclipped",
        )
        np.testing.assert_array_equal(permuted, actual[permutation])
        # Two different prompts in one batch must also remain separate groups.
        same_batch, _ = compute_shared_exp_shifts(
            rewards, np.zeros(4), np.array([7, 8, 7, 8]), 2, variant="unclipped",
        )
        np.testing.assert_array_equal(same_batch, actual)

    def test_reward_std_floor_applies_to_small_nonconstant_rewards(self):
        shifts, stats = compute_shared_exp_shifts(
            [0.0, 1e-8], [0, 0], [5, 5], 2, variant="unclipped",
        )
        positive_shift = (np.exp(0.01) - 1.0) / (np.exp(0.01) + 1.0)
        np.testing.assert_allclose(shifts, [-positive_shift, positive_shift], rtol=1e-6)
        self.assertEqual(stats["gamma"], 1e-6)

    def test_equal_rewards_have_zero_shift_and_no_clipping(self):
        # A large common offset must not overflow exp(r / 1e-6).
        for variant in ("unclipped", "upper", "two_sided"):
            shifts, stats = compute_shared_exp_shifts(
                np.full(48, 1e6), np.repeat([0, 1], 24), np.zeros(48), 24,
                variant=variant, epsilon_plus=0.3, epsilon_minus=0.3,
            )
            np.testing.assert_array_equal(shifts, np.zeros(48))
            self.assertEqual(stats["gamma"], 1e-6)
            self.assertEqual(stats["c_eff_mean"], 1.0)
            self.assertEqual(stats["zero_std_group_ratio"], 1.0)
            self.assertEqual(stats["negative_bound_active_fraction"], 0.0)
            self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_negative_bound_diagnostics_include_inactive_groups(self):
        rewards = np.r_[np.tile([-1.0, 1.0], 12), 10.0, np.zeros(23)]
        batches = np.repeat([0, 1], 24)
        prompts = np.zeros(48)
        upper, upper_stats = compute_shared_exp_shifts(
            rewards, batches, prompts, 24, epsilon_plus=2.0, epsilon_minus=0.3,
        )
        both, both_stats = compute_shared_exp_shifts(
            rewards, batches, prompts, 24, variant="two_sided", epsilon_plus=2.0, epsilon_minus=0.3,
        )
        raw, _ = compute_shared_exp_shifts(rewards, batches, prompts, 24, variant="unclipped")
        self.assertEqual(both_stats["negative_bound_active_fraction"], 0.5)
        self.assertEqual(upper_stats["negative_bound_active_fraction"], 0.5)
        self.assertEqual(upper_stats["negative_bound_0.3/active_fraction"], 0.5)
        self.assertEqual(upper_stats["epsilon_plus"], 2.0)
        self.assertEqual(upper_stats["epsilon_minus"], 0.3)
        upper_coefficients = upper[[0, 24]] / raw[[0, 24]]
        both_coefficients = both[[0, 24]] / raw[[0, 24]]
        self.assertAlmostEqual(
            both_stats["negative_bound_extra_shrink_mean"],
            float(np.mean(upper_coefficients - both_coefficients)), places=7,
        )
        np.testing.assert_array_equal(upper[24:], both[24:])
        self.assertAlmostEqual(float(both[:24].min()), -0.3, places=6)
        self.assertAlmostEqual(float(upper[24:].max()), 2.0, places=6)

    def test_upper_only_does_not_apply_counterfactual_negative_bound(self):
        args = (np.tile([-1.0, 1.0], 12), np.zeros(24), np.zeros(24), 24)
        first, default_stats = compute_shared_exp_shifts(*args, epsilon_minus=1.0)
        second, stats = compute_shared_exp_shifts(*args, epsilon_minus=0.3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(stats["negative_bound_active_fraction"], 1.0)
        self.assertEqual(default_stats["negative_bound_active_fraction"], 0.0)
        for negative in (0.6, 0.3):
            _, selected_stats = compute_shared_exp_shifts(*args, epsilon_minus=negative)
            self.assertEqual(default_stats[f"negative_bound_{negative}/active_fraction"], 1.0)
            self.assertEqual(
                default_stats[f"negative_bound_{negative}/extra_shrink_mean"],
                selected_stats["negative_bound_extra_shrink_mean"],
            )

    def test_input_arrays_are_not_mutated(self):
        inputs = [np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1]), np.array([5, 5, 5, 5])]
        snapshots = [array.copy() for array in inputs]
        compute_shared_exp_shifts(*inputs, 2)
        for original, snapshot in zip(inputs, snapshots):
            np.testing.assert_array_equal(original, snapshot)

    def test_invalid_inputs_fail_before_training(self):
        args = ([0.0, 1.0], [0, 0], [3, 3], 2)
        for kwargs in (
            {"variant": "elementwise"}, {"epsilon_plus": 0}, {"epsilon_plus": np.nan},
            {"epsilon_plus": np.inf}, {"epsilon_minus": 0}, {"epsilon_minus": 1.1},
            {"epsilon_minus": np.nan}, {"epsilon_minus": np.inf},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compute_shared_exp_shifts(*args, **kwargs)
        for bad_args in (
            ([], [], [], 2), ([np.nan, 0], [0, 0], [3, 3], 2),
            ([np.inf, 0], [0, 0], [3, 3], 2), ([0, 1], [0], [3, 3], 2),
            ([0, 1], [0, 1], [3, 3], 2), ([0, 1], [0, 0], [3, 3], 3),
            ([0, 1], [0, 0], [3, 3], 1), ([0, 1], [0, 0], [3, 3], 2.5),
        ):
            with self.subTest(args=bad_args), self.assertRaises(ValueError):
                compute_shared_exp_shifts(*bad_args)


if __name__ == "__main__":
    unittest.main()

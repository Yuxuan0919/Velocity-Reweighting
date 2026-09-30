"""CPU integration checks for the step-420 Linear scale intervention."""

import ast
import copy
import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

try:
    import ml_collections
except ImportError:
    ml_collections = None

from flow_grpo.checkpoint_fork import resolve_linear_fork_checkpoint
from flow_grpo.target_scale import TargetScaleSchedule
from test_linear_ablation_trainer import TRAINER, TREE, trainer_functions


class LinearCheckpointForkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original/checkpoints/checkpoint-420"
        self.output = self.root / "fork-output"
        self.config = SimpleNamespace(
            resume_from="", save_dir=str(self.output), logdir=str(self.root / "fork-logs"),
            num_epochs=1000, train=SimpleNamespace(ema=True, learning_rate=3e-4),
            per_prompt_stat_tracking=True,
        )
        self.flags = SimpleNamespace(
            fork_from_checkpoint=str(self.source), require_resume=True,
            reward_mapping="linear", linear_target_scale=5.0, mass_shift_transform="none",
        )
        self.state = {
            "epoch": 420, "global_step": 420,
            "reward_mapping": "linear", "linear_target_scale": 15.0,
        }
        self.parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.optimizer = torch.optim.AdamW([self.parameter], lr=3e-4)
        self.parameter.grad = torch.tensor([0.4, -0.3])
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.write_checkpoint(self.source, self.state)
        self.namespace = trainer_functions(
            "_has_saved_adapter", "_is_resumable_checkpoint", "_find_latest_checkpoint",
            "_validate_resume_objective", "resolve_resume_checkpoint", "save_ckpt",
            re=re, _CHECKPOINT_PATTERN=re.compile(r"^checkpoint-(\d+)$"),
            FLAGS=self.flags, resolve_linear_fork_checkpoint=resolve_linear_fork_checkpoint,
            dist=SimpleNamespace(broadcast_object_list=lambda *args, **kwargs: None),
        )

    def write_checkpoint(self, directory, state):
        for adapter in ("default", "old"):
            path = directory / f"lora_{adapter}"
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}")
            value = self.parameter.detach().clone() if adapter == "default" else torch.tensor([0.9, 1.9])
            torch.save(value, path / "adapter_model.bin")
        for name, value in {
            "training_state": state,
            "optimizer": self.optimizer.state_dict(),
            "scaler": {"scale": 65536.0, "growth_tracker": 7},
            "ema_state": {"shadow": torch.tensor([0.95, 1.95]), "step": 420},
            "stat_tracker": {"prompt": [0.8, 0.9]},
        }.items():
            torch.save(value, directory / f"{name}.pt")
        (directory / "_SUCCESS").write_text(f"global_step={state['global_step']}\nepoch={state['epoch']}\n")

    def resolve(self):
        # The real resolver broadcasts its selection; emulate only the CUDA
        # device lookup, keeping real checkpoint discovery and torch.load.
        with mock.patch.object(torch.cuda, "current_device", return_value=0):
            return self.namespace["resolve_resume_checkpoint"](self.config, 0, TargetScaleSchedule(5.0))

    def source_hashes(self):
        return {str(p.relative_to(self.source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.source.rglob("*") if p.is_file()}

    def test_initial_fork_keeps_source_immutable_and_has_explicit_provenance(self):
        before = self.source_hashes()
        self.assertEqual(self.resolve(), str(self.source))
        self.assertEqual(self.config.checkpoint_fork["source_global_step"], 420)
        self.assertEqual(self.config.checkpoint_fork["source_target_scale"], 15.0)
        self.assertEqual(self.config.checkpoint_fork["target_scale"], 5.0)
        self.assertFalse(self.config.checkpoint_fork["source_rng_restored"])
        self.assertFalse(self.output.exists())
        self.assertEqual(before, self.source_hashes())

    @unittest.skipIf(ml_collections is None, "ml_collections is a training dependency")
    def test_real_locked_training_config_accepts_declared_provenance_field(self):
        from config.base import get_config

        config = get_config()
        config.save_dir = self.config.save_dir
        config.logdir = self.config.logdir
        config.num_epochs = self.config.num_epochs
        config.train.ema = True
        config.per_prompt_stat_tracking = True
        config.lock()
        self.config = config
        self.assertEqual(self.resolve(), str(self.source))
        self.assertTrue(config.is_locked)
        self.assertEqual(config.checkpoint_fork.source_global_step, 420)
        self.assertEqual(config.checkpoint_fork.target_scale, 5.0)

    def test_invalid_source_state_cannot_bypass_normal_objective_checks(self):
        changes = (
            {"global_step": 450}, {"epoch": 419}, {"global_step": True},
            {"linear_target_scale": 5.0}, {"reward_mapping": "exponential"},
            {"mass_shift_transform": "square_both"},
            {"target_scale_schedule": TargetScaleSchedule(15.0, "linear_decay", 5.0).metadata()},
        )
        for change in changes:
            with self.subTest(change=change):
                torch.save({**self.state, **change}, self.source / "training_state.pt")
                with self.assertRaises(FileNotFoundError):
                    self.resolve()
        with self.assertRaises(ValueError):
            self.namespace["_validate_resume_objective"](self.state, "linear", 5.0)

    def test_missing_source_or_incomplete_state_never_starts_from_scratch(self):
        for name in ("_SUCCESS", "optimizer.pt", "ema_state.pt", "scaler.pt", "stat_tracker.pt",
                     "lora_old/adapter_model.bin"):
            with self.subTest(missing=name):
                path = self.source / name
                content = path.read_bytes()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    self.resolve()
                path.write_bytes(content)
        self.flags.fork_from_checkpoint = str(self.root / "missing/checkpoint-420")
        with self.assertRaises(FileNotFoundError):
            self.resolve()

    def test_wrong_path_and_overlapping_or_symlinked_output_are_rejected(self):
        original_source = self.flags.fork_from_checkpoint
        for value in ("checkpoint-420", str(self.source.parent / "checkpoint-450")):
            self.flags.fork_from_checkpoint = value
            with self.assertRaises(FileNotFoundError):
                self.resolve()
        self.flags.fork_from_checkpoint = original_source
        source_run = self.source.parent.parent
        for value in (source_run, self.source, source_run / "other-output", self.root):
            self.config.save_dir = str(value)
            with self.assertRaises(FileNotFoundError):
                self.resolve()
        alias = self.root / "source-alias"
        alias.symlink_to(source_run, target_is_directory=True)
        self.config.save_dir = str(alias / "fork")
        with self.assertRaises(FileNotFoundError):
            self.resolve()

    def test_partial_destination_does_not_silently_restart_from_source(self):
        (self.output / "checkpoints/checkpoint-450").mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, "nonempty"):
            self.resolve()

    def test_child_save_and_resume_preserve_fork_and_do_not_rewind_to_420(self):
        self.resolve()
        before = self.source_hashes()
        child = self.output / "checkpoints/checkpoint-450"
        self.write_checkpoint(child, {**self.state, "epoch": 450, "global_step": 450})
        self.namespace["save_ckpt"](
            str(self.output), SimpleNamespace(module=mock.Mock()), 450, 0,
            SimpleNamespace(state_dict=lambda: {"step": 450}), self.config, self.optimizer,
            SimpleNamespace(state_dict=lambda: {"scale": 65536.0}), 450,
            stat_tracker=SimpleNamespace(state_dict=lambda: {"prompt": [0.9]}),
            target_scale_schedule=TargetScaleSchedule(5.0),
            successful_optimizer_updates=30, optimizer_count_start_step=420,
        )
        state = torch.load(child / "training_state.pt", weights_only=True)
        self.assertEqual(state["linear_target_scale"], 5.0)
        self.assertEqual(state["checkpoint_fork"], self.config.checkpoint_fork)
        self.assertEqual(state["global_step"], 450)
        self.assertEqual(state["optimizer_count_start_step"], 420)
        self.assertEqual(before, self.source_hashes())
        shutil.rmtree(self.source.parent.parent)
        self.assertEqual(self.resolve(), str(child))
        self.config.resume_from = str(child)
        self.assertEqual(self.resolve(), str(child))

    def test_unrelated_c5_child_and_explicit_source_as_resume_are_rejected(self):
        self.resolve()
        child = self.output / "checkpoints/checkpoint-450"
        self.write_checkpoint(child, {**self.state, "epoch": 450, "global_step": 450,
                                      "linear_target_scale": 5.0})
        with self.assertRaisesRegex(FileNotFoundError, "does not belong"):
            self.resolve()
        self.config.resume_from = str(self.source)
        with self.assertRaisesRegex(FileNotFoundError, "does not belong"):
            self.resolve()

    def test_fork_child_cannot_use_ordinary_resume_to_overwrite_source(self):
        self.resolve()
        child = self.output / "checkpoints/checkpoint-450"
        self.write_checkpoint(child, {**self.state, "epoch": 450, "global_step": 450,
                                      "linear_target_scale": 5.0,
                                      "checkpoint_fork": self.config.checkpoint_fork})
        self.flags.fork_from_checkpoint = ""
        self.config.resume_from = str(child)
        self.config.save_dir = str(self.source.parent.parent)
        before = self.source_hashes()
        with self.assertRaisesRegex(FileNotFoundError, "fork continuation requires"):
            self.resolve()
        self.assertEqual(self.source_hashes(), before)

    def run_actual_resume_block(self):
        self.config.resume_from = self.resolve()
        default = torch.nn.Parameter(torch.zeros(2))
        old = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        optimizer = torch.optim.AdamW([default], lr=3e-4)
        restored = {}

        def load_adapter(directory, adapter_name, is_trainable):
            target = default if adapter_name == "default" else old
            with torch.no_grad():
                target.copy_(torch.load(Path(directory) / "adapter_model.bin", weights_only=True))

        model = SimpleNamespace(load_adapter=load_adapter, set_adapter=lambda name: None)
        namespace = dict(self.namespace)
        namespace.update(
            config=self.config, transformer_ddp=SimpleNamespace(module=model), optimizer=optimizer,
            scaler=SimpleNamespace(load_state_dict=lambda state: restored.update(scaler=state)),
            ema=SimpleNamespace(load_state_dict=lambda state, **kwargs: restored.update(ema=state)),
            stat_tracker=SimpleNamespace(load_state_dict=lambda state: restored.update(stat_tracker=state)),
            device="cpu", enable_amp=True, first_epoch=0, global_step=0,
        )
        main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        block = next(node for node in main.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Attribute) and node.test.attr == "resume_from")
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(TRAINER), "exec"), namespace)
        return namespace, default, old, optimizer, restored

    def test_actual_resume_restores_models_optimizer_scaler_ema_and_step(self):
        namespace, default, old, optimizer, restored = self.run_actual_resume_block()
        torch.testing.assert_close(default, self.parameter)
        torch.testing.assert_close(old, torch.tensor([0.9, 1.9]))
        original_moments = self.optimizer.state[self.parameter]
        for key, value in original_moments.items():
            torch.testing.assert_close(optimizer.state[default][key], value)
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)
        self.assertEqual(namespace["global_step"], 420)
        self.assertEqual(namespace["first_epoch"], 420)
        self.assertEqual(namespace["optimizer_count_start_step"], 420)
        self.assertEqual(restored["scaler"]["growth_tracker"], 7)
        self.assertEqual(restored["ema"]["step"], 420)
        self.assertEqual(restored["stat_tracker"], {"prompt": [0.8, 0.9]})

    def test_loaded_learning_rate_mismatch_fails_without_overriding_it(self):
        state = copy.deepcopy(self.optimizer.state_dict())
        state["param_groups"][0]["lr"] = 1e-4
        torch.save(state, self.source / "optimizer.pt")
        with self.assertRaisesRegex(ValueError, "will not override optimizer hyperparameters"):
            self.run_actual_resume_block()
        self.assertEqual(torch.load(self.source / "optimizer.pt", weights_only=True)["param_groups"][0]["lr"], 1e-4)


if __name__ == "__main__":
    unittest.main()

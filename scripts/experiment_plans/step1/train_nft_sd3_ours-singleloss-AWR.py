# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import re
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_enum(
    "reward_mapping",
    "exponential",
    ["exponential", "linear"],
    "Reward mapping f: FlowAWR exp(r/std) or Step 1 max(r, 0). Only f changes.",
)
flags.DEFINE_float(
    "linear_target_scale",
    1.0,
    "Multiplier c on the Linear velocity-target correction (Step 1.1). "
    "1 preserves Step 1; other values require reward_mapping=linear.",
)
flags.DEFINE_bool(
    "awr_variance_gate",
    True,
    "Normalize advantages by their rollout-batch std for OCR/GenEval. "
    "Disable for the raw Step 1 mass shift; keep the same setting in AWR comparisons.",
)
flags.DEFINE_float("awr_variance_floor", 1e-2, "Lower bound for the advantage std in variance gating.")
flags.DEFINE_float("awr_gamma_floor", 1e-6, "Numerical lower bound for the rollout reward std.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def _has_saved_adapter(adapter_dir):
    if not os.path.isdir(adapter_dir):
        return False
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weight_paths = (
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(adapter_dir, "adapter_model.bin"),
    )
    return os.path.isfile(config_path) and any(
        os.path.isfile(path) and os.path.getsize(path) > 0 for path in weight_paths
    )


def _is_resumable_checkpoint(checkpoint_dir, config):
    default_adapter_dir = os.path.join(checkpoint_dir, "lora_default")
    legacy_adapter_dir = os.path.join(checkpoint_dir, "lora")
    if not (_has_saved_adapter(default_adapter_dir) or _has_saved_adapter(legacy_adapter_dir)):
        return False
    old_adapter_dir = os.path.join(checkpoint_dir, "lora_old")
    if not (_has_saved_adapter(old_adapter_dir) or _has_saved_adapter(os.path.join(old_adapter_dir, "old"))):
        return False

    required_files = [
        os.path.join(checkpoint_dir, "optimizer.pt"),
        os.path.join(checkpoint_dir, "scaler.pt"),
        os.path.join(checkpoint_dir, "training_state.pt"),
    ]
    if config.train.ema:
        required_files.append(os.path.join(checkpoint_dir, "ema_state.pt"))
    if config.per_prompt_stat_tracking:
        required_files.append(os.path.join(checkpoint_dir, "stat_tracker.pt"))

    return all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in required_files)


def _find_latest_checkpoint(search_dir, config):
    if not search_dir:
        return ""
    search_dir = os.path.abspath(os.path.expanduser(search_dir))
    exact_match = _CHECKPOINT_PATTERN.fullmatch(os.path.basename(search_dir))
    if exact_match:
        return search_dir if _is_resumable_checkpoint(search_dir, config) else ""

    nested_checkpoint_dir = os.path.join(search_dir, "checkpoints")
    checkpoint_root = nested_checkpoint_dir if os.path.isdir(nested_checkpoint_dir) else search_dir
    if not os.path.isdir(checkpoint_root):
        return ""

    candidates = []
    for entry in os.scandir(checkpoint_root):
        match = _CHECKPOINT_PATTERN.fullmatch(entry.name)
        if entry.is_dir() and match:
            candidates.append((int(match.group(1)), entry.path))

    uses_completion_markers = any(
        os.path.isfile(os.path.join(checkpoint_dir, "_SUCCESS"))
        for _, checkpoint_dir in candidates
    )
    for _, checkpoint_dir in sorted(candidates, reverse=True):
        if uses_completion_markers and not os.path.isfile(os.path.join(checkpoint_dir, "_SUCCESS")):
            logger.warning("Ignoring checkpoint without completion marker: %s", checkpoint_dir)
            continue
        if _is_resumable_checkpoint(checkpoint_dir, config):
            return checkpoint_dir
        logger.warning("Ignoring incomplete checkpoint: %s", checkpoint_dir)
    return ""


def _validate_resume_objective(training_state, reward_mapping, linear_target_scale):
    # Legacy checkpoints predate target scaling, so their scale is 1.
    saved_scale = training_state.get("linear_target_scale", 1.0)
    saved_mapping = training_state.get("reward_mapping")
    if saved_scale != linear_target_scale or (
        saved_mapping is not None and saved_mapping != reward_mapping
    ):
        raise ValueError(
            "Checkpoint objective does not match this run: "
            f"saved reward_mapping={saved_mapping}, linear_target_scale={saved_scale}; "
            f"requested reward_mapping={reward_mapping}, linear_target_scale={linear_target_scale}. "
            "Use a separate save_dir and clear resume_from to start a new experiment."
        )


def resolve_resume_checkpoint(config, rank):
    requested_path = str(config.resume_from).strip()
    search_dir = requested_path or config.save_dir
    resolution = {"path": "", "error": ""}

    if is_main_process(rank):
        try:
            checkpoint_path = _find_latest_checkpoint(search_dir, config)
            if requested_path and not checkpoint_path:
                resolution["error"] = f"No complete checkpoint found at or under: {requested_path}"
            else:
                if checkpoint_path:
                    training_state = torch.load(
                        os.path.join(checkpoint_path, "training_state.pt"), map_location="cpu"
                    )
                    _validate_resume_objective(
                        training_state, FLAGS.reward_mapping, FLAGS.linear_target_scale
                    )
                resolution["path"] = checkpoint_path
        except Exception as error:
            resolution["error"] = f"Failed to resolve resume checkpoint from {search_dir}: {error}"

    payload = [resolution]
    dist.broadcast_object_list(
        payload,
        src=0,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    resolution = payload[0]
    if resolution["error"]:
        raise FileNotFoundError(resolution["error"])

    checkpoint_path = resolution["path"]
    if is_main_process(rank):
        if checkpoint_path:
            mode = "explicit" if requested_path else "automatic"
            logger.info("Using %s resume checkpoint: %s", mode, checkpoint_path)
        else:
            logger.info("No checkpoint found in %s; starting a new training run.", search_dir)
    return checkpoint_path


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}, "index": idx}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        indices = [example["index"] for example in examples]
        return prompts, metadatas, indices


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "index": idx}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        indices = [example["index"] for example in examples]
        return prompts, metadatas, indices


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def flowawr_reference_decay(rollout_iteration):
    """Old-policy retention coefficient, updated once after each rollout round."""
    return min(0.001 * rollout_iteration, 0.5)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()



def compute_flowawr_advantages(
    rewards, rollout_batch_ids, prompt_indices, group_size, gamma_floor=1e-6,
    variance_gate=False, variance_floor=1e-2, reward_mapping="exponential",
):
    """Return Delta w = f(R) / mean_group(f(R)) - 1 in gathered order.

    Exponential is the unchanged FlowAWR mapping, with rollout-wide gamma.
    Linear follows jingdong_experiment_plan.tex, Eq. (linear-weight):
    f(R) = max(R, 0), with Delta w = 0 when the whole group maps to zero.

    The optional variance gate is an explicit implementation choice for the
    rule-based tasks; leave it disabled for the raw Step 1 mass shift.
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    rollout_batch_ids = np.asarray(rollout_batch_ids).reshape(-1)
    prompt_indices = np.asarray(prompt_indices).reshape(-1)
    if not (len(rewards) == len(rollout_batch_ids) == len(prompt_indices)):
        raise ValueError("Rewards and group identifiers must have the same length")
    if rewards.size == 0:
        raise ValueError("FlowAWR requires a nonempty rollout buffer")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("FlowAWR requires finite rewards")
    if group_size < 2 or gamma_floor <= 0 or variance_floor <= 0:
        raise ValueError("group_size must be at least 2 and numerical floors must be positive")
    if reward_mapping not in ("exponential", "linear"):
        raise ValueError(f"Unknown reward mapping: {reward_mapping!r}")

    gamma = max(float(rewards.std()), gamma_floor) if reward_mapping == "exponential" else None
    advantages = np.empty_like(rewards)
    groups = defaultdict(list)
    for sample_idx, key in enumerate(zip(rollout_batch_ids, prompt_indices, strict=True)):
        groups[(int(key[0]), int(key[1]))].append(sample_idx)

    zero_std_groups = 0
    zero_weight_groups = 0
    for key, indices in groups.items():
        if len(indices) != group_size:
            raise ValueError(f"FlowAWR group {key} has {len(indices)} samples; expected {group_size}")
        group_rewards = rewards[indices]
        zero_std_groups += int(np.all(group_rewards == group_rewards[0]))
        if reward_mapping == "exponential":
            logits = group_rewards / gamma
            weights = np.exp(logits - logits.max())
        else:
            weights = np.maximum(group_rewards, 0.0)
            maximum = weights.max()
            if maximum == 0.0:
                # w = 1, Delta w = 0: still regress to the old-policy target.
                zero_weight_groups += 1
                weights = np.ones_like(weights)
            else:
                # Common scaling cancels in w; avoid overflow/underflow in mean.
                weights = weights / maximum
        advantages[indices] = weights / weights.mean() - 1.0

    raw_advantage_std = float(advantages.std())
    if variance_gate:
        advantages /= max(raw_advantage_std, variance_floor)
    stats = {
        "raw_advantage_std": raw_advantage_std,
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "advantage_min": float(advantages.min()),
        "advantage_max": float(advantages.max()),
        "zero_std_group_ratio": zero_std_groups / len(groups),
        "group_count": len(groups),
        "variance_gate": int(variance_gate),
    }
    if reward_mapping == "exponential":
        stats["gamma"] = gamma
    else:
        stats["negative_reward_ratio"] = float(np.mean(rewards < 0.0))
        stats["zero_weight_group_ratio"] = zero_weight_groups / len(groups)
    return advantages.astype(np.float32), stats


def log_scalars(writer, scalars, step):
    if writer is None:
        return
    for key, value in scalars.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        elif isinstance(value, np.ndarray):
            value = float(np.mean(value))
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            writer.add_scalar(key, value, step)


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
    writer,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata, _ = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}
        log_scalars(
            writer,
            {f"eval_reward/{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
            global_step,
        )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, config, optimizer, scaler, epoch, stat_tracker=None
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora_default = os.path.join(save_root, "lora_default")
        save_root_lora_old = os.path.join(save_root, "lora_old")
        completion_marker = os.path.join(save_root, "_SUCCESS")
        os.makedirs(save_root_lora_default, exist_ok=True)
        os.makedirs(save_root_lora_old, exist_ok=True)
        if os.path.exists(completion_marker):
            os.remove(completion_marker)

        model = transformer_ddp.module

        # Save the actual optimized parameters. Evaluation EMA is kept separately
        # in ema_state.pt, so optimizer and model state agree after resume.
        model.set_adapter("default")
        model.save_pretrained(save_root_lora_default, selected_adapters=["default"])

        # 保存 old 适配器
        model.set_adapter("old")
        model.save_pretrained(save_root_lora_old, selected_adapters=["old"])

        model.set_adapter("default")  # 恢复为 default

        # 保存优化器、scaler
        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        # 保存 epoch 和 global_step 等元信息
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "reward_mapping": FLAGS.reward_mapping,
            "linear_target_scale": FLAGS.linear_target_scale,
        }
        torch.save(training_state, os.path.join(save_root, "training_state.pt"))

        # 保存 EMA 影子参数
        if config.train.ema and ema is not None:
            ema_state_path = os.path.join(save_root, "ema_state.pt")
            torch.save(ema.state_dict(), ema_state_path)
            
        # 保存 PerPromptStatTracker 状态（仅在启用时）
        if config.per_prompt_stat_tracking and stat_tracker is not None:
            stat_tracker_path = os.path.join(save_root, "stat_tracker.pt")
            torch.save(stat_tracker.state_dict(), stat_tracker_path)

        marker_tmp = f"{completion_marker}.tmp"
        with open(marker_tmp, "w", encoding="utf-8") as marker_file:
            marker_file.write(f"global_step={global_step}\nepoch={epoch}\n")
        os.replace(marker_tmp, completion_marker)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config
    linear_target_scale = FLAGS.linear_target_scale
    if not np.isfinite(linear_target_scale) or linear_target_scale <= 0:
        raise ValueError("linear_target_scale must be finite and strictly positive")
    if FLAGS.reward_mapping != "linear" and linear_target_scale != 1.0:
        raise ValueError("linear_target_scale != 1 requires reward_mapping=linear")
    if not config.use_lora:
        raise ValueError("This FlowAWR script requires LoRA for the old-policy adapter")
    if config.sample.guidance_scale != 1.0 or not config.sample.deterministic:
        raise ValueError("FlowAWR rollout requires deterministic sampling with guidance_scale=1.0")
    if config.sample.num_image_per_prompt < 2:
        raise ValueError("FlowAWR needs at least two images per prompt")
    if not 0 < config.train.timestep_fraction <= 1:
        raise ValueError("train.timestep_fraction must be in (0, 1]")
    rule_reward = any(name in config.reward_fn for name in ("ocr", "geneval"))
    variance_gate = FLAGS.awr_variance_gate and rule_reward

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    config.resume_from = resolve_resume_checkpoint(config, rank)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    writer = None
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("reward_mapping", FLAGS.reward_mapping, 0)
        writer.add_scalar("linear_target_scale", linear_target_scale, 0)
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    stat_tracker = PerPromptStatTracker(config.sample.global_std) if config.per_prompt_stat_tracking else None

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    logger.info("  FlowAWR variance gate = %s (rule reward = %s)", variance_gate, rule_reward)
    logger.info("  Reward mapping = %s", FLAGS.reward_mapping)
    logger.info("  Linear target scale = %s", linear_target_scale)

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # 加载 default 适配器
        lora_default_path = os.path.join(config.resume_from, "lora_default")
        if os.path.exists(lora_default_path):
            default_adapter_path = lora_default_path
        else:
            # 向后兼容旧格式
            lora_path = os.path.join(config.resume_from, "lora")
            if os.path.exists(lora_path):
                default_adapter_path = lora_path
            else:
                raise FileNotFoundError(f"No default LoRA adapter found in {config.resume_from}")
        transformer_ddp.module.load_adapter(default_adapter_path, adapter_name="default", is_trainable=True)

        # 加载 old 适配器
        lora_old_path = os.path.join(config.resume_from, "lora_old")
        old_adapter_candidates = [
            os.path.join(lora_old_path, "old"),
            lora_old_path,
            os.path.join(lora_default_path, "old"),
        ]
        old_adapter_path = next(
            (path for path in old_adapter_candidates if _has_saved_adapter(path)),
            "",
        )
        if old_adapter_path:
            transformer_ddp.module.load_adapter(old_adapter_path, adapter_name="old", is_trainable=False)
        else:
            # 如果没有 old，则复制 default 作为起点（但会丢失旧策略信息）
            logger.warning("No 'old' adapter found in checkpoint. Cloning 'default' as 'old'.")
            # 需要先确保 default 已加载，然后拷贝
            # 因为 PeftModel 没有直接的克隆方法，可采用以下方式：
            # 重新加载 default 的权重并作为 old（简单方案：直接 load_adapter 同路径）
            transformer_ddp.module.load_adapter(default_adapter_path, adapter_name="old", is_trainable=False)

        # 确保适配器名称正确
        transformer_ddp.module.set_adapter("default")

        # 加载优化器
        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        # 加载 scaler
        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # 加载训练状态（epoch 和 global_step）
        training_state_path = os.path.join(config.resume_from, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            # Checkpoints are saved before sampling/training the recorded epoch.
            first_epoch = training_state.get("epoch", 0)
            global_step = training_state.get("global_step", 0)
            logger.info(f"Resuming from epoch={first_epoch}, global_step={global_step}")
        else:
            # 尝试从 checkpoint 目录名解析 global_step
            try:
                global_step = int(os.path.basename(config.resume_from).split("-")[-1])
                logger.info(f"Parsed global_step from dirname: {global_step}")
            except ValueError:
                logger.warning(f"Could not parse global_step from dirname: {config.resume_from}. Starting from 0.")
                global_step = 0
                
        # 恢复 EMA 影子参数
        if config.train.ema and ema is not None:
            ema_state_path = os.path.join(config.resume_from, "ema_state.pt")
            if os.path.exists(ema_state_path):
                ema_state = torch.load(ema_state_path, map_location=device)
                ema.load_state_dict(ema_state, device=device)
                logger.info("EMA shadow parameters restored from checkpoint.")
            else:
                logger.warning("EMA was enabled but no ema_state.pt found in checkpoint. EMA will start fresh.")
        # 恢复 PerPromptStatTracker 状态
        if config.per_prompt_stat_tracking:
            stat_tracker_path = os.path.join(config.resume_from, "stat_tracker.pt")
            if os.path.exists(stat_tracker_path):
                stat_tracker.load_state_dict(torch.load(stat_tracker_path, map_location=device))
                logger.info("PerPromptStatTracker state restored.")
            else:
                logger.info("No stat_tracker.pt found, starting fresh.")

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    if num_train_timesteps < 1:
        raise ValueError("The timestep fraction selected no training timesteps")

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # 只在首次训练时同步 old 适配器
    if not config.resume_from:
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)
            assert src_param is not tgt_param
    else:
        # 确保 old 适配器参数不会被意外覆盖，保持 checkpoint 加载的状态
        logger.info("Skipping old adapter initialization (resuming from checkpoint).")
        # 顺手验证一下 old 是否真正与 default 不同（如果是不同步的，说明加载正确）
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata, prompt_indices = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                    writer,
                )

            if (
                i == 0
                and epoch % config.save_freq == 0
                and is_main_process(rank)
                and not config.debug
                and not (config.resume_from and epoch == first_epoch)
            ):
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    config,
                    optimizer,
                    scaler,
                    epoch,  # 新增参数
                    stat_tracker=stat_tracker if config.per_prompt_stat_tracking else None,
                )

            transformer_ddp.module.set_adapter("old")
            # transformer_ddp.module.set_adapter("default")  # Algorithm 1: roll out with the current model theta.
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)
            sigmas = pipeline.scheduler.sigmas[:-1].repeat(len(prompts), 1).to(device)
            if sigmas.shape != timesteps.shape:
                raise ValueError("Sampler sigmas and timesteps have different shapes")

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "sigmas": sigmas,
                    "latents_clean": latents[:, -1],
                    "prompt_indices": torch.as_tensor(prompt_indices, device=device, dtype=torch.long),
                    "rollout_batch_ids": torch.full((len(prompts),), i, device=device, dtype=torch.long),
                    "rewards_future": rewards_future,  # Store future
                }
            )

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # One raw reward per generated image; do not use NFT's normalized advantages.
        collated_samples["rewards"]["avg"] = collated_samples["rewards"]["avg"].reshape(-1, 1)

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            log_scalars(
                writer,
                {
                    "epoch": epoch,
                    **{
                        f"reward/{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                global_step,
            )

        rollout_batch_ids_all = gather_tensor_to_all(collated_samples["rollout_batch_ids"], world_size).numpy()
        prompt_indices_all = gather_tensor_to_all(collated_samples["prompt_indices"], world_size).numpy()
        awr_advantages, awr_stats = compute_flowawr_advantages(
            gathered_rewards_dict["avg"][:, 0],
            rollout_batch_ids_all,
            prompt_indices_all,
            group_size=config.sample.num_image_per_prompt,
            gamma_floor=FLAGS.awr_gamma_floor,
            variance_gate=variance_gate,
            variance_floor=FLAGS.awr_variance_floor,
            reward_mapping=FLAGS.reward_mapping,
        )

        if stat_tracker is not None:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.numpy(), skip_special_tokens=True
            )
            stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                log_scalars(
                    writer,
                    {
                        "stats/group_size": group_size,
                        "stats/trained_prompt_num": trained_prompt_num,
                        "stats/zero_std_ratio": zero_std_ratio,
                        "stats/reward_std_mean": reward_std_mean,
                        "stats/mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "stats/mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "stats/mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "stats/mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "stats/mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    global_step,
                )
            stat_tracker.clear()

        if is_main_process(rank):
            log_scalars(writer, {f"awr/{key}": value for key, value in awr_stats.items()}, global_step)
            logger.info("FlowAWR (%s mapping) rollout: %s", FLAGS.reward_mapping, awr_stats)

        # Every rank sees the same gathered order: rank-major, then local sample order.
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if awr_advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                awr_advantages.reshape(world_size, samples_per_gpu)[rank]
            ).to(device=device, dtype=torch.float32)
        else:
            raise ValueError("Gathered FlowAWR advantage count does not match the DDP rollout")

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        del collated_samples["rollout_batch_ids"]
        del collated_samples["prompt_indices"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "sigmas"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    sigma = train_sample_batch["sigmas"][:, j_idx].float()
                    sigma_expanded = sigma.view(-1, *([1] * (x0.ndim - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - sigma_expanded) * x0.float() + sigma_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # SD3 predicts noise - clean in descending-sigma time.
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        # Match the baseline's fixed pretrained-model reference:
                        # disable LoRA only for this no-grad forward pass.
                        with torch.no_grad():
                            with transformer_ddp.module.disable_adapter():
                                ref_forward_prediction = transformer_ddp(
                                    hidden_states=xt,
                                    timestep=train_sample_batch["timesteps"][:, j_idx],
                                    encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds,
                                    return_dict=False,
                                )[0]
                            transformer_ddp.module.set_adapter("default")

                    advantage = train_sample_batch["advantages"].float()
                    with torch.no_grad():
                        conditional_velocity = noise - x0.float()
                        old_velocity = old_prediction.float()
                        correction = advantage.view(-1, *([1] * (x0.ndim - 1))) * (
                            conditional_velocity - old_velocity
                        )
                        # Step 1.1: scale only the target correction, not weights or loss.
                        if linear_target_scale != 1.0:
                            correction = linear_target_scale * correction
                        target_velocity = old_velocity + correction

                    # Negating all three velocities gives the paper's forward-time
                    # convention and leaves this MSE unchanged (Eq. 10).
                    policy_loss = (forward_prediction.float() - target_velocity).square().flatten(1).mean(1).mean()
                    kl_div_loss = (forward_prediction.float() - ref_forward_prediction.float()).square().flatten(1).mean(1).mean()
                    loss = policy_loss + config.train.beta * kl_div_loss
                    loss_terms = {
                        "policy_loss": policy_loss.detach(),
                        "kl_div_loss": kl_div_loss.detach(),
                        "kl_div": kl_div_loss.detach(),
                        "old_kl_div": (old_velocity - ref_forward_prediction.float()).square().mean().detach(),
                        "total_loss": loss.detach(),
                        "old_deviate": (forward_prediction.float() - old_velocity).square().mean().detach(),
                        "correction_norm": correction.square().mean().detach(),
                        "advantage_abs_mean": advantage.abs().mean().detach(),
                        "sigma_mean": sigma.mean().detach(),
                    }

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            log_scalars(
                                writer,
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                },
                                global_step,
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = flowawr_reference_decay(epoch + 1)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

    if is_main_process(rank):
        writer.flush()
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)

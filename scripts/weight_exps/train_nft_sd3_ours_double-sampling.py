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
from torchvision.utils import make_grid
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

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def add_file_log_handler(log_dir):
    """Write rank-0 Python logging records beside the TensorBoard event files."""
    log_path = os.path.abspath(os.path.join(log_dir, "train.log"))
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    return log_path


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

    required_files = [
        os.path.join(checkpoint_dir, "optimizer.pt"),
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
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


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


def gather_tensor_to_all_device(tensor, world_size):
    r"""All-gather a detached tensor while keeping the result on the current device."""
    tensor = tensor.detach().contiguous()
    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


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




def compute_reinforced_flow_weights(
    prompts,
    rollout_batch_ids,
    nft_advantages,
    advantage_clip,
    coverage_beta,
    epsilon,
    advantage_mode="all",
    group_size=None,
):
    r"""Map NFT advantages to \hat A/W/\bar Z and build same-prompt correction groups."""
    prompts = np.asarray(prompts)
    rollout_batch_ids = np.asarray(rollout_batch_ids)
    nft_advantages = np.asarray(nft_advantages, dtype=np.float64)
    if nft_advantages.ndim > 1:
        # NFT repeats each clean-sample advantage over training timesteps.
        nft_advantages = nft_advantages[:, 0]
    if advantage_clip <= 0:
        raise ValueError(f"advantage_clip must be positive, got {advantage_clip}")
    if not 0.0 <= coverage_beta <= 1.0:
        raise ValueError(f"coverage_beta must be in [0, 1], got {coverage_beta}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not (len(prompts) == len(rollout_batch_ids) == len(nft_advantages)):
        raise ValueError("prompts, rollout_batch_ids, and nft_advantages must have the same length")

    advantages_clip = np.clip(nft_advantages, -advantage_clip, advantage_clip)
    if advantage_mode == "positive_only":
        advantages_clip = np.clip(advantages_clip, 0.0, advantage_clip)
    elif advantage_mode == "negative_only":
        advantages_clip = np.clip(advantages_clip, -advantage_clip, 0.0)
    elif advantage_mode == "one_only":
        advantages_clip = np.where(advantages_clip > 0.0, 1.0, 0.0)
    elif advantage_mode == "binary":
        advantages_clip = np.sign(advantages_clip)

    # This is exactly 2 * r - 1 in the original NFT code:
    # r = clip((clip(adv, -A, A) / A) / 2 + 0.5, 0, 1).
    normalized_advantages = advantages_clip / advantage_clip
    importance_weights = 1.0 + normalized_advantages
    prompt_normalizers = np.empty_like(importance_weights)
    if group_size is None or int(group_size) <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size}")
    group_size = int(group_size)
    correction_group_indices = np.full(
        (len(importance_weights), group_size),
        fill_value=-1,
        dtype=np.int64,
    )

    prompt_groups = defaultdict(list)
    for sample_idx, (rollout_batch_id, prompt) in enumerate(zip(rollout_batch_ids, prompts, strict=True)):
        prompt_groups[(int(rollout_batch_id), str(prompt))].append(sample_idx)

    for sample_indices in prompt_groups.values():
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        if len(sample_indices) != group_size:
            raise ValueError(
                "Each (rollout_batch_id, prompt) correction group must contain "
                f"exactly K={group_size} samples, but found {len(sample_indices)}. "
                "If duplicate prompt strings represent distinct groups, propagate a unique prompt_group_id."
            )
        if len(np.unique(sample_indices)) != group_size:
            raise ValueError("Correction group indices must be unique")
        prompt_z_bar = 1.0 + coverage_beta * np.mean(importance_weights[sample_indices] - 1.0)
        prompt_normalizers[sample_indices] = prompt_z_bar
        correction_group_indices[sample_indices] = sample_indices[None, :]

    if np.any(correction_group_indices < 0):
        raise RuntimeError("Some rollout samples were not assigned to a correction group")
    sample_ids = np.arange(len(importance_weights), dtype=np.int64)[:, None]
    if not np.all(np.any(correction_group_indices == sample_ids, axis=1)):
        raise RuntimeError("Every outer sample must be included in its own correction group")

    # Original single-sample return kept for reference:
    # return normalized_advantages, importance_weights, prompt_normalizers
    return normalized_advantages, importance_weights, prompt_normalizers, correction_group_indices


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


def get_nested_config_value(config, path, default):
    value = config
    for key in path.split("."):
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
    return value


def get_image_log_settings(config):
    num_prompts = get_nested_config_value(config, "sample.log_num_prompts", None)
    if num_prompts is None:
        num_prompts = int(os.getenv("NFT_LOG_NUM_PROMPTS", 5))

    num_images_per_prompt = get_nested_config_value(config, "sample.log_num_images_per_prompt", None)
    if num_images_per_prompt is None:
        num_images_per_prompt = int(
            os.getenv("NFT_LOG_NUM_IMAGES_PER_PROMPT", int(config.sample.num_image_per_prompt))
        )

    return max(1, int(num_prompts)), max(1, int(num_images_per_prompt))


def format_reward_value(value):
    value = np.asarray(value)
    if value.size == 0:
        return None
    scalar = float(value.reshape(-1)[0])
    if scalar == -10:
        return None
    return f"{scalar:.2f}"


def select_prompt_image_logs(images, prompts, rewards, max_prompts, max_images_per_prompt):
    prompt_to_indices = defaultdict(list)
    selected_prompt_order = []
    for idx, prompt in enumerate(prompts):
        prompt_key = str(prompt)
        if prompt_key not in prompt_to_indices:
            if len(selected_prompt_order) >= max_prompts:
                continue
            selected_prompt_order.append(prompt_key)
        if len(prompt_to_indices[prompt_key]) < max_images_per_prompt:
            prompt_to_indices[prompt_key].append(idx)

    selected_indices = []
    captions = []
    for prompt_idx, prompt in enumerate(selected_prompt_order):
        for image_idx, sample_idx in enumerate(prompt_to_indices[prompt]):
            selected_indices.append(sample_idx)
            reward_text_parts = []
            for reward_key, reward_values in rewards.items():
                if sample_idx >= len(reward_values):
                    continue
                reward_value = format_reward_value(reward_values[sample_idx])
                if reward_value is not None:
                    reward_text_parts.append(f"{reward_key}: {reward_value}")
            reward_text = " | ".join(reward_text_parts)
            captions.append(
                f"prompt {prompt_idx}, image {image_idx} | {prompt[:1000]}"
                + (f" | {reward_text}" if reward_text else "")
            )

    if not selected_indices:
        return images[:0], []
    return images[selected_indices], captions


def select_prompt_image_log_indices(prompt_counts, prompts, max_prompts, max_images_per_prompt):
    selected_indices = []
    selected_prompts = []
    for idx, prompt in enumerate(prompts):
        prompt_key = str(prompt)
        if prompt_key not in prompt_counts:
            if len(prompt_counts) >= max_prompts:
                continue
            prompt_counts[prompt_key] = 0
        if prompt_counts[prompt_key] >= max_images_per_prompt:
            continue
        prompt_counts[prompt_key] += 1
        selected_indices.append(idx)
        selected_prompts.append(prompt)
    return selected_indices, selected_prompts


def append_prompt_image_log_batch(log_batches, prompt_counts, images, prompts, rewards, max_prompts, max_images_per_prompt):
    selected_indices, selected_prompts = select_prompt_image_log_indices(
        prompt_counts, prompts, max_prompts, max_images_per_prompt
    )
    if not selected_indices:
        return

    selected_rewards = {
        reward_key: np.asarray(reward_values)[selected_indices] for reward_key, reward_values in rewards.items()
    }
    log_batches.append((images.detach().cpu()[selected_indices], selected_prompts, selected_rewards))


def log_image_grid(writer, tag, images, captions, step, max_images=15, nrow=None):
    if writer is None:
        return
    num_images = min(max_images, len(images))
    if num_images == 0:
        return
    image_grid = make_grid(images[:num_images].float().clamp(0, 1), nrow=nrow or min(5, num_images))
    writer.add_image(tag, image_grid, step)
    if captions:
        caption_text = "\n".join(f"{idx}. {caption}" for idx, caption in enumerate(captions[:num_images]))
        writer.add_text(f"{tag}_captions", caption_text, step)


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
    image_log_num_prompts, image_log_num_images_per_prompt = get_image_log_settings(config)
    eval_image_log_prompt_items = []
    eval_image_log_prompt_set = set()

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
        prompts, prompt_metadata = test_batch
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
        if is_main_process(rank):
            for prompt, metadata in zip(prompts, prompt_metadata):
                prompt_key = str(prompt)
                if prompt_key in eval_image_log_prompt_set:
                    continue
                if len(eval_image_log_prompt_items) >= image_log_num_prompts:
                    break
                eval_image_log_prompt_set.add(prompt_key)
                eval_image_log_prompt_items.append((prompt, metadata))

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        if eval_image_log_prompt_items:
            eval_image_log_batches = []
            eval_image_log_prompt_counts = {}
            eval_log_prompts = [
                prompt
                for prompt, _ in eval_image_log_prompt_items
                for _ in range(image_log_num_images_per_prompt)
            ]
            eval_log_metadata = [
                metadata
                for _, metadata in eval_image_log_prompt_items
                for _ in range(image_log_num_images_per_prompt)
            ]
            for start in range(0, len(eval_log_prompts), config.sample.test_batch_size):
                batch_prompts = eval_log_prompts[start : start + config.sample.test_batch_size]
                batch_metadata = eval_log_metadata[start : start + config.sample.test_batch_size]
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    batch_prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
                )
                current_batch_size = len(prompt_embeds)
                with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
                    with torch.no_grad():
                        images, _, _ = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds[:current_batch_size],
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[:current_batch_size],
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
                rewards_future = executor.submit(reward_fn, images, batch_prompts, batch_metadata, only_strict=False)
                time.sleep(0)
                rewards, reward_metadata = rewards_future.result()
                append_prompt_image_log_batch(
                    eval_image_log_batches,
                    eval_image_log_prompt_counts,
                    images,
                    batch_prompts,
                    rewards,
                    image_log_num_prompts,
                    image_log_num_images_per_prompt,
                )

            images_to_log = torch.cat([batch_images for batch_images, _, _ in eval_image_log_batches], dim=0)
            prompts_to_log = [
                prompt for _, batch_prompts, _ in eval_image_log_batches for prompt in batch_prompts
            ]
            rewards_to_log = {
                reward_key: np.concatenate(
                    [np.asarray(batch_rewards[reward_key]) for _, _, batch_rewards in eval_image_log_batches], axis=0
                )
                for reward_key in eval_image_log_batches[0][2]
            }
            images_to_log, captions = select_prompt_image_logs(
                images_to_log,
                prompts_to_log,
                rewards_to_log,
                image_log_num_prompts,
                image_log_num_images_per_prompt,
            )
            log_image_grid(
                writer,
                "eval/images",
                images_to_log,
                captions,
                global_step,
                max_images=image_log_num_prompts * image_log_num_images_per_prompt,
                nrow=image_log_num_images_per_prompt,
            )
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
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler, epoch, stat_tracker=None
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

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        # 保存 default 适配器
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
        }
        torch.save(training_state, os.path.join(save_root, "training_state.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        
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
        log_path = add_file_log_handler(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text("config", f"```text\n{config}\n```", 0)
        logger.info("Writing command-line logs to %s", log_path)
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

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

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

            prompts, prompt_metadata = next(train_iter)

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
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                    epoch,  # 新增参数
                    stat_tracker=stat_tracker if config.per_prompt_stat_tracking else None,
                )

            # transformer_ddp.module.set_adapter("old")
            transformer_ddp.module.set_adapter("default")  # Algorithm 1: roll out with the current model theta.
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

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
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

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()
            num_to_log = min(15, len(images_to_log))
            captions = [
                f"{str(prompts_to_log[idx])[:100]} | avg: {float(rewards_to_log[idx]):.2f}"
                for idx in range(num_to_log)
            ]
            log_image_grid(writer, "train/images", images_to_log, captions, global_step)
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

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

        # if config.per_prompt_stat_tracking:
        #     prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
        #     prompts_all_decoded = pipeline.tokenizer.batch_decode(
        #         prompt_ids_all.cpu().numpy(), skip_special_tokens=True
        #     )
        prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
        prompts_all_decoded = pipeline.tokenizer.batch_decode(
            prompt_ids_all.cpu().numpy(), skip_special_tokens=True
        )
        rollout_batch_ids_all = gather_tensor_to_all(collated_samples["rollout_batch_ids"], world_size).numpy()
        algorithm_epsilon = 1e-5

        if config.per_prompt_stat_tracking:
            # Stat tracker update expects numpy arrays for rewards
            # advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])
            # stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

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
        else:
            # avg_rewards_all = gathered_rewards_dict["avg"]
            # advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
            # pass
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)

        # Original single-sample weight return kept for reference:
        # normalized_advantages, importance_weights, prompt_normalizers = compute_reinforced_flow_weights(
        normalized_advantages, importance_weights, prompt_normalizers, correction_group_indices = (
            compute_reinforced_flow_weights(
                prompts_all_decoded,
                rollout_batch_ids_all,
                advantages,
                advantage_clip=float(config.train.adv_clip_max),
                coverage_beta=float(config.beta),
                epsilon=algorithm_epsilon,
                advantage_mode=getattr(config.train, "adv_mode", "all"),
                group_size=int(config.sample.num_image_per_prompt),
            )
        )
        total_rollout_size = len(importance_weights)  # Algorithm 1: D = K * C.
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        # The K same-prompt correction samples may be distributed over multiple
        # ranks and training micro-batches.  Keep one rank-major global bank on
        # every rank; its order matches prompt_ids_all and the global W/Z arrays.
        with torch.no_grad():
            global_corr_latents = gather_tensor_to_all_device(
                collated_samples["latents_clean"], world_size
            )
            global_corr_importance_weights = torch.as_tensor(
                importance_weights, device=device, dtype=torch.float32
            )
            global_corr_prompt_normalizers = torch.as_tensor(
                prompt_normalizers, device=device, dtype=torch.float32
            )

        if not (
            global_corr_latents.shape[0]
            == global_corr_importance_weights.shape[0]
            == global_corr_prompt_normalizers.shape[0]
            == correction_group_indices.shape[0]
            == total_rollout_size
        ):
            raise RuntimeError("Global correction bank tensors must use the same rank-major sample order")

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
            collated_samples["importance_weights"] = torch.from_numpy(
                importance_weights.reshape(world_size, samples_per_gpu)[rank]
            ).to(device=device, dtype=torch.float32)
            collated_samples["prompt_normalizers"] = torch.from_numpy(
                prompt_normalizers.reshape(world_size, samples_per_gpu)[rank]
            ).to(device=device, dtype=torch.float32)
            collated_samples["correction_group_indices"] = torch.from_numpy(
                correction_group_indices.reshape(
                    world_size,
                    samples_per_gpu,
                    int(config.sample.num_image_per_prompt),
                )[rank]
            ).to(device=device, dtype=torch.long)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")
            logger.info(
                "Importance weights mean: %.6f; prompt normalizers mean: %.6f; D: %d; K: %d",
                collated_samples["importance_weights"].mean().item(),
                collated_samples["prompt_normalizers"].mean().item(),
                total_rollout_size,
                int(config.sample.num_image_per_prompt),
            )

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        del collated_samples["rollout_batch_ids"]

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
            for key in ["timesteps", "next_timesteps"]:
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

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

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

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    loss_terms = {}
                    # # Policy Gradient Loss
                    # advantages_clip = torch.clamp(
                    #     train_sample_batch["advantages"][:, j_idx],
                    #     -config.train.adv_clip_max,
                    #     config.train.adv_clip_max,
                    # )
                    # if hasattr(config.train, "adv_mode"):
                    #     if config.train.adv_mode == "positive_only":
                    #         advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                    #     elif config.train.adv_mode == "negative_only":
                    #         advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                    #     elif config.train.adv_mode == "one_only":
                    #         advantages_clip = torch.where(
                    #             advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                    #         )
                    #     elif config.train.adv_mode == "binary":
                    #         advantages_clip = torch.sign(advantages_clip)

                    # # normalize advantage
                    # normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    # r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()

                    # Original DiffusionNFT two-branch objective (kept for reference):
                    # positive_prediction = (
                    #     config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    # )
                    # implicit_negative_prediction = (
                    #     1.0 + config.beta
                    # ) * old_prediction.detach() - config.beta * forward_prediction
                    #
                    # # adaptive weighting
                    # x0_prediction = xt - t_expanded * positive_prediction
                    # with torch.no_grad():
                    #     weight_factor = (
                    #         torch.abs(x0_prediction.double() - x0.double())
                    #         .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                    #         .clip(min=0.00001)
                    #     )
                    # positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
                    #     dim=tuple(range(1, x0.ndim))
                    # )
                    # negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    # with torch.no_grad():
                    #     negative_weight_factor = (
                    #         torch.abs(negative_x0_prediction.double() - x0.double())
                    #         .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                    #         .clip(min=0.00001)
                    #     )
                    # negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                    #     dim=tuple(range(1, x0.ndim))
                    # )
                    # ori_policy_loss = (
                    #     r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    # )
                    # policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    # Algorithm 1: reward-induced target velocity and its
                    # importance-weighted, timestep-adapted regression loss.
                    importance_weight = train_sample_batch["importance_weights"].float()
                    prompt_normalizer = train_sample_batch["prompt_normalizers"].float()

                    # Double-sampling approximation: for every fixed outer x_t,
                    # average corrections from all K same-prompt clean rollouts.
                    with torch.no_grad():
                        correction_group_indices = train_sample_batch["correction_group_indices"].long()
                        expected_group_size = int(config.sample.num_image_per_prompt)
                        if correction_group_indices.shape != (
                            current_micro_batch_size,
                            expected_group_size,
                        ):
                            raise RuntimeError(
                                "correction_group_indices must have shape "
                                f"[{current_micro_batch_size}, {expected_group_size}], got "
                                f"{list(correction_group_indices.shape)}"
                            )

                        corr_z = global_corr_latents[correction_group_indices].float()
                        corr_importance_weight = global_corr_importance_weights[
                            correction_group_indices
                        ]
                        corr_prompt_normalizer = global_corr_prompt_normalizers[
                            correction_group_indices
                        ]

                        corr_t_expanded = t.float().view(
                            -1,
                            1,
                            *([1] * (x0.ndim - 1)),
                        )
                        fixed_xt = xt.detach().float().unsqueeze(1)
                        conditional_velocity = (fixed_xt - corr_z) / corr_t_expanded

                        fixed_old_prediction = old_prediction.detach().float().unsqueeze(1)
                        velocity_discrepancy = conditional_velocity - fixed_old_prediction
                        inner_feature_dims = tuple(range(2, velocity_discrepancy.ndim))
                        trajectory_alpha = 1.0 / (
                            torch.abs(velocity_discrepancy).mean(
                                dim=inner_feature_dims,
                                keepdim=True,
                            )
                            + algorithm_epsilon
                        )
                        # Normalize independently over the K correction samples
                        # for each fixed outer x_t: mean_j alpha_{i,j} = 1.
                        trajectory_alpha = trajectory_alpha / trajectory_alpha.mean(
                            dim=1,
                            keepdim=True,
                        ).clamp_min(algorithm_epsilon)

                        correction_coefficient = (
                            float(config.beta)
                            * (corr_importance_weight - 1.0)
                            / corr_prompt_normalizer
                        )
                        correction_coefficient_expanded = correction_coefficient.view(
                            current_micro_batch_size,
                            expected_group_size,
                            *([1] * (x0.ndim - 1)),
                        )
                        correction_terms = (
                            correction_coefficient_expanded
                            * trajectory_alpha
                            * velocity_discrepancy
                        )

                        mean_velocity_correction = correction_terms.mean(dim=1)
                        target_velocity = old_prediction.detach().float() + mean_velocity_correction

                        if config.debug and not torch.isfinite(target_velocity).all():
                            raise FloatingPointError("Non-finite double-sampling target velocity")

                    target_velocity_loss = ((forward_prediction - target_velocity) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    target_importance = importance_weight / prompt_normalizer
                    
                    ori_policy_loss = t.float() * target_importance * target_velocity_loss
                    


                    policy_loss = float(config.train.adv_clip_max) * ori_policy_loss.mean()

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()
                    loss_terms["importance_weight"] = importance_weight.mean().detach()
                    loss_terms["importance_weight_max"] = importance_weight.max().detach()
                    loss_terms["importance_weight_min"] = importance_weight.min().detach()
                    loss_terms["prompt_normalizer"] = prompt_normalizer.mean().detach()
                    loss_terms["prompt_normalizer_max"] = prompt_normalizer.max().detach()
                    loss_terms["prompt_normalizer_min"] = prompt_normalizer.min().detach()
                    loss_terms["target_importance"] = target_importance.mean().detach()
                    loss_terms["target_importance_max"] = target_importance.max().detach()
                    loss_terms["target_importance_min"] = target_importance.min().detach()
                    loss_terms["trajectory_alpha"] = trajectory_alpha.mean().detach()
                    loss_terms["trajectory_alpha_max"] = trajectory_alpha.max().detach()
                    loss_terms["trajectory_alpha_min"] = trajectory_alpha.min().detach()
                    loss_terms["correction_coefficient_abs_mean"] = correction_coefficient.abs().mean().detach()
                    loss_terms["correction_coefficient_max"] = correction_coefficient.max().detach()
                    loss_terms["correction_coefficient_min"] = correction_coefficient.min().detach()
                    loss_terms["mean_velocity_correction_norm"] = (
                        mean_velocity_correction.square()
                        .mean(dim=tuple(range(1, mean_velocity_correction.ndim)))
                        .sqrt()
                        .mean()
                        .detach()
                    )
                    loss_terms["mean_velocity_correction_norm_max"] = (
                        mean_velocity_correction.square()
                        .mean(dim=tuple(range(1, mean_velocity_correction.ndim)))
                        .sqrt()
                        .max()
                        .detach()
                    )
                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    loss_terms["total_loss"] = loss.detach()

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
            decay = return_decay(global_step, config.decay_type)
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

#!/usr/bin/env python3
"""Resolve a fixed experiment and delegate execution to a Step 1 launcher."""

import argparse
import json
import os
from pathlib import Path
import shlex
import sys


DIRECTORY = Path(__file__).resolve().parent
MANIFEST = json.loads((DIRECTORY / "manifest.json").read_text())
RUNS = {run["run_id"]: run for run in MANIFEST["runs"]}


def resolved_path(value):
    return str(Path(value).expanduser().resolve())


def resolve_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Fixed Linear / square ablations; --dry-run has no launch side effects.",
        allow_abbrev=False,
    )
    parser.add_argument("run_id", choices=RUNS)
    parser.add_argument("--layout", choices=("h200_8gpu", "a6000_16gpu"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config.pretrained.model", dest="model")
    parser.add_argument("--config.resume_from", dest="resume_from")
    parser.add_argument("--config.save_dir", dest="save_dir")
    parser.add_argument("--config.logdir", dest="logdir")
    args = parser.parse_args(argv)
    if args.resume_from and not Path(args.resume_from).expanduser().is_absolute():
        parser.error("--config.resume_from must be an absolute, complete checkpoint directory path")

    run = RUNS[args.run_id]
    layout = args.layout or run["default_layout"]
    variant = run["layout_variants"][layout]
    repo = Path(os.environ.get("REPO_DIR") or DIRECTORY.parents[2]).expanduser().resolve()
    save_dir = resolved_path(args.save_dir) if args.save_dir else str(repo / variant["save_dir"])
    logdir = resolved_path(args.logdir) if args.logdir else str(repo / variant["logdir"])
    model = args.model or os.environ.get("SD3_MODEL") or str(repo / "pretrained_models/sd3.5-medium")
    resume_from = resolved_path(args.resume_from) if args.resume_from else None
    fixed = MANIFEST["fixed_config"]
    batches = variant["rollout_batches"]
    accumulation = variant["gradient_accumulation_steps"]
    require_resume = run["require_resume"]

    env = dict(os.environ)
    env.update(
        REPO_DIR=str(repo), TASK="pickscore", PER_DEVICE_BATCH="6",
        AWR_VARIANCE_GATE="false", REWARD_MAPPING="linear", SD3_MODEL=model,
        NNODES="1" if layout == "h200_8gpu" else "2", NPROC_PER_NODE="8",
        LOGDIR=logdir, SAVE_DIR=save_dir, RUN_NAME=variant["run_name"],
    )
    base_name = "run_sd3_h200_8gpu_step1_linear_kl1e-4.sh" if layout == "h200_8gpu" else "run_sd3_a6000_16gpu_step1_linear_kl1e-4.sh"
    base_launcher = repo / "scripts/experiment_plans/step1" / base_name
    config = {
        "seed": fixed["seed"],
        "mixed_precision": fixed["mixed_precision"],
        "train.learning_rate": fixed["learning_rate"],
        "train.beta": fixed["beta"],
        "num_epochs": fixed["num_epochs"],
        "eval_freq": fixed["eval_freq"],
        "save_freq": fixed["save_freq"],
        "sample.num_steps": fixed["sample_num_steps"],
        "sample.eval_num_steps": fixed["sample_eval_num_steps"],
        "sample.num_image_per_prompt": fixed["group_size"],
        "sample.train_batch_size": fixed["per_device_batch"],
        "sample.test_batch_size": fixed["per_device_batch"],
        "sample.num_batches_per_epoch": batches,
        "train.batch_size": fixed["per_device_batch"],
        "train.gradient_accumulation_steps": accumulation,
        "train.timestep_fraction": fixed["timestep_fraction"],
    }
    protocol = [
        "--reward_mapping=linear",
        f"--linear_target_scale={run['c']}",
        f"--mass_shift_transform={run['transform']}",
        f"--target_scale_schedule={run['schedule']}",
    ]
    if run["schedule"] == "linear_decay":
        protocol.append(f"--target_scale_final={run['target_scale_final']}")
    protocol.extend([
        f"--target_scale_decay_start={run['target_scale_decay_start']}",
        f"--target_scale_decay_end={run['target_scale_decay_end']}",
        "--final_eval_and_save=true",
        f"--require_resume={'true' if require_resume else 'false'}",
        "--awr_variance_gate=false",
    ])
    extra = [f"--config.resume_from={resume_from}"] if resume_from else []
    # Only whitelisted path options are accepted; fixed training/method flags are last.
    extra.extend(f"--config.{key}={value}" for key, value in config.items())
    extra.extend(protocol)

    # Mirror the unchanged Step 1 CLI for inspection without invoking it.
    requirements = []
    if layout == "h200_8gpu":
        distributed = ["--standalone", "--nnodes=1", "--nproc_per_node=8"]
    else:
        node_rank = next((env.get(key) for key in ("NODE_RANK", "SENSECORE_PYTORCH_NODE_RANK", "RANK", "SLURM_NODEID") if env.get(key)), None)
        master_addr = env.get("MASTER_ADDR") or "127.0.0.1"
        master_port = env.get("MASTER_PORT") or "29532"
        if node_rank is None:
            requirements.append("Set NODE_RANK=0 / 1 on the two A6000 nodes")
        elif not node_rank.isdigit() or int(node_rank) not in (0, 1):
            requirements.append("NODE_RANK must be 0 or 1")
        if master_addr in ("127.0.0.1", "localhost"):
            requirements.append("Set MASTER_ADDR to node rank 0's reachable hostname or IP")
        distributed = [
            "--nnodes=2", f"--node_rank={node_rank if node_rank is not None else '<NODE_RANK>'}",
            f"--master_addr={master_addr}", f"--master_port={master_port}", "--nproc_per_node=8",
        ]
    torchrun_args = distributed + [
        "scripts/experiment_plans/step1/train_nft_sd3_ours-singleloss-AWR.py",
        "--config=config/nft.py:sd3_pickscore",
        f"--config.pretrained.model={model}",
        f"--config.logdir={logdir}", f"--config.save_dir={save_dir}",
        f"--config.run_name={variant['run_name']}",
        "--config.sample.train_batch_size=6", "--config.sample.test_batch_size=6",
        f"--config.sample.num_batches_per_epoch={batches}", "--config.train.batch_size=6",
        f"--config.train.gradient_accumulation_steps={accumulation}",
        "--config.train.timestep_fraction=1.0", "--config.train.beta=0.0001",
        "--reward_mapping=linear", "--awr_variance_gate=false",
    ] + extra
    checkpoint_directory = resume_from or save_dir
    result = {
        "run_id": run["run_id"], "sheet_id": run["sheet_id"],
        "action": run["action"], "layout": layout,
        "code_branch": MANIFEST["code_branch"], "repo_dir": str(repo),
        "run_name": variant["run_name"], "save_dir": save_dir, "logdir": logdir,
        "model": model, "task": "pickscore", "transform": run["transform"],
        "linear_target_scale": run["c"], "target_scale_schedule": run["schedule"],
        "target_scale_final": run["target_scale_final"],
        "target_scale_decay_start": run["target_scale_decay_start"],
        "target_scale_decay_end": run["target_scale_decay_end"],
        "config": config, "num_gpus": variant["num_gpus"],
        "images_per_round": fixed["images_per_round"],
        "optimizer_steps_per_round": fixed["optimizer_steps_per_round"],
        "final_eval_and_save": True, "require_resume": require_resume,
        "resume_from": resume_from, "resume_search_dir": save_dir,
        "resume_directory_exists": Path(checkpoint_directory).is_dir(),
        "resume_note": "Complete checkpoint and objective compatibility are validated by the trainer; use --config.resume_from=/absolute/path/to/checkpoint if needed.",
        "do_not_restart_running_job": run["action"] == "continue",
        "launcher_requirements": requirements,
        "base_launcher": str(base_launcher),
        "base_launcher_args": extra,
        "torchrun_args": torchrun_args,
        "command": shlex.join(["torchrun", *torchrun_args]),
        "environment": {key: env[key] for key in ("TASK", "PER_DEVICE_BATCH", "AWR_VARIANCE_GATE", "NNODES", "NPROC_PER_NODE", "LOGDIR", "SAVE_DIR", "RUN_NAME")},
    }
    return parser, args, result, env


def main():
    parser, args, result, env = resolve_arguments()
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if result["resume_from"] and not result["resume_directory_exists"]:
        parser.error(f"Checkpoint directory does not exist: {result['resume_from']}; pass --config.resume_from=/absolute/path/to/a/complete/checkpoint")
    if result["require_resume"] and not result["resume_directory_exists"]:
        parser.error(f"Resume is required, but the historical checkpoint directory is missing: {result['resume_search_dir']}; pass --config.resume_from=/absolute/path/to/a/complete/checkpoint. No new training was started.")
    if result["launcher_requirements"]:
        parser.error("; ".join(result["launcher_requirements"]))
    if not Path(result["base_launcher"]).is_file():
        parser.error(f"Step 1 launcher not found: {result['base_launcher']}")
    if result["do_not_restart_running_job"]:
        print("Optional continuation: use only after the original C2 job has stopped; do not start a second copy of a running job.", file=sys.stderr)
    if result["require_resume"]:
        print(f"Resume required; checkpoint search: {result['resume_from'] or result['resume_search_dir']}. Use --config.resume_from for an explicit complete checkpoint path.", file=sys.stderr)
    os.execvpe("bash", ["bash", result["base_launcher"], *result["base_launcher_args"]], env)


if __name__ == "__main__":
    main()

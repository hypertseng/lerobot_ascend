import logging
import time
import os
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
from torch.optim import Optimizer
from termcolor import colored

# 提前初始化日志
from lerobot.utils.utils import init_logging as _init_logging_at_import
_init_logging_at_import()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NPU_AVAILABLE = False
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    device = "npu:0"
    torch.npu.set_device(device)
    try:
        torch.npu.init()
    except Exception as e:
        logging.warning(f"torch.npu.init() 失败: {e}")
    NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
    device_count = getattr(torch.npu, "device_count", lambda: "NA")()
    _rank_env = int(os.environ.get("RANK", "0"))
    if _rank_env == 0:
        logging.info("NPU检测: available=%s, device_count=%s", NPU_AVAILABLE, device_count)
        logging.info("昇腾NPU支持已启用" if NPU_AVAILABLE else "NPU不可用，回退CPU")
except ImportError:
    if int(os.environ.get("RANK", "0")) == 0:
        logging.warning("未安装torch_npu，将使用CPU训练")

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_eval import eval_policy_all

import wandb

import datasets.features.features as features
_OLD_GENERATE_FROM_DICT = features.generate_from_dict
def _new_generate_from_dict(obj):
    if isinstance(obj, dict) and obj.get("_type") == "List":
        obj["_type"] = "Sequence"
    return _OLD_GENERATE_FROM_DICT(obj)
features.generate_from_dict = _new_generate_from_dict

# ---------------- 分布式初始化 ----------------
def setup_distributed():
    """设置分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if NPU_AVAILABLE:
            backend = "hccl"
            torch.npu.set_device(local_rank)
            device = torch.device(f"npu:{local_rank}")
        else:
            backend = "gloo"
            device = torch.device("cpu")

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        return rank, world_size, local_rank, device
    else:
        # 单进程训练
        device = torch.device("cuda:0" if torch.cuda.is_available() else "npu:0" if NPU_AVAILABLE else "cpu")
        return 0, 1, 0, device

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------- Policy更新 ----------------
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. It also handles mixed-precision training via a GradScaler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        grad_scaler: The GradScaler for automatic mixed-precision training.
        lr_scheduler: An optional learning rate scheduler.
        use_amp: A boolean indicating whether to use automatic mixed precision.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()

    # forward
    if use_amp:
        if NPU_AVAILABLE and device.type == "npu":
            with torch.npu.amp.autocast():
                loss, output_dict = policy.forward(batch)
        else:
            loss, output_dict = policy.forward(batch)
    else:
        loss, output_dict = policy.forward(batch)

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), grad_clip_norm, error_if_nonfinite=False
    )

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

# ---------------- 训练主函数 ----------------
@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
    """
    rank, world_size, local_rank, device = setup_distributed()

    cfg.validate()
    if world_size > 1:
        barrier_device_ids = [local_rank] if device.type != "cpu" else None
        dist.barrier(device_ids=barrier_device_ids)

    if rank == 0:
        logging.info(pformat(cfg.to_dict()))
        logging.info(f"分布式训练配置: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # WandB
    if cfg.wandb.enable and cfg.wandb.project and rank == 0:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if rank == 0:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed + rank)

    # 数据集
    if rank == 0:
        logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # eval env
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env and rank == 0:
        logging.info("Creating eval env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # 策略
    if rank == 0:
        logging.info("Creating policy")
    cfg.policy.device = str(device)
    policy = make_policy(cfg.policy, dataset.meta)
    policy = policy.to(device)

    # DDP包装
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank] if device.type != "cpu" else None,
                    find_unused_parameters=True)
        dist.barrier(device_ids=[local_rank] if device.type != "cpu" else None)
        
    
    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats
        
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.module.config.input_features, **policy.module.config.output_features},
                "norm_map": policy.module.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.module.config.output_features,
                "norm_map": policy.module.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # 优化器 & scheduler
    if rank == 0:
        logging.info("Creating optimizer and scheduler")

    # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # # 如果 cfg.optimizer 为 None，设置默认值（兼容原逻辑）
    # if cfg.optimizer is None:
    #     from dataclasses import dataclass

    #     @dataclass
    #     class DefaultOptimizerConfig:
    #         name: str = "adamw"
    #         lr: float = 1e-4
    #         weight_decay: float = 0.01
    #         betas: tuple = (0.9, 0.95)
    #         grad_clip_norm: float = 1.0

    #         def build(self, params):
    #             if self.name == "adamw":
    #                 import torch.optim as optim
    #                 return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)
    #             else:
    #                 raise ValueError(f"Unsupported optimizer: {self.name}")

    #     cfg.optimizer = DefaultOptimizerConfig()
    #     if rank == 0:
    #         logging.warning(
    #             colored("cfg.optimizer was None, using default AdamW optimizer.", "yellow")
    #         )
    # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    base_policy = policy.module if hasattr(policy, "module") else policy
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, base_policy)

    # GradScaler
    if NPU_AVAILABLE and device.type == "npu":
        grad_scaler = torch_npu.npu.amp.GradScaler(device.type, enabled=cfg.policy.use_amp)
    else:
        grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    # 恢复训练状态
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # dataloader
    base_sampler = None
    shuffle = True
    if hasattr(cfg.policy, "drop_n_last_frames") and world_size == 1:
        base_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
        shuffle = False

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True and base_sampler is None
        )
        shuffle = False
    else:
        sampler = base_sampler

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        drop_last=False,
        persistent_workers=True if cfg.num_workers > 0 else False,
        prefetch_factor=2,
    )
    dl_iter = cycle(dataloader)

    # Metrics
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step)

    if rank == 0:
        from tqdm import tqdm
        progress_bar = tqdm(range(step, cfg.steps), desc="Training", ncols=100)

    # ---------------- 训练循环 ----------------
    for step_idx in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, grad_scaler, lr_scheduler, use_amp=cfg.policy.use_amp
        )
        step += 1
        train_tracker.step()

        if rank == 0:
            progress_bar.set_postfix({
                "loss": f"{train_tracker.loss.val:.3f}",
                "grad": f"{train_tracker.grad_norm.val:.2f}",
                "lr": f"{train_tracker.lr.val:.2e}",
            })
            progress_bar.update(1)

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step and rank == 0:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if is_saving_step and cfg.save_checkpoint and rank == 0:
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_policy = policy.module if isinstance(policy, DDP) else policy
            save_checkpoint(
                checkpoint_dir, step, cfg, save_policy, optimizer, lr_scheduler, preprocessor, postprocessor
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if is_eval_step and cfg.env and rank == 0:
            step_id = get_step_identifier(step, cfg.steps)
            with torch.no_grad(), (
                torch.npu.amp.autocast() if NPU_AVAILABLE and device.type == "npu" and cfg.policy.use_amp else nullcontext()
            ):
                eval_info = eval_policy_all(
                    envs=eval_env,  # dict[suite][task_id] -> vec_env
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            # overall metrics (suite-agnostic)
            aggregated = eval_info["overall"]

            # optional: per-suite logging
            for suite, suite_info in eval_info.items():
                logging.info("Suite %s aggregated: %s", suite, suite_info)

            # meters/tracker
            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step)
            eval_tracker.eval_s = aggregated.pop("eval_s")
            eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
            eval_tracker.pc_success = aggregated.pop("pc_success")
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

    if eval_env:
        close_envs(eval_env)
    logging.info("End of training")

    cleanup_distributed()

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)
        preprocessor.push_to_hub(cfg.policy.repo_id)
        postprocessor.push_to_hub(cfg.policy.repo_id)


def main():
    init_logging()
    train()


if __name__ == "__main__":
    main()
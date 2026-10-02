"""Single-device staged fine-tuning with reproducible epoch-boundary resume."""

from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import LengthBucketBatchSampler, collate_speech_examples


@dataclass(frozen=True)
class TrainingStage:
    epochs: int
    max_seconds: float
    train_backbone: bool
    adapter_lr: float = 1e-4
    backbone_lr: float = 1e-5

    def __post_init__(self):
        if self.epochs < 1 or not 0 < self.max_seconds <= 60:
            raise ValueError("stage epochs must be positive and max_seconds must be in (0, 60]")
        if not math.isfinite(self.adapter_lr) or self.adapter_lr <= 0:
            raise ValueError("adapter_lr must be finite and positive")
        if not math.isfinite(self.backbone_lr) or self.backbone_lr <= 0:
            raise ValueError("backbone_lr must be finite and positive")


@dataclass(frozen=True)
class TrainingConfig:
    stages: tuple[TrainingStage, ...]
    batch_size: int = 2
    accumulation_steps: int = 8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    precision: str = "bf16"
    num_workers: int = 0
    seed: int = 42
    validation_seed: int = 1234
    latent_fps: int = 40
    gradient_checkpointing: bool = True
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "diff_loss": 1.0, "stop_loss": 0.1,
        "ar_l1_loss": 1e-4, "vae_projected_l1_loss": 1e-4,
    })

    def __post_init__(self):
        if not self.stages:
            raise ValueError("at least one training stage is required")
        if self.batch_size < 1 or self.accumulation_steps < 1 or self.num_workers < 0:
            raise ValueError("invalid batch size, accumulation steps or worker count")
        if self.precision not in {"none", "bf16", "fp16"}:
            raise ValueError("precision must be none, bf16 or fp16")
        if self.warmup_steps < 0 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("invalid warmup_steps or min_lr_ratio")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.latent_fps != 40:
            raise ValueError("the current SVAE clock is fixed to 40 latent frames/s")
        supported = {"diff_loss", "stop_loss", "ar_l1_loss", "vae_projected_l1_loss"}
        if not self.loss_weights or set(self.loss_weights) - supported:
            raise ValueError("loss_weights contains unsupported objectives")
        if any(not math.isfinite(value) or value < 0 for value in self.loss_weights.values()):
            raise ValueError("loss weights must be finite and nonnegative")
        if self.loss_weights.get("diff_loss", 0) <= 0:
            raise ValueError("diff_loss weight must be positive")
        durations = [stage.max_seconds for stage in self.stages]
        if durations != sorted(durations):
            raise ValueError("curriculum duration limits must not decrease")


def seed_training(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_batch(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_batch(item, device) for item in value]
    return value


def build_optimizer(model, config):
    """Keep backbone groups across freeze/unfreeze; avoid decay on norms/biases."""
    backbone_ids = {id(parameter) for parameter in model.causalAR.model.parameters()}
    encoder_ids = {id(parameter) for parameter in model.generator.parameters()}
    groups = {}
    for name, parameter in model.named_parameters():
        if id(parameter) in encoder_ids:
            continue
        kind = "backbone" if id(parameter) in backbone_ids else "adapter"
        decay = parameter.ndim >= 2 and not name.endswith("bias") and "embedding" not in name and "embed_tokens" not in name
        groups.setdefault((kind, decay), []).append(parameter)
    return torch.optim.AdamW([
        {"params": parameters, "kind": kind, "lr": 0.0,
         "weight_decay": config.weight_decay if decay else 0.0}
        for (kind, decay), parameters in groups.items()
    ])


def set_stage_learning_rate(optimizer, stage, step, total_steps, config):
    # The first update uses a nonzero LR. Clip warmup for short stages.
    warmup = min(config.warmup_steps, max(0, total_steps - 1))
    if step < warmup:
        factor = (step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total_steps - warmup - 1)
        factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * min(progress, 1))) / 2
    for group in optimizer.param_groups:
        base_lr = stage.backbone_lr if group["kind"] == "backbone" else stage.adapter_lr
        group["lr"] = base_lr * factor


def weighted_loss(metrics, weights):
    return sum(metrics[name] * weight for name, weight in weights.items())


class Trainer:
    def __init__(self, model, config, *, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if config.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; use none or bf16 on CPU")
        if config.precision == "bf16" and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("this CUDA device does not support bf16; use fp16 or none")
        self.model = model.to(self.device)
        if config.gradient_checkpointing:
            self.model.causalAR.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.model.LocDiT.checkpoint_activations = True
        self.optimizer = build_optimizer(model, config)
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        else:  # torch 2.2 compatibility
            self.scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16")
        self.global_step = 0
        self.completed_epochs = 0
        self.stage_step = 0
        self.best_validation = float("inf")

    def autocast(self):
        if self.config.precision == "none":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def train_epoch(self, loader, stage, *, total_stage_steps):
        self.model.causalAR.set_pretrained_body_trainable(stage.train_backbone)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {}
        samples = 0
        batch_count = len(loader)
        if not batch_count:
            raise ValueError("training loader is empty")
        for index, batch in enumerate(loader):
            batch = move_batch(batch, self.device)
            window_start = (index // self.config.accumulation_steps) * self.config.accumulation_steps
            window_size = min(self.config.accumulation_steps, batch_count - window_start)
            with self.autocast():
                metrics = self.model(**batch)
                loss = weighted_loss(metrics, self.config.loss_weights)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"nonfinite loss at optimizer step {self.global_step}")
            self.scaler.scale(loss / window_size).backward()
            count = batch["text_inputs"].shape[0]
            samples += count
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach()) * count
            if (index + 1) % self.config.accumulation_steps == 0 or index + 1 == batch_count:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    self.config.max_grad_norm, error_if_nonfinite=True,
                )
                set_stage_learning_rate(self.optimizer, stage, self.stage_step, total_stage_steps, self.config)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.stage_step += 1
        return {name: value / samples for name, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader):
        """Use fixed flow noise/time, restoring training RNG and model mode."""
        mode = self.model.training
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        totals, samples = {}, 0
        try:
            self.model.eval()
            with torch.random.fork_rng(devices=devices):
                # Seed only devices whose RNG is preserved by fork_rng.
                torch.random.default_generator.manual_seed(self.config.validation_seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(self.config.validation_seed)
                for batch in loader:
                    batch = move_batch(batch, self.device)
                    with self.autocast():
                        metrics = self.model(**batch)
                        metrics["loss"] = weighted_loss(metrics, self.config.loss_weights)
                    count = batch["text_inputs"].shape[0]
                    samples += count
                    for name, value in metrics.items():
                        if not torch.isfinite(value):
                            raise FloatingPointError(f"nonfinite validation metric: {name}")
                        totals[name] = totals.get(name, 0.0) + float(value) * count
        finally:
            self.model.train(mode)
        if not samples:
            raise ValueError("validation loader is empty")
        return {name: value / samples for name, value in totals.items()}

    def save(self, path, *, model_config, data_signature):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format_version": 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "model_config": model_config,
            "training_config": asdict(self.config),
            "data_signature": data_signature,
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "stage_step": self.stage_step,
            "best_validation": self.best_validation,
            "rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
            "torch_version": str(torch.__version__),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def restore(self, path, *, model_config, data_signature):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported training checkpoint format")
        if checkpoint["training_config"] != asdict(self.config) or checkpoint["model_config"] != model_config:
            raise ValueError("resume requires the original training and model configuration")
        if checkpoint["data_signature"] != data_signature:
            raise ValueError("resume requires the original data manifests")
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        for name in ("global_step", "completed_epochs", "stage_step", "best_validation"):
            setattr(self, name, checkpoint[name])
        torch.set_rng_state(checkpoint["rng"])
        random.setstate(checkpoint["python_rng"])
        if self.device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    def fit(self, train_data, validation_data, output_dir, *, model_config, data_signature):
        if train_data.speakers & validation_data.speakers:
            raise ValueError("training and validation speakers must be disjoint for zero-shot evaluation")
        # Keep validation fixed at the final duration limit across all stages.
        validation_loader = self.make_loader(
            validation_data, self.config.stages[-1].max_seconds, epoch=0, shuffle=False,
        )
        epoch_offset = 0
        for stage in self.config.stages:
            stage_end = epoch_offset + stage.epochs
            if self.completed_epochs >= stage_end:
                epoch_offset = stage_end
                continue
            for epoch in range(max(epoch_offset, self.completed_epochs), stage_end):
                if epoch == epoch_offset:
                    self.stage_step = 0
                loader = self.make_loader(train_data, stage.max_seconds, epoch=epoch, shuffle=True)
                updates_per_epoch = math.ceil(len(loader) / self.config.accumulation_steps)
                train_metrics = self.train_epoch(loader, stage, total_stage_steps=updates_per_epoch * stage.epochs)
                validation_metrics = self.evaluate(validation_loader)
                self.completed_epochs = epoch + 1
                improved = validation_metrics["loss"] < self.best_validation
                self.best_validation = min(self.best_validation, validation_metrics["loss"])
                self.save(Path(output_dir) / "latest.ckpt", model_config=model_config, data_signature=data_signature)
                if improved:
                    self.save(Path(output_dir) / "best.ckpt", model_config=model_config, data_signature=data_signature)
                yield {"epoch": epoch + 1, "step": self.global_step, "max_seconds": stage.max_seconds,
                       "train": train_metrics, "validation": validation_metrics}
            epoch_offset = stage_end

    def make_loader(self, dataset, max_seconds, *, epoch, shuffle):
        sampler = LengthBucketBatchSampler(
            dataset.lengths, self.config.batch_size,
            max_frames=int(max_seconds * self.config.latent_fps),
            seed=self.config.seed, epoch=epoch, shuffle=shuffle,
        )
        return DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate_speech_examples,
            num_workers=self.config.num_workers, pin_memory=self.device.type == "cuda",
            # A loader must not consume the model/flow RNG when iterated.
            generator=torch.Generator().manual_seed(self.config.seed + epoch),
        )

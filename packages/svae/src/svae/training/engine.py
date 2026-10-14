"""Single-device semantic codec GAN training and epoch-boundary resume."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from ..discriminators import SpeechDiscriminator
from ..loading import encoder_state
from ..losses import (SpectralReconstructionLoss, discriminator_loss,
                      feature_matching_loss, generator_adversarial_loss)
from ..runtime import accelerator_module, backend_name, resolve_device, validate_training_precision
from .data import require_disjoint


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 4
    clip_seconds: float = 3.0
    precision: str = "bf16"
    learning_rate: float = 1e-4
    lr_decay: float = 0.9996
    ema_decay: float = 0.999
    max_grad_norm: float = 1.0
    reconstruction_weight: float = 15.0
    kl_weight: float = 0.01
    adversarial_weight: float = 1.0
    feature_weight: float = 2.0
    semantic_weight: float = 1.0
    teacher_model: str = "microsoft/wavlm-large"
    teacher_revision: str | None = None
    fft_sizes: tuple[int, ...] = (2048, 1024, 512)
    n_mels: int = 80
    periods: tuple[int, ...] = (2, 3, 5, 7, 11)
    bands: tuple[tuple[float, float], ...] = ((0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.))
    discriminator_channels: int = 32
    num_workers: int = 0
    seed: int = 42
    validation_seed: int = 1234

    def __post_init__(self):
        for name, minimum in (("epochs", 1), ("batch_size", 1), ("num_workers", 0),
                              ("n_mels", 1), ("discriminator_channels", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"{name} must be an integer >={minimum}")
        for name in ("seed", "validation_seed"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("learning_rate", "max_grad_norm", "reconstruction_weight", "adversarial_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("kl_weight", "feature_weight", "semantic_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.lr_decay <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("lr_decay must be in (0,1] and ema_decay in [0,1)")
        if not math.isfinite(self.clip_seconds) or not 0 < self.clip_seconds <= 60:
            raise ValueError("clip_seconds must be finite and in (0,60]")
        if self.clip_samples < 400 or self.clip_samples % 400:
            raise ValueError("clip_seconds must use the 40 Hz latent grid (multiples of .025 seconds)")
        if self.semantic_weight > 0 and not self.teacher_model:
            raise ValueError("semantic alignment requires a pretrained teacher_model")
        object.__setattr__(self, "fft_sizes", tuple(self.fft_sizes))
        object.__setattr__(self, "periods", tuple(self.periods))
        object.__setattr__(self, "bands", tuple(tuple(band) for band in self.bands))

    @property
    def clip_samples(self):
        return round(self.clip_seconds * 16000)

    def resume_settings(self):
        values = asdict(self)
        # Extending the target epoch count does not alter the per-epoch LR schedule.
        values.pop("epochs")
        return values


def seed_training(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_codec(codec_config, state, ema_state, directory):
    """Publish a new immutable encoder/decoder pair with one directory rename."""
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"export destination already exists: {destination}")
    raw = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    ema = {key: value.detach().cpu().contiguous() for key, value in ema_state.items()}
    with tempfile.TemporaryDirectory(prefix=".svae-export-", dir=destination.parent) as temporary:
        root = Path(temporary) / "codec"
        (root / "dac").mkdir(parents=True)
        (root / "metainfo.json").write_text(json.dumps({"DAC": codec_config}, indent=2), encoding="utf-8")
        torch.save({"state_dict": raw}, root / "dac" / "weights.pth")
        torch.save({"ema_model." + key: value for key, value in ema.items()}, root / "dac" / "ema_state_dict.pth")
        save_file(encoder_state(ema), str(root / "encoder.safetensors"))
        os.replace(root, destination)
    return destination


class Trainer:
    format_version = 1

    def __init__(self, codec, config, *, codec_config, device="auto", teacher=None):
        self.device = resolve_device(device)
        validate_training_precision(self.device, config.precision)
        self.config, self.codec_config, self.teacher = config, copy.deepcopy(codec_config), teacher
        if codec.sample_rate != 16000 or codec.hop_length != 400:
            raise ValueError("the waveform trainer requires a 16 kHz / 40 Hz codec")
        if config.semantic_weight and (teacher is None or not codec.distill):
            raise ValueError("semantic training requires a real teacher and codec distill=true")
        if teacher is not None and teacher.hidden_size != codec.distill_hidden_dim:
            raise ValueError("teacher hidden_size must match codec distill_hidden_dim")
        self.codec = codec.to(self.device)
        self.discriminator = SpeechDiscriminator(periods=config.periods, fft_sizes=config.fft_sizes,
                                                bands=config.bands, channels=config.discriminator_channels).to(self.device)
        self.reconstruction = SpectralReconstructionLoss(fft_sizes=config.fft_sizes, n_mels=config.n_mels).to(self.device)
        self.optimizer = torch.optim.Adam(self.codec.parameters(), lr=config.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.lr_decay)
        self.discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=config.lr_decay)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=config.precision == "fp16")
        self.ema = {key: value.detach().clone() for key, value in self.codec.state_dict().items()}
        self.completed_epochs, self.global_step, self.best_validation = 0, 0, math.inf

    def _autocast(self):
        return torch.autocast(self.device.type, enabled=self.config.precision != "none",
                              dtype=torch.bfloat16 if self.config.precision == "bf16" else torch.float16)

    def _objective(self, output, real, real_outputs, fake_outputs):
        reconstruction = self.reconstruction(output["audio"], real)
        kl, semantic = output["vae/kl_loss"], output["vae/proj_loss"]
        adversarial = generator_adversarial_loss(fake_outputs)
        feature = feature_matching_loss(real_outputs, fake_outputs)
        cfg = self.config
        loss = (cfg.reconstruction_weight * reconstruction + cfg.kl_weight * kl +
                cfg.semantic_weight * semantic + cfg.adversarial_weight * adversarial + cfg.feature_weight * feature)
        return {"loss": loss, "reconstruction": reconstruction, "kl": kl, "semantic": semantic,
                "adversarial": adversarial, "feature_matching": feature}

    def _backward_step(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite SVAE objective; update aborted")
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("nonfinite SVAE gradients; update aborted")
        self.scaler.step(optimizer)

    @torch.no_grad()
    def _update_ema(self):
        # Codec buffers are fixed filter kernels, not running statistics.
        for name, value in self.codec.named_parameters():
            if value.grad is not None:
                self.ema[name].lerp_(value, 1 - self.config.ema_decay)

    def _loader(self, dataset, *, epoch, shuffle):
        dataset.epoch = epoch
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle,
                          num_workers=self.config.num_workers, pin_memory=self.device.type != "cpu",
                          generator=torch.Generator().manual_seed(self.config.seed + epoch))

    def train_epoch(self, dataset, *, epoch):
        self.codec.train()
        self.discriminator.train().requires_grad_(True)
        totals, count = {}, 0
        for cpu_audio in self._loader(dataset, epoch=epoch, shuffle=True):
            guidance = self.teacher(cpu_audio) if self.teacher is not None else None
            audio = cpu_audio.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = self.codec(audio, sample_rate=16000, guidance=guidance)
                d_loss = discriminator_loss(self.discriminator(audio), self.discriminator(output["audio"].detach()))
            self._backward_step(d_loss, self.discriminator_optimizer, self.discriminator.parameters())
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            self.discriminator.requires_grad_(False)
            try:
                with self._autocast():
                    with torch.no_grad():
                        real_outputs = self.discriminator(audio)
                    fake_outputs = self.discriminator(output["audio"])
                    metrics = self._objective(output, audio, real_outputs, fake_outputs)
                self._backward_step(metrics["loss"], self.optimizer, self.codec.parameters())
            finally:
                self.discriminator.requires_grad_(True)
            self.scaler.update()
            self._update_ema()
            self.global_step += 1
            metrics["discriminator"] = d_loss
            values = torch.stack([value.detach().float() for value in metrics.values()]).cpu().tolist()
            batch = len(cpu_audio)
            for key, value in zip(metrics, values):
                totals[key] = totals.get(key, 0.0) + value * batch
            count += batch
        self.scheduler.step()
        self.discriminator_scheduler.step()
        return {key: value / count for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, dataset):
        module = accelerator_module(self.device)
        devices = [] if module is None else [self.device.index]
        modes = self.codec.training, self.discriminator.training
        python_rng = random.getstate()
        try:
            with torch.random.fork_rng(devices=devices, device_type=self.device.type):
                random.seed(self.config.validation_seed)
                torch.random.default_generator.manual_seed(self.config.validation_seed)
                if module is not None:
                    with module.device(self.device):
                        module.manual_seed(self.config.validation_seed)
                self.codec.eval()
                self.discriminator.eval()
                totals, count = {}, 0
                for cpu_audio in self._loader(dataset, epoch=0, shuffle=False):
                    guidance = self.teacher(cpu_audio) if self.teacher is not None else None
                    audio = cpu_audio.to(self.device, non_blocking=True)
                    with self._autocast():
                        output = torch.func.functional_call(self.codec, self.ema, (audio,),
                                                            {"sample_rate": 16000, "guidance": guidance}, strict=True)
                        metrics = self._objective(output, audio, self.discriminator(audio), self.discriminator(output["audio"]))
                    values = torch.stack([value.float() for value in metrics.values()]).cpu().tolist()
                    if not all(math.isfinite(value) for value in values):
                        raise FloatingPointError("nonfinite SVAE validation objective")
                    for key, value in zip(metrics, values):
                        totals[key] = totals.get(key, 0.0) + value * len(cpu_audio)
                    count += len(cpu_audio)
                return {key: value / count for key, value in totals.items()}
        finally:
            random.setstate(python_rng)
            self.codec.train(modes[0])
            self.discriminator.train(modes[1])

    def save(self, path, *, data_signature):
        module = accelerator_module(self.device)
        checkpoint = {
            "svae_format_version": self.format_version, "config": self.config.resume_settings(),
            "target_epochs": self.config.epochs,
            "codec_config": self.codec_config, "data_signature": data_signature,
            "teacher_signature": self.teacher.signature if self.teacher is not None else None,
            "state_dict": self.codec.state_dict(), "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(), "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "discriminator_scheduler": self.discriminator_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(), "ema": self.ema,
            "completed_epochs": self.completed_epochs, "global_step": self.global_step, "best_validation": self.best_validation,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "accelerator_rng": {"backend": backend_name(self.device), "index": self.device.index,
                                "state": module.get_rng_state(self.device) if module is not None else None},
        }
        _atomic_save(checkpoint, path)

    def restore(self, path, *, data_signature):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = (self.config.resume_settings(), self.codec_config, data_signature,
                    self.teacher.signature if self.teacher is not None else None)
        actual = tuple(checkpoint.get(key) for key in ("config", "codec_config", "data_signature", "teacher_signature"))
        if checkpoint.get("svae_format_version") != self.format_version or expected != actual:
            raise ValueError("SVAE checkpoint config, codec, data or teacher differs from this run")
        rng = checkpoint["accelerator_rng"]
        if rng["backend"] != backend_name(self.device) or rng["index"] != self.device.index:
            raise ValueError("SVAE resume requires the original backend and device index")
        for model, key in ((self.codec, "state_dict"), (self.discriminator, "discriminator")):
            local, saved = model.state_dict(), checkpoint[key]
            if local.keys() != saved.keys() or any(local[name].shape != saved[name].shape for name in local):
                raise ValueError(f"SVAE checkpoint {key} has incompatible weights")
        self.codec.load_state_dict(checkpoint["state_dict"], strict=True)
        self.discriminator.load_state_dict(checkpoint["discriminator"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.ema = {name: value.to(self.device) for name, value in checkpoint["ema"].items()}
        self.completed_epochs = checkpoint["completed_epochs"]
        self.global_step, self.best_validation = checkpoint["global_step"], checkpoint["best_validation"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        module = accelerator_module(self.device)
        if module is not None:
            module.set_rng_state(rng["state"], self.device)

    def fit(self, train, validation, output_dir):
        require_disjoint(train, validation)
        signature = {"train": train.signature, "validation": validation.signature}
        output = Path(output_dir)
        for epoch in range(self.completed_epochs, self.config.epochs):
            train_metrics = self.train_epoch(train, epoch=epoch)
            validation_metrics = self.evaluate(validation)
            self.completed_epochs = epoch + 1
            improved = validation_metrics["loss"] < self.best_validation
            self.best_validation = min(self.best_validation, validation_metrics["loss"])
            self.save(output / "latest.ckpt", data_signature=signature)
            if improved:
                self.save(output / "best.ckpt", data_signature=signature)
            export = export_codec(self.codec_config, self.codec.state_dict(), self.ema,
                                  output / "exports" / f"epoch-{epoch + 1:06d}")
            yield {"epoch": epoch + 1, "step": self.global_step, "train": train_metrics,
                   "validation": validation_metrics, "export": str(export)}

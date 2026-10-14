"""Train a waveform codec, independently of CtrlSpeech and transcript processing."""

import argparse
import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..loading import DEFAULT_CODEC_CONFIG, build_codec, load_codec, read_codec_config
from ..runtime import backend_name, resolve_device, validate_training_precision
from .data import WaveDataset, require_disjoint
from .engine import Trainer, TrainingConfig, export_codec, seed_training
from .teacher import SemanticTeacher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="YAML containing codec and training sections")
    parser.add_argument("--init-from", type=Path, help="initialize from an exported encoder/decoder pair")
    parser.add_argument("--resume", type=Path, help="resume an epoch-boundary training checkpoint")
    parser.add_argument("--epochs", type=int, help="total target epoch count; may be extended on resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--clip-seconds", type=float)
    parser.add_argument("--precision", choices=("none", "bf16", "fp16"))
    parser.add_argument("--teacher-model", help="pretrained SSL model ID or local model directory")
    parser.add_argument("--semantic-weight", type=float, help="0 explicitly selects a vanilla acoustic VAE")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda[:N], rocm[:N], xpu[:N]")
    parser.add_argument("--validate-only", action="store_true", help="audit WAV crops without downloading models")
    args = parser.parse_args(argv)
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint.get("svae_format_version") != Trainer.format_version:
        raise ValueError("not a supported SVAE training checkpoint")
    codec_config = copy.deepcopy(checkpoint["codec_config"] if checkpoint else
                                 read_codec_config(args.init_from) if args.init_from else DEFAULT_CODEC_CONFIG)
    settings = dict(checkpoint["config"]) if checkpoint else {}
    if checkpoint:
        settings["epochs"] = checkpoint["target_epochs"]
    if args.config:
        document = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        if set(document) - {"codec", "training"}:
            raise ValueError("SVAE config accepts only codec and training sections")
        codec_config.update(document.get("codec", {}))
        settings.update(document.get("training", {}))
    for key in ("epochs", "batch_size", "clip_seconds", "precision", "teacher_model", "semantic_weight"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    config = TrainingConfig(**settings)
    if config.semantic_weight == 0:
        codec_config["distill"] = False
    elif not codec_config.get("distill", False):
        raise ValueError("semantic_weight>0 requires codec distill=true")
    train = WaveDataset(args.train_dir, config.clip_samples, seed=config.seed)
    validation = WaveDataset(args.validation_dir, config.clip_samples, seed=config.validation_seed)
    require_disjoint(train, validation)
    if args.validate_only:
        for dataset in (train, validation):
            for index in range(len(dataset)):
                dataset[index]
        print(json.dumps({"status": "valid", "train_examples": len(train), "validation_examples": len(validation),
                          "excluded_short_wavs": train.excluded + validation.excluded}))
        return
    device = resolve_device(args.device)
    validate_training_precision(device, config.precision)
    if (args.output_dir / "latest.ckpt").exists() and not args.resume:
        raise FileExistsError("output directory contains a training run; use --resume or a new directory")
    if checkpoint and config.epochs <= checkpoint["completed_epochs"]:
        raise ValueError("--epochs must exceed the checkpoint's completed epoch count")
    seed_training(config.seed)
    teacher = SemanticTeacher(config.teacher_model, device=device, revision=config.teacher_revision) if config.semantic_weight else None
    if teacher is not None and teacher.hidden_size != codec_config.get("distill_hidden_dim", 1024):
        raise ValueError("codec distill_hidden_dim must equal teacher hidden_size")
    if args.init_from:
        initial_config = read_codec_config(args.init_from)
        # distill controls the training objective, not the parameter architecture.
        initial_config["distill"] = codec_config.get("distill", False)
        if initial_config != codec_config:
            raise ValueError("--init-from architecture differs; do not override its codec configuration")
        codec = load_codec(args.init_from)
        codec.distill = codec_config.get("distill", False)
    else:
        codec = build_codec(codec_config)
    trainer = Trainer(codec, config, codec_config=codec_config, device=device, teacher=teacher)
    signature = {"train": train.signature, "validation": validation.signature}
    if args.resume:
        trainer.restore(args.resume, data_signature=signature)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"codec": codec_config, "training": settings}), args.output_dir / "config.yaml")
    print(json.dumps({"backend": backend_name(device), "device": str(device), "train_examples": len(train), "validation_examples": len(validation),
                      "excluded_short_wavs": train.excluded + validation.excluded,
                      "semantic_alignment": bool(config.semantic_weight)}), flush=True)
    for metrics in trainer.fit(train, validation, args.output_dir):
        print(json.dumps(metrics), flush=True)


def export_main(argv=None):
    parser = argparse.ArgumentParser(description="Export a paired SVAE encoder/decoder from a training checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("svae_format_version") != Trainer.format_version:
        raise ValueError("not a supported SVAE training checkpoint")
    directory = export_codec(checkpoint["codec_config"], checkpoint["state_dict"], checkpoint["ema"], args.output_dir)
    print(json.dumps({"export": str(directory), "completed_epochs": checkpoint["completed_epochs"]}))


if __name__ == "__main__":
    main()

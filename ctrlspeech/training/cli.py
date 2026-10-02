"""Train Japanese CtrlSpeech from audited, precomputed SVAE feature caches."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models import DiTar
from .data import CachedSpeechDataset, LengthBucketBatchSampler
from .engine import Trainer, TrainingConfig, TrainingStage, seed_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/japanese-lfm2-350m.yaml"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/japanese-training.yaml"))
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--svae-dir", type=Path, help="SVAE folder containing metainfo.json (required for training)")
    parser.add_argument("--encoder-weights", type=Path, help="trained SVAE encoder state dict matching the feature caches")
    parser.add_argument("--resume", type=Path, help="resume a training checkpoint at an epoch boundary")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("none", "bf16", "fp16"))
    parser.add_argument("--validate-only", action="store_true", help="audit all caches and splits without loading model weights")
    args = parser.parse_args(argv)
    if not args.validate_only and args.svae_dir is None:
        parser.error("training requires --svae-dir")
    if not args.validate_only and not args.resume and not args.encoder_weights:
        parser.error("a fresh run requires --encoder-weights; metainfo.json alone creates a random encoder")
    model_config = OmegaConf.load(args.model_config)
    if args.svae_dir:
        model_config.model.vocoder.path = args.svae_dir.resolve().as_posix()
    # A resumed checkpoint owns pretrained LFM weights; avoid downloading again.
    model_config.model.backbone.load_pretrained_weights = not bool(args.resume)
    settings = OmegaConf.to_container(OmegaConf.load(args.training_config), resolve=True)
    stages = tuple(TrainingStage(**stage) for stage in settings.pop("stages"))
    if args.precision:
        settings["precision"] = args.precision
    config = TrainingConfig(stages=stages, **settings)
    dataset_options = {"patch_size": model_config.model.patch_size, "audio_channels": model_config.model.audio_channels}
    train_data = CachedSpeechDataset(args.train_manifest, **dataset_options)
    validation_data = CachedSpeechDataset(args.validation_manifest, **dataset_options)
    if train_data.speakers & validation_data.speakers:
        raise ValueError("training and validation speaker IDs overlap")
    train_paths = {record["path"] for record in train_data.records}
    if train_paths & {record["path"] for record in validation_data.records}:
        raise ValueError("training and validation feature cache paths overlap")
    # Resolve all curriculum eligibility before expensive weight initialization.
    for stage in stages:
        LengthBucketBatchSampler(train_data.lengths, config.batch_size,
                                 max_frames=int(stage.max_seconds * config.latent_fps))
    LengthBucketBatchSampler(validation_data.lengths, config.batch_size,
                             max_frames=int(stages[-1].max_seconds * config.latent_fps))
    if args.validate_only:
        for dataset in (train_data, validation_data):
            for index in range(len(dataset)):
                dataset[index]
        print(json.dumps({"train_examples": len(train_data), "validation_examples": len(validation_data), "status": "valid"}))
        return
    seed_training(config.seed)
    model = DiTar(model_config.model)
    if args.encoder_weights and not args.resume:
        if args.encoder_weights.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(args.encoder_weights))
        else:
            state = torch.load(args.encoder_weights, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
        for prefix in ("model.generator.", "generator."):
            selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if selected:
                state = selected
                break
        model.generator.load_state_dict(state, strict=True)
    trainer = Trainer(model, config, device=args.device)
    # Initialization flags are runtime details, not checkpoint architecture.
    canonical_config = OmegaConf.to_container(model_config, resolve=True)
    canonical_config["model"]["backbone"]["load_pretrained_weights"] = False
    signature = {
        "train": hashlib.sha256(args.train_manifest.read_bytes()).hexdigest(),
        "validation": hashlib.sha256(args.validation_manifest.read_bytes()).hexdigest(),
    }
    if args.resume:
        trainer.restore(args.resume, model_config=canonical_config, data_signature=signature)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(canonical_config), args.output_dir / "config.yaml")
    for metrics in trainer.fit(train_data, validation_data, args.output_dir,
                               model_config=canonical_config, data_signature=signature):
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

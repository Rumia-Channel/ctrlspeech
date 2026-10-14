"""Real small LFM2/DiT/SARA paths; set CTRLSPEECH_TEST_DEVICE to require a GPU.

Weights and audio are synthetic. No codec encoder, sampler, decoder or optimizer
is mocked; LFM2's architecture is reduced and CPU speaker embeddings are cached.
"""

import json
import os

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf
from transformers import Lfm2Config

from ctrlspeech.assets import Assets, MODELS
from ctrlspeech.data import collate_japanese_sequences, encode_japanese_pair
from ctrlspeech.models.backbone import LFM2SpeechBackbone
from ctrlspeech.models.ditar import DiTar
from svae.loading import build_codec, encode_audio, load_encoder
from svae.training.engine import export_codec
from ctrlspeech.pipeline import CtrlSpeech
from svae.runtime import accelerator_module, resolve_device
from ctrlspeech.training import CachedSpeechDataset
from ctrlspeech.training.cli import main
from ctrlspeech.training.engine import Trainer, TrainingConfig, TrainingStage, seed_training


@pytest.fixture
def backend_assets(tmp_path, monkeypatch, small_codec_config):
    from ctrlspeech.models import ditar

    backbone_config = Lfm2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        layer_types=["conv", "full_attention"], conv_L_cache=3,
        block_multiple_of=16, block_auto_adjust_ff_dim=False, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )

    def small_backbone(**kwargs):
        return LFM2SpeechBackbone(
            phone_vocab_size=kwargs["phone_vocab_size"], config=backbone_config,
            load_pretrained_weights=False, validate_architecture=False,
        )

    monkeypatch.setattr(ditar, "LFM2SpeechBackbone", small_backbone)
    model_config = OmegaConf.load("configs/japanese-lfm2-350m.yaml")
    model_config.model.dim = 32
    model_config.model.mlp_hidden_dim = 16
    model_config.model.aggregation_encoder.update({
        "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 2,
    })
    model_config.model.loc_decoder.model.update({"dim": 32, "depth": 1, "heads": 2, "dim_head": 16})
    svae = tmp_path / "svae"
    seed_training(7)
    codec = build_codec(small_codec_config).eval()
    export_codec(codec.codec_config, codec.state_dict(), codec.state_dict(), svae)
    encoder = load_encoder(svae)
    weights = svae / "encoder.safetensors"
    model_config.model.vocoder.path = str(svae)
    config_path = tmp_path / "model.yaml"
    OmegaConf.save(model_config, config_path)
    prompt = torch.sin(torch.arange(3840, dtype=torch.float32) * (2 * torch.pi * 220 / 48000))[None] * 0.1
    target = torch.sin(torch.arange(4800, dtype=torch.float32) * (2 * torch.pi * 330 / 48000))[None] * 0.1
    sequence = encode_japanese_pair("あ", "い")
    text = collate_japanese_sequences([sequence])
    prompt_latents = encode_audio(prompt[:, None], encoder).transpose(1, 2)[0]
    target_latents = encode_audio(target[:, None], encoder).transpose(1, 2)[0]
    controls = torch.arange(20, dtype=torch.long)
    bounds = []
    for start, end, count in ((0, 8, len(sequence.prompt.phones)), (9, 19, len(sequence.target.phones))):
        if start:
            bounds.append((0, 0))
        boundaries = np.linspace(start, end, count + 1, dtype=int)
        bounds.extend(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))
    example = {
        "codec": {"name": "sara", "sample_rate": 48000, "latent_fps": 50, "latent_dim": 128},
        "vae_features": torch.cat((prompt_latents, target_latents)),
        "prompt_frames": len(prompt_latents), "speaker_embs": torch.zeros(192),
        "text_inputs": text["text_inputs"][0],
        "linguistic_features": {key: value[0] for key, value in text["linguistic_features"].items()},
        "pitch": controls + 20, "loudness": torch.full_like(controls, 32),
        "duration_segments": torch.tensor(bounds, dtype=torch.long),
    }
    manifests = []
    for split in ("train", "validation"):
        cache = tmp_path / f"{split}.pt"
        torch.save(example, cache)
        manifest = tmp_path / f"{split}.jsonl"
        manifest.write_text(json.dumps({"path": cache.name, "frames": len(example["vae_features"]),
                                        "speaker_id": split}) + "\n", encoding="utf-8")
        manifests.append(manifest)
    return model_config, config_path, svae, weights, prompt, text, example, manifests


@pytest.mark.parametrize("precision", ["none", "bf16"])
def test_real_cli_training_resume_and_waveform_inference(backend_assets, tmp_path, precision, monkeypatch):
    device = resolve_device(os.environ.get("CTRLSPEECH_TEST_DEVICE", "auto"))
    accelerator = accelerator_module(device)
    if precision == "bf16" and accelerator is not None:
        with accelerator.device(device):
            if not accelerator.is_bf16_supported():
                pytest.skip("device lacks bf16; the FP32 execution case still runs")
    model_config, config_path, svae, weights, prompt, text, example, manifests = backend_assets
    settings = {
        "stages": [{"epochs": 1, "max_seconds": 5, "train_backbone": True,
                    "adapter_lr": 0.001, "backbone_lr": 0.001}],
        "batch_size": 1, "accumulation_steps": 1, "precision": precision,
        "gradient_checkpointing": True, "warmup_steps": 0,
    }
    training_path = tmp_path / "training.yaml"
    OmegaConf.save(OmegaConf.create(settings), training_path)
    output = tmp_path / "checkpoints"
    main(["--model-config", str(config_path), "--training-config", str(training_path),
          "--train-manifest", str(manifests[0]), "--validation-manifest", str(manifests[1]),
          "--svae-dir", str(svae), "--encoder-weights", str(weights),
          "--output-dir", str(output), "--device", str(device)])
    checkpoint = torch.load(output / "latest.ckpt", map_location="cpu", weights_only=True)
    model = DiTar(model_config.model)
    stages = tuple(TrainingStage(**stage) for stage in settings["stages"])
    trainer = Trainer(model, TrainingConfig(stages=stages, **{key: value for key, value in settings.items() if key != "stages"}), device=device)
    trainer.restore(output / "latest.ckpt", model_config=checkpoint["model_config"], data_signature=checkpoint["data_signature"])
    train = CachedSpeechDataset(manifests[0])
    before = {name: parameter.detach().clone() for name, parameter in model.causalAR.model.named_parameters()}
    trainer.train_epoch(trainer.make_loader(train, 5, epoch=1, shuffle=False), stages[0], total_stage_steps=2)
    assert trainer.global_step == 2
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in model.causalAR.model.named_parameters())
    assert all(parameter.device == device for parameter in model.parameters())
    # The external CPU ONNX speaker model is not part of this accelerator check.
    import ctrlspeech.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "CosyVoiceSpeakerEmbedding", lambda **kwargs: None)
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"<pad>": 0}), encoding="utf-8")
    assets = Assets(
        spec=MODELS["japanese-lfm2-350m"], root=tmp_path, config_path=config_path,
        weights_path=output / "latest.ckpt", svae_dir=svae, vocab_path=vocab,
        campplus_path=tmp_path / "campplus.onnx",
    )
    pipeline = CtrlSpeech.from_assets(assets, device=device)
    arguments = dict(
        prompt_audio=prompt, speaker_emb=example["speaker_embs"],
        input_ids=text["input_ids"], text_masks=text["text_masks"],
        duration_segments=example["duration_segments"], pitch=example["pitch"], loudness=example["loudness"],
        max_seq_length=2, steps=2, cfg_strength=1.5, linguistic_features=text["linguistic_features"],
    )
    seed_training(17)
    audio = pipeline._sample(**arguments)
    seed_training(17)
    repeated = pipeline._sample(**arguments)
    assert audio.shape == (7680,)  # Two 80 ms AR patches decoded natively at 48 kHz.
    assert np.isfinite(audio).all()
    np.testing.assert_allclose(audio, repeated, rtol=1e-4, atol=1e-5)
    wav = tmp_path / "generated.wav"
    sf.write(wav, audio, 48000, subtype="FLOAT")
    decoded, rate = sf.read(wav, dtype="float32")
    assert rate == 48000
    np.testing.assert_array_equal(decoded, audio)


def test_hifigan_native_decoder_preserves_the_50hz_audio_clock(small_codec_config):
    from svae.models.sara import SARADecoder

    device = resolve_device(os.environ.get("CTRLSPEECH_TEST_DEVICE", "auto"))
    vocoder = SARADecoder(small_codec_config).to(device).eval()
    latents = torch.randn(1, 8, 128, device=device)
    with torch.no_grad():
        audio = vocoder.decode(latents)
        changed = vocoder.decode(latents + 0.1)
    assert audio.shape == (1, 1, 7680)
    assert torch.isfinite(audio).all()
    assert not torch.equal(audio, changed)


@pytest.mark.parametrize("component", ["codec", "local_decoder"])
def test_tts_rejects_inconsistent_latent_dimensions(backend_assets, component):
    config = OmegaConf.create(OmegaConf.to_container(backend_assets[0], resolve=True)).model
    config.audio_channels = 64
    if component == "codec":
        config.loc_decoder.model.out_dim = 64
    with pytest.raises(ValueError):
        DiTar(config)


import os


import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from svae.loading import build_codec, encode_audio, load_decoder, load_encoder
from svae.runtime import accelerator_module, resolve_device
from svae.training.cli import export_main, main
from svae.training.data import WaveDataset, require_disjoint
from svae.training.engine import Trainer, TrainingConfig, seed_training


@pytest.fixture
def waveform_assets(tmp_path, small_codec_config):
    for split, frequencies in (("train", (200, 220, 250)), ("validation", (330,))):
        folder = tmp_path / split
        folder.mkdir()
        for index, frequency in enumerate(frequencies):
            wave = .1 * np.sin(2 * np.pi * frequency * np.arange(9600) / 48000)
            sf.write(folder / f"{index}.wav", wave, 48000, subtype="FLOAT")
    settings = dict(epochs=2, batch_size=2, clip_seconds=.08, precision="none",
                    warmup_steps=3, fft_sizes=(64,), n_mels=8,
                    periods=(2,), bands=((0, 1),), discriminator_channels=2)
    return small_codec_config, settings


def components(tmp_path, codec_config, settings):
    config = TrainingConfig(**settings)
    device = resolve_device(os.environ.get("SVAE_TEST_DEVICE", "auto"))
    codec = build_codec(codec_config)
    trainer = Trainer(codec, config, device=device)
    train = WaveDataset(tmp_path / "train", config.clip_samples, seed=config.seed)
    validation = WaveDataset(tmp_path / "validation", config.clip_samples, seed=config.validation_seed)
    return trainer, train, validation


def test_real_semantic_gan_resume_matches_uninterrupted_updates(tmp_path, waveform_assets):
    codec, settings = waveform_assets
    seed_training(42)
    trainer, train, validation = components(tmp_path, codec, settings)
    teacher_before = {name: parameter.detach().clone() for name, parameter in trainer.codec.semantic.model.named_parameters()}
    critic_before = {name: parameter.detach().clone() for name, parameter in trainer.discriminator.named_parameters()}
    projector_before = {name: parameter.detach().clone() for name, parameter in trainer.codec.fusion.named_parameters()}
    trainer.train_epoch(train, epoch=0)
    assert any(not torch.equal(projector_before[name], parameter.detach()) for name, parameter in trainer.codec.fusion.named_parameters())
    assert any(not torch.equal(critic_before[name], parameter.detach()) for name, parameter in trainer.discriminator.named_parameters())
    assert all(parameter.grad is None and not parameter.requires_grad and torch.equal(teacher_before[name], parameter.detach())
               for name, parameter in trainer.codec.semantic.model.named_parameters())
    rng = torch.get_rng_state().clone()
    first = trainer.evaluate(validation)
    second = trainer.evaluate(validation)
    assert first == pytest.approx(second, rel=1e-4, abs=1e-5)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert trainer.codec.training and trainer.discriminator.training
    trainer.completed_epochs = 1
    signature = {"train": train.signature, "validation": validation.signature}
    trainer.save(tmp_path / "resume.ckpt", data_signature=signature)
    expected_metrics = trainer.train_epoch(train, epoch=1)
    expected = {key: value.detach().clone() for key, value in trainer.codec.state_dict().items()}
    expected_critic = {key: value.detach().clone() for key, value in trainer.discriminator.state_dict().items()}
    resumed, resumed_data, _ = components(tmp_path, codec, settings)
    resumed.restore(tmp_path / "resume.ckpt", data_signature=signature)
    actual_metrics = resumed.train_epoch(resumed_data, epoch=1)
    assert actual_metrics == pytest.approx(expected_metrics, rel=1e-4, abs=1e-5)
    for key, value in resumed.codec.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=1e-4, atol=1e-5)
    for key, value in resumed.discriminator.state_dict().items():
        torch.testing.assert_close(value, expected_critic[key], rtol=1e-4, atol=1e-5)
    assert resumed.global_step == trainer.global_step == 4


@pytest.mark.parametrize("precision", ["none", "bf16"])
def test_cli_trains_resumes_and_exports_usable_waveform_codec(tmp_path, waveform_assets, precision):
    codec, settings = waveform_assets
    device = resolve_device(os.environ.get("SVAE_TEST_DEVICE", "auto"))
    module = accelerator_module(device)
    if precision == "bf16" and module is not None:
        with module.device(device):
            if not module.is_bf16_supported():
                pytest.skip("device lacks BF16; FP32 execution remains required")
    settings = {**settings, "epochs": 1, "precision": precision}
    path = tmp_path / "config.yaml"
    OmegaConf.save(OmegaConf.create({"codec": codec, "training": settings}), path)
    output = tmp_path / "run"
    arguments = ["--train-dir", str(tmp_path / "train"), "--validation-dir", str(tmp_path / "validation"),
                 "--output-dir", str(output), "--device", str(device)]
    main([*arguments, "--config", str(path)])
    first = torch.load(output / "latest.ckpt", map_location="cpu", weights_only=True)
    main([*arguments, "--resume", str(output / "latest.ckpt"), "--epochs", "2"])
    checkpoint = torch.load(output / "latest.ckpt", map_location="cpu", weights_only=True)
    assert checkpoint["completed_epochs"] == 2 and checkpoint["global_step"] == 4
    assert any(not torch.equal(first["state_dict"][key], value) for key, value in checkpoint["state_dict"].items())
    exported = tmp_path / "paired"
    export_main(["--checkpoint", str(output / "latest.ckpt"), "--output-dir", str(exported)])
    encoder, decoder = load_encoder(exported).to(device), load_decoder(exported).to(device)
    audio = torch.sin(torch.arange(3840, device=device) * .08)[None, None] * .1
    with torch.no_grad():
        latent = encode_audio(audio, encoder)
        reconstructed = decoder.decode(latent.transpose(1, 2)).float().cpu()
    assert latent.shape == (1, 128, 4)
    assert reconstructed.shape == (1, 1, 3840) and torch.isfinite(reconstructed).all()
    sf.write(tmp_path / "reconstructed.wav", reconstructed[0, 0].numpy(), 48000, subtype="FLOAT")
    value, rate = sf.read(tmp_path / "reconstructed.wav", dtype="float32")
    assert rate == 48000
    np.testing.assert_array_equal(value, reconstructed[0, 0].numpy())


def test_changed_data_and_backend_are_rejected_before_mutation(tmp_path, waveform_assets):
    codec, settings = waveform_assets
    trainer, train, validation = components(tmp_path, codec, settings)
    signature = {"train": train.signature, "validation": validation.signature}
    trainer.save(tmp_path / "resume.ckpt", data_signature=signature)
    before = {key: value.detach().clone() for key, value in trainer.codec.state_dict().items()}
    with pytest.raises(ValueError):
        trainer.restore(tmp_path / "resume.ckpt", data_signature={**signature, "train": "different"})
    checkpoint = torch.load(tmp_path / "resume.ckpt", map_location="cpu", weights_only=True)
    checkpoint["accelerator_rng"]["backend"] = "unavailable-backend"
    torch.save(checkpoint, tmp_path / "wrong.ckpt")
    with pytest.raises(ValueError):
        trainer.restore(tmp_path / "wrong.ckpt", data_signature=signature)
    for key, value in trainer.codec.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_waveform_crops_are_independent_of_model_rng_and_resample(tmp_path):
    source = tmp_path / "audio"
    source.mkdir()
    value = .1 * np.sin(np.arange(8000) * .1)
    sf.write(source / "stereo.wav", np.stack([value, value], axis=1), 8000, subtype="FLOAT")
    data = WaveDataset(source, 3840, seed=7)
    first = data[0]
    torch.manual_seed(12)
    rng = torch.get_rng_state().clone()
    torch.testing.assert_close(data[0], first, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert first.shape == (1, 3840) and torch.isfinite(first).all()
    data.epoch = 1
    assert not torch.equal(data[0], first)
    with pytest.raises(ValueError):
        require_disjoint(data, WaveDataset(source, 3840))


def test_validation_failure_restores_mode_and_rng(tmp_path, waveform_assets):
    codec, settings = waveform_assets
    trainer, train, validation = components(tmp_path, codec, settings)
    trainer.codec.train()
    trainer.discriminator.eval()
    rng = torch.get_rng_state().clone()
    sf.write(tmp_path / "validation" / "0.wav", np.full(9600, np.nan), 48000, subtype="FLOAT")
    with pytest.raises(ValueError):
        trainer.evaluate(validation)
    assert trainer.codec.training and not trainer.discriminator.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_native_48khz_waveform_crops_preserve_content_above_12khz(tmp_path):
    source = tmp_path / "high-band"
    source.mkdir()
    frequency, rate, samples = 15000, 48000, 4800
    waveform = .1 * np.sin(2 * np.pi * frequency * np.arange(samples) / rate)
    sf.write(source / "voice.wav", waveform, rate, subtype="FLOAT")
    actual = WaveDataset(source, samples)[0][0]
    spectrum = torch.fft.rfft(actual)
    peak = int(spectrum.abs().argmax())
    assert peak * rate / samples == frequency
    assert float(2 * spectrum[peak].abs() / samples) == pytest.approx(.1, rel=1e-5)

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from svae.loading import build_codec, encode_audio, load_codec, load_decoder, load_encoder
from svae.models.sara import SARA
from svae.training.engine import export_codec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_kl_is_finite_at_encoder_variance_limit(dtype):
    mu = torch.full((2, 4, 64), 3000, dtype=dtype, requires_grad=True)
    variance = torch.full_like(mu, 12, requires_grad=True)
    actual = SARA.compute_kl_loss(mu, variance)
    expected = .5 * (mu.double().square() + variance.double().expm1() - variance.double()).sum(-1).mean()
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-3)
    actual.backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(variance.grad).all()
    assert SARA.compute_kl_loss(torch.zeros_like(mu), torch.zeros_like(variance)).item() == 0


def test_export_is_offline_strict_pair_with_single_posterior_sample(tmp_path, small_codec_config):
    torch.manual_seed(7)
    model = build_codec(small_codec_config).eval()
    directory = tmp_path / "codec"
    export_codec(model.codec_config, model.state_dict(), model.state_dict(), directory)
    shutil.rmtree(Path(small_codec_config["semantic_model"]))
    encoder, decoder, complete = load_encoder(directory), load_decoder(directory), load_codec(directory)
    audio = torch.sin(torch.arange(3840) * .05)[None, None] * .1
    torch.manual_seed(17)
    with torch.no_grad():
        expected = complete(audio, sample_rate=48000)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(17)
    latent = encode_audio(audio, encoder)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    torch.testing.assert_close(latent.transpose(1, 2), expected["z"], rtol=0, atol=0)
    reconstructed = decoder.decode(latent.transpose(1, 2))
    torch.testing.assert_close(reconstructed, expected["audio"], rtol=1e-5, atol=1e-6)
    assert reconstructed.shape == (1, 1, 3840) and not reconstructed.requires_grad
    state = load_file(str(directory / "ema.safetensors"))
    del state[next(key for key in state if key.startswith("decoder."))]
    save_file(state, str(directory / "ema.safetensors"))
    with pytest.raises(RuntimeError):
        load_decoder(directory)


def test_export_refuses_to_overwrite_an_existing_pair(tmp_path, small_codec_config):
    model = build_codec(small_codec_config)
    export_codec(model.codec_config, model.state_dict(), model.state_dict(), tmp_path / "codec")
    with pytest.raises(FileExistsError):
        export_codec(model.codec_config, model.state_dict(), model.state_dict(), tmp_path / "codec")


@pytest.mark.parametrize("use_ema", [False, True])
def test_raw_and_ema_exports_load_the_correct_pair(tmp_path, small_codec_config, use_ema):
    model = build_codec(small_codec_config).eval()
    raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ema = {key: value.clone() for key, value in raw.items()}
    bias = next(key for key in ema if key.startswith("decoder.") and key.endswith("bias"))
    ema[bias] += .25
    directory = tmp_path / "paired"
    export_codec(model.codec_config, raw, ema, directory)
    encoder = load_encoder(directory, use_ema=use_ema)
    decoder = load_decoder(directory, use_ema=use_ema)
    model.load_state_dict(ema if use_ema else raw, strict=True)
    wave = torch.sin(torch.arange(3840) * .1)[None, None] * .1
    torch.manual_seed(42)
    expected = model(wave, sample_rate=48000)["audio"]
    torch.manual_seed(42)
    latent = encode_audio(wave, encoder)
    torch.testing.assert_close(decoder.decode(latent.transpose(1, 2)), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("samples", [960, 3840, 4800, 5001])
def test_posterior_keeps_last_waveform_cell_and_forward_restores_length(small_codec_config, samples):
    model = build_codec(small_codec_config).eval()
    wave = torch.sin(torch.arange(samples) * .05)[None, None] * .1
    with torch.no_grad():
        output = model(wave, sample_rate=48000)
    assert output["z"].shape == (1, (samples + 959) // 960, 128)
    assert output["audio"].shape == wave.shape and torch.isfinite(output["audio"]).all()


def test_configured_64_dimension_pair_keeps_clock_and_rejects_wrong_latents(tmp_path, small_codec_config):
    model = build_codec({**small_codec_config, "vae_dim": 64}).eval()
    directory = tmp_path / "comparison"
    export_codec(model.codec_config, model.state_dict(), model.state_dict(), directory)
    encoder, decoder = load_encoder(directory), load_decoder(directory)
    wave = torch.sin(torch.arange(3840) * .07)[None, None] * .1
    latent = encode_audio(wave, encoder)
    assert latent.shape == (1, 64, 4)
    assert decoder.decode(latent.transpose(1, 2)).shape == (1, 1, 3840)
    with pytest.raises(ValueError):
        decoder.decode(torch.zeros(1, 4, 128))

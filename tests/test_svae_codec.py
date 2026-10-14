

import pytest
import torch

from svae.loading import build_codec, encode_audio, load_codec, load_decoder, load_encoder
from svae.models.dac.model import DAC
from svae.models.dac.model.regulator import InterpolateRegulator
from svae.training.engine import export_codec


@pytest.fixture
def small_codec_config():
    return dict(encoder_dim=4, encoder_rates=[4, 4, 5, 5], latent_dim=64,
                decoder_dim=64, decoder_rates=[5, 5, 4, 4], vae_dim=64,
                sample_rate=16000, decoder_type="dac", distill=False, distill_hidden_dim=8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_kl_is_finite_at_encoder_variance_limit(dtype):
    mu = torch.full((2, 4, 64), 3000, dtype=dtype, requires_grad=True)
    variance = torch.full_like(mu, 12, requires_grad=True)
    actual = DAC.compute_kl_loss(None, mu, variance)
    expected = .5 * (mu.double().square() + variance.double().expm1() - variance.double()).sum(-1).mean()
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-3)
    actual.backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(variance.grad).all()
    prior = DAC.compute_kl_loss(None, torch.zeros_like(mu), torch.zeros_like(variance))
    assert prior.item() == 0


def test_projection_uses_valid_input_and_target_lengths():
    torch.manual_seed(42)
    projection = InterpolateRegulator([0, 1], 4, 8, 8)
    values = torch.randn(2, 8, 4)
    values[0, 4:] = float("nan")
    values.requires_grad_()
    actual, _ = projection(values, [4, 8], [8, 6])
    single, _ = projection(values[:1, :4], [4], [8])
    torch.testing.assert_close(actual[:1], single)
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0, 4:]).item() > 0
    assert torch.count_nonzero(actual[1, 6:]).item() == 0
    actual.sum().backward()
    assert torch.count_nonzero(values.grad[0, 4:]).item() == 0
    assert torch.count_nonzero(values.grad[0, :4]).item() > 0


def test_semantic_loss_observes_teacher_frames_after_latent_length(small_codec_config):
    config = {**small_codec_config, "distill": True}
    model = build_codec(config).train()
    audio = torch.randn(1, 1, 1600) * .05
    torch.manual_seed(11)
    latent = model.encode(audio)[0]
    guidance, _ = model.projectors(latent, [4], [8])
    guidance = guidance.detach()
    torch.manual_seed(11)
    positive = model(audio, sample_rate=16000, guidance=guidance)["vae/proj_loss"]
    flipped = guidance.clone()
    flipped[:, 4:] *= -1
    torch.manual_seed(11)
    changed = model(audio, sample_rate=16000, guidance=flipped)["vae/proj_loss"]
    torch.testing.assert_close(positive, torch.tensor(-1.), atol=1e-5, rtol=0)
    torch.testing.assert_close(changed, torch.tensor(0.), atol=1e-5, rtol=0)
    model.eval()
    torch.manual_seed(11)
    torch.testing.assert_close(model(audio, sample_rate=16000, guidance=guidance)["vae/proj_loss"], positive)


def test_export_is_a_strict_paired_codec_with_single_posterior_sample(tmp_path, small_codec_config):
    torch.manual_seed(7)
    model = build_codec(small_codec_config).eval()
    directory = tmp_path / "codec"
    export_codec(small_codec_config, model.state_dict(), model.state_dict(), directory)
    encoder, decoder, complete = load_encoder(directory), load_decoder(directory), load_codec(directory)
    audio = torch.sin(torch.arange(1600) * .05)[None, None] * .1
    torch.manual_seed(17)
    expected = complete(audio, sample_rate=16000)
    torch.manual_seed(17)
    latent = encode_audio(audio, encoder)
    torch.testing.assert_close(latent.transpose(1, 2), expected["z"], rtol=0, atol=0)
    reconstructed = decoder.decode(latent.transpose(1, 2))
    torch.testing.assert_close(reconstructed, expected["audio"], rtol=1e-5, atol=1e-6)
    assert reconstructed.shape == (1, 1, 1600)
    assert not reconstructed.requires_grad
    checkpoint = torch.load(directory / "dac" / "ema_state_dict.pth", weights_only=True)
    del checkpoint[next(key for key in checkpoint if key.startswith("ema_model.decoder."))]
    torch.save(checkpoint, directory / "dac" / "ema_state_dict.pth")
    with pytest.raises(RuntimeError):
        load_decoder(directory)


def test_export_refuses_to_overwrite_an_existing_pair(tmp_path, small_codec_config):
    model = build_codec(small_codec_config)
    export_codec(small_codec_config, model.state_dict(), model.state_dict(), tmp_path / "codec")
    with pytest.raises(FileExistsError):
        export_codec(small_codec_config, model.state_dict(), model.state_dict(), tmp_path / "codec")

@pytest.mark.parametrize("use_ema", [False, True])
def test_raw_and_ema_exports_load_the_correct_pair(tmp_path, small_codec_config, use_ema):
    model = build_codec(small_codec_config).eval()
    raw = {key: value.detach().clone() for key, value in model.state_dict().items()}
    ema = {key: value.clone() for key, value in raw.items()}
    bias = next(key for key in ema if key.startswith("decoder.") and key.endswith("bias"))
    ema[bias] += .25
    directory = tmp_path / "paired"
    export_codec(small_codec_config, raw, ema, directory)
    path = directory / "dac" / ("ema_state_dict.pth" if use_ema else "weights.pth")
    stored = torch.load(path, weights_only=True)
    if use_ema:
        stored.update(initted=torch.tensor(True), step=torch.tensor(5))
    else:
        stored["state_dict"].update(initted=torch.tensor(True), step=torch.tensor(5))
    torch.save(stored, path)
    encoder = load_encoder(directory, use_ema=use_ema)
    decoder = load_decoder(directory, use_ema=use_ema)
    model.load_state_dict(ema if use_ema else raw, strict=True)
    wave = torch.sin(torch.arange(1600) * .1)[None, None] * .1
    torch.manual_seed(42)
    expected = model(wave, sample_rate=16000)["audio"]
    torch.manual_seed(42)
    latent = encode_audio(wave, encoder)
    torch.testing.assert_close(decoder.decode(latent.transpose(1, 2)), expected, rtol=1e-5, atol=1e-6)
    state = torch.load(path, weights_only=True)
    values = state if use_ema else state["state_dict"]
    del values[next(key for key in values if key.startswith("ema_model.decoder." if use_ema else "decoder."))]
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_decoder(directory, use_ema=use_ema)

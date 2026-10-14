import pytest
import torch

from svae.discriminators import SpeechDiscriminator
from svae.losses import (SpectralReconstructionLoss, discriminator_loss,
                        feature_matching_loss, generator_adversarial_loss)


def test_hinge_objective_rejects_reversed_real_fake_classification():
    good_real, good_fake = [(torch.tensor([2.]), [])], [(torch.tensor([-2.]), [])]
    assert discriminator_loss(good_real, good_fake).item() == 0
    assert discriminator_loss(good_fake, good_real).item() == 6
    fake = torch.tensor([2.], requires_grad=True)
    generator_adversarial_loss([(fake, [])]).backward()
    torch.testing.assert_close(fake.grad, torch.tensor([-1.]))


@pytest.mark.parametrize("autocast", [False, True])
def test_short_waveforms_have_real_spectral_and_gan_gradients(autocast):
    critic = SpeechDiscriminator(periods=(2, 3), fft_sizes=(64,), bands=((0, .5), (.5, 1)), channels=2)
    spectral = SpectralReconstructionLoss(fft_sizes=(64,), n_mels=8)
    real = (torch.randn(2, 1, 37) * .1).requires_grad_()
    fake = (torch.randn(2, 1, 37) * .1).requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual, expected = critic(fake), critic(real)
        loss = spectral(fake, real) + generator_adversarial_loss(actual) + feature_matching_loss(expected, actual)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(fake.grad).all() and torch.count_nonzero(fake.grad).item() > 0
    assert real.grad is None
    for parameter in critic.parameters():
        parameter.grad = None
    d_loss = discriminator_loss(critic(real.detach()), critic(fake.detach()))
    d_loss.backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0 for parameter in critic.parameters())
    torch.testing.assert_close(spectral(real, real), torch.tensor(0.))


@pytest.mark.parametrize("kwargs", [{"periods": (1,)}, {"bands": ((.5, .5),)}, {"channels": 0}])
def test_invalid_discriminator_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SpeechDiscriminator(**kwargs)

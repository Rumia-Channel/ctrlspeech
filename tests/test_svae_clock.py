import copy

import pytest

from svae.loading import DEFAULT_DECODER_CONFIG, build_codec


@pytest.mark.parametrize("decoder_type", ["dac", "bigvgan"])
def test_overlong_decoder_is_rejected_before_training_can_truncate_it(decoder_type):
    config = dict(encoder_dim=4, encoder_rates=[4, 4, 5, 5], latent_dim=64,
                  decoder_dim=64, vae_dim=64, sample_rate=16000, distill=False,
                  distill_hidden_dim=8, decoder_type=decoder_type)
    if decoder_type == "dac":
        config["decoder_rates"] = [10, 5, 4, 4]  # 800 samples/latent, encoder has 400.
    else:
        decoder = copy.deepcopy(DEFAULT_DECODER_CONFIG)
        decoder["upsample_rates"] = [5, 5, 2, 2, 2, 4]
        config["decoder_dict"] = decoder
    with pytest.raises(ValueError, match="upsampling factor.*encoder hop"):
        build_codec(config)

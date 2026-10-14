import json

import pytest

from svae.loading import DEFAULT_CODEC_CONFIG, build_codec, read_codec_config


def test_overlong_decoder_is_rejected_before_training_can_truncate_it():
    config = {**DEFAULT_CODEC_CONFIG, "decoder_rates": [10, 4, 4, 3, 4]}
    with pytest.raises(ValueError, match="upsampling factor.*encoder hop"):
        build_codec(config)


def test_legacy_codec_cannot_be_loaded_as_sara(tmp_path):
    (tmp_path / "metainfo.json").write_text(json.dumps({"DAC": {"sample_rate": 16000}}))
    with pytest.raises(ValueError):
        read_codec_config(tmp_path)

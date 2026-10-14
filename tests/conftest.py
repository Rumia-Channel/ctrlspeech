"""CPU regressions with optional, explicit ONNX isolation.

CTRLSPEECH_TEST_ISOLATE_ONNX=1 never imports the native ONNX runtime. It checks
Python integration and text-only OpenJTalk, not speaker/neural ONNX inference.
"""
import os
import sys
from types import ModuleType
from importlib.machinery import ModuleSpec

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

if os.environ.get('CTRLSPEECH_TEST_ISOLATE_ONNX') == '1':
    if 'onnxruntime' in sys.modules:
        raise RuntimeError('ONNX runtime was imported before the isolated test boundary')
    runtime = ModuleType('onnxruntime')
    runtime.__spec__ = ModuleSpec('onnxruntime', loader=None)
    runtime.disable_telemetry_events = lambda: None
    # No InferenceSession: pyopenjtalk's optional neural reading correction
    # follows its documented missing-runtime fallback. Actual ONNX inference
    # fails explicitly rather than pretending to produce a result.
    sys.modules['onnxruntime'] = runtime
else:
    import onnxruntime
    onnxruntime.disable_telemetry_events()


import pytest


@pytest.fixture
def small_codec_config(tmp_path):
    """Real reduced SSL architecture with synthetic local test weights."""
    import torch
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel
    from svae.loading import DEFAULT_CODEC_CONFIG

    directory = tmp_path / "semantic"
    with torch.random.fork_rng():
        torch.manual_seed(7)
        semantic = Wav2Vec2BertModel(Wav2Vec2BertConfig(
            hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
            feature_projection_input_dim=160, conv_depthwise_kernel_size=3,
            position_embeddings_type="relative_key", apply_spec_augment=False,
            hidden_dropout=0, attention_dropout=0, conformer_conv_dropout=0, layerdrop=0))
        semantic.save_pretrained(directory)
    SeamlessM4TFeatureExtractor().save_pretrained(directory)
    return {**DEFAULT_CODEC_CONFIG, "semantic_model": str(directory), "encoder_dim": 2,
            "acoustic_dim": 8, "decoder_dim": 64,
            "resblock_kernel_sizes": [3], "resblock_dilations": [1, 3]}

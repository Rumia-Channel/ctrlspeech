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

import importlib

import onnxruntime

import ctrlspeech


def test_package_disables_onnx_telemetry_before_speech_processing(monkeypatch):
    calls = []
    monkeypatch.setattr(onnxruntime, 'disable_telemetry_events', lambda: calls.append('disabled'))
    importlib.reload(ctrlspeech)
    assert calls == ['disabled']

import pytest
from ctrlspeech import assets


def complete_assets(root):
    for name in ('japanese-lfm2-350m/config.yaml', 'japanese-lfm2-350m/model.safetensors',
                 'svae/metainfo.json', 'svae/dac/ema_state_dict.pth',
                 'shared/vocab.json', 'shared/campplus.onnx'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_local_assets_resolve_relative_path_and_require_decoder(tmp_path, monkeypatch):
    complete_assets(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    monkeypatch.setenv('CTRLSPEECH_ASSETS', tmp_path.name)
    result = assets.download_assets()
    assert result.root.is_absolute()
    (tmp_path / 'svae/dac/ema_state_dict.pth').unlink()
    with pytest.raises(FileNotFoundError, match='ema_state_dict.pth'):
        assets.download_assets()


def test_directory_is_not_a_checkpoint(tmp_path, monkeypatch):
    complete_assets(tmp_path)
    checkpoint = tmp_path / 'japanese-lfm2-350m/model.safetensors'
    checkpoint.unlink()
    checkpoint.mkdir()
    monkeypatch.setenv('CTRLSPEECH_ASSETS', str(tmp_path))
    with pytest.raises(FileNotFoundError, match='model.safetensors'):
        assets.download_assets()


def test_repository_environment_is_read_at_call_time(tmp_path, monkeypatch):
    import huggingface_hub
    complete_assets(tmp_path)
    monkeypatch.delenv('CTRLSPEECH_ASSETS', raising=False)
    monkeypatch.setenv('CTRLSPEECH_HF_REPO', 'owner/new-ja')
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    assets.download_assets()
    assert calls[0]['repo_id'] == 'owner/new-ja'


def test_unset_repository_environment_does_not_reuse_import_snapshot(monkeypatch):
    monkeypatch.delenv("CTRLSPEECH_ASSETS", raising=False)
    monkeypatch.delenv("CTRLSPEECH_HF_REPO", raising=False)
    monkeypatch.setattr(assets, "DEFAULT_REPO_ID", "owner/stale")
    with pytest.raises(RuntimeError, match="No public"):
        assets.download_assets()

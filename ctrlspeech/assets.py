"""Resolve the weights and support files a CtrlSpeech model needs.

Everything is fetched from one Hugging Face repo, laid out as::

    <model>/{config.yaml, model.safetensors}
    svae/{metainfo.json, config.json, dac/{ema_state_dict.pth, weights.pth}}
    shared/{vocab.json, campplus.onnx}

Set ``CTRLSPEECH_ASSETS`` to a directory with that same layout to work fully
offline; nothing is downloaded then.
"""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPO_ID = os.environ.get("CTRLSPEECH_HF_REPO")


@dataclass(frozen=True)
class ModelSpec:
    """One published checkpoint."""

    key: str
    folder: str
    params: str
    controllable: bool
    description: str


MODELS = {
    "japanese-lfm2-350m": ModelSpec(
        key="japanese-lfm2-350m",
        folder="japanese-lfm2-350m",
        params="LFM2.5-350M backbone",
        controllable=True,
        description=(
            "Japanese CtrlSpeech with pure LiquidAI/LFM2.5-350M-Base backbone."
        ),
    )
}


@dataclass(frozen=True)
class Assets:
    """Absolute paths to everything ``CtrlSpeech.from_pretrained`` needs."""

    spec: ModelSpec
    root: Path
    config_path: Path
    weights_path: Path
    svae_dir: Path
    vocab_path: Path
    campplus_path: Path

    @property
    def controllable(self):
        return self.spec.controllable


def _layout(root, spec):
    root = Path(root)
    folder = root / spec.folder
    weights = folder / "model.safetensors"
    if not weights.exists():
        # Accept the raw Lightning export too, so a local checkpoint directory
        # works before anything has been converted.
        legacy = folder / "model.ckpt"
        if legacy.exists():
            weights = legacy
    return Assets(
        spec=spec,
        root=root,
        config_path=folder / "config.yaml",
        weights_path=weights,
        svae_dir=root / "svae",
        vocab_path=root / "shared" / "vocab.json",
        campplus_path=root / "shared" / "campplus.onnx",
    )


def _verify(assets):
    missing = [
        str(path)
        for path in (
            assets.config_path,
            assets.weights_path,
            assets.svae_dir / "metainfo.json",
            assets.vocab_path,
            assets.campplus_path,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "CtrlSpeech assets are incomplete under "
            f"{assets.root}:\n  " + "\n  ".join(missing)
        )
    return assets


def download_assets(
    model="japanese-lfm2-350m",
    repo_id=None,
    revision=None,
    cache_dir=None,
):
    """Return local paths for ``model``, downloading from the Hub if needed.

    Only the requested model folder plus the shared files are fetched, so
    picking ``control-150m`` does not pull the 600M weights.
    """
    if model not in MODELS:
        raise ValueError(
            f"Unknown model {model!r}. Available: {', '.join(sorted(MODELS))}"
        )
    spec = MODELS[model]

    local_root = os.environ.get("CTRLSPEECH_ASSETS")
    if local_root:
        return _verify(_layout(local_root, spec))

    resolved_repo_id = repo_id or DEFAULT_REPO_ID
    if not resolved_repo_id:
        raise RuntimeError(
            "No public CtrlSpeech-JA checkpoint is configured yet. "
            "Set CTRLSPEECH_ASSETS to a local trained asset directory or pass "
            "repo_id / CTRLSPEECH_HF_REPO after publishing a checkpoint."
        )

    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id=resolved_repo_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=[f"{spec.folder}/*", "svae/*", "shared/*"],
    )
    return _verify(_layout(root, spec))

"""LFM2.5-350M speech backbone.

CtrlSpeech uses the official Hugging Face LFM2 implementation rather than a
vendored/forked language-model source file. This keeps the 10 LIV convolution
+ 6 GQA layout and lets Transformers own the mixed convolution/KV cache used
during incremental decoding.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from transformers import Lfm2Config, Lfm2Model


DEFAULT_LFM2_MODEL_ID: Final[str] = "LiquidAI/LFM2.5-350M-Base"

LFM2_350M_LAYER_TYPES: Final[tuple[str, ...]] = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)


def lfm2_350m_config() -> Lfm2Config:
    """Return the architecture config for LiquidAI/LFM2.5-350M-Base.

    Keeping a local architecture definition makes inference from a future
    CtrlSpeech-JA checkpoint possible without downloading the original LFM
    config. Pretrained initialization still uses from_pretrained.
    """
    return Lfm2Config(
        vocab_size=65536,
        hidden_size=1024,
        intermediate_size=6656,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=128_000,
        initializer_range=0.02,
        norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=7,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
        },
        conv_bias=False,
        conv_L_cache=3,
        block_multiple_of=256,
        block_ffn_dim_multiplier=1.0,
        block_auto_adjust_ff_dim=True,
        layer_types=list(LFM2_350M_LAYER_TYPES),
    )


def _validate_lfm2_350m(config: Lfm2Config) -> None:
    """Reject silently incompatible LFM variants."""
    expected = {
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "conv_L_cache": 3,
    }
    mismatches = [
        f"{key}={getattr(config, key, None)!r} (expected {value!r})"
        for key, value in expected.items()
        if getattr(config, key, None) != value
    ]
    if tuple(config.layer_types) != LFM2_350M_LAYER_TYPES:
        mismatches.append("layer_types do not match LFM2.5-350M")

    if mismatches:
        raise ValueError(
            "CtrlSpeech-JA is pinned to the pure LFM2.5-350M architecture: "
            + "; ".join(mismatches)
        )


class LFM2SpeechBackbone(nn.Module):
    """LFM2.5-350M adapted to continuous speech embeddings.

    The LFM body remains unmodified. CtrlSpeech-specific information is
    injected before the model through separate phone and modality embeddings.
    This preserves LFM's native token embedding table for future raw-text
    conditioning instead of repurposing it as a phoneme vocabulary.
    """

    PHONE_MODALITY: Final[int] = 0
    AUDIO_MODALITY: Final[int] = 1
    NATIVE_TEXT_MODALITY: Final[int] = 2
    NUM_MODALITIES: Final[int] = 3

    def __init__(
        self,
        *,
        phone_vocab_size: int,
        model_id: str = DEFAULT_LFM2_MODEL_ID,
        load_pretrained_weights: bool = True,
        config: Lfm2Config | None = None,
        weighted_layers: bool = False,
        validate_architecture: bool = True,
    ):
        super().__init__()
        if phone_vocab_size < 2:
            raise ValueError(
                "phone_vocab_size must include at least padding and unknown tokens"
            )
        if weighted_layers:
            raise ValueError(
                "weighted_layers is not supported for the LFM2 backbone; "
                "CtrlSpeech-JA uses the final LFM2 hidden state."
            )
        if config is not None and load_pretrained_weights:
            raise ValueError(
                "pass either config or load_pretrained_weights=True, not both"
            )

        if load_pretrained_weights:
            # Keep the whole DiTAR graph in float32 by default. Training can
            # still use autocast/bfloat16 globally, but mixed parameter dtypes
            # between a bf16 base checkpoint and new speech adapters would make
            # inputs_embeds integration fragile.
            self.model = Lfm2Model.from_pretrained(
                model_id,
                dtype=torch.float32,
            )
        else:
            self.model = Lfm2Model(config or lfm2_350m_config())

        if validate_architecture:
            _validate_lfm2_350m(self.model.config)
        self.model_id = model_id
        self.hidden_size = int(self.model.config.hidden_size)

        self.phone_embedding = nn.Embedding(
            phone_vocab_size,
            self.hidden_size,
            padding_idx=0,
        )
        self.modality_embedding = nn.Embedding(
            self.NUM_MODALITIES,
            self.hidden_size,
        )
        self._init_adapter_weights()

    def _init_adapter_weights(self) -> None:
        std = float(getattr(self.model.config, "initializer_range", 0.02))
        nn.init.normal_(self.phone_embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.phone_embedding.weight[0].zero_()

    def embed_phone_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the canonical CtrlSpeech-JA phone vocabulary."""
        return self.phone_embedding(input_ids)

    def embed_native_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed native LFM tokenizer IDs for optional raw-text conditioning."""
        return self.model.embed_tokens(input_ids)

    def set_pretrained_body_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze only the pretrained LFM body."""
        for parameter in self.model.parameters():
            parameter.requires_grad = bool(trainable)

    def _add_modality(
        self,
        input_embeds: torch.Tensor,
        modality_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if modality_type_ids is None:
            return input_embeds
        if modality_type_ids.shape != input_embeds.shape[:2]:
            raise ValueError(
                "modality_type_ids must have shape [batch, sequence], got "
                f"{tuple(modality_type_ids.shape)} for embeddings "
                f"{tuple(input_embeds.shape)}"
            )
        if modality_type_ids.numel():
            min_id = int(modality_type_ids.min().item())
            max_id = int(modality_type_ids.max().item())
            if min_id < 0 or max_id >= self.NUM_MODALITIES:
                raise ValueError(
                    f"modality ids must be in [0, {self.NUM_MODALITIES - 1}]"
                )
        return input_embeds + self.modality_embedding(modality_type_ids)

    def forward(
        self,
        input_embeds: torch.Tensor,
        modality_type_ids: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._add_modality(input_embeds, modality_type_ids)
        outputs = self.model(
            inputs_embeds=x,
            attention_mask=padding_mask,
            use_cache=False,
        )
        return outputs.last_hidden_state

    def inference(
        self,
        input_embeds: torch.Tensor,
        modality_type_ids: torch.Tensor | None,
        padding_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = True,
    ):
        """Incrementally decode embeddings with the native LFM2 cache.

        Cached decoding receives only new acoustic token(s), while attention
        masks and callers may still carry the complete prefix history. Slice
        the modality history to the unprocessed tail before adding embeddings.
        """
        if modality_type_ids is not None:
            current_length = input_embeds.shape[1]
            if modality_type_ids.shape[0] != input_embeds.shape[0]:
                raise ValueError(
                    "modality_type_ids batch size does not match input"
                )
            if modality_type_ids.shape[1] < current_length:
                raise ValueError(
                    "modality_type_ids is shorter than the incremental input: "
                    f"{modality_type_ids.shape[1]} < {current_length}"
                )
            if modality_type_ids.shape[1] != current_length:
                modality_type_ids = modality_type_ids[:, -current_length:]

        x = self._add_modality(input_embeds, modality_type_ids)
        outputs = self.model(
            inputs_embeds=x,
            attention_mask=padding_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return outputs.last_hidden_state, outputs.past_key_values

import torch
from transformers import Lfm2Config

from ctrlspeech.models.backbone import LFM2SpeechBackbone, lfm2_350m_config


def _tiny_lfm2_config():
    return Lfm2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        conv_bias=False,
        conv_L_cache=3,
        block_multiple_of=16,
        block_auto_adjust_ff_dim=False,
        layer_types=["conv", "full_attention"],
    )


def test_lfm2_350m_local_config_matches_target_architecture():
    config = lfm2_350m_config()
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 16
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.layer_types.count("conv") == 10
    assert config.layer_types.count("full_attention") == 6


def test_lfm2_cached_decode_matches_full_decode():
    torch.manual_seed(0)
    backbone = LFM2SpeechBackbone(
        phone_vocab_size=16,
        load_pretrained_weights=False,
        config=_tiny_lfm2_config(),
        validate_architecture=False,
    ).eval()

    phone_ids = torch.tensor([[2, 3, 4, 5, 6]])
    input_embeds = backbone.embed_phone_tokens(phone_ids)
    modality = torch.full(
        phone_ids.shape,
        backbone.PHONE_MODALITY,
        dtype=torch.long,
    )
    full_mask = torch.ones_like(phone_ids, dtype=torch.bool)

    with torch.no_grad():
        full = backbone(
            input_embeds=input_embeds,
            modality_type_ids=modality,
            padding_mask=full_mask,
        )

        _, cache = backbone.inference(
            input_embeds=input_embeds[:, :-1],
            modality_type_ids=modality[:, :-1],
            padding_mask=full_mask[:, :-1],
            use_cache=True,
        )
        cached_tail, _ = backbone.inference(
            input_embeds=input_embeds[:, -1:],
            # Real CtrlSpeech sampling retains full modality history while only
            # passing the new acoustic token after the first cached step.
            modality_type_ids=modality,
            padding_mask=full_mask,
            past_key_values=cache,
            use_cache=True,
        )

    torch.testing.assert_close(
        cached_tail[:, -1],
        full[:, -1],
        rtol=1e-4,
        atol=1e-5,
    )


def test_phone_embedding_is_separate_from_native_lfm_embedding():
    backbone = LFM2SpeechBackbone(
        phone_vocab_size=16,
        load_pretrained_weights=False,
        config=_tiny_lfm2_config(),
        validate_architecture=False,
    )
    ids = torch.tensor([[2, 3]])
    phone = backbone.embed_phone_tokens(ids)
    native = backbone.embed_native_tokens(ids)
    assert phone.shape == native.shape
    assert phone.data_ptr() != native.data_ptr()


def test_compose_text_prefix_preserves_native_lfm_and_phone_modalities():
    backbone = LFM2SpeechBackbone(
        phone_vocab_size=16,
        load_pretrained_weights=False,
        config=_tiny_lfm2_config(),
        validate_architecture=False,
    ).eval()

    phone_ids = torch.tensor([[2, 3, 0]])
    phone_embeds = backbone.embed_phone_tokens(phone_ids)
    phone_mask = phone_ids.ne(0)
    native_ids = torch.tensor([[1, 7]])

    embeds, mask, modality = backbone.compose_text_prefix(
        phone_embeds=phone_embeds,
        phone_mask=phone_mask,
        native_text_ids=native_ids,
    )

    assert embeds.shape == (1, 5, 32)
    assert mask.tolist() == [[True, True, True, True, False]]
    assert modality.tolist() == [[
        backbone.NATIVE_TEXT_MODALITY,
        backbone.NATIVE_TEXT_MODALITY,
        backbone.PHONE_MODALITY,
        backbone.PHONE_MODALITY,
        backbone.PHONE_MODALITY,
    ]]
    torch.testing.assert_close(
        embeds[:, :2],
        backbone.embed_native_tokens(native_ids),
    )

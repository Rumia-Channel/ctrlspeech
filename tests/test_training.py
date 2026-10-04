"""Exercise real speech modules at tiny scale without downloading weights."""

import copy
import json

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers import Lfm2Config

from ctrlspeech.frontend import JAPANESE_PHONE_TO_ID
from ctrlspeech.models.backbone import LFM2SpeechBackbone
from ctrlspeech.models.ditar import DiTar
from ctrlspeech.models.modules import AggregationEncoder
from ctrlspeech.models.training_utils import dropout_condition, flow_matching_path, left_pad_text_prefix, masked_mean
from ctrlspeech.training import CachedSpeechDataset, LengthBucketBatchSampler, collate_speech_examples, validate_example
from ctrlspeech.training.cli import main
from ctrlspeech.training.engine import Trainer, TrainingConfig, TrainingStage, seed_training


@pytest.fixture
def tiny_model(monkeypatch):
    from ctrlspeech.models import ditar
    config = OmegaConf.load("configs/japanese-lfm2-350m.yaml").model
    config.dim = 32
    config.mlp_hidden_dim = 16
    config.aggregation_encoder.update({"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2})
    config.loc_decoder.model.update({"dim": 32, "depth": 1, "heads": 2, "dim_head": 16})
    backbone_config = Lfm2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        layer_types=["conv", "full_attention"], conv_L_cache=3,
        block_multiple_of=16, block_auto_adjust_ff_dim=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pad_token_id=0,
    )
    def backbone(**kwargs):
        return LFM2SpeechBackbone(
            phone_vocab_size=52, config=backbone_config,
            load_pretrained_weights=False, validate_architecture=False,
        )
    monkeypatch.setattr(ditar, "LFM2SpeechBackbone", backbone)
    monkeypatch.setattr(ditar, "load_state", lambda **kwargs: nn.Sequential(nn.Linear(1, 1), nn.Dropout(0.5)))
    seed_training(7)
    return DiTar(config)


def example(frames=9, *, native=False, controls=False):
    phones = torch.tensor([JAPANESE_PHONE_TO_ID[phone] for phone in ("a", "|", "i", "u")])
    result = {
        "vae_features": torch.randn(frames, 64),
        "speaker_embs": torch.randn(192),
        "text_inputs": phones,
        "prompt_frames": 4,
        "linguistic_features": {
            "accent_pitch": torch.tensor([1, 0, 2, 1]),
            "phrase_boundary": torch.tensor([3, 0, 1, 2]),
            "accent_nucleus": torch.tensor([1., 0., 1., 1.]),
            "phrase_mora_count": torch.tensor([1., 0., 2., 2.]),
            "valid": torch.tensor([True, False, True, True]),
        },
    }
    if native:
        result["native_text_inputs"] = torch.tensor([1, 10, 7])
    if controls:
        result.update({"pitch": torch.ones(30, dtype=torch.long), "loudness": torch.ones(30, dtype=torch.long),
                       "duration_segments": torch.tensor([[0, 10], [0, 0], [10, 20], [20, 30]])})
    return result


def training_config(*, accumulation=2, epochs=2):
    return TrainingConfig(
        stages=(TrainingStage(epochs, 5, False, adapter_lr=1e-3), TrainingStage(1, 15, True)),
        precision="none", batch_size=2, accumulation_steps=accumulation,
        warmup_steps=0, max_grad_norm=100, loss_weights={"diff_loss": 1.0, "stop_loss": 0.1},
        gradient_checkpointing=False,
    )


def test_flow_velocity_matches_interpolant_derivative():
    clean = torch.randn(2, 5, 3, dtype=torch.float64)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.2, 0.8], dtype=torch.float64)[:, None, None]
    for curved in (False, True):
        _, velocity = flow_matching_path(clean, noise, time, curved=curved)
        left, _ = flow_matching_path(clean, noise, time - 1e-6, curved=curved)
        right, _ = flow_matching_path(clean, noise, time + 1e-6, curved=curved)
        torch.testing.assert_close(velocity, (right - left) / 2e-6)


def test_masked_reduction_excludes_padding_and_uses_float32():
    values = torch.tensor([[[2., 4.], [float("nan"), float("inf")]]], dtype=torch.bfloat16)
    mean = masked_mean(values, torch.tensor([[True, False]]))
    assert mean.dtype == torch.float32
    assert mean == 3
    assert masked_mean(values, torch.zeros(1, 2, dtype=torch.bool)) == 0


def test_condition_dropout_is_per_row_and_disabled_for_validation():
    torch.manual_seed(2)
    values = torch.ones(32, 3, 2)
    dropped = dropout_condition(values, 0.5, training=True)
    assert (dropped[:, 0, 0] == 0).any() and (dropped[:, 0, 0] == 1).any()
    assert torch.equal(dropped[:, 0], dropped[:, 1])
    assert dropout_condition(values, 1, training=False) is values


def test_prefix_compaction_preserves_gradients_and_last_text_position():
    embeds = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1).requires_grad_()
    mask = torch.tensor([[True, False, False, True, True, False], [True] * 6])
    modality = torch.tensor([[2, 2, 2, 0, 0, 0], [2, 2, 0, 0, 0, 0]])
    packed, valid, types = left_pad_text_prefix(embeds, mask, modality)
    assert valid[0].tolist() == [False, False, False, True, True, True]
    assert packed[0, -3:, 0].tolist() == [0., 3., 4.]
    assert types[0, -3:].tolist() == [2, 0, 0]
    packed.sum().backward()
    assert torch.equal(embeds.grad.squeeze(-1), mask.float())


@pytest.mark.parametrize("pool_type", ["avg", "cls"])
def test_empty_acoustic_patches_are_finite_zero_with_finite_gradients(pool_type):
    encoder = AggregationEncoder(16, 32, 1, 2, 4, pool_type=pool_type)
    values = torch.randn(2, 8, 16, requires_grad=True)
    mask = torch.tensor([[True] * 5 + [False] * 3, [True] * 4 + [False] * 4])
    output, valid = encoder(values, mask)
    assert valid.tolist() == [[True, True], [True, False]]
    assert torch.isfinite(output).all()
    assert torch.equal(output[1, 1], torch.zeros(16))
    output.square().sum().backward()
    assert torch.isfinite(values.grad).all()


def test_ditar_initialization_keeps_zero_decoder_and_frozen_encoder(tiny_model):
    assert torch.count_nonzero(tiny_model.LocDiT.proj_out.weight) == 0
    assert torch.count_nonzero(tiny_model.LocDiT.transformer_blocks[0].attn_norm.linear.weight) == 0
    tiny_model.train()
    assert not tiny_model.generator.training
    assert all(not parameter.requires_grad for parameter in tiny_model.generator.parameters())


@pytest.mark.parametrize("random_time,shared", [(False, False), (True, False), (True, True)])
def test_real_ditar_variable_length_training_backward(tiny_model, random_time, shared):
    tiny_model.LocDiT_config.random_time = random_time
    tiny_model.LocDiT_config.time_schedule = True
    tiny_model.use_seperate_linear = not shared
    batch = collate_speech_examples([example(9, native=True, controls=True), example(6, native=True, controls=True)])
    losses = tiny_model(**batch)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    sum(value for name, value in losses.items() if name != "stop_acc").backward()
    assert tiny_model.LocDiT.proj_out.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in tiny_model.parameters())
    assert all(parameter.grad is None for parameter in tiny_model.generator.parameters())


def test_short_text_predicts_first_patch_from_last_valid_phone(tiny_model):
    tiny_model.eval()
    long = example(9)
    short = example(6)
    short["text_inputs"] = short["text_inputs"][:-1]
    short["linguistic_features"] = {name: values[:-1] for name, values in short["linguistic_features"].items()}
    batched = collate_speech_examples([short, long])
    single = collate_speech_examples([short])
    def first_hidden(batch):
        valid = torch.arange(batch["vae_features"].shape[1])[None] < batch["vae_lengths"][:, None]
        result = tiny_model.get_ar_input(batch["vae_features"], valid, batch["speaker_embs"], batch["text_inputs"],
                                         batch["text_masks"], linguistic_features=batch["linguistic_features"])
        inputs, modality, _, _, _, text_masks, audio_masks = result
        hidden = tiny_model.causalAR(inputs, modality, torch.cat([text_masks, audio_masks], dim=1))
        return hidden[0, text_masks.shape[1] - 1]
    torch.testing.assert_close(first_hidden(batched), first_hidden(single), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("strategy", ["only_drop_ctx", "drop_ctx_or_his", "drop_ctx_and_his"])
def test_sampling_supports_all_cfg_modes_and_short_prompt(tiny_model, monkeypatch, strategy):
    from ctrlspeech.models import ditar
    monkeypatch.setattr(ditar, "process_online", lambda *args: torch.randn(1, 64, 2))
    tiny_model.eval()
    tiny_model.drop_condition = strategy
    result = tiny_model.sample(torch.zeros(1, 800), torch.zeros(1, 192),
                               torch.tensor([[2, 3, 4]]), torch.ones(1, 3, dtype=torch.bool),
                               max_seq_length=2, steps=1, use_cache=True)
    assert result.shape == (1, 8, 64)
    assert torch.isfinite(result).all()


def test_cache_validation_and_target_only_collation():
    first, second = example(9, native=True, controls=True), example(6, native=True, controls=True)
    validate_example(first)
    batch = collate_speech_examples([first, second])
    assert batch["audio_loss_mask"].sum(dim=1).tolist() == [5, 2]
    assert not batch["audio_loss_mask"][:, :4].any()
    first["prompt_frames"] = 3
    with pytest.raises(ValueError, match="patch boundary"):
        validate_example(first)
    first["prompt_frames"] = 4
    first["linguistic_features"]["valid"][1] = True
    with pytest.raises(ValueError, match="separator"):
        validate_example(first)


def test_length_buckets_are_reproducible_and_do_not_truncate():
    lengths = [20, 300, 150, 70, 500, 100]
    sampler = LengthBucketBatchSampler(lengths, 2, max_frames=200, seed=3)
    batches = list(sampler)
    assert batches == list(sampler)
    assert sorted(index for batch in batches for index in batch) == [0, 2, 3, 5]
    assert len(sampler) == 2
    with pytest.raises(ValueError, match="no examples"):
        LengthBucketBatchSampler(lengths, 2, max_frames=1)


class TinyObjective(nn.Module):
    """Known scalar objective for checking accumulation and optimizer state."""
    def __init__(self):
        super().__init__()
        self.causalAR = nn.Module()
        self.causalAR.model = nn.Linear(1, 1, bias=False)
        self.causalAR.set_pretrained_body_trainable = lambda trainable: self.causalAR.model.requires_grad_(trainable)
        self.generator = nn.Linear(1, 1)
        self.adapter = nn.Linear(1, 1, bias=False)

    def forward(self, text_inputs, **kwargs):
        prediction = self.adapter(text_inputs.float()) + self.causalAR.model(text_inputs.float())
        return {"diff_loss": prediction.square().mean(), "stop_loss": prediction.sum() * 0}


def test_accumulation_remainder_matches_individual_optimizer_updates():
    seed_training(1)
    model = TinyObjective()
    other = copy.deepcopy(model)
    # deepcopy of lambda closures would refer to the first model; bind explicitly.
    other.causalAR.set_pretrained_body_trainable = lambda enabled: other.causalAR.model.requires_grad_(enabled)
    accumulated = Trainer(model, training_config(accumulation=2))
    reference = Trainer(other, training_config(accumulation=1))
    batches = [{"text_inputs": torch.tensor([[value]])} for value in (1, 3, 5)]
    combined = [{"text_inputs": torch.tensor([[1], [3]])}, batches[-1]]
    accumulated.train_epoch(batches, accumulated.config.stages[0], total_stage_steps=2)
    reference.train_epoch(combined, reference.config.stages[0], total_stage_steps=2)
    assert accumulated.global_step == reference.global_step == 2
    torch.testing.assert_close(model.adapter.weight, other.adapter.weight)
    assert model.causalAR.model.weight.grad is None


def test_nonfinite_loss_does_not_update_model():
    model = TinyObjective()
    trainer = Trainer(model, training_config())
    before = model.adapter.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="nonfinite loss"):
        trainer.train_epoch([{"text_inputs": torch.tensor([[float("nan")]])}],
                            trainer.config.stages[0], total_stage_steps=1)
    torch.testing.assert_close(model.adapter.weight, before)
    assert model.adapter.weight.grad is None
    assert trainer.global_step == 0
    with pytest.raises(ValueError, match="training loader is empty"):
        trainer.train_epoch([], trainer.config.stages[0], total_stage_steps=1)


def test_validation_preserves_rng_and_resume_matches_next_update(tiny_model, tmp_path):
    config = training_config()
    trainer = Trainer(tiny_model, config)
    loader = [collate_speech_examples([example(9), example(6)])]
    stage = config.stages[0]
    trainer.train_epoch(loader, stage, total_stage_steps=4)
    trainer.completed_epochs = 1
    rng = torch.get_rng_state().clone()
    first = trainer.evaluate(loader)
    second = trainer.evaluate(loader)
    assert first == second
    assert tiny_model.training and not tiny_model.generator.training
    assert torch.equal(rng, torch.get_rng_state())
    path = tmp_path / "resume.ckpt"
    trainer.save(path, model_config={"tiny": True}, data_signature={"train": "a"})
    clone = copy.deepcopy(tiny_model)
    resumed = Trainer(clone, config)
    trainer.train_epoch(loader, stage, total_stage_steps=4)
    expected = {name: value.clone() for name, value in tiny_model.state_dict().items()}
    resumed.restore(path, model_config={"tiny": True}, data_signature={"train": "a"})
    resumed.train_epoch(loader, stage, total_stage_steps=4)
    assert resumed.global_step == trainer.global_step
    for name, value in clone.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="original data"):
        resumed.restore(path, model_config={"tiny": True}, data_signature={"train": "changed"})


def test_checkpointed_bf16_training_is_finite_and_updates(tiny_model):
    config = TrainingConfig(stages=(TrainingStage(1, 5, False),), precision="bf16", warmup_steps=0)
    trainer = Trainer(tiny_model, config)
    loader = [collate_speech_examples([example(9), example(6)])]
    result = trainer.train_epoch(loader, config.stages[0], total_stage_steps=1)
    assert result["loss"] > 0
    assert tiny_model.causalAR.model.is_gradient_checkpointing
    assert tiny_model.LocDiT.checkpoint_activations
    assert trainer.global_step == 1


def write_dataset(tmp_path, speaker):
    path = tmp_path / f"{speaker}.pt"
    torch.save(example(), path)
    manifest = tmp_path / f"{speaker}.jsonl"
    manifest.write_text(json.dumps({"path": path.name, "frames": 9, "speaker_id": speaker}) + "\n", encoding="utf-8")
    return manifest


def test_fit_runs_stages_and_writes_resumable_inference_checkpoint(tiny_model, tmp_path):
    config = training_config(epochs=1)
    trainer = Trainer(tiny_model, config)
    train = CachedSpeechDataset(write_dataset(tmp_path, "train-speaker"))
    validation = CachedSpeechDataset(write_dataset(tmp_path, "unseen-speaker"))
    output = tmp_path / "output"
    history = list(trainer.fit(train, validation, output, model_config={}, data_signature={}))
    assert [item["max_seconds"] for item in history] == [5, 15]
    assert trainer.completed_epochs == 2
    assert all(parameter.requires_grad for parameter in tiny_model.causalAR.model.parameters())
    assert (output / "best.ckpt").is_file()
    checkpoint = torch.load(output / "latest.ckpt", weights_only=True)
    assert checkpoint["completed_epochs"] == 2
    assert "generator.0.weight" in checkpoint["state_dict"]
    assert list(trainer.fit(train, validation, output, model_config={}, data_signature={})) == []


def test_validate_only_audits_caches_without_model_download(tmp_path, capsys):
    train = write_dataset(tmp_path, "train-speaker")
    validation = write_dataset(tmp_path, "unseen-speaker")
    main(["--train-manifest", str(train), "--validation-manifest", str(validation),
          "--output-dir", str(tmp_path / "output"), "--svae-dir", str(tmp_path / "svae"), "--validate-only"])
    assert json.loads(capsys.readouterr().out)["status"] == "valid"
    assert not (tmp_path / "output").exists()
    with pytest.raises(ValueError, match="speaker IDs overlap"):
        main(["--train-manifest", str(train), "--validation-manifest", str(train),
              "--output-dir", str(tmp_path / "output"), "--svae-dir", str(tmp_path), "--validate-only"])


@pytest.mark.parametrize("field,value,message", [
    ("vae_features", [], "vae_features must be a tensor"),
    ("speaker_embs", [0] * 192, "speaker_embs must be a tensor"),
    ("text_inputs", [2, 3, 4], "text_inputs must be a tensor"),
    ("linguistic_features", None, "linguistic_features must be a dictionary"),
    ("native_text_inputs", None, "native_text_inputs must be a tensor"),
    ("prompt_frames", True, "prompt_frames"),
])
def test_cache_schema_rejects_wrong_types_with_actionable_errors(field, value, message):
    cached = example()
    cached[field] = value
    with pytest.raises(ValueError, match=message):
        validate_example(cached)


def test_cache_root_and_linguistic_fields_require_tensors():
    with pytest.raises(ValueError, match="feature cache must be a dictionary"):
        validate_example([])
    cached = example()
    cached["linguistic_features"]["accent_nucleus"] = [1., 0., 1., 1.]
    with pytest.raises(ValueError, match="accent_nucleus must be a tensor"):
        validate_example(cached)
    cached["linguistic_features"]["accent_nucleus"] = torch.ones(4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="nonnegative real numeric"):
        validate_example(cached)


def test_cache_controls_require_one_ordered_shared_timeline():
    cached = example(controls=True)
    cached["loudness"] = cached["loudness"][:-1]
    with pytest.raises(ValueError, match="same control timeline length"):
        validate_example(cached)
    cached = example(controls=True)
    cached["duration_segments"][2] = torch.tensor([5, 20])
    with pytest.raises(ValueError, match="ordered and nonoverlapping"):
        validate_example(cached)
    # Empty segments are allowed anywhere and carry no timeline ordering.
    cached["duration_segments"][2] = torch.tensor([1, 1])
    validate_example(cached)


def test_collation_does_not_silently_drop_partial_controls():
    cached = example(controls=True)
    del cached["duration_segments"]
    with pytest.raises(ValueError, match="provided together"):
        collate_speech_examples([cached])


def test_collation_normalizes_cached_float_dtypes_for_real_model(tiny_model):
    half, double = example(6), example(9)
    for cached, dtype in ((half, torch.float16), (double, torch.float64)):
        for name in ("vae_features", "speaker_embs"):
            cached[name] = cached[name].to(dtype)
        validate_example(cached)
    batch = collate_speech_examples([half, double])
    assert batch["vae_features"].dtype == batch["speaker_embs"].dtype == torch.float32
    torch.testing.assert_close(batch["vae_features"][1], double["vae_features"].float())
    losses = tiny_model(**batch)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["diff_loss"].backward()
    assert tiny_model.LocDiT.proj_out.weight.grad is not None


@pytest.mark.parametrize("entry,message", [
    ("[]", "manifest entry must be an object"),
    ("{", "invalid JSON"),
    ('{"frames": true, "speaker_id": "speaker", "path": "cached.pt"}', "frames must be a positive integer"),
    ('{"frames": 9, "speaker_id": "speaker"}', "path must be a nonempty string"),
    ('{"frames": 9, "speaker_id": "speaker", "path": 42}', "path must be a nonempty string"),
    ('{"frames": 9, "speaker_id": "  ", "path": "cached.pt"}', "speaker_id is required"),
])
def test_manifest_schema_errors_identify_file_and_line(tmp_path, entry, message):
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text("\n" + entry + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message) as error:
        CachedSpeechDataset(manifest)
    assert f"{manifest}:2:" in str(error.value)


def test_dataset_schema_errors_identify_cache_path(tmp_path):
    manifest = write_dataset(tmp_path, "speaker")
    cache_path = tmp_path / "speaker.pt"
    torch.save({"vae_features": torch.zeros(9, 64)}, cache_path)
    with pytest.raises(ValueError, match="feature cache is missing") as error:
        CachedSpeechDataset(manifest)[0]
    assert str(cache_path) in str(error.value)


@pytest.mark.parametrize("setting,value", [
    ("batch_size", True), ("batch_size", 1.5), ("accumulation_steps", 1.5),
    ("num_workers", 0.5), ("warmup_steps", 0.5), ("seed", 4.5),
    ("validation_seed", False), ("gradient_checkpointing", "false"),
])
def test_training_config_rejects_noninteger_counts_and_ambiguous_flags(setting, value):
    with pytest.raises(ValueError, match=setting):
        TrainingConfig(stages=(TrainingStage(1, 5, False),), **{setting: value})


@pytest.mark.parametrize("epochs,train_backbone,message", [
    (True, False, "epochs"), (1.5, False, "epochs"), (1, "false", "train_backbone"),
])
def test_stage_rejects_noninteger_epochs_and_ambiguous_freezing(epochs, train_backbone, message):
    with pytest.raises(ValueError, match=message):
        TrainingStage(epochs, 5, train_backbone)


def test_nonfinite_gradient_clears_accumulation_without_updating():
    model = TinyObjective()
    trainer = Trainer(model, training_config())
    before = model.adapter.weight.detach().clone()
    hook = model.adapter.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            trainer.train_epoch([{"text_inputs": torch.ones(1, 1)}], trainer.config.stages[0], total_stage_steps=1)
    finally:
        hook.remove()
    torch.testing.assert_close(model.adapter.weight, before)
    assert model.adapter.weight.grad is None
    assert trainer.global_step == trainer.stage_step == 0


def test_validation_preserves_python_rng_even_after_an_error(monkeypatch):
    import random

    model = TinyObjective()
    original_forward = model.forward

    def randomized_forward(**batch):
        result = original_forward(**batch)
        result["diff_loss"] = result["diff_loss"] * random.random()
        return result

    monkeypatch.setattr(model, "forward", randomized_forward)
    trainer = Trainer(model, training_config())
    loader = [{"text_inputs": torch.ones(1, 1)}]
    before = random.getstate()
    assert trainer.evaluate(loader) == trainer.evaluate(loader)
    assert random.getstate() == before
    with pytest.raises(FloatingPointError, match="nonfinite validation"):
        trainer.evaluate([{"text_inputs": torch.full((1, 1), float("nan"))}])
    assert random.getstate() == before
    assert model.training


@pytest.mark.parametrize("completed_epochs", [1, 2])
def test_fit_resume_matches_frozen_to_unfrozen_curriculum(tiny_model, tmp_path, completed_epochs):
    config = TrainingConfig(
        stages=(TrainingStage(1, 5, False), TrainingStage(2, 15, True)),
        precision="none", batch_size=2, accumulation_steps=2, warmup_steps=0,
        gradient_checkpointing=False,
    )
    initial_model = copy.deepcopy(tiny_model)
    train = CachedSpeechDataset(write_dataset(tmp_path, "train-speaker"))
    validation = CachedSpeechDataset(write_dataset(tmp_path, "unseen-speaker"))
    checkpoint_args = {"model_config": {"tiny": True}, "data_signature": {"train": "fixed"}}
    seed_training(17)
    uninterrupted = Trainer(tiny_model, config)
    expected_history = list(uninterrupted.fit(train, validation, tmp_path / "full", **checkpoint_args))
    expected_rng = torch.get_rng_state().clone()

    seed_training(17)
    interrupted = Trainer(copy.deepcopy(initial_model), config)
    iterator = interrupted.fit(train, validation, tmp_path / "partial", **checkpoint_args)
    for _ in range(completed_epochs):
        next(iterator)
    iterator.close()
    resumed = Trainer(copy.deepcopy(initial_model), config)
    resumed.restore(tmp_path / "partial/latest.ckpt", **checkpoint_args)
    remaining = list(resumed.fit(train, validation, tmp_path / "resumed", **checkpoint_args))
    assert remaining == expected_history[completed_epochs:]
    assert resumed.completed_epochs == uninterrupted.completed_epochs == 3
    assert resumed.global_step == uninterrupted.global_step == 3
    assert resumed.stage_step == uninterrupted.stage_step == 2
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(parameter.requires_grad for parameter in resumed.model.causalAR.model.parameters())
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, uninterrupted.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["pitch", "loudness", "duration_segments"])
def test_real_model_rejects_partial_acoustic_controls(tiny_model, missing):
    batch = collate_speech_examples([example(controls=True)])
    del batch[missing]
    with pytest.raises(ValueError, match="provided together"):
        tiny_model(**batch)


@pytest.mark.parametrize("setting,value,message", [
    ("steps", 0, "positive integers"), ("steps", True, "positive integers"),
    ("steps", 1.5, "positive integers"), ("max_seq_length", 0, "positive integers"),
    ("max_seq_length", False, "positive integers"), ("max_seq_length", 2.5, "positive integers"),
    ("cfg_strength", float("nan"), "finite and non-negative"),
    ("cfg_strength", float("inf"), "finite and non-negative"),
    ("cfg_strength", -1, "finite and non-negative"),
])
def test_sampling_rejects_invalid_arguments_before_encoding(tiny_model, monkeypatch, setting, value, message):
    from ctrlspeech.models import ditar

    def unexpected_encoding(*args, **kwargs):
        pytest.fail("invalid sampling arguments must be rejected before encoding")

    monkeypatch.setattr(ditar, "process_online", unexpected_encoding)
    tiny_model.eval()
    with pytest.raises(ValueError, match=message):
        tiny_model.sample(torch.zeros(1, 800), torch.zeros(1, 192),
                          torch.tensor([[2, 3, 4]]), torch.ones(1, 3, dtype=torch.bool),
                          **{setting: value})

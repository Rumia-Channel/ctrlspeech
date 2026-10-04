import math

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from einops import rearrange
from torchdiffeq import odeint

from .embeds import DurationConditioner, JapaneseLinguisticConditioner, TimestepEmbedding, VAEProjector
from .backbone import DEFAULT_LFM2_MODEL_ID, LFM2SpeechBackbone
from .modules import AggregationEncoder, MLPStopPredictor
from .backbone.dit import DiT
from .vae.online_feature import load_state, process_online
from .training_utils import (
    dropout_condition, flow_matching_path, left_pad_text_prefix, masked_mean,
)


@dataclass
class LocDiTConfig:
    name: Literal["DiT"] = "DiT"

    # Model
    model: dict = field(default_factory=dict)
    history_vae_window_size: int = 4

    # Training
    random_time: bool = False
    time_schedule: bool = False
    drop_cond_prob: float = 0.1
    speaker_drop_prob: float = 0.5
    emotion_drop_prob: float = 0.1
    # Inference
    odeint_kwargs: dict = field(default_factory=lambda: {
        # atol = 1e-5,
        # rtol = 1e-5,
        "method": "euler",
    })

    def __post_init__(self):
        if self.name != "DiT":
            raise ValueError("only the DiT local decoder is implemented")
        if self.history_vae_window_size < 1:
            raise ValueError("history_vae_window_size must be positive")
        for name in ("drop_cond_prob", "speaker_drop_prob", "emotion_drop_prob"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


class DiTar(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_channels = config.audio_channels
        self.dim = config.dim
        self.patch_size = config.patch_size
        if self.patch_size < 1:
            raise ValueError("patch_size must be positive")
        self.text_vocab_size = config.text_vocab_size
        self.mlp_hidden_dim = config.mlp_hidden_dim

        backbone_model_id = getattr(
            config.backbone, "model_id", DEFAULT_LFM2_MODEL_ID
        )
        phone_vocab_size = int(
            getattr(config.backbone, "phone_vocab_size", self.text_vocab_size)
        )
        self.causalAR = LFM2SpeechBackbone(
            phone_vocab_size=phone_vocab_size,
            model_id=backbone_model_id,
            load_pretrained_weights=bool(
                getattr(config.backbone, "load_pretrained_weights", True)
            ),
            weighted_layers=bool(
                getattr(config.backbone, "weighted_layers", False)
            ),
        )
        if self.causalAR.hidden_size != self.dim:
            raise ValueError(
                "CtrlSpeech model dim must match LFM2 hidden size: "
                f"{self.dim} != {self.causalAR.hidden_size}"
            )
        if bool(getattr(config.backbone, "freeze_pretrained_body", False)):
            self.causalAR.set_pretrained_body_trainable(False)

        self.vae_projector = VAEProjector(
            in_dim=self.audio_channels, 
            d_model=self.dim,
            hidden=self.mlp_hidden_dim,
            dropout=0.1,
        )
        self.use_seperate_linear = config.use_seperate_linear
        if self.use_seperate_linear:
            self.vae_projector_for_dit = VAEProjector(
                in_dim=self.audio_channels,
                d_model=self.dim,
                hidden=self.mlp_hidden_dim,
                dropout=0.1,
            )

        self.time_embedding = TimestepEmbedding(dim=self.dim)
        self.cond_projection = nn.Sequential(
            nn.Linear(self.dim + 192, self.dim * 2),
            nn.SiLU(),
            nn.Linear(self.dim * 2, self.dim)
        )
        self.aggregation_encoder = AggregationEncoder(**config.aggregation_encoder)
        
        self.LocDiT_config = LocDiTConfig(**config.loc_decoder)
        self.LocDiT = DiT(**self.LocDiT_config.model)

        self.stop_predictor_type = config.stop_predictor_type

        self.stop_projection = MLPStopPredictor(
            dim=self.dim,
            output_dim=3,
            hidden_dim=self.dim // 2,
            dropout=0.5
        )

        self.drop_condition = config.drop_condition
        if self.drop_condition not in {
            "only_drop_ctx", "drop_ctx_or_his", "drop_ctx_and_his"
        }:
            raise ValueError(f"unsupported drop_condition: {self.drop_condition}")
        self.control_drop_prob = float(getattr(config, "control_drop_prob", 0.5))
        if not 0 <= self.control_drop_prob <= 1:
            raise ValueError("control_drop_prob must be in [0, 1]")

        self.audio_type = config.audio_type
        vocoder_path = Path(config.vocoder.path).expanduser()
        generator = load_state(save_path=str(vocoder_path.parent), tag=vocoder_path.name).eval()
        self.generator = generator
        self.generator.requires_grad_(False)

        self.use_ar_l1_loss = config.loss.use_ar_l1_loss
        self.use_stop_loss = config.loss.use_stop_loss
        self.use_vae_projected_l1_loss = config.loss.use_vae_projected_l1_loss

        self.pitch_embedding = nn.Embedding(128, self.dim)
        self.loudness_embedding = nn.Embedding(64, self.dim)
        self.duration_conditioner = DurationConditioner(
            self.dim,
            hidden_dim=max(128, self.dim // 4),
            max_reference_frames=6000,
        )

        self.japanese_linguistic_conditioner = JapaneseLinguisticConditioner(
            self.dim
        )

        # Preserve pretrained LFM2 and SVAE weights.  Only CtrlSpeech-specific
        # modules are initialized here; a future checkpoint will overwrite them
        # during inference as usual.
        for name, module in self.named_children():
            if name in {"causalAR", "generator"}:
                continue
            module.apply(self._init_weights)

        # Generic initialization above must not overwrite AdaLN-Zero or the
        # zero output projection used to stabilize a new flow decoder.
        self.LocDiT.initialize_weights()
        self.japanese_linguistic_conditioner.zero_padding_embeddings()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.generator.eval()
        return self

    def forward(
        self, raw_audio=None, audio_lengths=None, speaker_embs=None,
        text_inputs=None, text_masks=None, duration_segments=None,
        pitch=None, loudness=None, stresses=None, emotions=None,
        linguistic_features=None,
        native_text_inputs=None,
        native_text_masks=None,
        vae_features=None,
        vae_lengths=None,
        audio_loss_mask=None,
    ):
        device = self.device
        if vae_features is None:
            if raw_audio is None or audio_lengths is None:
                raise ValueError("provide raw_audio/audio_lengths or vae_features/vae_lengths")
            with torch.no_grad():
                vae_features = process_online(raw_audio, self.generator).transpose(1, 2)
            vae_lengths = torch.ceil(audio_lengths / self.generator.hop_length).long()
        elif raw_audio is not None:
            raise ValueError("raw_audio and vae_features are mutually exclusive")
        if vae_features.ndim != 3 or vae_features.shape[-1] != self.audio_channels:
            raise ValueError("vae_features must have shape [B, T, audio_channels]")
        vae_features = vae_features.detach().to(device)
        if vae_lengths is None:
            raise ValueError("vae_lengths is required for cached latents")
        vae_lengths = torch.as_tensor(vae_lengths, device=device, dtype=torch.long)
        if vae_lengths.shape != (vae_features.shape[0],) or (
            (vae_lengths <= 0) | (vae_lengths > vae_features.shape[1])
        ).any():
            raise ValueError("vae_lengths must be positive and within the latent sequence")
        vae_padding_masks = (
            torch.arange(vae_features.shape[1], device=device) < vae_lengths[:, None]
        )
        if audio_loss_mask is None:
            audio_loss_mask = vae_padding_masks
        else:
            audio_loss_mask = torch.as_tensor(audio_loss_mask, device=device, dtype=torch.bool)
            if audio_loss_mask.shape != vae_padding_masks.shape:
                raise ValueError("audio_loss_mask must match latent sequence dimensions")
            audio_loss_mask = audio_loss_mask & vae_padding_masks
        if not audio_loss_mask.any(dim=1).all():
            raise ValueError("every sample must have at least one supervised audio frame")
        if speaker_embs is None or speaker_embs.shape != (vae_features.shape[0], 192):
            raise ValueError("speaker_embs must have shape [B, 192]")

        (
            ar_input_embeds, 
            modality_type_ids, 
            padded_vae_features, 
            padded_vae_padding_masks,
            vae_projected, 
            text_masks, 
            vae_aggregated_masks
        ) = self.get_ar_input(
            vae_features=vae_features,
            vae_padding_masks=vae_padding_masks,
            speaker_embs=speaker_embs,
            text_inputs=text_inputs,
            text_masks=text_masks,
            duration_segments=duration_segments,
            pitch=pitch,
            loudness=loudness,
            linguistic_features=linguistic_features,
            native_text_inputs=native_text_inputs,
            native_text_masks=native_text_masks,
        )

        ar_padding_mask = torch.cat([text_masks, vae_aggregated_masks], dim=1)

        ar_pred = self.causalAR(
            input_embeds=ar_input_embeds,
            padding_mask=ar_padding_mask,
            modality_type_ids=modality_type_ids,
        )

        ar_pred = ar_pred[:, text_masks.shape[-1]-1:-1]
        loss_mask = F.pad(
            audio_loss_mask, (0, padded_vae_features.shape[1] - audio_loss_mask.shape[1]),
            value=False,
        )
        patch_loss_mask = loss_mask.reshape(loss_mask.shape[0], -1, self.patch_size).any(-1)

        if self.use_ar_l1_loss:
            ar_l1_loss = masked_mean(ar_pred.abs(), patch_loss_mask)
        else:
            ar_l1_loss = ar_pred.new_zeros(())

        # Flow matching from Gaussian noise at t=0 to clean latents at t=1.
        batch_size, n_patches = ar_pred.shape[:2]
        if not self.LocDiT_config.random_time:
            time = torch.rand((batch_size,), dtype=torch.float32, device=device)
            t = time[..., None, None]  # (B, 1, 1)
        else:
            time = torch.rand((batch_size, n_patches), dtype=torch.float32, device=device)
            t = time.repeat_interleave(self.patch_size, dim = 1)  # (B, n_patches * patch_size)
            t = t[..., None]  # (B, n_patches * patch_size, 1)
            time = time.view(-1)

        
        x1 = padded_vae_features
        x0 = torch.randn_like(x1)

        xt, flow = flow_matching_path(
            x1, x0, t, curved=self.LocDiT_config.time_schedule,
        )

        locdit_input, locdit_mask, vae_projected_l1_loss = self.get_decoder_input(
            vae_features=padded_vae_features,
            # vae_projected=vae_projected,      # (B, T_padded, D_model)
            vae_padding_mask=padded_vae_padding_masks,
            h_predict=ar_pred,
            noisy_input=xt,
            loss_mask=loss_mask,
        )

        time_embed = self.time_embedding(time)  # (B, D_model)
        speaker_embs = dropout_condition(
            speaker_embs, self.LocDiT_config.speaker_drop_prob, training=self.training,
        )
        if self.LocDiT_config.random_time:
            speaker_embs = speaker_embs.repeat_interleave(n_patches, dim=0)
        cond_embed = torch.cat([time_embed, speaker_embs], dim=-1)  # (B, D_model + 192)
        cond_embed = self.cond_projection(cond_embed)  # (B, D_model)
        
        pred = self.LocDiT(
            x=locdit_input,
            t=cond_embed,
            mask=locdit_mask,
            patch_size=self.patch_size,
            random_time=self.LocDiT_config.random_time,
        )
        pred = pred.reshape(batch_size, n_patches * self.patch_size, self.audio_channels)

        diff_loss = masked_mean((pred.float() - flow.float()).abs(), loss_mask)

        if self.use_stop_loss:
            input_for_stop = ar_pred.detach() * 0.5 + ar_pred * 0.5
            stop_logits, _ = self.stop_projection(input_for_stop, vae_aggregated_masks)

            # Stop loss - Three-class classification: 0 (first), 1 (middle), 2 (last)
            _, aggr_seq_len = vae_aggregated_masks.shape
            
            # Find first valid token index
            first_true_index = vae_aggregated_masks.int().argmax(dim=1, keepdim=True)  # (B, 1)
            
            # Find last valid token index
            aggr_mask_flipped = vae_aggregated_masks.flip(dims=[1])
            last_true_index = aggr_seq_len - aggr_mask_flipped.int().argmax(dim=1, keepdim=True) - 1  # (B, 1)
            
            _idxs = torch.arange(aggr_seq_len, device=device)[None, :]
            
            # Create three-class targets for speech tokens
            # 0: first token, 1: middle tokens, 2: last token
            stop_targets = torch.ones(ar_pred.shape[0], aggr_seq_len, dtype=torch.long, device=device)
            stop_targets[_idxs == first_true_index] = 0  # First token
            stop_targets[_idxs == last_true_index] = 2  # Last token
            # Middle tokens remain as 1
            # Set padding positions to -100 (will be ignored in loss calculation)
            stop_targets[~patch_loss_mask] = -100
            
            # Use cross entropy loss for multi-class classification with ignore_index
            stop_loss = F.cross_entropy(
                stop_logits.float().reshape(-1, stop_logits.shape[-1]),
                stop_targets.view(-1),
                ignore_index=-100
            )
            
            # Calculate accuracy (only on valid positions, excluding ignore_index=-100)
            stop_pred = stop_logits.argmax(dim=-1)  # (B, seq_len)
            valid_mask = (stop_targets != -100)  # Mask for non-ignored positions
            correct_predictions = (stop_pred == stop_targets) & valid_mask
            stop_acc = correct_predictions.sum().float() / valid_mask.sum().float()
        else:
            stop_loss = torch.tensor(0, device=device)
            stop_acc = torch.tensor(0, device=device)

        return {
            "diff_loss": diff_loss, 
            "stop_loss": stop_loss, 
            "stop_acc": stop_acc,
            "ar_l1_loss": ar_l1_loss,
            "vae_projected_l1_loss": vae_projected_l1_loss,
        }

    def _aggregate_segment_embed(self, frame_embed, segments):
        """Mean-pool frame-level embeddings into one embedding per duration segment.

        ``duration`` (``segments``) may span a longer range than the available
        ``pitch``/``loudness`` frames. Segments that fall entirely beyond the
        available frames contribute a zero embedding (no pitch/loudness
        conditioning there); partially-covered segments are averaged over only
        the frames that exist.
        """
        bounds = torch.as_tensor(segments, dtype=torch.long, device=frame_embed.device)
        if bounds.numel() == 0:
            return frame_embed.new_zeros((0, frame_embed.shape[-1]))
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("duration segments must have shape [N, 2]")
        starts, ends = bounds.unbind(-1)
        if ((starts < 0) | (ends < starts)).any():
            raise ValueError("duration segments must satisfy 0 <= start <= end")
        starts = starts.clamp_max(frame_embed.shape[0])
        ends = ends.clamp_max(frame_embed.shape[0])
        lengths = ends - starts
        # Prefix sums pool every phone at once instead of launching one mean
        # reduction per phone. Empty/out-of-range intervals return exact zero.
        dtype = torch.float32 if frame_embed.dtype in {torch.float16, torch.bfloat16} else frame_embed.dtype
        cumulative = F.pad(frame_embed.to(dtype).cumsum(dim=0), (0, 0, 1, 0))
        pooled = (cumulative[ends] - cumulative[starts]) / lengths.clamp_min(1)[:, None]
        return pooled.to(frame_embed.dtype)

    def get_pitch_loudness_embed(self, pitch, loudness, segment):
        pitch_embeds = []
        loudness_embeds = []
        duration_embeds = []

        if not len(pitch) == len(loudness) == len(segment):
            raise ValueError("pitch, loudness and duration_segments batch sizes must match")
        for p, l, s in zip(pitch, loudness, segment):
            pitch_embed = self.pitch_embedding(torch.as_tensor(p, dtype=torch.long, device=self.device))
            pitch_embeds.append(self._aggregate_segment_embed(pitch_embed, s))

            loudness_embed = self.loudness_embedding(torch.as_tensor(l, dtype=torch.long, device=self.device))
            loudness_embeds.append(self._aggregate_segment_embed(loudness_embed, s))

            bounds = torch.as_tensor(s, device=self.device, dtype=torch.long).reshape(-1, 2)
            duration = (bounds[:, 1] - bounds[:, 0]).float()
            duration_valid = duration > 0
            duration_embed = self.duration_conditioner(duration)
            duration_embed = duration_embed * duration_valid.unsqueeze(-1).to(
                duration_embed.dtype
            )
            duration_embeds.append(duration_embed)
        return pitch_embeds, loudness_embeds, duration_embeds

    def get_ar_input(
        self, vae_features, vae_padding_masks, speaker_embs, text_inputs, text_masks,
        duration_segments=None, pitch=None, loudness=None,
        linguistic_features=None,
        native_text_inputs=None,
        native_text_masks=None,
    ):
        if text_masks is None:
            text_masks = text_inputs.ne(0)
        text_embeds = self.causalAR.embed_phone_tokens(text_inputs)

        if linguistic_features is not None:
            text_embeds = (
                text_embeds
                + self.japanese_linguistic_conditioner(
                    linguistic_features,
                    batch_size=text_embeds.shape[0],
                    sequence_length=text_embeds.shape[1],
                ).to(text_embeds.dtype)
            )

        controls = (duration_segments, pitch, loudness)
        if any(value is not None for value in controls) and not all(value is not None for value in controls):
            raise ValueError("duration_segments, pitch and loudness must be provided together")
        if all(value is not None for value in controls):
            pitch_embeds, loudness_embeds, duration_embeds = self.get_pitch_loudness_embed(
                pitch, loudness, duration_segments
            )
            pitch_embeds = rnn.pad_sequence(pitch_embeds, padding_value=0, batch_first=True)
            loudness_embeds = rnn.pad_sequence(loudness_embeds, padding_value=0, batch_first=True)
            duration_embeds = rnn.pad_sequence(duration_embeds, padding_value=0, batch_first=True)
            for control in (pitch_embeds, loudness_embeds, duration_embeds):
                if control.shape[0] != text_embeds.shape[0] or control.shape[1] > text_embeds.shape[1]:
                    raise ValueError("prosody controls must align with the phone sequence")
                control = F.pad(control, (0, 0, 0, text_embeds.shape[1] - control.shape[1]))
                text_embeds = text_embeds + dropout_condition(
                    control, self.control_drop_prob, training=self.training,
                )

        B, L, D = vae_features.shape
        pad_length = (-L) % self.patch_size

        vae_features = vae_features.masked_fill(~vae_padding_masks.unsqueeze(-1), 0)
        padded_vae_features = F.pad(vae_features, (0, 0, 0, pad_length))
        padded_vae_padding_masks = F.pad(vae_padding_masks, (0, pad_length), value=False)

        vae_projected = self.vae_projector(padded_vae_features)

        if self.patch_size == 1:
            vae_aggregated = vae_projected
            vae_aggregated_masks = padded_vae_padding_masks
        else:
            vae_aggregated, vae_aggregated_masks = self.aggregation_encoder(
                vae_projected, 
                padding_mask=padded_vae_padding_masks
            )

        text_embeds, text_masks, text_modality_ids = (
            self.causalAR.compose_text_prefix(
                phone_embeds=text_embeds,
                phone_mask=text_masks,
                native_text_ids=native_text_inputs,
                native_text_mask=native_text_masks,
            )
        )
        text_embeds, text_masks, text_modality_ids = left_pad_text_prefix(
            text_embeds, text_masks, text_modality_ids,
        )
        audio_modality_ids = torch.full(
            vae_aggregated.shape[:-1],
            self.causalAR.AUDIO_MODALITY,
            device=self.device,
            dtype=torch.int64,
        )
        input_embeds = torch.cat([text_embeds, vae_aggregated], dim=1)
        modality_type_ids = torch.cat(
            [text_modality_ids, audio_modality_ids],
            dim=1,
        )

        return (
            input_embeds,
            modality_type_ids,
            padded_vae_features,
            padded_vae_padding_masks,
            vae_projected,
            text_masks,
            vae_aggregated_masks,
        )
    
    def get_decoder_input(self, vae_features, vae_padding_mask, h_predict, noisy_input, loss_mask=None):
        device = self.device

        B, n_patches, D_model = h_predict.shape
        history_vae_window_size = self.LocDiT_config.history_vae_window_size

        # ------------------------------------------------------------------
        # 1. Context ("ctx") feature – the current AR hidden state
        # ------------------------------------------------------------------
        ctx = h_predict  # (B, n_patches, D_model)
        folded_ctx = rearrange(ctx, 'b n_patches d_model -> (b n_patches) 1 d_model')
        
        # Pad *left* with n_history_vae zero‑tokens so the first window contains all zeros.
        projector = self.vae_projector_for_dit if self.use_seperate_linear else self.vae_projector
        dit_vae_projected = projector(vae_features)
        if self.use_vae_projected_l1_loss:
            vae_projected_l1_loss = masked_mean(
                dit_vae_projected.abs(), vae_padding_mask if loss_mask is None else loss_mask,
            )
        else:
            vae_projected_l1_loss = dit_vae_projected.new_zeros(())
        vae_projected__left_padded = F.pad(
            dit_vae_projected, (0, 0, history_vae_window_size, 0),
        )
    
        _left_pad_mask = torch.zeros(
            (B, history_vae_window_size), dtype=torch.bool, device=device
        )
        vae_padding_mask__left_padded = torch.cat([
            _left_pad_mask, vae_padding_mask,
        ], dim=1)  # (B, T_vae + n_history_vae)

        #滑动窗口，它沿着指定的维度滑动，并把窗口内的数据提取出来作为一个新的维度。
        hist_emb = vae_projected__left_padded.unfold(
            dimension = 1, 
            size = history_vae_window_size,  #指定每个窗口的大小
            step = self.patch_size, #指定窗口每次滑动的步长
        )[:, :n_patches, ...]  # (B, n_patches, D_model, n_history_vae) #(B, n_windows, D_model, size)
        hist_msk = vae_padding_mask__left_padded.unfold(
            dimension = 1, 
            size = history_vae_window_size,
            step = self.patch_size,
        )[:, :n_patches, ...]  # (B, n_patches, n_history_vae)

        # Merge batch & time for LocDiT.
        hist_emb = rearrange(
            hist_emb,
            'b n_patches d_model n_history_vae -> (b n_patches) n_history_vae d_model',
        )
        hist_msk = rearrange(
            hist_msk,
            'b n_patches n_history_vae -> (b n_patches) n_history_vae',
        )

        # 3. Current noisy patch  x_t  → embed to D_model
        # ------------------------------------------------------------------
        if self.use_seperate_linear:
            noisy_emb = self.vae_projector_for_dit(noisy_input)
        else:
            noisy_emb = self.vae_projector(noisy_input)
        
        noisy_emb = rearrange(
            noisy_emb, 
            'b (n_patches patch_size) d_model -> (b n_patches) patch_size d_model', 
            patch_size=self.patch_size
        )
        folded_vae_mask = rearrange(
            vae_padding_mask, 
            'b (n_patches patch_size) -> (b n_patches) patch_size', 
            patch_size=self.patch_size
        )

        # 4. Concatenate [ctx | history | noisy]
        # ------------------------------------------------------------------
        if self.drop_condition == "only_drop_ctx":
            folded_ctx = dropout_condition(
                folded_ctx, self.LocDiT_config.drop_cond_prob, training=self.training,
            )
        elif self.drop_condition == "drop_ctx_or_his":
            folded_ctx = dropout_condition(
                folded_ctx, self.LocDiT_config.drop_cond_prob, training=self.training,
            )
            hist_emb = dropout_condition(
                hist_emb, self.LocDiT_config.drop_cond_prob, training=self.training,
            )
        elif self.drop_condition == "drop_ctx_and_his":
            combined = dropout_condition(
                torch.cat([folded_ctx, hist_emb], dim=1),
                self.LocDiT_config.drop_cond_prob, training=self.training,
            )
            folded_ctx, hist_emb = combined[:, :1], combined[:, 1:]
        else:
            raise NotImplementedError
            
        combined_input = torch.cat((folded_ctx, hist_emb, noisy_emb), dim=1)
        
        ctx_mask = torch.ones((B * n_patches, 1), dtype=torch.bool, device=self.device)
        combined_mask = torch.cat((ctx_mask, hist_msk, folded_vae_mask), dim=1)

        return combined_input, combined_mask, vae_projected_l1_loss


    @torch.no_grad()
    def sample(
        self,
        prompt_audio,
        speaker_embs,
        text_inputs,
        text_masks,
        duration_segments=None,
        pitch=None,
        loudness=None,
        linguistic_features=None,
        native_text_inputs=None,
        native_text_masks=None,
        max_seq_length: int = 300,
        steps: int = 32,
        cfg_strength: float = 1.5,
        use_cache: bool = False,
        progress: bool = False,
    ):
        device = self.device
        if self.training:
            raise RuntimeError("call model.eval() before sampling")
        if prompt_audio.shape[0] != 1:
            raise ValueError("sampling currently supports a batch size of one")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (steps, max_seq_length)):
            raise ValueError("steps and max_seq_length must be positive integers")
        if not math.isfinite(cfg_strength) or cfg_strength < 0:
            raise ValueError("cfg_strength must be finite and non-negative")
        def odeint_fn(t, x):
            time_embed = self.time_embedding(t.unsqueeze(0))
            # Apply speaker embedding processing (same as forward)
            cond_embed = torch.cat([time_embed, speaker_embs], dim=-1)  # (B, D_model + 192)
            cond_embed = self.cond_projection(cond_embed)  # (B, D_model)

            if self.use_seperate_linear:
                x = self.vae_projector_for_dit(x)
            else:
                x = self.vae_projector(x)
            cond_input = torch.cat([ctx, historical_patch, x], dim=1)

            cond_locdit_mask = torch.cat([
                torch.ones((1, 1), dtype=torch.bool, device=device),
                historical_mask,
                torch.ones((1, self.patch_size), dtype=torch.bool, device=device)
            ], dim=1)

            pred = self.LocDiT(
                x=cond_input,
                t=cond_embed,
                mask=cond_locdit_mask,
                patch_size=self.patch_size,
                random_time=self.LocDiT_config.random_time,
            )

            if cfg_strength == 0:
                return pred
            if self.drop_condition in {"only_drop_ctx", "drop_ctx_or_his", "drop_ctx_and_his"}:
                # For unconditional: use zero speaker embedding
                uncond_speaker_embs = torch.zeros_like(speaker_embs)
                uncond_cond_embed = torch.cat([time_embed, uncond_speaker_embs], dim=-1)
                uncond_cond_embed = self.cond_projection(uncond_cond_embed)

                uncond_history = (
                    torch.zeros_like(historical_patch)
                    if self.drop_condition == "drop_ctx_and_his" else historical_patch
                )
                uncond_input = torch.cat([torch.zeros_like(ctx), uncond_history, x], dim=1)
                uncond_locdit_mask = cond_locdit_mask

                null_pred = self.LocDiT(
                    x=uncond_input,
                    t=uncond_cond_embed,
                    mask=uncond_locdit_mask,
                    patch_size=self.patch_size,
                    random_time=self.LocDiT_config.random_time,
                )
            return pred + (pred - null_pred) * cfg_strength

        if len(prompt_audio.shape) == 2:
            prompt_audio = prompt_audio.unsqueeze(1)
        
        prompt_vae_features = process_online(prompt_audio, self.generator).transpose(1, 2) # (B, T, D)
        prompt_vae_padding_masks = torch.ones(
            prompt_vae_features.shape[:-1], dtype=torch.bool, device=device)

        (
            ar_input_embeds, 
            modality_type_ids, 
            padded_vae_features, 
            padded_vae_padding_masks,
            vae_projected, 
            text_masks, 
            vae_aggregated_masks
        ) = self.get_ar_input(
            vae_features=prompt_vae_features,
            vae_padding_masks=prompt_vae_padding_masks,
            speaker_embs=speaker_embs,
            text_inputs=text_inputs,
            text_masks=text_masks,
            duration_segments=duration_segments,
            pitch=pitch,
            loudness=loudness,
            linguistic_features=linguistic_features,
            native_text_inputs=native_text_inputs,
            native_text_masks=native_text_masks,
        )

        ar_padding_mask = torch.cat([text_masks, vae_aggregated_masks], dim=1)

        vae_results = torch.empty((1, 0, self.audio_channels), device=device)
        past_key_values = None
        stop_past_key_values = None
        for _ in tqdm(range(max_seq_length), disable=not progress):
            h_predict, past_key_values = self.causalAR.inference(
                input_embeds=ar_input_embeds,
                padding_mask=ar_padding_mask,
                modality_type_ids=modality_type_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            
            h_predict_for_stop = h_predict
            stop_mask = torch.ones(
                (1, h_predict_for_stop.shape[1]), 
                dtype=torch.bool, device=device
            )
            
            if use_cache:
                stop_output = self.stop_projection(
                    h_predict_for_stop, stop_mask, 
                    past_key_values=stop_past_key_values,
                    use_cache=True
                )
                stop_logits, stop_past_key_values = stop_output
                stop_logits = stop_logits[:, -1]
            else:
                stop_logits, _ = self.stop_projection(h_predict_for_stop, stop_mask)
                stop_logits = stop_logits[:, -1]

            h_predict = h_predict[:, [-1], :]

            y0 = torch.randn(1, self.patch_size, self.audio_channels, device=device)
            
            t_start = 0
            t = torch.linspace(t_start, 1, steps+1, device=device)

            # Sway sampling 
            sway_sampling_coef = -1
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t) #[33][nfe+1]
            
            if vae_results.numel() == 0:
                historical_patch = prompt_vae_features[:, -self.LocDiT_config.history_vae_window_size:, :]
            else:
                current_length = vae_results.shape[1]
                if current_length < self.LocDiT_config.history_vae_window_size:
                    historical_patch = torch.cat([prompt_vae_features[
                        :, -(self.LocDiT_config.history_vae_window_size-current_length):, :
                    ], vae_results], dim=1)
                else:
                    historical_patch = vae_results[:, -self.LocDiT_config.history_vae_window_size:, :]
            
            history_length = historical_patch.shape[1]
            projector = self.vae_projector_for_dit if self.use_seperate_linear else self.vae_projector
            historical_patch = projector(historical_patch)
            historical_patch = F.pad(
                historical_patch,
                (0, 0, self.LocDiT_config.history_vae_window_size - history_length, 0),
            )
            historical_mask = torch.arange(
                self.LocDiT_config.history_vae_window_size, device=device,
            )[None, :] >= self.LocDiT_config.history_vae_window_size - history_length
            
            ctx = h_predict
            trajectory = odeint(odeint_fn, y0, t, **self.LocDiT_config.odeint_kwargs)
            sampled = trajectory[-1]
            vae_results = torch.cat((vae_results, sampled), dim=1)

            # 停止判断（全batch判断）- Check if predicted class is 2 (last token)
            stop_pred_class = stop_logits.argmax(dim=-1)
            if stop_pred_class == 2 and vae_results.shape[1] > 10:
                break
        
            # Update for the next step
            input_vae_patch_emb = self.vae_projector(sampled)
            vae_patch_mask = torch.ones((1, self.patch_size), dtype=torch.bool, device=device)
            if self.patch_size == 1:
                aggregation_emb = input_vae_patch_emb
            else:
                aggregation_emb, aggregation_mask = self.aggregation_encoder(
                    input_vae_patch_emb, padding_mask=vae_patch_mask
                )  # B T/4 D
            
            if use_cache:
                ar_input_embeds = aggregation_emb
            else:
                ar_input_embeds = torch.cat([ar_input_embeds, aggregation_emb], dim=1)
            
            ar_padding_mask = torch.cat([
                ar_padding_mask, torch.ones((1, 1), dtype=torch.bool, device=device)
            ], dim=1)
            modality_type_ids = torch.cat([
                modality_type_ids, torch.ones((1, 1), dtype=torch.int64, device=device)
            ], dim=1)
    
        return vae_results

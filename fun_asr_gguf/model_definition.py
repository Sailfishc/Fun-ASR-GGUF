import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import base64
import numpy as np

# ============================================================================
# Basic Building Blocks (from export_torch_model.py)
# ============================================================================

class SinusoidalPositionEncoder(nn.Module):
    def __init__(self, d_model=80, dropout_rate=0.1):
        super().__init__()

    def encode(
        self,
        positions: torch.Tensor = None,
        depth: int = None,
        dtype: torch.dtype = torch.float32,
    ):
        batch_size = positions.size(0)
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(
            torch.tensor([10000], dtype=dtype, device=device)
        ) / (depth / 2 - 1)
        inv_timescales = torch.exp(
            torch.arange(depth / 2, device=device).type(dtype)
            * (-log_timescale_increment)
        )
        inv_timescales = inv_timescales.unsqueeze(0)
        scaled_time = positions.unsqueeze(-1) * inv_timescales.unsqueeze(1)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)
        return x + position_encoding

class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate, activation=None):
        super().__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        if activation is None:
            activation = torch.nn.ReLU()
        self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

# ============================================================================
# SANM Components (from export_torch_model.py)
# ============================================================================

class MultiHeadedAttentionSANM(nn.Module):
    def __init__(
        self,
        n_head,
        in_feat,
        n_feat,
        dropout_rate,
        kernel_size,
        sanm_shfit=0,
    ):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fsmn_block = nn.Conv1d(
            n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False
        )
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask):
        b, t, d = inputs.size()
        if mask is not None:
            mask = mask.unsqueeze(-1)
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x

    def forward_qkv(self, x):
        b, t, d = x.size()
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, int(self.h * self.d_k), dim=-1)
        q_h = q.unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        k_h = k.unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        v_h = v.unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        return q_h, k_h, v_h, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.permute(0, 2, 1, 3).flatten(2)
        return self.linear_out(x)

    def forward(self, x, mask):
        q_h, k_h, v_h, v = self.forward_qkv(x)
        fsmn_memory = self.forward_fsmn(v, mask)
        q_h = q_h * self.d_k ** (-0.5)
        scores = torch.matmul(q_h, k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask)
        return att_outs + fsmn_memory

class EncoderLayerSANM(nn.Module):
    def __init__(
        self,
        in_size,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after

    def forward(self, x, mask):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat) if self.in_size != self.size else residual + self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + self.dropout(self.self_attn(x, mask))
            else:
                x = self.dropout(self.self_attn(x, mask))
                return x, mask

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask

# ============================================================================
# Main Model Components (from export_torch_model.py)
# ============================================================================

class SenseVoiceEncoderSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 80 * 7
        self.output_size = 512
        self.attention_heads = 4
        self.linear_units = 2048
        self.num_blocks = 50
        self.tp_blocks = 20
        self.dropout_rate = 0.1
        self.attention_dropout_rate = 0.1
        self.kernel_size = 11
        self.sanm_shfit = 0

        self.embed = SinusoidalPositionEncoder()

        self.encoders0 = nn.ModuleList([
            EncoderLayerSANM(
                self.input_size, self.output_size,
                MultiHeadedAttentionSANM(self.attention_heads, self.input_size, self.output_size, self.attention_dropout_rate, self.kernel_size, self.sanm_shfit),
                PositionwiseFeedForward(self.output_size, self.linear_units, self.dropout_rate),
                self.dropout_rate
            ) for _ in range(1)
        ])

        self.encoders = nn.ModuleList([
            EncoderLayerSANM(
                self.output_size, self.output_size,
                MultiHeadedAttentionSANM(self.attention_heads, self.output_size, self.output_size, self.attention_dropout_rate, self.kernel_size, self.sanm_shfit),
                PositionwiseFeedForward(self.output_size, self.linear_units, self.dropout_rate),
                self.dropout_rate
            ) for _ in range(self.num_blocks - 1)
        ])

        self.tp_encoders = nn.ModuleList([
            EncoderLayerSANM(
                self.output_size, self.output_size,
                MultiHeadedAttentionSANM(self.attention_heads, self.output_size, self.output_size, self.attention_dropout_rate, self.kernel_size, self.sanm_shfit),
                PositionwiseFeedForward(self.output_size, self.linear_units, self.dropout_rate),
                self.dropout_rate
            ) for _ in range(self.tp_blocks)
        ])

        self.after_norm = LayerNorm(self.output_size)
        self.tp_norm = LayerNorm(self.output_size)

    def forward(self, xs_pad: torch.Tensor):
        masks = None
        xs_pad = xs_pad * (self.output_size**0.5)
        xs_pad = self.embed(xs_pad)

        for layer in self.encoders0:
            xs_pad, masks = layer(xs_pad, masks)
        for layer in self.encoders:
            xs_pad, masks = layer(xs_pad, masks)
        
        xs_pad = self.after_norm(xs_pad)
        for layer in self.tp_encoders:
            xs_pad, masks = layer(xs_pad, masks)
        
        xs_pad = self.tp_norm(xs_pad)
        return xs_pad

class CTC(nn.Module):
    def __init__(self, odim, encoder_output_size):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, odim)

    def forward(self, hs_pad):
        return self.ctc_lo(hs_pad)

# ============================================================================
# DML Optimized Adaptor Components (from 01-Export-Encoder-Adaptor-CTC.py)
# ============================================================================

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        q = self.linear_q(query).unflatten(-1, (self.h, self.d_k))
        k = self.linear_k(key).unflatten(-1, (self.h, self.d_k))
        v = self.linear_v(value).unflatten(-1, (self.h, self.d_k))
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_attention(self, value, scores, mask):
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        x = torch.matmul(self.dropout(attn), value)
        x = x.transpose(1, 2).flatten(2)
        return self.linear_out(x)

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k ** -0.5)
        return self.forward_attention(v, scores, mask)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate, normalize_before=True):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, x, mask=None):
        residual = x
        if self.normalize_before: x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, mask))
        if not self.normalize_before: x = self.norm1(x)

        residual = x
        if self.normalize_before: x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before: x = self.norm2(x)
        return x, mask

class CorrectTransformerAdaptor(nn.Module):
    def __init__(self, downsample_rate=1, encoder_dim=512, llm_dim=1024, ffn_dim=2048, n_layer=2, **kwargs):
        super().__init__()
        self.k = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, llm_dim)
        self.blocks = nn.ModuleList([
            EncoderLayer(
                llm_dim,
                MultiHeadedAttention(kwargs.get("attention_heads", 8), llm_dim, kwargs.get("attention_dropout_rate", 0.0)),
                PositionwiseFeedForward(llm_dim, llm_dim // 4, kwargs.get("dropout_rate", 0.0)),
                kwargs.get("dropout_rate", 0.0)
            ) for _ in range(n_layer)
        ]) if n_layer > 0 else None

    def forward(self, x, ilens=None):
        batch_size, seq_len, dim = x.size()
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        x = F.pad(x, (0, 0, 0, pad_num), value=0.0)
        x = x.unflatten(1, (chunk_num, self.k)).flatten(2)
        
        x = self.linear2(self.relu(self.linear1(x)))
        masks = None
        if self.blocks is not None:
            for block in self.blocks: x, masks = block(x, masks)
        return x, None

# ============================================================================
# Hybrid Model & Export Wrappers
# ============================================================================

class HybridSenseVoice(nn.Module):
    def __init__(self, encoder_dim=512, llm_dim=1024, vocab_size=60515):
        super().__init__()
        self.audio_encoder = SenseVoiceEncoderSmall()
        self.audio_adaptor = CorrectTransformerAdaptor(downsample_rate=1, encoder_dim=encoder_dim, llm_dim=llm_dim, n_layer=2)
        self.ctc_decoder = CorrectTransformerAdaptor(downsample_rate=1, encoder_dim=encoder_dim, llm_dim=encoder_dim, n_layer=5)
        self.ctc_proj = CTC(odim=vocab_size, encoder_output_size=encoder_dim)
        
    def load_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("audio_encoder."): new_state_dict[k] = v
            elif k.startswith("audio_adaptor."): new_state_dict[k] = v
            elif k.startswith("ctc_decoder."): new_state_dict[k] = v
            elif k.startswith("ctc.ctc_lo."): new_state_dict[k.replace("ctc.ctc_lo", "ctc_proj.ctc_lo")] = v
        self.load_state_dict(new_state_dict, strict=False)

# ============================================================================
# STFT Frontend Component (from export_torch_model.py)
# ============================================================================

class STFT_Process(nn.Module):
    def __init__(self, model_type, n_fft=400, win_length=400, hop_len=160, max_frames=0, window_type='hamming'):
        super().__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.half_n_fft = n_fft // 2
        
        window = torch.hamming_window(win_length, periodic=True).float()
        if win_length < n_fft:
            pl = (n_fft - win_length) // 2
            pr = n_fft - win_length - pl
            window = F.pad(window, (pl, pr))
            
        t = torch.arange(n_fft).float().unsqueeze(0)
        f = torch.arange(self.half_n_fft + 1).float().unsqueeze(1)
        omega = 2 * torch.pi * f * t / n_fft
        self.register_buffer('cos_kernel', (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1))
        self.register_buffer('sin_kernel', (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1))

    def forward(self, x):
        # Default center padding
        x_padded = F.pad(x, (self.half_n_fft, self.half_n_fft), mode='constant')
        real = F.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        imag = F.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)
        return real, imag

class EncoderExportWrapper(nn.Module):
    def __init__(self, hybrid_model, stft_model, fbank, pre_emphasis=0.97, lfr_m=7, lfr_n=6):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.stft_model = stft_model
        self.pre_emphasis_val = float(pre_emphasis)
        self.register_buffer('pre_emphasis', torch.tensor(pre_emphasis, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('fbank', fbank)
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.lfr_m_factor = (lfr_m - 1) // 2

    def forward(self, audio):
        # Audio Processing
        audio = audio.float()
        audio = audio - torch.mean(audio)
        if self.pre_emphasis_val > 0:
            audio = torch.cat([audio[..., :1], audio[..., 1:] - self.pre_emphasis * audio[..., :-1]], dim=-1)
        real, imag = self.stft_model(audio)
        mel = (torch.matmul(self.fbank, real * real + imag * imag).transpose(1, 2) + 1e-7).log()
        
        # LFR Processing
        T = mel.shape[1]
        T_lfr = (T + self.lfr_n - 1) // self.lfr_n
        pad_len = (T_lfr * self.lfr_n + self.lfr_m) - T
        left_pad = mel[:, [0]].repeat(1, self.lfr_m_factor, 1)
        right_pad = mel[:, [-1]].repeat(1, pad_len, 1)
        padded = torch.cat([left_pad, mel, right_pad], dim=1)
        
        lfr_list = []
        for i in range(self.lfr_m):
            feat = padded[:, i : i + T_lfr * self.lfr_n : self.lfr_n]
            lfr_list.append(feat[:, :T_lfr, :])
        
        x = torch.cat(lfr_list, dim=-1)
        
        # 1. Encoder Execution
        enc_output = self.hybrid_model.audio_encoder(x)
        
        # 2. Adaptor Execution
        adaptor_output, _ = self.hybrid_model.audio_adaptor(enc_output, None)
        
        # 3. Official Slicing (Simulated downsampling)
        olens_1 = 1 + (T_lfr - 3 + 2) // 2
        olens_2 = 1 + (olens_1 - 3 + 2) // 2
        target_len = (olens_2 - 1) // 2 + 1
        
        return enc_output, adaptor_output[:, :target_len, :]

class CTCHeadExportWrapper(nn.Module):
    def __init__(self, hybrid_model):
        super().__init__()
        self.ctc_decoder = hybrid_model.ctc_decoder
        self.ctc_proj = hybrid_model.ctc_proj
        
    def forward(self, enc_output):
        h, _ = self.ctc_decoder(enc_output, None)
        logits = self.ctc_proj.ctc_lo(h)
        indices = torch.argmax(logits, dim=-1).to(torch.int32)
        return indices

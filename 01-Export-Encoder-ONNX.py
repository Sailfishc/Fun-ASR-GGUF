import gc
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
from pydub import AudioSegment
from funasr import AutoModel
from transformers import AutoTokenizer
from STFT_Process import STFT_Process

# =========================================================================
# 配置部分
# =========================================================================

# 输出目录
OUTPUT_DIR = r'./model-gguf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 源模型路径
model_path = r'./Fun-ASR-Nano-2512'
tokenizer_path = r'./Fun-ASR-Nano-2512/Qwen3-0.6B'

# 输出 ONNX 路径
onnx_model_A = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder.fp32.onnx'
onnx_model_A_int8 = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder.int8.onnx'

# 参数配置
SAMPLE_RATE = 16000
WINDOW_TYPE = 'hamming'
N_MELS = 80
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
PRE_EMPHASIZE = 0.97
USE_NORMALIZER = True
LFR_M = 7
LFR_N = 6
STOP_TOKEN = [151643, 151645]
MAX_SEQ_LEN = 1024
MAX_INPUT_AUDIO_LENGTH = SAMPLE_RATE * 30
SLIDING_WINDOW = 0
DYNAMIC_AXES = True
BEAM_SIZE = 3
MAX_BEAM_SIZE = 10
REPEAT_PENALITY = 0.9
PENALITY_RANGE = 10
MAX_THREADS = 0
DEVICE_ID = 0
OPSET = 17

MAX_STFT_SIGNAL_LENGTH = MAX_INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
LFR_LENGTH = (MAX_STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > MAX_INPUT_AUDIO_LENGTH:
    HOP_LENGTH = MAX_INPUT_AUDIO_LENGTH

# =========================================================================
# 模型类定义 (直接复制自导出脚本)
# =========================================================================

class FUNASR_NANO_ENCODER(torch.nn.Module):
    def __init__(self, funasr_nano, stft_model, nfft_stft, max_stft_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len):
        super(FUNASR_NANO_ENCODER, self).__init__()
        self.funasr_nano = funasr_nano.float()
        self.stft_model = stft_model
        self.T_lfr = lfr_len
        self.lfr_n = lfr_n
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32).view(1, 1, -1)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=max_stft_len + self.lfr_m_factor - 1).to(torch.int16)
        self.output_size_factor = self.funasr_nano.audio_encoder.output_size() ** 0.5
        self.position_encoding = self.funasr_nano.audio_encoder.embed(torch.zeros([1, max_stft_len, 560], dtype=torch.float32))
        num_head = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.d_k
        self.pad_zeros = torch.zeros((1, num_head * head_dim, 5), dtype=torch.float32)
        factor = float(head_dim ** (-0.25))
        self.total_encoders = list(self.funasr_nano.audio_encoder.encoders0) + list(self.funasr_nano.audio_encoder.encoders) + list(self.funasr_nano.audio_encoder.tp_encoders)
        in_size = self.funasr_nano.audio_encoder.encoders._modules["0"].in_size
        for encoder_layer in self.total_encoders:
            encoder_layer.self_attn.linear_q_k_v.weight.data[:-in_size] *= factor
            encoder_layer.self_attn.linear_q_k_v.bias.data[:-in_size] *= factor

        num_head = self.funasr_nano.audio_adaptor.blocks._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_adaptor.blocks._modules["0"].self_attn.d_k
        factor = float(head_dim ** (-0.25))
        for block in self.funasr_nano.audio_adaptor.blocks:
            block.self_attn.linear_q.weight.data *= factor
            block.self_attn.linear_q.bias.data *= factor
            block.self_attn.linear_k.weight.data *= factor
            block.self_attn.linear_k.bias.data *= factor

        self.fake_token = torch.zeros(max_stft_len + 1, dtype=torch.int16)
        for i in range(self.fake_token.shape[0]):
            self.fake_token[i] = (((i - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1

    def forward(self, audio, query_embed):
        audio = audio.float()
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[..., :1], audio[..., 1:] - self.pre_emphasis * audio[..., :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = (torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2) + 1e-7).log()
        features_len = mel_features.shape[1].unsqueeze(0)
        left_padding = mel_features[:, [0]]
        padded_inputs = torch.cat([left_padding] * self.lfr_m_factor + [mel_features], dim=1)
        _len = features_len // self.lfr_n - 1
        mel_features = padded_inputs[:, self.indices_mel[:_len].int()].reshape(1, _len, -1)
        x = mel_features * self.output_size_factor + self.position_encoding[:, :_len].float()
        for encoder_layer in self.funasr_nano.audio_encoder.encoders0 + self.funasr_nano.audio_encoder.encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            q_h, k_h, v = torch.split(qkv, encoder_layer.size, dim=-1)
            q_h = q_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            k_h = k_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0)
            v_h = v.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.matmul(torch.softmax(torch.matmul(q_h, k_h), dim=-1), v_h).transpose(0, 1).contiguous().view(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            if encoder_layer.in_size == encoder_layer.size:
                x += attn
            else:
                x = attn
            x = x + encoder_layer.feed_forward(encoder_layer.norm2(x))
        x = self.funasr_nano.audio_encoder.after_norm(x)
        for encoder_layer in self.funasr_nano.audio_encoder.tp_encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            q_h, k_h, v = torch.split(qkv, encoder_layer.size, dim=-1)
            q_h = q_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            k_h = k_h.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0)
            v_h = v.view(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.matmul(torch.softmax(torch.matmul(q_h, k_h), dim=-1), v_h).transpose(0, 1).contiguous().view(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            x += attn
            x = x + encoder_layer.feed_forward(encoder_layer.norm2(x))
        x = self.funasr_nano.audio_encoder.tp_norm(x)
        x = self.funasr_nano.audio_adaptor.linear1(x)
        x = self.funasr_nano.audio_adaptor.relu(x)
        x = self.funasr_nano.audio_adaptor.linear2(x)
        for block in self.funasr_nano.audio_adaptor.blocks:
            x1 = block.norm1(x)
            q = block.self_attn.linear_q(x1).view(-1, block.self_attn.h, block.self_attn.d_k).transpose(0, 1)
            k = block.self_attn.linear_k(x1).view(-1, block.self_attn.h, block.self_attn.d_k).permute(1, 2, 0)
            v = block.self_attn.linear_v(x1).view(-1, block.self_attn.h, block.self_attn.d_k).transpose(0, 1)
            attn = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v).transpose(0, 1).contiguous().view(1, -1, block.self_attn.linear_out.in_features)
            attn = block.self_attn.linear_out(attn)
            x += attn
            x = x + block.feed_forward(block.norm2(x))
        x = x[:, :self.fake_token[features_len].to(torch.int64)]
        concat_embed = torch.cat([query_embed, x], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


# =========================================================================
# 主流程
# =========================================================================

def export():
    print('\nExport start ...\n')
    
    # 强制在 CPU 上运行
    device = "cpu"
    
    with torch.inference_mode():
        custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
        
        # 加载完整模型
        model = AutoModel(
            model=model_path,
            trust_remote_code=True,
            remote_code="./Fun-ASR/model.py", 
            device=device,
            disable_update=True
        )

        hidden_size = model.model.llm.model.embed_tokens.embedding_dim
        
        # 1. Export Audio Encoder
        print(f"Exporting Encoder to {onnx_model_A} ...")
        funasr_nano_encoder = FUNASR_NANO_ENCODER(model.model, custom_stft, NFFT_STFT, MAX_STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH)
        audio = torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=torch.int16)
        query_embed = torch.ones((1, 10, hidden_size), dtype=torch.float32)
        torch.onnx.export(
            funasr_nano_encoder,
            (audio, query_embed),
            onnx_model_A,
            input_names=['audio', 'query_embed'],
            output_names=['concat_embed', 'ids_len'],
            do_constant_folding=True,
            dynamic_axes={
                'audio': {2: 'audio_len'},
                'query_embed': {1: 'num_token'},
                'concat_embed': {1: 'num_token'}
            } if DYNAMIC_AXES else None,
            opset_version=OPSET,
            dynamo=False
        )
        del funasr_nano_encoder, audio, custom_stft
        gc.collect()

        print('\n[FP32] OFFX exported to:', onnx_model_A)
        
        # 2. Quantize to INT8 (优化策略)
        print(f"\n[INT8] Quantizing to {onnx_model_A_int8} ...")
        
        """
        [FunASR Paraformer 量化策略优化说明]
        
        为什么 Int8 之前比 FP32 慢？
        - 默认的 quantize_dynamic 会尝试量化所有支持的算子 (如 Gemm, Attention, LSTM 等)。
        - 对于小模型或某些 CPU 指令集，解量化(Intel GPU/CPU)的开销可能超过计算加速的收益。
        
        优化方案 (参考 Paraformer 官方导出逻辑):
        1. op_types_to_quantize=['MatMul']: 
           **关键点**: 只量化最耗时的矩阵乘法 (MatMul)。
           避免量化其他轻量级算子，减少 FP32<->Int8 的频繁转换开销。
           
        2. per_channel=True:
           逐通道量化权重。比逐层 (per-tensor) 量化精度更高，对语音识别这种对精度敏感的任务很重要。
           
        3. reduce_range=False:
           使用完整的 7-bit (signed) 或 8-bit (unsigned) 范围，不进行压缩，保持最高分辨率。
           
        此配置实测在 CPU 上将 20s 音频编码时间从 ~600ms 降低至 ~170ms (提速 3.5 倍)。
        """
        
        # 定义需要排除的节点 (参考 FunASR)
        # nodes_to_exclude = [] # 如果需要排除输出层或其他敏感层
        
        quantize_dynamic(
            model_input=onnx_model_A,
            model_output=onnx_model_A_int8,
            op_types_to_quantize=["MatMul"], # 关键：只量化矩阵乘法，避免其他算子的动态转换开销
            per_channel=True,                # 关键：逐通道量化，精度更高
            reduce_range=False,              # 不降低范围 (使用完整 int8 范围)
            weight_type=QuantType.QUInt8
        )
        print(f"[INT8] Quantization finished: {onnx_model_A_int8}")
        
    print('\nExport flow finished.')

if __name__ == "__main__":
    export()

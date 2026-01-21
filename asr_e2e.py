"""
端到端 ASR 推理脚本 (End-to-End ASR Inference)

流程:
1. 加载音频文件 (MP3/WAV/etc) -> PCM 16kHz
2. ONNX Encoder: 音频 -> Audio Embedding
3. 从 GGUF 读取 token_embd.weight，生成 prefix/suffix embedding
4. 拼接 [prefix + audio + suffix]
5. llama.dll 直接注入 embedding 并生成文本

依赖:
- onnxruntime
- pydub (音频处理)
- gguf (读取 GGUF 模型权重)
- llama.dll / ggml.dll (推理)
"""

import sys
import os
import ctypes
import numpy as np
import time
import gguf

# =========================================================================
# Vulkan 选项
# =========================================================================
# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止 Vulkan 用独显（强制用集显）
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 VulkanFP16 计算（Intel集显fp16有溢出问题）

# =========================================================================



# =========================================================================
# 配置区 (硬编码，直接修改这些值)
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径
MODEL_DIR           = os.path.join(SCRIPT_DIR, "model-gguf")
ENCODER_ONNX_PATH   = os.path.join(MODEL_DIR, "Fun-ASR-Nano-Encoder.int8.onnx")  
CTC_ONNX_PATH       = os.path.join(MODEL_DIR, "Fun-ASR-Nano-CTC.int8.onnx")
DECODER_GGUF_PATH   = os.path.join(MODEL_DIR, "Fun-ASR-Nano-Decoder.q8_0.gguf")
TOKENS_PATH         = os.path.join(MODEL_DIR, "tokens.txt")

# llama.cpp 编译版本的 DLL 所在目录
# 下载地址：https://github.com/ggml-org/llama.cpp/releases/download/b7786/llama-b7786-bin-win-vulkan-x64.zip
BIN_DIR             = os.path.join(SCRIPT_DIR, "bin")
GGML_DLL_PATH       = os.path.join(BIN_DIR, "ggml.dll")
LLAMA_DLL_PATH = os.path.join(BIN_DIR, "llama.dll")
GGML_BASE_DLL_PATH = os.path.join(BIN_DIR, "ggml-base.dll")

# 输入音频
INPUT_AUDIO = os.path.join(SCRIPT_DIR, "input2.mp3")

# ASR Prompts
# 默认热词表（在本例中用于简单匹配）
DEFAULT_HOTWORDS = ['Claude Code', 'Antigravity', 'SenseVoice', 'FunASR']
STOP_TOKENS = [151643, 151645]

# 音频参数
SAMPLE_RATE = 16000
USE_NORMALIZER = True
MAX_INPUT_AUDIO_LENGTH = SAMPLE_RATE * 30  # 最大音频长度 (samples)

# 推理参数
core_num = os.cpu_count()
N_PREDICT = 512                 # 最大生成 token 数
N_THREADS = core_num // 2       # 生成阶段，内存带宽受限，使用物理核心数
N_THREADS_BATCH = core_num      # 读取阶段，计算密集，线程全开
QUIET_MODE = True               # 静默模式 (关闭 llama.cpp 调试信息)
INJECT_CHUNK_SIZE = 512         # 注入 Embedding 时的分块大小 
N_UBATCH = 512                  # llama.cpp 内部物理 batch 大小

# =========================================================================
# DLL Loading
# =========================================================================



# Load backends - 临时切换到 bin 目录，让 Windows 找到依赖 DLL (如 MKL, SYCL, Vulkan 等)
original_cwd = os.getcwd()      # 保存原工作目录
os.chdir(BIN_DIR)               # 临时切换到 bin 目录
try:
    ggml = ctypes.CDLL(GGML_DLL_PATH)
    ggml_base = ctypes.CDLL(GGML_BASE_DLL_PATH)
    llama = ctypes.CDLL(LLAMA_DLL_PATH)

    ggml_backend_load_all = ggml.ggml_backend_load_all

    ggml_backend_load_all.argtypes = []
    ggml_backend_load_all.restype = None
    ggml_backend_load_all()         # 加载后端

finally:
    os.chdir(original_cwd)          # 立即切换回原目录

# GGML Device Enumeration Bindings
ggml_backend_dev_count = ggml.ggml_backend_dev_count
ggml_backend_dev_count.argtypes = []
ggml_backend_dev_count.restype = ctypes.c_size_t

ggml_backend_dev_get = ggml.ggml_backend_dev_get
ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
ggml_backend_dev_get.restype = ctypes.c_void_p

ggml_backend_dev_description = ggml_base.ggml_backend_dev_description
ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
ggml_backend_dev_description.restype = ctypes.c_char_p

ggml_backend_dev_name = ggml_base.ggml_backend_dev_name
ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
ggml_backend_dev_name.restype = ctypes.c_char_p


# =========================================================================
# Type Definitions
# =========================================================================

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

# =========================================================================
# Function Prototypes
# =========================================================================

# Logging
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def quiet_log_callback(level, message, user_data):
    pass
llama_log_set = llama.llama_log_set
llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
llama_log_set.restype = None

# Backend
llama_backend_init = llama.llama_backend_init
llama_backend_init.argtypes = []
llama_backend_init.restype = None

llama_backend_free = llama.llama_backend_free
llama_backend_free.argtypes = []
llama_backend_free.restype = None

# Model
llama_model_default_params = llama.llama_model_default_params
llama_model_default_params.argtypes = []
llama_model_default_params.restype = llama_model_params

llama_model_load_from_file = llama.llama_model_load_from_file
llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
llama_model_load_from_file.restype = ctypes.c_void_p

llama_model_free = llama.llama_model_free
llama_model_free.argtypes = [ctypes.c_void_p]
llama_model_free.restype = None

llama_model_get_vocab = llama.llama_model_get_vocab
llama_model_get_vocab.argtypes = [ctypes.c_void_p]
llama_model_get_vocab.restype = ctypes.c_void_p

# Context
llama_context_default_params = llama.llama_context_default_params
llama_context_default_params.argtypes = []
llama_context_default_params.restype = llama_context_params

llama_init_from_model = llama.llama_init_from_model
llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
llama_init_from_model.restype = ctypes.c_void_p

llama_free = llama.llama_free
llama_free.argtypes = [ctypes.c_void_p]
llama_free.restype = None

# Batch
llama_batch_init = llama.llama_batch_init
llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
llama_batch_init.restype = llama_batch

llama_batch_free = llama.llama_batch_free
llama_batch_free.argtypes = [llama_batch]
llama_batch_free.restype = None

# State
llama_state_get_size = llama.llama_state_get_size
llama_state_get_size.argtypes = [ctypes.c_void_p]
llama_state_get_size.restype = ctypes.c_size_t

llama_state_get_data = llama.llama_state_get_data
llama_state_get_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
llama_state_get_data.restype = ctypes.c_size_t

llama_state_set_data = llama.llama_state_set_data
llama_state_set_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
llama_state_set_data.restype = ctypes.c_size_t

# Decode
llama_decode = llama.llama_decode
llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
llama_decode.restype = ctypes.c_int32

# Logits
llama_get_logits = llama.llama_get_logits
llama_get_logits.argtypes = [ctypes.c_void_p]
llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

# Tokenize
llama_tokenize = llama.llama_tokenize
llama_tokenize.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(llama_token), ctypes.c_int32,
    ctypes.c_bool, ctypes.c_bool,
]
llama_tokenize.restype = ctypes.c_int32

# Vocab
llama_vocab_n_tokens = llama.llama_vocab_n_tokens
llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
llama_vocab_n_tokens.restype = ctypes.c_int32

llama_vocab_eos = llama.llama_vocab_eos
llama_vocab_eos.argtypes = [ctypes.c_void_p]
llama_vocab_eos.restype = llama_token

llama_token_to_piece = llama.llama_token_to_piece
llama_token_to_piece.argtypes = [ctypes.c_void_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
llama_token_to_piece.restype = ctypes.c_int

# =========================================================================
# CTC & Hotword Functions
# =========================================================================

import base64

def load_ctc_tokens(filename):
    """加载 CTC 词表"""
    id2token = dict()
    if not os.path.exists(filename):
        return id2token
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) == 1:
                t, i = " ", parts[0]
            else:
                t, i = parts
            id2token[int(i)] = t
    return id2token

def decode_ctc(logits, id2token):
    """解码 CTC Logits 并计算时间戳"""
    indices = np.argmax(logits[0], axis=-1)
    blank_id = max(id2token.keys()) if id2token else 0
    
    frame_shift_ms = 60
    offset_ms = -30
    
    results = []
    last_idx = -1
    for i, idx in enumerate(indices):
        if idx == blank_id or idx == last_idx:
            last_idx = idx
            continue
        last_idx = idx
        token_b64 = id2token.get(idx, "")
        if not token_b64: continue
        
        try:
            token_text = base64.b64decode(token_b64).decode("utf-8")
        except:
            continue
            
        timestamp = max((i * frame_shift_ms + offset_ms) / 1000.0, 0.0)
        results.append({"text": token_text, "start": timestamp})
                
    full_text = "".join([r["text"] for r in results])
    return full_text, results

def match_hotwords(text, hotword_list):
    """简单的热词匹配逻辑"""
    matched = []
    target_text = text.lower()
    for hw in hotword_list:
        if hw.lower() in target_text:
            matched.append(hw)
    return matched

# =========================================================================
# Helper Functions
# =========================================================================

def text_to_tokens(vocab, text):
    """使用 llama.dll 进行文本分词"""
    text_bytes = text.encode("utf-8")
    n_tokens_max = len(text_bytes) + 32
    tokens = (llama_token * n_tokens_max)()
    
    n = llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens_max, False, True)
    if n < 0:
        return []
    return [tokens[i] for i in range(n)]

def get_token_embeddings_gguf(model_path, cache_dir=None):
    """
    使用 gguf 库从 GGUF 读取 token_embd.weight。
    支持 F16/F32 和 Q8_0 量化格式
    使用缓存机制：首次读取后保存为 .npy 文件，后续直接加载缓存
    """
    # 生成缓存文件路径
    if cache_dir is None:
        cache_dir = os.path.dirname(model_path)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    cache_path = os.path.join(cache_dir, f"{model_name}.embd.npy")
    
    # 如果缓存存在且比模型新，直接加载
    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(model_path):
            return np.load(cache_path)
    
    # 从 GGUF 读取
    reader = gguf.GGUFReader(model_path, mode='r')
    
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            # GGML_TYPE_Q8_0 = 8
            if t.tensor_type == 8:
                # Q8_0 解量化
                # Block 结构: d (float16, 2字节) + qs (int8[32], 32字节) = 34 字节
                block_size_bytes = 34
                num_values_per_block = 32
                
                raw_data = t.data
                data_u8 = np.frombuffer(raw_data, dtype=np.uint8)
                n_blocks = data_u8.size // block_size_bytes
                
                blocks = data_u8.reshape(n_blocks, block_size_bytes)
                deltas = blocks[:, :2].view(np.float16).flatten()
                quants = blocks[:, 2:].view(np.int8)
                
                # value = delta * quant
                data = (deltas[:, np.newaxis] * quants).flatten().astype(np.float32).reshape(-1, 1024)
            else:
                # F16 或 F32
                data = t.data
                if data.dtype == np.float16:
                    data = data.astype(np.float32)
            
            # 保存缓存
            np.save(cache_path, data)
            return data
    
    return None

def token_to_bytes(vocab, token_id):
    """将 token 转换为原始字节 (用于 BPE 字节级 token)"""
    buf = ctypes.create_string_buffer(256)
    n = llama_token_to_piece(vocab, token_id, buf, ctypes.sizeof(buf), 0, True)
    if n > 0:
        return buf.raw[:n]
    return b""

class ByteDecoder:
    """
    字节级解码器，用于处理 BPE 拆分的 UTF-8 字符
    """
    def __init__(self):
        self.buffer = b""
    
    def decode(self, raw_bytes):
        self.buffer += raw_bytes
        result = ""
        while self.buffer:
            try:
                result += self.buffer.decode('utf-8')
                self.buffer = b""
                break
            except UnicodeDecodeError as e:
                if e.reason == 'unexpected end of data' or 'invalid continuation' in e.reason:
                    if e.start > 0:
                        result += self.buffer[:e.start].decode('utf-8', errors='replace')
                        self.buffer = self.buffer[e.start:]
                    break
                else:
                    result += self.buffer[:1].decode('utf-8', errors='replace')
                    self.buffer = self.buffer[1:]
        return result
    
    def flush(self):
        if self.buffer:
            result = self.buffer.decode('utf-8', errors='replace')
            self.buffer = b""
            return result
        return ""

def normalizer(audio, target_value=8192.0):
    """音频归一化处理"""
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean((audio * audio), dtype=np.float32), dtype=np.float32)
    audio *= (target_value / (rms + 1e-7))
    np.clip(audio, -32768.0, 32767.0, out=audio)
    return audio.astype(np.int16)

def load_audio(audio_path):
    """加载音频文件并转换为 16kHz PCM"""
    from pydub import AudioSegment
    
    audio = np.array(
        AudioSegment.from_file(audio_path)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .get_array_of_samples(),
        dtype=np.int16
    )
    
    if USE_NORMALIZER:
        audio = normalizer(audio, 8192.0)
    
    return audio

def encode_audio(audio, encoder_sess):
    """使用 ONNX Encoder 获取 LLM 嵌入和 CTC 特征"""
    import onnxruntime
    
    # Reshape: (1, 1, audio_len)
    audio_input = audio.reshape(1, 1, -1)
    query_embed = np.ones((1, 10, 1024), dtype=np.float32)
    
    in_names = [x.name for x in encoder_sess.get_inputs()]
    out_names = [x.name for x in encoder_sess.get_outputs()]
    
    # 输入: audio, query_embed
    # 输出: concat_embed, ids_len, enc_output
    input_feed = {
        in_names[0]: onnxruntime.OrtValue.ortvalue_from_numpy(audio_input, 'cpu', 0),
        in_names[1]: onnxruntime.OrtValue.ortvalue_from_numpy(query_embed, 'cpu', 0),
    }
    
    outputs = encoder_sess.run_with_ort_values(out_names, input_feed)
    
    audio_embd = outputs[0].numpy().squeeze(0) # [N, 1024]
    enc_output = outputs[2].numpy()           # [1, T, 512]
    
    return audio_embd, enc_output

# =========================================================================
# 高级封装函数 (High Level Functions)
# =========================================================================

def load_onnx_models():
    """步骤 1: 加载 ONNX 音频编码器和 CTC Head"""
    print("\n[1] 加载 ONNX Models (Encoder + CTC)...")
    import onnxruntime
    
    t_start = time.perf_counter()
    session_opts = onnxruntime.SessionOptions()
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    encoder_sess = onnxruntime.InferenceSession(
        ENCODER_ONNX_PATH, 
        sess_options=session_opts, 
        providers=['CPUExecutionProvider']
    )
    
    ctc_sess = onnxruntime.InferenceSession(
        CTC_ONNX_PATH, 
        sess_options=session_opts, 
        providers=['CPUExecutionProvider']
    )
    
    t_cost = time.perf_counter() - t_start
    print(f"    Encoder: {os.path.basename(ENCODER_ONNX_PATH)}")
    print(f"    CTC Head: {os.path.basename(CTC_ONNX_PATH)}")
    print(f"    ONNX Models Loaded in {t_cost:.2f}s")
    
    return encoder_sess, ctc_sess, t_cost

def load_gguf_model():
    """步骤 2: 加载 GGUF LLM 解码器"""
    print(f"\n[2] 加载 GGUF LLM Decoder")
    t_start = time.perf_counter()
    
    llama_backend_init()
    
    model_params = llama_model_default_params()
    model = llama_model_load_from_file(DECODER_GGUF_PATH.encode('utf-8'), model_params)
    
    t_cost = time.perf_counter() - t_start
    if not model:
        print("    ERROR: Failed to load model")
        return None, None, None, None, t_cost
        
    print(f"    Decoder: {os.path.basename(DECODER_GGUF_PATH)} (耗时: {t_cost:.2f}s)")
    
    vocab = llama_model_get_vocab(model)
    eos_token = llama_vocab_eos(vocab)
    
    return model, vocab, eos_token, t_cost

def load_embedding_weights():
    """步骤 3: 加载 token embedding 权重"""
    print("\n[3] 加载 token embedding 权重...")
    t_start = time.perf_counter()
    
    embedding_table = get_token_embeddings_gguf(DECODER_GGUF_PATH)
    if embedding_table is None:
        return None, 0
    
    t_read = time.perf_counter() - t_start
    print(f"    Embedding table: {embedding_table.shape} (耗时: {t_read*1000:.2f}ms)")
    return embedding_table, t_read

def process_audio_file(audio_path, encoder_sess):
    """步骤 4: 加载并编码音频"""
    print(f"\n[4] 加载音频和 Encode: {os.path.basename(audio_path)}")
    
    audio = load_audio(audio_path)
    audio_len = len(audio)
    print(f"    音频长度: {audio_len} samples ({audio_len/SAMPLE_RATE:.2f}s)")
    
    t_start = time.perf_counter()
    audio_embd, enc_output = encode_audio(audio, encoder_sess)
    t_cost = time.perf_counter() - t_start
    
    print(f"    耗时: {t_cost*1000:.2f}ms")
    return audio_embd, enc_output, audio_len, t_cost

def run_ctc_pass(ctc_sess, enc_output, ctc_id2token, hotword_list):
    """步骤 5: CTC Decode & Hotword Matching"""
    print("\n[5] CTC Decode")
    t_start = time.perf_counter()
    
    ctc_logits = ctc_sess.run(None, {"enc_output": enc_output})[0]
    ctc_text, ctc_results = decode_ctc(ctc_logits, ctc_id2token)
    
    t_cost = time.perf_counter() - t_start
    
    print(f"    CTC 识别结果：{ctc_text}")
    
    # 格式化时间戳输出
    ts_list = [float(f"{r['start']:.2f}") for r in ctc_results]
    print(f"    CTC 时间戳：{ts_list[:10]} ......")
    
    matched_hws = match_hotwords(ctc_text, hotword_list)
    print(f"    热词：{matched_hws}")
    
    return matched_hws, t_cost

def prepare_prompt_embeddings(vocab, embedding_table, matched_hotwords=None):
    """步骤 6: 生成 Prompt"""
    print("\n[6] 生成 Prompt")
    
    PREFIX_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    SUFFIX_PROMPT = "\n<|im_end|>\n<|im_start|>assistant"

    if matched_hotwords:
        hotwords = ", ".join(matched_hotwords)
        PREFIX_PROMPT += f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
        PREFIX_PROMPT += f"热词列表：[{hotwords}]\n"
    
    PREFIX_PROMPT += "\n语音转写：\n"
    
    prefix_tokens = text_to_tokens(vocab, PREFIX_PROMPT)
    suffix_tokens = text_to_tokens(vocab, SUFFIX_PROMPT)
    
    prefix_embd = embedding_table[prefix_tokens].astype(np.float32)
    suffix_embd = embedding_table[suffix_tokens].astype(np.float32)
    
    print(f"    Prefix: {len(prefix_tokens)} tokens")
    print(f"    Suffix: {len(suffix_tokens)} tokens")
    
    return prefix_embd, suffix_embd, len(prefix_tokens), len(suffix_tokens)

def setup_inference_context(model, full_embd, n_tokens_input):
    """步骤 7: 创建上下文并注入 Embedding"""
    print(f"\n[7] 注入 embeddings ({n_tokens_input} tokens)...")
    
    ctx_params = llama_context_default_params()
    ctx_params.n_ctx = 2048
    ctx_params.n_batch = 2048
    ctx_params.n_ubatch = N_UBATCH
    ctx_params.embeddings = False
    ctx_params.no_perf = True
    ctx_params.n_threads = N_THREADS
    ctx_params.n_threads_batch = N_THREADS_BATCH
    
    ctx = llama_init_from_model(model, ctx_params)
    if not ctx:
        return None, 0
    
    t_start = time.perf_counter()
    batch_embd = llama_batch_init(n_tokens_input, full_embd.shape[1], 1)
    
    # Prepare batch
    batch_embd.n_tokens = n_tokens_input
    batch_embd.token = ctypes.cast(None, ctypes.POINTER(llama_token))
    
    if not full_embd.flags['C_CONTIGUOUS']:
        full_embd = np.ascontiguousarray(full_embd)
    ctypes.memmove(batch_embd.embd, full_embd.ctypes.data, full_embd.nbytes)
    
    for k in range(n_tokens_input):
        batch_embd.pos[k] = k
        batch_embd.n_seq_id[k] = 1
        batch_embd.seq_id[k][0] = 0
        batch_embd.logits[k] = 1 if k == n_tokens_input - 1 else 0
        
    ret = llama_decode(ctx, batch_embd)
    llama_batch_free(batch_embd)
    
    if ret != 0:
        print(f"    ERROR: Decode failed (ret={ret})")
        llama_free(ctx)
        return None, 0

    t_cost = time.perf_counter() - t_start
    print(f"    注入耗时: {t_cost*1000:.2f}ms")
    return ctx, t_cost

def run_generation(ctx, vocab, eos_token, n_input_tokens):
    """步骤 8: 生成文本"""
    print(f"\n[8] 生成文本 (最大 {N_PREDICT} tokens)...")
    print("=" * 70)
    
    vocab_size = llama_vocab_n_tokens(vocab)
    batch_text = llama_batch_init(1, 0, 1)
    batch_text.n_tokens = 1
    
    generated_text = ""
    current_pos = n_input_tokens
    tokens_generated = 0
    decoder = ByteDecoder()
    
    t_gen_start = time.perf_counter()
    
    try:
        for _ in range(N_PREDICT):
            logits_ptr = llama_get_logits(ctx)
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            token_id = int(np.argmax(logits_arr))
            
            if token_id == eos_token or token_id in STOP_TOKENS:
                break
            
            raw_bytes = token_to_bytes(vocab, token_id)
            text_piece = decoder.decode(raw_bytes)
            
            print(text_piece, end="", flush=True)
            generated_text += text_piece
            tokens_generated += 1
            
            batch_text.token[0] = token_id
            batch_text.pos[0] = current_pos
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1
            
            if llama_decode(ctx, batch_text) != 0:
                break
            
            current_pos += 1
            
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    remaining = decoder.flush()
    if remaining:
        print(remaining, end="", flush=True)
        generated_text += remaining
    
    t_cost = time.perf_counter() - t_gen_start
    print("\n" + "=" * 70)
    
    llama_batch_free(batch_text)
    return generated_text, tokens_generated, t_cost

# =========================================================================
# 主函数 (Main)
# =========================================================================

def main():
    print("=" * 70)
    print("SenseVoice Hybrid ASR 推理 (CTC + LLM)")
    print("=" * 70)
    
    if QUIET_MODE:
        cb = LOG_CALLBACK(quiet_log_callback)
        llama_log_set(cb, None)
    
    # 加载 ONNX Models
    encoder_sess, ctc_sess, t_load_onnx = load_onnx_models()
    
    # 加载GGUF LLM Decoder
    model, vocab, eos_token, t_load_dec = load_gguf_model()
    if not model: return 1
    
    # 加载 Embedding 权重
    embedding_table, t_load_embd = load_embedding_weights()
    if embedding_table is None: return 1
    
    # 加载 CTC 词表
    ctc_id2token = load_ctc_tokens(TOKENS_PATH)

    print("\n" + "=" * 70)
    print("模型加载完成，准备处理音频...")
    print("=" * 70)
    
    # 音频编码
    audio_embd, enc_output, audio_len, t_encode_audio = process_audio_file(INPUT_AUDIO, encoder_sess)
    
    # CTC 解码
    matched_hws, t_ctc_cost = run_ctc_pass(ctc_sess, enc_output, ctc_id2token, DEFAULT_HOTWORDS)
    
    # 准备提示词
    prefix_embd, suffix_embd, n_prefix, n_suffix = prepare_prompt_embeddings(vocab, embedding_table, matched_hws)
    
    # 拼接 embd
    full_embd = np.concatenate([prefix_embd, audio_embd.astype(np.float32), suffix_embd], axis=0)
    n_input_tokens = full_embd.shape[0]
    
    # 注入 embd
    ctx, t_inject = setup_inference_context(model, full_embd, n_input_tokens)
    if not ctx: return 1
    
    # LLM 解码
    text, n_gen, t_gen = run_generation(ctx, vocab, eos_token, n_input_tokens)
    
    # 统计
    tps_out = n_gen / t_gen if t_gen > 0 else 0
    tps_in = n_input_tokens / t_inject if t_inject > 0 else 0
    t_total = t_encode_audio + t_ctc_cost + t_inject + t_gen
    
    print(f"\n[统计]")
    print(f"  音频长度: {audio_len/SAMPLE_RATE:.2f}s")
    print(f"  Decoder输入: {tps_in:5.0f} tokens/s (all: {n_input_tokens}, prefix:{n_prefix}, audio:{audio_embd.shape[0]}, suffix:{n_suffix})")
    print(f"  Decoder输出: {tps_out:5.0f} tokens/s (all: {n_gen})")

    print(f"\n[加载耗时]")
    print(f"  - ONNX加载： {t_load_onnx*1000:5.0f}ms")
    print(f"  - GGUF加载： {t_load_dec*1000:5.0f}ms")
    print(f"  - Embd读取： {t_load_embd*1000:5.0f}ms")
    
    print(f"\n[转录耗时]")
    print(f"  - 音频编码： {t_encode_audio*1000:5.0f}ms")
    print(f"  - CTC解码：  {t_ctc_cost*1000:5.0f}ms")
    print(f"  - LLM读取：  {t_inject*1000:5.0f}ms")
    print(f"  - LLM生成：  {t_gen*1000:5.0f}ms")
    print(f"  - 总耗时：   {t_total:5.2f}s")
    
    # 清理
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
    
    print("\n[完成]")
    return 0

if __name__ == "__main__":
    exit(main())


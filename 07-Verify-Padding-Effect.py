import torch
import numpy as np
import os
import sys
import torchaudio
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.getcwd())
import fun_asr_gguf.model_definition_paddable as model_paddable

def run_debug_test():
    # --- Configuration ---
    model_dir = r'./Fun-ASR-Nano-2512'
    weight_path = os.path.join(model_dir, "model.pt")
    
    SAMPLE_RATE = 16000
    N_MELS = 80
    NFFT_STFT = 400
    WINDOW_LENGTH = 400
    HOP_LENGTH = 160
    
    print("--- Running Padding Consistency Debug Test ---")

    # 1. Load Paddable Model
    print("[1] Loading Paddable PyTorch model...")
    hybrid = model_paddable.HybridSenseVoice(vocab_size=60515)
    hybrid.load_weights(weight_path)
    hybrid.eval()
    
    stft = model_paddable.STFT_Process(n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH).eval()
    fbank = (torchaudio.functional.melscale_fbanks(NFFT_STFT // 2 + 1, 20, SAMPLE_RATE // 2, N_MELS, SAMPLE_RATE, None,'htk')).transpose(0, 1).unsqueeze(0)
    
    wrapper = model_paddable.EncoderExportWrapperPaddable(hybrid, stft, fbank).eval()

    # 2. Prepare Inputs
    print("[2] Preparing audio data...")
    # Fixed seed for reproducibility across runs
    torch.manual_seed(42)
    s5_samples = SAMPLE_RATE * 5
    audio_5s_torch = torch.randn(1, 1, s5_samples)
    ilens_5s = torch.tensor([s5_samples], dtype=torch.long)

    # 3. Run 5s Inference (Logs will go to logs/model-debug.log)
    print("[3] Running 5s audio inference...")
    with torch.no_grad():
        wrapper(audio_5s_torch, ilens_5s)
    
    # 4. Run 30s-padded Inference
    print("[4] Running 30s-padded audio inference...")
    s30_samples = SAMPLE_RATE * 30
    audio_30s_torch = F.pad(audio_5s_torch, (0, s30_samples - s5_samples), value=0.0)
    ilens_30s = torch.tensor([s5_samples], dtype=torch.long) 
    
    with torch.no_grad():
        wrapper(audio_30s_torch, ilens_30s)

    print("\n[Done] Results logged to logs/model-debug.log")
    print("Check the file to see where the stats begin to diverge.")

if __name__ == "__main__":
    run_debug_test()

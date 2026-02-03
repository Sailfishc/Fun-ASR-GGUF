import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
import fun_asr_gguf.model_definition_paddable as model_paddable

def verify_final_consistency():
    # --- Configuration ---
    model_dir = r'./Fun-ASR-Nano-2512'
    weight_path = os.path.join(model_dir, "model.pt")
    
    SAMPLE_RATE = 16000
    N_MELS = 80
    NFFT_STFT = 400
    WINDOW_LENGTH = 400
    HOP_LENGTH = 160
    
    print("--- Final PyTorch Padding Consistency Test: 5s vs. 30s-Padded ---")
    print(f"Strategy: Replicate Padding + Length-Aware Masking")

    # 1. Load Model
    hybrid = model_paddable.HybridSenseVoice(vocab_size=60515)
    hybrid.load_weights(weight_path)
    hybrid.eval()
    
    stft = model_paddable.STFT_Process(n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH).eval()
    fbank = (torchaudio.functional.melscale_fbanks(NFFT_STFT // 2 + 1, 20, SAMPLE_RATE // 2, N_MELS, SAMPLE_RATE, None,'htk')).transpose(0, 1).unsqueeze(0)
    
    wrapper = model_paddable.EncoderExportWrapperPaddable(hybrid, stft, fbank).eval()

    # 2. Prepare Data
    torch.manual_seed(42)
    s5_samples = SAMPLE_RATE * 5
    audio_5s = torch.randn(1, 1, s5_samples)
    ilens_5s = torch.tensor([s5_samples], dtype=torch.long)

    s30_samples = SAMPLE_RATE * 30
    audio_30s = F.pad(audio_5s, (0, s30_samples - s5_samples), value=0.0)
    ilens_30s = torch.tensor([s5_samples], dtype=torch.long)

    # 3. Inference
    print("\n[Inferencing] Running both models...")
    with torch.no_grad():
        _, output_5s = wrapper(audio_5s, ilens_5s)
        _, output_30s = wrapper(audio_30s, ilens_30s)

    # 4. Compare
    print("\n--- Numerical Analysis (Final Output) ---")
    
    diff = torch.abs(output_5s - output_30s)
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    
    # Cosine Similarity
    # Flatten to (N, Dim) to calculate per-frame similarity, then mean
    cos = torch.nn.CosineSimilarity(dim=-1)
    sim = cos(output_5s, output_30s).mean().item()
    
    # Check shape consistency
    print(f"  Shape 5s:  {list(output_5s.shape)}")
    print(f"  Shape 30s: {list(output_30s.shape)}")
    
    print(f"  Max Absolute Error: {max_err:.6e}")
    print(f"  Mean Absolute Error: {mean_err:.6e}")
    print(f"  Cosine Similarity:   {sim:.10f}") # Ideally 1.0000000000

    # Tolerance check
    atol = 1e-3
    if max_err < atol and sim > 0.999999:
        print(f"\n✅ PASS: The models are numerically consistent!")
        print(f"The padding effect has been successfully mitigated (Similarity > 0.999999).")
    else:
        print(f"\n❌ FAIL: Discrepancy detected exceeds desired range.")

if __name__ == "__main__":
    verify_final_consistency()

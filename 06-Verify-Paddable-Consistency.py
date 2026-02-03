import torch
import numpy as np
import os
import sys
import torchaudio

# Add project root to path
sys.path.append(os.getcwd())
import fun_asr_gguf.model_definition as model_old
import fun_asr_gguf.model_definition_paddable as model_new

def verify_pytorch_versions():
    # --- Configuration ---
    model_dir = r'./Fun-ASR-Nano-2512'
    weight_path = os.path.join(model_dir, "model.pt")
    
    SAMPLE_RATE = 16000
    N_MELS = 80
    NFFT_STFT = 400
    WINDOW_LENGTH = 400
    HOP_LENGTH = 160
    
    print("--- PyTorch Internal Verification: Original vs. Paddable Definition ---")

    # 1. Load Original Model (Base)
    print("[1] Loading Original PyTorch model...")
    hybrid_old = model_old.HybridSenseVoice(vocab_size=60515)
    hybrid_old.load_weights(weight_path)
    hybrid_old.eval()
    
    stft_old = model_old.STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH).eval()
    fbank = (torchaudio.functional.melscale_fbanks(NFFT_STFT // 2 + 1, 20, SAMPLE_RATE // 2, N_MELS, SAMPLE_RATE, None,'htk')).transpose(0, 1).unsqueeze(0)
    
    wrapper_old = model_old.EncoderExportWrapper(hybrid_old, stft_old, fbank).eval()

    # 2. Load New Paddable Model
    print("[2] Loading Paddable PyTorch model...")
    hybrid_new = model_new.HybridSenseVoice(vocab_size=60515)
    hybrid_new.load_weights(weight_path) # Use SAME weights
    hybrid_new.eval()
    
    stft_new = model_new.STFT_Process(n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH).eval()
    
    wrapper_new = model_new.EncoderExportWrapperPaddable(hybrid_new, stft_new, fbank).eval()

    # 3. Prepare Same Random Input (5 seconds)
    print("[3] Preparing random 5s audio...")
    samples = SAMPLE_RATE * 5
    audio_np = np.random.randn(1, 1, samples).astype(np.float32)
    audio_torch = torch.from_numpy(audio_np)
    ilens = torch.tensor([samples], dtype=torch.long)

    # 4. Run Original Inference
    print("[4] Running Original inference...")
    with torch.no_grad():
        out_old_enc, out_old_adapt = wrapper_old(audio_torch)
    
    # 5. Run New Paddable Inference
    print("[5] Running Paddable inference...")
    with torch.no_grad():
        # Pass ilens to the new wrapper
        out_new_enc, out_new_adapt = wrapper_new(audio_torch, ilens)

    # 6. Numerical Comparison
    print("\n--- Numerical Comparison (PyTorch Only) ---")
    
    def compare_tensors(t1, t2, name):
        diff = torch.abs(t1 - t2)
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        print(f"  [{name}]")
        print(f"    Shape: old={t1.shape}, new={t2.shape}")
        print(f"    Max Error:  {max_err:.6e}")
        print(f"    Mean Error: {mean_err:.6e}")
        
        if torch.allclose(t1, t2, atol=1e-5):
            print(f"    ✅ {name} matches (atol=1e-5)")
        else:
            print(f"    ❌ {name} mismatch detected!")

    compare_tensors(out_old_enc, out_new_enc, "Encoder Output")
    compare_tensors(out_old_adapt, out_new_adapt, "Adaptor Output")

if __name__ == "__main__":
    verify_pytorch_versions()

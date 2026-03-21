"""
run_mp3_va_auto_fixed.py

Auto-inspects checkpoint to set patch/frame sizes, builds ASTRegressor from trainv2.py,
preprocesses an MP3 into log-mel windows, converts windows into patches, runs inference,
and prints per-window & aggregated Valence/Arousal means.

Edit MP3_PATH and CHECKPOINT_PATH at the top if needed, then run:
    python run_mp3_va_auto_fixed.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import v2.trainv2 as trainv2

# ========== USER VARIABLES ==========
MP3_PATH = "songs/psychosocial.mp3"  # ← set path to input MP3
CHECKPOINT_PATH = "best_ast_va(409val).pt"  # ← set path to checkpoint
DEVICE = "cpu"  # "cpu", "cuda", or "mps"

# Audio / preprocessing defaults (tweak if needed)
SAMPLE_RATE = 32000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 1024

# Windowing (you can change WINDOW_SEC and HOP_SEC)
WINDOW_SEC = 5.0       # seconds per input window
HOP_SEC = 2.5          # seconds between successive windows

EPS = 1e-6
# =====================================

def inspect_checkpoint_for_patch_settings(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        patch_dim = None
        max_patches = None
        # try direct names
        for k in sd.keys():
            if k.endswith("patch_proj.weight"):
                w = sd[k]
                patch_dim = int(w.shape[1])
            if "pos_embed" in k:
                pe = sd[k]
                max_patches = int(pe.shape[1]) - 1
            if patch_dim is not None and max_patches is not None:
                break
        return patch_dim, max_patches
    else:
        return None, None

def build_model_from_checkpoint_settings(patch_dim, max_patches, device):
    if patch_dim is None or max_patches is None:
        raise RuntimeError("Could not infer patch_dim/max_patches from checkpoint. Edit script manually.")
    if not hasattr(trainv2, "ASTRegressor"):
        raise RuntimeError("trainv2.ASTRegressor not found. Ensure trainv2.py exports ASTRegressor class.")
    ASTRegressor = getattr(trainv2, "ASTRegressor")
    model = ASTRegressor(patch_area=int(patch_dim), max_patches=int(max_patches))
    return model.to(device)

def load_checkpoint_into_model(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        for k in ("model_state_dict","state_dict","state_dict_ema","state_dict_best","model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                try:
                    model.load_state_dict(sd)
                    return model.to(device)
                except Exception:
                    model.load_state_dict(sd, strict=False)
                    return model.to(device)
        try:
            model.load_state_dict(ckpt)
            return model.to(device)
        except Exception:
            pass
        if "model" in ckpt and isinstance(ckpt["model"], nn.Module):
            return ckpt["model"].to(device)
    else:
        if isinstance(ckpt, nn.Module):
            return ckpt.to(device)
    model.load_state_dict(ckpt, strict=False)
    return model.to(device)

def load_log_mel(mp3_path):
    audio, sr = librosa.load(mp3_path, sr=None, mono=True)
    if sr != SAMPLE_RATE:
        # Use explicit keyword args to avoid signature mismatch across librosa versions
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, n_mels=N_MELS, power=2.0
    )
    log_mel = np.log(np.maximum(mel, EPS))
    mean = log_mel.mean(axis=1, keepdims=True)
    std = log_mel.std(axis=1, keepdims=True) + 1e-6
    return ((log_mel - mean) / std).astype(np.float32)

def seconds_to_frames(sec):
    return int(round(sec * SAMPLE_RATE / HOP_LENGTH))

def windowize_mel(mel, window_sec, hop_sec):
    win_frames = seconds_to_frames(window_sec)
    hop_frames = seconds_to_frames(hop_sec)
    windows = []
    total = mel.shape[1]
    for start in range(0, total - win_frames + 1, hop_frames):
        windows.append(mel[:, start:start+win_frames])
    if not windows:
        pad = np.zeros((mel.shape[0], win_frames), dtype=mel.dtype)
        pad[:, :mel.shape[1]] = mel
        windows = [pad]
    return np.stack(windows, axis=0)

def window_to_patches(window, patch_size_frames, overlap_frames):
    n_mels, frames = window.shape
    step = max(1, patch_size_frames - overlap_frames)
    patches = []
    for start in range(0, frames - patch_size_frames + 1, step):
        patch = window[:, start:start+patch_size_frames]
        patches.append(patch.reshape(-1))
    if not patches:
        pad = np.zeros((n_mels, patch_size_frames), dtype=window.dtype)
        pad[:, :frames] = window
        patches = [pad.reshape(-1)]
    return np.stack(patches, axis=0).astype(np.float32)

def windows_to_model_inputs(windows, patch_size_frames, overlap_frames, device):
    all_patches = [window_to_patches(w, patch_size_frames, overlap_frames) for w in windows]
    n_windows = len(all_patches)
    n_patches = max(p.shape[0] for p in all_patches)
    patch_dim = all_patches[0].shape[1]
    padded = np.zeros((n_windows, n_patches, patch_dim), dtype=np.float32)
    padding_masks = np.ones((n_windows, n_patches), dtype=bool)
    for i,p in enumerate(all_patches):
        cnt = p.shape[0]
        padded[i,:cnt,:] = p
        padding_masks[i,:cnt] = False
    inputs = torch.from_numpy(padded).to(device)
    padding_masks = torch.from_numpy(padding_masks).to(device=device, dtype=torch.bool)
    return inputs, padding_masks

def predict_va(mp3_path, model, device, window_sec=WINDOW_SEC, hop_sec=HOP_SEC, patch_size_frames=None, overlap_frames=None):
    mel = load_log_mel(mp3_path)
    windows = windowize_mel(mel, window_sec, hop_sec)
    inputs, padding_masks = windows_to_model_inputs(windows, patch_size_frames, overlap_frames, device)
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(inputs, padding_mask=padding_masks)
        except TypeError:
            outputs = model(inputs)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        out_np = outputs.detach().cpu().numpy()
    val_means = out_np[:, 0]
    aro_means = out_np[:, 2]
    return val_means, aro_means, out_np

def main():
    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
    assert os.path.exists(MP3_PATH), f"MP3 not found: {MP3_PATH}"

    patch_dim, max_patches = inspect_checkpoint_for_patch_settings(CHECKPOINT_PATH)
    if patch_dim is None:
        raise RuntimeError("Could not infer patch_dim from checkpoint. Please open the checkpoint and provide patch_dim.")
    if max_patches is None:
        raise RuntimeError("Could not infer max_patches (pos_embed length) from checkpoint. Please open the checkpoint and provide it.")

    print(f"Inferred patch_dim={patch_dim}, max_patches={max_patches}")

    if patch_dim % N_MELS != 0:
        raise RuntimeError(f"patch_dim {patch_dim} is not divisible by N_MELS {N_MELS}. Adjust N_MELS or inspect checkpoint.")
    patch_size_frames = patch_dim // N_MELS
    overlap_frames = max(0, patch_size_frames - 1)
    print(f"Inferred patch_size_frames={patch_size_frames}, using overlap_frames={overlap_frames}")

    if DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = build_model_from_checkpoint_settings(patch_dim, max_patches, device)

    model = load_checkpoint_into_model(model, CHECKPOINT_PATH, device)
    print("Model loaded.")

    val_means, aro_means, raw = predict_va(MP3_PATH, model, device,
                                           window_sec=WINDOW_SEC, hop_sec=HOP_SEC,
                                           patch_size_frames=patch_size_frames, overlap_frames=overlap_frames)

    print("\nPer-window valence, arousal:")
    for i,(v,a) in enumerate(zip(val_means, aro_means)):
        print(f"{i:03d}: valence_mean={float(v):.4f}, arousal_mean={float(a):.4f}")

    print("\nAggregated means:")
    print(f"Valence mean = {float(val_means.mean()):.4f}")
    print(f"Arousal mean = {float(aro_means.mean()):.4f}")

    print("\nSCALED means:")
    print(f"Valence mean = {float((val_means.mean() - 4)/8):.4f}")
    print(f"Arousal mean = {float((aro_means.mean() - 4)/8):.4f}")

if __name__ == "__main__":
    main()

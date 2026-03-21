#!/usr/bin/env python3
"""
test_ast_on_mp3_nocli_full.py

Non-CLI script to run a saved AST regressor checkpoint on a single MP3.
Put your hyperparameters / paths at the top and run the script (python3 test_ast_on_mp3_nocli_full.py).

This version includes a robust batching/inference helper and diagnostics to
catch the "all windows produce same output / zero std" problem.
"""

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# audio
import librosa

# Try to reuse your training file's model & metadata loader
try:
    from v3.trainv3 import ASTRegressor, load_metadata
except Exception as e:
    raise ImportError(
        "Could not import ASTRegressor/load_metadata from trainv3.py. "
        "Make sure the training file is in the same directory or adjust the import. "
        f"(Error: {e})"
    )

# ---------------- USER CONFIGURATION (edit these) ----------------
MP3_PATH = "songs/understand.mp3"                         # path to mp3 you want to test
CKPT_PATH = "ast_checkpoints/ast_best.pth"                # path to checkpoint (ast_best.pth)
METADATA_PATH = "dataspectrogram/_metadata.json"          # path to metadata JSON used in training

# model hyperparameters (must match training)
EMBED_DIM = 768
N_LAYERS = 6
N_HEADS = 8
MLP_DIM = 2048
PATCH_OVERLAP = 6
PATCH_SIZE = 16     # both freq & time patch size (training used PATCH_SIZE)
# inference batching / windowing
WINDOW_SECONDS = 4.0
WINDOW_OVERLAP = 2.0
BATCH_SIZE = 8

# device override: set to "cpu" or "cuda" (or None to auto-detect)
DEVICE_STR = None
# -----------------------------------------------------------------


def compute_mel_db(y: np.ndarray, sr: int, n_mels: int, hop_length: int, n_fft: int = 2048, power: float = 2.0):
    """
    Compute mel spectrogram and convert to dB (power_to_db) to match training .npy stats.
    Returns shape (n_mels, n_frames) float32
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        power=power
    )
    mel_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return mel_db


def frames_for_window(window_seconds: float, sr: int, hop_length: int):
    hop_ms = hop_length / float(sr) * 1000.0
    window_ms = int(round(window_seconds * 1000.0))
    target_frames = max(1, int(round(window_ms / max(1e-6, hop_ms))))
    return target_frames


def make_windows_from_mel(mel: np.ndarray, window_seconds: float, overlap_seconds: float, sr: int, hop_length: int):
    """
    Slice mel spectrogram into windows of window_seconds with overlap_seconds
    Returns list of mel windows shaped (n_mels, target_frames)
    """
    n_mels, n_frames = mel.shape
    hop_ms = hop_length / float(sr) * 1000.0
    window_ms = int(round(window_seconds * 1000.0))
    stride_ms = int(round((window_seconds - overlap_seconds) * 1000.0)) if (overlap_seconds < window_seconds) else window_ms
    if stride_ms <= 0:
        stride_ms = window_ms
    target_frames = frames_for_window(window_seconds, sr, hop_length)

    stride_frames = max(1, int(round(stride_ms / hop_ms)))

    windows = []
    cur = 0
    while cur < n_frames:
        end = cur + target_frames
        win = mel[:, cur:end]
        if win.shape[1] < target_frames:
            pad = target_frames - win.shape[1]
            win = np.pad(win, ((0, 0), (0, pad)), mode='constant', constant_values=0.0)
        windows.append(win)
        if end >= n_frames:
            break
        cur = cur + stride_frames
    return windows


def run_inference_on_windows(windows: List[np.ndarray], model: torch.nn.Module, device: torch.device, batch_size: int = 8):
    """
    Robust inference helper that:
      - converts windows (list of n_mels x frames) -> torch tensor (B,1,n_mels,frames)
      - manual batching (no DataLoader) to avoid tuple/packing bugs
      - prints simple diagnostics to ensure windows differ
    Returns numpy array shape (num_windows, 2)
    """
    if len(windows) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Quick diagnostics: check a few windows' stats to ensure they're not identical
    def _win_stats(w):
        return (float(w.min()), float(w.max()), float(w.mean()), float(w.std()))

    print("Window diagnostics (first 6 windows):")
    for i, w in enumerate(windows[:6]):
        mn, mx, mean, std = _win_stats(w)
        print(f"  win {i:03d}: shape={w.shape}, min={mn:.6f}, max={mx:.6f}, mean={mean:.6f}, std={std:.6f}")

    # Check if all windows are exactly equal (fast path)
    eq0 = True
    ref = windows[0]
    for w in windows[1:]:
        if w.shape != ref.shape or not np.allclose(w, ref, atol=1e-6):
            eq0 = False
            break
    if eq0:
        print("Warning: all windows are (nearly) identical! This explains identical predictions. Check windowing code.")
    else:
        print("Windows appear to differ (not identical). Proceeding with batching inference.")

    # stack and convert once (B,1,n_mels,frames)
    X = np.stack(windows, axis=0).astype(np.float32)
    X = X[:, None, :, :]
    X_t = torch.from_numpy(X)
    total = X_t.shape[0]

    preds_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, total, batch_size):
            batch = X_t[i:i + batch_size].to(device, non_blocking=True)
            out = model(batch)  # (B,2)
            preds_list.append(out.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    if preds.shape[0] != total:
        print(f"Warning: predictions length {preds.shape[0]} != windows {total}")
    return preds


def load_checkpoint(ckpt_path: str, device: torch.device, n_mels: int, patch_freq: int, patch_time: int, embed_dim: int, n_layers: int, n_heads: int, mlp_dim: int, overlap: int):
    """
    Load model structure and weights (using ASTRegressor signature).
    """
    ck = torch.load(ckpt_path, map_location=device)
    model = ASTRegressor(n_mels=n_mels, patch_freq=patch_freq, patch_time=patch_time,
                         embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
                         mlp_dim=mlp_dim, overlap=overlap)
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
    else:
        state = ck
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    return model.to(device)


def main():
    # device
    if DEVICE_STR:
        device = torch.device(DEVICE_STR)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # paths
    mp3_path = Path(MP3_PATH)
    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 not found: {mp3_path}")
    ckpt_path = Path(CKPT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load metadata
    metadata = load_metadata(METADATA_PATH)
    sr = int(metadata.get("sr", 16000))
    hop_length = int(metadata.get("hop_length", 160))
    n_mels = int(metadata.get("n_mels", 128))
    print(f"Metadata: sr={sr}, hop_length={hop_length}, n_mels={n_mels}")

    # load audio
    print(f"Loading audio '{mp3_path}' (sr={sr}) ...")
    y, _ = librosa.load(str(mp3_path), sr=sr, mono=True)

    # compute mel spectrogram (dB: matches training .npy)
    print("Computing mel spectrogram (power_to_db)...")
    mel = compute_mel_db(y, sr=sr, n_mels=n_mels, hop_length=hop_length)

    # debug: print overall mel stats
    print("Mel spectrogram stats: shape", mel.shape, "min/max/mean/std:",
          float(mel.min()), float(mel.max()), float(mel.mean()), float(mel.std()))

    # make windows
    print("Slicing into windows...")
    windows = make_windows_from_mel(mel, window_seconds=WINDOW_SECONDS, overlap_seconds=WINDOW_OVERLAP, sr=sr, hop_length=hop_length)
    if len(windows) == 0:
        raise RuntimeError("No windows produced from audio. Check length and window parameters.")
    print(f"Produced {len(windows)} windows of shape {windows[0].shape}")

    # load model
    print("Loading model checkpoint:", ckpt_path)
    model = load_checkpoint(str(ckpt_path), device=device, n_mels=n_mels,
                            patch_freq=PATCH_SIZE, patch_time=PATCH_SIZE,
                            embed_dim=EMBED_DIM, n_layers=N_LAYERS,
                            n_heads=N_HEADS, mlp_dim=MLP_DIM, overlap=PATCH_OVERLAP)

    # small forward-test on first two windows to sanity-check model differs
    if len(windows) >= 2:
        model.eval()
        with torch.no_grad():
            a = torch.from_numpy(windows[0][None, None, :, :].astype(np.float32)).to(device)
            b = torch.from_numpy(windows[1][None, None, :, :].astype(np.float32)).to(device)
            out0 = model(a).cpu().numpy()
            out1 = model(b).cpu().numpy()
        print("Sample single-window outputs (first two windows):")
        print("  out0:", out0.flatten().tolist())
        print("  out1:", out1.flatten().tolist())
    else:
        print("Less than 2 windows; skipping single-window forward-check.")

    # inference
    print("Running inference on windows ...")
    preds = run_inference_on_windows(windows, model, device=device, batch_size=BATCH_SIZE)

    if preds.shape[0] == 0:
        print("No predictions were made.")
        return

    # Print per-window and averaged predictions
    print("\nPer-window predictions (valence, arousal):")
    for i, p in enumerate(preds):
        print(f"  window {i:03d}:  valence={p[0]:.5f}, arousal={p[1]:.5f}")

    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    print("\nAggregated prediction (mean ± std):")
    print(f"  valence  = {mean_pred[0]:.5f} ± {std_pred[0]:.5f}")
    print(f"  arousal  = {mean_pred[1]:.5f} ± {std_pred[1]:.5f}")

    # write out to file next to mp3
    out_path = mp3_path.with_suffix(".ast_pred.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Per-window valence arousal predictions (valence, arousal)\n")
        for i, p in enumerate(preds):
            f.write(f"{i}\t{p[0]:.8f}\t{p[1]:.8f}\n")
        f.write(f"\n# Aggregated (mean,std)\n")
        f.write(f"valence_mean\t{mean_pred[0]:.8f}\n")
        f.write(f"valence_std\t{std_pred[0]:.8f}\n")
        f.write(f"arousal_mean\t{mean_pred[1]:.8f}\n")
        f.write(f"arousal_std\t{std_pred[1]:.8f}\n")
    print(f"\nWrote predictions to: {out_path}")


if __name__ == "__main__":
    main()

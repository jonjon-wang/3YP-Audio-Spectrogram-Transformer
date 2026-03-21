# make_mels.py
# Edit INPUT_DIR and OUTPUT_DIR below, then run:
# python make_mels.py
#
# Produces: .npy files (shape = (n_mels, T), dtype=float32) and out_dir/_metadata.json

import sys
import os
import json
from pathlib import Path
from typing import Tuple
import numpy as np
from tqdm import tqdm
import librosa

# ------------------ USER SETTABLE PATHS ------------------
# Set these two variables to point to your input MP3 folder and desired output folder.
# Example:
# INPUT_DIR = "/home/you/datasets/mp3s"
# OUTPUT_DIR = "/home/you/datasets/mels"
INPUT_DIR = "data/datamp3"   # <-- change this
OUTPUT_DIR = "data/dataspectrogram" # <-- change this
# -------------------------------------------------------

# ------------------ CONFIG ------------------
SR = 22050            # target sample rate (change if you prefer)
N_MELS = 128
N_FFT = int(0.025 * SR)   # 25 ms window
HOP_LENGTH = int(0.010 * SR)  # 10 ms hop (=> ~100 frames/sec)
POWER = 2.0           # power spectrogram (energy)
REF = 1.0             # librosa.power_to_db ref
EXCLUDE_PREFIX = "_"  # files starting with this in output are ignored for stats
# --------------------------------------------

def compute_log_mel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y= y / (np.sqrt(np.mean(y**2)) + 1e-8),
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=POWER
    )
    mel_db = librosa.power_to_db(mel, ref=REF)
    return mel_db.astype(np.float32)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def safe_rel_path(input_path: Path, input_root: Path) -> Path:
    try:
        return input_path.relative_to(input_root)
    except Exception:
        return Path(input_path.name)

def process_file(in_path: Path, out_path: Path):
    ensure_dir(out_path.parent)
    y, sr = librosa.load(str(in_path), sr=SR, mono=True)
    mel_db = compute_log_mel(y, sr)
    np.save(str(out_path), mel_db)
    return mel_db.shape

def iter_mp3_files(root: Path):
    for p in root.rglob("*.mp3"):
        yield p

def streaming_mean_std(sum_x, sum_x2, n):
    mean = sum_x / n
    var = (sum_x2 / n) - (mean ** 2)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, std

def compute_global_mean_std(out_root: Path):
    sum_x = np.zeros((N_MELS,), dtype=np.float64)
    sum_x2 = np.zeros((N_MELS,), dtype=np.float64)
    total_frames = 0

    npy_paths = list(out_root.rglob("*.npy"))
    npy_paths = [p for p in npy_paths if p.name != "_metadata.json" and not p.name.startswith(EXCLUDE_PREFIX)]

    for p in tqdm(npy_paths, desc="Computing global stats"):
        arr = np.load(str(p))  # shape (n_mels, T)
        s = arr.sum(axis=1).astype(np.float64)
        s2 = (arr.astype(np.float64) ** 2).sum(axis=1)
        frames = arr.shape[1]
        sum_x += s
        sum_x2 += s2
        total_frames += frames

    if total_frames == 0:
        raise RuntimeError("No .npy spectrogram files found in output folder to compute stats.")

    mean, std = streaming_mean_std(sum_x, sum_x2, total_frames)
    return mean.astype(np.float32), std.astype(np.float32), int(total_frames)

def process_folder(input_dir: str, output_dir: str, preserve_subdirs: bool = True):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    ensure_dir(out_root)

    mp3_files = list(iter_mp3_files(in_root))
    if not mp3_files:
        print("No mp3 files found in", in_root)
        return

    print(f"Found {len(mp3_files)} mp3 files. Starting conversion ...")
    shapes = {}
    for in_path in tqdm(mp3_files, desc="Converting mp3 -> mel (dB)"):
        rel = safe_rel_path(in_path.with_suffix(".npy"), in_root) if preserve_subdirs else Path(in_path.stem + ".npy")
        out_path = out_root / rel
        try:
            shp = process_file(in_path, out_path)
            shapes[str(out_path.relative_to(out_root))] = shp
        except Exception as e:
            print(f"ERROR processing {in_path}: {e}")

    mean, std, total_frames = compute_global_mean_std(out_root)

    metadata = {
        "sr": SR,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "power": POWER,
        "ref": REF,
        "num_files": len(shapes),
        "total_frames": total_frames,
        "global_mean_per_bin": mean.tolist(),
        "global_std_per_bin": std.tolist(),
        "file_shapes": {k: list(v) for k, v in shapes.items()}
    }

    meta_path = out_root / "_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done. Saved spectrograms to", out_root)
    print("Metadata saved to", meta_path)
    return metadata

def main():
    # Use the top-of-file variables
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    if not input_dir or not output_dir:
        print("Please set INPUT_DIR and OUTPUT_DIR at the top of the file before running.")
        return

    # Basic checks for user friendliness
    in_root = Path(input_dir)
    if not in_root.exists() or not in_root.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        return

    out_root = Path(output_dir)
    ensure_dir(out_root)

    try:
        process_folder(input_dir, output_dir)
    except Exception as e:
        print("Conversion failed:", e)

if __name__ == "__main__":
    main()

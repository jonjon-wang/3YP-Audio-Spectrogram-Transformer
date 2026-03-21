#!/usr/bin/env python3
"""
trainjj_fixed_full.py

Full training script for Audio Spectrogram Transformer (valence/arousal regression).
This file contains:
 - metadata & annotation loading
 - spectrogram chunk loader (returns writable arrays)
 - a dataset that enforces fixed frames per sample (pads/truncates to window)
 - AST-like transformer regressor
 - training loop with optional AMP, grad clipping, ReduceLROnPlateau scheduler
 - normalization of id_periods input (numpy array or dict)
"""

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Optional, Any, Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# ----------HYPERPARAMETERS ----------
# folders / paths
SPECTROGRAM_DIR = "dataspectrogram"   # folder containing .npy files and _metadata.json
METADATA_PATH = "dataspectrogram/_metadata.json"  # path to metadata JSON file
AROUSAL_CSV = "dynamic_annotations/arousal.csv"      # path to arousal CSV annotations
VALENCE_CSV = "dynamic_annotations/valence.csv"      # path to valence CSV annotations

# model / training
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
NUM_WORKERS = min(((os.cpu_count() or 5) - 4), 14)

WINDOW_SECONDS = 4.0   # crop length in seconds
WINDOW_OVERLAP = 2.0   # overlap between crops in seconds

N_MELS = 128
PATCH_SIZE = 16
PATCH_OVERLAP = 6
MODEL_DIM = 768
NUM_LAYERS = 6  # paper uses 12
NUM_HEADS = 8   # paper uses 12
MLP_DIM = 2048
MLP_HEAD_FLOOR = 128
DROPOUT = 0.1

# misc training tweaks
USE_GPU = True
USE_AMP = False            # enable mixed precision (only for CUDA)
GRAD_CLIP_NORM = 1.0     # clip gradients to this norm (set None to disable)
SCHEDULER_PATIENCE = 2   # for ReduceLROnPlateau
SEED = 42

# ---------- SEED AND DEVICE SETUP ----------
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available() and USE_GPU:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ---------- GLOBALS ----------
METADATA: dict = {}
_DATA: Dict[int, Dict[str, np.ndarray]] = {}

# ---------- UTILS ----------
def _parse_ms_col_to_seconds(colname: str) -> float:
    m = re.search(r'(-?\d+\.?\d*)\s*ms', colname)
    if m:
        return float(m.group(1)) / 1000.0
    m = re.search(r'(-?\d+\.?\d*)$', colname)
    if m:
        v = float(m.group(1))
        return v / 1000.0 if v > 1000 else v
    raise ValueError(f"Can't parse time from column '{colname}'")

def _to_ms_int(seconds: float) -> int:
    return int(round(seconds * 1000.0))

def _ms_to_frame_indices(start_ms: int, end_ms: int, sr: int, hop_length: int) -> Tuple[int, int]:
    start_s = start_ms / 1000.0
    end_s   = end_ms   / 1000.0
    start_frame = int(np.floor(start_s * sr / hop_length))
    end_frame   = int(np.ceil (end_s   * sr / hop_length))
    if start_frame < 0:
        start_frame = 0
    if end_frame < 0:
        end_frame = 0
    return start_frame, end_frame

# ---------- METADATA LOADER ----------
def load_metadata(metadata_path: str = METADATA_PATH) -> dict:
    global METADATA
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        METADATA = json.load(f)
    # Basic defaults/validation
    if "sr" not in METADATA:
        raise KeyError("METADATA missing 'sr'")
    if "hop_length" not in METADATA:
        raise KeyError("METADATA missing 'hop_length'")
    if "file_shapes" not in METADATA:
        METADATA["file_shapes"] = {}
    if "n_mels" not in METADATA:
        METADATA["n_mels"] = 128
    return METADATA

# ---------- ANNOTATION PREPARATION ----------
def prepare_and_index_ms(valence_csv: str, arousal_csv: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Build global _DATA and return id_periods_map mapping int(song_id) -> list of period dicts:
      period dict: {"start_ms", "end_ms", "valence_mean", "arousal_mean"}
    """
    global _DATA
    _DATA = {}

    vdf = pd.read_csv(valence_csv)
    adf = pd.read_csv(arousal_csv)

    id_col = 'song_id' if 'song_id' in vdf.columns else vdf.columns[0]
    adf_id_col = 'song_id' if 'song_id' in adf.columns else adf.columns[0]

    # parse time columns -> seconds
    time_cols = [c for c in vdf.columns if c != id_col]
    secs = []
    valid_cols = []
    for c in time_cols:
        try:
            s = _parse_ms_col_to_seconds(c)
            secs.append(s)
            valid_cols.append(c)
        except Exception:
            continue

    if not valid_cols:
        raise RuntimeError("No valid time columns found in valence CSV")

    ms_times = np.array([_to_ms_int(s) for s in secs], dtype=np.int64)

    # load valence into _DATA
    for _, row in vdf.iterrows():
        try:
            sid_raw = row[id_col]
            sid = int(float(sid_raw))
        except Exception:
            continue
        vals = np.asarray(row[valid_cols], dtype=float)
        _DATA[sid] = {
            "timestamps_ms": ms_times,
            "values": np.vstack([vals, np.full_like(vals, np.nan)]).T
        }

    # fill arousal
    for _, row in adf.iterrows():
        try:
            sid_raw = row[adf_id_col]
            sid = int(float(sid_raw))
        except Exception:
            continue
        if sid not in _DATA:
            continue
        vals = np.asarray(row[valid_cols], dtype=float)
        _DATA[sid]["values"][:, 1] = vals

    # build periods map
    id_periods_map: Dict[int, List[Dict[str, Any]]] = {}
    for sid, d in _DATA.items():
        vals = d["values"]
        mask = (~np.isnan(vals[:,0])) | (~np.isnan(vals[:,1]))
        if not np.any(mask):
            continue
        t_ms = d["timestamps_ms"]
        start_ms = int(t_ms[mask][0])
        end_ms   = int(t_ms[mask][-1])
        mean_vals = np.nanmean(vals[mask], axis=0)
        period = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "valence_mean": float(mean_vals[0]) if not np.isnan(mean_vals[0]) else float("nan"),
            "arousal_mean": float(mean_vals[1]) if not np.isnan(mean_vals[1]) else float("nan"),
        }
        id_periods_map.setdefault(sid, []).append(period)

    return id_periods_map

def get_stats_for_id_ms(sid: Any, start_ms: int, end_ms: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        key = int(float(sid))
    except Exception:
        print("I don't know")
        return None, None
    if key not in _DATA:
        print("I don't know")
        return None, None
    d = _DATA[key]
    t_ms = d["timestamps_ms"]
    vals = d["values"]
    if t_ms.size == 0:
        print("I don't know")
        return None, None
    i0 = np.searchsorted(t_ms, start_ms, side="left")
    i1 = np.searchsorted(t_ms, end_ms, side="right")
    if i1 <= i0:
        print("I don't know")
        return None, None
    segment = vals[i0:i1]
    if segment.size == 0:
        print("I don't know")
        return None, None
    means = np.nanmean(segment, axis=0)
    stds = np.nanstd(segment, axis=0)
    if np.isnan(means).all():
        print("I don't know")
        return None, None
    return means, stds

# ---------- SPECTROGRAM LOADER (writable copy) ----------
def load_spectrogram_chunk(
    sid: Any,
    start_ms: int,
    end_ms: int,
    spectrogram_dir: str = SPECTROGRAM_DIR,
    mmap: bool = True
) -> Optional[np.ndarray]:
    """
    Return a writable float32 numpy array of shape (n_mels, n_frames) for the requested ms window,
    or None if file/window not available.
    """
    global METADATA
    if not METADATA:
        load_metadata(METADATA_PATH)

    try:
        fname = f"{int(sid)}.npy"
    except Exception:
        fname = f"{sid}.npy"

    spec_path = Path(spectrogram_dir) / fname
    if not spec_path.exists():
        return None

    sr = int(METADATA["sr"])
    hop_length = int(METADATA["hop_length"])
    file_shapes = METADATA.get("file_shapes", {})

    available_frames = None
    if fname in file_shapes:
        try:
            available_frames = int(file_shapes[fname][1])
        except Exception:
            available_frames = None

    start_frame, end_frame = _ms_to_frame_indices(start_ms, end_ms, sr, hop_length)
    if available_frames is not None:
        if start_frame >= available_frames:
            return None
        end_frame = min(end_frame, available_frames)

    try:
        if mmap:
            arr = np.load(str(spec_path), mmap_mode="r")
        else:
            arr = np.load(str(spec_path))
    except Exception:
        return None

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return None

    n_mels, n_frames = arr.shape
    start_frame = max(0, min(start_frame, n_frames))
    end_frame   = max(0, min(end_frame, n_frames))
    if end_frame <= start_frame:
        return None

    chunk = arr[:, start_frame:end_frame]
    # ensure writable float32 copy
    return np.array(chunk, dtype=np.float32, copy=True)

# ---------- NORMALIZE id_periods_map INPUT ----------
def _normalize_id_periods_map(id_periods_map):
    """
    Accept dict or numpy array returned from older helpers.
    Return dict mapping int(sid) -> list of period dicts.
    """
    if isinstance(id_periods_map, dict):
        return id_periods_map
    if isinstance(id_periods_map, np.ndarray):
        out = {}
        if id_periods_map.size == 0:
            return out
        for row in id_periods_map:
            try:
                sid = int(float(row[0]))
                start_ms = int(row[1])
                end_ms = int(row[2])
            except Exception:
                continue
            period = {"start_ms": start_ms, "end_ms": end_ms,
                      "valence_mean": float("nan"), "arousal_mean": float("nan")}
            out.setdefault(sid, []).append(period)
        return out
    raise TypeError("id_periods_map must be dict or numpy.ndarray")

# ---------- DATASET (fixed frames per sample) ----------
class SpectrogramValenceArousalDataset(Dataset):
    def __init__(self,
                 ids: List,
                 id_periods_map: Dict,
                 window_seconds: float = WINDOW_SECONDS,
                 sr: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 n_mels: Optional[int] = None,
                 transform=None,
                 min_frames: int = 4):
        self.ids = ids
        self.id_periods_map = id_periods_map
        self.window_seconds = float(window_seconds)
        self.transform = transform

        if sr is None or hop_length is None or n_mels is None:
            if not METADATA:
                load_metadata(METADATA_PATH)
            sr = sr or int(METADATA.get("sr", 16000))
            hop_length = hop_length or int(METADATA.get("hop_length", 160))
            n_mels = n_mels or int(METADATA.get("n_mels", 128))

        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.hop_ms = self.hop_length / float(self.sr) * 1000.0
        self.window_ms = int(round(self.window_seconds * 1000.0))
        # target frames computed from window_ms/hop_ms
        self.target_frames = max(1, int(round(self.window_ms / max(1e-6, self.hop_ms))))
        self.min_frames = max(min_frames, self.target_frames)

        # build sample list
        self.samples = []
        stride_ms = int(round((self.window_seconds - WINDOW_OVERLAP) * 1000.0)) if WINDOW_OVERLAP < self.window_seconds else self.window_ms
        if stride_ms <= 0:
            stride_ms = self.window_ms

        for sid in self.ids:
            periods = id_periods_map.get(int(sid), [])
            for p in periods:
                try:
                    start_ms = int(p["start_ms"])
                    end_ms = int(p["end_ms"])
                    val = float(p.get("valence_mean", p.get("valence", float("nan"))))
                    aro = float(p.get("arousal_mean", p.get("arousal", float("nan"))))
                except Exception:
                    continue
                if math.isnan(val) and math.isnan(aro):
                    continue
                cur = start_ms
                while cur + self.window_ms <= end_ms:
                    self.samples.append((sid, cur, cur + self.window_ms, val, aro))
                    cur += stride_ms

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check annotation CSVs and metadata.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, start_ms, end_ms, val, aro = self.samples[idx]
        spec = load_spectrogram_chunk(sid, start_ms, end_ms)
        if spec is None:
            spec = np.zeros((self.n_mels, self.target_frames), dtype=np.float32)
        else:
            if spec.ndim == 3:
                spec = spec.squeeze()
            if spec.ndim != 2:
                spec = np.zeros((self.n_mels, self.target_frames), dtype=np.float32)
            n_frames = spec.shape[1]
            if n_frames < self.target_frames:
                pad = self.target_frames - n_frames
                spec = np.pad(spec, ((0,0),(0,pad)), mode='constant', constant_values=0.0)
            elif n_frames > self.target_frames:
                spec = spec[:, :self.target_frames]

        # ensure writable copy
        spec = np.array(spec, dtype=np.float32, copy=True)
        tensor_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, frames)

        if self.transform is not None:
            tensor_spec = self.transform(tensor_spec)

        label = torch.tensor([val if not math.isnan(val) else 0.0,
                              aro if not math.isnan(aro) else 0.0], dtype=torch.float32)
        return tensor_spec, label

# ---------- MODEL ----------
class PatchEmbed(nn.Module):
    def __init__(self, n_mels:int, patch_freq:int=16, patch_time:int=16, embed_dim:int=128, overlap:int=6):
        super().__init__()
        stride_f = max(1, patch_freq - overlap)
        stride_t = max(1, patch_time - overlap)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(patch_freq, patch_time),
                              stride=(stride_f, stride_t))

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                    # (B, embed_dim, P_f, P_t)
        B, C, Pf, Pt = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, N_patches, embed_dim)
        return x, (Pf, Pt)

class ASTRegressor(nn.Module):
    def __init__(self,
                 n_mels: int = N_MELS,
                 patch_freq: int = PATCH_SIZE,
                 patch_time: int = PATCH_SIZE,
                 embed_dim: int = MODEL_DIM,
                 n_layers: int = NUM_LAYERS,
                 n_heads: int = NUM_HEADS,
                 mlp_dim: int = MLP_DIM,
                 mlp_head_floor: int = MLP_HEAD_FLOOR,
                 dropout: float = DROPOUT,
                 overlap: int = PATCH_OVERLAP):
        super().__init__()

        self.patch_embed = PatchEmbed(n_mels=n_mels, patch_freq=patch_freq,
                                      patch_time=patch_time, embed_dim=embed_dim,
                                      overlap=overlap)
        
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout,
                                                   activation='gelu', batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.reg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, max(mlp_head_floor, embed_dim//4)),
            nn.GELU(),
            nn.Linear(max(mlp_head_floor, embed_dim//4), 2)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _ensure_pos_embed(self, n_patches: int):
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != 1 + n_patches):
            pe = torch.zeros(1, 1 + n_patches, self.embed_dim, device=self.cls_token.device)
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = nn.Parameter(pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches, (Pf, Pt) = self.patch_embed(x)   # (B, N, embed_dim)
        N = patches.shape[1]
        self._ensure_pos_embed(N)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)   # (B, 1+N, E)
        x = x + self.pos_embed[:, : (1+N), :]
        x = self.transformer(x)
        cls_out = x[:, 0]
        out = self.reg_head(cls_out)
        return out

# ---------- TRAIN / EVAL ----------
def train_epoch(model: nn.Module, dl: DataLoader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(dl, desc="train", leave=False)
    for spec, label in pbar:
        spec = spec.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(spec)
                loss = criterion(out, label)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(spec)
            loss = criterion(out, label)
            loss.backward()
            if GRAD_CLIP_NORM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        total_loss += loss.item() * spec.size(0)
        n += spec.size(0)
        pbar.set_postfix(loss=total_loss/max(1,n))
    return total_loss / max(1, n)

def eval_epoch(model: nn.Module, dl: DataLoader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        pbar = tqdm(dl, desc="eval", leave=False)
        for spec, label in pbar:
            spec = spec.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            out = model(spec)
            loss = criterion(out, label)
            total_loss += loss.item() * spec.size(0)
            n += spec.size(0)
            pbar.set_postfix(loss=total_loss/max(1,n))
    return total_loss / max(1, n)

# ---------- DATALOADER HELPERS ----------
def make_dataloaders(id_periods_map: Dict[int, List[Dict[str, Any]]],
                     batch_size: int = BATCH_SIZE,
                     window_seconds: float = WINDOW_SECONDS,
                     num_workers: int = NUM_WORKERS):
    n_mels = int(METADATA.get("n_mels", 128))
    sr = int(METADATA.get("sr", 16000))
    hop_length = int(METADATA.get("hop_length", 160))

    ids = list(id_periods_map.keys())
    random.shuffle(ids)
    n_val = max(1, int(len(ids) * 0.1))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]

    train_ds = SpectrogramValenceArousalDataset(train_ids, id_periods_map, window_seconds=window_seconds,
                                                sr=sr, hop_length=hop_length, n_mels=n_mels)
    val_ds = SpectrogramValenceArousalDataset(val_ids, id_periods_map, window_seconds=window_seconds,
                                              sr=sr, hop_length=hop_length, n_mels=n_mels)
    pin_memory = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, train_ids, val_ids

# ---------- MAIN TRAIN FUNCTION ----------
def main_train(id_periods_map,
               epochs: int = EPOCHS,
               batch_size: int = BATCH_SIZE,
               lr: float = LR,
               window_seconds: float = WINDOW_SECONDS,
               embed_dim: int = MODEL_DIM,
               n_layers: int = NUM_LAYERS,
               n_heads: int = NUM_HEADS,
               mlp_dim: int = MLP_DIM,
               overlap: int = PATCH_OVERLAP,
               save_dir: str = "ast_checkpoints",
               resume: Optional[str] = None):
    os.makedirs(save_dir, exist_ok=True)

    # normalize id_periods input (accept numpy array or dict)
    id_periods_map = _normalize_id_periods_map(id_periods_map)

    train_loader, val_loader, train_ids, val_ids = make_dataloaders(id_periods_map, batch_size=batch_size,
                                                                    window_seconds=window_seconds,
                                                                    num_workers=NUM_WORKERS)

    n_mels = int(METADATA.get("n_mels", 128))
    model = ASTRegressor(n_mels=n_mels,
                         patch_freq=PATCH_SIZE, patch_time=PATCH_SIZE,
                         embed_dim=embed_dim, n_layers=n_layers, n_heads=n_heads,
                         mlp_dim=mlp_dim, overlap=overlap).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # reduce on plateau without 'verbose' kw to support older/newer PyTorch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, factor=0.5)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE.type=="cuda") else None

    # resume support: load model/optim and restore start epoch and best_val if present
    start_epoch = 0
    best_val = float("inf")
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=DEVICE)
        if isinstance(ck, dict) and "model" in ck:
            try:
                model.load_state_dict(ck.get("model"))
            except Exception:
                # attempt partial load / strict=False if state dict shape mismatch
                model.load_state_dict(ck.get("model"), strict=False)
            if "optim" in ck and isinstance(ck["optim"], dict):
                try:
                    optimizer.load_state_dict(ck["optim"])
                except Exception:
                    # optimizer state may not be compatible across devices/versions; ignore if fails
                    print("Warning: could not fully restore optimizer state from resume checkpoint.")
            start_epoch = int(ck.get("epoch", 0)) + 1
            best_val = float(ck.get("best_val", best_val))
            print(f"Resumed from {resume}: start_epoch={start_epoch}, best_val={best_val}")
        else:
            # ck might be a bare state_dict
            try:
                model.load_state_dict(ck)
                print(f"Loaded model state_dict from {resume}")
            except Exception:
                print(f"Warning: resume file {resume} could not be loaded as a model checkpoint.")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler=scaler)
        val_loss = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)
        t1 = time.time()
        print(f"Epoch {epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} time={t1-t0:.1f}s")

        # save latest checkpoint (includes best_val for resume safety)
        ckpath = os.path.join(save_dir, f"ast_epoch{epoch:03d}.pth")
        torch.save({
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch,
            "best_val": best_val
        }, ckpath)

        # save best model
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(save_dir, "ast_best.pth")
            torch.save({
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val
            }, best_path)
            print(f"✔ New best model saved to {best_path} (val_loss={best_val:.5f})")

    print("Training finished. Best val loss:", best_val)
    return model

# ---------- OPTIONAL PLOTTING ----------
def plot_spectrogram(spec: np.ndarray, title: str = "Spectrogram", sr: int | None = None, hop_length: int | None = None):
    plt.figure(figsize=(10, 4))
    if sr is not None and hop_length is not None:
        times = np.arange(spec.shape[1]) * hop_length / sr
        extent = [times[0], times[-1], 0, spec.shape[0]]
        plt.imshow(spec, origin="lower", aspect="auto", extent=extent)
        plt.xlabel("Time (s)")
    else:
        plt.imshow(spec, origin="lower", aspect="auto")
        plt.xlabel("Frames")
    plt.ylabel("Mel bins")
    plt.title(title)
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--window_seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--embed_dim", type=int, default=MODEL_DIM)
    parser.add_argument("--n_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--n_heads", type=int, default=NUM_HEADS)
    parser.add_argument("--mlp_dim", type=int, default=MLP_DIM)
    parser.add_argument("--overlap", type=int, default=PATCH_OVERLAP)
    parser.add_argument("--save_dir", type=str, default="ast_checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    load_metadata(METADATA_PATH)
    id_periods_map = prepare_and_index_ms(VALENCE_CSV, AROUSAL_CSV)

    if len(id_periods_map) == 0:
        raise RuntimeError("No labeled tracks found. Check CSV files and timestamps.")

    print(f"Found {len(id_periods_map)} tracks with annotations. Starting training.")
    main_train(id_periods_map,
               epochs=args.epochs,
               batch_size=args.batch,
               lr=args.lr,
               window_seconds=args.window_seconds,
               embed_dim=args.embed_dim,
               n_layers=args.n_layers,
               n_heads=args.n_heads,
               mlp_dim=args.mlp_dim,
               overlap=args.overlap,
               save_dir=args.save_dir,
               resume=args.resume)

import json
import math
import os
import random
import re
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

from sklearn.model_selection import train_test_split
import copy
import time


# ----------HYPERPARAMETERS ----------
# folders / paths
SPECTROGRAM_DIR = "data/dataspectrogram"   # folder containing .npy files and _metadata.json
METADATA_PATH = "data/dataspectrogram/_metadata.json"  # path to metadata JSON file
AROUSAL_CSV = "data/dynamic_annotations/arousal.csv"      # path to arousal CSV annotations
VALENCE_CSV = "data/dynamic_annotations/valence.csv"      # path to valence CSV annotations

MODEL_SAVE_DIR = "models/best_model"

# model / training
VAL_FRACTION = 0.1
BATCH_SIZE = 8
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5  # set to integer to enable early stopping

LR = 1e-5
WEIGHT_DECAY = 1e-2

cpu = os.cpu_count() or 4
workers = cpu - 2
NUM_WORKERS_TRAIN = max(0, min(math.floor(workers * 2 / 3), 6))
NUM_WORKERS_VAL = max(0, min(workers - math.floor(workers * 2 / 3), 3))

WINDOW_SECONDS = 4.0   # crop length in seconds
WINDOW_OVERLAP = 2.0   # overlap between crops in seconds

PATCH_SIZE = 16 # paper uses 16
PATCH_OVERLAP = 6 # paper uses 6
MODEL_DIM = 768 # paper uses 768
NUM_LAYERS = 12 # paper uses 12
NUM_HEADS = 12 # paper uses 12
MLP_DIM = 3072 # paper uses 3072
DROPOUT = 0.1 # paper uses 0.1

# misc training tweaks
USE_GPU = True
USE_AMP = True            # enable mixed precision
PIN_MEMORY = True
GRAD_CLIP_NORM = 1.0     # clip gradients to this norm (set None to disable)
SCHEDULER_PATIENCE = 2   # for ReduceLROnPlateau
SEED = 42



# ---------- SEED AND DEVICE SETUP ----------




# ---------- GLOBAL VARIABLES ----------

_DATA: Dict[int, Dict[str, np.ndarray]] = {}
_METADATA: dict = {}



# ---------- UTILITIES ----------
_num_re = re.compile(r'-?\d+')

def _parse_sid(value: Any) -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Invalid song id: {value}") from exc


def _parse_ms(value: Any, name: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"Invalid {name} in ms: {value}") from exc

def _parse_str_col_to_ms(colname: str) -> int:
    m = _num_re.search(colname)

    if not m:
        raise ValueError(f"Can't parse integer time from column '{colname}'")
    
    return int(m.group())



# ---------- METADATA ----------

def load_metadata(path: str = METADATA_PATH) -> Dict[str, Any]:
    global _METADATA

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    _METADATA = json.loads(p.read_text(encoding="utf-8"))

    # minimal validation
    for k in ("sr", "hop_length", "file_shapes"):
        if k not in _METADATA:
            raise KeyError(f"metadata missing '{k}'")
        
    return _METADATA



def _ms_to_frame_indices(start_ms: int, end_ms: int, sr: int, hop_length: int) -> Tuple[int, int]:
    start = int(np.floor(start_ms/1000.0 * sr / hop_length))
    end   = int(np.ceil (end_ms  /1000.0 * sr / hop_length))
    return max(0, start), max(0, end)



# ---------- ANNOTATIONS ----------

def prepare_and_index_ms(valence_csv: str = VALENCE_CSV, arousal_csv: str = AROUSAL_CSV) -> np.ndarray:
    """
    Loads valence/arousal CSVs into _DATA and returns periods array [[id, start_ms, end_ms], ...]
    """
    global _DATA
    _DATA = {}
    vdf = pd.read_csv(valence_csv)
    adf = pd.read_csv(arousal_csv)

    if 'song_id' in vdf.columns:
        id_col = 'song_id' 
    else:
        id_col = vdf.columns[0]
    
    time_cols = []
    for c in vdf.columns:
        if c != id_col:
            time_cols.append(c)

    parsed = []
    valid_cols = []
    for c in time_cols:
        try:
            parsed.append(_parse_str_col_to_ms(c))
            valid_cols.append(c)
        except Exception:
            raise RuntimeError("Could not parse time columns in valence CSV")

    t_ms = np.array(parsed, dtype=np.int64)

    # fill valence, reserve second column for arousal
    for _, row in vdf.iterrows():
        try:
            sid = _parse_sid(row[id_col])
        except Exception:
            raise RuntimeError("Could not parse song id in valence CSV")
        vals = np.asarray(row[valid_cols], dtype=float)
        _DATA[sid] = {"timestamps_ms": t_ms, "values": np.vstack([vals, np.full_like(vals, np.nan)]).T}

    adf_id = adf.columns[0]

    common_cols = []
    for c in valid_cols:
        if c in adf.columns:
            common_cols.append(c)

    if common_cols != valid_cols:
        missing = [c for c in valid_cols if c not in adf.columns]
        extra = [c for c in adf.columns if c not in valid_cols and c != adf_id]
        raise ValueError(
            "Arousal CSV time columns do not match valence CSV.\n"
            f"Missing columns: {missing}\n"
            f"Extra columns:   {extra}"
        )

    for _, row in adf.iterrows():
        try:
            sid = _parse_sid(row[adf_id])
        except Exception:
            raise RuntimeError("Could not parse song id in arousal CSV")
        if sid in _DATA:
            vals = np.asarray(row[common_cols], dtype=float)
            # ensure shape matches timestamps
            if vals.shape[0] != _DATA[sid]["timestamps_ms"].shape[0]:
                # try to broadcast or skip
                print(f"Mismatch annotation length for id {sid} — skipping arousal")
                continue
            _DATA[sid]["values"][:, 1] = vals

    rows = []
    for sid, d in _DATA.items():
        mask = (~np.isnan(d["values"][:, 0])) | (~np.isnan(d["values"][:, 1]))
        if not np.any(mask):
            continue
        t = d["timestamps_ms"]
        rows.append([int(sid), int(t[mask][0]), int(t[mask][-1])])

    if not rows:
        raise Exception("No valid annotation periods found")
    
    return np.asarray(rows, dtype=np.int64)


def get_target_VA_means(sid: Any, start_ms: int, end_ms: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (means, stds) for [start_ms, end_ms).
    On failure, prints a clear reason and returns (None, None).
    """

    # normalize song id
    try:
        key = _parse_sid(sid)
    except Exception:
        print(f"Invalid song id: {sid}")
        return None, None

    try:
        start_ms = _parse_ms(start_ms, "start_ms")
        end_ms = _parse_ms(end_ms, "end_ms")
    except Exception as exc:
        print(str(exc))
        return None, None

    if key not in _DATA:
        print(f"Song id {key} not found")
        return None, None

    d = _DATA[key]

    t = d.get("timestamps_ms")
    if t is None or len(t) == 0:
        print(f"No timestamps available for song id {key}")
        return None, None

    if start_ms >= end_ms:
        print("Invalid time window: start_ms must be < end_ms")
        return None, None

    # find index range
    i0 = np.searchsorted(t, start_ms, side="left")
    i1 = np.searchsorted(t, end_ms,   side="right")

    if i1 <= i0:
        print("No annotations found in the requested time window")
        return None, None

    values = d.get("values")
    if values is None or values.size == 0:
        print(f"No annotation values available for song id {key}")
        return None, None

    seg = values[i0:i1]

    if seg.size == 0:
        print("Empty annotation slice after indexing")
        return None, None

    if np.isnan(seg).all():
        print("Annotations in the requested window are all NaN")
        return None, None

    means = np.nanmean(seg, axis=0)
    stds  = np.nanstd(seg,  axis=0)

    return means, stds



# ---------- SPECTROGRAM I/O AND WINDOW GENERATION ----------

def load_spectrogram_chunk(
    sid: Any,
    start_ms: int,
    end_ms: int,
    spectrogram_dir: str = SPECTROGRAM_DIR,
    mmap: bool = False
) -> Optional[np.ndarray]:
    """
    Returns a heap-owned, contiguous np.float32 array shaped (n_mels, n_frames_slice),
    or None on error (and prints a clear message).

    This version normalizes the returned slice to the expected number of frames
    for the requested (start_ms, end_ms) window by trimming or padding with zeros.
    """
    # ensure metadata loaded
    if not _METADATA:
        try:
            load_metadata()
        except Exception as exc:
            print("Metadata not available")
            return None

    # build filename (prefer integer song ids)
    try:
        sid_int = _parse_sid(sid)
    except Exception:
        print(f"Invalid song id: {sid}")
        return None

    try:
        start_ms = _parse_ms(start_ms, "start_ms")
        end_ms = _parse_ms(end_ms, "end_ms")
    except Exception as exc:
        print(str(exc))
        return None

    if start_ms >= end_ms:
        print("Invalid time range: start_ms must be < end_ms")
        return None

    fname = f"{sid_int}.npy"

    p = Path(spectrogram_dir) / fname
    if not p.exists():
        print(f"Spectrogram file not found: {p}")
        return None

    # metadata helpers
    sr = int(_METADATA.get("sr", 0))
    hop = int(_METADATA.get("hop_length", 0))
    file_shapes = _METADATA.get("file_shapes", {})

    # expected number of frames for the requested window (float -> ceil)
    expected_frames = int(np.ceil((end_ms - start_ms) / 1000.0 * sr / hop))

    # optional available frames from metadata; may be None
    available = None
    if fname in file_shapes:
        try:
            available = int(file_shapes[fname][1])
        except Exception:
            # shape present but malformed
            print(f"No file shape info for: {fname}")
            return None

    # convert ms -> frame indices (current logic)
    sframe, eframe = _ms_to_frame_indices(start_ms, end_ms, sr, hop)

    # If available frames known, clip indices to file bounds
    if available is not None and sframe >= available:
        print("Requested start beyond available frames")
        return None
    if available is not None:
        eframe = min(eframe, available)

    # load file (mmap if requested)
    try:
        arr = np.load(str(p), mmap_mode='r' if mmap else None)
    except Exception as exc:
        print(f"Failed to load spectrogram file: {p}")
        return None

    # validate shape
    if arr.ndim != 2:
        print(f"Spectrogram file has unexpected shape: {getattr(arr, 'shape', None)}")
        return None

    n_mels, n_frames = arr.shape

    # clip to array bounds (sframe/eframe already clipped above if available)
    sframe = max(0, min(sframe, n_frames))
    eframe = max(0, min(eframe, n_frames))
    if eframe <= sframe:
        print("Requested time range results in empty slice")
        return None

    # slice
    chunk = arr[:, sframe:eframe]

    # convert to float32 and contiguous heap-owned array
    if chunk.dtype != np.float32 or not chunk.flags['OWNDATA'] or not chunk.flags['C_CONTIGUOUS']:
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
    else:
        # ensure we make a heap-owned copy (if mmap was used this makes it safe)
        chunk = np.array(chunk, dtype=np.float32, copy=True)

    # Now normalize to expected_frames
    cur_frames = chunk.shape[1]
    if cur_frames == expected_frames:
        return chunk

    if cur_frames < expected_frames:
        # pad on the right with zeros
        pad_amount = expected_frames - cur_frames
        # pad shape: ((0,0), (0,pad_amount))
        chunk = np.pad(chunk, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0.0)
        return np.ascontiguousarray(chunk, dtype=np.float32)
    else:
        # cur_frames > expected_frames: trim extra frames on the right
        chunk = chunk[:, :expected_frames]
        return np.ascontiguousarray(chunk, dtype=np.float32)



def generate_windows(start_ms: int, end_ms: int, window_ms: int, step_ms: int):
    """Yield (win_start_ms, win_end_ms) covering [start_ms, end_ms].
    Always yields at least one window of length window_ms (clips to end).
    """
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    window_ms = int(window_ms)
    step_ms = int(step_ms)
    total = end_ms - start_ms
    if total <= window_ms:
        yield start_ms, end_ms
        return

    # regularly spaced starts
    starts = np.arange(start_ms, end_ms - window_ms + 1, step_ms, dtype=np.int64)
    for s in starts:
        yield int(s), int(s + window_ms)

    # ensure final tail is covered (one window ending at end_ms)
    last_start = int(end_ms - window_ms)
    if last_start > starts[-1]:
        yield last_start, end_ms


# ---------- PLOTTING (OPTIONAL) ----------

def plot_spectrogram(spec: np.ndarray, title: str = "Spectrogram", sr: int | None = None, hop_length: int | None = None):
    plt.figure(figsize=(10, 4))
    if sr and hop_length:
        times = np.arange(spec.shape[1]) * hop_length / sr
        extent = [times[0], times[-1], 0, spec.shape[0]]
        plt.imshow(spec, origin="lower", aspect="auto", extent=extent)
        plt.xlabel("Time (s)")
    else:
        plt.imshow(spec, origin="lower", aspect="auto")
        plt.xlabel("Frames")
    plt.ylabel("Mel bins"); plt.title(title); plt.colorbar(label="Amplitude"); plt.tight_layout(); plt.show()



# ---------- DATASET ----------

class EmotionWindowDataset(Dataset):
    def __init__(self, rows, spectrogram_dir=SPECTROGRAM_DIR):
        """
        rows: list of (sid, start_ms, end_ms, target_mean)
        """
        self.rows = rows
        self.spectrogram_dir = spectrogram_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sid, start_ms, end_ms, target = self.rows[idx]

        spec = load_spectrogram_chunk(
            sid,
            start_ms,
            end_ms,
            spectrogram_dir=self.spectrogram_dir,
            mmap=False
        )

        if spec is None:
            raise IndexError("Missing spectrogram chunk")

        # ensure float32 contiguous heap-owned array, then convert to tensor
        # torch.tensor(...) always makes a copy and yields a CPU tensor that is safe for collate
        spec = torch.from_numpy(spec).float().unsqueeze(0).contiguous()  # (1, n_mels, n_frames)

        target = torch.tensor(target, dtype=torch.float32)  # (2,)

        return spec, target



# ----- Minimal AST-like regression model -----

# keep your global defaults or define them above as needed:
# PATCH_SIZE, PATCH_OVERLAP, MODEL_DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, DROPOUT

class SimpleASTRegressor(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_frames: int,
        patch_size: int = PATCH_SIZE,
        patch_overlap: int = PATCH_OVERLAP,
        model_dim: int = MODEL_DIM,
        num_layers: int = NUM_LAYERS,
        num_heads: int = NUM_HEADS,
        mlp_dim: int = MLP_DIM,
        dropout: float = DROPOUT,
        out_dim: int = 2
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_stride = max(1, patch_size - patch_overlap)

        # compute patch grid counts (H x W)
        self.n_patches_h = (n_mels - patch_size) // self.patch_stride + 1
        self.n_patches_w = (n_frames - patch_size) // self.patch_stride + 1
        if self.n_patches_h <= 0 or self.n_patches_w <= 0:
            raise ValueError("Patch size too large for spectrogram shape")

        self.n_patches = self.n_patches_h * self.n_patches_w

        # patch embedding: project flattened patch -> model_dim
        self.input_channels = 1
        self.patch_dim = self.input_channels * patch_size * patch_size
        self.patch_proj = nn.Linear(self.patch_dim, model_dim)

        # small normalization on patch embeddings (ViT-style)
        self.patch_norm = nn.LayerNorm(model_dim)

        # dropout for embeddings (matches typical AST/ViT training)
        self.proj_dropout = nn.Dropout(dropout)
        self.token_dropout = nn.Dropout(dropout)

        # cls token + positional embeddings (initialized for the default input size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, model_dim))

        # transformer encoder (uses PyTorch's TransformerEncoderLayer)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # regression head
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, out_dim)

        # initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H=n_mels, W=n_frames)
        # Use unfold to extract flattened patches
        kernel = (self.patch_size, self.patch_size)
        stride = (self.patch_stride, self.patch_stride)
        patches = F.unfold(x, kernel_size=kernel, stride=stride)  # (B, C*patch_h*patch_w, L)
        patches = patches.transpose(1, 2)  # (B, L, patch_dim)
        return patches  # L should == current n_patches

    def _get_pos_embed_for(self, cur_n_patches_h: int, cur_n_patches_w: int, device: torch.device):
        """
        Return positional embeddings sized for 1 + (cur_n_patches_h * cur_n_patches_w).
        If the original pos_embed matches, return directly; otherwise do bilinear interpolation
        of the patch grid (excluding the cls token) from (self.n_patches_h, self.n_patches_w)
        to (cur_n_patches_h, cur_n_patches_w).
        """
        pos = self.pos_embed  # (1, 1 + orig_n_patches, dim)
        B1, L, D = pos.shape
        orig_h, orig_w = self.n_patches_h, self.n_patches_w

        # quick path: same size
        if (cur_n_patches_h == orig_h) and (cur_n_patches_w == orig_w):
            return pos.to(device)

        # split cls and patch part
        cls_pos = pos[:, :1, :].to(device)              # (1,1,D)
        patch_pos = pos[:, 1:, :].to(device)            # (1, orig_h*orig_w, D)

        # reshape -> (1, D, orig_h, orig_w)
        patch_pos = patch_pos.reshape(1, orig_h, orig_w, D).permute(0, 3, 1, 2)

        # interpolate to new (cur_h, cur_w)
        patch_pos_interp = F.interpolate(
            patch_pos, size=(cur_n_patches_h, cur_n_patches_w),
            mode='bilinear', align_corners=False
        )  # (1, D, cur_h, cur_w)

        # flatten back to (1, cur_h*cur_w, D)
        patch_pos_interp = patch_pos_interp.permute(0, 2, 3, 1).reshape(1, cur_n_patches_h * cur_n_patches_w, D)

        # concat cls
        new_pos = torch.cat([cls_pos, patch_pos_interp], dim=1)  # (1, 1 + cur_n_patches, D)
        return new_pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, n_mels, n_frames)  (n_frames can vary)
        returns: (B, 2) — predicted [valence, arousal]
        """
        B, C, H, W = x.shape

        patches = self._extract_patches(x)                 # (B, cur_n_patches, patch_dim)
        cur_n_patches = patches.shape[1]

        # infer current patch grid dims:
        # we assume n_patches_h (along frequency) stays the same as built from constructor
        cur_n_patches_h = self.n_patches_h
        if cur_n_patches % cur_n_patches_h != 0:
            # fallback: try to compute both dims from geometry if something unexpected happens
            cur_n_patches_h = int((H - self.patch_size) // self.patch_stride + 1)
        cur_n_patches_w = cur_n_patches // cur_n_patches_h

        if cur_n_patches_h <= 0 or cur_n_patches_w <= 0:
            raise RuntimeError("Computed invalid patch grid from input")

        if cur_n_patches != cur_n_patches_h * cur_n_patches_w:
            raise RuntimeError("Mismatch in inferred patch grid and actual patches")

        # project patches
        emb = self.patch_proj(patches)                     # (B, cur_n_patches, model_dim)
        emb = self.patch_norm(emb)
        emb = self.proj_dropout(emb)

        # prepare cls token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(emb.device)  # (B,1,model_dim)

        # positional embedding: handle shape mismatch by interpolation if needed
        pos = self._get_pos_embed_for(cur_n_patches_h, cur_n_patches_w, device=emb.device)
        if pos.shape[1] != (1 + cur_n_patches):
            # sanity check
            raise RuntimeError(f"Positional embedding size mismatch: expected {1 + cur_n_patches}, got {pos.shape[1]}")

        # concat and add pos
        tokens = torch.cat([cls_tokens, emb], dim=1)       # (B, 1 + cur_n_patches, model_dim)
        tokens = tokens + pos  # broadcasting over batch
        tokens = self.token_dropout(tokens)

        # transformer expects (B, S, D) given batch_first=True
        tokens = self.transformer(tokens)                  # (B, 1 + cur_n_patches, model_dim)

        cls_out = tokens[:, 0, :]                          # (B, model_dim)
        out = self.norm(cls_out)
        out = self.head(out)                               # (B, out_dim)
        return out




# -------- CCC (Concordance Correlation Coefficient) --------

def concordance_correlation_coefficient(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8):
    """
    Compute CCC per-dimension and return mean CCC across dims.
    Keep computations in float32 to support MPS (which doesn't support float64).
    y_pred, y_true: (B, D)
    returns: tensor scalar (mean CCC) dtype=float32
    """
    # ensure float32 for MPS compatibility
    y_pred = y_pred.float()
    y_true = y_true.float()

    mean_pred = torch.mean(y_pred, dim=0)
    mean_true = torch.mean(y_true, dim=0)

    var_pred = torch.var(y_pred, dim=0, unbiased=False)
    var_true = torch.var(y_true, dim=0, unbiased=False)

    cov = torch.mean((y_pred - mean_pred) * (y_true - mean_true), dim=0)

    ccc = (2.0 * cov) / (var_pred + var_true + (mean_pred - mean_true) ** 2 + eps)

    # return mean over dims as float32 tensor
    return torch.mean(ccc)
    
    
def ccc_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
    """Loss = 1 - mean(CCC). Keep dtype float32."""
    ccc = concordance_correlation_coefficient(y_pred, y_true)
    return 1.0 - ccc


# -------- build windowed rows (examples) ----------

def build_rows_from_annotations(window_seconds: float = WINDOW_SECONDS, overlap_seconds: float = WINDOW_OVERLAP, min_valid: float = 0.0):
    """
    Builds rows: list of (sid, start_ms, end_ms, target_mean_array (2,))
    Uses _DATA populated by prepare_and_index_ms().
    min_valid: minimum fraction of annotated points within a window to accept (not used strictly here,
               but kept for future extension).
    """
    rows = []
    # ensure metadata / annotations loaded already
    if not _METADATA:
        try:
            load_metadata()
        except Exception:
            print("Warning: metadata not loaded before building rows")

    # prepare index (this loads/validates _DATA)
    try:
        periods = prepare_and_index_ms()
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare annotations: {exc}")

    window_ms = int(window_seconds * 1000)
    step_ms = int((window_seconds - overlap_seconds) * 1000)

    for sid, start_ms, end_ms in periods:
        # generate windows inside annotated period
        for w0, w1 in generate_windows(int(start_ms), int(end_ms), window_ms, step_ms):
            means_stds = get_target_VA_means(sid, w0, w1)
            if means_stds is None:
                # skip invalid windows
                continue
            means, stds = means_stds
            if means is None:
                continue
            # convert to python list/np for storage
            target = np.asarray(means, dtype=np.float32)
            # optional filter: ensure at least some non-nan (get_target_VA_means already checks)
            rows.append((int(sid), int(w0), int(w1), target))
    if not rows:
        raise RuntimeError("No rows built — check your annotations and metadata")
    return rows



# -------- collate function (if you want any custom handling) ----------

def cpu_collate(batch):
    """
    Collate in CPU-only form: stack CPU tensors and return them (no .to(device) here).
    This avoids worker processes trying to serialize non-CPU storage.
    """
    specs = [b[0] if isinstance(b[0], torch.Tensor) else torch.tensor(b[0]) for b in batch]
    targets = [b[1] if isinstance(b[1], torch.Tensor) else torch.tensor(b[1]) for b in batch]
    specs = torch.stack(specs, dim=0)   # (B,1,n_mels,n_frames)
    targets = torch.stack(targets, dim=0)
    return specs, targets



def _worker_init_fn(worker_id):
    # seed
    seed = SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ensure worker has metadata loaded (important for spawn start on macOS)
    global _METADATA

    if not _METADATA:
        try:
            load_metadata()
        except Exception:
            # if loading fails, print for debugging; the worker must have metadata
            print(f"    Worker {worker_id + 1}: failed to load metadata")



def save_model_with_config(
    save_dir: str,
    model: torch.nn.Module,
    config: dict[str, Any] |  None = None,
    extra: Dict[str, Any] | None = None,
):
    """
    Creates `save_dir/` and saves:
      - model.pt     (state_dict)
      - config.json  (model dimensions + hyperparams)

    Assumes model is SimpleASTRegressor.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- extract model config (must match constructor args) ----
    if extra is not None:
        config["extra"] = extra

    # ---- save weights ----
    torch.save(model.state_dict(), save_dir / "model.pt")

    # ---- save config ----
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {save_dir}")
    print(f"  ├─ model.pt")
    print(f"  └─ config.json")



# THIS IS LOOKING GOOD IF LOSS CAN GET DOWN
# WORK ON SAVING MODEL FULLY SO TESTING IS EASIER
# WRITE TESTING FILE



def train_loop(model, opt, train_loader, val_loader, model_save_path, device, config):

    if EARLY_STOPPING_PATIENCE is not None:
        early_stopping_patience = EARLY_STOPPING_PATIENCE
    else:
        early_stopping_patience = float('inf')

    best_val_loss = float('inf')
    epochs_no_improve = 0



    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        train_mses = []
        train_cccs = []

        pbar = tqdm(train_loader, desc = f"Training Epoch {epoch}", leave = True)
        for specs, targets in pbar:
            specs = specs.to(device)
            targets = targets.to(device)

            opt.zero_grad()
            preds = model(specs)
            mse = F.mse_loss(preds, targets)
            loss_ccc = ccc_loss(preds, targets)
            loss = 0.5 * mse + 0.5 * loss_ccc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            with torch.no_grad():
                batch_ccc = concordance_correlation_coefficient(preds, targets).item()

            train_losses.append(loss.item())
            train_cccs.append(batch_ccc)
            train_mses.append(mse.item())

            pbar.set_postfix({"loss": f"{np.mean(train_losses):.4f}", "mse": f"{np.mean(train_mses):.4f}", "ccc": f"{np.mean(train_cccs):.4f}"})

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_train_mse  = float(np.mean(train_mses))  if train_mses else 0.0
        avg_train_ccc  = float(np.mean(train_cccs))  if train_cccs else 0.0

        print(f"    loss: {avg_train_loss:.4f}, mse: {avg_train_mse:.6f}, ccc: {avg_train_ccc:.4f}")



        model.eval()
        val_losses = []
        val_mses = []
        val_cccs = []

        with torch.no_grad():
            vbar = tqdm(val_loader, desc = f"Validation Epoch {epoch}", leave = True)
            for specs, targets in vbar:
                specs = specs.to(device)
                targets = targets.to(device)

                preds = model(specs)

                mse = F.mse_loss(preds, targets).item()
                loss_ccc = ccc_loss(preds, targets).item()
                combined_loss = 0.5 * mse + 0.5 * loss_ccc

                batch_ccc = concordance_correlation_coefficient(preds, targets).item()

                val_losses.append(combined_loss)
                val_mses.append(mse)
                val_cccs.append(batch_ccc)

                vbar.set_postfix({"val_loss": f"{np.mean(val_losses):.4f}", "val_mse": f"{np.mean(val_mses):.4f}", "val_ccc": f"{np.mean(val_cccs):.4f}"})

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        avg_val_mse  = float(np.mean(val_mses))  if val_mses else 0.0
        avg_val_ccc  = float(np.mean(val_cccs))  if val_cccs else 0.0

        print(f"    val_loss: {avg_val_loss:.4f}, val_mse: {avg_val_mse:.6f}, val_ccc: {avg_val_ccc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model_with_config(
                model_save_path,
                model,
                config=config,
                extra={"val_loss": avg_val_loss, "val_mse": avg_val_mse, "val_ccc": avg_val_ccc,}
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered.")
                return

    return




# -------- convenience entry point ----------

def run_training_pipeline(spectrogram_dir=SPECTROGRAM_DIR, model_save_path=MODEL_SAVE_DIR):
    # set seeds
    print("Initializing device and setting seeds...")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # device setup
    global USE_AMP, PIN_MEMORY, device
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        USE_AMP = False  # AMP not supported on MPS
        PIN_MEMORY = False  # pin_memory not supported on MPS
    else:
        device = torch.device("cpu")

    print(f"    Using device: {device}")
    
    # Load metadata and annotation/index
    print("Preparing rows from annotations...")

    load_metadata()
    rows = build_rows_from_annotations(window_seconds=WINDOW_SECONDS, overlap_seconds=WINDOW_OVERLAP)
    train_rows, val_rows = train_test_split(
        rows,
        test_size = VAL_FRACTION,
        random_state = int(SEED),
        shuffle=True
    )

    print(f"    Built {len(rows)} examples, Training examples: {len(train_rows)}, Validation examples: {len(val_rows)}")

    train_ds = EmotionWindowDataset(train_rows, spectrogram_dir=spectrogram_dir)
    val_ds = EmotionWindowDataset(val_rows, spectrogram_dir=spectrogram_dir)

    spec, _ = train_ds[0] 
    n_mels = spec.shape[1] 
    n_frames = spec.shape[2]

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS_TRAIN,
        pin_memory=PIN_MEMORY,
        collate_fn=cpu_collate,
        worker_init_fn=_worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS_VAL,
        pin_memory=PIN_MEMORY,
        collate_fn=cpu_collate,
        worker_init_fn=_worker_init_fn,
    )

    config = {
        "sample_rate": int(_METADATA.get("sr", 0)),
        "hop_length": int(_METADATA.get("hop_length", 0)),
        "n_fft": int(_METADATA.get("n_fft", 0)),
        "power": float(_METADATA.get("power", 1.0)),
        "ref": float(_METADATA.get("ref", 1.0)),
        "window_seconds": WINDOW_SECONDS,
        "window_overlap": WINDOW_OVERLAP,

        "n_mels": n_mels,
        "n_frames": n_frames,
        "patch_size": PATCH_SIZE,
        "patch_overlap": PATCH_OVERLAP,
        "model_dim": MODEL_DIM,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "mlp_dim": MLP_DIM,
        "dropout": DROPOUT,
        "out_dim": 2
    }

    model = SimpleASTRegressor(
        n_mels=config["n_mels"],
        n_frames=config["n_frames"],
        patch_size=config["patch_size"],
        patch_overlap=config["patch_overlap"],
        model_dim=config["model_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_dim=config["mlp_dim"],
        dropout=config["dropout"],
        out_dim=config["out_dim"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_loop(model, opt, train_loader, val_loader, model_save_path, device, config)

    return




# Replace training loop with single call to sanity_test_pipeline()
if __name__ == "__main__":
    run_training_pipeline()


# maybe add 10 fold validation
# maybe EMA and KL stopping
# ADAMW optimizer
# check if volume is being normalized because louder songs may score better
# learning rate scheduler?
# redo the spectrograms with normilization
# does not work well with rap
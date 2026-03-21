# train_ast_va_fixed.py
"""
Train a compact AST-like transformer to predict valence & arousal from precomputed log-mel .npy spectrograms.
Drop this file next to your `dataspectrogram/` folder (which must include `_metadata.json` and .npy files).
Edit the USER SETTINGS section below before running.

Changes vs original:
- pos_embed created in __init__ (fixed shape using computed max_patches)
- collate_fn returns padding_mask and model accepts it so Transformer ignores padded patches
- deterministically seed DataLoader workers
- optional mixed precision (AMP) training with GradScaler
- gradient clipping and ReduceLROnPlateau scheduler
- a few robustness checks / dtype fixes
"""

import json
import math
import random
import re
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------- USER SETTINGS ----------------
DATA_DIR = "dataspectrogram"   # folder containing .npy files and _metadata.json
STATIC_CSV = "annotations/static_annotations_averaged_songs_1_2000.csv"  # path to CSV annotations
USE_GPU = True
SEED = 42

# model / training
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 20
NUM_WORKERS = 4
TARGET_SECONDS = 10.0   # crop length in seconds
PATCH_SIZE = 16
PATCH_OVERLAP = 6
MODEL_DIM = 768
NUM_LAYERS = 6 #paper uses 12
NUM_HEADS = 8 # paper uses 12
MLP_DIM = 2048
DROPOUT = 0.1

# misc training tweaks
USE_AMP = True            # enable mixed precision
GRAD_CLIP_NORM = 1.0     # clip gradients to this norm (set None to disable)
SCHEDULER_PATIENCE = 2   # for ReduceLROnPlateau
# -----------------------------------------------

# --- seeds / device
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
print("Device:", device)

# ---------------- CSV loading & expansion ----------------
def expand_id_field(s: str) -> List[str]:
    """Expand a CSV ID cell into a list of individual id strings.
       Supports values like "2,3", "5-7", "10", "2;3", "2 3", etc."""
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return []
    parts = re.split(r"[,;\/]", s)  # split on common separators
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # ranges "5-7" or "5:7"
        m = re.match(r"^(\d+)\s*[-:]\s*(\d+)$", p)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b:
                out.extend([str(i) for i in range(a, b+1)])
            else:
                out.extend([str(i) for i in range(b, a+1)])
            continue
        # multiple digits inside token: extract all numbers
        nums = re.findall(r"\d+", p)
        if nums:
            out.extend([str(int(x)) for x in nums])
            continue
        # fallback: the token itself
        out.append(p)
    # dedupe preserve order
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)
    return res

def infer_annotation_columns(df: pd.DataFrame) -> Tuple[str,str,str]:
    """Infer id, valence_mean, arousal_mean columns (or approximate)."""
    cols = [c.lower() for c in df.columns]
    # id
    id_candidates = ["song_id","id","track_id","track","filename","file"]
    for c in id_candidates:
        for col in df.columns:
            if col.lower() == c:
                id_col = col
                break
        else:
            continue
        break
    else:
        id_col = df.columns[0]
    # val/arousal
    val_col = None; aro_col = None
    for col in df.columns:
        low = col.lower()
        if "valence" in low and "mean" in low:
            val_col = col
        if "arousal" in low and "mean" in low:
            aro_col = col
    # fallback to any val/arousal-like tokens
    if val_col is None:
        for col in df.columns:
            if "val" in col.lower() and "std" not in col.lower():
                val_col = col; break
    if aro_col is None:
        for col in df.columns:
            if "aro" in col.lower() or "arous" in col.lower():
                aro_col = col; break
    # last resort: take first two numeric columns not id
    if val_col is None or aro_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        numeric_cols = [c for c in numeric_cols if c != id_col]
        if len(numeric_cols) >= 2:
            val_col, aro_col = numeric_cols[0], numeric_cols[1]
        else:
            raise RuntimeError("Couldn't infer valence/arousal columns. Edit STATIC_CSV or set columns explicitly.")
    return id_col, val_col, aro_col

def load_annotations_expanded(csv_path: str) -> Dict[str, Tuple[float,float]]:
    """Load CSV, expand grouped ids and return mapping key->(valence,arousal)."""
    df = pd.read_csv(csv_path)
    id_col, val_col, aro_col = infer_annotation_columns(df)
    print(f"Inferred columns: id_col='{id_col}', valence='{val_col}', arousal='{aro_col}'")
    mapping: Dict[str, Tuple[float,float]] = {}
    for _, r in df.iterrows():
        ids = expand_id_field(r[id_col])
        if not ids:
            continue
        try:
            val = float(r[val_col])
            aro = float(r[aro_col])
        except Exception:
            # skip rows with invalid numbers
            continue
        for k in ids:
            mapping[str(k).strip()] = (val, aro)
    print(f"Loaded {len(mapping)} expanded annotation keys (after expansion).")
    return mapping

# ---------------- dataset ----------------
class SpectrogramVADataset(Dataset):
    def __init__(self, data_root: str, annotations: Dict[str, Tuple[float,float]], metadata_path: str,
                 fixed_seconds: float = TARGET_SECONDS, train: bool = True):
        self.root = Path(data_root)
        # load metadata
        meta = json.load(open(metadata_path, "r"))
        self.mean = np.array(meta["global_mean_per_bin"], dtype=np.float32)
        self.std = np.array(meta["global_std_per_bin"], dtype=np.float32)
        self.n_mels = int(meta["n_mels"])
        self.hop_length = int(meta["hop_length"])
        self.sr = int(meta.get("sr", 22050))
        # compute frames per second and fixed frames
        fps = float(self.sr) / float(self.hop_length)
        self.fixed_frames = int(round(fixed_seconds * fps))
        self.annotations = annotations
        # gather files but filter to those present in annotations (robust matching)
        all_files = sorted([p for p in self.root.rglob("*.npy") if p.name != "_metadata.json"])
        self.files = []
        for p in all_files:
            if self._has_annotation(p):
                self.files.append(p)
        print(f"Dataset: {len(self.files)} labeled spectrogram files found (out of {len(all_files)} total).")
        self.train = train

    def __len__(self):
        return len(self.files)

    def _has_annotation(self, path: Path) -> bool:
        try:
            _ = self._match_annotation(path)
            return True
        except KeyError:
            return False

    def _match_annotation(self, path: Path) -> Tuple[float,float]:
        """Robust matching between file path and annotation keys (expanded)."""
        stem = path.stem
        filename = path.name
        rel = str(path.relative_to(self.root))
        keys = list(self.annotations.keys())
        keys_lower = {k.lower(): k for k in keys}

        candidates = []
        candidates.append(stem)
        candidates.append(filename)
        candidates.append(rel)
        # numeric collapse
        try:
            candidates.append(str(int(float(stem))))
        except Exception:
            pass
        candidates.append(stem.lstrip("0") or "0")
        # add keys that contain the stem as substring (useful when CSV grouped keys like "2,3")
        for k in keys:
            if stem.lower() in str(k).lower():
                candidates.append(k)
        # normalize punctuation matches
        def normalize(x):
            return re.sub(r"[:\-_,\s]+", " ", str(x)).strip().lower()
        norm_stem = normalize(stem)
        for k in keys:
            if normalize(k) == norm_stem:
                candidates.append(k)
        # check all candidates
        for c in candidates:
            if c in self.annotations:
                return self.annotations[c]
            if str(c).lower() in keys_lower:
                return self.annotations[keys_lower[str(c).lower()]]
        # final fuzzy numeric match
        m = re.search(r"(\d+)", stem)
        if m:
            d = m.group(1)
            for k in keys:
                mm = re.search(r"(\d+)", str(k))
                if mm and mm.group(1) == d:
                    return self.annotations[k]

        # fail with helpful debug sample
        sample_keys = list(keys)[:50]
        raise KeyError(f"No annotation for file {path}. Tried candidates: {candidates[:10]}. Sample keys: {sample_keys}")

    def _load_and_normalize(self, path: Path) -> np.ndarray:
        arr = np.load(str(path))
        if arr.ndim != 2 or arr.shape[0] != self.n_mels:
            raise RuntimeError(f"Unexpected spectrogram shape {arr.shape} for {path}")
        arr = (arr - self.mean[:, None]) / (self.std[:, None] + 1e-9)
        return arr.astype(np.float32)

    def _random_crop(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[1]
        if T == self.fixed_frames:
            return x
        if T < self.fixed_frames:
            pad = np.repeat(x[:, -1:], repeats=(self.fixed_frames - T), axis=1)
            return np.concatenate([x, pad], axis=1)
        if self.train:
            start = np.random.randint(0, T - self.fixed_frames + 1)
        else:
            start = (T - self.fixed_frames) // 2
        return x[:, start:start + self.fixed_frames]

    def __getitem__(self, idx: int):
        p = self.files[idx]
        va = self._match_annotation(p)
        x = self._load_and_normalize(p)
        x = self._random_crop(x)
        if self.train:
            x = spec_augment(x, max_mask_time=0.08, max_mask_freq=0.15)
        # return float32 numpy array (make_patches will convert to tensor)
        return x.astype(np.float32), torch.tensor(va, dtype=torch.float32)

def spec_augment(spec: np.ndarray, max_mask_time=0.1, max_mask_freq=0.2) -> np.ndarray:
    n_mels, T = spec.shape
    # freq mask
    freq_mask_len = int(n_mels * random.uniform(0.0, max_mask_freq))
    if freq_mask_len > 0 and n_mels > 0:
        f0 = random.randint(0, max(0, n_mels - freq_mask_len))
        spec[f0:f0+freq_mask_len, :] = spec.mean()
    # time mask
    time_mask_len = int(T * random.uniform(0.0, max_mask_time))
    if time_mask_len > 0 and T > 0:
        t0 = random.randint(0, max(0, T - time_mask_len))
        spec[:, t0:t0+time_mask_len] = spec.mean()
    return spec

def _pad_reflect_2d(t: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """
    Reflect-pad a 2D tensor `t` (H, W) by (pad_h rows, pad_w cols) on the bottom/right.
    Returns padded 2D tensor.
    """
    t4 = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pad = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    t4p = F.pad(t4, pad, mode='reflect')
    return t4p.squeeze(0).squeeze(0)  # back to (H2, W2)

def make_patches(spec: torch.Tensor, patch_size: int = PATCH_SIZE, overlap: int = PATCH_OVERLAP) -> torch.Tensor:
    """
    Convert spectrogram (n_mels, frames) into flattened patches with overlap.
    Returns tensor shape (N_patches, patch_area).
    """
    if not torch.is_tensor(spec):
        spec = torch.from_numpy(spec)
    spec = spec.float()
    C, W = spec.shape
    stride = patch_size - overlap
    # compute pad amounts so tiling fits exactly
    if C >= patch_size:
        pad_h = (stride - ((C - patch_size) % stride)) % stride
    else:
        pad_h = patch_size - C
    if W >= patch_size:
        pad_w = (stride - ((W - patch_size) % stride)) % stride
    else:
        pad_w = patch_size - W

    if pad_h != 0 or pad_w != 0:
        spec_padded = _pad_reflect_2d(spec, pad_h=pad_h, pad_w=pad_w)
    else:
        spec_padded = spec

    C2, W2 = spec_padded.shape
    patches = []
    for top in range(0, C2 - patch_size + 1, stride):
        for left in range(0, W2 - patch_size + 1, stride):
            p = spec_padded[top:top+patch_size, left:left+patch_size].reshape(-1)
            patches.append(p)
    if len(patches) == 0:
        patches = [spec_padded.reshape(-1)]
    return torch.stack(patches, dim=0)  # (N_patches, patch_area)

# ---------------- Transformer / Model ----------------
class PatchTransformer(nn.Module):
    def __init__(self, patch_dim: int, model_dim: int = MODEL_DIM, n_layers: int = NUM_LAYERS,
                 n_heads: int = NUM_HEADS, mlp_dim: int = MLP_DIM, dropout: float = DROPOUT,
                 max_patches: int = 256):
        super().__init__()
        self.model_dim = model_dim
        self.patch_proj = nn.Linear(patch_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        # pre-allocate pos embed for maximum sequence length (cls + max_patches)
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + max_patches, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads,
                                                   dim_feedforward=mlp_dim, dropout=dropout,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, patches: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        patches: (B, N, Dpatch)
        padding_mask: (B, N) with True for padded patch positions (NOT including cls token)
        returns: cls token embedding (B, model_dim)
        """
        B, N, D = patches.shape
        x = self.patch_proj(patches)  # (B, N, model_dim)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, model_dim)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, model_dim)
        # slice pos_embed to appropriate length and move to device
        if self.pos_embed.shape[1] < (N + 1):
            raise RuntimeError(f"pos_embed length {self.pos_embed.shape[1]} too small for N+1={N+1}. "
                               "Increase max_patches when constructing model.")
        x = x + self.pos_embed[:, : (N + 1), :].to(x.device)
        # Transformer expects input shape (seq_len, batch, dim)
        # src_key_padding_mask has shape (batch, seq_len) with True for positions that should be masked
        if padding_mask is not None:
            # padding_mask: (B, N) -> expand to include cls token at front (cls is never padded)
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, padding_mask.to(x.device)], dim=1)  # (B, N+1)
        else:
            src_key_padding_mask = None
        x = x.transpose(0, 1)  # (seq_len, batch, dim)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (seq_len, batch, dim)
        return x[0]  # cls token (batch, dim)

class ASTRegressor(nn.Module):
    def __init__(self, patch_area: int, max_patches: int = 256):
        super().__init__()
        self.backbone = PatchTransformer(patch_dim=patch_area, max_patches=max_patches)
        self.head = nn.Sequential(
            nn.LayerNorm(MODEL_DIM),
            nn.Linear(MODEL_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

    def forward(self, patches: torch.Tensor, padding_mask: torch.Tensor = None):
        x = self.backbone(patches, padding_mask=padding_mask)
        return self.head(x)

# ---------------- collate & metrics ----------------
def collate_fn(batch):
    """
    batch: list of (spec_numpy (n_mels, frames), y tensor (2,))
    returns: patches_tensor (B, maxN, patch_dim), ys_t (B,2), padding_mask (B, maxN) where True indicates padded patch
    """
    specs, ys = zip(*batch)  # specs: each numpy array (n_mels, frames)
    patches_list = [make_patches(s, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP) for s in specs]
    N_list = [p.shape[0] for p in patches_list]
    maxN = max(N_list)
    padded = []
    padding_masks = []
    for p, n in zip(patches_list, N_list):
        if p.shape[0] < maxN:
            pad = torch.zeros((maxN - p.shape[0], p.shape[1]), dtype=p.dtype)
            p2 = torch.cat([p, pad], dim=0)
            # False = real token, True = padded token
            mask = torch.tensor([False]*p.shape[0] + [True]*(maxN - p.shape[0]), dtype=torch.bool)
        else:
            p2 = p
            mask = torch.tensor([False]*maxN, dtype=torch.bool)
        padded.append(p2)
        padding_masks.append(mask)
    patches_tensor = torch.stack(padded, dim=0)  # (B, maxN, patch_dim)
    padding_mask = torch.stack(padding_masks, dim=0)  # (B, maxN) True for padded
    ys_t = torch.stack([y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.float32) for y in ys], dim=0)
    return patches_tensor, ys_t, padding_mask

# ---------------- training loop ----------------
def worker_init_fn(worker_id):
    # deterministic seeds for workers
    worker_seed = SEED + worker_id + 1
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def compute_max_patches_from_specs(n_mels: int, fixed_frames: int, patch_size: int = PATCH_SIZE, overlap: int = PATCH_OVERLAP) -> int:
    stride = patch_size - overlap
    # number along mel axis
    if n_mels >= patch_size:
        nx = ((n_mels - patch_size) + stride) // stride
        nx = nx + 1
    else:
        nx = 1
    # number along time axis
    if fixed_frames >= patch_size:
        ny = ((fixed_frames - patch_size) + stride) // stride
        ny = ny + 1
    else:
        ny = 1
    return int(nx * ny)

def main():
    # load + expand annotations
    ann_map = load_annotations_expanded(STATIC_CSV)
    meta_path = Path(DATA_DIR) / "_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata at {meta_path}")
    # dataset (train) and a separate dataset for val (train=False) to control cropping
    full_ds = SpectrogramVADataset(DATA_DIR, ann_map, str(meta_path), fixed_seconds=TARGET_SECONDS, train=True)
    n = len(full_ds)
    if n == 0:
        raise RuntimeError("No labeled spectrogram files found. Check CSV and dataspectrogram contents.")
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(0.8 * n)
    train_idx, val_idx = idxs[:split], idxs[split:]
    train_set = torch.utils.data.Subset(full_ds, train_idx)
    # val set uses same underlying files but deterministic cropping
    val_ds = SpectrogramVADataset(DATA_DIR, ann_map, str(meta_path), fixed_seconds=TARGET_SECONDS, train=False)
    val_set = torch.utils.data.Subset(val_ds, val_idx)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn, worker_init_fn=worker_init_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn, worker_init_fn=worker_init_fn, pin_memory=True)

    # build model using dataset metadata to compute patch area and max patches
    sample_spec, _ = full_ds[0]
    sample_patches = make_patches(sample_spec, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP)
    patch_area = sample_patches.shape[1]
    # compute a safe max_patches analytically based on metadata (no heavy loop)
    max_patches = compute_max_patches_from_specs(full_ds.n_mels, full_ds.fixed_frames, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP)
    # add a margin
    max_patches = max(16, max_patches + 8)

    print("Patch area:", patch_area, "sample num patches:", sample_patches.shape[0], "max_patches:", max_patches)

    model = ASTRegressor(patch_area=patch_area, max_patches=max_patches).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=0.5, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        it = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}")
        for patches, ys, padding_mask in pbar:
            patches = patches.to(device)
            ys = ys.to(device)
            padding_mask = padding_mask.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                preds = model(patches, padding_mask=padding_mask)
                loss = criterion(preds, ys)
            scaler.scale(loss).backward()
            # gradient clipping
            if GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            it += 1
            pbar.set_postfix({"train_loss": f"{running / max(1, it):.4f}"})
        train_loss = running / max(1, it)

        # validation
        model.eval()
        vloss = 0.0
        vsteps = 0
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for patches, ys, padding_mask in tqdm(val_loader, desc="Validate"):
                patches = patches.to(device)
                ys = ys.to(device)
                padding_mask = padding_mask.to(device)
                with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                    preds = model(patches, padding_mask=padding_mask)
                    loss = criterion(preds, ys)
                vloss += loss.item()
                vsteps += 1
                all_preds.append(preds.cpu())
                all_trues.append(ys.cpu())
        val_loss = vloss / max(1, vsteps)
        preds_cat = torch.cat(all_preds, dim=0).numpy()
        trues_cat = torch.cat(all_trues, dim=0).numpy()
        if preds_cat.shape[0] > 1:
            # guard against NaNs
            try:
                p_val = np.corrcoef(preds_cat[:,0], trues_cat[:,0])[0,1]
            except Exception:
                p_val = float("nan")
            try:
                p_aro = np.corrcoef(preds_cat[:,1], trues_cat[:,1])[0,1]
            except Exception:
                p_aro = float("nan")
        else:
            p_val = p_aro = float("nan")
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} pearson_val={p_val:.4f} pearson_aro={p_aro:.4f}")

        # scheduler step
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_ast_va.pt")
            print("Saved best model -> best_ast_va.pt")

    print("Done. best val loss:", best_val)

if __name__ == "__main__":
    main()

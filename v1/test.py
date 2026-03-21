# predict_va_from_mp3.py
"""
Predict valence & arousal for an MP3 using a trained checkpoint `best_ast_va.pt`
and the preprocessing metadata `_metadata.json` produced by make_mels.

Edit INPUT_MP3 / MODEL_PATH / METADATA_PATH below, then run:
    python predict_va_from_mp3.py

Outputs per-window and aggregated predictions.

This version includes:
- deterministic inference flags (seed + cudnn deterministic)
- safer checkpoint key mapping (strips common prefixes)
- automatic copying of checkpoint `pos_embed` / `cls_token` into the model
  so they are not randomly initialized at first forward
- forces FP32 inference to avoid AMP variance
"""
import json
from pathlib import Path
import numpy as np
import math
import random
import warnings

import torch
import torch.nn as nn

# ---------------- Deterministic / precision configuration ----------------
# If you need maximum reproducibility, enable the deterministic settings below.
# WARNING: may slow down GPU inference.
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# If your PyTorch version supports it and you want stricter behavior, you can:
# torch.use_deterministic_algorithms(True)

# Ensure we do inference in FP32 (avoid AMP/autocast)
# Do NOT use torch.cuda.amp.autocast() in this script if you require reproducibility.
# ---------------------------------------------------------------------------

# optional faster mp3 decode without ffmpeg
try:
    import miniaudio
    HAVE_MINIAUDIO = True
except Exception:
    HAVE_MINIAUDIO = False

import librosa
import torch.nn.functional as F

# ---------------- USER PATHS ----------------
INPUT_MP3 = "songs/clarity.mp3"  # <-- set this
MODEL_PATH = "best_ast_va(840val).pt"  # <-- set if different
METADATA_PATH = "dataspectrogram/_metadata.json"  # <-- metadata produced during precompute
# -------------------------------------------

# -------- model / patching hyperparams (must match training) ----
PATCH_SIZE = 16
PATCH_OVERLAP = 6
TARGET_SECONDS = 10.0  # same as training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------------------------

# ---------------- audio decode utilities -------------------------
def decode_mp3_miniaudio(path: Path, sr: int):
    """
    Decode using miniaudio (if available) and return a mono float32 numpy array at sample rate sr.
    This handles different miniaudio return shapes/fields across versions.
    """
    try:
        # prefer decode_file which takes a filename
        if hasattr(miniaudio, "decode_file"):
            dec = miniaudio.decode_file(str(path), sample_rate=sr)
        elif hasattr(miniaudio, "decode"):
            dec = miniaudio.decode(str(path), sample_rate=sr)
        else:
            raise RuntimeError("miniaudio decode function not found")

        # dec may be a tuple (samples, sr) or an object with attributes
        # Normalize to numpy array samples and sample rate
        if isinstance(dec, tuple) and len(dec) == 2:
            samples, sr_orig = dec
            y = np.asarray(samples)
        elif hasattr(dec, "samples"):
            y = np.asarray(dec.samples)
            sr_orig = getattr(dec, "sample_rate", sr)
        elif hasattr(dec, "data"):
            y = np.asarray(dec.data)
            sr_orig = getattr(dec, "sample_rate", sr)
        else:
            # fallback: try to coerce the returned object to array
            arr = np.asarray(dec)
            if arr.size == 0:
                raise RuntimeError("miniaudio returned empty data")
            y = arr
            sr_orig = sr

        # If stereo or multiple channels, average to mono
        if y.ndim > 1:
            # assume shape (samples, channels) or (channels, samples)
            if y.shape[0] < y.shape[1]:
                # (channels, samples) -> average axis 0
                y = y.mean(axis=0)
            else:
                # (samples, channels) -> average axis 1
                y = y.mean(axis=1)

        # Convert integer PCM to float range [-1,1] if needed
        if np.issubdtype(y.dtype, np.integer):
            max_val = float(np.iinfo(y.dtype).max)
            y = y.astype("float32") / max_val
        else:
            y = y.astype("float32")

        # If decode already returned at desired sr, skip resample; otherwise resample
        sr_orig = int(sr_orig) if sr_orig is not None else sr
        if sr_orig != sr:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)

        return y.astype("float32")

    except Exception as e:
        warnings.warn(f"miniaudio failed to decode; falling back to librosa: {e}")
        raise


def load_audio(path: Path, sr: int):
    """
    Try miniaudio first (fast); on any failure fallback to librosa using the file path.
    """
    if HAVE_MINIAUDIO:
        try:
            return decode_mp3_miniaudio(path, sr)
        except Exception:
            # decode_mp3_miniaudio already issued a warning
            pass
    # guaranteed fallback: pass the filename to librosa (never pass decoded objects)
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype("float32")


# -----------------------------------------------------------------

# ---------------- patch helper (reflect pad) ---------------------
def _pad_reflect_2d(t: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    # t: (H, W)
    t4 = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    pad = (0, pad_w, 0, pad_h)  # left,right,top,bottom
    t4p = F.pad(t4, pad, mode="reflect")
    return t4p.squeeze(0).squeeze(0)


def make_patches(spec: torch.Tensor, patch_size: int = PATCH_SIZE, overlap: int = PATCH_OVERLAP) -> torch.Tensor:
    """Spec (n_mels, frames) -> patches tensor (N_patches, patch_area)"""
    if not torch.is_tensor(spec):
        spec = torch.from_numpy(spec)
    spec = spec.float()
    C, W = spec.shape
    stride = patch_size - overlap
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
            p = spec_padded[top:top + patch_size, left:left + patch_size].reshape(-1)
            patches.append(p)
    if len(patches) == 0:
        patches = [spec_padded.reshape(-1)]
    return torch.stack(patches, dim=0)  # (N, patch_area)


# -----------------------------------------------------------------

# ---------------- model classes (must match training file) -------
class PatchTransformer(nn.Module):
    def __init__(self, patch_dim: int, model_dim: int = 768, n_layers: int = 6, n_heads: int = 8,
                 mlp_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.patch_proj = nn.Linear(patch_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        # pos_embed will be set either from checkpoint copy or initialized if needed during forward
        self.pos_embed = None
        # ensure transformer operates with batch-first tensors (B, S, E)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, N, patch_dim)
        B, N, D = patches.shape
        x = self.patch_proj(patches)  # (B, N, model_dim)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, model_dim)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, model_dim)
        # ensure pos_embed exists and is on the correct device
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != x.shape[1]):
            # If pos_embed wasn't set from checkpoint, initialize deterministically using the global seed state
            self.pos_embed = nn.Parameter(torch.randn(1, x.shape[1], self.model_dim).to(x.device))
        x = x + self.pos_embed  # (B, N+1, model_dim)
        # with batch_first=True the transformer expects (B, S, E) so do not transpose
        x = self.transformer(x)  # (B, N+1, model_dim)
        # return the cls token embedding for the batch: (B, model_dim)
        return x[:, 0, :]


class ASTRegressor(nn.Module):
    def __init__(self, patch_area: int):
        super().__init__()
        self.backbone = PatchTransformer(patch_dim=patch_area)
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

    def forward(self, patches: torch.Tensor):
        x = self.backbone(patches)
        return self.head(x)


# -----------------------------------------------------------------

# ---------------- main predict routine -----------------------------
def predict_on_mp3(mp3_path: str, model_path: str, metadata_path: str, target_seconds: float = TARGET_SECONDS,
                   stride_seconds: float = None):
    # load metadata
    meta = json.load(open(metadata_path, "r"))
    sr = int(meta.get("sr", 22050))
    n_mels = int(meta["n_mels"])
    n_fft = int(meta["n_fft"])
    hop_length = int(meta["hop_length"])
    power = float(meta.get("power", 2.0))
    ref = float(meta.get("ref", 1.0))
    mean = np.array(meta["global_mean_per_bin"], dtype=np.float32)
    std = np.array(meta["global_std_per_bin"], dtype=np.float32)

    # load audio
    p = Path(mp3_path)
    if not p.exists():
        raise FileNotFoundError(mp3_path)
    y = load_audio(p, sr=sr)

    # compute spectrogram (full)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                         power=power)
    mel_db = librosa.power_to_db(mel, ref=ref)  # shape (n_mels, T)
    # normalize
    mel_norm = (mel_db - mean[:, None]) / (std[:, None] + 1e-9)

    # windowing
    fps = float(sr) / float(hop_length)  # frames per second
    win_frames = int(round(target_seconds * fps))
    if stride_seconds is None:
        stride_seconds = target_seconds / 2.0
    hop_frames = int(round(stride_seconds * fps))
    T = mel_norm.shape[1]
    starts = list(range(0, max(1, T - win_frames + 1), hop_frames))
    if len(starts) == 0:
        starts = [0]
    # prepare model
    # build a sample patch to get patch_area
    # single window sample (pad if needed)
    def get_window(i):
        s = starts[i]
        e = s + win_frames
        if e <= T:
            w = mel_norm[:, s:e]
        else:
            # pad by repeating last frame
            pad_len = e - T
            pad = np.repeat(mel_norm[:, -1:], repeats=pad_len, axis=1)
            w = np.concatenate([mel_norm[:, s:T], pad], axis=1)
        return w.astype(np.float32)

    sample_w = get_window(0)
    sample_patches = make_patches(torch.from_numpy(sample_w), patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP)
    patch_area = sample_patches.shape[1]

    # ---------------- prepare & load model (robust auto-instantiation) ----------------
    model = ASTRegressor(patch_area=patch_area).to(DEVICE)

    # load raw checkpoint
    ckpt_raw = torch.load(model_path, map_location=DEVICE)

    # If the checkpoint is a dict containing a state-dict under common keys, extract it
    if isinstance(ckpt_raw, dict):
        if "state_dict" in ckpt_raw:
            state_dict = ckpt_raw["state_dict"]
        elif "model_state_dict" in ckpt_raw:
            state_dict = ckpt_raw["model_state_dict"]
        else:
            # assume the dict *is* the state_dict mapping
            state_dict = ckpt_raw
    else:
        # fallback: loaded object is a state_dict
        state_dict = ckpt_raw

    # --- Try to copy pos_embed / cls_token from checkpoint into the model before load ---
    # This prevents random initialization of those params when they are present in the checkpoint.
    def copy_ckpt_param_into_model(sd, model, basename):
        """
        sd: checkpoint state_dict
        model: instantiated model
        basename: e.g. "pos_embed" or "cls_token"
        """
        for ck_key, v in sd.items():
            if ck_key == basename or ck_key.endswith("." + basename):
                try:
                    param = nn.Parameter(torch.empty_like(v).to(DEVICE))
                    with torch.no_grad():
                        param.data.copy_(v.to(DEVICE))
                    # try attaching to backbone first
                    if hasattr(model, "backbone") and hasattr(model.backbone, basename):
                        setattr(model.backbone, basename, param)
                        print(f"Copied checkpoint '{ck_key}' into model.backbone.{basename}")
                    elif hasattr(model, basename):
                        setattr(model, basename, param)
                        print(f"Copied checkpoint '{ck_key}' into model.{basename}")
                    else:
                        # if no attribute, skip (will be created on forward)
                        print(f"Found checkpoint key '{ck_key}' but model has no attribute '{basename}' to attach to.")
                except Exception as e:
                    print(f"Could not copy checkpoint key {ck_key}: {e}")
                break  # handle only the first matching key

    copy_ckpt_param_into_model(state_dict, model, "pos_embed")
    copy_ckpt_param_into_model(state_dict, model, "cls_token")
    copy_ckpt_param_into_model(state_dict, model, "class_token")

    # safer filtering: strip common leading prefixes like "module." or "model."
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            stripped[k[len("module."):]] = v
        elif k.startswith("model."):
            stripped[k[len("model."):]] = v
        else:
            stripped[k] = v

    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in stripped.items() if k in model_keys}

    # Load filtered state dict non-strictly (so missing keys are allowed)
    load_res = model.load_state_dict(filtered, strict=False)

    # Force FP32 (avoid AMP) for deterministic numeric behavior
    model = model.float()
    model.to(DEVICE)

    print("Model load summary:")
    print("  Missing keys:", load_res.missing_keys)
    print("  Unexpected keys ignored:", load_res.unexpected_keys)
    # -----------------------------------------------------------------------------------

    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(len(starts)):
            w = get_window(i)
            patches = make_patches(torch.from_numpy(w), patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP)  # (N, patch_area)
            patches = patches.unsqueeze(0).to(DEVICE)  # (1, N, patch_area)
            # ensure input dtype is float32
            patches = patches.float()
            out = model(patches)  # (1,2)
            preds.append(out.cpu().numpy()[0])

    preds = np.stack(preds, axis=0)  # (num_windows, 2)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return {
        "per_window": preds,
        "mean": mean_pred,
        "std": std_pred,
        "starts_frames": starts,
        "frames_per_second": fps,
        "target_seconds": target_seconds
    }


# ---------------- CLI --------------------------
if __name__ == "__main__":
    res = predict_on_mp3(INPUT_MP3, MODEL_PATH, METADATA_PATH)
    print("Per-window predictions (valence, arousal):")
    for i, (v, a) in enumerate(res["per_window"]):
        start_s = res["starts_frames"][i] / res["frames_per_second"]
        print(f" window {i:02d} start {start_s:.1f}s -> valence={v:.3f}, arousal={a:.3f}")
    print("Aggregated (mean ± std):")
    print(f" valence = {res['mean'][0]:.3f} ± {res['std'][0]:.3f}")
    print(f" arousal = {res['mean'][1]:.3f} ± {res['std'][1]:.3f}")

    valence_mean = (res['mean'][0] - 5) / 4
    valence_std = res['std'][0] / 4

    arousal_mean = (res['mean'][1] - 5) / 4
    arousal_std = res['std'][1] / 4

    print(f" SCALED valence = {valence_mean:.3f} ± {valence_std:.3f}")
    print(f" SCALED arousal = {arousal_mean:.3f} ± {arousal_std:.3f}")

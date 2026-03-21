import os
from typing import Any, Optional, Tuple

import numpy as np
import librosa

import torch
import torch.nn as nn 
import torch.nn.functional as F

import matplotlib.pyplot as plt

import json

import math

# PATHS
SONG_PATH = "songs/sparks.mp3"
MODEL = "large"

MODEL_PATH = "models/" + MODEL + "/model.pt"
MODEL_DATA_PATH = "models/" + MODEL + "/config.json"



DTYPE = np.float32

with open(MODEL_DATA_PATH, "r") as f:
    _DATA = json.load(f)

_METADATA = {
    "sr": _DATA["sample_rate"],
    "n_mels": _DATA["n_mels"],
    "n_fft": _DATA["n_fft"],
    "hop_length": _DATA["hop_length"],
    "power": _DATA["power"],
    "ref": _DATA["ref"],
}

WINDOW_SECONDS = _DATA["window_seconds"]  # window length in seconds
WINDOW_OVERLAP = _DATA["window_overlap"]  # window overlap in seconds



def compute_log_mel(audio_path: str) -> np.ndarray:
    """Load an audio file and compute the log-mel spectrogram exactly as in the original script.

    Args:
        audio_path: Path to the input audio (mp3, wav, etc.)

    Returns:
        log_mel: np.ndarray of shape (N_MELS, T), dtype float32, in dB (power_to_db with ref=1.0)
    """
    # Load audio (mono, resampled to SR)
    y, used_sr = librosa.load(audio_path, sr=_METADATA["sr"], mono=True)

    # Compute mel-spectrogram (power)
    mel = librosa.feature.melspectrogram(
        y = y / (np.sqrt(np.mean(y**2)) + 1e-8),  # normalize RMS to 1.0
        sr=used_sr,
        n_fft=_METADATA["n_fft"],
        hop_length=_METADATA["hop_length"],
        n_mels=_METADATA["n_mels"],
        power=_METADATA["power"],
    )

    # Convert power to dB using the same reference as the original file
    log_mel = librosa.power_to_db(mel, ref=_METADATA["ref"])
    # Ensure dtype
    return log_mel.astype(DTYPE, copy=False)


def mp3_to_npy(mp3_path: str, out_npy: Optional[str] = None) -> np.ndarray:
    """Compute the log-mel spectrogram for an MP3 and optionally save to .npy.

    Args:
        mp3_path: path to input mp3
        out_npy: if provided, save the resulting array to this path

    Returns:
        np.ndarray: log-mel spectrogram (N_MELS, T) as float32
    """
    spec = compute_log_mel(mp3_path)

    if out_npy:
        # create parent dirs if necessary
        os.makedirs(os.path.dirname(os.path.abspath(out_npy)), exist_ok=True)
        np.save(out_npy, spec)

    return spec

def _ms_to_frame_indices(start_ms: int, end_ms: int, sr: int, hop_length: int) -> Tuple[int, int]:
    start = int(np.floor(start_ms/1000.0 * sr / hop_length))
    end   = int(np.ceil (end_ms  /1000.0 * sr / hop_length))
    return max(0, start), max(0, end)

def get_spectrogram_chunk(
    arr: np.ndarray,
    start_ms: int,
    end_ms: int,
    sr: int = _METADATA["sr"],
    hop: int = _METADATA["hop_length"],
    mmap: bool = False,
) -> Optional[np.ndarray]:
    """
    Returns a heap-owned, contiguous np.float32 array shaped (n_mels, n_frames_slice),
    or None on error (and prints a clear message).

    This version normalizes the returned slice to the expected number of frames
    for the requested (start_ms, end_ms) window by trimming or padding with zeros.
    """

    if start_ms >= end_ms:
        print("Invalid time range: start_ms must be < end_ms")
        return None

    file_shapes = arr.shape

    # expected number of frames for the requested window (float -> ceil)
    expected_frames = int(np.ceil((end_ms - start_ms) / 1000.0 * sr / hop))


    # convert ms -> frame indices (current logic)
    sframe, eframe = _ms_to_frame_indices(start_ms, end_ms, sr, hop)


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

def plot_spectrogram(spec: np.ndarray, title: str = "Spectrogram", sr: int = _METADATA["sr"], hop_length: int = _METADATA["hop_length"]) -> None:
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




import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# keep your global defaults or define them above as needed:
# PATCH_SIZE, PATCH_OVERLAP, MODEL_DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, DROPOUT

class SimpleASTRegressor(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_frames: int,
        patch_size: int,
        patch_overlap: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
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







if __name__ == "__main__":
    # LOAD MODEL
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = SimpleASTRegressor(
        n_mels=_DATA["n_mels"], 
        n_frames=_DATA["n_frames"], 
        patch_size=_DATA["patch_size"], 
        patch_overlap=_DATA["patch_overlap"], 
        model_dim=_DATA["model_dim"],
        num_layers=_DATA["num_layers"], 
        num_heads=_DATA["num_heads"], 
        mlp_dim=_DATA["mlp_dim"]
        ).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    print("Loaded model from", MODEL_PATH)



    # PROCESS A SONG
    spec = mp3_to_npy(SONG_PATH)

    song_length_ms = spec.shape[1] * _METADATA["hop_length"] / _METADATA["sr"] * 1000.0

    print(f"Loaded song from {SONG_PATH} of length (ms): {song_length_ms}")



    valence_list = []
    arousal_list = []

    for window in range(0, int(song_length_ms - WINDOW_SECONDS * 1000) + 1, int((WINDOW_SECONDS - WINDOW_OVERLAP) * 1000)):
        start_ms = window
        end_ms = window + int(WINDOW_SECONDS * 1000)

        chunk = get_spectrogram_chunk(arr=spec, start_ms=start_ms, end_ms=end_ms)
        if chunk is None:
            print(f"Failed to get spectrogram chunk for window {start_ms}-{end_ms} ms")
            continue

        # convert to tensor and add batch/channel dims
        input_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_mels, n_frames)

        with torch.no_grad():
            output = model(input_tensor)  # (1, 2)
            valence, arousal = output[0].cpu().numpy()

            print(
                f"Window {start_ms // 1000:>4}-{end_ms // 1000:<4} s | "
                f"Valence = {valence:>7.4f} | Arousal = {arousal:>7.4f}"
            )

            valence_list.append(valence)
            arousal_list.append(arousal)

    print("Finished processing song.")
    print(f"    Valence predicted: {np.mean(valence_list)} ± {np.std(valence_list)}")
    print(f"    Arousal predicted: {np.mean(arousal_list)} ± {np.std(arousal_list)}")

    
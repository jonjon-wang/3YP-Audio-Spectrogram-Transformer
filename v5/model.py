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

from config import Hyperparameters


class AudioSpectrogramTransformer(nn.Module):
    """
    An implementation of the Audio Spectrogram Transformer (AST) architecture for regression tasks.
    """
    def __init__(self, hyperparameters: Hyperparameters | None = None):
        super().__init__()

        if hyperparameters is None:
            hyperparameters = Hyperparameters()
        self.hyperparameters = hyperparameters

        self.n_mels = hyperparameters.metadata["n_mels"]
        timestep_ms = self.hyperparameters.metadata["hop_length"] / self.hyperparameters.metadata["sr"] * 1000
        self.n_frames = int(self.hyperparameters.window_seconds_ms // timestep_ms)
        self.patch_size = hyperparameters.patch_size
        self.patch_overlap = hyperparameters.patch_overlap
        self.patch_stride = max(1, self.patch_size - self.patch_overlap)

        self.n_patches_h = (self.n_mels - self.patch_size) // self.patch_stride + 1
        self.n_patches_w = (self.n_frames - self.patch_size) // self.patch_stride + 1
        if self.n_patches_h <= 0 or self.n_patches_w <= 0:
            raise ValueError("Patch size too large for spectrogram shape")



            #Think about maybe loading metadata into hyperparametrs i think that makes more sense

        


class SimpleASTRegressor(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_frames: int,
        patch_size: int = 1, #PATCH_SIZE,
        patch_overlap: int =1, #PATCH_OVERLAP,
        model_dim: int = 1, #MODEL_DIM,
        num_layers: int = 1, #NUM_LAYERS,
        num_heads: int = 1, #NUM_HEADS,
        mlp_dim: int = 1, #MLP_DIM,
        dropout: float = 0.1, #DROPOUT,
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

# IMPORTS
from dataclasses import dataclass, field
from typing import Any, Optional
import math
import torch
import os
import json


# HYPERPARAMETERS AND CONFIGURATION
@dataclass(frozen = True)
class Hyperparameters:
    '''
    This class holds all the hyperparameters and configuration settings for the project.
    '''

    # FOLDERS AND PATHS
    spectrogram_dir: str = "data/dataspectrogram"   # folder containing .npy files
    metadata_path: str = "data/dataspectrogram/_metadata.json"  # path to metadata JSON file

    arousal_csv: str = "data/dynamic_annotations/arousal.csv"
    valence_csv: str = "data/dynamic_annotations/valence.csv"

    model_save_dir: str = "models/best_model"

    # TRAINING HYPERPARAMETERS
    val_fraction: float = 0.1
    batch_size: int = 8
    epochs: int = 20
    early_stopping_patience: int = 5  # set to integer to enable early stopping

    lr: float = 1e-5
    weight_decay: float = 1e-2

    max_workers: int = 6   # maximum number of workers for data loading (set to 0 to disable multiprocessing)
    workers: int = field(init=False)
    num_workers_train: int = field(init=False)
    num_workers_val: int = field(init=False)

    use_gpu: bool = True
    device: torch.device = field(init=False)
    use_amp: bool = field(init=False)
    pin_memory: bool = field(init=False)

    grad_clip_norm: Optional[float] = 1.0     # clip gradients to this norm (set None to disable)
    scheduler_patience: int = 2   # for ReduceLROnPlateau
    seed: int | None = 42   # set to integer for reproducibility, or None to disable

    # MODEL HYPERPARAMETERS
    window_seconds_ms: int = 4000
    window_overlap_ms: int = 2000

    patch_size: int = 16 # paper uses 16
    patch_overlap: int = 6 # paper uses 6
    model_dim: int = 768 # paper uses 768
    num_layers: int = 12 # paper uses 12
    num_heads: int = 12 # paper uses 12
    mlp_dim: int = 3072 # paper uses 3072
    dropout: float = 0.1 # paper uses 0.1

    # DATA AUGMENTATION HYPERPARAMETERS
    bins_per_axis: int = 12
    
    crop_valence: tuple[float, float] = (-1.0, 1.0) 
    crop_arousal: tuple[float, float] = (-1.0, 1.0) 
    max_magnitude: float = 1.0 # do not change this one for meaningful tests as it alters the input data distribution

    # METADATA
    metadata: dict = field(init=False)

    def __post_init__(self):
        if os.cpu_count():
            cpu = os.cpu_count()
        else:
            cpu = 4

        workers = max(0, min(self.max_workers, cpu - 2))
        num_workers_train = max(0, min(math.floor(workers * 2 / 3), 6))
        num_workers_val = max(0, min(workers - math.floor(workers * 2 / 3), 3))

        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "num_workers_train", num_workers_train)
        object.__setattr__(self, "num_workers_val", num_workers_val)

        if self.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            use_amp = True
            pin_memory = True
        elif self.use_gpu and torch.backends.mps.is_available():
            device = torch.device("mps")
            use_amp = False
            pin_memory = False
        else:
            device = torch.device("cpu")
            use_amp = False
            pin_memory = False

        object.__setattr__(self, "device", device)
        object.__setattr__(self, "use_amp", use_amp)
        object.__setattr__(self, "pin_memory", pin_memory)

        object.__setattr__(self, "metadata", self._load_metadata())

    def _load_metadata(self) -> dict[str, Any]:
        """
        Loads metadata from JSON file and validates required keys. Returns metadata dict.
        """

        path = self.metadata_path

        metadata = {}

        try:
            with open(path, "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(path)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}") from e

        for k in ("sr", "hop_length", "file_shapes", "n_fft", "power", "ref"):
            if k not in metadata:
                raise KeyError(f"'{k}' missing from metadata")
            
        return metadata
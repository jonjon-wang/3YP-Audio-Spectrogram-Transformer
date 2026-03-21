# IMPORTS
import json
import random
import numpy as np
import pandas as pd
from typing import Iterable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from config import Hyperparameters



# UTILITY FUNCTIONS
def plot_valence_arousal_histogram(data: dict[int, np.ndarray], bins: int = 8) -> None:
    """
    Plots separate histograms for valence and arousal distributions.
    """

    arr = np.array(data)

    if arr.shape[1] != 3:
        raise ValueError("Each element must be (song_id, valence, arousal)")

    valence = arr[:, 1].astype(float)
    arousal = arr[:, 2].astype(float)

    plt.figure()
    plt.xlim(-1.1, 1.1)
    plt.hist(valence, bins=bins)
    plt.title("Average Valence Distribution")
    plt.xlabel("Valence")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    plt.xlim(-1.1, 1.1)
    plt.hist(arousal, bins=bins)
    plt.title("Average Arousal Distribution")
    plt.xlabel("Arousal")
    plt.ylabel("Frequency")
    plt.show()



def plot_valence_arousal_scatter(
        data: list[tuple[int, float, float]], 
        valence_bound: tuple[float, float] = (-1.1, 1.1), 
        arousal_bound: tuple[float, float] = (-1.1, 1.1)
    ) -> None:
    """
    Plots a 2D scatter of valence vs arousal.
    """

    values = np.array(data)

    if values.shape[1] != 3:
        raise ValueError("Each element must be (song_id, valence, arousal)")

    valence = values[:, 1]
    arousal = values[:, 2]

    plt.figure()
    plt.scatter(valence, arousal, s=10, alpha=0.3)
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Average Valence vs Average Arousal (2D Scatter)")

    plt.xlim(valence_bound[0], valence_bound[1])
    plt.ylim(arousal_bound[0], arousal_bound[1])

    plt.gca().set_aspect('equal', adjustable='box')

    plt.axhline(0)
    plt.axvline(0)
    plt.show()



def plot_spectrogram(
        spectrogram: np.ndarray,
        sr: int | None = 22050,
        hop_length: int | None = 220,
        cmap: str = "magma"
    ) -> None:
    """
    Plots a 2D spectrogram (freq_bins x time_frames). If sr and hop_length are provided, the x-axis is shown in seconds.
    """

    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be 2D (freq_bins x time_frames)")

    plt.figure()

    if sr is not None and hop_length is not None:
        time_axis = np.arange(spectrogram.shape[1]) * hop_length / sr
        extent = [time_axis[0], time_axis[-1], 0, spectrogram.shape[0]]

        plt.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap
        )
        plt.xlabel("Time (s)")
    else:
        plt.imshow(
            spectrogram,
            aspect="auto",
            origin="lower",
            cmap=cmap
        )
        plt.xlabel("Frame")

    plt.ylabel("Frequency Bin")
    plt.title("Spectrogram")
    plt.colorbar(label="Magnitude")

    plt.tight_layout()
    plt.show()



# DATASET PROCESSOR CLASS
class DatasetProcessor():
    """
    Class that processes the dataset and provides pytorch dataset rows.
    """

    def __init__(self, hyperparameters: Hyperparameters | None = None) -> None:
        """
        Initializes the DatasetProcessor class with the given hyperparameters and loads metadata and annotation periods into global variables.
        """
        
        if hyperparameters is None:
            hyperparameters = Hyperparameters()

        # CUSTOM MODIFICATION FUNCTION VARIABLES
        self.bins_per_axis = hyperparameters.bins_per_axis

        self.crop_valence = hyperparameters.crop_valence
        self.crop_arousal = hyperparameters.crop_arousal
        self.max_magnitude = hyperparameters.max_magnitude
        
        self.hyperparameters = hyperparameters

        self.metadata = hyperparameters.metadata

        if self.hyperparameters.seed is not None:
            random.seed(self.hyperparameters.seed)

        self.valence_df = pd.read_csv(self.hyperparameters.valence_csv)
        self.arousal_df = pd.read_csv(self.hyperparameters.arousal_csv)

        self.annotation_periods = self._load_annotation_periods()
        self.spectrogram_paths = self._load_spectrogram_paths()
        self.factor = self.max_magnitude / max(abs(self.crop_valence[0]), abs(self.crop_valence[1]), abs(self.crop_arousal[0]), abs(self.crop_arousal[1])) # scaling factor to expand cropped values to -1 to 1 range (set in dataset_modification_function)

        self.dataset = None



    @staticmethod
    def _parse_ms_from_col_name(col_name: str) -> int:
        """
        Parses a column name like "t_12345" to extract the timestamp in ms as an integer. Raises ValueError if format is invalid.
        """

        numdic = {}
        for i in range(10):
            numdic[str(i)] = None
        
        output = ""

        for c in col_name:
            if c in numdic:
                output += c

        return int(output)



    def _load_annotation_periods(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Loads valence/arousal CSVs in the format Dict[songid, (timestamps_ms, valence, arousal)] and return the same thing.
        """

        annotation_periods = {}

        valence_df = self.valence_df
        arousal_df = self.arousal_df

        if "song_id" not in valence_df.columns:
            raise KeyError("song_id is not a column in valence CSV")
        if "song_id" not in arousal_df.columns:
            raise KeyError("song_id is not a column in arousal CSV")
        
        if not valence_df["song_id"].equals(arousal_df["song_id"]):
            raise ValueError("song_id column values do not match between valence and arousal CSVs")
        
        valence_df = valence_df.set_index("song_id")
        arousal_df = arousal_df.set_index("song_id")
        
        timestamp_dict = {}
        for col in valence_df.columns:
            if col != "song_id":
                timestamp_dict[col] = self._parse_ms_from_col_name(col)

        for song_id in valence_df.index:
            valence_row = valence_df.loc[song_id].dropna()
            arousal_row = arousal_df.loc[song_id].dropna()

            common_cols = valence_row.index.intersection(arousal_row.index)

            timestamps_ms = []
            valence_annotations = []
            arousal_annotations = []

            for col in common_cols:
                timestamps_ms.append(timestamp_dict[col])
                valence_annotations.append(valence_row[col])
                arousal_annotations.append(arousal_row[col])
            
            annotation_periods[int(song_id)] = (np.array(timestamps_ms), np.array(valence_annotations), np.array(arousal_annotations))
            
        return annotation_periods
    


    def _load_spectrogram_paths(self) -> dict[int, str]:
        """
        Loads spectrogram paths from disk and returns a dict of song_id to spectrogram file path.
        """

        spectrogram_paths = {}

        if "song_id" not in self.valence_df.columns:
            raise KeyError("song_id is not a column in valence CSV")
        
        for song_id in self.valence_df["song_id"]:
            path = Path(self.hyperparameters.spectrogram_dir) / f"{song_id}.npy"

            if not path.is_file():
                raise FileNotFoundError(f"Spectrogram file not found for song_id {song_id} at path {path}")
            
            spectrogram_paths[int(song_id)] = Path(path)

        return spectrogram_paths



    def load_dataset(self) -> list[tuple[int, float, float]]:
        """
        Gathers and returns the average annotations per songid in the format Dict[songid, np.ndarray(valence avg, arousal avg)].
        """

        valence_df = self.valence_df
        arousal_df = self.arousal_df

        if "song_id" not in valence_df.columns:
            raise KeyError("song_id is not a column in valence CSV")
        if "song_id" not in arousal_df.columns:
            raise KeyError("song_id is not a column in arousal CSV")
        
        if not valence_df["song_id"].equals(arousal_df["song_id"]):
            raise ValueError("song_id column values do not match between valence and arousal CSVs")
        
        song_ids = valence_df["song_id"].values
        valence_avgs = valence_df.drop(columns = ["song_id"]).mean(axis = 1).values
        arousal_avgs = arousal_df.drop(columns = ["song_id"]).mean(axis = 1).values

        output = []

        for i in range(len(song_ids)):
            output.append((int(song_ids[i]), valence_avgs[i], arousal_avgs[i]))

        return output



    def get_avg_annotations_from_window(self, song_id: int, window_start: int, window_end: int) -> np.ndarray:
        """
        Return the average valence and arousal annotations for a given song_id over a specified time window (in ms). Uses the global _ANNOTATION_PERIODS variable.
        """

        if song_id not in self.annotation_periods:
            raise KeyError(f"song_id {song_id} not in annotation periods")

        timestamps_ms, valence_annotations, arousal_annotations = self.annotation_periods[song_id]

        if window_end <= timestamps_ms[-1]:
            mask = (timestamps_ms >= window_start) & (timestamps_ms < window_end)
        if window_end > timestamps_ms[-1]:
            mask = (timestamps_ms >= window_start) & (timestamps_ms <= timestamps_ms[-1])

        if not np.any(mask):
            raise ValueError(f"No annotations found for song_id {song_id} in window {window_start}-{window_end} ms")

        avg_valence = np.mean(valence_annotations[mask])
        avg_arousal = np.mean(arousal_annotations[mask])

        return np.array([avg_valence, avg_arousal])



    def get_spectrogram_data_from_window(self, song_id: int, window_start: int, window_end: int) -> tuple[Path, int, int]:
        """
        Returns the file path of the spectrogram chunk for a given song_id and time window (in frames).
        """

        timestep_ms = self.metadata["hop_length"] / self.metadata["sr"] * 1000 
        window_frames = int(self.hyperparameters.window_seconds_ms // timestep_ms)

        start_frame = int(round(window_start / timestep_ms))
        end_frame = start_frame + window_frames

        return (self.spectrogram_paths[song_id], start_frame, end_frame)



    def generate_windows(self, data: list[tuple[int, float, float]]) -> Iterable[tuple[int, int, int]]:
        """
        Generates and yields windows for a given song_id based on the window hyperparameters.
        """

        if self.hyperparameters.window_overlap_ms >= self.hyperparameters.window_seconds_ms:
            raise ValueError("window_overlap_ms must be less than window_seconds_ms to avoid infinite loop")

        for song_id, _, _ in data:
            annotations_timestamps = self.annotation_periods[song_id][0]
            curr = int(annotations_timestamps[0])
            end = int(annotations_timestamps[-1])

            while curr < end:
                window_start = curr
                window_end = curr + int(self.hyperparameters.window_seconds_ms)

                yield int(song_id), int(window_start), int(window_end)

                curr += int(self.hyperparameters.window_seconds_ms - self.hyperparameters.window_overlap_ms)



    def dataset_modification_function(
            self,
            annotated_averages: list[tuple[int, float, float]], 
            print_info: bool = False
        ) -> np.ndarray:
        """
        Modifies the distribution of the data to be more uniform by applying a custom function and return a numpy array of the desired songids.
        """

        if print_info:
            print(f"scaling factor: {self.factor}")

        valence_bin_size = abs(self.crop_valence[1] - self.crop_valence[0]) / self.bins_per_axis
        arousal_bin_size = abs(self.crop_arousal[1] - self.crop_arousal[0]) / self.bins_per_axis

        bins = []
        bin_counts = []
        for x in range(self.bins_per_axis):
            bins_row = []
            bin_counts_row = []
            for y in range(self.bins_per_axis):
                bins_row.append([])
                bin_counts_row.append(0)
            bins.append(bins_row)
            bin_counts.append(bin_counts_row)

        cut = 0
        for songid, valence, arousal in annotated_averages:
            if valence > self.crop_valence[0] and valence < self.crop_valence[1] and arousal > self.crop_arousal[0] and arousal < self.crop_arousal[1]:
                valence_bin = int((valence - self.crop_valence[0]) // valence_bin_size)
                arousal_bin = int((arousal - self.crop_arousal[0]) // arousal_bin_size)
                bins[valence_bin][arousal_bin].append((songid, valence, arousal))
                bin_counts[valence_bin][arousal_bin] += 1
            else:
                cut += 1

        bin_mean = 0
        zeroes_in_bin = 0
        for valence_bin in range(self.bins_per_axis):
            for arousal_bin in range(self.bins_per_axis):
                if bin_counts[valence_bin][arousal_bin] > 0:
                    bin_mean += bin_counts[valence_bin][arousal_bin]
                else:
                    zeroes_in_bin += 1
        bin_mean = int(round(bin_mean / (self.bins_per_axis * self.bins_per_axis - zeroes_in_bin)))

        if print_info:
            print(f"bin mean: {bin_mean}")
            print(f"cut: {cut}")
            print(f"bin counts: {bin_counts}")
            if zeroes_in_bin > 0:
                print(f"Warning: {zeroes_in_bin} bins are empty.")
        
        for valence_bin in range(self.bins_per_axis):
            for arousal_bin in range(self.bins_per_axis):
                if bin_counts[valence_bin][arousal_bin] == 0:
                    continue
                while bin_counts[valence_bin][arousal_bin] != bin_mean:
                    if bin_counts[valence_bin][arousal_bin] > bin_mean:
                        bins[valence_bin][arousal_bin].pop(random.randint(0, len(bins[valence_bin][arousal_bin]) - 1))
                        bin_counts[valence_bin][arousal_bin] -= 1
                    else:
                        bins[valence_bin][arousal_bin].append(random.choice(bins[valence_bin][arousal_bin]))
                        bin_counts[valence_bin][arousal_bin] += 1

        if print_info:
            print(f"new bin counts: {bin_counts}")

        output = []
        for valence_bin in range(self.bins_per_axis):
            for arousal_bin in range(self.bins_per_axis):
                for songid, valence, arousal in bins[valence_bin][arousal_bin]:
                    output.append((songid, valence * self.factor, arousal * self.factor))

        return output
    


    def process_data(self) -> tuple[list[tuple[Path, int, int]], list[np.ndarray]]:
        """
        Processes the dataset to generate spectrogram windows and corresponding average valence/arousal labels. Returns a tuple of (spectrogram_data, targets).
        """

        if self.dataset is None:
            self.dataset = self.dataset_modification_function(self.load_dataset())

        data = self.dataset.copy()

        spectrogram_data_output = []
        targets_output = []

        for song_id, window_start, window_end in self.generate_windows(data):
            sd = self.get_spectrogram_data_from_window(song_id, window_start, window_end)
            t = np.asarray(self.get_avg_annotations_from_window(song_id, window_start, window_end), dtype=np.float32)
            spectrogram_data_output.append(sd)
            targets_output.append(t)

        return spectrogram_data_output, targets_output



class DatasetClass(Dataset):
    """
    PyTorch Dataset class that provides spectrogram windows and corresponding average valence/arousal labels.
    """

    def __init__(self, spectrograms_data: list[tuple[Path, int, int]], targets: list[np.ndarray]) -> None:
        self.spectrogram_data = spectrograms_data
        self.targets = targets

        self._memmaps = {}

        if len(spectrograms_data) != len(targets):
            raise ValueError("Spectrograms and targets must have same length")
        
    def get_spectrogram_from_window(self, spectrogram_path: Path, start_frame: int, end_frame: int) -> np.ndarray:
        """
        Return the spectrogram chunk for a given song_id and time window (in ms).
        """

        if start_frame < 0 or end_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"Invalid window: start {start_frame} frames, end {end_frame} frames")

        if spectrogram_path not in self._memmaps:
            self._memmaps[spectrogram_path] = np.load(spectrogram_path, mmap_mode='r')
        spectrogram = self._memmaps[spectrogram_path]

        if start_frame >= spectrogram.shape[1] or end_frame <= 0:
            raise ValueError(f"Window {start_frame}-{end_frame} frames is out of bounds for spectrogram {spectrogram_path} with shape {spectrogram.shape}")

        if end_frame > spectrogram.shape[1]:
            window_width  = end_frame - start_frame
            available_width = spectrogram.shape[1] - start_frame
            repeats = window_width // available_width + 1

            repeated_part = spectrogram[:, start_frame:]

            tiled = np.tile(repeated_part, (1, repeats))

            return tiled[:, :window_width]
        else:
            return spectrogram[:, start_frame:end_frame]
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        spectrogram_path, start_frame, end_frame = self.spectrogram_data[idx]

        spectrogram_window = self.get_spectrogram_from_window(spectrogram_path, start_frame, end_frame)

        x = torch.from_numpy(spectrogram_window).unsqueeze(0).float()
        y = torch.from_numpy(self.targets[idx]).float()

        if x.ndim != 3:
            raise ValueError(f"Invalid spectrogram shape {x.shape}")
        if y.shape != (2,):
            raise ValueError(f"Invalid target shape {y.shape}")

        return x, y
from data_loader import DatasetProcessor, DatasetClass, plot_valence_arousal_scatter, plot_valence_arousal_histogram, plot_spectrogram
from config import Hyperparameters


import torch
from tqdm import tqdm  # optional but useful

def check_dataset_dimensions(dataset):
    """
    Iterates through entire dataset and verifies that all samples
    have identical input and target shapes.
    """

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    first_x, first_y = dataset[0]
    expected_x_shape = first_x.shape
    expected_y_shape = first_y.shape

    print(f"Expected input shape:  {expected_x_shape}")
    print(f"Expected target shape: {expected_y_shape}")

    mismatches = []

    for idx in tqdm(range(len(dataset))):
        x, y = dataset[idx]

        if x.shape != expected_x_shape or y.shape != expected_y_shape:
            mismatches.append(
                (idx, x.shape, y.shape)
            )

    if len(mismatches) == 0:
        print("All samples have consistent dimensions.")
    else:
        print(f"Found {len(mismatches)} mismatched samples.")
        for m in mismatches[:10]:  # print first few only
            print(f"Index {m[0]}: x={m[1]}, y={m[2]}")

    return mismatches



if __name__ == "__main__":
    hyperparameters = Hyperparameters()
    dataset_processor = DatasetProcessor(hyperparameters = hyperparameters)

    plot = False

    data = dataset_processor.load_dataset()
    
    if plot:
        plot_valence_arousal_histogram(data)
        plot_valence_arousal_scatter(data)

    moddata = dataset_processor.dataset_modification_function(data)

    if plot:
        plot_valence_arousal_histogram(moddata)
        plot_valence_arousal_scatter(moddata)


    spectrograms, targets = dataset_processor.process_data()

    dataset = DatasetClass(spectrograms, targets)
    
    mismatches = check_dataset_dimensions(dataset)

    timestep_ms = hyperparameters.metadata["hop_length"] / hyperparameters.metadata["sr"] * 1000
    expected_frames = int(hyperparameters.window_seconds_ms // timestep_ms)
    print(f"Expected frames per spectrogram based on window size: {expected_frames}")
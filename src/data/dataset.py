import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    def __init__(self, feature_files, target_files, chunk_length, overlap, augment=False, augment_prob=0.2):
        """
        Args:
            feature_files (list): List of file paths to CSV files containing feature columns.
            target_files (list): List of file paths to CSV files containing target columns (one-to-one correspondence with feature_files).
            chunk_length (int): Length of each chunk.
            overlap (int): Overlap between consecutive chunks.
            augment (bool): Whether to apply augmentation to the data.
            augment_prob (float): Probability of applying augmentation.
        """
        assert len(feature_files) == len(target_files), "Feature files and target files must match in number."
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.file_pairs = list(zip(feature_files, target_files))
        self.augment = augment
        self.augment_prob = augment_prob
        self.chunks = self._process_all_file_pairs()

    def _process_all_file_pairs(self):
        all_chunks = []
        for feature_file, target_file in self.file_pairs:
            feature_df = pd.read_csv(feature_file)
            target_df = pd.read_csv(target_file)
            feature_df.drop(feature_df.columns[0], axis=1, inplace=True)
            target_df.drop(target_df.columns[0], axis=1, inplace=True)
            chunks = self._create_chunks(feature_df, target_df)
            all_chunks.extend(chunks)
        return all_chunks

    def _create_chunks(self, feature_df, target_df):
        """
        Split data into overlapping chunks and track overlap contributions.
        """
        step = self.chunk_length - self.overlap
        num_chunks = (len(feature_df) - self.overlap) // step + 1
        chunks = []

        for i in range(num_chunks):
            start = i * step
            end = start + self.chunk_length
            chunk_features = feature_df.iloc[start:end]
            chunk_targets = target_df.iloc[start:end]

            if len(chunk_features) < self.chunk_length:
                padding_rows = self.chunk_length - len(chunk_features)
                feature_padding = pd.DataFrame(
                    np.repeat([[0] * feature_df.shape[1]], padding_rows, axis=0),
                    columns=feature_df.columns,
                )
                target_padding = pd.DataFrame(
                    np.repeat([[-1] * target_df.shape[1]], padding_rows, axis=0),
                    columns=target_df.columns,
                )

                chunk_features = pd.concat([chunk_features, feature_padding], ignore_index=True)
                chunk_targets = pd.concat([chunk_targets, target_padding], ignore_index=True)

            features = chunk_features.values
            targets = chunk_targets.values
            chunks.append((features, targets))

        return chunks

    @staticmethod
    def augment_coordinates(features, augment_prob=0.2):
        """
        Applies augmentations to coordinate data, treating the first 62 columns as 
        x, y coordinate pairs and leaving the rest as flags.

        Parameters:
        - features: pd.DataFrame or np.ndarray
            Feature data. If ndarray, it is converted to a DataFrame with default column names.
        - augment_prob: float
            Probability of applying augmentation.

        Returns:
        - np.ndarray
            Augmented feature array.
        """
        if np.random.rand() > augment_prob:
            return features  # No augmentation applied

        # Convert to DataFrame if input is ndarray
        if isinstance(features, np.ndarray):
            features = pd.DataFrame(features)

        df = features.copy()

        # Split coordinates and flags
        coordinate_df = df.iloc[:, :62]  # First 62 columns are coordinates
        flag_df = df.iloc[:, 62:]        # Remaining columns are flags

        # Ensure coordinate columns are treated as pairs (x, y)
        num_pairs = coordinate_df.shape[1] // 2
        if coordinate_df.shape[1] % 2 != 0:
            raise ValueError("Number of coordinate columns is not even. Ensure x, y pairs.")

        noise_level = 1
        noise = np.random.normal(0, noise_level, (len(coordinate_df), 2)) 
        scale_range = (0.5,1.5)
        scale = np.random.uniform(*scale_range)
        angle_range = (-5, 5)
        angle = np.radians(np.random.uniform(*angle_range))
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        translation_range = (-1, 1)
        translation = np.random.uniform(*translation_range, size=2)

        augmented_coords = []
        for i in range(num_pairs):
            x_col = coordinate_df.iloc[:, 2 * i]    
            y_col = coordinate_df.iloc[:, 2 * i + 1]  

            coords = np.column_stack((x_col, y_col))

            coords += noise
            coords *= scale
            coords = np.dot(coords, rotation_matrix.T)
            coords += translation

            augmented_coords.append(coords[:, 0])  
            augmented_coords.append(coords[:, 1])  

        augmented_df = pd.concat([pd.DataFrame(np.column_stack(augmented_coords)), flag_df], axis=1)

        return augmented_df.values  


    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        features, targets = self.chunks[idx]

        if self.augment:
            features = self.augment_coordinates(features, augment_prob=self.augment_prob)

        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        return features, targets
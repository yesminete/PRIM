import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, feature_files, target_files, chunk_length, overlap):
        """
        Args:
            feature_files (list): List of file paths to CSV files containing feature columns.
            target_files (list): List of file paths to CSV files containing target columns (one-to-one correspondence with feature_files).
            chunk_length (int): Length of each chunk.
            overlap (int): Overlap between consecutive chunks.
        """
        assert len(feature_files) == len(target_files), "Feature files and target files must match in number."
        self.chunk_length = chunk_length
        self.overlap = overlap
        # Pair feature and target files
        self.file_pairs = list(zip(feature_files, target_files))
        # Process all file pairs to generate chunks
        self.chunks = self._process_all_file_pairs()

    def _process_all_file_pairs(self):
        all_chunks = []
        for feature_file, target_file in self.file_pairs:
            feature_df = pd.read_csv(feature_file)
            target_df = pd.read_csv(target_file)
            # Drop the first column if it's an index
            feature_df.drop(feature_df.columns[0], axis=1, inplace=True)
            target_df.drop(target_df.columns[0], axis=1, inplace=True)
            # Create chunks and local row-to-chunks mapping
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
                # Pad features and targets to the required length
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


    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        features, targets = self.chunks[idx]
        features = torch.tensor(features, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        return features, targets

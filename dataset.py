import pandas as pd
import librosa
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path


class AudioDataset(Dataset):
    """
    Custom class for audiodataset
    """
    def __init__(self, path_to_sound_files: str, path_to_csv: str, sr=44100, transform=None,
                 n_fft=2048, hop_length=512) -> None:
        """

        Args:
            path_to_sound_files: path to folder where sound files are stored
            path_to_csv: path to csv with metadata
            sr: sample rate
            transform: custom transforms if we need them later
            n_fft: length of the FFT window
            hop_length: number of samples between successive frames
        """
        self.dataframe = pd.read_csv(Path(path_to_csv).absolute()).iloc[:20]
        self.path_to_sound_files = Path(path_to_sound_files).absolute()
        self.sr = sr
        self.transform = transform
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        """
        Loads audio file, build a melspectrogram and returns it with necessary reshape into
        (num_channels, time, frequency) as well as target with one hot encoding
        Args:
            idx: id of an sample

        Returns: dict with features and target

        """
        filename = self.dataframe['filename'][idx]
        cols = self.dataframe.columns
        target = self.dataframe[cols[1:]].iloc[idx]
        target = np.array(target)

        audio, _ = librosa.load(self.path_to_sound_files / filename, sr=self.sr)
        features = librosa.feature.melspectrogram(audio, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)
        features = features.reshape(1, features.shape[0], features.shape[1])

        if self.transform:
            features = self.transform(features)

        sample = {'features': features, 'target': target}
        return sample

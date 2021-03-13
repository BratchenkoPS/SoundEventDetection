import pandas as pd
import librosa
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path


class AudioDataset(Dataset):
    def __init__(self, path_to_sound_files: str, path_to_csv: str, sr=44100, num_classes=50, transform=None,
                 n_fft=2048, hop_length=512):
        self.dataframe = pd.read_csv(Path(path_to_csv).absolute()).iloc[:20]
        self.path_to_sound_files = Path(path_to_sound_files).absolute()
        self.sr = sr
        self.num_classes = num_classes
        self.transform = transform
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
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

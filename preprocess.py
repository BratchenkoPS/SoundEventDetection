import librosa
import soundfile
import pandas as pd
import logging

from tqdm import tqdm
from pathlib import Path


class DataPreprocess:
    """
    Preprocesses data from DESED and ESC-50 into train and test data from multilabel classiftication (audio tagging)
    """

    def __init__(self,
                 sr: int,
                 max_length: float,
                 path_to_esc_audio='data/ESC-50-master/audio',
                 path_to_desed_audio='data/DESED/dataset/audio/eval/public',
                 esc_df_path='data/ESC-50-master/meta/esc50.csv',
                 desed_df_path='data/DESED/dataset/metadata/eval/public.tsv'
                 ) -> None:
        """

        Args:
            sr: samplerate of audio files
            max_length: maximum duration of audio files to split into
            path_to_esc_audio: path to ESC audio folder
            path_to_desed_audio: path to DESED audio folder
            esc_df_path: path to ESC csv metadata
            desed_df_path: path to DESED csv metadata
        """
        self.sr = sr
        self.max_length = max_length
        self.path_to_esc_audio = Path(path_to_esc_audio).absolute()
        self.path_to_desed_audio = Path(path_to_desed_audio).absolute()
        self.esc_df = pd.read_csv(Path(esc_df_path).absolute())
        self.desed_df = pd.read_csv(Path(desed_df_path).absolute(), sep='\t')

    def get_test_data(self, path_to_test_data='data/test') -> None:
        """
        Splits 10 second audio files from DESED dataset into 1 sec files with multiple (if needed) tags
        Args:
            path_to_test_data: path to store test audio and metadata

        Returns: None

        """
        path_to_test_audio = Path(path_to_test_data).absolute() / 'audio'
        path_to_test_csv = Path(path_to_test_data).absolute() / 'meta'

        path_to_test_audio.mkdir(exist_ok=True, parents=True)
        path_to_test_csv.mkdir(exist_ok=True, parents=True)

        labels = []
        files = []

        logging.info('Starting to build test dataset from DESED')
        for i in tqdm(range(len(self.desed_df)-1)):
            filename = self.desed_df['filename'][i]
            onset = self.desed_df['onset'][i]
            label = self.desed_df['event_label'][i]
            offset = self.desed_df['offset'][i]

            while onset < offset:  # loop for cases when there is one class for more than 1 sec

                if self.desed_df['filename'][i + 1] == filename and onset + 1 < self.desed_df['onset'][i + 1] and \
                        self.desed_df['event_label'][i + 1] != label:
                    label = label + ', ' + self.desed_df['event_label'][i + 1]  # check for multilabel

                if offset - onset < 1.0:
                    duration = offset - onset  # check for the shorter duration parts
                else:
                    duration = 1.0

                audio, _ = librosa.load(self.path_to_desed_audio / filename, offset=onset, sr=44100, duration=duration)

                if audio.shape[0] < 44100:  # padd for shorter duration audio
                    audio = librosa.util.fix_length(audio, size=44100)

                new_filename = path_to_test_audio / (str(onset) + filename)

                soundfile.write(new_filename, audio, samplerate=44100)
                onset += 1
                files.append(new_filename)
                labels.append(label)

            test_meta = pd.DataFrame({'filename': files, 'labels': labels})
            test_meta.to_csv(path_to_test_csv/ 'test.csv', index=False)

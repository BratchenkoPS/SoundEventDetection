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
                 classes_mapping: dict,
                 path_to_esc_audio='data/ESC-50-master/audio',
                 path_to_desed_audio='data/DESED/dataset/audio/eval/public',
                 esc_df_path='data/ESC-50-master/meta/esc50.csv',
                 desed_df_path='data/DESED/dataset/metadata/eval/public.tsv'
                 ) -> None:
        """

        Args:
            sr: samplerate of audio files
            max_length: maximum duration of audio files to split into
            classes_mapping: mapping of classes from one dataset to another (DESED - ESC)
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
        self.classes_mapping = classes_mapping

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

        self.desed_df = self.desed_df[self.desed_df['event_label'].isin(self.classes_mapping.keys())]
        self.desed_df['event_label'] = self.desed_df['event_label'].apply(lambda x: self.classes_mapping[x])

        self.desed_df.sort_values(by=['filename', 'onset'], inplace=True)
        self.desed_df.reset_index(inplace=True, drop=True)

        logging.info('Starting to build test dataset from DESED')
        multilabel_flag = False
        for i in tqdm(range(len(self.desed_df) - 1)):
            filename = self.desed_df['filename'][i]
            onset = self.desed_df['onset'][i]
            label = self.desed_df['event_label'][i]
            offset = self.desed_df['offset'][i]

            if multilabel_flag:
                label = label + ',' + label_mem
                multilabel_flag = False

            while onset < offset:  # loop for cases when there is one class for more than 1 sec

                if self.desed_df['filename'][i + 1] == filename and onset + 1 < self.desed_df['onset'][i + 1] and \
                        self.desed_df['event_label'][i + 1] != label:
                    multilabel_flag = True  # flag to make the next line multilabel as well
                    label_mem = label
                    label = label + ',' + self.desed_df['event_label'][i + 1]  # check for multilabel

                if offset - onset < self.max_length:
                    duration = offset - onset  # check for the shorter duration parts
                else:
                    duration = self.max_length

                audio, _ = librosa.load(self.path_to_desed_audio / filename, offset=onset, sr=self.sr,
                                        duration=duration)

                if audio.shape[0] < self.sr:  # padd for shorter duration audio
                    audio = librosa.util.fix_length(audio, size=self.sr)

                new_filename = (str(onset) + filename)

                soundfile.write(path_to_test_audio / new_filename, audio, samplerate=self.sr)
                onset += 1
                files.append(new_filename)
                labels.append(label)

        test_meta = pd.DataFrame({'filename': files, 'labels': labels})
        test_meta = self.get_one_hot_encoding(test_meta)
        test_meta.to_csv(path_to_test_csv / 'test.csv', index=False)

    def get_train_data(self, path_to_train_data='data/train'):
        """
        Splits 5 second audio files from ESC-50 with relevant classes into 1 sec files
        Args:
            path_to_train_data: folder to store train data

        Returns: None

        """
        path_to_train_audio = Path(path_to_train_data).absolute() / 'audio'
        path_to_train_csv = Path(path_to_train_data).absolute() / 'meta'

        path_to_train_audio.mkdir(exist_ok=True, parents=True)
        path_to_train_csv.mkdir(exist_ok=True, parents=True)

        self.esc_df = self.esc_df[self.esc_df['category'].isin(self.classes_mapping.values())]
        self.esc_df.sort_values(by='filename', inplace=True)
        self.esc_df.reset_index(inplace=True, drop=True)

        files = []
        labels = []
        logging.info('Starting to build train dataset from ESC-50')
        for i in tqdm(range(len(self.esc_df))):
            filename = self.esc_df['filename'][i]
            label = self.esc_df['category'][i]

            audio, _ = librosa.load(self.path_to_esc_audio / filename, sr=self.sr)
            audio_length = int(audio.shape[0] // self.sr)

            audio_part_start = 0
            for k in range(audio_length):
                new_filename = str(k) + filename
                audio_part_end = (self.sr * (k + 1))
                audio_part = audio[audio_part_start:audio_part_end]

                soundfile.write(path_to_train_audio / new_filename, audio_part, samplerate=self.sr)
                audio_part_start = audio_part_end

                files.append(new_filename)
                labels.append(label)

        train_meta = pd.DataFrame({'filename': files, 'labels': labels})
        train_meta = self.get_one_hot_encoding(train_meta)
        train_meta.to_csv(path_to_train_csv / 'train.csv', index=False)

    def get_one_hot_encoding(self, df: pd.DataFrame):
        """
        Makes one hot encodings for labels in dataframe
        Args:
            df: dataframe with filenames and labels

        Returns: dataframe with one hot encodings

        """
        df['labels'] = df['labels'].apply(lambda x: self.fix_dups(x))
        unique_labels = df['labels'].unique()
        true_unique_labels = []
        for labels in unique_labels:
            for label in labels.split(','):
                if label not in true_unique_labels:
                    true_unique_labels.append(label)

        for label in true_unique_labels:
            df[label] = [0 for i in range(len(df))]

        for i in range(len(df)):
            labels = df['labels'][i]
            for label in labels.split(','):
                df[label][i] = 1
        df.drop('labels', inplace=True, axis=1)
        df.reset_index(inplace=True, drop=True)
        return df

    @staticmethod
    def fix_dups(string: str):
        """
        Fixes dup bugs in labels in final DF (like 'cat, cat, dog')
        Args:
            string: string with labels

        Returns: fixed string

        """
        values = string.split(',')
        values = sorted(list(set(values)))
        values = ','.join(values)
        return values

    # def get_oversampling(self, path_to_train_audio):
    #     files = []
    #     labels = []
    #     for i in tqdm(range(len(self.esc_df))):
    #         filename = self.esc_df['filename'][i]
    #         label = self.esc_df['category'][i]
    #
    #         for j in range(10):
    #
    #             audio, _ = librosa.load(self.path_to_esc_audio / filename, sr=self.sr, offset=j*0.1)
    #             audio_length = int(audio.shape[0] // self.sr)
    #
    #             audio_part_start = 0
    #             for k in range(audio_length):
    #                 new_filename = str(k) + str(j) + filename
    #                 audio_part_end = (self.sr * (k + 1))
    #                 audio_part = audio[audio_part_start:audio_part_end]
    #
    #                 soundfile.write(path_to_train_audio / new_filename, audio_part, samplerate=self.sr)
    #                 audio_part_start = audio_part_end
    #
    #                 files.append(new_filename)
    #                 labels.append(label)

import logging
import wget
import tarfile
import zipfile

from pathlib import Path


class DataLoader:
    """
    Downloads and unpacks DESED and ESC-50 datasets into given folders.
    """

    def __init__(self,
                 desed_link: str,
                 esc_50_link: str) -> None:
        """

        Args:
            desed_link: link to DESED dataset
            esc_50_link: link to ESC-50 dataset
        """
        self.desed_link = desed_link
        self.esc_50_link = esc_50_link

    def download_data(self) -> None:
        """
        Downloads both datasets

        Returns: None

        """
        curr_path = Path('').absolute()
        data_path = curr_path / 'data'
        desed_path = data_path / 'DESED'
        desed_path.mkdir(exist_ok=True, parents=True)
        data_path.mkdir(exist_ok=True, parents=True)

        logging.info('Starting to download data (might take a while)')
        wget.download(self.desed_link, str(data_path))  # wget only works with strings
        wget.download(self.esc_50_link, str(data_path))

    @staticmethod
    def unpack_data() -> None:
        """
        Unpacks the archives with data

        Returns: None

        """
        desed = tarfile.open('data/DESEDpublic_eval.tar.gz')
        desed.extractall('data/DESED')

        esc = zipfile.ZipFile('data/ESC-50-master.zip')
        esc.extractall('data/')

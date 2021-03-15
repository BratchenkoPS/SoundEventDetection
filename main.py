import yaml
import logging
import torch
import random
import numpy as np

from downloader import Downloader
from dataset import AudioDataset
from torch.utils.data import DataLoader
from model import Resnet18Multi
from train_eval import train
from preprocess import DataPreprocess

if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = yaml.load(file)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if config['Download']:
        downloader = Downloader(config['Downloader']['desed_link'],
                                config['Downloader']['esc_link'])
        downloader.download_data()
        downloader.unpack_data()

    SEED = 42

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    preprocess = DataPreprocess(config['DataPreprocess']['sr'],
                                config['DataPreprocess']['max_length'],
                                config['DataPreprocess']['classes_mapping'])
    preprocess.get_train_data()
    preprocess.get_test_data()

    train_dataset = AudioDataset(path_to_sound_files='data/train/audio',
                                 path_to_csv='data/train/meta/train.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = AudioDataset(path_to_sound_files='data/test/audio',
                                path_to_csv='data/test/meta/test.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Resnet18Multi(config['Model']['num_classes'])
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logging.info('Total num of model parameters: {}'.format(pytorch_total_params))

    optimizer = torch.optim.Adam(model.parameters(), lr=config['Model']['learning_rate'])
    criterion = torch.nn.BCELoss()

    epochs = config['Model']['num_epochs']
    threshold = config['Model']['threshold']
    train(epochs, model, device, train_dataloader, test_dataloader, optimizer, criterion, threshold)



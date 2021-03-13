import yaml
import logging
import torch
import numpy as np

from downloader import Downloader
from preprocess import DataPreprocess
from dataset import AudioDataset
from torch.utils.data import DataLoader
from model import Resnet18Multi
from tqdm import tqdm
from metrics import calculate_f1
from sklearn.metrics import f1_score

if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = yaml.load(file)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if config['Download']:
        downloader = Downloader(config['Downloader']['desed_link'],
                                config['Downloader']['esc_link'])
        downloader.download_data()
        downloader.unpack_data()

    # preprocess = DataPreprocess(config['DataPreprocess']['sr'],
    #                             config['DataPreprocess']['max_length'],
    #                             config['DataPreprocess']['classes_mapping'])
    # preprocess.get_train_data()
    # preprocess.get_test_data()

    train_dataset = AudioDataset(path_to_sound_files='data/train/audio',
                                 path_to_csv='data/train/meta/train.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_dataset = AudioDataset(path_to_sound_files='data/test/audio',
                                path_to_csv='data/test/meta/test.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Resnet18Multi(5)
    model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    epochs = 10
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        for obj in tqdm(train_dataloader):
            imgs, targets = obj['features'], obj['target']
            targets = targets.type(torch.float)
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            model_result = model(imgs)
            loss = criterion(model_result, targets)

            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()

            epoch_losses.append(batch_loss_value)

        epoch_loss = np.mean(epoch_losses)
        print('Epoch {} loss={}'.format(epoch, epoch_loss))
        print('Calucating metrics')

        model.eval()
        model_result = []
        targets_full = []

        for obj in tqdm(test_dataloader):
            imgs, targets = obj['features'], obj['target']
            targets = targets.type(torch.float)
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.no_grad():
                model_batch_res = model(imgs)
                model_result.append(np.array(model_batch_res))
                targets_full.append(np.array(targets))

        model_result = np.array(model_result)
        targets_full = np.array(targets_full)

        pred = np.array(model_result > 0.6, dtype=float)
        true_pred = []
        for prediction in pred:
            true_pred.append(prediction[0])
        true_target = []
        for targ in targets_full:
            true_target.append(targ[0])
        # print(true_pred)
        # print(true_target)
        metrics = f1_score(true_target, true_pred, average='macro')
        print(metrics)



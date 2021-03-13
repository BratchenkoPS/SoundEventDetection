import yaml
import logging

from dataloader import DataLoader
from preprocess import DataPreprocess

if __name__ == '__main__':
    with open('config.yml', 'r') as file:
        config = yaml.load(file)

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    if config['Download']:
        dataloader = DataLoader(config['DataLoader']['desed_link'],
                                config['DataLoader']['esc_link'])
        dataloader.download_data()
        dataloader.unpack_data()

    preprocess = DataPreprocess(config['DataPreprocess']['sr'],
                                config['DataPreprocess']['max_length'])
    preprocess.get_test_data()























# import torch
# import numpy as np
#
# from dataset import AudioDataset
# from model import Resnet18Multi
# from sklearn.metrics import f1_score
# from torch.utils.data import DataLoader
# from tqdm import tqdm


# dataset = AudioDataset(path_to_sound_files='fluff/ESC-50-master/audio',
#                        path_to_csv='fluff/ESC-50-master/meta/esc50.csv')
#
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#
# def calculate_metrics(pred, target):
#     return {
#         'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
#     }
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# model = Resnet18Multi(50)
# model.to(device)
#
# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print(pytorch_total_params)
#
# batch_size = 4
# max_epoch_number = 35
# learning_rate = 1e-3
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# epochs = 10
# iteration = 0
# criterion = torch.nn.CrossEntropyLoss()
# test_freq = 50
#
# batch_losses = []
# for epoch in range(epochs):
#     for obj in tqdm(dataloader):
#         model.train()
#         imgs, targets = obj['features'], obj['target']
#         targets = targets.type(torch.float)
#         imgs, targets = imgs.to(device), targets.to(device)
#
#         optimizer.zero_grad()
#
#         model_result = model(imgs)
#         loss = criterion(model_result, torch.max(targets, 1)[1])
#
#         batch_loss_value = loss.item()
#         loss.backward()
#         optimizer.step()
#
#         batch_losses.append(batch_loss_value)
#         model.eval()
#         with torch.no_grad():
#             true_res = []
#             for tensor in model_result:
#                 ind = torch.argmax(tensor)
#                 res_temp = np.zeros(50)
#                 res_temp[ind] = 1
#                 true_res.append(res_temp)
#             result = calculate_metrics(np.array(true_res), np.array(targets))
#             print(result)
#
#         # if iteration % test_freq == 0:
#         #     model.eval()
#         #     with torch.no_grad():
#         #         model_result = []
#         #         res = []
#         #         for obj in dataloader:
#         #             imgs, targets = obj['features'], obj['target']
#         #             model_batch_result = model(imgs)
#         #             model_result.extend(model_batch_result.cpu().numpy())
#         #             res.append(targets.cpu().numpy())
#
#         # result = calculate_metrics(np.array(model_result), np.array(targets))
#         # print(result)
#
#         iteration += 1
#
#         loss_value = np.mean(batch_losses)
#         # print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch + 1, iteration, loss_value))

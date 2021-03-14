import torch
import numpy as np
import logging

from tqdm import tqdm
from sklearn.metrics import f1_score
from pathlib import Path
from typing import List


def train(epochs: int,
          model: torch.nn.Module,
          device: torch.device,
          train_dataloader: torch.utils.data.Dataloader,
          test_dataloader: torch.utils.data.Dataloader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          threshold: float) -> None:
    """
    Train loop for multilabel resnet architecture
    Args:
        epochs: num of epochs for train loop
        model: resnet18 multilabel architecture
        device: device to train on
        train_dataloader: dataloader with train data
        test_dataloader: dataloader with test data
        optimizer: optimizer
        criterion: criterion
        threshold: threshold for sigmoid values

    Returns: None

    """
    best_metric = 0
    curr_path = Path('').absolute()
    model_path = curr_path / 'models'
    model_path.mkdir(exist_ok=True, parents=True)
    logging.info('Starting to train a model')

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
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
        logging.info('Epoch {} train loss={}'.format(epoch + 1, epoch_loss))
        metrics = evaluate(model, test_dataloader, device, threshold)
        if metrics > best_metric:
            logging.info('New best score! Saving model. \n')
            torch.save(model.state_dict(), 'models/best-model.pt')
            best_metric = metrics


def evaluate(model: torch.nn.Module,
             test_dataloader: torch.utils.data.Dataloader,
             device: torch.device,
             threshold: float) -> float:
    """
    Calculates macro f1 score on test data with necessary preproces for model output
    Args:
        model: resnet18 multilabel architecture
        test_dataloader: dataloader with test data
        device: device to evaluate at
        threshold: threshold for sigmoid values

    Returns: avg f1 macro for test data

    """
    model.eval()
    predicted_full = []
    targets_full = []
    logging.info('Starting to calculate metrics')
    for obj in tqdm(test_dataloader):
        imgs, targets = obj['features'], obj['target']
        targets = targets.type(torch.float)
        imgs, targets = imgs.to(device), targets.to(device)

        with torch.no_grad():
            predicted_batch = model(imgs)
            predicted_full.extend(predicted_batch.cpu().numpy())
            targets_full.extend(targets.cpu().numpy())

    predicted_full = np.array(predicted_full, dtype=object)
    predicted_full = np.array(predicted_full > threshold, dtype=float)
    targets_full = np.array(targets_full, dtype=object)

    metrics = calc_macro_f1(targets_full, predicted_full)
    logging.info('Test f1 macro is {} \n'.format(metrics))
    return metrics


def calc_macro_f1(targets: np.array, pred: np.array) -> float:
    """
    Calculates avg F1 macro score for given target and prediction
    Args:
        targets: array of targets np.array[np.array[np.array]]
        pred: array of prediction np.array[np.array[np.array]]

    Returns: avg F1 macro score for given target and prediction

    """
    true_pred = []
    for prediction in pred:
        true_pred.append(prediction[0])
    true_target = []
    for targ in targets:
        true_target.append(targ[0])

    metrics = f1_score(true_target, true_pred, average='macro')
    return metrics

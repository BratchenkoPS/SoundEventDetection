import numpy as np

from sklearn.metrics import f1_score


def calculate_f1(pred, target, threshold=0.6):
    pred = np.array(pred > threshold, dtype=float)
    true_pred = []
    for prediction in pred:
        true_pred.append(prediction[0])
    true_target = []
    for targ in target:
        true_target.append(targ[0])
    print(true_pred)
    print(true_target)
    metrics = f1_score(true_target, true_pred, average='macro')
    return metrics


# a = np.array([[1., 0.], [1., 0.]])
# b = np.array([[0, 1.], [1., 0.]])
# metrics = calculate_f1(a, b)
# print(metrics)

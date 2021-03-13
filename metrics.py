from sklearn.metrics import f1_score


def calculate_metrics(pred, target):
    return {
        'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
    }

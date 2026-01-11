import numpy as np


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    report = {}
    eps = 1e-8
    num_classes = max(y_true.max(), y_pred.max()) + 1
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true == c) & (y_pred != c))
        fn = np.sum((y_true != c) & (y_pred == c))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        report[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return report

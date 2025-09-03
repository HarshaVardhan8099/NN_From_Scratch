import cupy as cp

def accuracy(preds, targets):
    pred_labels = cp.argmax(preds, axis = 1)
    return cp.mean(pred_labels == targets)

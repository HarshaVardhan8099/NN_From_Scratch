from backend import xp

def accuracy(preds, targets):
    """
    Calculates the accuracy of the model's predictions.
    
    Args:
        preds (xp.ndarray): The predicted probabilities from the model's output layer.
                            Shape: (batch_size, num_classes).
        targets (xp.ndarray): The true labels (integer indices).
                              Shape: (batch_size,).
                              
    Returns:
        float: The accuracy as a fraction (e.g., 0.95).
    """
    # Find the index of the highest probability for each sample to get the predicted label.
    pred_labels = xp.argmax(preds, axis=1)
    
    # Compare the predicted labels to the true labels.
    # xp.mean() calculates the average of the boolean array (True=1, False=0).
    return xp.mean(pred_labels == targets)
from backend import xp

class CrossEntropyLoss:
    def forward(self, preds, targets):
        """
        Calculates the Cross-Entropy loss.
        
        Args:
            preds (xp.ndarray): The predicted probabilities from the model's output layer.
                                Shape: (batch_size, num_classes).
            targets (xp.ndarray): The true labels (integer indices).
                                  Shape: (batch_size,).
                                  
        Returns:
            float: The mean loss over the batch.
        """
        # Get the number of samples in the batch.
        m = preds.shape[0]
        
        # Calculate the negative log-likelihood of the correct class.
        # xp.arange(m) creates an index for each sample, and `targets` selects the column
        # corresponding to the true label. `1e-9` is added to prevent taking the log of zero.
        log_likelihood = -xp.log(preds[xp.arange(m), targets] + 1e-9)
        
        # Calculate the average loss over the entire batch.
        loss = xp.sum(log_likelihood) / m
        return loss

    def backward(self, preds, targets):
        """
        Calculates the gradient of the loss with respect to the model's output.
        
        Args:
            preds (xp.ndarray): The predicted probabilities.
            targets (xp.ndarray): The true labels.
            
        Returns:
            xp.ndarray: The gradient to be passed back to the model's output layer.
        """
        # Get the number of samples in the batch.
        m = preds.shape[0]
        
        # Initialize the gradient with a copy of the predictions.
        grad = preds.copy()
        
        # For the correct class of each sample, subtract 1 from the gradient.
        # This is the simplified gradient for the combination of Softmax and Cross-Entropy loss.
        grad[xp.arange(m), targets] -= 1
        
        # Normalize the gradient by the batch size.
        grad /= m
        return grad
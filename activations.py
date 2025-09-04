from backend import xp

class ReLU:
    """
    The Rectified Linear Unit (ReLU) activation function.
    It returns the element itself for positive values and zero for negative values.
    """
    def forward(self, X):
        """
        Performs the forward pass for ReLU.
        
        Args:
            X (xp.ndarray): The input data.
            
        Returns:
            xp.ndarray: The element-wise max of 0 and the input.
        """
        # Create a boolean mask of the input where values are greater than 0.
        # This mask is used in the backward pass to efficiently handle the gradient.
        self.mask = (X > 0)
        return xp.maximum(0, X)

    def backward(self, dOut):
        """
        Performs the backward pass for ReLU.
        
        Args:
            dOut (xp.ndarray): The gradient from the subsequent layer.
            
        Returns:
            xp.ndarray: The gradient for the ReLU layer's input.
        """
        # The gradient of ReLU is 1 for positive inputs and 0 for negative/zero inputs.
        # We multiply the incoming gradient (dOut) by our pre-computed mask.
        return dOut * self.mask

class Softmax:
    """
    The Softmax activation function, typically used in the output layer for multi-class classification.
    It converts a vector of numbers into a probability distribution.
    """
    def forward(self, X):
        """
        Performs the forward pass for Softmax.
        
        Args:
            X (xp.ndarray): The input data.
            
        Returns:
            xp.ndarray: A probability distribution where each value is between 0 and 1, and the
                        sum of each row is 1.
        """
        # Subtract the maximum value from each row to prevent numerical overflow
        # when calculating the exponent of large numbers.
        exps = xp.exp(X - xp.max(X, axis=1, keepdims=True))
        
        # Normalize the exponentiated values to get probabilities.
        self.out = exps / xp.sum(exps, axis=1, keepdims=True)
        return self.out

    def backward(self, dOut):
        """
        Softmax backward pass.
        
        The gradient for Softmax is simplified when used with CrossEntropyLoss.
        The `dOut` from the loss function already incorporates the Softmax derivative,
        so this layer's backward method simply passes the gradient through.
        """
        return dOut
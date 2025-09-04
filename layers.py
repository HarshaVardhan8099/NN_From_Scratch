from backend import xp

class Dense:
    def __init__(self, in_features, out_features, lr=0.01):
        """
        Initializes a Dense layer.

        Args:
            in_features (int): The number of input features (neurons in the previous layer).
            out_features (int): The number of output features (neurons in this layer).
            lr (float): The learning rate for weight updates.
        """
        # Initialize weights (W) with small random values to break symmetry.
        # The shape is (in_features, out_features) for matrix multiplication.
        self.W = xp.random.randn(in_features, out_features) * 0.01
        
        # Initialize biases (b) to zeros. The shape is (1, out_features) for broadcasting.
        self.b = xp.zeros((1, out_features))
        
        self.lr = lr # Store the learning rate.

    def forward(self, X):
        """
        Performs the forward pass for the dense layer.
        
        Args:
            X (xp.ndarray): The input data from the previous layer, with shape (batch_size, in_features).
            
        Returns:
            xp.ndarray: The output of the layer, with shape (batch_size, out_features).
        """
        # Store the input for use in the backward pass (for calculating dW).
        self.X = X
        
        # The core linear transformation: y = X @ W + b
        # '@' is the matrix multiplication operator. The bias 'b' is automatically broadcasted.
        return X @ self.W + self.b

    def backward(self, dOut):
        """
        Performs the backward pass to calculate gradients and update weights.
        
        Args:
            dOut (xp.ndarray): The gradient of the loss with respect to the output of this layer.
                               Shape: (batch_size, out_features).
                               
        Returns:
            xp.ndarray: The gradient of the loss with respect to the input of this layer (dX).
                        Shape: (batch_size, in_features). This is passed to the previous layer.
        """
        # Calculate the gradient of the loss with respect to weights (dW).
        # This is derived from the chain rule: dL/dW = X.T @ dL/dOut
        dW = self.X.T @ dOut
        
        # Calculate the gradient of the loss with respect to biases (db).
        # We sum the gradients along the batch dimension (axis=0) because the bias is shared.
        db = xp.sum(dOut, axis=0, keepdims=True)
        
        # Calculate the gradient of the loss with respect to the input (dX).
        # This is passed back to the previous layer in the network.
        # dL/dX = dL/dOut @ W.T
        dX = dOut @ self.W.T

        # Update weights and biases using the calculated gradients and learning rate.
        # This is a form of gradient descent.
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dX
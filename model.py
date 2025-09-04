class SimpleNN:
    def __init__(self, layers):
        """
        Initializes the SimpleNN model.
        
        Args:
            layers (list): A list of layer objects (e.g., Dense, ReLU, Softmax) in the order
                           they should be applied.
        """
        self.layers = layers

    def forward(self, X):
        """
        Performs the forward pass through the entire network.
        
        Args:
            X (xp.ndarray): The input data to the network.
            
        Returns:
            xp.ndarray: The output of the final layer.
        """
        # Pass the output of one layer as the input to the next layer in sequence.
        for layer in self.layers:
            X = layer.forward(X)
            
        return X

    def backward(self, grad):
        """
        Performs the backward pass to propagate gradients back through the network.
        
        Args:
            grad (xp.ndarray): The initial gradient from the loss function.
        """
        # Propagate the gradient back through the layers in reverse order.
        # Each layer's backward method takes the incoming gradient and returns the
        # gradient for the previous layer.
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
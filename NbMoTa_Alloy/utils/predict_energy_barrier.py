import numpy as np

class MLPModel:
    """Multi-layer perceptron model for energy barrier prediction.
    
    This class implements a feedforward neural network with batch normalization
    layers for predicting energy barriers in atomic diffusion processes.
    
    Attributes:
        layers (list): List of hidden layer parameters
        output_layer (dict): Output layer parameters
    """
    
    def __init__(self, model_weights_np):
        """Initialize the MLP model with pre-trained weights.
        
        Args:
            model_weights_np (list): List of numpy arrays containing model weights
                Expected format: [W1, b1, gamma1, beta1, mean1, var1, W2, b2, ...]
        """
        self.layers = []
        n_layers = len(model_weights_np) // 6

        # Initialize hidden layers with batch normalization
        for i in range(n_layers):
            W, b = model_weights_np[i * 6 + 0], model_weights_np[i * 6 + 1]
            gamma, beta = model_weights_np[i * 6 + 2], model_weights_np[i * 6 + 3]
            mean, var = model_weights_np[i * 6 + 4], model_weights_np[i * 6 + 5]

            layer_params = {
                'W': W.astype(np.float32),     # Weight matrix
                'b': b.astype(np.float32),     # Bias vector
                'gamma': gamma.astype(np.float32),  # BN scale parameter
                'beta': beta.astype(np.float32),    # BN shift parameter
                'mean': mean.astype(np.float32),    # BN running mean
                'var': var.astype(np.float32)       # BN running variance
            }
            self.layers.append(layer_params)

        # Initialize output layer (no batch normalization)
        W_out = model_weights_np[n_layers * 6 + 0]
        b_out = model_weights_np[n_layers * 6 + 1]
        self.output_layer = {
            'W': W_out.astype(np.float32),
            'b': b_out.astype(np.float32)
        }

    def forward(self, x):
        """Forward pass through the neural network.
        
        Args:
            x (np.ndarray): Input tensor of shape (batch_size, input_features)
        
        Returns:
            np.ndarray: Output predictions of shape (batch_size,) or (batch_size, output_dim)
        """
        x = x.astype(np.float32)
        
        for layer in self.layers:
            # Linear transformation
            x = np.dot(x, layer['W']) + layer['b']
            
            # Batch normalization
            x_normalized = (x - layer['mean']) / np.sqrt(layer['var'] + 1e-5)
            x = x_normalized * layer['gamma'] + layer['beta']
            
            # ReLU activation function
            x = np.maximum(0, x)
        
        # Output layer (no activation)
        x = np.dot(x, self.output_layer['W']) + self.output_layer['b']
        
        # Squeeze last dimension if it's size 1
        return x.squeeze(-1) if x.shape[-1] == 1 else x

def predict_barriers(model, input_array):
    B, n_dirs, feat = input_array.shape
    
    # Reshape input for batch processing: (B * n_dirs, feat)
    input_reshaped = input_array.reshape(-1, feat).astype(np.float32)
    
    # Forward pass through the model
    output = model.forward(input_reshaped)
    
    # Reshape output back to original structure: (B, n_dirs)
    return output.reshape(B, n_dirs)

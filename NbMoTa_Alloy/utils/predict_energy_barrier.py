
import numpy as np

class MLPModel:
    def __init__(self, model_weights_np):
        self.layers = []
        n_layers = len(model_weights_np) // 6

        for i in range(n_layers):
            W, b = model_weights_np[i * 6 + 0], model_weights_np[i * 6 + 1]
            gamma, beta = model_weights_np[i * 6 + 2], model_weights_np[i * 6 + 3]
            mean, var = model_weights_np[i * 6 + 4], model_weights_np[i * 6 + 5]

            layer_params = {
                'W': W.astype(np.float32),  # Keep original shape, don't transpose
                'b': b.astype(np.float32),
                'gamma': gamma.astype(np.float32),
                'beta': beta.astype(np.float32),
                'mean': mean.astype(np.float32),
                'var': var.astype(np.float32)
            }
            self.layers.append(layer_params)

        # Output layer
        W_out, b_out = model_weights_np[n_layers * 6 + 0], model_weights_np[n_layers * 6 + 1]
        self.output_layer = {
            'W': W_out.astype(np.float32),  # Keep original shape, don't transpose
            'b': b_out.astype(np.float32)
        }

    def forward(self, x):
        x = x.astype(np.float32)
        
        for layer in self.layers:
            # Linear layer
            x = np.dot(x, layer['W']) + layer['b']
            
            # Batch normalization
            x = (x - layer['mean']) / np.sqrt(layer['var'] + 1e-5)
            x = x * layer['gamma'] + layer['beta']
            
            # ReLU activation
            x = np.maximum(0, x)
        
        # Output layer
        x = np.dot(x, self.output_layer['W']) + self.output_layer['b']
        return x.squeeze(-1) if x.shape[-1] == 1 else x

def predict_barriers(model, input_array):
    B, n_dirs, feat = input_array.shape
    
    # Reshape for forward pass
    input_reshaped = input_array.reshape(-1, feat).astype(np.float32)
    
    # Forward pass
    output = model.forward(input_reshaped)
    
    # Reshape back to original batch structure
    return output.reshape(B, n_dirs)

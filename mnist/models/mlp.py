import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self, num_layers,input_dim,hidden_dim,output_dim):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(indim, outdim)
            for indim, outdim in zip(layer_sizes[:-1],layer_sizes[1:])
        ]
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = mx.maximum(layer(x),0.0) # relu
        return self.layers[-1](x)
import torch.nn as nn

from bitnet import BitLinear

class BaseModel(nn.Module):

    def __init__(self, 
                 input_size: int, 
                 hidden_layers: int,
                 hidden_units: int, 
                 output_size: int,
                 activation:nn.Module=nn.Tanh()):
        super().__init__()
        self.name = "Base"
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_units)
        
        # criando hidden layers iterativamente
        self.hidden_layers = nn.Sequential()
        for _ in range(hidden_layers):
            self.hidden_layers.append(module=nn.Linear(in_features=hidden_units, 
                                                       out_features=hidden_units))
            self.hidden_layers.append(module=activation)
        
        self.output_layer = nn.Linear(in_features=hidden_units, out_features=output_size)
        self.activation = activation
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class BitModel(nn.Module):

    def __init__(self, 
                 input_size:int, 
                 hidden_layers: int,
                 hidden_units:int, 
                 output_size:int,
                 activation:nn.Module=nn.Tanh()):
        super().__init__()
        self.name = "Bitnet"
        self.input_layer = BitLinear(in_features=input_size, out_features=hidden_units)
        
        # criando hidden layers iterativamente
        self.hidden_layers = nn.Sequential()
        for _ in range(hidden_layers):
            self.hidden_layers.append(module=BitLinear(in_features=hidden_units, 
                                                       out_features=hidden_units))
            self.hidden_layers.append(activation)
        
        self.output_layer = BitLinear(in_features=hidden_units, out_features=output_size)
        self.activation = activation

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
    
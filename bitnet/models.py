import torch.nn as nn

from bitnet import BitLinear

class BaseModel(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, output_size: int):
        super().__init__()
        self.name = "Base"
        self.linear_layer1 = nn.Linear(in_features=input_size, out_features=hidden_units)
        self.linear_layer2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.linear_layer3 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.linear_layer4 = nn.Linear(in_features=hidden_units, out_features=output_size)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.linear_layer1(x)
        x = self.activation(x)
        x = self.linear_layer2(x)
        x = self.activation(x)
        x = self.linear_layer3(x)
        x = self.activation(x)
        x = self.linear_layer4(x)
        return x


class BitModel(nn.Module):
    def __init__(self, input_size:int, hidden_units:int, output_size:int):
        super().__init__()
        self.name = "Bitnet"
        self.bit_layer1 = BitLinear(in_features=input_size, out_features=hidden_units)
        self.bit_layer2 = BitLinear(in_features=hidden_units, out_features=hidden_units)
        self.bit_layer3 = BitLinear(in_features=hidden_units, out_features=hidden_units)
        self.bit_layer4 = BitLinear(in_features=hidden_units, out_features=output_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.bit_layer1(x)
        x = self.activation(x)
        x = self.bit_layer2(x)
        x = self.activation(x)
        x = self.bit_layer3(x)
        x = self.activation(x)
        x = self.bit_layer4(x)
        return x
    
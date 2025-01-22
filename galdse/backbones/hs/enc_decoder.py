import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,layers):
        super(Encoder,self).__init__()
        self.layers = nn.ModuleList([nn.Sequential() for _ in range(len(layers))])
        i = 0
        for layer in layers:
            self.layers[i].add_module("1",nn.Conv2d(in_channels=layer[0][0], out_channels=layer[0][1], kernel_size=layer[1],stride=layer[2],padding=layer[3]))
            self.layers[i].add_module("2",nn.BatchNorm2d(layer[0][1]))
            self.layers[i].add_module("3",nn.PReLU())
            i += 1
    def forward(self,x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
class Decoder(nn.Module):
    def __init__(self,layers,is_excitation=False,is_tanh=False):
        super(Decoder,self).__init__()
        self.layers = nn.ModuleList([nn.Sequential() for _ in range(len(layers))])
        self.is_excitation=is_excitation
        i = 0
        for layer in layers:
            self.layers[i].add_module("1",nn.ConvTranspose2d(*layer[0], kernel_size=layer[1], stride=layer[2], padding=layer[3], output_padding=(1 if i == 1 or i == 2 else 0,0)))
            self.layers[i].add_module("2",nn.BatchNorm2d(layer[0][1]))
            if i != 2:
                self.layers[i].add_module("3",nn.PReLU())  
            elif is_tanh == True:
                self.layers[i].add_module("3", nn.Tanh())
            else:
                self.layers[i].add_module("3", nn.Sigmoid())
            i+=1
    def forward(self,x,medium_array):
        for i in range(3): 
            a = medium_array[2-i]
            x = torch.cat((x,a),dim=1)
            x = self.layers[i](x)
        return x


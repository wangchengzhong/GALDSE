"""
@author: Chengzhong Wang
"""
from .hs.dprnn import DPRNN
from .hs.cbam import CBAM 
from .hs.enc_decoder import Encoder,Decoder
import torch
import torch.nn as nn
from .shared import FrontendRegistry


def create_decoder_layers(last_channel_size):
    return [((128,32),(1,1),(1,1),(0,0)),
                  ((64,16),(5,2),(2,1),(2,1)),
                  ((32,last_channel_size),(5,2),(2,1),(2,1))]

encoder_layers = [((3,16), (5,2), (2,1), (2,1)),
                  ((16,32),(5,2),(2,1),(2,1)),
                  ((32,64),(1,1),(1,1),(0,0))]

@FrontendRegistry.register("MagMask")
class MagMask(nn.Module):
    @staticmethod
    def add_argparse_args(parser):
        return parser
    def __init__(self, 
            num_freq_bins=256, 
            hidden_size=128,# 113, #
            encoder_params=encoder_layers,
            decoder_params=create_decoder_layers(1) ,
            **kwargs):
        super(MagMask,self).__init__()

        self.encoder_blocks = Encoder(encoder_params)

        self.complex_spectrum_decoder = Decoder(decoder_params)
        rnn_size = num_freq_bins//4 # [B, F, T]
        self.dprnns = nn.Sequential(*(DPRNN("LSTM",rnn_size,hidden_size,rnn_size) for _ in range(2)))

        self.cbams = nn.ModuleList([nn.ModuleList([CBAM(16),CBAM(32),CBAM(64)]) for _ in range(1)])


    def forward(self,y_r, y_c, y_mag):

        x = torch.cat([y_r, y_c, y_mag], dim=1) # [B, 4, F, T]
        encoder_outputs = self.encoder_blocks(x)

        medium = [self.cbams[0][i](encoder_outputs[i]) for i in range(3)]

        o = self.dprnns(encoder_outputs[2])

        mask = self.complex_spectrum_decoder(o,medium)
        output_mag = mask * y_mag

        return 1-mask, output_mag

if __name__ == "__main__":
    # python -m galdse.backbones.se_hs_mask
    model = MagMask().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info


    def input_constructor(input_shape):
        batch = 1
        y_r = torch.ones(()).new_empty((batch, *input_shape))
        y_c = torch.ones(()).new_empty((batch, *input_shape))
        y_mag = torch.ones(()).new_empty((batch, *input_shape))
        return {'y_r': y_r, 'y_c': y_c, 'y_mag': y_mag}

    flops, params = get_model_complexity_info(
        model, 
        (1, 256, 256),
        input_constructor=input_constructor,
        as_strings=True,
        print_per_layer_stat=True, 
        verbose=True
    )
    
    print(f'Computational complexity: {flops}')
    print(f'Number of parameters: {params}')
    # 3.6 GMac/s
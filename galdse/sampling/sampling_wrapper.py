
import torch.nn as nn
import torch


class SamplingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.score_fn = model


    def forward(self, y, mask):
        with torch.no_grad():
            out = self.score_fn.diffusion.p_sample_loop_progressive(
                y, self.score_fn, mask, noise=None, device='cuda', clip_denoised=False, progress=False
            )
            
            return out
import warnings

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from galdse.backbones import BackboneRegistry, FrontendRegistry
from galdse.stochastic_process import StoProcessRegistry
from galdse.util.inference import evaluate_model

from torch_pesq import PesqLoss
from galdse.util.si_snr_loss import si_snr_loss


class GaldSE(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--num_eval_files", type=int, default=824, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--finetuned", action="store_true", help="Enable finetuning mode with sqrt noise")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")

        return parser

    def __init__(
        self, backbone, frontend, sto_process, lr=1e-4, ema_decay=0.999,
        num_eval_files=824, loss_type='mse', data_module_cls=None, pesq_weight=0.001, si_snr_weight=0.00003, l1_weight=0.0005, 
        finetuned=False, **kwargs
    ):

        super().__init__()
        # Initialize Backbone and Frontend DNN
        front_end_cls = FrontendRegistry.get_by_name(frontend)
        self.frontend = front_end_cls(**kwargs)
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize Stochastic Process
        sto_process_cls = StoProcessRegistry.get_by_name(sto_process)
        self.diffusion = sto_process_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.model_parameters = {name: param for name, param in self.named_parameters() if 'pesq_loss' not in name}
        self.ema = ExponentialMovingAverage(self.model_parameters.values(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.save_hyperparameters(ignore=['no_wandb', 'start_point'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.finetuned = finetuned

        self._reduce_op = lambda *args, **kwargs: torch.mean(*args, **kwargs)
        self.loss_fn_denoiser = lambda x, y: self._reduce_op(torch.square(x - y))

        self.pesq_weight = pesq_weight
        self.l1_weight = l1_weight
        self.si_snr_weight = si_snr_weight
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=16000).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False

    def configure_optimizers(self):
        torch.set_default_dtype(torch.float32)
        optimizer = torch.optim.AdamW(
            self.model_parameters.values(),
            lr=self.lr
        )
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model_parameters.values())

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)
    
    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.model_parameters.values())        # store current params in EMA
                self.ema.copy_to(self.model_parameters.values())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.model_parameters.values())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)


    def _loss_per_batch(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()

        loss = torch.sum(losses.reshape(losses.shape[0], -1), dim=-1)
        return loss

    def _step(self, batch, batch_idx):
        x_, y_ = batch

        x = torch.cat([x_.real, x_.imag], dim=1)
        y = torch.cat([y_.real, y_.imag], dim=1)
        y_mag = torch.sqrt(y[:,[0],:,:]**2 + y[:,[1],:,:]**2)
        mask_mag, _ = self.frontend(y[:,[0],:,:], y[:,[1],:,:], y_mag)
        
        tt = torch.randint(0, self.diffusion.num_timesteps, size=(x.shape[0],))

        mask = mask_mag.detach()

        if self.finetuned:
            noise = torch.randn_like(x) * torch.sqrt(mask)
        else:
            noise = torch.randn_like(x) * mask

        loss_spec = self.loss_fn_denoiser((1-mask_mag) * y, x)

        z_t = self.diffusion.q_sample(x, y, tt, noise)

        model_output = self.dnn(torch.cat([z_t, y, (mask)], dim=1), tt.to(x.device))
        loss = self.loss_fn_denoiser(x, model_output)

        target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
        x_hat_td = self.to_audio(torch.view_as_complex(model_output.permute(0,2,3,1).contiguous()).squeeze(), target_len)
        x_td = self.to_audio(x_.squeeze(), target_len)
        losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
        losses_l1 = torch.mean(0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))
        losses_si_snr = si_snr_loss(x_hat_td, x_td)

        if self.pesq_weight > 0.0:
            losses_pesq = self.pesq_loss(x_td, x_hat_td)
            losses_pesq = torch.mean(losses_pesq)
            # combine the losses
            loss_td = self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq + self.si_snr_weight * losses_si_snr
            loss = loss + loss_td

        return loss + loss_spec

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss
    def forward(self, input, t):
        ex0 = self.dnn(input, t)
        return ex0

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    
    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        back_spec = self._backward_transform(spec).squeeze(1)
        return self._istft(back_spec, length)
    
    def _istft_device(self, spec, device='cuda'):
        return torch.istft(spec, window=(torch.hann_window(self.data_module.n_fft)).to(device), 
                           center=True, hop_length=self.data_module.hop_length, n_fft=self.data_module.n_fft, return_complex=False)

    
    def to_audio_multi(self, spec, length=None):
        back_spec = self._backward_transform(spec).squeeze(1)
        return_value = []
        for spec in back_spec:
            return_value.append(self._istft_device(spec))
        return torch.stack(return_value)
    
    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

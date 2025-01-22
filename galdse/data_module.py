from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import torch.nn.functional as F

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            normalize="noisy", spec_transform=None,
            stft_kwargs=None, **ignored_kwargs):

        # Read file paths according to file naming format.

        self.clean_files = sorted(glob(join(data_dir, subset) + '/clean/*.wav'))
        self.noisy_files = sorted(glob(join(data_dir, subset) + '/noisy/*.wav'))

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):

        x, _ = load(self.clean_files[i])
        y, _ = load(self.noisy_files[i])

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')


        return x, y

    def __len__(self):

        if self.dummy:
            return int(len(self.clean_files)/500)
        else:
            return len(self.clean_files)



class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=48, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.5, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        parser.add_argument("--augment_type", type=str, choices=("Remix", "EqRemix", "None"), default="Remix", help="Augment type.")
        return parser

    def __init__(
        self, base_dir, batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=48, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent", augment_type="EqRemix", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs
        self.augment_type = augment_type
        
    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True,
                normalize=self.normalize, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='valid',
                dummy=self.dummy, shuffle_spec=False,
                normalize=self.normalize, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False,
                normalize=self.normalize, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})
      
    
    def apply_band_eq(self, noises, n_bands=31, min_gain=0.1, max_gain=3.0, sample_rate=16000):
        B, T = noises.shape
        gains = torch.empty(B, n_bands).uniform_(min_gain, max_gain)
        
        # original_rms = torch.sqrt(torch.mean(noises**2, dim=1, keepdim=True))
    
        specs = torch.stft(noises, 
                        n_fft=1024, 
                        hop_length=512,
                        window=torch.hann_window(1024).to(noises.device),
                        return_complex=True)
        
        freqs = torch.fft.rfftfreq(1024, d=1/sample_rate)
        freq_edges = torch.logspace(1, np.log10(sample_rate/2), n_bands+1)
        
        # Apply gains to log-spaced bands
        for i in range(n_bands):
            mask = (freqs >= freq_edges[i]) & (freqs < freq_edges[i+1])
            specs[:, mask] *= gains[:, i:i+1].unsqueeze(-1)
        
        processed = torch.istft(specs,
                            n_fft=1024,
                            hop_length=512,
                            window=torch.hann_window(1024).to(noises.device),
                            length=T)
        # processed_rms = torch.sqrt(torch.mean(processed**2, dim=1, keepdim=True))
        # processed = processed * (original_rms / (processed_rms + 1e-8))
        
        return processed
    def remix_collate_fn(self, batch):
        clean_signals, orig_noisy_signals = zip(*batch)

        clean_signals = torch.stack(clean_signals).squeeze(1)
        orig_noisy_signals = torch.stack(orig_noisy_signals).squeeze(1)
        
        noises = orig_noisy_signals - clean_signals
        noise_indices = torch.randperm(len(noises))
        shuffled_noises = noises[noise_indices]

        noise_scales = torch.empty(len(noises)).uniform_(0.5, 2)
        if self.augment_type == "EqRemix":
            eq_prob = torch.rand(len(shuffled_noises))  # [B]
            eq_mask = (eq_prob < 0.5) 
            if eq_mask.any():
                eq_noises = self.apply_band_eq(shuffled_noises[eq_mask])
                shuffled_noises[eq_mask] = eq_noises

        remixed_noisy = clean_signals + torch.clamp(shuffled_noises * noise_scales.unsqueeze(1), min=-1, max=1)

        norm_factors = remixed_noisy.abs().max(dim=1)[0]
        clean_signals = clean_signals / norm_factors.unsqueeze(1)
        remixed_noisy = remixed_noisy / norm_factors.unsqueeze(1)
        
        clean_specs = self.spec_fwd(torch.stft(clean_signals, **self.stft_kwargs))
        noisy_specs = self.spec_fwd(torch.stft(remixed_noisy, **self.stft_kwargs))
        
        return clean_specs.unsqueeze(1), noisy_specs.unsqueeze(1)
    
    def collate_fn(self, batch):
        clean_signals, orig_noisy_signals = zip(*batch)

        clean_signals = torch.stack(clean_signals).squeeze(1)
        noisy_signals = torch.stack(orig_noisy_signals).squeeze(1)
        
        norm_factors = noisy_signals.abs().max(dim=1)[0]
        clean_signals = clean_signals / norm_factors.unsqueeze(1)
        noisy_signals = noisy_signals / norm_factors.unsqueeze(1)
        
        clean_specs = self.spec_fwd(torch.stft(clean_signals, **self.stft_kwargs))
        noisy_specs = self.spec_fwd(torch.stft(noisy_signals, **self.stft_kwargs))
        
        return clean_specs.unsqueeze(1), noisy_specs.unsqueeze(1)


    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True, collate_fn=self.remix_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False, collate_fn=self.collate_fn
        )

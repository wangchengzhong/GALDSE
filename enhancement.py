import glob
from argparse import ArgumentParser
from os.path import join

import torch
from soundfile import write
from torchaudio import load
from tqdm import tqdm

from galdse.model import GaldSE
from galdse.util.other import ensure_dir, pad_spec
import os
import torch

from torch.utils.data import DataLoader, Dataset

import glob
from torch.nn import DataParallel
from galdse.sampling.sampling_wrapper import SamplingWrapper

def split_signals(noisy_signals, segment_length=192000, overlap=32):
    segments_list = []
    signal = noisy_signals.squeeze()
    total_length = signal.size(0)
    
    start = 0
    while start + segment_length <= total_length:
        segment = signal[start:start+segment_length]
        segments_list.append(segment.unsqueeze(0))
        start += segment_length - overlap
    
    if start < total_length:
        segment = signal[start:]
        segments_list.append(segment.unsqueeze(0))
    
    return segments_list

class offlineDataset(Dataset):

    def __init__(self, datafolder, fs=16000):

        self.noisyfolder = datafolder#os.path.join(datafolder, ')
        noisy_files = sorted(glob.glob('{}/*.wav'.format(self.noisyfolder)))
        
        self.noisy_chunks = []
        self.noisy_chunks_T = []
        self.filenames_batch = []
        self.chunk_norm_factors = []
        for noisy_file in tqdm(noisy_files):
            filename = noisy_file.split('/')[-1]   
            # Load wav
            y, _ = load(noisy_file) 
              
            segments = split_signals(y)
            # T_long = y.size(1)
            for segment in segments:
                # Normalize
                y = segment
                norm_factor = y.abs().max()
                self.chunk_norm_factors.append(norm_factor)
                self.noisy_chunks.append(segment)
                self.noisy_chunks_T.append(y.size(1))
                self.filenames_batch.append(filename)


    def __getitem__(self, index):

        data = self.noisy_chunks[index]

        return torch.Tensor(data), self.noisy_chunks_T[index], self.filenames_batch[index], self.chunk_norm_factors[index]

    def __len__(self):
        return len(self.noisy_chunks)

class PairedDataLoader(object):
    def __init__(self, data_set, batch_size, is_shuffle=False):
        self.data_loader = DataLoader(dataset=data_set, 
                                      batch_size=batch_size, 
                                      shuffle=is_shuffle,                                 
                                      drop_last=False, 
                                      collate_fn=self.collate_fn, 
                                      num_workers=32)

    @staticmethod
    def collate_fn(batch):
        data_ = []
        T_orig_ = []
        filenames_ = []
        norm_factors_ = []
        for item in batch:   
            data_.append(item[0])
            T_orig_.append(item[1])  
            filenames_.append(item[2])
            norm_factors_.append(item[3])
        max_length = max([x.shape[-1] for x in data_])
        padded_data = [torch.nn.functional.pad(x, (0, max_length-x.shape[-1]), 'constant', 0) for x in data_]
   
        padded_data = torch.stack(padded_data, dim=0)

        return padded_data, T_orig_, filenames_, torch.tensor(norm_factors_)

    def get_data_loader(self):
        return self.data_loader

def validation(model_wrapper, model, validation_data_loader):
    current_filename = None
    current_segments = []
    def write_and_clear_current_segments(filename, segments):
        if segments:
            base, ext = os.path.splitext(filename)
            filename_with_params = f"{base}_kappa{model.diffusion.kappa}_numsteps{model.diffusion.num_timesteps}{ext}"
            if len(segments) == 1:
                write(join(target_dir, filename), segments[0].cpu().numpy(), 16000)
            else:
                enhanced_signal = concatenate_segments_crossfade(segments, segment_length=len(segments[0]))
                write(join(target_dir, filename), enhanced_signal.cpu().numpy(), 16000)
        segments.clear()

    for val_batch_idx, (noisy_data_chunk, T_origs_batch, filenames_batch, norm_factors) in enumerate(validation_data_loader.get_data_loader()):
        y = noisy_data_chunk.squeeze(1)
        y = y / norm_factors.to(y.device)[:,None]
        Y = model._forward_transform(model._stft(y.cuda())).unsqueeze(1)
        Y_ = pad_spec(Y).to(device='cuda')

        Y = torch.cat([Y_.real, Y_.imag], dim=1)
        # Reverse sampling
        Y_mag = torch.sqrt(Y[:,[0],:,:]**2 + Y[:,[1],:,:]**2)
        mask, _ = model.frontend(Y[:,[0],:,:], Y[:,[1],:,:], Y_mag)


        sample = model_wrapper(Y, mask)

        # Backward transform in time domain
        max_length = max(T_origs_batch)
        sample_decode = torch.view_as_complex(torch.stack([sample[:,0,:,:], sample[:,1,:,:]], dim=-1))
        x_hat = model.to_audio(sample_decode.squeeze(1), max_length) 


        # Renormalize

        x_hat = x_hat * norm_factors.to(x_hat.device)[:,None]
        for i, filename in enumerate(filenames_batch):
            if filename != current_filename:
                write_and_clear_current_segments(current_filename, current_segments)
                current_filename = filename
            x_hat_trimmed = x_hat[i, :T_origs_batch[i]]
            current_segments.append(x_hat_trimmed)
    write_and_clear_current_segments(current_filename, current_segments)




def concatenate_segments_crossfade(segments, segment_length=192000, overlap=32):
    count = len(segments)
    total_length = int(segment_length * count - overlap * (count - 1))
    
    output_signal = torch.zeros(total_length).float().to(device=segments[0].device)
    
    fade_in = torch.linspace(0, 1, overlap).to(device=segments[0].device)
    fade_out = torch.linspace(1, 0, overlap).to(device=segments[0].device)
    
    start = 0
    for i, segment in enumerate(segments):
        end = start + segment_length
        segment = segment.squeeze()
        if i > 0:
            print('omg')
            output_signal[start:start+overlap] *= fade_out
            output_signal[start:start+overlap] += segment[:overlap] * fade_in
            actual_length = min(segment_length - 2 * overlap, segment.size(0) - 2 * overlap)
            output_signal[start+overlap:start+overlap+actual_length] = segment[overlap:overlap+actual_length]
        else:
            if segment.size(0) < total_length:
                output_signal[:segment.size(0)] = segment
            else:
                output_signal[start:end-overlap] = segment[:-overlap]
        
        if i < count - 1:
            output_signal[end-overlap:end] = segment[-overlap:] * fade_out
        
        start += segment_length - overlap
    
    return output_signal[:(count-1) * len(segments[0]) + len(segments[-1])]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the test data (must have subdirectory noisy/)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--batch_size", type=int, default=12)
    args = parser.parse_args()

    noisy_dir = args.noisy_dir
    checkpoint_file = args.ckpt

    global target_dir
    target_dir = args.enhanced_dir
    ensure_dir(target_dir)
    batch_size = args.batch_size
    # Settings
    sr = 16000


    # Load score model 
    model = GaldSE.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=3, num_workers=0, kwargs=dict(gpu=False)).cuda()
    model.train(False)

    # model.diffusion.power=0.39
    # model.hparams.power = 1
    # model.hparams.num_diffusion_steps=5
    print(model.hparams)
    
    for param in model.parameters():
        param.requires_grad = False

    model_wrapper = SamplingWrapper(model)
    model_wrapper = DataParallel(model_wrapper)
    model_wrapper.cuda()

    test_data_set = offlineDataset(noisy_dir)
    test_data_loader = PairedDataLoader(test_data_set, batch_size, False)
    validation(model_wrapper, model, test_data_loader)
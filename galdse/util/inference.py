import torch
from torchaudio import load

from pesq import pesq
from pystoi import stoi

from .other import si_sdr, pad_spec
# Settings
sr = 16000


def evaluate_model(model, num_eval_files):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 
        T_orig = x.size(1)   

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y_ = pad_spec(Y)
        y = y * norm_factor
        Y = torch.cat([Y_.real, Y_.imag], dim=1)
        y_mag = torch.sqrt(Y[:,[0],:,:]**2 + Y[:,[1],:,:]**2)
        mask_mag, _ = model.frontend(Y[:,[0],:,:], Y[:,[1],:,:], y_mag)

        mask = (mask_mag).detach()
        sample_decode = model.diffusion.p_sample_loop_progressive(
            y=Y,
            model=model,
            mask=mask,
            noise=None,
            clip_denoised=False,
            progress=False)

        sample_decode = torch.view_as_complex(torch.stack([sample_decode[:,0,:,:], sample_decode[:,1,:,:]], dim=-1))
        x_hat = model.to_audio(sample_decode.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
 
        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files


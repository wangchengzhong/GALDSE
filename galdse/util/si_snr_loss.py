import torch
smallVal = 1e-12
def si_snr_loss(y_pred_batch, y_true_batch, reduction="mean"):
    """Compute the si_snr score and return -1 * that score.

    This function can be used as a loss function for training
    with SGD-based updates.

    Arguments
    ---------
    y_pred_batch : torch.Tensor
        The degraded (enhanced) waveforms.
    y_true_batch : torch.Tensor
        The clean (reference) waveforms.
    reduction : str
        The type of reduction ("mean" or "batch") to use.

    Example
    -------
    """

    y_pred_batch = torch.squeeze(y_pred_batch, dim=-1)
    y_true_batch = torch.squeeze(y_true_batch, dim=-1)

    batch_size = y_pred_batch.shape[0]
    SI_SNR = torch.zeros(batch_size)

    for i in range(0, batch_size):  # Run over mini-batches
        s_target = y_true_batch[i]
        s_estimate = y_pred_batch[i]

        # s_target = <s', s>s / ||s||^2
        dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)
        s_target_energy = (
            torch.sum(s_target ** 2, dim=0, keepdim=True) + smallVal
        )
        proj = dot * s_target / s_target_energy

        # e_noise = s' - s_target
        e_noise = s_estimate - proj

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
            torch.sum(e_noise ** 2, dim=0) + smallVal
        )
        SI_SNR[i] = 10 * torch.log10(si_snr_beforelog + smallVal)

    if reduction == "mean":
        return -SI_SNR.mean()

    return -SI_SNR
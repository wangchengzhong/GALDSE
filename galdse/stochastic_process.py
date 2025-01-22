
import torch
import numpy as np
import math
from galdse.util.registry import Registry
StoProcessRegistry = Registry("StoProcess") # stochastic process registry

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def get_named_beta_schedule_(schedule_name, num_diffusion_timesteps, beta_start, beta_end):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return np.linspace(
            beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=np.float64
        )**2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def get_named_eta_schedule(
        schedule_name='exponential',
        num_diffusion_timesteps=15,
        min_noise_level=0.005, #original 0.01
        etas_end=0.99,
        kappa=0.5,
        power=0.3):
    """
    Get a pre-defined eta schedule for the given name.

    The eta schedule library consists of eta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    """
    if schedule_name == 'exponential':
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = torch.exp(torch.tensor(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start)))
        base = torch.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = torch.linspace(0, 1, num_diffusion_timesteps)**power
        power_timestep *= (num_diffusion_timesteps-1)
        sqrt_etas = torch.pow(base, power_timestep) * etas_start

    else:
        raise ValueError(f"Unknown schedule_name {schedule_name}")

    return sqrt_etas

@StoProcessRegistry.register("RESSHIFT")
class RESSHIFT():
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--kappa", type=float, default=0.5, help="controls variance of the overall noise")
        parser.add_argument("--num_diffusion_steps", type=int, default=6, help="number of diffusion steps for training and sampling.")
        parser.add_argument("--power", type=float, default=0.39, help="controls how the diffusion evolves along time axis")
        parser.add_argument("--min_noise_level", type=float, default=0.005, help="minimum noise applied at the start of the diffusion process")
        parser.add_argument("--etas_end", type=float, default=0.99, help="maximum noise level at the end of the diffusion chain")
        return parser
    
    def __init__(self, kappa=1, num_diffusion_steps=6, power=1.0, min_noise_level=0.001, etas_end=0.99, **ignored_kwargs):

        # super().__init__()
        self.kappa = kappa
        self.num_diffusion_steps = num_diffusion_steps
        self.power = power
        self.min_noise_level = min_noise_level
        self.etas_end = etas_end
        sqrt_etas = get_named_eta_schedule(kappa=kappa, num_diffusion_timesteps=num_diffusion_steps, power=power, min_noise_level=min_noise_level, etas_end=etas_end)
        self.sqrt_etas = sqrt_etas.float()
        self.etas = self.sqrt_etas ** 2
        assert len(self.etas.shape) == 1, "etas must be 1-D"
        assert (self.etas > 0).all() and (self.etas <= 1).all()

        self.num_timesteps = int(self.etas.shape[0])
        self.etas_prev = np.append(0.0, self.etas[:-1])
        self.alpha = self.etas - self.etas_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = kappa**2 * self.etas_prev / self.etas * self.alpha
        self.posterior_variance_clipped = np.append(
                self.posterior_variance[1], self.posterior_variance[1:]
                )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(self.posterior_variance_clipped)
        self.posterior_mean_coef1 = self.etas_prev / self.etas
        self.posterior_mean_coef2 = self.alpha / self.etas
        

    def q_mean_variance(self, x_start, y, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        cur_etas = self.etas[t].float()
        mean = cur_etas[:,None,None,None] * (y - x_start) + x_start
        variance = cur_etas[:,None,None,None] * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance


    def q_sample(self, x_start, y, t, noise):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
 
        assert noise.shape == x_start.shape
        cur_etas = self.etas[t].to(x_start.device)
        cur_sqrt_etas = self.sqrt_etas[t].to(x_start.device) * self.kappa
        return cur_etas[:,None,None,None] * (y-x_start) + x_start + cur_sqrt_etas[:,None,None,None] * noise
        return (
            _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas * self.kappa, t, x_start.shape) * noise
        )
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        cur_posterior_mean_coef1 = self.posterior_mean_coef1[t].to(x_t.device).float()
        cur_posterior_mean_coef2 = self.posterior_mean_coef2[t].to(x_t.device).float()
        cur_posterior_variance = self.posterior_variance[t].to(x_t.device).float()
        cur_posterior_log_variance_clipped = torch.tensor(self.posterior_log_variance_clipped[t]).float()
        posterior_mean = cur_posterior_mean_coef1[:,None,None,None] * x_t + cur_posterior_mean_coef2[:,None,None,None] * x_start

        posterior_variance = cur_posterior_variance[:,None,None,None].expand(x_t.shape)
        posterior_log_variance_clipped = cur_posterior_log_variance_clipped[...,None,None,None].expand(x_t.shape)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(
        self, model, x_t, y, mask, t,
        denoised_fn=None,
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x_t.shape[:2]
        assert t.shape == (B,)

        model_output = model(torch.cat([x_t, y, mask], dim=1), t.to(y.device), **model_kwargs)

        cur_posterior_variance = self.posterior_variance[t].float()
        cur_posterior_log_variance_clipped = torch.tensor(self.posterior_log_variance_clipped[t]).float()

        model_variance = cur_posterior_variance[:,None,None,None].expand(x_t.shape).to(x_t.device).float()
        model_log_variance = cur_posterior_log_variance_clipped[...,None,None,None].expand(x_t.shape).to(x_t.device).float()

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            return x


        pred_xstart = process_xstart(model_output)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )

        assert (
            model_mean.shape == pred_xstart.shape == x_t.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    def p_sample(self, model, x, y, mask, t, denoised_fn=None, model_kwargs=None, noise_repeat=False):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            y,
            mask,
            t,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x) * mask 
        if noise_repeat:
            noise = noise[0,].repeat(x.shape[0], 1, 1, 1)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1))).to(x.device)
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "mean":out["mean"]}


    def p_sample_loop_progressive(
            self, y, model, mask,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        z_y = y

        # generating noise
        if noise is None:
            noise = torch.randn_like(z_y) * mask
        if noise_repeat:
            noise = noise[0,].repeat(z_y.shape[0], 1, 1, 1)
        z_sample = self.prior_sample(z_y, noise)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        # indices = [9, 7, 5, 3, 0]
        # print(indices)
        for i in indices:
            t = torch.tensor([i] * y.shape[0])
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    z_sample,
                    z_y,
                    mask,
                    t,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat,
                )
                # yield out
                z_sample = out["sample"]
        return out['sample']

    def prior_sample(self, y, noise):
        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """
        t = torch.tensor([self.num_timesteps-1,] * y.shape[0]).long()
        cur_sqrt_etas = self.sqrt_etas[t].to(y.device).float() * self.kappa
        return y + cur_sqrt_etas[:,None,None,None] * noise

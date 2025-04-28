"""
Inspired from: https://github.com/yang-song/score_sde_pytorch
"""

import argparse
import numpy as np
import pickle
from scipy import integrate
import fmnist_sde_lib
import torch

import fmnist_dataset as ds_edm


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn

def get_likelihood_fn(model, sde, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45'):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
        model: A score model.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
        rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
        atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
        method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.

    Returns:
        A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def score_fn(x, t):
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
        return model(x, labels)

    def drift_fn(x, t):
        """The drift function of the reverse-time SDE."""
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def div_fn(x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(xx, tt))(x, t, noise)

    def likelihood_fn(data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            data: A PyTorch tensor.

        Returns:
            bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
            z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
                probability flow ODE.
            nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(drift_fn(sample, vec_t))
                logp_grad = to_flattened_numpy(div_fn(sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
            solution = integrate.solve_ivp(ode_func, (0.002, 80), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
            prior_logp = sde.prior_logp(z)
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            offset = np.log2(127.5) # https://arxiv.org/pdf/1705.05263 (2.4)
            bpd = bpd + offset
            return bpd, z, nfe

    return likelihood_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Path to the trained FMNIST model', required=True)
    parser.add_argument('--data-path', help='Path to the preprocessed FMNIST dataset', required=True)
    args = parser.parse_args()

    # Load FMNIST model
    device = torch.device('cuda')
    with open(args.model_path, 'rb') as f:
        fmnist_model = pickle.load(f)['ema'].to(device)

    # Load FMNIST dataset
    fmnist_ds = ds_edm.ImageFolderDataset(args.data_path)
    fmnist_dl = torch.utils.data.DataLoader(fmnist_ds, batch_size=1, shuffle=False)
    fmnist_iter = iter(fmnist_dl)

    # Compute log-likelihoods (bits/dim)
    sde = fmnist_sde_lib.EDM(sigma_min=0.002, sigma_max=80)
    likelihood_fn = get_likelihood_fn(fmnist_model, sde)
    bpds = []

    for batch_id in range(1000):
        print(f"----- {batch_id} -----")
        batch, _ = next(fmnist_iter)
        eval_batch = batch.to(device).to(torch.float32) / 127.5 - 1
        bpd, _, _ = likelihood_fn(eval_batch)
        print(f"bpd: {bpd}")
        bpds.extend(bpd.cpu())

    print('-'*5)
    print(f"mean bpd: {sum(bpds) / len(bpds)}")

if __name__ == "__main__":
    main()

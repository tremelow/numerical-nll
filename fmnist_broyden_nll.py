import argparse
import numpy as np
import os
import pickle
import torch

import fmnist_dataset as ds_edm


def get_likelihood_fn(model, t_steps, sigma_max, device):
    def drift_fn(x, t):
        return (x - model(x, t, None)) / t

    def step_edm_flat(x_cur, t_cur, t_next, shape):
        x_cur = x_cur.reshape(shape)
        drift = drift_fn(x_cur, t_cur)
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)
        return x_cur + (t_next - t_cur) * drift

    def flat_edm_with_jacobian(xk, t_cur, t_next, shape):                    
        with torch.enable_grad():
            xk.requires_grad_(True)
            F = step_edm_flat(xk, t_next, t_cur, shape)
            F_flat = F.flatten(start_dim=-3)
            grad_xk = lambda vec: torch.autograd.grad(vec.sum(), xk, create_graph=True)[0]
            gradients = []
            for i in range(F_flat.shape[-1]):
                if i % 1000 == 0:
                    print(f"Assembling the jacobian: {i}/{F_flat.shape[-1]}")
                grad_out = grad_xk(F_flat[:, i])
                gradients.append(grad_out.detach())
            F_prime = torch.stack(gradients, dim=-3).squeeze(dim=-1)
        xk.requires_grad_(False)
        return F_flat, F_prime

    def broyden_method(x_next_init, x_cur, t_cur, t_next):
        shape = x_next_init.shape
        vect = lambda u: u.flatten(start_dim=-3).unsqueeze(-1)
        x_cur_flat = vect(x_cur)
        x_next_flat = vect(x_next_init)
        eye = torch.eye(x_cur_flat.shape[-2], device=device).unsqueeze(0)
        invJ = eye
        get_res = lambda xk: vect(step_edm_flat(xk, t_next, t_cur, shape)) - x_cur_flat
        res = get_res(x_next_flat)

        print("Starting Broyden's method iterations...")
        while True:
            dx = -torch.matmul(invJ, res)
            x_next_flat = x_next_flat + 0.9 * dx

            err_batch = res.square().sum((-1, -2))
            err = err_batch.max()
            if err < 1e-6:
                break

            res_new = get_res(x_next_flat)
            dR = res_new - res
            invJ_dR = torch.matmul(invJ, dR)
            dy = (dx - invJ_dR) / (1e-4 + torch.sum(dx * invJ_dR, -2, keepdim=True))

            dx_invJ = torch.matmul(dx.transpose(-1, -2), invJ)
            invJ = invJ + dy * dx_invJ

            res = res_new

        _, F_prime = flat_edm_with_jacobian(x_next_flat, t_cur, t_next, shape)
        log_det = torch.linalg.slogdet(F_prime)[1]

        return x_next_flat.reshape(shape), log_det

    def prior_logp_fn(z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * sigma_max ** 2)

    def likelihood_fn(data):
        with torch.no_grad():
            log_det_list = []
            x_next = data
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                print(f"Step: {i + 1}/{len(t_steps) - 1}")
                x_cur = x_next

                drift = drift_fn(x_cur, t_cur)
                x_next = x_cur + (t_next - t_cur) * drift
                drift_prime = drift_fn(x_next, t_next)
                x_next = x_cur + (t_next - t_cur) * (0.5 * drift + 0.5 * drift_prime)

                x_next, log_det = broyden_method(x_next, x_cur, t_cur, t_next)
                log_det_list.append(log_det)
            
            x_final = x_next
            prior_logp = prior_logp_fn(x_final)
            nll_batch = torch.stack(log_det_list, dim=1).sum(dim=1) - prior_logp
            nll_bpd = nll_batch / (np.log(2.) * np.prod(x_final.shape[1:]))
            offset = np.log2(127.5) # https://arxiv.org/pdf/1705.05263 (2.4)
            nll_bpd += offset

            return nll_bpd
    
    return likelihood_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='Path to the trained FMNIST model', required=True)
    parser.add_argument('--data-path', help='Path to the preprocessed FMNIST dataset', required=True)
    parser.add_argument('--num-steps', help='Total number of timesteps', required=True)
    parser.add_argument('--batch-size', help='Batch size', required=True)
    args = parser.parse_args()

    # Load FMNIST model
    device = torch.device('cuda')
    with open(args.model_path, 'rb') as f:
        fmnist_model = pickle.load(f)['ema'].to(device)

    # Load FMNIST dataset
    batch_size = int(args.batch_size)
    fmnist_ds = ds_edm.ImageFolderDataset(args.data_path)
    fmnist_dl = torch.utils.data.DataLoader(fmnist_ds, batch_size=batch_size, shuffle=False)
    fmnist_iter = iter(fmnist_dl)

    # Prepare timesteps
    sigma_min = 0.002
    sigma_max = 80.
    steps = int(args.num_steps)
    rho = 7
    step_indices = torch.arange(steps, dtype=torch.float32, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.flip(t_steps, dims=(0,)).to(device) # t_steps[0] < t_steps[1] < ... < t_steps[n] (forward)

    # Compute log-likelihoods (bits/dim)
    save_dir = f'nll_vals/{args.num_steps}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    likelihood_fn = get_likelihood_fn(fmnist_model, t_steps, sigma_max, device)
    bpds = []
    file_id = 0

    for batch_id in range(1000 // batch_size):
        print(f"----- {batch_id} -----")
        batch, _ = next(fmnist_iter)
        eval_batch = batch.to(device).to(torch.float32) / 127.5 - 1
        bpd = likelihood_fn(eval_batch)
        print(f"bpd: {bpd}")
        bpds.extend(bpd.cpu())

        torch.save(torch.stack(bpds), os.path.join(save_dir, f'{file_id}.pt'))
        bpds = []
        file_id += 1

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from gaussian import DynamicMultivariateNormal


def hexa_vertices(num_sides):
    th = 2.0 * np.pi / num_sides
    rot_th = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    var = np.diag([0.02, 0.001])
    mean = np.array([0.0, -1.0/np.sqrt(2)])
    mean_offset = 0.0 * np.array([20.0, 10.0])
    norms = []
    for _ in range(num_sides):
        norms.append(DynamicMultivariateNormal(2, mean + mean_offset, var))
        var = rot_th @ var @ rot_th.T
        mean = rot_th @ mean
    return norms

def cube_vertices(dim, side_len=1.0, var=1e-2):
    vertices1d = np.array([-side_len, side_len])
    all_vertices1d = vertices1d.reshape(1, 2).repeat(dim, axis=0)
    all_vertices = np.meshgrid(*all_vertices1d)
    vertices = np.stack(all_vertices, axis=-1).reshape(-1, dim)
    var = var * np.eye(dim)
    return [DynamicMultivariateNormal(dim, vertex, var) for vertex in vertices]

def plot_simulation(mix, t, x, show_every=50):
    num_plots = (len(t) - 1) // show_every + 1
    fig, ax = plt.subplots(1, num_plots, figsize=(15, 3))

    for i in range(num_plots):
        si = i * show_every
        ti = t[si]
        xi = x[:, si, :]

        x1_cont = np.linspace(xi[:, 0].min() - 1.0, xi[:, 0].max() + 1.0, 200)
        x2_cont = np.linspace(xi[:, 1].min() - 1.0, xi[:, 1].max() + 1.0, 200)
        x_cont = np.stack(np.meshgrid(x1_cont, x2_cont), -1)
        x1_quiv = x1_cont[5::10]
        x2_quiv = x2_cont[5::10]
        x_quiv = np.stack(np.meshgrid(x1_quiv, x2_quiv), -1)
        score, div_score = mix.score_with_div(ti, x_quiv)
    
        ax[i].scatter(*xi.T, s=1)
        ax[i].contour(x_cont[:, :, 0], x_cont[:, :, 1], np.log(1e-8 + mix.density(ti, x_cont)), levels=10, alpha=0.5, cmap="plasma")
        ax[i].quiver(x_quiv[:, :, 0], x_quiv[:, :, 1], score[:, :, 0], score[:, :, 1], div_score, alpha=0.8)
    return fig, ax

def solve_numerical_scheme(solver, mix, n_samples, t_min, tf, n_timesteps, linear_ts):
    x_init = mix.sample(n_samples)
    sol = solver(mix, t_min, tf, n_timesteps, linear_ts)

    return sol(x_init)

def solve_flow(mix, prior, n_data=5000, t_min=1e-6, tf=1.0):
    def flat_extended_ode(t, x_cumdiv_flat):
        x, _ = np.split(x_cumdiv_flat.reshape(-1, mix.dim + 1), [mix.dim], -1)
        dx, dlogp = mix.extended_ode(t, x)
        return np.concatenate([dx, dlogp], 1).flatten()

    x_data = mix.sample(n_data)
    delta_logp = np.zeros((n_data, 1))
    x_logp_init = np.concatenate([x_data, delta_logp], axis=1)

    sol = solve_ivp(flat_extended_ode, (t_min, tf), x_logp_init.flatten())

    x_logp_fin = sol.y[:, -1].reshape(n_data, mix.dim + 1)
    x, delta_logp = np.split(x_logp_fin, [mix.dim], -1)
    prior_fin = np.log(prior.density(x))
    return x, -(delta_logp[:, 0] + prior_fin).mean() / np.log(2.0) / mix.dim

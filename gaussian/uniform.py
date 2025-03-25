import numpy as np
from numpy.random import SeedSequence, default_rng

from .static import MultivariateNormal, Mixture

DEFAULT_SEED = 42


from .dynamic import DynamicMixture

class DynamicUniformGaussianMixture(DynamicMixture):
    """
    Starting from an empirical measure sum_k \delta(x - x_k).
    """
    def __init__(self, locs: np.ndarray, init_var=1e-2):
        self.locs = locs  # [N, d]
        self.num, self.dim = locs.shape
        self.init_var = init_var

        if weights == ():
            weights = np.ones(self.num_rv)
        weights = np.array(weights)
        self.weights = weights / np.sum(weights)
        self.norm_2pi = 1.0 / pow(np.sqrt(2.0 * np.pi), self.dim)

    def density(self, t, x):
        s_t, mu_t, noise2_t = self.evol_t(t)
        inv_var_t = 1.0 / (np.square(s_t) * self.init_var + noise2_t)
        mean_t = s_t * (self.locs + mu_t)

        out = np.zeros(x.shape[:-1])
        for w_k, mean_k in zip(self.weights, mean_t):
            dist_x_mean_k = np.square(x - mean_k).sum(-1)
            out += w_k * np.exp(-0.5 * dist_x_mean_k * inv_var_t)
        return out * self.norm_2pi * np.pow(inv_var_t, self.dim)

    def score(self, t, x):
        s_t, mu_t, noise2_t = self.evol_t(t)
        inv_std_t = 1.0 / np.sqrt(np.square(s_t) * self.init_var + noise2_t)
        mean_t = s_t * (self.locs + mu_t)

        tot_score = np.zeros_like(x)
        tot_prob = np.zeros((*x.shape[:-1], 1))
        for w_k, mean_k in zip(self.weights, mean_t):
            y_k = (mean_k - x) * inv_std_t
            prob_k = w_k * np.exp(-0.5 * np.sum(np.square(y_k), -1, keepdims=True))
            tot_prob += prob_k
            tot_score += prob_k * y_k
        return inv_std_t * tot_score / tot_prob

    def score_with_div(self, t, x):
        s_t, mu_t, noise2_t = self.evol_t(t)
        var_t = np.square(s_t) * self.init_var + noise2_t
        inv_std_t = 1.0 / np.sqrt(var_t)
        mean_t = s_t * (self.locs + mu_t)

        tot_score = np.zeros_like(x)
        tot_prob = np.zeros((*x.shape[:-1], 1))
        tot_norm_sq_score = np.zeros_like(tot_prob)
        for w_k, mean_k in zip(self.weights, mean_t):
            y_k = (mean_k - x) * inv_std_t
            norm_sq_y_k = np.sum(np.square(y_k), -1, keepdims=True)
            prob_k = w_k * np.exp(-0.5 * norm_sq_y_k)
            tot_prob += prob_k
            tot_score += prob_k * y_k
            tot_norm_sq_score += prob_k * norm_sq_y_k
        score = inv_std_t * tot_score / tot_prob
        norm_sq_tot_score = np.sum(np.square(score), axis=1, keepdims=True)
        # tot_div_score = -self.dim / (tot_prob * var_t)
        # tot_norm_sq_score /= (tot_prob * var_t)
        div_score = (tot_norm_sq_score - self.dim) / (tot_prob * var_t) - norm_sq_tot_score
        return score, div_score

    def ode(self, t, x):
        """Score-matching ODE in forward time"""
        return self.f(t, x) - 0.5 * self.g_sq(t) * self.score(t, x)

    def extended_ode(self, t, x):
        # in forward time, ODE in x concatenated with the volume variation, the time-differential of log(p)
        f_tx, div_f_tx = self.f_with_div(t, x)
        score_tx, div_score_tx = self.score_with_div(t, x)
        g_sq_t = self.g_sq(t)
        ode = f_tx - 0.5 * g_sq_t * score_tx
        vol = div_f_tx - 0.5 * g_sq_t * div_score_tx
        return ode, vol
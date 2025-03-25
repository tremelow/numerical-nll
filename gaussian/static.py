from abc import abstractmethod

import numpy as np
from numpy.random import SeedSequence, default_rng

DEFAULT_SEED = 42


class Covariance:
    """
    Attributes: cov, inv_cov, std, inv_std, sqrt_det, trace_inv_cov
    """

    @abstractmethod
    def _apply(self, x, mat):
        pass

    def expand(self, y):  # for sampling
        return self._apply(y, self.std)

    def reduce(self, x):  # for the density
        return self._apply(x, self.inv_std)

    def reduce_sq(self, x):  # for the score
        return self._apply(x, self.inv_cov)

    def reduce_grad_and_divgrad(self, x):  # for the score and its divergence
        return self._apply(x, self.inv_cov), self.trace_inv_cov

    def reduce_grad_and_jac(self, x):  # for the score and its divergence
        return self._apply(x, self.inv_cov), self.inv_cov


class CovarianceMat(Covariance):
    def __init__(self, cov: np.ndarray):
        U, S, Uh = np.linalg.svd(cov, hermitian=True)
        sqrt_S = np.sqrt(S)
        self.cov = cov
        self.std = (U * sqrt_S) @ Uh
        self.inv_std = (U / sqrt_S) @ Uh
        self.inv_cov = (U / S) @ Uh
        self.sqrt_det = sqrt_S.prod()
        self.trace_inv_cov = np.linalg.trace(self.inv_cov)

    def _apply(self, y, mat):
        return y @ mat.T


class CovarianceVec(Covariance):
    def __init__(self, cov: np.ndarray):
        self.cov = cov
        self.std = np.sqrt(cov)
        self.inv_std = 1.0 / self.std
        self.inv_cov = 1.0 / cov
        self.sqrt_det = self.std.prod()
        self.trace_inv_cov = self.inv_cov.sum()

    def _apply(self, y, mat):
        return y * mat


class MultivariateNormal:
    SQRT_2PI = np.sqrt(2.0 * np.pi)

    def __init__(
        self, dim: int, mean: np.ndarray | float = 0.0, cov: np.ndarray | float = 1.0
    ):
        assert isinstance(dim, int), "`dim` must be integer."
        self.dim = dim

        self.mean = np.array(mean)
        assert self.mean.ndim < 2, "`mean` must be at most 1-dimensional."
        if self.mean.ndim == 1:
            assert (
                len(self.mean) == dim
            ), "`mean` must be either scalar or of size `dim`."

        cov = np.array(cov)
        assert cov.ndim < 3, "`cov` must be at most 2-dimensional."
        if cov.ndim == 1:
            assert len(cov) == dim, "If `cov` is a vector, it must be of size `dim`."

        if cov.ndim == 2:
            assert cov.shape == (
                dim,
                dim,
            ), "If `cov` is 2-dimensional, it must be of shape `(dim, dim)`."
            self.cov = CovarianceMat(cov)
        else:
            self.cov = CovarianceVec(cov)

        self.norm_cst = 1.0 / (pow(self.SQRT_2PI, self.dim) * self.cov.sqrt_det)

    def density(self, x):
        y = self.cov.reduce(x - self.mean)
        y_sq = np.sum(np.square(y), axis=-1)
        return np.exp(-0.5 * y_sq) * self.norm_cst

    def score(self, x):
        return -self.cov.reduce_sq(x - self.mean)

    def score_with_div(self, x):
        score, div_score = self.cov.reduce_grad_and_divgrad(x - self.mean)
        return -score, -div_score

    def score_with_jac(self, x):
        score, jac_score = self.cov.reduce_grad_and_jac(x - self.mean)
        return -score, -jac_score

    def sample(self, size: tuple | int = (), seed: SeedSequence | int = DEFAULT_SEED):
        if isinstance(size, int):
            size = (size,)
        y_sample = default_rng(seed).normal(size=(*size, self.dim))
        return self.cov.expand(y_sample) + self.mean


class Mixture:
    def __init__(self, rand_vars: tuple[MultivariateNormal], weights=()):
        self.rv = list(rand_vars)
        self.num_rv = len(rand_vars)
        assert self.num_rv > 0, "At least one density necessary"
        self.dim = rand_vars[0].dim
        assert all(
            f.dim == self.dim for f in self.rv
        ), "All random variables must have the same dimension"

        if weights == ():
            weights = np.ones(self.num_rv)
        weights = np.array(weights)
        self.weights = weights / np.sum(weights)

    def density(self, x, **kwargs):
        out = np.zeros(x.shape[:-1])
        for wk, rvk in zip(self.weights, self.rv):
            out += wk * rvk.density(x, **kwargs)
        return out

    def score(self, x, **kwargs):
        tot_score = np.zeros_like(x)
        tot_prob = np.zeros((*x.shape[:-1], 1))
        for wk, rvk in zip(self.weights, self.rv):
            probk = (wk * rvk.density(x, **kwargs))[..., None]
            tot_prob += probk
            tot_score += probk * rvk.score(x, **kwargs)
        return tot_score / tot_prob

    def score_with_div(self, x, **kwargs):
        tot_prob = np.zeros((*x.shape[:-1], 1))

        tot_score = np.zeros_like(x)
        tot_norm_sq_score = np.zeros_like(tot_prob)
        tot_div_score = np.zeros_like(tot_prob)
        
        for w_k, rv_k in zip(self.weights, self.rv):
            prob_k = w_k * rv_k.density(x, **kwargs)[..., None]
            tot_prob += prob_k

            score_k, div_score_k = rv_k.score_with_div(x, **kwargs)
            tot_score += prob_k * score_k

            tot_norm_sq_score += prob_k * np.sum(np.square(score_k), -1, keepdims=True)
            tot_div_score += prob_k * div_score_k

        score = tot_score / (1e-8 + tot_prob)
        norm_sq_tot_score = np.sum(np.square(score), axis=-1, keepdims=True)
        div_score = (tot_norm_sq_score + tot_div_score) / (1e-8 + tot_prob) - norm_sq_tot_score
        return score, div_score

    def score_with_jac(self, x, **kwargs):
        tot_prob = np.zeros((*x.shape[:-1], 1, 1))

        tot_score = np.zeros_like(x)[..., None]
        tot_tensor_score = np.zeros((*x.shape[:-1], self.dim, self.dim))
        tot_jac_score = np.zeros_like(tot_tensor_score)

        for w_k, rv_k in zip(self.weights, self.rv):
            prob_k = w_k * rv_k.density(x, **kwargs)[..., None, None]
            tot_prob += prob_k

            score_k, jac_score_k = rv_k.score_with_jac(x, **kwargs)
            score_k = score_k[..., :, None]
            tot_score += prob_k * score_k

            tot_tensor_score += prob_k * score_k * np.moveaxis(score_k, -1, -2)
            tot_jac_score += prob_k * jac_score_k

        score = tot_score / (1e-8 + tot_prob)
        tensor_tot_score = score * np.moveaxis(score, -1, -2)
        jac_score = (tot_tensor_score + tot_jac_score) / (1e-8 + tot_prob) - tensor_tot_score
        return score[..., 0], jac_score

    def sample(
        self, size: tuple | int = (), seed: SeedSequence | int = DEFAULT_SEED, **kwargs
    ):
        if isinstance(size, int):
            size = (size,)
        if isinstance(seed, int):
            seed = SeedSequence(seed)
        seeds = seed.spawn(self.num_rv + 1)

        rng = default_rng(seeds[-1])
        choice = rng.choice(self.num_rv, size=(*size, 1), p=self.weights)

        x = np.zeros((*size, self.dim))
        for k, rvk in enumerate(self.rv):
            # sample greedily and filter to keep only k-th distribution
            x += (choice == k) * rvk.sample(size, seeds[k], **kwargs)
        return x

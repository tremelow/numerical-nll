from abc import abstractmethod

import numpy as np
from numpy.random import SeedSequence, default_rng

from .static import MultivariateNormal, Mixture

DEFAULT_SEED = 42


##########################
####       MEAN       ####
##########################


class DynamicMean:
    """
    Utilities for the mean, taking into account transformations
    m <- s * (m - d)
    """

    def __init__(self, mean):
        self.mean = np.array(mean)

    def __call__(self, scaling=0.0, drift=0.0):
        return scaling * (self.mean + drift)


def build_dynamic_mean(dim: int, mean: float | np.ndarray) -> DynamicMean:
    mean = np.array(mean)
    assert mean.ndim < 2, "`mean` must be at most 1-dimensional."
    if mean.ndim == 1:
        assert len(mean) == dim, "`mean` must be either scalar or of size `dim`."
    return DynamicMean(mean)


##########################
####    COVARIANCE    ####
##########################


class DynamicCovariance:
    """
    Utilities for covariance matrices, taking into account transformations
    C <- s * C + B * I
    """

    @abstractmethod
    def expand(self, y, scaling=1.0, added_noise_sq=0.0):
        """For sampling. Returns $\Sigma y$"""
        pass

    @abstractmethod
    def reduced_norm_sq(self, x, scaling=1.0, added_noise_sq=0.0):
        """For the density. Returns $x^T C^{-1} x$ and $det(C)$."""
        pass

    @abstractmethod
    def reduce_grad(self, x, scaling=1.0, added_noise_sq=0.0):
        """For the score. Returns $C^{-1} x$."""
        pass


class DynamicCovarianceMat(DynamicCovariance):
    def __init__(self, cov: np.ndarray):
        self.cov = cov

        U, S, Uh = np.linalg.svd(cov, hermitian=True)
        self.to_sing_vec = U
        self.sing_vals = S
        self.from_sing_vec = Uh

    def expand(self, y, scaling=1.0, added_noise_sq=0.0):
        std = np.sqrt(scaling * self.sing_vals + added_noise_sq)
        return ((y @ self.to_sing_vec) * std) @ self.from_sing_vec

    def reduced_norm_sq(self, x, scaling=1.0, added_noise_sq=0.0):
        diag_cov = scaling * self.sing_vals + added_noise_sq
        # simplification: \| Uh @ z \|^2 = \|z\|^2
        y_norm_sq = np.square((x @ self.to_sing_vec) / np.sqrt(diag_cov)).sum(-1)
        return y_norm_sq, diag_cov.prod()

    def reduce_grad(self, x, scaling=1.0, added_noise_sq=0.0):
        diag_cov = scaling * self.sing_vals + added_noise_sq
        return ((x @ self.to_sing_vec) / diag_cov) @ self.from_sing_vec

    def reduce_grad_and_divgrad(self, x, scaling=1.0, added_noise_sq=0.0):
        inv_diag_cov = 1.0 / (scaling * self.sing_vals + added_noise_sq)
        y = ((x @ self.to_sing_vec) * inv_diag_cov) @ self.from_sing_vec
        return y, np.sum(inv_diag_cov)

    def reduce_grad_and_jac(self, x, scaling=1.0, added_noise_sq=0.0):
        inv_diag_cov = 1.0 / (scaling * self.sing_vals + added_noise_sq)
        # y = ((x @ self.to_sing_vec) * inv_diag_cov) @ self.from_sing_vec
        inv_cov = (self.to_sing_vec * inv_diag_cov) @ self.from_sing_vec
        y = x @ inv_cov
        return y, inv_cov


class DynamicCovarianceVec(DynamicCovariance):
    def __init__(self, cov: np.ndarray):
        self.cov = cov

    def expand(self, y, scaling=1.0, added_noise_sq=0.0):
        std = np.sqrt(scaling * self.cov + added_noise_sq)
        return y * std

    def reduced_norm_sq(self, x, scaling=1.0, added_noise_sq=0.0):
        diag_cov = scaling * self.cov + added_noise_sq
        return np.square(x / np.sqrt(diag_cov)).sum(-1), diag_cov.prod()

    def reduce_grad(self, x, scaling=1.0, added_noise_sq=0.0):
        cov = scaling * self.cov + added_noise_sq
        return x / cov

    def reduce_grad_and_divgrad(self, x, scaling=1.0, added_noise_sq=0.0):
        inv_cov = scaling * self.cov + added_noise_sq
        return x * inv_cov, inv_cov.sum()
    
    def reduce_grad_and_jac(self, x, scaling=1.0, added_noise_sq=0.0):
        inv_cov = scaling * self.cov + added_noise_sq
        return x * inv_cov, inv_cov


class DynamicCovarianceScal(DynamicCovarianceVec):
    def __init__(self, dim: int, cov: np.ndarray):
        self.dim = dim
        self.cov = cov

    def reduced_norm_sq(self, x, scaling=1.0, added_noise_sq=0.0):
        diag_cov = scaling * self.cov + added_noise_sq
        return np.square(x / np.sqrt(diag_cov)).sum(-1), np.pow(diag_cov, self.dim)

    def reduce_grad_and_divgrad(self, x, scaling=1.0, added_noise_sq=0.0):
        inv_cov = scaling * self.cov + added_noise_sq
        return x * inv_cov, self.dim * inv_cov


def build_dynamic_cov(dim: int, cov: float | np.ndarray) -> DynamicCovariance:
    cov = np.array(cov)

    match cov.ndim:
        case 0:
            return DynamicCovarianceScal(dim, cov)
        case 1:
            assert len(cov) == dim, "If `cov` is a vector, it must be of size `dim`."
            return DynamicCovarianceVec(cov)
        case 2:
            assert cov.shape == (
                dim,
                dim,
            ), "If `cov` is 2-dimensional, it must be of shape `(dim, dim)`."
            return DynamicCovarianceMat(cov)
        case _:
            raise ValueError("`cov` must be at most 2-dimensional.")


################################
####    RANDOM VARIABLES    ####
################################


class DynamicMultivariateNormal:
    SQRT_2PI = np.sqrt(2.0 * np.pi)

    def __init__(
        self, dim: int, mean: np.ndarray | float = 0.0, cov: np.ndarray | float = 1.0
    ):
        assert isinstance(dim, int), "`dim` must be integer."
        self.dim = dim
        self.mean = build_dynamic_mean(dim, mean)
        self.cov = build_dynamic_cov(dim, cov)
        self.base_norm_cst = 1.0 / pow(self.SQRT_2PI, self.dim)

    def density(self, x, scaling=1.0, drift=0.0, added_noise_sq=0.0):
        x_cent = x - self.mean(scaling, drift)
        y_norm_sq, det_cov = self.cov.reduced_norm_sq(
            x_cent, scaling=scaling, added_noise_sq=added_noise_sq
        )
        return np.exp(-0.5 * y_norm_sq) * self.base_norm_cst / np.sqrt(det_cov)

    def score(self, x, scaling=1.0, drift=0.0, added_noise_sq=0.0):
        x_cent = x - self.mean(scaling, drift)
        return -self.cov.reduce_grad(
            x_cent, scaling=scaling, added_noise_sq=added_noise_sq
        )

    def score_with_div(self, x, scaling=1.0, drift=0.0, added_noise_sq=0.0):
        x_cent = x - self.mean(scaling, drift)
        y, tr = self.cov.reduce_grad_and_divgrad(
            x_cent, scaling=scaling, added_noise_sq=added_noise_sq
        )
        return -y, -tr

    def score_with_jac(self, x, scaling=1.0, drift=0.0, added_noise_sq=0.0):
        x_cent = x - self.mean(scaling, drift)
        y, jac = self.cov.reduce_grad_and_jac(
            x_cent, scaling=scaling, added_noise_sq=added_noise_sq
        )
        return -y, -jac

    def sample(
        self,
        size: tuple | int = (),
        seed: SeedSequence | int = DEFAULT_SEED,
        scaling=1.0,
        drift=0.0,
        added_noise_sq=0.0,
    ):
        if isinstance(size, int):
            size = (size,)
        y_sample = default_rng(seed).normal(size=(*size, self.dim))
        x_cent = self.cov.expand(
            y_sample, scaling=scaling, added_noise_sq=added_noise_sq
        )
        return x_cent + self.mean(scaling, drift)


class DynamicMixture(Mixture):
    @abstractmethod
    def a(self, t):
        pass

    @abstractmethod
    def b(self, t):
        pass

    def f(self, t, x):
        return self.a(t) * x + self.b(t)

    def f_with_div(self, t, x):
        a_t, b_t = self.a(t), self.b(t)
        return a_t * x + b_t, self.dim * a_t
    
    def f_with_jac(self, t, x):
        a_t, b_t = self.a(t), self.b(t)
        return a_t * x + b_t, a_t * np.eye(self.dim)

    @abstractmethod
    def g(self, t):
        # sqrt(alpha'(t)) with alpha(t) = t ** 2
        pass

    def g_sq(self, t):
        return self.g(t) ** 2

    @abstractmethod
    def scaling(self, t):
        # exp(int_0^t a(u) du)
        pass

    @abstractmethod
    def mean_drift(self, t):
        # int_0^t b(u) / s(u) du
        pass

    @abstractmethod
    def added_noise_sq(self, t):
        # scaled noise (s(t) * sigma(t)) ** 2
        # int_0^t (s(t) * g(u) / s(u)) ** 2 du
        pass

    def evol_t(self, t):
        # return self.scaling(t), self.mean_drift(t), self.added_noise_sq(t)
        return {
            "scaling": self.scaling(t),
            "drift": self.mean_drift(t),
            "added_noise_sq": self.added_noise_sq(t),
        }

    def density(self, t, x):
        return Mixture.density(self, x, **self.evol_t(t))

    def score(self, t, x):
        return Mixture.score(self, x, **self.evol_t(t))

    def score_with_div(self, t, x):
        return Mixture.score_with_div(self, x, **self.evol_t(t))

    def score_with_jac(self, t, x):
        return Mixture.score_with_jac(self, x, **self.evol_t(t))

    def ode(self, t, x):
        return self.f(t, x) - 0.5 * self.g_sq(t) * self.score(t, x)

    def extended_ode(self, t, x):
        # in forward time, ODE in x concatenated with the volume variation, the time-differential of log(p)
        f_tx, div_f_tx = self.f_with_div(t, x)
        score_tx, div_score_tx = self.score_with_div(t, x)
        g_sq_t = self.g_sq(t)
        ode = f_tx - 0.5 * g_sq_t * score_tx
        vol = div_f_tx - 0.5 * g_sq_t * div_score_tx
        return ode, vol

    def ode_with_jac(self, t, x):
        # in forward time, ODE in x concatenated with the volume variation, the time-differential of log(p)
        f_tx, jac_f_tx = self.f_with_jac(t, x)
        score_tx, jac_score_tx = self.score_with_jac(t, x)
        g_sq_t = self.g_sq(t)
        ode = f_tx - 0.5 * g_sq_t * score_tx
        jac = jac_f_tx - 0.5 * g_sq_t * jac_score_tx
        return ode, jac

    def sample(
        self, size: tuple | int = (), t: float = 0.0, seed: SeedSequence | int = DEFAULT_SEED, **kwargs
    ):
        if isinstance(size, int):
            size = (size,)
        if isinstance(seed, int):
            seed = SeedSequence(seed)
        seeds = seed.spawn(self.num_rv + 1)

        rng = default_rng(seeds[-1])
        choice = rng.choice(self.num_rv, size=(*size, 1), p=self.weights)

        x = np.zeros((*size, self.dim))
        evol_t = self.evol_t(t)
        for k, rvk in enumerate(self.rv):
            # sample greedily and filter to keep only k-th distribution
            x += (choice == k) * rvk.sample(size, seeds[k], **evol_t, **kwargs)
        return x


###########################
####       SDEs       ####
###########################


class VarianceExploding(DynamicMixture):
    def a(self, t):
        return 0.0

    def b(self, t):
        return 0.0

    def f(self, t, x):
        return 0.0

    def f_with_div(self, t, x):
        return 0.0, 0.0

    def g(self, t):
        return np.sqrt(2.0 * t)

    def g_sq(self, t):
        return 2.0 * t

    def scaling(self, t):
        return 1.0

    def mean_drift(self, t):
        return 0.0

    def added_noise_sq(self, t):
        return np.square(t)

    def added_noise(self, t):
        return t
    


class VariancePreserving(DynamicMixture):
    def __init__(self, rand_vars: tuple[MultivariateNormal], weights=(), beta_min=1e-2, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min
        super().__init__(rand_vars, weights)

    def alpha(self, t):
        return (0.5 * self.beta_d * t + self.beta_min) * t

    def alpha_prime(self, t):
        return self.beta_d * t + self.beta_min

    def a(self, t):
        return -0.5 * self.alpha_prime(t)

    def b(self, t):
        return 0.0

    def g(self, t):
        return np.sqrt(self.alpha_prime(t))

    def g_sq(self, t):
        return self.alpha_prime(t)

    def scaling(self, t):
        return np.exp(-0.5 * self.alpha(t))

    def mean_drift(self, t):
        return 0.0

    def added_noise_sq(self, t):
        return 1.0 - np.exp(-self.alpha(t))

    def eval_t(self, t):
        alpha_t = self.alpha(t)
        scaling = np.exp(-0.5 * alpha_t)
        drift = 0.0
        added_noise_sq = 1.0 - np.square(scaling)
        return {
            "scaling": scaling,
            "drift": drift,
            "added_noise_sq": added_noise_sq
        }


class SubVariancePreserving(DynamicMixture):
    def __init__(self, rand_vars: tuple[MultivariateNormal], weights=(), beta_min=1e-2, beta_max=20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min
        super().__init__(rand_vars, weights)

    def alpha(self, t):
        return (0.5 * self.beta_d * t + self.beta_min) * t

    def alpha_prime(self, t):
        return self.beta_d * t + self.beta_min
    
    def a(self, t):
        return -0.5 * self.alpha_prime(t)

    def b(self, t):
        return 0.0

    def g(self, t):
        return np.sqrt(self.g_sq(t))

    def g_sq(self, t):
        return self.alpha_prime(t) * (1.0 - np.exp(-2.0 * self.alpha(t)))

    def scaling(self, t):
        return np.exp(-0.5 * self.alpha(t))

    def mean_drift(self, t):
        return 0.0

    def added_noise_sq(self, t):
        return (1.0 - np.exp(-self.alpha(t))) ** 2

    def eval_t(self, t):
        alpha_t = self.alpha(t)
        scaling = np.exp(-0.5 * alpha_t)
        drift = 0.0
        added_noise_sq = np.square(1.0 - np.square(scaling))
        return {
            "scaling": scaling,
            "drift": drift,
            "added_noise_sq": added_noise_sq
        }

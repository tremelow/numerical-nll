"""
Inspired from: https://github.com/yang-song/score_sde_pytorch
"""

import abc
import torch
import numpy as np


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      # def sde(self, x, t):
      #   """Create the drift and diffusion functions for the reverse SDE/ODE."""
      #   drift, diffusion = sde_fn(x, t) # Eq.191 EDM page 32
      #   score = score_fn(x, t) # grad_x s_theta(x,t)
      #   drift = drift - diffusion * score # drift = drift - ... équivalent à f_tilde(x,t) = f(x,t) - ... (Eq.38)
      #   # Set the diffusion function to zero for ODEs.
      #   diffusion = 0. if self.probability_flow else diffusion
      #   return drift, diffusion
      
      def sde(self, x, t):
        #sigma_t = sde_fn(t)
        D_theta = score_fn(x, t)
        drift = (x - D_theta) / t[0]
        return drift, 0.

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

class EDM(SDE):
  def __init__(self, sigma_min=0.002, sigma_max=80, N=1000):
    super().__init__(N)
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
    self.N = N

  @property
  def T(self):
    return 1

  # def sde(self, x, t):
  #   sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
  #   # sigma = self.sigma_min * torch.sqrt((self.sigma_max / self.sigma_min) ** (2*t) - 1) # std distribution bruitée (Eq.30, Eq.199 EDM)
  #   drift = torch.zeros_like(x) # pas de dt = pas de drift (Eq.30, Eq.192 EDM)
  #   diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device)) # (Eq.30, Eq.191 EDM)
  #   return drift, diffusion

  def sde(self, t):
    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    return sigma

  def marginal_prob(self, x, t):
    std = t
    # std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    # std = self.sigma_min * torch.sqrt((self.sigma_max / self.sigma_min) ** (2*t) - 1) # std distribution bruitée (Eq.30, Eq.199 EDM)
    mean = x
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def prior_logp(self, z): # log p_T(x_T) (1,)
    shape = z.shape # (1, 3, 32, 32)
    N = np.prod(shape[1:]) # 3072
    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)

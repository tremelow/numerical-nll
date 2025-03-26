import numpy as np
from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate._ivp.common import warn_extraneous


class EulerSolver(OdeSolver):
    """
    Euler method solver for solve_ivp
    """

    def __init__(self, fun, t0, y0, t_bound, nt, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized=False)
        self.t_steps = np.linspace(t0, t_bound, nt + 1)
        self.cur_t_idx = 0
        
    def _step_impl(self):
        t = self.t
        t_next = self.t_steps[self.cur_t_idx + 1]
        y = self.y
        
        self.y_old = y
        self.y = y + (t_next - t) * self.fun(t, y)
        self.cur_t_idx += 1
        self.t = self.t_steps[self.cur_t_idx]

        return (True, None)
        
    def _dense_output_impl(self):
        pass

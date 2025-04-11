import numpy as np
# from scipy.integrate._ivp.base import OdeSolver
# from scipy.integrate._ivp.common import warn_extraneous


class EulerSolver():
    """
    Explicit Euler method solver
    """

    def __init__(self, mix, t_min, tf, nt, linear_ts=False):
        self.ode_with_jac = mix.ode_with_jac
        self.tf = tf

        if linear_ts:
            self.t_eval = np.linspace(t_min, tf, nt + 1, dtype=np.float32)
        else:
            rho = 7
            step_indices = np.arange(nt, dtype=np.float32)
            t_steps = (tf ** (1 / rho) + step_indices / (nt - 1) * (t_min ** (1 / rho) - tf ** (1 / rho))) ** rho
            self.t_eval = np.flip(t_steps)

    def prior_logp_fn(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.tf ** 2) - np.sum(z ** 2, axis=-1) / (2 * self.tf ** 2)
        
    def __call__(self, x_cur):
        x_list = [x_cur]
        log_det_list = []
        eye = np.eye(x_cur.shape[-1])

        for t_cur, t_next in zip(self.t_eval[:-1], self.t_eval[1:]):
            drift, jac = self.ode_with_jac(t_cur, x_cur)
            x_next = x_cur + (t_next - t_cur) * drift

            log_det = np.linalg.slogdet(eye + (t_next - t_cur) * jac)[1]

            x_list.append(x_next)
            log_det_list.append(log_det)
            x_cur = x_next
        
        x_final = x_next
        prior_logp = self.prior_logp_fn(x_final)
        nll_batch = -(np.sum(log_det_list, axis=0) + prior_logp)
        nll_bpd = nll_batch / (np.log(2.) * x_final.shape[-1])

        return self.t_eval, np.array(x_list).transpose(1, 0, 2), nll_bpd

# class EulerSolverSolveIVP(OdeSolver):
#     """
#     Explicit Euler method solver for solve_ivp
#     """

#     def __init__(self, fun, t0, y0, t_bound, nt, **extraneous):
#         warn_extraneous(extraneous)
#         super().__init__(fun, t0, y0, t_bound, vectorized=False)
#         self.t_steps = np.linspace(t0, t_bound, nt + 1)
#         self.cur_t_idx = 0
        
#     def _step_impl(self):
#         t = self.t
#         t_next = self.t_steps[self.cur_t_idx + 1]
#         y = self.y
        
#         self.y_old = y
#         self.y = y + (t_next - t) * self.fun(t, y)
#         self.cur_t_idx += 1
#         self.t = self.t_steps[self.cur_t_idx]

#         return (True, None)
        
#     def _dense_output_impl(self):
#         pass

import numpy as np
# from scipy.integrate._ivp.base import OdeSolver
# from scipy.integrate._ivp.common import warn_extraneous


class BroydenSolver:
    """
    Broyden method solver
    """

    def __init__(self, ode, t_min, tf, nt):
        self.ode = ode
        self.t_eval = np.linspace(t_min, tf, nt + 1)
    
    def step(self, x_cur, t_cur, t_next):
        x_cur = np.squeeze(x_cur, axis=-1)
        drift = np.expand_dims(self.ode(t_cur, x_cur), axis=-1)
        return (t_next - t_cur) * drift

    def broyden_method(self, x_next_init, x_cur, t_cur, t_next):
        x_next = np.expand_dims(x_next_init, axis=-1)
        x_cur = np.expand_dims(x_cur, axis=-1)

        eye = np.expand_dims(np.eye(x_cur.shape[-2]), axis=0)
        invJ = eye
        J = eye

        get_res = lambda xk: xk + self.step(xk, t_next, t_cur) - x_cur
        res = get_res(x_next)

        while True:
            dx = -np.matmul(invJ, res)
            x_next = x_next + 0.9 * dx

            err_sq = np.sum(np.square(dx))
            if err_sq < 1e-6:
                break

            res_new = get_res(x_next)
            dR = res_new - res
            invJ_dR = np.matmul(invJ, dR)
            dy = (dx - invJ_dR) / (1e-4 + np.sum(dx * invJ_dR, axis=-2, keepdims=True))
            dy_J = (dR - np.matmul(J, dx)) / (1e-4 + np.sum(np.square(dx), axis=-2, keepdims=True))

            dx_invJ = np.matmul(np.transpose(dx, (0, -1, -2)), invJ)
            invJ = invJ + dy * dx_invJ
            J = J + dy_J * np.transpose(dx, (0, -1, -2))

            res = res_new

        x_next = np.squeeze(x_next, axis=-1)

        return x_next

    def __call__(self, x):
        x_list = [x]
        for t_cur, t_next in zip(self.t_eval[:-1], self.t_eval[1:]):
            drift = self.ode(t_cur, x)
            x_euler = x + (t_next - t_cur) * drift
            drift_prime = self.ode(t_next, x_euler)
            x_heun = x + (t_next - t_cur) * (0.5 * drift + 0.5 * drift_prime)

            x = self.broyden_method(x_heun, x, t_cur, t_next)

            x_list.append(x)

        return self.t_eval, np.array(x_list).transpose(1, 0, 2)

# class BroydenSolver(OdeSolver):
#     """
#     Broyden method solver for solve_ivp
#     Warning: too slow to be used in practice
#     """

#     def __init__(self, fun, t0, y0, t_bound, nt, **extraneous):
#         warn_extraneous(extraneous)
#         super().__init__(fun, t0, y0, t_bound, vectorized=True)
#         self.t_steps = np.linspace(t0, t_bound, nt + 1)
#         self.cur_t_idx = 0
    
#     def euler_step(self, x_cur, t_cur, t_next):
#         x_cur = np.squeeze(x_cur, axis=-1)
#         drift = np.expand_dims(self.fun(t_cur, x_cur), axis=-1)
#         return (t_next - t_cur) * drift

#     def broyden_method(self, x_next_init, x_cur, t_cur, t_next):
#         x_next = np.expand_dims(x_next_init, axis=-1)
#         x_cur = np.expand_dims(x_cur, axis=-1)

#         eye = np.expand_dims(np.eye(x_cur.shape[-2]), axis=0)
#         invJ = eye
#         J = eye

#         get_res = lambda xk: xk + self.euler_step(xk, t_next, t_cur) - x_cur
#         res = get_res(x_next)

#         while True:
#             dx = -np.matmul(invJ, res)
#             x_next = x_next + 0.9 * dx

#             err_sq = np.sum(np.square(dx))
#             if err_sq < 1e-6:
#                 break

#             res_new = get_res(x_next)
#             dR = res_new - res
#             invJ_dR = np.matmul(invJ, dR)
#             dy = (dx - invJ_dR) / (1e-4 + np.sum(dx * invJ_dR, axis=-2, keepdims=True))
#             dy_J = (dR - np.matmul(J, dx)) / (1e-4 + np.sum(np.square(dx), axis=-2, keepdims=True))

#             dx_invJ = np.matmul(np.transpose(dx, (0, -1, -2)), invJ)
#             invJ = invJ + dy * dx_invJ
#             J = J + dy_J * np.transpose(dx, (0, -1, -2))

#             res = res_new
        
#         x_next = np.squeeze(x_next, axis=-1)

#         return x_next
        
#     def _step_impl(self):
#         t = self.t
#         t_next = self.t_steps[self.cur_t_idx + 1]
#         y = self.y

#         drift = self.fun(t, y)
#         y_euler = y + (t_next - t) * drift
#         drift_prime = self.fun(t_next, y_euler)
#         y_heun = y + (t_next - t) * (0.5 * drift + 0.5 * drift_prime)

#         self.y_old = y
#         self.y = self.broyden_method(y_heun, y, t, t_next)
#         self.cur_t_idx += 1
#         self.t = self.t_steps[self.cur_t_idx]

#         return (True, None)
        
#     def _dense_output_impl(self):
#         pass

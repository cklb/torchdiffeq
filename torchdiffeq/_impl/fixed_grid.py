from .solvers import FixedGridODESolver
from . import rk_common


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        return tuple(dt * f_ for f_ in func(t, y, u))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y, u)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid, u))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y, u):
        return rk_common.rk4_alt_step_func(func, t, dt, y, u)

    @property
    def order(self):
        return 4

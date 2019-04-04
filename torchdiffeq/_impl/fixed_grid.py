from .solvers import FixedGridODESolver
from . import rk_common


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y, *args):
        return tuple(dt_ * f_ for dt_, f_ in zip(dt, func(t, y, *args)))

    @property
    def order(self):
        return 1


class Midpoint(FixedGridODESolver):

    def step_func(self, func, t, dt, y, *args):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y, *args)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid, *args))

    @property
    def order(self):
        return 2


class RK4(FixedGridODESolver):

    def step_func(self, func, t, dt, y, *args):
        return rk_common.rk4_alt_step_func(func, t, dt, y, *args)

    @property
    def order(self):
        return 4

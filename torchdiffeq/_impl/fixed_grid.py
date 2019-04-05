from .solvers import FixedGridODESolver
from . import rk_common


class Euler(FixedGridODESolver):

    def step_func(self, func, t, dt, y, *args):
        if not isinstance(dt, tuple):
            dt = dt,
        y_dt = func(t, y, *args)
        res = []
        for dt_, f_ in zip(dt, y_dt):
            # TODO fix this hack
            if dt_.dim() > 0:
                # broadcast dt to
                dim_diff = f_.dim() - dt_.dim()
                dt_ = dt_[[slice(None)] + dim_diff*[None]]
            res.append(dt_ * f_)
        return tuple(res)

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

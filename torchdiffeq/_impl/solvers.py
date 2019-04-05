import abc
import torch
from .misc import _assert_increasing, _handle_unused_kwargs


class AdaptiveStepsizeODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, atol, rtol, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol

    def before_integrate(self, t):
        pass

    @abc.abstractmethod
    def advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        _assert_increasing(t)
        solution = [self.y0]
        t = t.to(self.y0[0].device, torch.float64)
        self.before_integrate(t)
        for i in range(1, len(t)):
            y = self.advance(t[i])
            solution.append(y)
        return tuple(map(torch.stack, tuple(zip(*solution))))


class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, u=None, step_size=None, grid_constructor=None, **unused_kwargs):
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.u = u

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda f, y0, t: t
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, func, t, dt, y, *args):
        pass

    def integrate(self, t):
        _assert_increasing(t)
        t = t.type_as(self.y0[0])
        if t.dim() > 1:
            t_mod = t.t()
        else:
            t_mod = t

        time_grid = self.grid_constructor(self.func, self.y0, t)
        if t.dim() > 1:
            assert all(time_grid[0] == t[0]) and all(time_grid[-1] == t[-1])
        else:
            assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])

        solution = [self.y0]

        j = 1
        y0 = self.y0
        if t.dim() > 1:
            times = zip(time_grid.t()[:-1], time_grid.t()[1:])
        else:
            times = zip(time_grid[:-1], time_grid[1:])
        for idx, (t0, t1) in enumerate(times):
            extra_args = []
            if self.u is not None:
                if self.u[0].dim() > 3:
                    u = tuple(_u[:, idx] for _u in self.u)
                else:
                    u = tuple(_u[idx] for _u in self.u)
                extra_args.append(u)

            dy = self.step_func(self.func, t0, t1 - t0, y0, *extra_args)
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))

            def _in_solution_region(idx, t_values, curr_time):
                if idx >= len(t_values):
                    return False
                if curr_time.dim() == 0:
                    return curr_time >= t_values[idx]
                else:
                    return all(curr_time >= t_values[idx])

            # while j < len(t_mod) and t1 >= t_mod[j]:
            while _in_solution_region(j, t_mod, t1):
                solution.append(self._linear_interp(t0, t1, y0, y1, t_mod[j]))
                j += 1

            y0 = y1

        sol = tuple(map(torch.stack, tuple(zip(*solution))))

        if t.dim() > 1:
            sol[0].transpose_(0, 1)

        return sol

        # sol = torch.stack(tuple(zip(*solution)))
        # return sol

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t0.dim() == 0:
            if t == t0:
                return y0
            if t == t1:
                return y1
            t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
            slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
            return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))
        else:
            # vectorized variant
            if all(t == t0):
                return y0
            if all(t == t1):
                return y1
            t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
            slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
            return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_false')
args = parser.parse_args()

if args.adjoint:
    # from torchdiffeq import odeint_adjoint as odeint
    from torchdiffeq import odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

state_dim = 3
# state_dim = 2
# state_dim = 1
input_dim = 1
net_dim = state_dim + input_dim

true_y0 = torch.tensor([[1., 1., 1.]])
# true_y0 = torch.tensor([[2., 0.]])
# true_y0 = torch.tensor([[2.]])


class RealModel(nn.Module):

    # true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
    # true_B = torch.tensor([[0.], [1]])
    true_A = torch.tensor([[-1.]])
    true_B = torch.tensor([[1.]])

    def __init__(self):
        super().__init__()

    def forward(self, t, y, *args):
        x = y
        u = args[0]

        x_dt = torch.zeros_like(y)
        if 0:
            x_dt = x @ self.true_A.t() + u @ self.true_B.t()
        else:
            sigma = 10
            rho = 28
            beta = 8/3
            x_dt[..., 0] = sigma * (x[..., 1] - x[..., 0])
            x_dt[..., 1] = x[..., 0] * (rho - x[..., 2]) - x[..., 2]
            x_dt[..., 2] = x[..., 0] * x[..., 1] - beta * x[..., 2]
        return x_dt


def create_data():
    t_values = torch.linspace(0., 10, args.data_size)
    # u_values = torch.sin(t_values)[:, None, None]
    # u_values = torch.randn(len(t_values), 1, 1)
    u_values = 0*torch.ones(len(t_values), 1, 1)

    with torch.no_grad():
        y_values = odeint(RealModel(),
                          true_y0,
                          t_values,
                          u=u_values,
                          method='euler')

    return t_values, y_values, u_values


def separate_data(data):
    test_ratio = .25
    val_ratio = .25
    train_ratio = 1 - val_ratio - test_ratio

    test_idx = int(data[0].shape[0] * test_ratio)
    val_idx = int(data[0].shape[0] * val_ratio)
    train_idx = int(data[0].shape[0] * train_ratio)

    train_data = [dat[:train_idx] for dat in data]
    val_data = [dat[train_idx:train_idx+val_idx] for dat in data]
    test_data = [dat[train_idx+val_idx:] for dat in data]

    return train_data, val_data, test_data


def get_batch(t_values, y_values, u_values):
    set_size = 10  # time steps per dataset
    batch_size = 20  # data sets per batch
    set_idxs = np.arange(0, t_values.shape[0], set_size, dtype=np.int64)
    all_selectors = np.random.choice(set_idxs,
                                     batch_size,
                                     replace=False)
    set_selectors = torch.tensor([all_selectors + i for i in range(set_size)]).t()

    batch_y0 = y_values[all_selectors]
    batch_t = t_values[set_selectors]
    batch_y = y_values[set_selectors]
    batch_u = u_values[set_selectors]

    # def _extract_sets(values, selectors):
    #     sets = torch.stack(tuple(values[sel] for sel in selectors), dim=1)  # (T, M, D)
    #     return sets
    #
    # batch_t = _extract_sets(t_values, set_selectors)
    # batch_y = _extract_sets(y_values, set_selectors)
    # batch_u = _extract_sets(u_values, set_selectors)

    return batch_y0, batch_t, batch_y, batch_u


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_phase = fig.add_subplot(221, frameon=False)
    ax_vecfield = fig.add_subplot(222, frameon=False)
    ax_traj = fig.add_subplot(223, frameon=False)
    ax_errors = fig.add_subplot(224, frameon=False)
    plt.show(block=False)


def visualize(data, pred_y, odefunc, itr, errors, ground_truth=None):
    train_t, train_y, train_u = data
    if ground_truth is not None:
        all_t, all_y, all_u = ground_truth
        all_pred = odeint(func, all_y[0], all_t, all_u)

    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    for idx in range(all_y.shape[-1]):
        ax_traj.plot(all_t.numpy(), all_y.numpy()[..., idx], label="ground truth x")
        ax_traj.scatter(train_t.numpy(), train_y.numpy()[..., idx], label="train samples x")
        ax_traj.plot(all_t.numpy(), all_pred.numpy()[..., idx], "--", label="prediction x")

    # ax_traj.plot(train_t.numpy(),
    #              train_y.numpy()[:, 0, 1],
    #              '-g',
    #              label="real y",
    #              )
    # ax_traj.plot(train_t.numpy(),
    #              pred_y.numpy()[:, 0, 0],
    #              '--',
    #              label="pred x")
    # ax_traj.plot(train_t.numpy(),
    #              pred_y.numpy()[:, 0, 1],
    #              'b--',
    #              label="pred y")
    ax_traj.set_xlim(all_t.min(), all_t.max())
    # ax_traj.set_ylim(all_y.min(), all_y.min())
    ax_traj.legend()

    if 0:
        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(all_y.numpy()[..., 0], all_y.numpy()[..., 1], label="ground truth")
        ax_phase.scatter(train_y.numpy()[:, 0, 0], train_y.numpy()[:, 0, 1], label="train samples")
        ax_phase.plot(all_pred.numpy()[..., 0], all_pred.numpy()[..., 1], label="prediction")
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)
        ax_phase.legend()

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        vec_init_times = torch.ones(21**2, 1) * train_t[0]
        vec_init_states = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 1, 2))
        vec_inputs = train_u[21**2*[0]]
        dydt = odefunc(vec_init_times, vec_init_states, vec_inputs).cpu().detach().numpy()
        dydt = dydt.reshape(21, 21, 2)
        mag = np.sqrt(dydt[..., 0]**2 + dydt[..., 1]**2)
        dydt = (dydt.T / mag).T

        ax_vecfield.streamplot(x, y, dydt[..., 0], dydt[..., 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

    ax_errors.cla()
    ax_errors.set_title('Errors')
    ax_errors.set_xlabel('Iteration')
    ax_errors.set_ylabel('RSME / N')
    for lbl, err in errors.items():
        ax_errors.plot(err, label=lbl)
    ax_errors.legend()

    fig.tight_layout()
    # plt.savefig('png/{:03d}'.format(itr))
    plt.draw()
    plt.pause(0.001)


class ODEFunc(nn.Module):

    deep = True
    # deep = False

    def __init__(self):
        super(ODEFunc, self).__init__()

        if self.deep:
            self.net = nn.Sequential(
                # nn.Linear(state_dim+input_dim, state_dim),
                nn.Linear(net_dim, 50),
                nn.ReLU(),
                nn.Linear(50, state_dim),
            )
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)
        else:
            self.a_net = nn.Linear(state_dim, state_dim, bias=False)
            self.b_net = nn.Linear(input_dim, state_dim, bias=False)
            nn.init.normal_(self.a_net.weight, mean=0, std=0.1)
            nn.init.normal_(self.b_net.weight, mean=0, std=0.1)

    def forward(self, t, y, u):
        if self.deep:
            q = torch.cat((y, u), dim=-1)
            # q = torch.cat((y, u.unsqueeze(1)), dim=-1)
            return self.net(q)
        else:
            return self.a_net(y) + self.b_net(u)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def loss_func(output, target, model=None, reg=False):
    loss = torch.tensor(0.)

    # rmse
    err = torch.mean(torch.abs(output - target))
    loss += err

    if reg:
        if model is None:
            raise ValueError("Model parameters are needed for regularization")
        lamda = 0
        # lamda = 1e-3
        reg = torch.tensor(0., requires_grad=False)
        for p in model.parameters():
            reg += torch.norm(p, p=1)  # L1
            # reg += torch.norm(p, p=2)  # L2
        loss += lamda * reg

    return loss


def calc_error(data, func):
    t_data, y_data, u_data = data
    y_pred = odeint(func, y_data[0], t_data, u_data)
    err = loss_func(y_pred, y_data, func, reg=False)
    return err.item()/len(y_pred), y_pred


if __name__ == '__main__':

    # handle data
    data = create_data()
    train_data, val_data, test_data = separate_data(data)

    func = ODEFunc()
    # optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
    optimizer = optim.SGD(func.parameters(), lr=1e-1)
    end = time.time()

    # time_meter = RunningAverageMeter(0.97)
    # loss_meter = RunningAverageMeter(0.97)

    errors = {lbl: [] for lbl in ["train", "validation", "test"]}

    ii = 0
    for itr in range(1, args.niters + 1):
        def _closure():
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y, batch_u = get_batch(*train_data)
            pred_y = odeint(func, batch_y0, batch_t, batch_u)
            loss = loss_func(batch_y, pred_y, func, reg=True)
            loss.backward()
        optimizer.step(_closure)

        # time_meter.update(time.time() - end)
        # loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                # calc errors
                train_err, train_pred = calc_error(train_data, func)
                val_err, val_pred = calc_error(val_data, func)
                test_err, test_pred = calc_error(test_data, func)
                errors["train"].append(train_err)
                errors["validation"].append(val_err)
                errors["test"].append(test_err)

                if args.viz:
                    visualize(train_data, train_pred, func, ii, errors,
                              ground_truth=data)
                ii += 1

        end = time.time()

    test_err = calc_error(test_data, func)

    print("Real Values:")
    print("A:", RealModel.true_A)
    print("B:", RealModel.true_B)

    print("Learned Values:")
    for name, param in func.named_parameters():
        print(name, param)

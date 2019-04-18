import logging
from datetime import datetime
import os
import argparse
import numpy as np

# for reproducible results
np.random.seed(20190417)

import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt


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


def distort_data(data):
    t_data, y_data, u_data = data
    y_arr = y_data.numpy()
    y_arr += np.random.normal(size=y_data.shape)
    return t_data, y_data, u_data


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


class ODEFunc(nn.Module):

    deep = True
    # deep = False

    def __init__(self, layers):
        super(ODEFunc, self).__init__()
        self.all_layers = [net_input_dim] + layers + [net_outpout_dim]

        if self.deep:
            # build structure
            stages = []
            for d1, d2 in zip(self.all_layers[:-1], self.all_layers[1:]):
                layer = nn.Linear(d1, d2)
                stages.append(layer)
                if args.activation == "relu":
                    act = nn.ReLU()
                else:
                    raise NotImplementedError(args.activation)
                stages.append(act)

            self.net = nn.Sequential(*stages)

            # init weights
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


def train_model(train_data, val_data, model, optimizer, checkpoint=None, eval_cb=None):
    def _train_closure():
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y, batch_u = get_batch(*train_data)
        pred_y = odeint(model, batch_y0, batch_t, batch_u)
        loss = loss_func(batch_y, pred_y, model, reg=True)
        loss.backward()

    # try to load checkpoint
    state = load_checkpoint(checkpoint, model.all_layers)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        errors = state["errors"]
        start_epoch = state["epoch"]
    else:
        logging.info("Training new model")
        errors = {lbl: [] for lbl in ["train", "validation", "test"]}
        start_epoch = 0
    try:
        for itr in range(start_epoch, args.niters):
            # run training step
            optimizer.step(_train_closure)

            if itr % args.test_freq == 0:
                # calc errors
                train_err, train_pred = calc_error(train_data, model)
                val_err, val_pred = calc_error(val_data, model)
                errors["train"].append(train_err)
                errors["validation"].append(val_err)

                logging.info("Train Error: {}".format(train_err))
                logging.info("Val Error: {}".format(val_err))

                if eval_cb is not None:
                    eval_cb(itr, errors)
        logging.info("Finished training over {} epochs".format(itr))

    except BaseException as e:
        logging.error("An exception occurred, training aborted.")
        logging.exception(e)

    finally:
        state = {"epoch": itr,
                 "errors": errors,
                 "model": model.state_dict(),
                 "optimizer": optimizer.state_dict()
                 }
        create_checkpoint(state, model.all_layers)

    return errors


class Visualiser:

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 4), facecolor='white')
        self.ax_phase = self.fig.add_subplot(221, frameon=False)
        self.ax_vecfield = self.fig.add_subplot(222, frameon=False)
        self.ax_traj = self.fig.add_subplot(223, frameon=False)
        self.ax_errors = self.fig.add_subplot(224, frameon=False)
        plt.show(block=False)
        # makedirs('png')

    def visualize(self, raw_data, train_data, val_data, test_data, model, itr, errors):
        all_t, all_y, all_u = raw_data
        train_t, train_y, train_u = train_data
        val_t, val_y, val_u = train_data

        with torch.no_grad():
            all_err, all_pred = calc_error(raw_data, model)
        errors["test"].append(all_err)

        self.ax_traj.cla()
        self.ax_traj.set_title('Trajectories for Epoch {}'.format(itr))
        self.ax_traj.set_xlabel('t')
        self.ax_traj.set_ylabel('x,y')
        for idx in range(all_y.shape[-1]):
            var = "x{}".format(idx)
            self.ax_traj.plot(all_t.numpy(),
                              all_y.numpy()[..., idx],
                              "--",
                              label="ground truth {}".format(var))
            # plt.gca().set_prop_cycle(None)
            self.ax_traj.scatter(train_t.numpy(),
                                 train_y.numpy()[..., idx],
                                 label="train samples {}".format(var))
            self.ax_traj.scatter(val_t.numpy(),
                                 val_y.numpy()[..., idx],
                                 marker="+",
                                 label="validation samples {}".format(var))
            # plt.gca().set_prop_cycle(None)
            self.ax_traj.plot(all_t.numpy(),
                              all_pred.numpy()[..., idx],
                              label="prediction {}".format(var))

        # self.ax_traj.plot(train_t.numpy(),
        #              train_y.numpy()[:, 0, 1],
        #              '-g',
        #              label="real y",
        #              )
        # self.ax_traj.plot(train_t.numpy(),
        #              pred_y.numpy()[:, 0, 0],
        #              '--',
        #              label="pred x")
        # ax_traj.plot(train_t.numpy(),
        #              pred_y.numpy()[:, 0, 1],
        #              'b--',
        #              label="pred y")
        self.ax_traj.set_xlim(all_t.min(), all_t.max())
        # self.ax_traj.set_ylim(all_y.min(), all_y.min())
        self.ax_traj.legend()

        if 0:
            self.ax_phase.cla()
            self.ax_phase.set_title('Phase Portrait')
            self.ax_phase.set_xlabel('x')
            self.ax_phase.set_ylabel('y')
            self.ax_phase.plot(all_y.numpy()[..., 0], all_y.numpy()[..., 1], label="ground truth")
            self.ax_phase.scatter(train_y.numpy()[:, 0, 0], train_y.numpy()[:, 0, 1], label="train samples")
            self.ax_phase.plot(all_pred.numpy()[..., 0], all_pred.numpy()[..., 1], label="prediction")
            self.ax_phase.set_xlim(-2, 2)
            self.ax_phase.set_ylim(-2, 2)
            self.ax_phase.legend()

            self.ax_vecfield.cla()
            self.ax_vecfield.set_title('Learned Vector Field')
            self.ax_vecfield.set_xlabel('x')
            self.ax_vecfield.set_ylabel('y')

            y, x = np.mgrid[-2:2:21j, -2:2:21j]
            vec_init_times = torch.ones(21**2, 1) * train_t[0]
            vec_init_states = torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 1, 2))
            vec_inputs = train_u[21**2*[0]]
            dydt = odefunc(vec_init_times, vec_init_states, vec_inputs).cpu().detach().numpy()
            dydt = dydt.reshape(21, 21, 2)
            mag = np.sqrt(dydt[..., 0]**2 + dydt[..., 1]**2)
            dydt = (dydt.T / mag).T

            self.ax_vecfield.streamplot(x, y, dydt[..., 0], dydt[..., 1], color="black")
            self.ax_vecfield.set_xlim(-2, 2)
            self.ax_vecfield.set_ylim(-2, 2)

        self.ax_errors.cla()
        self.ax_errors.set_title('Errors')
        self.ax_errors.set_xlabel('Iteration')
        self.ax_errors.set_ylabel('RSME / N')
        for lbl, err in errors.items():
            self.ax_errors.plot(err, label=lbl)
        self.ax_errors.legend()

        self.fig.tight_layout()
        # plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


def main():
    # handle data
    raw_data = create_data()
    train_data, val_data, test_data = separate_data(raw_data)

    # add noise
    train_data = distort_data(train_data)

    # build model and optimizer
    model = ODEFunc(args.structure)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    # optimizer = optim.SGD(func.parameters(), lr=1e-1)

    if args.viz:
        vis = Visualiser()

        def _eval_callback(itr_idx, errors):
            vis.visualize(raw_data, train_data, val_data, test_data,
                          model,
                          itr_idx,
                          errors
                          )
    else:
        _eval_callback = None

    # run training
    errors = train_model(train_data, val_data, model, optimizer,
                         checkpoint="latest",
                         eval_cb=_eval_callback)

    if 0:
        print("Real Values:")
        print("A:", RealModel.true_A)
        print("B:", RealModel.true_B)
        print("Learned Values:")
        for name, param in model.named_parameters():
            print(name, param)


def load_checkpoint(name, structure):
    if name is None:
        return {}

    path = args.checkpoint_path
    try:
        files = os.listdir(path)
    except FileNotFoundError:
        os.makedirs(path)
        return

    if name is "latest":
        chkpt_files = []
        for f in files:
            if "checkpoint" not in f:
                continue
            if str(structure) not in f:
                continue
            chkpt_files.append(f)
        if not chkpt_files:
            logging.warning("No appropriate checkpoint found")
            return {}

        # get most up to date checkpoint
        dates = [datetime.strptime(f.split("_")[0], checkpoint_mark)
                 for f in chkpt_files]
        sorted_chkpts = sorted(zip(dates, chkpt_files), key=lambda x: x[0])
        chkpt_file = sorted_chkpts[-1][1]
    else:
        chkpt_file = name

    file = os.sep.join([path, chkpt_file])
    state = torch.load(file)
    logging.info("Loaded checkpoint file '{}'".format(file))
    return state


def create_checkpoint(state, structure):
    path = args.checkpoint_path
    date = datetime.now().strftime(checkpoint_mark)
    fname = "{}_{}_checkpoint.torch".format(date, structure)
    file = os.path.sep.join([path, fname])
    torch.save(state, file)
    logging.info("Checkpoint '{}' created".format(file))


if __name__ == '__main__':
    checkpoint_mark = "%Y-%m-%d %H:%M:%S"
    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--method', type=str, choices=['dopri5', 'adams'],
                        default='dopri5')
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--batch_time', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--niters', type=int, default=10000)
    parser.add_argument('--test_freq', type=int, default=10)
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--adjoint', action='store_false')
    parser.add_argument('--checkpoint_path', type=str, default=".checkpoints")
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--structure', type=list, default=[10, 10])
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
    net_input_dim = state_dim + input_dim
    net_outpout_dim = state_dim

    true_y0 = torch.tensor([[1., 1., 1.]])
    # true_y0 = torch.tensor([[2., 0.]])
    # true_y0 = torch.tensor([[2.]])

    # init logging
    logging.basicConfig(filename="training_{}.log".format(args.structure),
                        format='%(asctime)s %(message)s',
                        datefmt=checkpoint_mark,
                        level=logging.INFO)

    main()

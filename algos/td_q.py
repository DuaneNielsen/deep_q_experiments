import torch
from algos.schedules import *
from tensorboardX import SummaryWriter
import random
from monitoring import SingleLogger
from data import OneObsToState
import matplotlib.pyplot as plt
from util import Timer

timer = Timer()


class Q:
    def __init__(self, env, critic, behaviour_policy, greedy_policy, exp_buffer, join=OneObsToState(), device='cuda', viewer=None, tb=None, plot=None):
        self.env = env
        self.state = self.env.reset()
        self.join = join
        self.exp_buffer = exp_buffer
        self.v = viewer
        self.state = env.reset()
        self.critic = critic
        self.behavior_policy = behaviour_policy
        self.greedy_policy = greedy_policy
        self.device = device
        self.tb = tb
        self.plot = plot
        self.epsilon = None

    def run(self, run_id, steps, discount_factor, batch_size,
            lr_sched=ConstantSched(0.05), eps_sched=None, logging_freq=10):

        self.state = self.env.reset()

        # learning rates
        lr = lr_sched.get(0)
        optim = torch.optim.SGD(self.critic.parameters(), lr=lr)

        # monitoring
        t = SummaryWriter(f'runs/{run_id}_{random.randint(0, 10000)}')
        e_log = SingleLogger(self.exp_buffer, logging_freq, t, self.critic, self.join)

        for step in range(steps):

            self.epsilon = eps_sched.get(step)

            render = e_log.render_episode(step)

            state, reward, done, reset = self.step(render=render)

            lr = lr_sched.get(step)
            for param_group in optim.param_groups:
                param_group['lr'] = lr

            e_log.log_progress(state, reward, done, reset, step, self.epsilon, lr)

            self.train(discount_factor, batch_size, optim, step)


            critic = self.train(discount_factor, batch_size, optim, step)

    def step(self, render=False, display_observation=False):

        action_dist = self.behavior_policy(self.join(self.state), epsilon=self.epsilon)

        action = action_dist.sample()

        next_state, reward, done, reset, info = self.env.step(action)

        self.exp_buffer.add(self.state, action, reward, done, reset, next_state)

        self.state = next_state

        if render:
            self.env.render(mode='human')
        if display_observation:
            self.v.render(self.state)

        return next_state, reward, done, reset

    def train(self, discount_factor, batch_size, optim, step):

        self.exp_buffer.batch_size = batch_size

        for state, action, reward, done, next_state, index, i_w in self.exp_buffer:

            with torch.no_grad():
                # zero the bootstrapped value of terminal states
                zero_if_terminal = (~done).to(next_state.dtype)

                # extract and convert obs to states
                state = self.join(state)
                next_state = self.join(next_state)

                # softmax and lack of logprob will affect the calculation here!
                next_action = self.greedy_policy(self.join(next_state)).sample().to(self.device)

                next_value = self.critic(next_state, next_action)
                bootstrapped = reward + zero_if_terminal * discount_factor * next_value

            optim.zero_grad()
            q_value = self.critic(state, action)
            td_error = bootstrapped - q_value
            loss = torch.mean((td_error * i_w) ** 2)
            loss.backward()

            if self.tb is not None:
                total_norm = 0
                for p in self.critic.parameters():
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.tb.add_scalar('grad_norm', total_norm, step)
                if hasattr(self.critic, 'gain'):
                    self.tb.add_scalar('gain', self.critic.gain.data, step)
                self.tb.add_scalar('value_mean', q_value.mean(), step)
                self.tb.add_scalar('value_std', q_value.std(), step)

            if self.plot is not None:
                self.plot.update(self.critic, state, action, bootstrapped)

            optim.step()
            self.exp_buffer.update_td_error(index, td_error)

            break

        return self.critic


class FastPlot:
    def __init__(self, actions, device='cuda', resolution=11, fig_size=(10, 16)):
        self.fig = plt.figure(1)
        self.fig.set_figheight(fig_size[0])
        self.fig.set_figwidth(fig_size[1])
        self.ax = self.fig.add_subplot()
        self.ax.set_ylim(0.0, 2.5)
        self.fig.show()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.lines = []
        self.scatter = []
        x = np.linspace(0, 1.0, resolution)
        y = np.zeros_like(x)
        x_s, y_s = [], []
        for a in range(actions):
            self.lines.append(self.ax.plot(x, y, alpha=0.6, label=str(a)))
            self.scatter.append(self.ax.scatter(x_s, y_s, alpha=0.6, label=str(a)))
        self.actions = actions
        self.device = device
        self.resolution = resolution

    def update(self, critic, state, act, td_est):
        with torch.no_grad():
            x = torch.linspace(0.0, 1.0, self.resolution, device=self.device, requires_grad=False).unsqueeze(1)
            for a in range(self.actions):
                self.fig.canvas.restore_region(self.background)
                v = critic(x, torch.ones(self.resolution, dtype=torch.long, device=self.device, requires_grad=False) * a)
                self.lines[a][0].set_ydata(v.cpu().numpy())
                action_mask = a == act
                self.scatter[a].set_offsets(np.c_[state[action_mask].cpu().numpy(), td_est[action_mask].cpu().numpy()])
                self.ax.draw_artist(self.lines[a][0])
            self.fig.canvas.blit(self.ax.bbox)




class PlotQ:
    def __init__(self, device='cuda'):
        self.device = device
        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot()
        plt.ion()

        # Let's capture the background of the figure
        self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.ax]

    def plot(self, critic, actions, state, act, td_est):
        with torch.no_grad():
            x = torch.linspace(0.0, 1.0, 10, device=self.device, requires_grad=False)

            self.ax.cla()
            self.ax.set_ylim(0.0, 5.0)
            for action in range(actions):
                v = critic(x, torch.ones(10, dtype=torch.long, device=self.device, requires_grad=False) * action)
                self.ax.plot(x.cpu().numpy(), v.cpu().numpy(), alpha=0.6, label=str(action))
                action_mask = action == act
                self.ax.scatter(state[action_mask].cpu().numpy(), td_est[action_mask].cpu().numpy(), label=str(action))


            self.fig.legend()

            timer.start('draw')
            self.fig.canvas.restore_region(self.backgrounds)
            timer.elapsed('draw')

            plt.pause(0.001)


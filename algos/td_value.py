import torch

from models import VPolicy, EpsilonGreedyProperDiscreteDist, GreedyDist
import matplotlib.pyplot as plt
from data import OneObsToState
from tensorboardX import SummaryWriter
from monitoring import SingleLogger
from algos.schedules import *
from random import randint

# matplotlib
plt.ion()
fig = plt.figure(figsize=(4, 10))
plt.ylim(-1.2, 1.2)


# axes = plt.gca()
# axes.set_xlim([xmin, xmax])
# axes.set_ylim([-1.2, 1.2])

def plot_values(critic, state, next, target, fig, device):
    with torch.no_grad():
        x = torch.linspace(0.0, 1.0, 100, device=device, requires_grad=False)
        plt.clf()
        plt.ylim(-1.2, 1.2)
        v = critic(x)
        plt.plot(x.cpu().numpy(), v.cpu().numpy(), alpha=0.6, label='value')
        plt.scatter(state.cpu().numpy(), target.cpu().numpy())
        # for i, s in enumerate(state):
        #     plt.plot(state[i].cpu().numpy(), next[i].cpu().numpy())

        plt.legend()
        fig.canvas.draw_idle()
        plt.pause(0.025)


class TDValue:
    def __init__(self, env, critic, behaviour_policy, greedy_policy, exp_buffer, join=OneObsToState(), device='cuda', viewer=None, plot=None):
        self.env = env
        self.state = self.env.reset()
        self.join = join
        self.exp_buffer = exp_buffer
        self.v = viewer
        self.tb = None
        self.critic = critic
        self.behaviour_policy = behaviour_policy
        self.greedy_policy = greedy_policy
        self.device = device
        self.plotter = plot

    def run(self, run_id, steps, discount_factor, batch_size,
            lr_sched=ConstantSched(0.05), eps_sched=None, logging_freq=10):

        self.state = self.env.reset()

        # learning rates
        lr = lr_sched.get(0)
        optim = torch.optim.SGD(self.critic.parameters(), lr=lr)

        # monitoring
        self.tb = SummaryWriter(f'runs/{run_id}_{randint(0, 10000)}')
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

        with torch.no_grad():

            lookahead_states, lookahead_reward, lookahead_done, info = self.env.lookahead()

            behaviour_action = self.behaviour_policy(self.join(lookahead_states), lookahead_reward, lookahead_done).sample()
            greedy_action = self.greedy_policy(self.join(lookahead_states), lookahead_reward, lookahead_done).sample()

            next_state, reward, done, reset, info = self.env.step(behaviour_action)

            greedy_state = lookahead_states[0, greedy_action]
            greedy_done = lookahead_done[0, greedy_action]

            self.exp_buffer.add(self.state, behaviour_action, reward, greedy_done, reset, greedy_state)
            self.state = next_state

            if render:
                self.env.render(mode='human')
            if display_observation:
                self.v.render(self.state)

        return next_state, reward, done, reset

    def train(self, discount_factor, batch_size, optim, step):

        self.exp_buffer.batch_size = batch_size

        for state, action, reward, done, next_state, index, i_w in self.exp_buffer:

            # zero the bootstrapped value of terminal states
            zero_if_terminal = (~done).to(next_state.dtype)

            # extract and convert obs to states
            state = self.join(state)

            with torch.no_grad():
                next_value = self.critic(next_state)
                target = reward + discount_factor * next_value * zero_if_terminal
                target.detach()

            optim.zero_grad()
            value = self.critic(state)
            td_error = (target - value)
            loss = torch.mean((td_error * i_w) ** 2)
            loss.backward()

            total_norm = 0
            for p in self.critic.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #print(critic.ln.weight.grad.data)
            # tb.add_scalar('gain', critic.gain.data, step)
            self.tb.add_scalar('td_error_mag', torch.abs(td_error).mean().cpu().item(), step)
            self.tb.add_scalar('td_error', td_error.mean().cpu().item(), step)
            self.tb.add_scalar('grad_norm', total_norm, step)
            self.tb.add_scalar('value_mean', value.mean(), step)
            self.tb.add_scalar('value_std', value.std(), step)
            self.tb.add_scalar('target_mean', target.mean(), step)
            self.tb.add_scalar('target_std', target.std(), step)
            #plot_values(critic, state, next_state, target, fig, device)

            optim.step()

            self.exp_buffer.update_td_error(index, td_error)

            break

        return self.critic
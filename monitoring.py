import torch
from colorama import Fore, Style
from matplotlib import pyplot as plt

from util import RunningReward
import logging

logger = logging.getLogger(__name__)


class SingleLogger:
    def __init__(self, exp_buffer, logging_freq, tensorboard, critic=None, join=None):
        self.reward_counter = 0
        self.step_counter = 0
        self.prev_reward = 0
        self.prev_step = 0
        self.exp_buffer = exp_buffer
        self.logging_freq = logging_freq
        self.t = tensorboard
        self.episode_counter = 0
        self.done = False
        self.critic = critic
        self.join = join

    def log_progress(self, state, reward, done, reset, step, epsilon, lr):
        self.done = done.item() == 1
        self.t.add_scalar('lr', lr, step)
        self.t.add_scalar('epsilon', epsilon, step)
        self.t.add_scalar('reward', reward.item(), step)
        self.t.add_scalar('episode_length', self.prev_step, step)
        self.reward_counter += reward.item()
        self.step_counter += 1
        if self.done:
            trajectory = self.exp_buffer.get_trajectory()
            logger.info(
                f'{Fore.GREEN}reward {self.reward_counter} {Style.RESET_ALL} steps {self.step_counter} epsilon {epsilon} trajectory {trajectory}')
            self.reward_counter = 0
            self.step_counter = 0
            self.episode_counter += 1
        # if self.critic is not None:
            # value = self.critic(self.join(state))
            # self.t.add_scalar('value', value, step)

    def render_episode(self, step):
        return self.episode_counter % self.logging_freq == 0


class VectoredLogger:
    def __init__(self, ll_runs, logging_freq, tensorboard, device):
        self.rw = RunningReward(ll_runs)
        self.rec_reward = RunningReward(ll_runs)
        self.t = tensorboard
        self.logging_freq = logging_freq
        self.done = torch.zeros(1, dtype=torch.uint8, device=device)

    def log_progress(self, state, reward, done, reset, step, epsilon, lr):
        self.t.add_scalar('epsilon', epsilon, step)
        self.t.add_scalar('lr', lr, step)
        self.rw.add(reward, done, reset)
        self.rec_reward.add(reward, done, reset)
        self.t.add_scalar('reward', self.rec_reward.reward, step)
        self.t.add_scalar('episode_length', self.rec_reward.episode_length, step)
        self.rec_reward.reset()
        if step % self.logging_freq == 0:
            self.rw.log()
            self.rw.reset()

    def render_episode(self, step):
        return step % self.logging_freq == 0





def print_qvalues(weights):
    ms = torch.argmax(weights, dim=0)
    arrows = [u'\N{BLACK LEFT-POINTING TRIANGLE}', u'\N{BLACK RIGHT-POINTING TRIANGLE}',
              u'\N{BLACK UP-POINTING TRIANGLE}', u'\N{BLACK DOWN-POINTING TRIANGLE}']

    for i in range(ms.size(0)):
        s = ''
        for j in range(ms.size(1)):
            s = s + arrows[ms[i, j].item()]
        logger.info(s)


def print_values(weigths):
    for i in range(weigths.size(0)):
        for j in range(weigths.size(1)):
            logger.info(weigths[i, j])
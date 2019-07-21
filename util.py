import logging
import time
from collections import deque
from statistics import mean

import numpy as np
import torch
from colorama import Fore, Style

reward_logger = logging.getLogger('reward')
time_logger = logging.getLogger('timing_log')


class Timer:
    def __init__(self):
        self.timers = {}

    def start(self, id):
        self.timers[id] = time.time()

    def elapsed(self, id):
        elapsed = time.time() - self.timers[id]
        time_logger.info(f"{id} : {elapsed}")


class RewardAccumulator:
    def __init__(self, n, device='cpu'):
        self.accum = torch.zeros(n, requires_grad=False, device=device)
        self.reward_total = []

    def add(self, reward, done):
        with torch.no_grad():
            self.accum += reward
            self.reward_total.append(self.accum[done])
            self.accum[done] = 0.0

    def ave_reward(self):
        with torch.no_grad():
            r = torch.cat(self.reward_total)
            return torch.mean(r).item(), r.size(0)


class RewardAccumulatorNumpy:
    def __init__(self, n):
        self.n = n
        self.accum = np.zeros(n)
        self.episode_reward = np.array([])

    def add(self, reward, done):
        self.accum += reward
        d = done.astype(np.bool)
        self.episode_reward = np.append(self.episode_reward, self.accum[d], axis=0)
        self.accum[d] = 0.0

    def reset(self):
        self.accum = np.zeros(self.n)
        self.episode_reward = np.array([])


class RunningReward:
    def __init__(self, n):
        self.n = n
        self.accum = np.zeros(n)
        self.steps = np.zeros(n)
        self.recent = deque([])
        self.epi_len = deque([])

    def add(self, reward, done, reset):

        reward = reward.cpu().numpy()
        d = done.cpu().numpy().astype(np.bool)
        reset = reset.cpu().numpy().astype(np.bool)

        self.accum += reward * ~reset
        self.steps += 1 * ~reset
        self.recent.extend(self.accum[d].tolist())
        self.epi_len.extend(self.steps[d].tolist())
        self.accum[d] = 0.0
        self.steps[d] = 0

    def reset(self):
        self.recent.clear()
        self.epi_len.clear()

    @property
    def episode_length(self):
        if len(self.epi_len) > 0:
            return mean(self.epi_len)
        else:
            return 0

    @property
    def reward(self):
        if len(self.recent) > 0:
            return mean(self.recent)
        else:
            return 0

    def log(self):
        if len(self.recent) > 0:
            reward_logger.info(
                f'{Fore.GREEN} reward {self.reward} {Style.RESET_ALL} epi {len(self.recent)} mean_epi_len {self.episode_length}')
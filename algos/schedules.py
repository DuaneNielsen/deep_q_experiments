import math
import numpy as np


class EpsSchedule:
    def __init__(self, warmup_len=10000, finish=50000):
        kickoff = np.linspace(1.0, 0.1, warmup_len)
        finish = np.linspace(0.1, 0.01, finish)
        self.schedule = np.concatenate((kickoff, finish), axis=0)

    def epsilon_schedule(self, step):
        if step < len(self.schedule):
            return self.schedule[step]
        else:
            return 0.01


class ConstantSched:
    def __init__(self, eps):
        self.eps = eps

    def get(self, step):
        return self.eps


class ExponentialDecay:
    def __init__(self, half_life, scale, bias, steps):
        self.x = np.linspace(0.0, steps, steps)
        lmda = half_life / math.log(2)
        self.x = scale * np.exp(-self.x / lmda) + bias

    def get(self, step):
        return self.x[step]


class Cos:
    def __init__(self, frequency, scale, bias, steps):
        self.x = np.linspace(0.0, steps, steps)
        self.x = 0.5 * np.cos(self.x / frequency * np.pi) + 0.5
        self.x = self.x * scale + bias

    def get(self, step):
        return self.x[step]
import gym
from gym.wrappers import TimeLimit
from gym_duane.wrappers import StackedObs, BatchTensor, Reset, NormalizeFunctional
import gym_duane
import torch
from random import randint
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt


def gen_norms(env):
    env = Reset(env)
    env = BatchTensor(env, device='cuda')
    buf = []
    obs = env.reset()
    r = []

    for _ in range(100000):
        action = torch.tensor([randint(0, env.action_space.n - 1)]).unsqueeze(0)
        obs, reward, done, reset, info = env.step(action)
        buf.append(obs)
        r.append(reward.item())

    b = torch.stack(buf)
    print(b.squeeze().mean(0))
    print(b.squeeze().std(0))

    print(mean(r))
    print(stdev(r))


def test_gen_norms_cartpole():
    env = gym.make('CartPole-v0')
    gen_norms(env)


def test_gen_norms_f5():
    env = gym.make('F5-v0')
    env = TimeLimit(env, max_episode_steps=20)
    gen_norms(env)



def test_generate_norms_lunarlander():

    env = gym.make('LunarLander-v2')
    env = Reset(env)
    env = BatchTensor(env, device='cuda')
    buf = []
    obs = env.reset()
    total_reward = 0.0

    for _ in range(10000):
        action = torch.tensor([randint(0, env.action_space.n - 1)]).unsqueeze(0)
        obs, reward, done, reset, info = env.step(action)
        buf.append(obs)
        total_reward += reward.item()

    b = torch.stack(buf)
    b = b.squeeze()
    print(b.mean(0))
    print(b.std(0))
    print(b.min(0))
    print(b.max(0))
    for i in range(8):
        plot_values(b[:,i].cpu().numpy())


def normalize(obs):
    mean = torch. tensor([-0.0115,  0.9516, -0.0119, -0.6656, -0.0055, -0.0090,  0.0275,  0.0246], device='cuda:0')
    std = torch.tensor([0.3110, 0.4772, 0.6319, 0.4803, 0.5030, 0.5246, 0.1635, 0.1550], device='cuda:0')
    obs = (obs - mean) / std
    return obs


def normalize_min_max(obs):
    min = np.array([-1.0089, -0.4125, -2.0318, -1.9638, -3.6345, -6.7558,  0.0000,  0.0000])
    max = np.array([1.0165, 1.6740, 2.1364, 0.5349, 3.5584, 5.9297, 1.0000, 1.0000])
    return (obs - min) / (max - min)


def plot_values(x):
    plt.hist(x, density=True, bins=30)
    plt.show()


def test_plot_values():
    plot_values(0)


def normalize_lunar_lander(obs):
    mean = np.array([-0.0115, 0.9516, -0.0119, -0.6656, -0.0055, -0.0090, 0.0275, 0.0246])
    std = np.array([0.3110, 0.4772, 0.6319, 0.4803, 0.5030, 0.5246, 0.1635, 0.1550])
    return (obs - mean) / std


def normalize_lunar_lander_reward(reward):
    return (reward + 300.0) / (200 + 300)

def test_normalize_lunarlander():
    env = gym.make('LunarLander-v2')
    env = NormalizeFunctional(env, obs_f=normalize_min_max, reward_f=normalize_lunar_lander_reward)
    env = Reset(env)
    env = BatchTensor(env, device='cuda')
    buf = []
    obs = env.reset()
    total_reward = 0.0
    rew = []

    for _ in range(4):
        for _ in range(10000):
            action = torch.tensor([randint(0, env.action_space.n - 1)]).unsqueeze(0)
            obs, reward, done, reset, info = env.step(action)
            buf.append(obs)
            total_reward += reward.item()
            rew.append(reward)

        b = torch.stack(buf).squeeze()
        r = torch.stack(rew).squeeze()
        #print(b.mean(0))
        #print(b.squeeze().std(0))
        plot_values(r.cpu().numpy())
        for i in range(8):
            plot_values(b[:, i].cpu().numpy())


def test_print_normalize():
    mean = torch.tensor([-1.0435e-02, 9.5823e-01, -6.0627e-03, -6.6439e-01, 1.6793e-03, -2.4789e-04, 2.4840e-02, 2.4490e-02], device='cuda:0')
    std = torch.tensor([0.3054, 0.4723, 0.6153, 0.4805, 0.4986, 0.5198, 0.1556, 0.1546], device='cuda:0')

from train import *
import torch
import logging
import gym
import gym_duane

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.DEBUG)

#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


lawn = """
[
[S, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, T(1.0)]
]
"""


frozen_lake = """
[
[S, E, E, E, E],
[E, E, E, E, E],
[E, E, T, E, T],
[E, E, E, E, E],
[E, E, T, E, T(1.0)]
]
"""

cliff_walk = """
[
[E, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, E],
[S, E, E, E, E],
[T, T, T, T, T(1.0)]
]
"""


def test_frozenlake_value_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s,
                     exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                     steps=steps, logging_freq=100, run_id=f'frozenlake_value{i}', warmup=10, lr=0.05)


def test_cliffwalk_value_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=cliff_walk, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s, exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                     steps=steps, logging_freq=100, run_id=f'cliffwalk_value{i}', warmup=10)


def test_frozenlake_q_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_q_{i}', warmup=10)


def test_cliffwalk_q_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=cliff_walk, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'cliffwalk_q_{i}', warmup=10)


def test_frozenlake_value_prioritized():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=False)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s,
                     exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                     steps=steps, logging_freq=100, run_id=f'frozenlake_value_prioritized_{i}', warmup=10, lr=0.05)


def test_frozenlake_value_importance_sampled():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=True)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s,
                     exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                     steps=steps, logging_freq=100, run_id=f'frozenlake_value_imps_{i}', warmup=10, lr=0.05)


def test_frozenlake_q_prioritized():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=False)

        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_q_prioritized_{i}', warmup=10)


def test_frozenlake_q_importance_sampled():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=True)

        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_q_imps{i}', warmup=10)


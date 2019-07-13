import gym
import numpy as np
from statistics import mean
import logging
import gym_duane

from data import *
from models import *
from colorama import Style, Fore
from collections import deque

from train import batch_episode, one_step, train, train_one, train_one_curiosity
from util import Timer, RunningReward
from tensorboardX import SummaryWriter
import random
import torch
import math

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.DEBUG)
timer = Timer()


def test_reset():
    env = gym.make('SimpleGrid-v2', n=1, device="cpu", map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)

    device = 'cpu'

    episode = []

    action = torch.tensor([0])

    state = env.reset()
    next_state, reward, done, info = env.step(action)

    episode.append(BatchStep(state.numpy(), action.numpy(), reward.numpy(), done.numpy(), next_state.numpy()))

    state = next_state
    next_state, reward, done, info = env.step(action)

    episode.append(BatchStep(state.numpy(), action.numpy(), reward.numpy(), done.numpy(), next_state.numpy()))

    dataset = SARSGridDataset(episode=episode)

    state, action, reward, done, reset, next_state = dataset[1]

    print(state)
    print(next_state)

    assert reset == 1


def test_bandit():
    device = 'cuda'
    ll_runs = 1000
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)

    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), 2).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, 2, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    ave_reward = 0

    while ave_reward < 0.95:
        print(f'paramters: {critic.weights.data}')
        episode, entropy, ave_reward, episodes = batch_episode(env, policy, device, max_rollout_len=50, render=False)
        logger.info(f'{Fore.GREEN}ave_reward {ave_reward} episodes {episodes} {Style.RESET_ALL}')
        policy, critic = train(episode, critic, device, optim, actions=2, epsilon=0.05, logging_freq=0)


def test_bandit_deepq():
    device = 'cuda'
    ll_runs = 1000
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)

    critic = DiscreteQTable((env.height, env.width), 2).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, 2, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    rw = RunningReward(ll_runs, 2000)
    ave_reward = 0
    exp_buffer = []

    state = env.reset()

    while ave_reward < 0.95:

        state, reward, done = one_step(env, state, policy, exp_buffer)
        rw.add(reward.cpu().numpy(), done.cpu().numpy())
        logger.info(f'reward {mean(rw.recent)} epi {len(rw.recent)}')
        policy, critic = train_one(exp_buffer, critic, device, optim, actions=2, epsilon=0.05, logging_freq=0)
        if len(exp_buffer) > 100:
            exp_buffer.pop(0)

        print_qvalues(critic.weights.data)


def test_shortwalk_deepq():
    device = 'cuda'
    ll_runs = 8000
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """)

    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.1).to(device)
    rw = RunningReward(ll_runs)
    ave_reward = 0
    exp_buffer = []

    state = env.reset()

    while ave_reward < 0.95:

        state, reward, done = one_step(env, state, policy, exp_buffer, render=True)
        rw.add(reward.cpu().numpy(), done.cpu().numpy())
        if len(rw.recent) > 0:
            logger.info(f'reward {mean(rw.recent)} epi {len(rw.recent)}')
        else:
            logger.info(f'reward 0 epi {len(rw.recent)}')
        rw.reset()

        policy, critic = train_one(exp_buffer, critic, device, optim, actions=actions, epsilon=0.1, logging_freq=0)
        if len(exp_buffer) > 100:
            exp_buffer.pop(0)

        print_qvalues(critic.weights.data)


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


class ExpExponentialDecay:
    def __init__(self, half_life, minimum, steps):
        self.x = np.linspace(0.0, steps, steps)
        lmda = half_life / math.log(2)
        self.x = (1.0 - minimum) * np.exp(-self.x / lmda) + minimum

    def epsilon_schedule(self, step):
        return self.x[step]


def run_deep_q_on(env, ll_runs, critic, policy, exp_buffer, eps_sched=None, batch_size=10000, workers=12,
                  logging_freq=10, join=OneObsToState(),
                  run_id='default', lr=0.05, steps=12000, warmup=500, actions=4, discount=0.99):
    # parameters
    device = 'cuda'

    state = env.reset()

    # setup model
    optim = torch.optim.SGD(critic.parameters(), lr=lr)

    # monitoring
    rw = RunningReward(ll_runs)
    rec_reward = RunningReward(ll_runs)
    t = SummaryWriter(f'runs/{run_id}_{random.randint(0, 10000)}')

    # warmup to avoid correlations
    for i in range(warmup):

        if i % logging_freq == 0:
            timer.start('warmup_loop')
            logger.info(f"{Fore.LIGHTBLUE_EX}exp buffer: {len(exp_buffer)}{Style.RESET_ALL}")

        state, reward, done = one_step(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('warmup_loop')

    exp_buffer.clear()

    for i in range(steps):

        if i % logging_freq == 0:
            timer.start('main_loop')
            logger.info(f"{Fore.LIGHTBLUE_EX}exp buffer: {len(exp_buffer)}{Style.RESET_ALL}")
            timer.start('step')

        state, reward, done = one_step(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('step')

        r = reward.cpu().numpy()
        d = done.cpu().numpy()
        rw.add(r, d)
        rec_reward.add(r, d)
        t.add_scalar('reward', rec_reward.reward, i)
        t.add_scalar('episode_length', rec_reward.episode_length, i)
        rec_reward.reset()

        if i % logging_freq == 0:
            rw.log()
            rw.reset()
            timer.start('train_one')

        epsilon = eps_sched.epsilon_schedule(i)
        t.add_scalar('epsilon', epsilon, i)

        policy, critic = train_one(exp_buffer, critic, join, device, optim, actions=actions, epsilon=epsilon,
                                   batch_size=batch_size, logging_freq=0, num_workers=workers, discount_factor=discount)

        if i % logging_freq == 0:
            timer.elapsed('train_one')
            timer.elapsed('main_loop')
            print_qvalues(critic.weights.data)


def run_deep_q_with_curios_on(map, ll_runs, replay_window=1000, epsilon=0.06, batch_size=10000, workers=12,
                              logging_freq=10, run_id='default', lr=0.05, curio_lr=0.05, steps=12000):
    device = 'cuda'
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=map)

    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    target = DiscreteQTable((env.height, env.width), actions).to(device)
    learner = DiscreteQTable((env.height, env.width), actions).to(device)
    optim = torch.optim.SGD(critic.parameters(), lr=lr)
    optim_learner = torch.optim.SGD(learner.parameters(), lr=curio_lr)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=epsilon).to(device)
    rw = RunningReward(ll_runs)
    ave_reward = 0
    exp_buffer = deque(maxlen=replay_window)

    t = SummaryWriter(f'runs/{run_id}_{random.randint(0, 10000)}')

    state = env.reset()

    for i in range(steps):

        if i % logging_freq == 0:
            logger.info(f"{Fore.LIGHTBLUE_EX}exp buffer: {len(exp_buffer)}{Style.RESET_ALL}")
            timer.start('main_loop')
            timer.start('step')
        state, reward, done = one_step(env, state, policy, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('step')

        r = reward.cpu().numpy()
        d = done.cpu().numpy()
        rw.add(r, d)
        t.add_scalar('reward', r.mean().item(), i)
        t.add_scalar('done', d.sum().item(), i)
        t.add_scalar('episode_length', rw.episode_length, i)

        if i % logging_freq == 0:
            rw.log()
            rw.reset()

            timer.start('train_one')
        policy, critic = train_one_curiosity(exp_buffer, critic, target, learner, device, optim, optim_learner,
                                             actions=actions, epsilon=epsilon,
                                             batch_size=batch_size, logging_freq=0, num_workers=workers)
        if i % logging_freq == 0:
            timer.elapsed('train_one')
            print_qvalues(critic.weights.data)
            timer.elapsed('main_loop')


def print_qvalues(weights):
    ms = torch.argmax(weights, dim=0)
    arrows = [u'\N{BLACK LEFT-POINTING TRIANGLE}', u'\N{BLACK RIGHT-POINTING TRIANGLE}',
              u'\N{BLACK UP-POINTING TRIANGLE}', u'\N{BLACK DOWN-POINTING TRIANGLE}']

    for i in range(ms.size(0)):
        s = ''
        for j in range(ms.size(1)):
            s = s + arrows[ms[i, j].item()]
        logger.info(s)


frozen_lake = """
[
[T, T, T, T, T],
[S, E, E, E, E],
[E, E, E, E, E],
[E, E, T, E, T],
[E, E, E, E, E],
[E, E, T, E, T(1.0)]
]
"""


def test_frozenlake_baseline():
    for i in range(10):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=frozen_lake)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        exp_buffer = PrioritizedExpBuffer(replay_window, *env.observation_space_shape)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_baseline_{i}', warmup=1000)


def test_frozenlake_init():

    for i in range(10):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=frozen_lake)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        critic.weights.data.normal_()
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        flattener = Flattener(*env.observation_space_shape)
        exp_buffer = ExpBuffer(replay_window, flattener)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_init_norm_{i}', warmup=1000)


puddle_jumping = """
[
[S, E, E, E, E     , E, E, E, E, E     , E, E, E, E, E     ],
[E, E, E, E, E     , E, E, E, E, E     , E, E, E, E, E     ],
[E, E, T, E, T     , E, E, T, E, T     , E, E, T, E, T     ],
[E, E, E, E, E     , E, E, E, E, E     , E, E, E, E, E     ],
[E, E, T, E, E(1.0), E, E, T, E, E(1.0), E, E, T, E, T(1.0)]
]
"""

def test_puddlejump_baseline():
    for i in range(10):
        ll_runs = 600
        steps = 40000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=puddle_jumping)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        exp_buffer = ExpBuffer(replay_window, *env.observation_space_shape)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'puddle_baseline_{i}', warmup=1000)


shortjump = """
[
[S, E, E, E,  E(1.0), E, E, E, E(1.0), E, E, E, E, T(1.0)]
]
"""


def test_shortjump_baseline():
    for i in range(10):
        ll_runs = 600
        steps = 40000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=shortjump)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        exp_buffer = ExpBuffer(replay_window, *env.observation_space_shape)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'short_baseline_{i}', warmup=1000)



def test_frozenlake_paper():
    for _ in range(3):
        ll_runs = 8000
        steps = 120000
        ep_s = EpsSchedule(warmup_len=10000, finish=10000)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=frozen_lake)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        run_deep_q_on(env=env, critic=critic, policy=policy, ll_runs=ll_runs, eps_sched=ep_s,
                      replay_window=replay_window, batch_size=32 * 8000,
                      workers=1,
                      steps=steps, logging_freq=100, run_id='frozenlake_paper', warmup=1000)
        # run_deep_q_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=8000 * 400, batch_size=32, workers=1,
        #               steps=6000, logging_freq=100, run_id='frozenlake_400')
        # run_deep_q_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=8000 * 200, batch_size=32, workers=1,
        #               steps=6000, logging_freq=100, run_id='frozenlake_200')
        # run_deep_q_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=8000 * 100, batch_size=32, workers=1,
        #               steps=6000, logging_freq=100, run_id='frozenlake_100')


def test_frozenlake_window_sizes():
    for _ in range(3):
        ll_runs = 600
        steps = 30000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 20
        batch_size = 32 * ll_runs

        batch_size = 8 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window, batch_size=8 * ll_runs,
                      workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)

        batch_size = 16 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window,
                      batch_size=16 * ll_runs, workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)

        batch_size = 32 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window,
                      batch_size=32 * ll_runs, workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)

        batch_size = 64 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window,
                      batch_size=64 * ll_runs, workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)


def test_fake_lunar_lander():
    for i in range(2):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 5
        env = gym.make('GridLunarLander-v0', n=ll_runs, device=device)
        length = sum([np.prod(shape) for shape in env.observation_space_shape])
        critic = DiscreteQTable((length,), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(replay_window, *env.observation_space_shape)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_baseline_{i}', warmup=1000)


def test_shortwalk_curio():
    map = """
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """
    run_deep_q_with_curios_on(map, ll_runs=8000, epsilon=0.1, replay_window=20, batch_size=10000, workers=12)


def test_frozen_lake_curio():
    run_deep_q_with_curios_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=20, batch_size=10000, workers=12)


def test_fake_lunar_lander_curio():
    map = """
    [
    [T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, S, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0)]
    ]
    """
    run_deep_q_with_curios_on(map, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000, workers=12)


anthill = """
[
[L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, T(1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, S, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(1.0), E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, T(1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L]
]
"""


def test_anthill():
    run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000, workers=12,
                  run_id='anthill')


def test_anthill_curio():
    run_deep_q_with_curios_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000, workers=12,
                              run_id='anthill_curio')


def test_anthill_curio_batch():
    steps = 20000
    for _ in range(2):
        run_deep_q_with_curios_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                                  run_id='anthill_curio_lr_05', lr=0.05, steps=steps)
        run_deep_q_with_curios_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                                  run_id='anthill_curio_lr_01', lr=0.01, steps=steps)

        run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                      run_id='anthill_lr_10', lr=0.05, steps=steps)
        run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                      run_id='anthill_lr_05', lr=0.05, steps=steps)
        run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                      run_id='anthill_lr_01', lr=0.05, steps=steps)


def test_grid_walk():
    device = 'cuda'
    ll_runs = 8000
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """)

    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    ave_reward = 0
    episodes = 0

    while episodes < ll_runs:
        print(f'paramters: {critic.weights.data}')
        print_qvalues(critic.weights.data)
        episode, entropy, ave_reward, episodes = batch_episode(env, policy, device, max_rollout_len=4, render=False)
        logger.info(f'{Fore.GREEN}ave_reward {ave_reward} episodes {episodes} {Style.RESET_ALL}')
        policy, critic = train(episode, critic, device, optim, actions=actions, epsilon=0.05, logging_freq=0)

import random

import gym
import math
from statistics import mean

from colorama import Fore, Style
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data import *
from data import OneObsToState
from models import *
from models import DiscreteQTable, QPolicy, EpsilonGreedyProperDiscreteDist
from util import RewardAccumulator, Timer, RunningReward
import logging

logger = logging.getLogger(__name__)
timer = Timer()


def one_step(env, state, policy, join, exp_buffer, v=None, render=False, display_observation=False):

    action_dist = policy(join(state))

    action = action_dist.sample()

    next_state, reward, done, reset, info = env.step(action)

    exp_buffer.add(state, action, reward, done, reset, next_state)

    if render:
        env.render(mode='human')
    if display_observation:
        v.render(state)

    return next_state, reward, done, reset


def one_step_value(env, state, policy, join, exp_buffer, v=None, render=False, display_observation=False):

    lookahead_states, lookahead_reward, lookahead_done, info = env.lookahead()

    action_dist = policy(join(lookahead_states), lookahead_reward, lookahead_done)

    action = action_dist.sample()

    next_state, reward, done, reset, info = env.step(action)

    exp_buffer.add(state, action, reward, done, reset, next_state)

    if render:
        env.render(mode='human')
    if display_observation:
        v.render(state)

    return next_state, reward, done, reset


def train_one(exp_buffer, critic, join, device, optim, actions, discount_factor=0.99, epsilon=0.05, logging_freq=10,
              batch_size=10000):

    greedy_policy = QPolicy(critic, actions=actions, dist_class=GreedyDist)

    exp_buffer.batch_size = batch_size

    for state, action, reward, done, next_state, index, i_w in exp_buffer:

        with torch.no_grad():
            # zero the bootstrapped value of terminal states
            zero_if_terminal = (~done).to(next_state.dtype)

            # extract and convert obs to states
            state = join(state)
            next_state = join(next_state)

            # softmax and lack of logprob will affect the calculation here!
            next_action = greedy_policy(join(next_state)).sample().to(device)

            next_value = critic(next_state, next_action)
            target = reward + zero_if_terminal * discount_factor * next_value

        optim.zero_grad()
        predicted = critic(state, action)
        td_error = target - predicted
        loss = torch.mean((td_error * i_w) ** 2)
        loss.backward()
        optim.step()

        exp_buffer.update_td_error(index, td_error)

        break

    # return an epsilon greedy policy as actor
    return QPolicy(critic, actions=actions, dist_class=EpsilonGreedyProperDiscreteDist, epsilon=epsilon), critic


def train_one_value(exp_buffer, critic, join, device, optim, actions, discount_factor=0.99, epsilon=0.05, logging_freq=10,
              batch_size=10000):

    exp_buffer.batch_size = batch_size

    for state, action, reward, done, next_state, index, i_w in exp_buffer:

        # zero the bootstrapped value of terminal states
        zero_if_terminal = (~done).to(next_state.dtype)

        # extract and convert obs to states
        state = join(state)
        next_state = join(next_state)

        optim.zero_grad()
        value = critic(state)
        next_value = critic(next_state).detach()
        target = reward + discount_factor * next_value * zero_if_terminal
        td_error = (target - value)
        loss = torch.mean((td_error * i_w) ** 2)
        loss.backward()
        optim.step()

        exp_buffer.update_td_error(index, td_error)

        break

    # return an epsilon greedy policy as actor
    return VPolicy(critic, actions=actions, dist_class=EpsilonGreedyProperDiscreteDist, epsilon=epsilon), critic


def train_one_curiosity(episode, critic, tn, learner, device, optim, optim_learner, actions, discount_factor=0.99,
                        epsilon=0.05, logging_freq=10, batch_size=10000, num_workers=12):
    dataset = SARSGridDataset(episode)
    loader = SARSGridDataLoader(dataset, batch_size=batch_size)

    greedy_policy = QPolicy(critic, actions=actions, dist_class=GreedyDist)

    for state, action, reward, done, reset, next_state in loader:
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        done = done.to(device)
        reset = reset.to(device)
        next_state = next_state.to(device)

        # remove transtions going from Terminal => Start
        state = state[~reset]
        action = action[~reset]
        reward = reward[~reset]
        done = done[~reset]
        next_state = next_state[~reset]

        # zero the boostrapped value of terminal states
        zero_if_terminal = (~done).to(next_state.dtype)

        # softmax and lack of logprob will affect the calculation here!
        next_action = greedy_policy(next_state).sample().to(device)

        next_value = critic(next_state, next_action)
        target = reward + zero_if_terminal * discount_factor * next_value

        optim.zero_grad()
        predicted = critic(state, action)
        error = (target - predicted)
        loss = torch.mean(error ** 2)
        loss.backward()
        optim.step()

        # curiosity update
        optim_learner.zero_grad()
        t = tn(state, action)
        l = learner(state, action)
        loss = torch.mean((t - l) ** 2)
        loss.backward()
        optim_learner.step()

        break

    # return an epsilon greedy policy as actor
    return QPolicyCuriosity(critic, tn, learner, actions=actions, dist_class=EpsilonGreedyProperDiscreteDist,
                            epsilon=epsilon), critic


def log_stats(action, critic, dataset, i, loss, predicted, state, target):
    with torch.no_grad():
        current = critic(state, action)
        total_diff = torch.abs(predicted - current).sum().item()
        mean_diff = total_diff / len(dataset)
        magnitude = gradnorm(critic)
    logger.info(f'loss {loss.item()}')
    logger.info(f'grdnrm {magnitude}')
    logger.info(f'mean_dif {mean_diff}')
    logger.info(
        f'prev mean {predicted.mean()} std {predicted.std()} max {predicted.max()} min {predicted.min()}')
    logger.info(f'target mean {target.mean()} std {target.std()} max {target.max()} min {target.min()}')
    logger.info(
        f'current mean {current.mean()} std {current.std()} max {current.max()} min {current.min()}')
    logger.info(f'iterations {i}')


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


class ConstantEps:
    def __init__(self, eps):
        self.eps = eps

    def epsilon_schedule(self, step):
        return self.eps


class ExpExponentialDecay:
    def __init__(self, half_life, scale, bias, steps):
        self.x = np.linspace(0.0, steps, steps)
        lmda = half_life / math.log(2)
        self.x = scale * np.exp(-self.x / lmda) + bias

    def epsilon_schedule(self, step):
        return self.x[step]


class Cos:
    def __init__(self, frequency, scale, bias, steps):
        self.x = np.linspace(0.0, steps, steps)
        self.x = 0.5 * np.cos(self.x / frequency * np.pi) + 0.5
        self.x = self.x * scale + bias

    def epsilon_schedule(self, step):
        return self.x[step]


def run_on(env, stepper, learner, ll_runs, critic, policy, exp_buffer, eps_sched=None, batch_size=10000, workers=12,
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

        state, reward, done, reset = stepper(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('warmup_loop')

    exp_buffer.clear()

    for i in range(steps):

        if i % logging_freq == 0:
            timer.start('main_loop')
            logger.info(f"{Fore.LIGHTBLUE_EX}exp buffer: {len(exp_buffer)}{Style.RESET_ALL}")
            timer.start('step')

        state, reward, done, reset = stepper(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('step')

        rw.add(reward, done, reset)
        rec_reward.add(reward, done, reset)
        t.add_scalar('reward', rec_reward.reward, i)
        t.add_scalar('episode_length', rec_reward.episode_length, i)
        rec_reward.reset()

        if i % logging_freq == 0:
            rw.log()
            rw.reset()
            timer.start('train_one')

        epsilon = eps_sched.epsilon_schedule(i)
        t.add_scalar('epsilon', epsilon, i)

        policy, critic = learner(exp_buffer, critic, join, device, optim, actions=actions, epsilon=epsilon,
                                   batch_size=batch_size, logging_freq=0, discount_factor=discount)

        if i % logging_freq == 0:
            timer.elapsed('train_one')
            timer.elapsed('main_loop')
            print(critic.weights.data)


def run_value_on(env, ll_runs, critic, policy, exp_buffer, eps_sched=None, batch_size=10000, workers=12,
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

        state, reward, done, reset = one_step_value(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('warmup_loop')

    exp_buffer.clear()

    for i in range(steps):

        if i % logging_freq == 0:
            timer.start('main_loop')
            logger.info(f"{Fore.LIGHTBLUE_EX}exp buffer: {len(exp_buffer)}{Style.RESET_ALL}")
            timer.start('step')

        state, reward, done, reset = one_step_value(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('step')

        rw.add(reward, done, reset)
        rec_reward.add(reward, done, reset)
        t.add_scalar('reward', rec_reward.reward, i)
        t.add_scalar('episode_length', rec_reward.episode_length, i)
        rec_reward.reset()

        if i % logging_freq == 0:
            rw.log()
            rw.reset()
            timer.start('train_one')

        epsilon = eps_sched.epsilon_schedule(i)
        t.add_scalar('epsilon', epsilon, i)

        policy, critic = train_one_value(exp_buffer, critic, join, device, optim, actions=actions, epsilon=epsilon,
                                   batch_size=batch_size, logging_freq=0, num_workers=workers, discount_factor=discount)

        if i % logging_freq == 0:
            timer.elapsed('train_one')
            timer.elapsed('main_loop')
            print(critic.weights.data)


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

        state, reward, done, reset = one_step(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('warmup_loop')

    exp_buffer.clear()

    for i in range(steps):

        if i % logging_freq == 0:
            timer.start('main_loop')
            logger.info(f"{Fore.LIGHTBLUE_EX}exp buffer: {len(exp_buffer)}{Style.RESET_ALL}")
            timer.start('step')

        state, reward, done, reset = one_step(env, state, policy, join, exp_buffer, render=i % logging_freq == 0)

        if i % logging_freq == 0:
            timer.elapsed('step')

        rw.add(reward, done, reset)
        rec_reward.add(reward, done, reset)
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


def print_values(weigths):
    for i in range(weigths.size(0)):
        for j in range(weigths.size(1)):
            logger.info(weigths[i, j])
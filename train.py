from statistics import mean

import torch
from torch.utils.data import DataLoader

from data import *
from models import QPolicy, GreedyDist, EpsilonGreedyProperDiscreteDist, QPolicyCuriosity, gradnorm
from util import RewardAccumulator, Timer
import logging

logger = logging.getLogger(__name__)
timer = Timer()


def batch_episode(env, policy, device, max_rollout_len=4000, v=None, render=False, display_observation=False):
    episode = []
    entropy = []

    state = env.reset()
    rwa = RewardAccumulator(state.size(0), device)

    if render:
        env.render()

    for _ in range(max_rollout_len):

        action_dist = policy(state)

        entropy.append(action_dist.entropy().mean().item())

        action = action_dist.sample()

        next_state, reward, done, info = env.step(action)

        episode.append(BatchStep(state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(), done.cpu().numpy(),
                                 next_state.cpu().numpy()))

        rwa.add(reward, done)

        if render:
            env.render(mode='human')
        if display_observation:
            v.render(state)

        state = next_state

    final_entropy = mean(entropy)
    ave_reward, episodes = rwa.ave_reward()
    return episode, final_entropy, ave_reward, episodes


def one_step(env, state, policy, join, exp_buffer, v=None, render=False, display_observation=False):

    action_dist = policy(join(state))

    action = action_dist.sample()

    next_state, reward, done, info = env.step(action)

    exp_buffer.add(state, action, reward, done, next_state)

    if render:
        env.render(mode='human')
    if display_observation:
        v.render(state)

    return next_state, reward, done


def train(episode, critic, join, device, optim, actions, discount_factor=0.99, epsilon=0.05, logging_freq=10,
          batch_size=10000, num_workers=12):
    dataset = SARSGridDataset(episode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

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

        loss = torch.tensor([100.0])
        prev_loss = torch.tensor([101.0])
        i = 0

        while loss.item() > 0.01 and abs(loss.item() - prev_loss.item()) > 0.0001:
            # for _ in range(1):
            i += 1
            prev_loss = loss

            # softmax and lack of logprob will affect the calculation here!
            next_action = greedy_policy(join(next_state)).sample().to(device)

            next_value = critic(join(next_state), next_action)
            target = reward + zero_if_terminal * discount_factor * next_value

            optim.zero_grad()
            predicted = critic(join(state), action)
            error = (target - predicted)
            loss = torch.mean(error ** 2)
            loss.backward()
            optim.step()

            if logging_freq > 0 and i % logging_freq == 0:
                log_stats(action, critic, dataset, i, loss, predicted, state, target)

        if logging_freq > 0:
            log_stats(action, critic, dataset, i, loss, predicted, state, target)
        logger.info(f'iterations {i}')
    # return an epsilon greedy policy as actor
    return QPolicy(critic, actions=actions, dist_class=EpsilonGreedyProperDiscreteDist, epsilon=epsilon), critic


def train_one(exp_buffer, critic, join, device, optim, actions, discount_factor=0.99, epsilon=0.05, logging_freq=10,
              batch_size=10000, num_workers=12):

    loader= SARSGridPrioritizedTensorDataLoader(exp_buffer, batch_size)
    greedy_policy = QPolicy(critic, actions=actions, dist_class=GreedyDist)

    for state, action, reward, done, reset, next_state, index in loader:

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

        # zero the bootstrapped value of terminal states
        zero_if_terminal = (~done).to(next_state.dtype)

        # extract and convert obs to states
        state = join(exp_buffer.flattener.unflatten(state))
        next_state = join(exp_buffer.flattener.unflatten(next_state))

        # softmax and lack of logprob will affect the calculation here!
        next_action = greedy_policy(next_state).sample().to(device)

        next_value = critic(next_state, next_action)
        target = reward + zero_if_terminal * discount_factor * next_value

        optim.zero_grad()
        predicted = critic(state, action)
        error = (target - predicted)
        exp_buffer.update_td_error(index, error)
        loss = torch.mean(error ** 2)
        loss.backward()
        optim.step()

        break

    # return an epsilon greedy policy as actor
    return QPolicy(critic, actions=actions, dist_class=EpsilonGreedyProperDiscreteDist, epsilon=epsilon), critic


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
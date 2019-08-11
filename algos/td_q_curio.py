import torch

from data import SARSGridDataset, SARSGridDataLoader
from models import QPolicy, GreedyDist, QPolicyCuriosity, EpsilonGreedyProperDiscreteDist


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
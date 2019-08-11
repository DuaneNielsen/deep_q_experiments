import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
from torch.nn.functional import one_hot, selu

class GreedyDist:
    def __init__(self, probs):
        self.probs = probs

    def sample(self):
        return torch.argmax(self.probs, dim=1)

    # not sure what this should actually be, below is entropy of a random draw
    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs))


class RandomPolicy:
    def __call__(self, state):
        p = torch.ones(state.size(0), 4) / 4
        return Categorical(p)


class EpsilonCuriosityDiscreteDist:
    def __init__(self, probs, curio_probs, epsilon=0.05):
        self.probs = probs
        maxprob, argmax = probs.max(1)
        self.p = curio_probs * (epsilon + epsilon / curio_probs.size(1))
        self.p[torch.arange(self.p.size(0)), argmax] = 1.0 - epsilon

    def sample(self):
        return Categorical(self.p).sample()

    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs), dim=1)

    def logprob(self, action):
        probs = torch.sum(self.p * action, dim=1)
        return torch.log(probs)


class EpsilonGreedyProperDiscreteDist:
    def __init__(self, probs, epsilon=0.05, win_threshold=1e-8):
        self.probs = probs
        self.epsilon = epsilon

        # if no clear winner, just do a random action
        first_prob = probs[:, 0]
        diff = probs[:, :] - first_prob[:]
        same = torch.abs(diff) < win_threshold

        e = self.epsilon / (self.probs.size(1) - 1)
        max = torch.argmax(self.probs, dim=1)
        self.p = torch.ones_like(self.probs, device=probs.device) * e
        self.p[torch.arange(self.p.size(0)), max] = 1.0 - self.epsilon
        self.p[same] = 1.0 / self.probs.size(1)

    def sample(self):
        return Categorical(self.p).sample()

    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs), dim=1)

    def logprob(self, action):
        probs = torch.sum(self.p * action, dim=1)
        return torch.log(probs)


class DiscreteDist:
    def __init__(self, probs, epsilon=0.05):
        self.p = probs
        self.epsilon = epsilon

    def sample(self):
        return Categorical(self.p).sample()

    def entropy(self):
        return torch.sum(- self.p * torch.log2(self.p), dim=1)

    def logprob(self, action):
        probs = torch.sum(self.p * action, dim=1)
        return torch.log(probs)


class QPolicyCuriosity(nn.Module):
    def __init__(self, qf, tn, ln, actions, dist_class, **kwargs):
        super().__init__()
        self.qf = qf
        self.tn = tn
        self.ln = ln

        self.actions = torch.arange(actions)
        self.num_actions = actions
        self.dist_class = dist_class
        self.kwargs = kwargs

    def parameters(self, recurse=True):
        return self.qf.parameters(recurse)

    def forward(self, state):
        batch_size = state.size(0)
        input_size = state.shape[1:]

        # copy the states * number of actions
        states = state.unsqueeze(1).expand(batch_size, self.num_actions, *input_size)
        states = states.reshape(batch_size * self.num_actions, *input_size)

        # repeat the actions for each state in the batch
        actions = self.actions.unsqueeze(0).expand(batch_size, -1)
        actions = actions.reshape(batch_size * self.num_actions)

        values = self.qf(states, actions)

        tn = self.tn(states, actions)
        ln = self.ln(states, actions)
        novelty = torch.abs(tn - ln)

        values = values.reshape(batch_size, self.num_actions)
        novelty = novelty.reshape(batch_size, self.num_actions)

        probs = torch.softmax(values, dim=1)
        curio_probs = torch.softmax(novelty, dim=1)

        return EpsilonCuriosityDiscreteDist(probs, curio_probs, epsilon=0.1)


class QPolicy(nn.Module):
    def __init__(self, qf, actions, dist_class):
        super().__init__()
        self.qf = qf
        self.actions = torch.arange(actions)
        self.num_actions = actions
        self.dist_class = dist_class

    def parameters(self, recurse=True):
        return self.qf.parameters(recurse)

    def forward(self, state, **kwargs):
        batch_size = state.size(0)
        input_size = state.shape[1:]

        # copy the states * number of actions
        states = state.unsqueeze(1).expand(batch_size, self.num_actions, *input_size)
        states = states.reshape(batch_size * self.num_actions, *input_size)

        # repeat the actions for each state in the batch
        actions = self.actions.unsqueeze(0).expand(batch_size, -1).to(device=state.device)
        actions = actions.reshape(batch_size * self.num_actions)

        values = self.qf(states, actions)
        values = values.reshape(batch_size, self.num_actions)

        probs = torch.softmax(values, dim=1)

        return self.dist_class(probs, **kwargs)


class VPolicy(nn.Module):
    def __init__(self, vf, actions, dist_class, **kwargs):
        super().__init__()
        self.vf = vf
        self.actions = torch.arange(actions)
        self.num_actions = actions
        self.dist_class = dist_class
        self.kwargs = kwargs

    def parameters(self, recurse=True):
        return self.qf.parameters(recurse)

    def forward(self, states, rewards, done):
        """
        :param states: lookahead in N, Action, State
        :param states: lookahead in N, rewards

        :return: Probability distribution over actions
        """

        b = states.size(0)
        a = states.size(1)
        obs_shape = states.shape[2:]
        states = states.view(b * a, *obs_shape)
        rewards = rewards.view(b * a)
        done = done.view(b * a)
        values = self.vf(states) * (~done).float() + rewards
        values = values.view(b, a)
        probs = torch.softmax(values, dim=1)

        return self.dist_class(probs, **self.kwargs)


def gradnorm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class DiscreteVTable(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(*features))
        self.features = features
        self.feature_dims = list(range(1, len(features) + 1))

    def forward(self, state):
        activations = state * self.weights
        return torch.sum(activations, dim=self.feature_dims)


class DiscreteQTable(nn.Module):
    def __init__(self, features, actions):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(actions, *features))

    def forward(self, state, action):
        return torch.sum(self.weights[action, :, :] * state, dim=[1, 2])


class FixupFCResLayer(nn.Module):
    def __init__(self, depth, in_features):
        super().__init__()
        self.c1 = nn.Linear(in_features, in_features)
        self.c1.weight.data.mul_(depth ** -0.5)
        self.c2 = nn.Linear(in_features, in_features)
        self.c2.weight.data.zero_()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])

    def forward(self, input):
        hidden = input + self.bias[0]
        hidden = self.c1(hidden) + self.bias[1]
        hidden = torch.relu(hidden) + self.bias[2]
        hidden = self.c2(hidden) * self.gain + self.bias[3]

        return torch.relu(hidden + input)


class FixupQ(nn.Module):
    def __init__(self, state_shape, actions, blocks):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.actions = actions
        self.features = self.state_features + actions
        self.resblocks = nn.Sequential(*[FixupFCResLayer((depth * 2) + 1, self.features) for depth in range(blocks)])
        self.value = nn.Linear(self.features, 1)
        self.value.weight.data.zero_()
        self.value.bias.data.zero_()
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, states, actions):
        """

        :param states: batch, shape1, shape2  state is flattened
        :param actions: batch, Long representing the action
        :return: batch, values of taking action in state
        """
        action_oh = one_hot(actions, num_classes=self.actions).float()
        input = torch.cat((states.view(-1, self.state_features), action_oh), dim=1)
        hidden = self.resblocks(input)
        values = self.value(hidden) * self.gain + self.bias
        return values.squeeze()


class ProjFixupQ(nn.Module):
    def __init__(self, state_shape, actions, hidden_size, blocks):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.actions = actions
        self.features = self.state_features + actions

        self.proj = nn.Linear(self.features, hidden_size, bias=True)
        self.resblocks = nn.Sequential(*[FixupFCResLayer((depth * 2) + 2, hidden_size) for depth in range(blocks)])
        self.value = nn.Linear(hidden_size, 1)
        self.value.weight.data.zero_()
        self.value.bias.data.zero_()
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, states, actions):
        """

        :param states: batch, shape1, shape2  state is flattened
        :param actions: batch, Long representing the action
        :return: batch, values of taking action in state
        """
        action_oh = one_hot(actions, num_classes=self.actions).float()
        input = torch.cat((states.view(-1, self.state_features), action_oh), dim=1)
        hidden = self.proj(input)
        hidden = self.resblocks(hidden)
        values = self.value(hidden) * self.gain + self.bias
        return values.squeeze()


class FixupV(nn.Module):
    def __init__(self, state_shape, blocks):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.features = self.state_features
        self.resblocks = nn.Sequential(*[FixupFCResLayer(depth * 2 + 1, self.features) for depth in range(blocks)])
        self.value = nn.Linear(self.features, 1)
        self.value.weight.data.mul_(blocks * 2 + 1 ** -0.5)
        #self.value.bias.data.zero
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        """

        :param states: batch, shape1, shape2  state is flattened
        :return: batch, values of taking action in state
        """
        input = states.view(-1, self.state_features)
        hidden = self.resblocks(input)
        values = self.value(hidden) * self.gain + self.bias
        return values.squeeze()


class SimpleV(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.features = self.state_features
        self.ln = nn.Linear(self.state_features, 1)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, states):
        input = states.view(-1, self.state_features)
        value = self.ln(input) * self.gain
        return value.squeeze()


class EnsembleV(nn.Module):
    def __init__(self, state_shape, ensembles):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.features = self.state_features
        self.ensemebles = ensembles
        self.ln = nn.Linear(self.state_features * ensembles, 3)

        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, states):
        input = states.view(-1, self.state_features)
        input = input.unsqueeze(2).expand(-1, self.state_features, self.ensemebles).reshape(-1, self.state_features * self.ensemebles)
        value = torch.sum(torch.relu(self.ln(input)), dim=1) / self.ensemebles * self.gain
        return value.squeeze()


class ProjectionV(nn.Module):
    def __init__(self, state_shape, hidden_size):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.features = self.state_features
        self.proj = nn.Linear(self.state_features, hidden_size, bias=True)
        self.ln = nn.Linear(hidden_size, 1, bias=False)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, states):
        input = states.view(-1, self.state_features)
        hidden = torch.relu(self.proj(input))
        value = self.ln(hidden) * self.gain
        return value.squeeze()


class DeepProjectionV(nn.Module):
    def __init__(self, state_shape, hidden_size, layers):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.features = self.state_features
        self.proj = nn.Linear(self.state_features, hidden_size, bias=True)
        self.resblocks = nn.Sequential(*[FixupFCResLayer(depth * 2 + 2, hidden_size) for depth in range(layers)])
        self.ln = nn.Linear(hidden_size, 1, bias=False)
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, states):
        input = states.view(-1, self.state_features)
        hidden = torch.relu(self.proj(input))
        hidden = self.resblocks(hidden)
        value = self.ln(hidden) * self.gain + self.bias
        return value.squeeze()


class OneHotV(nn.Module):
    def __init__(self, state_shape, buckets):
        super().__init__()
        self.state_features = np.prod(state_shape)
        self.features = self.state_features
        self.buckets = buckets
        self.ln = nn.Linear(self.buckets, 1)
        #self.gain = nn.Parameter(torch.ones(1))

    def forward(self, states):
        input = states.view(-1, self.state_features)
        input = torch.floor(input * (self.buckets - 1)).long()
        oh = one_hot(input, self.buckets).float()
        value = self.ln(oh)
        return value.squeeze()


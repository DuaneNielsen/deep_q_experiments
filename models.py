import torch
from torch import nn
from torch.distributions import Categorical


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
    def __init__(self, probs, epsilon=0.05):
        self.probs = probs
        self.epsilon = epsilon

        e = self.epsilon / (self.probs.size(1) - 1)
        max = torch.argmax(self.probs, dim=1)
        self.p = torch.ones_like(self.probs, device=probs.device) * e
        self.p[torch.arange(self.p.size(0)), max] = 1.0 - self.epsilon

    def sample(self):
        return Categorical(self.p).sample()

    def entropy(self):
        return torch.sum(- self.probs * torch.log2(self.probs), dim=1)

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
    def __init__(self, qf, actions, dist_class, **kwargs):
        super().__init__()
        self.qf = qf
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
        values = values.reshape(batch_size, self.num_actions)

        probs = torch.softmax(values, dim=1)

        return self.dist_class(probs, **self.kwargs)


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

    def forward(self, states):
        """
        :param states: lookahead in N, Action, State
        :return: Probability distribution over actions
        """
        b = states.size(0)
        a = states.size(1)
        obs_shape = states.shape[2:]
        states = states.view(b * a, *obs_shape)
        values = self.vf(states)
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
import numpy as np
import torch
from torch.utils.data import Dataset
from util import Timer
from torch.utils.data.dataloader import default_collate
from multiprocessing import Process
from collections import deque
from torch.utils.data import TensorDataset
from operator import mul
import matplotlib.pyplot as plt

timer = Timer()


class SimpleExpBuffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)

    def add(self, state, action, reward, done, next_state):
        self.buffer.append(BatchStep(state, action, reward, done, next_state))

    def dataset(self):
        return SARSGridDataset(self.buffer)

    def __len__(self):
        return len(self.buffer)


class BatchStep:
    def __init__(self, state, action, reward, done, next_state):
        self.state = state.cpu().numpy()
        self.action = action.cpu().numpy()
        self.reward = reward.cpu().numpy()
        self.done = done.cpu().numpy()
        self.next_state = next_state.cpu().numpy()

    def __getitem__(self, item):
        return Step(self.state[item], self.action[item], self.reward[item], self.done[item], self.next_state[item])


class Step:
    def __init__(self, state, action, reward, done, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.next_state = next_state


class SARSGridDataset(Dataset):
    def __init__(self, episode):
        super().__init__()
        self.episode = episode
        self.batch_size = episode[0].state.shape[0]

    def _transform(self, step, resetting):
        state = torch.from_numpy(step.state)
        action = torch.tensor(step.action.item())
        reward = torch.tensor(step.reward.item())
        done = torch.tensor(step.done.item(), dtype=torch.uint8)
        resetting = torch.tensor(resetting, dtype=torch.uint8)
        next_state = torch.from_numpy(step.next_state)
        return state, action, reward, done, resetting, next_state

    def __getitem__(self, item):
        t = item // self.batch_size
        offset = item % self.batch_size
        step = self.episode[t][offset]
        if t > 0:
            resetting = self.episode[t - 1][offset].done.item()
        else:
            resetting = 0
        return self._transform(step, resetting)

    def __len__(self):
        return len(self.episode) * self.batch_size


class SARSGridDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        timer.start('load_time')
        sample = np.random.randint(len(self.dataset), size=self.batch_size)
        data = []
        for item in np.nditer(sample):
            data.append(self.dataset[item])
        batch = default_collate(data)
        timer.elapsed('load_time')
        return batch


def load(n, dataset, q):
    sample = np.random.randint(len(dataset), size=n)
    for item in np.nditer(sample):
        q.put(dataset[item])


# todo use multiprocessing RawArray to write everything into a set of memory slices
# then load by copying offsets
class SARSGridLLDataLoader:
    def __init__(self, dataset, batch_size, workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.arr = Array('i', range(batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        timer.start('load_time')
        n = self.batch_size // self.workers
        # put array here
        threads = [Process(target=load, args=(n, self.dataset, data)) for _ in range(self.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        batch = default_collate(data)
        timer.elapsed('load_time')
        return batch


class ManyObsToState:
    def __call__(self, observations):
        return torch.cat([s.flatten(start_dim=1) for s in observations], dim=1)


class OneObsToState:
    def __call__(self, observation):
        return observation


class Flattener:
    def __init__(self, *shapes):
        self.shapes = shapes
        self.length = 0
        for shape in shapes:
            self.length += np.prod(list(shape)).item()

    def flatten(self, *args):
        with torch.no_grad():
            flat = []
            for tensor in args:
                flat.append(torch.flatten(tensor, start_dim=2))
            return torch.cat(flat, dim=1)

    def unflatten(self, data):
        tensors = []
        offset = 0
        batch_size = data.size(0)
        for shape in self.shapes:
            length = np.prod(list(shape)).item()
            slice = torch.narrow(data, 1, offset, length)
            slice = slice.reshape(batch_size, *shape)
            tensors.append(slice)
            offset += length

        if len(tensors) == 1:
            return tensors[0]
        else:
            return tuple(tensors)


class ExpBuffer:
    def __init__(self, max_timesteps, ll_runs, batch_size, observation_shape):

        self.observation_shape = observation_shape
        self.cursor = 0
        requires_grad = False
        self.window_size = max_timesteps * ll_runs
        self.ll_runs = ll_runs
        self.max_timesteps = max_timesteps
        self.state = torch.empty(size=(max_timesteps, ll_runs, *observation_shape), dtype=torch.float32,
                                 requires_grad=requires_grad,
                                 device='cpu')
        self.action = torch.empty(max_timesteps, ll_runs, dtype=torch.long, device='cpu')
        self.reward = torch.empty(max_timesteps, ll_runs, dtype=torch.float32, requires_grad=requires_grad,
                                  device='cpu')
        self.done = torch.zeros(max_timesteps, ll_runs, dtype=torch.uint8, device='cpu')
        self.reset = torch.zeros(max_timesteps, ll_runs, dtype=torch.uint8, device='cpu')
        self.next = torch.empty(max_timesteps, ll_runs, *observation_shape, dtype=torch.float32,
                                requires_grad=requires_grad,
                                device='cpu')
        self.full = False
        self.batch_size = batch_size

    def add(self, state, action, reward, done, reset, next):
        with torch.no_grad():
            self.state[self.cursor] = state.cpu()
            self.next[self.cursor] = next.cpu()
            self.action[self.cursor] = action.cpu()
            self.reward[self.cursor] = reward.cpu()
            self.done[self.cursor] = done.cpu()
            self.reset[self.cursor] = reset.cpu()

            self.cursor += 1
            if not self.full:
                self.full = self.cursor >= self.max_timesteps
            self.cursor = self.cursor % self.max_timesteps

    def update_td_error(self, index, td_error):
        pass

    def __len__(self):
        if self.full:
            return self.window_size
        else:
            return self.cursor * self.ll_runs

    def get_trajectory(self):
        # todo this is probably innaccurate, needs more work and refinement
        # todo probably should be using reset not done
        trajectory = []
        cursor = self.cursor - 1
        while cursor >= 0:
            state = self.state[cursor, 0]
            reward = self.reward[cursor, 0]
            action = self.action[cursor, 0]
            done = self.done[cursor, 0]
            trajectory.append(action)
            cursor -= 1
            if self.done[cursor, 0] == 1:
                break
        return list(reversed(trajectory))

    def clear(self):
        self.full = False
        self.cursor = 0

    def data(self):
        state = self.state.reshape(self.window_size, *self.observation_shape)
        action = self.action.reshape(self.window_size)
        reward = self.reward.reshape(self.window_size)
        done = self.done.reshape(self.window_size)
        reset = self.reset.reshape(self.window_size)
        next = self.next.reshape(self.window_size, *self.observation_shape)
        return state, action, reward, done, reset, next

    def __iter__(self):
        return SARSGridTensorDataLoader(self, batch_size=self.batch_size)


class SARSGridTensorDataLoader:
    def __init__(self, exp_buffer, batch_size, device='cuda'):
        self.exp_buffer = exp_buffer
        self.batch_size = batch_size
        self.n = 0
        self.state, self.action, self.reward, self.done, self.reset, self.next = exp_buffer.data()
        self.device = device

    def __iter__(self):
        return self

    def load_to_device(self, index, importance_weights):
        state = self.state[index].to(self.device)
        action = self.action[index].to(self.device)
        reward = self.reward[index].to(self.device)
        done = self.done[index].to(self.device)
        next = self.next[index].to(self.device)
        return state, action, reward, done, next, index, importance_weights

    def sample(self):
        sample = np.random.randint(len(self.exp_buffer), None, size=self.batch_size)
        index = torch.from_numpy(sample)

        # remove resets from the batch, causes variable size batches
        index = index[~self.reset[index]]
        importance_weights = torch.ones_like(index, dtype=torch.float32, device=self.device)
        return index, importance_weights

    def __next__(self):
        if self.n * self.batch_size > len(self.exp_buffer):
            raise StopIteration()
        self.n += 1

        index, importance_weights = self.sample()

        return self.load_to_device(index, importance_weights)


class PrioritizedExpBuffer(ExpBuffer):
    def __init__(self, max_timesteps, ll_runs, batch_size, observation_shape, importance_sample=True, prioritize=True):
        super().__init__(max_timesteps, ll_runs, batch_size, observation_shape)
        self.importance_sample = importance_sample
        self.td_error = torch.zeros(max_timesteps * ll_runs, dtype=torch.float32, device='cuda', requires_grad=False)
        self.td_error_empty = torch.ones(max_timesteps * ll_runs, dtype=torch.uint8, device='cuda', requires_grad=False)
        self.prioritize = prioritize

    def add(self, state, action, reward, done, reset, next):
        start = self.cursor * self.ll_runs
        end = start + self.ll_runs
        i = torch.arange(start, end)
        self.td_error[i] = float('Inf')
        self.td_error_empty[i] = False
        super().add(state, action, reward, done, reset, next)

    def update_td_error(self, index, td_error):
        with torch.no_grad():
            self.td_error[index.cuda()] = td_error

    def clear(self):
        self.td_error.fill_(-1.0)
        super().clear()

    def __iter__(self):
        return PrioritizedLoader(self, batch_size=self.batch_size, importance_sample=self.importance_sample,
                                 prioritize=self.prioritize)


def histogram(x, title):
    num_bins = 40
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.xlabel('value')
    plt.ylabel('number samples')
    plt.title(title)
    plt.legend()
    plt.show()


class PrioritizedLoader(SARSGridTensorDataLoader):
    def __init__(self, exp_buffer, batch_size, device='cuda', prioritize=True, importance_sample=False):
        super().__init__(exp_buffer, batch_size, device)
        self.importance_sample = importance_sample
        self.prioritize = prioritize

    def sample(self):
        with torch.no_grad():
            new = torch.isinf(self.exp_buffer.td_error)
            empty = self.exp_buffer.td_error_empty
            reset = self.reset.cuda()
            new_n = torch.sum(new).item()
            sample_size = self.batch_size - new_n
            sample_from = ~(new | empty | reset)
            eps = 1e-12
            p = torch.abs(self.exp_buffer.td_error.clone()) + eps
            sum = p[sample_from].sum()

            if self.prioritize:
                # prioritized replay sampling
                if new_n < self.batch_size:
                    p[sample_from] = p[sample_from] * sample_size / sum
                    p[~sample_from] = 0.0
                    r = torch.rand_like(p)
                    sampled = r < p
                    index = sampled | new
                else:
                    p[new] = self.batch_size / new_n
                    p[~new] = 0.0
                    r = torch.rand_like(p)
                    index = r < p
            else:
                prob = self.batch_size / self.exp_buffer.td_error.size(0)
                p = torch.full_like(self.exp_buffer.td_error, prob)
                r = torch.rand_like(p)
                index = r < p

            if self.importance_sample:
                # weighted importance sampling
                N = sample_from.sum()

                iw = torch.ones_like(self.exp_buffer.td_error, device='cuda')
                true_p = p[index]
                iw[index] = true_p.reciprocal() / sample_size / N
                iw[new] = 1.0
                iw[index] = iw[index] / iw[index].max()
                iw = iw[index]
            else:
                iw = torch.ones(index.sum().item(), device='cuda')

            index = index.nonzero().flatten().cpu()

            return index, iw


class SARSGridPrioritizedTensorDataLoader:
    def __init__(self, exp_buffer, batch_size, importance_sample=False, correct_batch_size=False):
        self.exp_buffer = exp_buffer
        self.batch_size = batch_size
        self.n = 0
        self.correct_batch_size = correct_batch_size
        self.importance_sample = importance_sample

    def __iter__(self):
        return self

    def __next__(self):
        with torch.no_grad():
            if self.n * self.batch_size > len(self.exp_buffer):
                raise StopIteration()
            self.n += 1

            # prioritized replay sampling

            # anything lower than zero is not yet initizized, so ignore it
            empty = self.exp_buffer.td_error < 0.0

            # anything with infinite error is new, and so automatically sampled
            new = torch.isinf(self.exp_buffer.td_error)

            # sample based on weighted TD error from the remainder
            sample_from = ~(empty | new)
            num_new = torch.sum(new)
            samples = self.batch_size - num_new
            eps = 1e-8
            td_error = torch.abs(self.exp_buffer.td_error[sample_from]) + eps

            w = torch.zeros_like(self.exp_buffer.td_error)

            # this is not quite accurate, as * samples causes the probability
            # of some samples being picked to go greater than 1.0, but this effect reduces as the
            # size of sample_from increases.. the impact of this error is smaller batch sizes than expected
            w[sample_from] = td_error * samples / torch.sum(td_error)
            r = torch.rand_like(self.exp_buffer.td_error)
            index = r < w

            if self.importance_sample:
                # weighted importance sampling
                N = torch.sum(index).item() + torch.sum(new).item()
                iw = torch.zeros_like(self.exp_buffer.td_error, device='cuda')
                iw[index] = w[index].reciprocal() / N
                iw[new] = 1.0 / N

            # add back in the new samples and remove resets from the batch
            index = index | new
            index = ~self.exp_buffer.reset.cuda() & index

            if self.importance_sample:
                # normalize importance weights
                iw[index] = iw[index] * iw[index].max().reciprocal()
            else:
                iw = torch.ones(index.size(0), dtype=torch.float32, device='cuda')

            index = index.cpu()
            return self.exp_buffer.state[index], self.exp_buffer.action[index], self.exp_buffer.reward[index], \
                   self.exp_buffer.done[index], self.exp_buffer.reset[index], self.exp_buffer.next[index], index, iw[
                       index]



class PrioritizedExpBufferOld:
    def __init__(self, sliding_window_size, batch_size, importance_sample, *observation_shapes):
        self.flattener = Flattener(*observation_shapes)

        self.cursor = 0
        requires_grad = False
        self.window_size = sliding_window_size
        self.state = torch.empty(sliding_window_size, self.flattener.length, dtype=torch.float32,
                                 requires_grad=requires_grad,
                                 device='cpu')
        self.action = torch.empty(sliding_window_size, dtype=torch.long, device='cpu')
        self.reward = torch.empty(sliding_window_size, dtype=torch.float32, requires_grad=requires_grad, device='cpu')
        self.done = torch.empty(sliding_window_size, dtype=torch.uint8, device='cpu')
        self.reset = torch.zeros(sliding_window_size, dtype=torch.uint8, device='cpu')
        self.next = torch.empty(sliding_window_size, self.flattener.length, dtype=torch.float32,
                                requires_grad=requires_grad,
                                device='cpu')
        self.td_error = torch.ones(sliding_window_size, dtype=torch.float32, device='cuda', requires_grad=False) * -1.0
        self.full = False
        self.batch_size = batch_size
        self.importance_sample = importance_sample

    def index(self, cursor, size):
        if cursor + size <= self.window_size:
            index = torch.arange(cursor, cursor + size)
        else:
            i1 = torch.arange(cursor, self.window_size)
            i2 = torch.arange(0, (cursor + size) % self.window_size)
            index = torch.cat((i1, i2))
        return index

    def add(self, state, action, reward, done, next):
        with torch.no_grad():
            size = state.size(0)
            index = self.index(self.cursor, size)
            reset = self.index(self.cursor + 1, size)

            state = self.flattener.flatten(state)
            next = self.flattener.flatten(next)

            self.state[index] = state[:].cpu()
            self.action[index] = action[:].cpu()
            self.reward[index] = reward[:].cpu()
            done = done.cpu()
            self.done[index] = done[:]
            self.reset[reset] = done[:]
            self.next[index] = next[:].cpu()
            self.td_error[index.cuda()] = float('Inf')

            self.cursor += size
            if not self.full:
                self.full = self.cursor >= self.window_size
            self.cursor = self.cursor % self.window_size

    def update_td_error(self, index, td_error):
        with torch.no_grad():
            self.td_error[index] = td_error

    def dataset(self):
        return TensorDataset(self.state, self.action, self.reward, self.done, self.reset, self.next)

    def __len__(self):
        if self.full:
            return self.window_size
        else:
            return self.cursor

    def clear(self):
        self.td_error.fill_(-1.0)
        self.full = False
        self.cursor = 0

    def __iter__(self):
        return SARSGridPrioritizedTensorDataLoader(self, batch_size=self.batch_size,
                                                   importance_sample=self.importance_sample)

import numpy as np
import torch
from torch.utils.data import Dataset
from util import Timer
from torch.utils.data.dataloader import default_collate
from multiprocessing import Process
from collections import deque
from torch.utils.data import TensorDataset
from operator import mul

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
        self.state = torch.empty(max_timesteps, ll_runs, *observation_shape, dtype=torch.float32,
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
        self.reset_buffer = torch.zeros(ll_runs, dtype=torch.uint8, device='cpu')
        self.full = False
        self.batch_size = batch_size

    def add(self, state, action, reward, done, next):
        with torch.no_grad():
            self.state[self.cursor] = state.cpu()
            self.next[self.cursor] = next.cpu()
            self.action[self.cursor] = action.cpu()
            self.reward[self.cursor] = reward.cpu()
            done = done.cpu()
            self.done[self.cursor] = done

            # save the done flag into buffer, use it to flag next T -> S transitions (resets)
            self.reset[self.cursor] = self.reset_buffer
            self.reset_buffer = done

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

    def clear(self):
        self.full = False
        self.cursor = 0

    def data(self):
        state = self.state.reshape(self.window_size, *self.observation_shape)
        action = self.action.reshape(self.window_size)
        reward = self.reward.reshape(self.window_size)
        done = self.done.reshape(self.window_size)
        reset = self.reset.reshape(self.window_size)
        next = self.state.reshape(self.window_size, *self.observation_shape)
        return state, action, reward, done, reset, next

    def __iter__(self):
        return SARSGridTensorDataLoader(self, batch_size=self.batch_size)


class SARSGridTensorDataLoader:
    def __init__(self, exp_buffer, batch_size):
        self.exp_buffer = exp_buffer
        self.batch_size = batch_size
        self.n = 0
        self.state, self.action, self.reward, self.done, self.reset, self.next = exp_buffer.data()

    def __iter__(self):
        return self

    def __next__(self):
        if self.n * self.batch_size > len(self.exp_buffer):
            raise StopIteration()
        self.n += 1
        sample = np.random.randint(len(self.exp_buffer), None, size=self.batch_size)
        index = torch.from_numpy(sample)
        index = index[~self.reset[index]]
        if torch.sum(torch.tensor([1.0, 0.0, 1.0]).expand(self.batch_size, 3) * self.state[index]) != 0:
            print(self.state[index])

        return self.state[index], self.action[index], self.reward[index], \
               self.done[index], self.reset[index], self.next[index], index, torch.ones_like(index, dtype=torch.float32,
                                                                                            device='cpu')


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


class PrioritizedExpBuffer:
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

from data import *
import numpy as np
import torch
from torch import tensor
import gym
import gym_duane


def test_SARSGridDataset():
    state = np.array([
        [[1, 0, 0]],
        [[1, 0, 0]]
    ])
    action = np.array([
        [0],
        [0]
    ])
    reward = np.array([
        [0.0],
        [0.0]
    ])
    done = np.array([
        [1],
        [1]
    ])
    next_state = np.array([
        [[0, 1, 0]],
        [[0, 1, 0]]
    ])

    a = BatchStep(state, action, reward, done, next_state)

    state = np.array([
        [[0, 1, 0]],
        [[0, 1, 0]]
    ])
    action = np.array([
        [0],
        [0]
    ])
    reward = np.array([
        [0.0],
        [0.0]
    ])
    done = np.array([
        [0],
        [0]
    ])
    next_state = np.array([
        [[1, 0, 0]],
        [[1, 0, 0]]
    ])

    b = BatchStep(state, action, reward, done, next_state)

    episode = [a, b]

    dataset = SARSGridDataset(episode)

    state, action, reward, done, reset, next_state = dataset[0]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[1]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[2]
    print(state, action, reward, done, reset)

    state, action, reward, done, reset, next_state = dataset[3]
    print(state, action, reward, done, reset)


def test_expbuffer():

    eb = ExpBuffer(max_timesteps=5, ll_runs=3, batch_size=2, observation_shape=(4, 4))

    state_b = torch.ones(5, 3, 4, 4) * torch.arange(5).float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
    action_b = torch.ones(5, 3, dtype=torch.long)
    reward_b = torch.ones(5, 3)
    done_b = torch.ones(5, 3, dtype=torch.uint8)
    next_b = torch.ones(5, 3, 4, 4) * torch.arange(5).float().unsqueeze(1).unsqueeze(1).unsqueeze(1)

    t = 0
    eb.add(state_b[t], action_b[t], reward_b[t], done_b[t], next_b[t])

    for state, action, reward, done, reset, next, index, iw in eb:
        assert state.shape == (2, 4, 4)
        assert action.shape == (2,)
        assert reward.shape == (2,)
        assert done.shape == (2,)
        assert next.shape == (2, 4, 4)



def test_expbuffer_with_bandit():
    eb = ExpBuffer(max_timesteps=2, ll_runs=3, batch_size=2, observation_shape=(1, 3))
    env = gym.make('SimpleGrid-v3', n=3, device='cuda', max_steps=40, map_string="""
        [
        [T(-1.0), S, T(1.0)]
        ]
        """)

    s = env.reset()
    for _ in range(1000):
        action = torch.randint(4, (3,)).cuda()
        n, reward, done, info = env.step(action)
        eb.add(s, action, reward, done, n)

        term = torch.sum(n * torch.tensor([1.0, 0.0, 1.0]).cuda(), dim=[1, 2])
        assert torch.allclose(term, done.float())

        for state, action, reward, done, reset, next, index, iw in eb:
            b = index.size(0)
            print(state, next, done, reset)
            assert state.shape == (b, 1, 3)
            assert action.shape == (b,)
            assert reward.shape == (b,)
            assert done.shape == (b,)
            assert next.shape == (b, 1, 3)
            term = torch.sum(state * torch.tensor([1.0, 0.0, 1.0]), dim=[1, 2])
            assert torch.allclose(term, torch.zeros_like(term))

        s = n.clone()

def test_flattener():
    flattener = Flattener((20, 20), (15,))

    image = torch.rand(5, 20, 20)
    instr = torch.rand(5, 15)

    flat = flattener.flatten(image, instr)

    assert flat.size(0) == 5

    out_image, out_instr = flattener.unflatten(flat)

    assert torch.allclose(image, out_image)
    assert torch.allclose(instr, out_instr)


def test_many_to_state():
    image = torch.rand(5, 20, 20)
    instr = torch.rand(5, 15)

    t = ManyObsToState()

    out = t((image, instr))

    assert out.size(0) == 5
    assert out.size(1) == 20 * 20 + 15


def test_prioritized_replay():
    batch_size = 2
    eb = PrioritizedExpBuffer(4, batch_size, True, (2, 2))

    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    action = torch.tensor([1], dtype=torch.int64)
    reward = torch.tensor([1])
    done = torch.tensor([1], dtype=torch.uint8)
    next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    eb.add(state, action, reward, done, next)

    #loader = SARSGridPrioritizedTensorDataLoader(eb, 2)
    for state, action, reward, done, reset, next, index in eb:
        assert torch.allclose(index, torch.tensor([1, 0, 0, 0], dtype=torch.uint8))
        error = torch.tensor([0.0]).cuda()
        eb.update_td_error(index, error)
        break

    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    action = torch.tensor([1], dtype=torch.int64)
    reward = torch.tensor([1])
    done = torch.tensor([1], dtype=torch.uint8)
    next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    eb.add(state, action, reward, done, next)

    loader = SARSGridPrioritizedTensorDataLoader(eb, 2)
    for state, action, reward, done, reset, next, index in loader:
        assert torch.allclose(index, torch.tensor([1, 1, 0, 0], dtype=torch.uint8))
        error = torch.tensor([0.5, 0.5]).cuda()
        eb.update_td_error(index, error)
        break

    def step():
        state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        action = torch.tensor([1], dtype=torch.int64)
        reward = torch.tensor([1])
        done = torch.tensor([1], dtype=torch.uint8)
        next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        eb.add(state, action, reward, done, next)

        loader = SARSGridPrioritizedTensorDataLoader(eb, 2)
        for state, action, reward, done, reset, next, index in loader:
            print(index)
            error = torch.rand(index.sum()).cuda()
            eb.update_td_error(index, error)
            break

    step()
    step()
    step()
    step()


def test_prioritized_large():
    eb = PrioritizedExpBuffer(1000, 10 * 32, True, (1,))

    def transition(num, size):
        state = torch.tensor([[num]]*size).float()
        action = torch.tensor([num]*size, dtype=torch.int64)
        reward = torch.tensor([num]*size).float()
        done = torch.tensor([0]*size, dtype=torch.uint8)
        next = torch.tensor([[num]]*size).float()
        return state, action, reward, done, next

    def step(expected_size):
        eb.add(*transition(1, 10))

        for state, action, reward, done, reset, next, index, i_s in eb:
            error = torch.rand(index.sum()).cuda()
            eb.update_td_error(index, error)
            print(action.size(0), expected_size)
            break

    for i in range(10, 320, 10):
        step(i)

    for _ in range(100):
        step(320)
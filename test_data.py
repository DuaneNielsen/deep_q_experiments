from data import *
import numpy as np
import torch

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

    eb = ExpBuffer(2, (2, 2))

    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    action = torch.tensor([1], dtype=torch.int64)
    reward = torch.tensor([1])
    done = torch.tensor([1], dtype=torch.uint8)
    next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    eb.add(state, action, reward, done, next)

    def check_state(index, state, next):
        out_state = eb.flattener.unflatten(eb.state[index].unsqueeze(0))
        out_next = eb.flattener.unflatten(eb.next[index].unsqueeze(0))
        assert torch.allclose(state, out_state)
        assert torch.allclose(next, out_next)

    check_state(0, state, next)
    assert eb.done[0]

    state2 = torch.rand(1, 2, 2)
    action2 = torch.tensor([1], dtype=torch.int64)
    reward2 = torch.tensor([1])
    done2 = torch.tensor([0], dtype=torch.uint8)
    next2 = torch.rand(1, 2, 2)

    eb.add(state2, action2, reward2, done2, next2)

    assert eb.reset[1].item() == 1
    check_state(1, state2, next2)

    state2 = torch.rand(1, 2, 2)
    action2 = torch.tensor([1], dtype=torch.int64)
    reward2 = torch.tensor([1])
    done2 = torch.tensor([1], dtype=torch.uint8)
    next2 = torch.rand(1, 2, 2)

    eb.add(state2, action2, reward2, done2, next2)

    assert eb.reset[0].item() == 0
    check_state(0, state2, next2)

    state2 = torch.rand(1, 2, 2)
    action2 = torch.tensor([1], dtype=torch.int64)
    reward2 = torch.tensor([1])
    done2 = torch.tensor([0], dtype=torch.uint8)
    next2 = torch.rand(1, 2, 2)

    eb.add(state2, action2, reward2, done2, next2)

    assert eb.reset[1].item() == 1
    check_state(1, state2, next2)


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
    eb = PrioritizedExpBuffer(4, (2, 2))

    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    action = torch.tensor([1], dtype=torch.int64)
    reward = torch.tensor([1])
    done = torch.tensor([1], dtype=torch.uint8)
    next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    eb.add(state, action, reward, done, next)

    loader = SARSGridPrioritizedTensorDataLoader(eb, 2)
    for state, action, reward, done, reset, next, index in loader:
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
    eb = PrioritizedExpBuffer(1000, (1,))

    def transition(num, size):
        state = torch.tensor([[num]]*size).float()
        action = torch.tensor([num]*size, dtype=torch.int64)
        reward = torch.tensor([num]*size).float()
        done = torch.tensor([num]*size, dtype=torch.uint8)
        next = torch.tensor([[num]]*size).float()
        return state, action, reward, done, next

    def step(expected_size):
        eb.add(*transition(1, 10))

        loader = SARSGridPrioritizedTensorDataLoader(eb, batch_size=10 * 32)
        for state, action, reward, done, reset, next, index in loader:
            error = torch.rand(index.sum()).cuda()
            eb.update_td_error(index, error)
            print(action.size(0), expected_size)
            break

    for i in range(10, 320, 10):
        step(i)

    for _ in range(100):
        step(320)
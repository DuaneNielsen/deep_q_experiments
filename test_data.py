from data import *
import numpy as np
import torch
from torch import tensor
import gym
import gym_duane
from models import *
from torch.distributions import Beta, Exponential, Uniform
import matplotlib.pyplot as plt


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
    for _ in range(10):
        action = torch.randint(4, (3,)).cuda()
        n, reward, done, reset, info = env.step(action)
        eb.add(s, action, reward, done, reset, n)

        term = torch.sum(n * torch.tensor([1.0, 0.0, 1.0]).cuda(), dim=[1, 2])
        assert torch.allclose(term, done.float())
        left = torch.tensor([[1.0, 0.0, 0.0]]).cuda()
        right = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
        left_action = (action == 0) & ~reset
        right_action = (action == 1) & ~reset
        assert torch.allclose(n[left_action], left)
        assert torch.allclose(n[right_action], right)

        for state, action, reward, done, next, index, iw in eb:
            b = index.size(0)
            assert state.shape == (b, 1, 3)
            assert action.shape == (b,)
            assert reward.shape == (b,)
            assert done.shape == (b,)
            assert next.shape == (b, 1, 3)
            term = torch.sum(state * torch.tensor([1.0, 0.0, 1.0], device='cuda'), dim=[1, 2])
            assert torch.allclose(term, torch.zeros_like(term))
            left = torch.tensor([[1.0, 0.0, 0.0]]).cuda()
            right = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
            left_action = (action == 0)
            right_action = (action == 1)
            assert torch.allclose(next[left_action], left)
            assert torch.allclose(next[right_action], right)

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
    eb = PrioritizedExpBuffer(4, 1, batch_size, (2, 2))

    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    action = torch.tensor([1], dtype=torch.int64)
    reward = torch.tensor([1])
    done = torch.tensor([1], dtype=torch.uint8)
    reset = torch.tensor([0], dtype=torch.uint8)
    next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    eb.add(state, action, reward, done, reset, next)

    for state, action, reward, done, next, index, imp in eb:
        # assert torch.allclose(index, torch.tensor([0]))
        error = torch.tensor([0.0]).cuda()
        eb.update_td_error(index, error)
        break

    # state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    # action = torch.tensor([1], dtype=torch.int64)
    # reward = torch.tensor([1])
    # done = torch.tensor([1], dtype=torch.uint8)
    # reset = torch.tensor([0], dtype=torch.uint8)
    # next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    #
    # eb.add(state, action, reward, done, reset, next)
    #
    # for state, action, reward, done, next, index, imp in eb:
    #     #assert torch.allclose(index, torch.tensor([0, 1]))
    #     #error = torch.tensor([0.5, 0.5]).cuda()
    #     eb.update_td_error(index, error)
    #     break

    def step():
        state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        action = torch.tensor([1], dtype=torch.int64)
        reward = torch.tensor([1])
        done = torch.tensor([1], dtype=torch.uint8)
        reset = torch.tensor([0], dtype=torch.uint8)
        next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        eb.add(state, action, reward, done, reset, next)

        for state, action, reward, done, next, index, imp in eb:
            print(index)
            error = torch.rand(index.shape).cuda()
            eb.update_td_error(index, error)
            break

    step()
    step()
    step()
    step()


def test_prioritized_large():
    t_steps = 100
    ll_runs = 10
    eb = PrioritizedExpBuffer(max_timesteps=t_steps, ll_runs=ll_runs, batch_size=320, observation_shape=(1,))

    def transition(num, size):
        state = torch.tensor([[num]] * size).float()
        action = torch.tensor([num] * size, dtype=torch.int64)
        reward = torch.tensor([num] * size).float()
        done = torch.tensor([0] * size, dtype=torch.uint8)
        reset = torch.tensor([0] * size, dtype=torch.uint8)
        next = torch.tensor([[num]] * size).float()
        return state, action, reward, done, reset, next

    def step(expected_size):
        eb.add(*transition(1, ll_runs))

        for state, action, reward, done, next, index, imp in eb:
            error = torch.rand(index.shape).cuda()
            eb.update_td_error(index, error)
            print(action.size(0), expected_size)
            break

    for i in range(10, 320, 10):
        step(i)

    for _ in range(100):
        step(320)


def test_imp_sample():
    t_steps = 1000
    ll_runs = 100
    eb = PrioritizedExpBuffer(max_timesteps=t_steps, ll_runs=ll_runs, batch_size=320, observation_shape=(5, 5),
                              prioritize=False)
    peb = PrioritizedExpBuffer(max_timesteps=t_steps, ll_runs=ll_runs, batch_size=320, observation_shape=(5, 5),
                               importance_sample=True)

    env = gym.make('SimpleGrid-v3', n=ll_runs, device='cuda', max_steps=40, map_string="""
[
[E, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, E],
[S, E, E, E, E],
[T, T, T, T, T(1.0)]
]
        """)

    actions = 4
    device = 'cuda'
    discount_factor = 0.99

    td_error_buff = Uniform(0.0, 10.0).sample(eb.td_error.shape).cuda()

    def calc_td_error(state, next_state):
        with torch.no_grad():
            size = state.size(0)
            #td_error = Exponential(0.98).sample((size,)).cuda()
            td_error = Uniform(0.0, 5.0).sample((size,)).cuda()
            return td_error

    def histogram(x, title):
        num_bins = 40
        n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.xlabel('value')
        plt.ylabel('number samples')
        plt.title(title)
        plt.legend()
        plt.show()

    s = env.reset()
    for _ in range(100):
        action = torch.randint(4, (ll_runs,)).cuda()
        n, reward, done, reset, info = env.step(action)
        eb.add(s, action, reward, done, reset, n)
        peb.add(s, action, reward, done, reset, n)

        # need to make sure all "news" are updated
        for _ in range(200):
            for state, action, reward, done, next_state, index, i_w in eb:
                td_error = td_error_buff[index]
                eb.update_td_error(index, td_error)
                peb.update_td_error(index, td_error)
                break

    cum_ave = 0
    cum_n = 0
    sampled = []
    for _ in range(20):
        for state, action, reward, done, next_state, index, i_w in eb:
            td_error = td_error_buff[index]
            td_error = torch.abs(td_error)
            sampled.append(td_error)
            cum_sum = cum_ave * cum_n
            batch_sum = td_error.sum()
            cum_n = cum_n + td_error.size(0)
            cum_ave = (cum_sum + batch_sum) / cum_n

    histogram(torch.cat(sampled).cpu().numpy(), 'uniform sampling')
    del sampled
    print(cum_ave)

    p_cum_ave = 0
    iw_cum_ave = 0
    cum_n = 0
    sampled = []
    imp_sample= []
    for _ in range(20):
        for state, action, reward, done, next_state, index, i_w in peb:
            td_error = td_error_buff[index]
            td_error = torch.abs(td_error)
            sampled.append(td_error)
            imp_sample.append(td_error * i_w)
            p_cum_sum = p_cum_ave * cum_n
            iw_cum_sum = iw_cum_ave * cum_n
            p_batch_sum = td_error.sum()
            iw_batch_sum = (td_error * i_w).sum()
            cum_n = cum_n + td_error.size(0)
            p_cum_ave = (p_cum_sum + p_batch_sum) / cum_n
            iw_cum_ave = (iw_cum_sum + iw_batch_sum) / cum_n
    print(p_cum_ave)
    print(iw_cum_ave)
    histogram(torch.cat(sampled).cpu().numpy(), 'prioritized sampling')
    histogram(torch.cat(imp_sample).cpu().numpy(), 'importance sampling')
    del sampled
    del imp_sample

from train import *
import torch
import logging
import gym
import gym_duane


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.DEBUG)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def test_reset():
    env = gym.make('SimpleGrid-v2', n=1, device="cpu", map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)

    device = 'cpu'

    episode = []

    action = torch.tensor([0])

    state = env.reset()
    next_state, reward, done, info = env.step(action)

    episode.append(BatchStep(state.numpy(), action.numpy(), reward.numpy(), done.numpy(), next_state.numpy()))

    state = next_state
    next_state, reward, done, info = env.step(action)

    episode.append(BatchStep(state.numpy(), action.numpy(), reward.numpy(), done.numpy(), next_state.numpy()))

    dataset = SARSGridDataset(episode=episode)

    state, action, reward, done, reset, next_state = dataset[1]

    print(state)
    print(next_state)

    assert reset == 1


def test_bandit():
    device = 'cuda'
    ll_runs = 1000
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [T(-1.0), S, T(1.0)]
    ]
    """)

    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), 2).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, 2, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    ave_reward = 0

    while ave_reward < 0.95:
        print(f'paramters: {critic.weights.data}')
        episode, entropy, ave_reward, episodes = batch_episode(env, policy, device, max_rollout_len=50, render=False)
        logger.info(f'{Fore.GREEN}ave_reward {ave_reward} episodes {episodes} {Style.RESET_ALL}')
        policy, critic = train(episode, critic, device, optim, actions=2, epsilon=0.05, logging_freq=0)


bandit = """
    [
    [T(-1.0), S, T(1.0)]
    ]
    """


def test_bandit_deepq():
    device = 'cuda'
    ll_runs = 1000
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=bandit)

    critic = DiscreteQTable((env.height, env.width), 2).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, 2, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    rw = RunningReward(ll_runs, 2000)
    ave_reward = 0
    exp_buffer = []

    state = env.reset()

    while ave_reward < 0.95:

        state, reward, done = one_step(env, state, policy, exp_buffer)
        rw.add(reward.cpu().numpy(), done.cpu().numpy())
        logger.info(f'reward {mean(rw.recent)} epi {len(rw.recent)}')
        policy, critic = train_one(exp_buffer, critic, device, optim, actions=2, epsilon=0.05, logging_freq=0)
        if len(exp_buffer) > 100:
            exp_buffer.pop(0)

        print_qvalues(critic.weights.data)


def test_bandit_td_value():
    for i in range(1):
        ll_runs = 5
        steps = 20
        ep_s = ConstantEps(0.05)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=bandit, max_steps=5)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_value_on(env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s,
                     exp_buffer=exp_buffer, batch_size=batch_size,
                     workers=1, discount=0.8,
                     steps=steps, logging_freq=1, run_id=f'bandit_value{i}', warmup=1)

walker = """
[
[S, E, T(1.0)]
]
"""

def test_walker_td_value():
    for i in range(1):
        ll_runs = 5
        steps = 100
        ep_s = ConstantEps(0.05)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=walker, max_steps=5)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_value_on(env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s,
                     exp_buffer=exp_buffer, batch_size=batch_size,
                     workers=1, discount=0.8,
                     steps=steps, logging_freq=1, run_id=f'walker_value{i}', warmup=1)




def test_shortwalk_deepq():
    device = 'cuda'
    ll_runs = 8000
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """)

    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    critic.weights.data[0, 0, 1] = 1.0
    critic.weights.data[1, 0, 1] = -1.0
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.1).to(device)
    rw = RunningReward(ll_runs)
    ave_reward = 0
    exp_buffer = []

    state = env.reset()

    while ave_reward < 0.95:

        state, reward, done = one_step(env, state, policy, exp_buffer, render=True)
        rw.add(reward.cpu().numpy(), done.cpu().numpy())
        if len(rw.recent) > 0:
            logger.info(f'reward {mean(rw.recent)} epi {len(rw.recent)}')
        else:
            logger.info(f'reward 0 epi {len(rw.recent)}')
        rw.reset()

        policy, critic = train_one(exp_buffer, critic, device, optim, actions=actions, epsilon=0.1, logging_freq=0)
        if len(exp_buffer) > 100:
            exp_buffer.pop(0)

        print_qvalues(critic.weights.data)


lawn = """
[
[S, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, T(1.0)]
]
"""


frozen_lake = """
[
[S, E, E, E, E],
[E, E, E, E, E],
[E, E, T, E, T],
[E, E, E, E, E],
[E, E, T, E, T(1.0)]
]
"""

cliff_walk = """
[
[E, E, E, E, E],
[E, E, E, E, E],
[E, E, E, E, E],
[S, E, E, E, E],
[T, T, T, T, T(1.0)]
]
"""

def test_frozenlake_value_baseline():
    for i in range(5):
        ll_runs = 600
        steps = 10000
        ep_s = ExpExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s,
                     exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                     steps=steps, logging_freq=100, run_id=f'frozenlake_value{i}', warmup=1, lr=0.05)


def test_cliffwalk_value_baseline():
    for i in range(1):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.95, 0.05, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=cliff_walk, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
                     ll_runs=ll_runs, eps_sched=ep_s, exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                     steps=steps, logging_freq=100, run_id=f'cliffwalk_value{i}', warmup=1)


def test_frozenlake_baseline():
    for i in range(10):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.5, 0.01, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_baseline_{i}', warmup=1000)


def test_frozenlake_prioritized():
    for i in range(5):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.95, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(replay_window, batch_size, False, *env.observation_space_shape)
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_prioritized_{i}', warmup=1000)


def test_frozenlake_importance_sampled():
    for i in range(5):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, scale=0.95, bias=0.05, steps=steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(replay_window, batch_size, True, *env.observation_space_shape)
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_imp_smp_{i}', warmup=1000)



puddle_jumping = """
[
[S, E, E, E, E     , E, E, E, E, E     , E, E, E, E, E     ],
[E, E, E, E, E     , E, E, E, E, E     , E, E, E, E, E     ],
[E, E, T, E, T     , E, E, T, E, T     , E, E, T, E, T     ],
[E, E, E, E, E     , E, E, E, E, E     , E, E, E, E, E     ],
[E, E, T, E, E(1.0), E, E, T, E, E(1.0), E, E, T, E, T(1.0)]
]
"""

def test_puddlejump_baseline():
    for i in range(5):
        ll_runs = 600
        steps = 40000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=puddle_jumping, max_steps=100)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        exp_buffer = ExpBuffer(replay_window, *env.observation_space_shape)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'puddle_baseline_{i}', warmup=1000)



def test_puddlejump_importance_sampled():
    for i in range(5):
        ll_runs = 600
        steps = 40000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=puddle_jumping, max_steps=150)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 32 * ll_runs
        exp_buffer = PrioritizedExpBuffer(replay_window, batch_size, True, *env.observation_space_shape)
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'puddlejmp_imp_smp_{i}', warmup=1000)


def test_puddlejump_importance_sampled_cos():
    for i in range(1):
        ll_runs = 600
        steps = 45000
        ep_s = Cos(1000, 0.2, bias=0.00, steps=steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=puddle_jumping)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 32 * ll_runs
        exp_buffer = PrioritizedExpBuffer(replay_window, batch_size, True, *env.observation_space_shape)
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'puddlejmp_imp_smp_cos{i}', warmup=1000)


shortjump = """
[
[S, E, E, E,  E(1.0), E, E, E, E(1.0), E, E, E, E, T(1.0)]
]
"""


def test_shortjump_baseline():
    for i in range(10):
        ll_runs = 600
        steps = 40000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=shortjump)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        exp_buffer = ExpBuffer(replay_window, *env.observation_space_shape)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'short_baseline_{i}', warmup=1000)



def test_frozenlake_paper():
    for _ in range(3):
        ll_runs = 8000
        steps = 120000
        ep_s = EpsSchedule(warmup_len=10000, finish=10000)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=frozen_lake)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        run_deep_q_on(env=env, critic=critic, policy=policy, ll_runs=ll_runs, eps_sched=ep_s,
                      replay_window=replay_window, batch_size=32 * 8000,
                      workers=1,
                      steps=steps, logging_freq=100, run_id='frozenlake_paper', warmup=1000)
        # run_deep_q_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=8000 * 400, batch_size=32, workers=1,
        #               steps=6000, logging_freq=100, run_id='frozenlake_400')
        # run_deep_q_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=8000 * 200, batch_size=32, workers=1,
        #               steps=6000, logging_freq=100, run_id='frozenlake_200')
        # run_deep_q_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=8000 * 100, batch_size=32, workers=1,
        #               steps=6000, logging_freq=100, run_id='frozenlake_100')


def test_frozenlake_window_sizes():
    for _ in range(3):
        ll_runs = 600
        steps = 30000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 20
        batch_size = 32 * ll_runs

        batch_size = 8 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window, batch_size=8 * ll_runs,
                      workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)

        batch_size = 16 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window,
                      batch_size=16 * ll_runs, workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)

        batch_size = 32 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window,
                      batch_size=32 * ll_runs, workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)

        batch_size = 64 * ll_runs
        run_deep_q_on(frozen_lake, ll_runs=ll_runs, eps_sched=ep_s, replay_window=replay_window,
                      batch_size=64 * ll_runs, workers=1,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_bt_{batch_size}', warmup=1000)


def test_fake_lunar_lander():
    for i in range(2):
        ll_runs = 600
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 5
        env = gym.make('GridLunarLander-v0', n=ll_runs, device=device)
        length = sum([np.prod(shape) for shape in env.observation_space_shape])
        critic = DiscreteQTable((length,), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(replay_window, *env.observation_space_shape)
        batch_size = 16 * ll_runs
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'frozenlake_baseline_{i}', warmup=1000)


def test_shortwalk_curio():
    map = """
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """
    run_deep_q_with_curios_on(map, ll_runs=8000, epsilon=0.1, replay_window=20, batch_size=10000, workers=12)


def test_frozen_lake_curio():
    run_deep_q_with_curios_on(frozen_lake, ll_runs=8000, epsilon=0.1, replay_window=20, batch_size=10000, workers=12)


def test_fake_lunar_lander_curio():
    map = """
    [
    [T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, S, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(-1.0)],
    [T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0), T(-1.0)]
    ]
    """
    run_deep_q_with_curios_on(map, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000, workers=12)


anthill = """
[
[L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, T(1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, S, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, T(1.0), E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, T(1.0), E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, E, L],
[L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L]
]
"""


def test_anthill_importance_sampled():
    for i in range(10):
        ll_runs = 600
        steps = 40000
        ep_s = ExpExponentialDecay(steps // 10, 0.05, steps)
        replay_window = ll_runs * steps // 10
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string=anthill)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(replay_window, batch_size, True, *env.observation_space_shape)
        run_deep_q_on(env=env, critic=critic, policy=policy,
                      ll_runs=ll_runs, eps_sched=ep_s,
                      exp_buffer=exp_buffer, batch_size=batch_size,
                      workers=1, discount=0.8,
                      steps=steps, logging_freq=100, run_id=f'anthill_imp_smp_{i}', warmup=1000)


def test_anthill():
    run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000, workers=12,
                  run_id='anthill')


def test_anthill_curio():
    run_deep_q_with_curios_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000, workers=12,
                              run_id='anthill_curio')


def test_anthill_curio_batch():
    steps = 20000
    for _ in range(2):
        run_deep_q_with_curios_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                                  run_id='anthill_curio_lr_05', lr=0.05, steps=steps)
        run_deep_q_with_curios_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                                  run_id='anthill_curio_lr_01', lr=0.01, steps=steps)

        run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                      run_id='anthill_lr_10', lr=0.05, steps=steps)
        run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                      run_id='anthill_lr_05', lr=0.05, steps=steps)
        run_deep_q_on(anthill, ll_runs=8000, epsilon=0.1, replay_window=100, batch_size=16000,
                      run_id='anthill_lr_01', lr=0.05, steps=steps)


def test_grid_walk():
    device = 'cuda'
    ll_runs = 8000
    actions = 4
    env = gym.make('SimpleGrid-v2', n=ll_runs, device=device, map_string="""
    [
    [S, E, E],
    [E, E, T(1.0)]
    ]
    """)

    obs = env.reset()
    critic = DiscreteQTable((env.height, env.width), actions).to(device)
    optim = torch.optim.SGD(critic.parameters(), lr=0.1)
    policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.05).to(device)
    ave_reward = 0
    episodes = 0

    while episodes < ll_runs:
        print(f'paramters: {critic.weights.data}')
        print_qvalues(critic.weights.data)
        episode, entropy, ave_reward, episodes = batch_episode(env, policy, device, max_rollout_len=4, render=False)
        logger.info(f'{Fore.GREEN}ave_reward {ave_reward} episodes {episodes} {Style.RESET_ALL}')
        policy, critic = train(episode, critic, device, optim, actions=actions, epsilon=0.05, logging_freq=0)


v_table = torch.tensor([
    [0.0137, 0.0122, 0.0140, 0.0185, 0.0216],
    [0.0149, 0.0120, 0.0036, 0.0184, 0.0083],
    [0.0178, 0.0049, 0.9256, 0.0218, 0.8017],
    [0.0222, 0.0259, 0.0262, 0.6383, 0.5538],
    [0.0180, 0.0096, 0.7984, 0.7097, 0.6495]], device='cuda:0')


def test_frozenlake_value_debug():
    for i in range(1):
        ll_runs = 1
        steps = 20000
        ep_s = ExpExponentialDecay(steps // 10, 0.5, 0.05, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        critic.weights.data = v_table
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.0).to(device)
        batch_size = 16 * ll_runs
        join = OneObsToState()

        done = torch.tensor([0], dtype=torch.uint8)
        state = env.reset()
        env.render()
        print("")
        while not done.all():
            lookahead_state, lookahead_reward, lookahead_done, info = env.lookahead()
            action_dist = policy(join(lookahead_state), lookahead_reward, lookahead_done)
            action = action_dist.sample()
            n, reward, done, reset, info = env.step(action)
            env.render()
            print("")


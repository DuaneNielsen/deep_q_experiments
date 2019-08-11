from algos.schedules import ConstantSched, ExponentialDecay
from algos.td_q import one_step, train_one
from algos.td_value import one_step_value, train_one_value
import monitoring
import gym
from gym_duane.wrappers import BatchTensor, LookAhead, Reset, StackedObs, Normalize

logger = monitoring.getLogger(__name__)
monitoring.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=monitoring.DEBUG)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


bandit = """
    [
    [T(-1.0), S, T(1.0)]
    ]
    """


def test_bandit_q():
    for i in range(1):
        ll_runs = 5
        steps = 200
        ep_s = ConstantSched(0.05)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=bandit, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='parallel',
               steps=steps, logging_freq=10, run_id=f'bandit_q_{i}', warmup_steps=0)


def test_bandit_deep_q():
    for i in range(1):
        ll_runs = 5
        steps = 200
        ep_s = ConstantSched(0.05)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=bandit, max_steps=40)
        critic = FixupQ((env.height, env.width), actions, 4).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='parallel',
               steps=steps, logging_freq=10, run_id=f'bandit_deepq_{i}', warmup_steps=0)


walker = """
[
[S, E, T(1.0)]
]
"""


def test_walker_deepq():
    for i in range(1):
        ll_runs = 5
        steps = 300
        ep_s = ConstantSched(0.05)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=walker, max_steps=40)
        critic = FixupQ((env.height, env.width), actions, 4).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='parallel',
               steps=steps, logging_freq=10, run_id=f'walker_deepq_{i}', warmup_steps=0)

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


def test_lawn_deepq_baseline():
    for i in range(3):
        ll_runs = 1
        steps = 10000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=lawn, max_steps=40)
        critic = FixupQ((env.height, env.width), actions, 4).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='parallel',
               steps=steps, logging_freq=100, run_id=f'lawn_deeoq_{i}', warmup_steps=10)

def test_cliffwalk_deepq_baseline():
    for i in range(3):
        ll_runs = 1
        steps = 10000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.1, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=cliff_walk, max_steps=40)
        critic = FixupQ((env.height, env.width), actions, 4).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, lr_sched=lr_s, rendermode='parallel',
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
               steps=steps, logging_freq=100, run_id=f'cliffwalk_deeoq_{i}', warmup_steps=10)


def test_frozenlake_deepq_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 15000
        ep_s = ExponentialDecay(half_life=steps // 7.0, scale=0.3, bias=0.02, steps=steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40, reward_per_timestep=-0.01)
        critic = FixupQ((env.height, env.width), actions, 4).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps // 8, ll_runs=ll_runs, batch_size=batch_size,
                               observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, lr_sched=lr_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=100, run_id=f'frozenlake_deepq_{i}', warmup_steps=10)


def test_frozenlake_deepq_grid_search():
    for _ in range(3):
        for discount in np.arange(0.84, 1.0, 0.04):
            ll_runs = 600
            batch_size = 16 * ll_runs
            steps = 15000
            ep_s = ExponentialDecay(half_life=steps // 7.0, scale=0.4, bias=0.02, steps=steps)
            lr_s = ConstantSched(0.05)
            device = 'cuda'
            actions = 4
            env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40, reward_per_timestep=-0.01)
            critic = FixupQ((env.height, env.width), actions, 4).to(device)
            policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
            exp_buffer = ExpBuffer(max_timesteps=steps // 8, ll_runs=ll_runs, batch_size=batch_size,
                                   observation_shape=env.observation_space_shape)
            run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
                   ll_runs=ll_runs, eps_sched=ep_s, lr_sched=lr_s,
                   exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                   steps=steps, logging_freq=100, run_id=f'frozenlake_deepq_discount_{discount}', warmup_steps=10)


def test_frozenlake_value_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 15000
        ep_s = ExponentialDecay(steps // 5, 0.4, 0.02, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40, reward_per_timestep=-0.01)
        critic = FixupV((env.height, env.width), 4).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//8, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=100, run_id=f'frozenlake_value{i}', warmup_steps=0, lr_sched=lr_s)


def test_frozenlake_value_grid():
    for i in range(3):
        for step_penalty in np.arange(0.002, 0.1, 0.002):
            eps = 0.4
            steps = 8000
            ll_runs = 600
            ep_s = ExponentialDecay(steps // 5, eps, 0.02, steps)
            lr_s = ConstantSched(0.05)
            device = 'cuda'
            actions = 2
            env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40, reward_per_timestep=-step_penalty)
            critic = FixupV((env.height, env.width), 4).to(device)
            policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
            batch_size = 16 * ll_runs
            exp_buffer = ExpBuffer(max_timesteps=steps//8, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

            run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
                   ll_runs=ll_runs, eps_sched=ep_s,
                   exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
                   steps=steps, logging_freq=100, run_id=f'frozenlake_step_{step_penalty}', warmup_steps=10, lr_sched=lr_s)


def test_lunar_lander_value_baseline():
    for i in range(10):
        epochs = 100
        steps_per_epoch = 50000
        steps = epochs * steps_per_epoch
        #ep_s = ExponentialDecay(steps // 5, 0.2, 0.02, steps)
        ep_s = ConstantSched(0.05)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 4
        ll_runs = 1
        batch_size = 640 * ll_runs
        resnet_blocks = 4

        mean = np.array([-0.0115, 0.9516, -0.0119, -0.6656, -0.0055, -0.0090, 0.0275, 0.0246])
        std = np.array([0.3110, 0.4772, 0.6319, 0.4803, 0.5030, 0.5246, 0.1635, 0.1550])

        env = gym.make('LunarLander-v2')
        env = Normalize(env, mean, std, 0.0, 200.0)
        env = StackedObs(env)
        env = LookAhead(env)
        env = Reset(env)
        env = BatchTensor(env, device='cuda')

        critic = FixupV((4, 8), resnet_blocks).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps_per_epoch // 6, ll_runs=ll_runs, batch_size=batch_size,
                               observation_shape=(4, 8))

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=200, run_id=f'lunar_lander_value_baseline', warmup_steps=10, lr_sched=lr_s)


def test_cartpole_baseline():
    for i in range(10):
        epochs = 100
        steps_per_epoch = 50000
        steps = epochs * steps_per_epoch
        #ep_s = ExponentialDecay(steps // 5, 0.2, 0.02, steps)
        ep_s = ConstantSched(0.05)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 2
        ll_runs = 1
        batch_size = 640 * ll_runs
        obs_shape = (4, 4)

        mean = np.array([0.0002, 0.0050, -0.0010, -0.0098])
        std = np.array([0.0944, 0.5519, 0.1007, 0.8333])
        r_mean = 0.95713
        r_std = 0.2025649856854196

        env = gym.make('CartPole-v0')
        env = Normalize(env, mean, std, r_mean, r_std)
        env = StackedObs(env)
        env = LookAhead(env)
        env = Reset(env)
        env = BatchTensor(env, device='cuda')

        critic = DiscreteQTable(obs_shape, actions).to(device)
        start_policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps_per_epoch // 6, ll_runs=ll_runs, batch_size=batch_size,
                               observation_shape=obs_shape)

        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=start_policy,
               ll_runs=ll_runs, eps_sched=ep_s, actions=actions,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=200, run_id=f'cartpole_q_baseline', warmup_steps=10, lr_sched=lr_s)

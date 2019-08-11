from algos.schedules import ConstantSched, ExponentialDecay
from algos.td_q import one_step, train_one
from algos.td_value import one_step_value, train_one_value
import monitoring
import gym

logger = monitoring.getLogger(__name__)
monitoring.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=monitoring.DEBUG)

#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40, reward_per_timestep=-0.01)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=100, run_id=f'frozenlake_value{i}', warmup_steps=10, lr_sched=lr_s)


def test_cliffwalk_value_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=cliff_walk, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=100, run_id=f'cliffwalk_value{i}', warmup_steps=10)


def test_frozenlake_q_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40, reward_per_timestep=-0.01)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
               steps=steps, logging_freq=100, run_id=f'frozenlake_q_{i}', warmup_steps=10)


def test_cliffwalk_q_baseline():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=cliff_walk, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size, observation_shape=env.observation_space_shape)
        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
               steps=steps, logging_freq=100, run_id=f'cliffwalk_q_{i}', warmup_steps=10)


def test_frozenlake_value_prioritized():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=False)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=100, run_id=f'frozenlake_value_prioritized_{i}', warmup_steps=10, lr=0.05)


def test_frozenlake_value_importance_sampled():
    for i in range(3):
        ll_runs = 600
        steps = 20000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 2
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteVTable((env.height, env.width)).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=0.5).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=True)

        run_on(stepper=one_step_value, learner=train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.99,
               steps=steps, logging_freq=100, run_id=f'frozenlake_value__b_imps_{i}', warmup_steps=10, lr=0.05)


def test_frozenlake_q_prioritized():
    for i in range(3):
        ll_runs = 600
        steps = 5000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=False)

        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
               steps=steps, logging_freq=100, run_id=f'frozenlake_q_prioritized_{i}', warmup_steps=10)


def test_frozenlake_q_importance_sampled():
    for i in range(3):
        ll_runs = 600
        steps = 20000
        ep_s = ExponentialDecay(steps // 10, 0.4, 0.02, steps)
        device = 'cuda'
        actions = 4
        env = gym.make('SimpleGrid-v3', n=ll_runs, device=device, map_string=frozen_lake, max_steps=40)
        critic = DiscreteQTable((env.height, env.width), actions).to(device)
        policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)
        batch_size = 16 * ll_runs
        exp_buffer = PrioritizedExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs, batch_size=batch_size,
                                          observation_shape=env.observation_space_shape, importance_sample=True)

        run_on(stepper=one_step, learner=train_one, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8,
               steps=steps, logging_freq=100, run_id=f'frozenlake_q_imps{i}', warmup_steps=10)


from algos.schedules import ConstantSched, ExponentialDecay
from algos.td_q import Q, FastPlot
import algos.td_value as td_value
import gym
from gym_duane.wrappers import *
from data import ExpBuffer, OneObsToState
from models import *
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s-%(module)s-%(message)s', level=logging.DEBUG)
logging.getLogger('font_manager').setLevel(logging.INFO)


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def norm_f5(obs):
    return (obs + 5.0) / 10.0


def test_F5_deep_q():
    for i in range(10):
        ll_runs = 1
        steps = 1000
        ep_s = ExponentialDecay(steps/15, 0.3, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (1,)
        batch_size = 16 * ll_runs

        env = gym.make('F5-v0')
        env = TimeLimit(env, max_episode_steps=50)
        env = NormalizeFunctional(env, obs_f=norm_f5, reward_f=normalize_reward)
        env = Reset(env)
        env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        critic = FixupQ(obs_shape, actions, 4).to(device)
        behaviour_policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist).to(device)
        greedy_policy = QPolicy(critic, actions, GreedyDist).to(device)
        exp_buffer = ExpBuffer(max_timesteps=steps // 10, ll_runs=ll_runs, batch_size=batch_size,
                               observation_shape=obs_shape)
        algo = Q(env, critic, behaviour_policy, greedy_policy, exp_buffer, device=device, plot=FastPlot(actions))
        algo.run(run_id='base_line', steps=steps, batch_size=batch_size, discount_factor=0.95, lr_sched=lr_s,
                 eps_sched=ep_s, logging_freq=10)

def test_F5_deep_q_proj():
    for i in range(10):
        ll_runs = 1
        steps = 1000
        ep_s = ExponentialDecay(steps/15, 0.3, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (1,)
        batch_size = 16 * ll_runs

        env = gym.make('F5-v0')
        env = TimeLimit(env, max_episode_steps=50)
        env = NormalizeFunctional(env, obs_f=norm_f5, reward_f=normalize_reward)
        env = Reset(env)
        env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        #critic = ProjFixupQ(obs_shape, actions, 20, 4).to(device)
        critic = EnsembleQ(obs_shape, actions, hidden=20, blocks=4).to(device)
        behaviour_policy = QPolicy(critic, actions, EpsilonGreedyProperDiscreteDist).to(device)
        greedy_policy = QPolicy(critic, actions, GreedyDist).to(device)
        exp_buffer = ExpBuffer(max_timesteps=steps // 10, ll_runs=ll_runs, batch_size=batch_size,
                               observation_shape=obs_shape)
        algo = Q(env, critic, behaviour_policy, greedy_policy, exp_buffer, device=device, plot=FastPlot(actions))
        algo.run(run_id='base_line', steps=steps, batch_size=batch_size, discount_factor=0.95, lr_sched=lr_s,
                 eps_sched=ep_s, logging_freq=10)



def normalize_obs(obs):
    return (obs + 4.0) / 8.0

def normalize_reward(reward):
    return reward

def test_F3_value():

    for i in range(3):
        ll_runs = 1
        steps = 10000
        ep_s = ExponentialDecay(steps/16, 0.5, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (1,)
        batch_size = 32

        env = gym.make('F3-v0')
        #env = RewardPerStep(env, reward_per_step=-0.01)
        env = TimeLimit(env, max_episode_steps=20)
        env = NormalizeFunctional(env, obs_f=normalize_obs, reward_f=normalize_reward)
        env = LookAhead(env)
        env = Reset(env)
        #env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        #critic = FixupV(obs_shape, 4).to(device)
        critic = SimpleV(obs_shape).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs,
                               batch_size=batch_size, observation_shape=obs_shape)

        stepper = td_value.Stepper(env, OneObsToState(), exp_buffer)

        run_on(stepper=stepper, learner=td_value.train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, actions=actions,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='episodic',
               steps=steps, logging_freq=1, run_id=f'f5_value_{i}', warmup_steps=0)

def test_F3_oh_value():

    for i in range(3):
        ll_runs = 1
        steps = 10000
        ep_s = ExponentialDecay(steps/16, 0.5, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (1,)
        batch_size = 32

        env = gym.make('F3-v0')
        #env = RewardPerStep(env, reward_per_step=-0.01)
        env = TimeLimit(env, max_episode_steps=20)
        env = NormalizeFunctional(env, obs_f=normalize_obs, reward_f=normalize_reward)
        env = LookAhead(env)
        env = Reset(env)
        #env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        #critic = FixupV(obs_shape, 4).to(device)
        critic = OneHotV(obs_shape, 12).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs,
                               batch_size=batch_size, observation_shape=obs_shape)

        stepper = td_value.Stepper(env, OneObsToState(), exp_buffer)

        run_on(stepper=stepper, learner=td_value.train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, actions=actions,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='episodic',
               steps=steps, logging_freq=1, run_id=f'f5_value_{i}', warmup_steps=0)

def test_F3_fixup_value():

    for i in range(3):
        ll_runs = 1
        steps = 10000
        ep_s = ExponentialDecay(steps/16, 0.5, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (1,)
        batch_size = 32

        env = gym.make('F3-v0')
        #env = RewardPerStep(env, reward_per_step=-0.01)
        env = TimeLimit(env, max_episode_steps=20)
        env = NormalizeFunctional(env, obs_f=normalize_obs, reward_f=normalize_reward)
        env = LookAhead(env)
        env = Reset(env)
        #env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        critic = DeepProjectionV(obs_shape, 8, 6).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs,
                               batch_size=batch_size, observation_shape=obs_shape)

        stepper = td_value.Stepper(env, OneObsToState(), exp_buffer)

        run_on(stepper=stepper, learner=td_value.train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, actions=actions,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='episodic',
               steps=steps, logging_freq=1, run_id=f'f5_value_{i}', warmup_steps=0)


def normalize_F5_obs(obs):
    return (obs + 6.0) / 12.0


def test_F5_fixup_value():

    for i in range(3):
        ll_runs = 1
        steps = 100000
        ep_s = ExponentialDecay(steps/160, 0.5, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (1,)
        batch_size = 32

        env = gym.make('F5-v0')
        #env = RewardPerStep(env, reward_per_step=-0.01)
        env = TimeLimit(env, max_episode_steps=50)
        env = NormalizeFunctional(env, obs_f=normalize_F5_obs, reward_f=normalize_reward)
        env = LookAhead(env)
        env = Reset(env)
        #env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        critic = DeepProjectionV(obs_shape, 8, 6).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs,
                               batch_size=batch_size, observation_shape=obs_shape)

        stepper = td_value.Stepper(env, OneObsToState(), exp_buffer)

        run_on(stepper=stepper, learner=td_value.train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, actions=actions,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='episodic',
               steps=steps, logging_freq=1, run_id=f'f5_value_{i}', warmup_steps=0)

mean = np.array([-0.0115, 0.9516, -0.0119, -0.6656, -0.0055, -0.0090, 0.0275, 0.0246])
std = np.array([0.3110, 0.4772, 0.6319, 0.4803, 0.5030, 0.5246, 0.1635, 0.1550])
min = np.array([-1.0089, -0.4125, -2.0318, -1.9638, -3.6345, -6.7558, 0.0000, 0.0000])
max = np.array([1.0165, 1.6740, 2.1364, 0.5349, 3.5584, 5.9297, 1.0000, 1.0000])


def normalize_lunar_lander(obs):
    return (obs - mean) /std


def normalize_min_max(obs):
    return (obs - min) / (max - min)


def normalize_lunar_lander_reward(reward):
       return reward / 150.0



def test_lunar_lander():

    for i in range(3):
        ll_runs = 1
        steps = 1000000
        ep_s = ExponentialDecay(steps/160, 0.45, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (8,)
        batch_size = 32


        env = gym.make('LunarLander-v2')
        #env = RewardPerStep(env, reward_per_step=-0.01)
        #env = TimeLimit(env, max_episode_steps=50)
        env = NormalizeFunctional(env, obs_f=normalize_min_max, reward_f=normalize_lunar_lander_reward)
        env = LookAhead(env)
        env = Reset(env)
        #env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        critic = DeepProjectionV(obs_shape, 100, 6).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs,
                               batch_size=batch_size, observation_shape=obs_shape)

        stepper = td_value.Stepper(env, OneObsToState(), exp_buffer)

        run_on(stepper=stepper, learner=td_value.train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, actions=actions,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='episodic',
               steps=steps, logging_freq=5, run_id=f'f5_value_{i}', warmup_steps=0)


def test_cartpole():

    for i in range(3):
        ll_runs = 1
        steps = 1000000
        ep_s = ExponentialDecay(steps/160, 0.45, 0.05, steps)
        lr_s = ConstantSched(0.05)
        device = 'cuda'
        actions = 3
        obs_shape = (8,)
        batch_size = 32


        env = gym.make('CartPole-v2')
        #env = RewardPerStep(env, reward_per_step=-0.01)
        #env = TimeLimit(env, max_episode_steps=50)
        env = NormalizeFunctional(env, obs_f=normalize_min_max, reward_f=normalize_lunar_lander_reward)
        env = LookAhead(env)
        env = Reset(env)
        #env = Monitor(env)
        env = BatchTensor(env, device='cuda')

        critic = DeepProjectionV(obs_shape, 100, 6).to(device)
        policy = VPolicy(critic, actions, EpsilonGreedyProperDiscreteDist, epsilon=1.0).to(device)

        exp_buffer = ExpBuffer(max_timesteps=steps//10, ll_runs=ll_runs,
                               batch_size=batch_size, observation_shape=obs_shape)

        stepper = td_value.Stepper(env, OneObsToState(), exp_buffer)

        run_on(stepper=stepper, learner=td_value.train_one_value, env=env, critic=critic, policy=policy,
               ll_runs=ll_runs, eps_sched=ep_s, actions=actions,
               exp_buffer=exp_buffer, batch_size=batch_size, discount=0.8, lr_sched=lr_s, rendermode='episodic',
               steps=steps, logging_freq=5, run_id=f'f5_value_{i}', warmup_steps=0)
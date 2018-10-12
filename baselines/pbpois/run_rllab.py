#!/usr/bin/env python3
from baselines import logger
import time, os, gym, logging
import numpy as np

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines.policy.neuron_hyperpolicy import MultiPeMlpPolicy
from baselines.policy.weight_hyperpolicy import PeMlpPolicy
from baselines.pbpois import pbpois, nbpois
from baselines.pbpois.parallel_sampler import ParallelSampler
from baselines.envs.rllab_wrappers import Rllab2GymWrapper

def rllab_env_from_name(env):
    if env == 'swimmer':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        return SwimmerEnv
    elif env == 'ant':
        from rllab.envs.mujoco.ant_env import AntEnv
        return AntEnv
    elif env == 'half-cheetah':
        from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
        return HalfCheetahEnv
    elif env == 'hopper':
        from rllab.envs.mujoco.hopper_env import HopperEnv
        return HopperEnv
    elif env == 'simple-humanoid':
        from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
        return SimpleHumanoidEnv
    elif env == 'full-humanoid':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        return HumanoidEnv
    elif env == 'walker':
        from rllab.envs.mujoco.walker2d_env import Walker2DEnv
        return Walker2DEnv
    elif env == 'cartpole':
        from rllab.envs.box2d.cartpole_env import CartpoleEnv
        return CartpoleEnv
    elif env == 'mountain-car':
        from rllab.envs.box2d.mountain_car_env import MountainCarEnv
        return MountainCarEnv
    elif env == 'inverted-pendulum':
        from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv as InvertedPendulumEnv
        return InvertedPendulumEnv
    elif env == 'acrobot':
        from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv as AcrobotEnv
        return AcrobotEnv
    elif env == 'inverted-double-pendulum':
        from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
        return InvertedDoublePendulumEnv
    else:
        raise Exception('Unrecognized rllab environment.')

def train(env, max_iters, num_episodes, horizon, iw_norm, bound, delta, gamma, seed, policy, max_offline_iters, aggregate, adaptive_batch, njobs=1):
    
    # Create the environment
    env_rllab_class = rllab_env_from_name(env)
    def make_env():
        env_rllab = env_rllab_class()
        _env = Rllab2GymWrapper(env_rllab)
        return _env

    # Create the policy
    if policy == 'linear':
        hid_layers = []
    elif policy == 'nn':
        hid_layers = [100, 50, 25]

    
    if aggregate=='none':
        learner = pbpois
        PolicyClass = PeMlpPolicy
    elif aggregate=='neuron':
        learner = nbpois
        PolicyClass = MultiPeMlpPolicy
    else:
        print("Unknown aggregation method, defaulting to none")
        learner = pbpois
        PolicyClass = PeMlpPolicy
        
    make_policy = lambda name, observation_space, action_space: PolicyClass(name,
                      observation_space,
                      action_space,
                      hid_layers,
                      use_bias=True,
                      seed=seed)

    sampler = ParallelSampler(make_env, make_policy, gamma, horizon, np.ravel, num_episodes, njobs, seed)

    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)
    
    
    learner.learn(
          make_env, 
          make_policy,
          sampler,
          gamma=gamma,
          n_episodes=num_episodes,
          horizon=horizon,
          max_iters=max_iters,
          verbose=1,
          feature_fun=np.ravel,
          iw_norm=iw_norm,
          bound = bound,
          max_offline_iters=max_offline_iters,
          delta=delta,
          center_return=False,
          line_search_type='parabola',
          adaptive_batch=adaptive_batch)

    sampler.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_norm', type=str, default='sn')
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--aggregate', type=str, default='none')
    parser.add_argument('--adaptive_batch', type=int, default=0)
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--policy', type=str, default='linear')
    parser.add_argument('--max_offline_iters', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='logs', format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)
    train(env=args.env,
          max_iters=args.max_iters,
          num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_norm=args.iw_norm,
          bound=args.bound,
          delta=args.delta,
          gamma=args.gamma,
          seed=args.seed,
          policy=args.policy,
          max_offline_iters=args.max_offline_iters,
          njobs=args.njobs,
          aggregate=args.aggregate,
          adaptive_batch=args.adaptive_batch)

if __name__ == '__main__':
    main()
    
    

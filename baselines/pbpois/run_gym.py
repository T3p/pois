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

def train(env, max_iters, num_episodes, horizon, iw_norm, bound, delta, gamma, seed, policy, max_offline_iters, aggregate, adaptive_batch, njobs=1):
    
    # Create the environment
    def make_env():
        env_gym = gym.make(env).unwrapped
        return env_gym

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
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
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
    
    

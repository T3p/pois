#!/usr/bin/env python3
import argparse
from baselines import bench, logger
import time, os, gym, logging
import tensorflow as tf
import numpy as np

import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.policy.mlp_policy import MlpPolicy
from baselines.pois2 import pois2

def train(env, max_iters, num_episodes, horizon, iw_method, iw_norm, natural, bound, delta, gamma, seed, policy, max_offline_iters, njobs=1):

    # Declare env and created the vectorized env
    def make_env(seed=0):
        def _thunk():
            _env = gym.make(env)
            _env.seed(seed)
            return _env
        return _thunk
    parallel_env = SubprocVecEnv([make_env(i + seed) for i in range(njobs)], terminating=True)

    # Create the policy
    if policy == 'linear':
        hid_size = num_hid_layers = 0
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3

    def make_policy(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=hid_size, num_hid_layers=num_hid_layers, gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         hidden_W_init=tf.contrib.layers.xavier_initializer(),
                         output_W_init=tf.contrib.layers.xavier_initializer())

    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs
    sess = U.make_session(affinity)
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.WARN)

    pois2.learn(parallel_env, make_policy, n_episodes=num_episodes, max_iters=max_iters,
               horizon=horizon, gamma=gamma, delta=delta, use_natural_gradient=natural,
               iw_method=iw_method, iw_norm=iw_norm, bound=bound, save_weights=True,
               center_return=True, render_after=None, max_offline_iters=max_offline_iters,)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='cartpole')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--file_name', type=str, default='progress')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--njobs', type=int, default=2)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--max_offline_iters', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()
    assert args.njobs > 1, "Njobs must be more than 1, use POIS1 otherwise."
    # Configure logging
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name
    logger.configure(dir='logs', format_strs=['stdout', 'csv'], file_name=file_name)
    # Start training
    train(env=args.env,
          max_iters=args.max_iters,
          num_episodes=args.num_episodes,
          horizon=args.horizon,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          natural=args.natural,
          bound=args.bound,
          delta=args.delta,
          gamma=args.gamma,
          seed=args.seed,
          policy=args.policy,
          max_offline_iters=args.max_offline_iters,
          njobs=args.njobs)


if __name__ == '__main__':
    main()

import sys
sys.path.append('/home/alberto/rllab')

import rllab
from rllab.envs.normalized_env import normalize
import gym
import gym.spaces

def convert_rllab_space(space):
    if isinstance(space, rllab.spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high)
    elif isinstance(space, rllab.spaces.Discrete):
        return gym.spaces.Discrete(n=space._n)
    elif isinstance(space, rllab.spaces.Tuple):
        return gym.spaces.Tuple([convert_rllab_space(x) for x in space._components])
    else:
        raise NotImplementedError

class Rllab2GymWrapper(gym.Env):

    def __init__(self, rllab_env):
        self.rllab_env = normalize(rllab_env)
        self.observation_space = convert_rllab_space(rllab_env.observation_space)
        self.action_space = convert_rllab_space(rllab_env.action_space)
        self.seed()
        self.reset()

    def step(self, action):
        res = self.rllab_env.step(action)
        return tuple(res)

    def reset(self):
        new_state = self.rllab_env.reset()
        return new_state

    def seed(self, seed=0):
        pass



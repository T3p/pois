from multiprocessing import Process, Queue, Event
import os
import baselines.common.tf_util as U
import time
from mpi4py import MPI
from baselines.common import set_global_seeds
import tensorflow as tf
import numpy as np

def traj_segment_function(env, pol, gamma, task_horizon, feature_fun, batch_size):
    '''
    Collects trajectories
    '''
    theta = pol.resample()
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_disc_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    disc_ep_rets = []
    ep_lens = []
    actor_params = []

    # Initialize history arrays
    i = 0
    j = 0
    samples_to_get = task_horizon*batch_size
    tot_samples = 0
    while True:
        ac = pol.act(ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        #if t > 0 and t % horizon == 0:
        if tot_samples>0 and tot_samples % samples_to_get == 0:
            return {"rets" : ep_rets, "disc_rets": disc_ep_rets, "lens" : ep_lens,
                    "actor_params": actor_params}

        ob, rew, new, _ = env.step(ac)
        ob = feature_fun(ob) if feature_fun else ob

        cur_ep_ret += rew
        cur_disc_ep_ret += rew * gamma**cur_ep_len
        cur_ep_len += 1
        tot_samples+=1
        
        j += 1
        if new or j == task_horizon or (tot_samples>0 and tot_samples % samples_to_get == 0):
            new = True
            env.done = True

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            disc_ep_rets.append(cur_disc_ep_ret)
            actor_params.append(np.array(theta))

            cur_ep_ret = 0
            cur_disc_ep_ret = 0
            cur_ep_len = 0
            theta = pol.resample()
            ob = env.reset()

            next_t = (i+1) * task_horizon

            t = next_t - 1
            i += 1
            j = 0
        t += 1


class Worker(Process):
    '''
    A worker is an independent process with its own environment and policy instantiated locally
    after being created. It ***must*** be runned before creating any tensorflow session!
    '''

    def __init__(self, output, input, event, make_env, make_pi, traj_segment_generator, seed):
        super(Worker, self).__init__()
        self.output = output
        self.input = input
        self.make_env = make_env
        self.make_pi = make_pi
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed

    def run(self):
        sess = U.single_threaded_session()
        sess.__enter__()

        env = self.make_env()
        workerseed = self.seed + 10000 * (MPI.COMM_WORLD.Get_rank() + 1)
        set_global_seeds(workerseed)
        env.seed(workerseed)

        pi = self.make_pi('pi%s' % os.getpid(), env.observation_space, env.action_space)
        print('Worker %s - Running with seed %s' % (os.getpid(), workerseed))

        while True:
            self.event.wait()
            self.event.clear()
            command, weights = self.input.get()
            if command == 'collect':
                #print('Worker %s - Collecting...' % os.getpid())
                pi.set_params(weights)
                samples = self.traj_segment_generator(env, pi)
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                print('Worker %s - Exiting...' % os.getpid())
                env.close()
                sess.close()
                break

class ParallelSampler(object):

    def __init__(self, env_maker, pol_maker, gamma, task_horizon, feature_fun, batch_size, n_workers=-1, seed=0):
        affinity = len(os.sched_getaffinity(0))
        if n_workers == -1:
            self.n_workers = affinity
        else:
            self.n_workers = min(n_workers, affinity)

        print('Using %s CPUs' % self.n_workers)

        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        n_episodes_per_process = batch_size // self.n_workers
        remainder = batch_size % self.n_workers

        f = lambda env, pol: traj_segment_function(env, pol, gamma, task_horizon, feature_fun, n_episodes_per_process)
        f_rem = lambda env, pol: traj_segment_function(env, pol, gamma, task_horizon, feature_fun, n_episodes_per_process+1)
        fun = [f] * (self.n_workers - remainder) + [f_rem] * remainder
        self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i], env_maker, pol_maker, fun[i], seed + i) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()


    def collect(self, weights):
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', weights))

        for e in self.events:
            e.set()

        sample_batches = []
        for i in range(self.n_workers):
            pid, samples = self.output_queue.get()
            sample_batches.append(samples)

        return self._merge_sample_batches(sample_batches)

    def _merge_sample_batches(self, sample_batches):
        list_fields = ['rets', 'disc_rets', 'lens', 'actor_params']

        new_dict = list(zip(list_fields,map(lambda f: sample_batches[0][f], list_fields)))
        new_dict = dict(new_dict)

        for batch in sample_batches[1:]:
            for f in list_fields:
                new_dict[f].extend(batch[f])

        return new_dict


    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()

        for w in self.workers:
            w.join()


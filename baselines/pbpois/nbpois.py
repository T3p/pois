#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:13:18 2018

@author: matteo
"""
"""References
    PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
        control." International Conference on Artificial Neural Networks. Springer,
        Berlin, Heidelberg, 2008.
"""

import numpy as np
from baselines import logger
import warnings
from contextlib import contextmanager
import time
from baselines.common import colorize

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))

def eval_trajectory(env, pol, gamma, horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t<horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ret += r
        disc_ret += gamma**t * r
        t+=1
        
    return ret, disc_ret, t

#BINARY line search
def line_search_binary(pol, newpol, actor_params, rets, alpha, natgrad, 
                normalize=True,
                use_rmax=True,
                use_renyi=True,
                max_search_ite=30, rmax=None, delta=0.2, reassign=None):
    rho_init = newpol.eval_params()
    bound_init = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
    n_bounds = len(bound_init)
    low = np.zeros(n_bounds)
    high = np.nan * np.ones(n_bounds)

    #old_delta_bound = 0.
    rho_opt = rho_init
    i_opt = 0.
    delta_bound_opt = np.zeros(n_bounds)
    epsilon_opt = np.zeros(n_bounds)
    epsilon = np.ones(n_bounds)

    if max_search_ite<=0:
        rho = rho_init + alpha*natgrad
        newpol.set_params(rho)
        delta_bound = newpol.eval_bound(actor_params, rets, pol, rmax,
                                      normalize, use_rmax, use_renyi, delta) - bound_init
        return rho, np.ones(len(epsilon)), delta_bound, 0   

    for i in range(max_search_ite):
        rho = rho_init + reassign(epsilon) * natgrad * alpha
        newpol.set_params(rho)
        bound = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
        delta_bound = bound - bound_init        
        cond = np.logical_or(delta_bound<=delta_bound_opt, np.isnan(bound))
        cond = np.logical_not(cond)
        if np.any(np.isnan(bound)):
            warnings.warn('Got NaN bound value')
        delta_bound = np.where(np.isnan(delta_bound), -np.inf*np.ones(n_bounds), delta_bound)

        high = np.where(cond, high, epsilon)
        low = np.where(cond, epsilon, low)
        rho_opt = np.where(reassign(cond), rho, rho_opt)
        if np.any(delta_bound>delta_bound_opt):
            i_opt = i
        delta_bound_opt = np.where(cond, delta_bound, delta_bound_opt)
        epsilon_opt = np.where(cond, epsilon, epsilon_opt)

        old_epsilon = epsilon
        
        epsilon = np.where(np.isnan(high), 2*epsilon, (low + high)/2)
            
        if np.linalg.norm(old_epsilon - epsilon) < 1e-6:
            break
    
    return rho_opt, epsilon_opt, delta_bound_opt, i_opt+1

def line_search_parabola(pol, newpol, actor_params, rets, alpha, natgrad, 
                normalize=True,
                use_rmax=True,
                use_renyi=True,
                max_search_ite=30, rmax=None, delta=0.2, reassign=None):
    
    rho_init = newpol.eval_params()
    bound_init = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
    n_bounds = len(bound_init)
    epsilon = np.ones(n_bounds)
    epsilon_old = np.zeros(n_bounds)
    max_increase=2. 
    delta_bound_tol=1e-4
    delta_bound_old = -np.inf * np.ones(n_bounds)
    
    rho_old = rho_init
    if max_search_ite<=0:
        rho = rho_init + alpha*natgrad
        newpol.set_params(rho)
        delta_bound = newpol.eval_bound(actor_params, rets, pol, rmax, 
                                 normalize, use_rmax, use_renyi, delta) - bound_init
        return rho, np.ones(len(epsilon)), delta_bound, 0

    for i in range(max_search_ite):
        stepsize = alpha*reassign(epsilon)
        stepsize = np.where(np.isnan(stepsize), np.zeros(len(stepsize)), stepsize)
        rho = rho_init + stepsize * natgrad
        newpol.set_params(rho)

        bound = newpol.eval_bound(actor_params, rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)

        if np.any(np.isnan(bound)):
            warnings.warn('Got NaN bound value!')
        if np.all(np.isnan(bound)):    
            return rho_old, epsilon_old, delta_bound_old, i + 1
	
        epsilon_old = epsilon
        delta_bound = bound - bound_init

        epsilon = np.where(delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old,
                           epsilon_old*max_increase,
                           epsilon_old ** 2 / (2 * (epsilon_old - delta_bound)))

        if np.all(delta_bound <= delta_bound_old + delta_bound_tol):
            if np.all(delta_bound_old < 0.):
                return rho_init, np.zeros(n_bounds), np.zeros(n_bounds), i + 1
            else:
                return rho_old, epsilon_old, delta_bound_old, i+1
        epsilon = np.where(np.logical_and(delta_bound <= delta_bound_old + delta_bound_tol,
                                              delta_bound_old < 0.),
                           np.zeros(n_bounds),
                           epsilon)
        epsilon = np.where(np.logical_and(delta_bound <= delta_bound_old + delta_bound_tol,
                                              delta_bound_old >= 0.),
                           epsilon_old,
                           epsilon)

        epsilon = np.where(np.isnan(epsilon), np.zeros(len(epsilon)), epsilon)
        delta_bound = np.where(np.isnan(delta_bound), np.zeros(len(delta_bound)), delta_bound)

        delta_bound_old = delta_bound
        rho_old = rho

    delta_bound_old = np.where(np.isnan(epsilon_old), np.zeros(len(delta_bound_old)), delta_bound_old)
    epsilon_old = np.where(np.isnan(epsilon_old), np.zeros(len(epsilon_old)), epsilon_old)
    epsilon_old = np.where(np.isinf(epsilon_old), np.zeros(len(epsilon_old)), epsilon_old)
    return rho_old, epsilon_old, delta_bound_old, i+1

def optimize_offline(pol, newpol, actor_params, rets, grad_tol=1e-4, bound_tol=1e-4, max_offline_ite=100, 
                     normalize=True, 
                     use_rmax=True,
                     use_renyi=True,
                     max_search_ite=30,
                     rmax=None, delta=0.2, use_parabola=False):

    rho = pol.eval_params()

    layer_lens = newpol.layer_lens
    n_bounds = len(layer_lens)
    def reassign(v):
        v = np.repeat(v, layer_lens)
        return np.concatenate((v, v))
    improvement = np.zeros(n_bounds)    
    
    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    print(titlestr % ("iter", "epsilon", "step size", "num line search", 
                      "gradient norm", "delta bound ite", "delta bound tot"))
    
    natgrad = None
    
    for i in range(max_offline_ite):
        #Candidate policy
        newpol.set_params(rho)

        #subsampling
        indexes = np.random.choice(len(rets), min(2000,len(rets)), replace=False)
        #indexes = np.argsort(rets)[-min(500, len(rets)):]
        _rets = np.take(rets, indexes, axis=0)
        _actor_params = np.take(actor_params, indexes, axis=0)        

        #Bound with gradient
        bound, grad = newpol.eval_bound_and_grad(_actor_params, _rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)
        if np.any(np.isnan(grad)):
            warnings.warn('Got NaN gradient! Stopping!')
            return rho, improvement
        if np.any(np.isnan(bound)):
            warnings.warn('Got NaN bound! Stopping!')
            return rho, improvement     

            
        #Natural gradient
        if newpol.diagonal: 
            natgrad = grad/(newpol.eval_fisher() + 1e-24)
        else:
            raise NotImplementedError
        
        #Step size search
        cum_layer_lens = np.cumsum(layer_lens)[:-1]
        grad_norms2 = grad*natgrad
        grad_norms2 = np.reshape(grad_norms2, (2, len(grad_norms2)//2))
        grad_norms2 = np.sum(grad_norms2, axis=0)
        grad_norms2 = np.split(grad_norms2, cum_layer_lens)        
        grad_norms2 = list(map(np.sum, grad_norms2))
        grad_norms2 = list(map(np.atleast_1d, grad_norms2))
        grad_norms2 = reassign(grad_norms2)
        alpha = 1. / grad_norms2
        grad_norms = np.sqrt(grad_norms2)
        alpha = np.where(np.isnan(alpha), np.zeros(len(alpha)), alpha)
        if np.sum(grad_norms) < grad_tol:
            print("stopping - gradient norm < gradient_tol")
            return rho, improvement
        grad_norm = np.max(grad_norms)
        #"""
        
        
        #"""
        
        line_search = line_search_parabola if use_parabola else line_search_binary
        rho, epsilon, delta_bound, num_line_search = line_search(pol, 
                                                                 newpol, 
                                                                 _actor_params, 
                                                                 _rets, 
                                                                 alpha, 
                                                                 natgrad, 
                                                                 normalize=normalize,
                                                                 use_rmax=use_rmax,
                                                                 use_renyi=use_renyi,
                                                                 max_search_ite=max_search_ite,
                                                                 rmax=rmax,
                                                                 delta=delta,
                                                                 reassign=reassign)
        
        stepsize = alpha*reassign(epsilon)
        stepsize = np.where(np.isnan(stepsize), np.zeros(len(stepsize)), stepsize)

        newpol.set_params(rho)
        improvement+=delta_bound
        print(fmtstr % (i+1, 
                        np.max(epsilon), 
                        np.max(stepsize), 
                        num_line_search, 
                        grad_norm, 
                        np.amax(delta_bound), 
                        np.amax(improvement)))
        if np.all(delta_bound < bound_tol):
            print("stopping - delta bound < bound_tol")
            return rho, improvement
    
    return rho, improvement


def learn(env_maker, pol_maker, sampler,
          gamma, n_episodes, horizon, max_iters, 
          feature_fun=None, 
          iw_norm='sn', 
          bound='max-ess',
          max_offline_iters=100, 
          max_search_ite=30,
          verbose=True, 
          save_weights=True,
          delta=0.2,
          center_return=False,
          line_search_type='parabola',
          adaptive_batch=False):
    
    #Initialization
    env = env_maker()
    pol = pol_maker('pol', env.observation_space, env.action_space)
    newpol = pol_maker('oldpol', env.observation_space, env.action_space)
    newpol.set_params(pol.eval_params())
    old_rho = pol.eval_params()
    batch_size = n_episodes
    normalize = True if iw_norm=='sn' else False
    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    
    if bound == 'std-d2':
        use_rmax = False
        use_renyi = True
    elif bound == 'max-d2':
        use_rmax = True
        use_renyi = True
    elif bound == 'max-ess':
        use_rmax = True
        use_renyi = False
    elif bound == 'std-ess':
        use_rmax = False
        use_renyi = False
    else: 
        raise NotImplementedError
        
    if line_search_type == 'parabola':
        use_parabola = True
    elif line_search_type == 'binary':
        use_parabola = False
    else:
        raise NotImplementedError
        
    promise = -np.inf
    actor_params, rets, disc_rets, lens = [], [], [], []    
    old_actor_params, old_rets, old_disc_rets, old_lens = [], [], [], []

    #Learning
    for it in range(max_iters):
        logger.log('\n********** Iteration %i ************' % it)
        rho = pol.eval_params() #Higher-order-policy parameters
        if verbose>1:
            logger.log('Higher-order parameters: ', rho)
        if save_weights: 
            w_to_save = rho
            
        #Add 50k samples to the batch
        with timed('Sampling'):
            if sampler:
                seg = sampler.collect(rho)
                _lens, _rets, _disc_rets, _actor_params = seg['lens'], seg['rets'], seg['disc_rets'], seg['actor_params']
                lens.extend(_lens)
                rets.extend(_rets)
                disc_rets.extend(_disc_rets)
                actor_params.extend(_actor_params)
                eps_this_iter = len(_lens)
            else:
                samples_to_get = n_episodes*horizon
                tot_samples = 0
                eps_this_iter = 0
                frozen_pol = pol.freeze()
                while tot_samples<samples_to_get:
                    eps_this_iter+=1
                    theta = frozen_pol.resample()
                    actor_params.append(theta)
                    _horizon = min(horizon, samples_to_get - tot_samples)
                    ret, disc_ret, ep_len = eval_trajectory(env, frozen_pol, gamma, _horizon, feature_fun)
                    tot_samples+=ep_len
                    rets.append(ret)
                    disc_rets.append(disc_ret)
                    lens.append(ep_len)
                    
        complete = len(rets)>=batch_size #Is the batch complete?
        #Normalize reward
        norm_disc_rets = np.array(disc_rets)
        if center_return:
            norm_disc_rets = norm_disc_rets - np.mean(norm_disc_rets)
        rmax = np.max(abs(norm_disc_rets))
        #Estimate online performance
        perf = np.mean(norm_disc_rets)
        episodes_so_far+=n_episodes
        timesteps_so_far+=sum(lens[-n_episodes:])
        
        with timed('summaries before'):
            logger.log("Performance (plain, undiscounted): ", np.mean(rets[-n_episodes:]))
            #Data regarding the episodes collected in this iteration
            logger.record_tabular("Iteration", it)
            logger.record_tabular("InitialBound", str(newpol.eval_bound(actor_params, norm_disc_rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta)))
            logger.record_tabular("EpLenMean", np.mean(lens[-n_episodes:]))
            logger.record_tabular("EpRewMean", np.mean(norm_disc_rets[-n_episodes:]))
            logger.record_tabular("UndEpRewMean", np.mean(norm_disc_rets[-n_episodes:]))
            logger.record_tabular("EpThisIter", n_episodes)
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("BatchSize", batch_size)
            logger.record_tabular("TimeElapsed", time.time() - tstart)
        
        
        if adaptive_batch and complete and perf<promise and batch_size<5*n_episodes:
            #The policy is rejected (unless batch size is already maximal)
            iter_type = 0
            if verbose: logger.log('Rejecting policy (expected at least %f, got %f instead)!\nIncreasing batch_size' % 
                                   (promise, perf))
            batch_size+=n_episodes #Increase batch size
            newpol.set_params(old_rho) #Reset to last accepted policy
            promise = -np.inf #No need to test last accepted policy
            #Reuse old trajectories
            actor_params = old_actor_params
            rets = old_rets
            disc_rets = old_disc_rets
            lens = old_lens
            if verbose: logger.log('Must collect more data (have %d/%d)' % (len(rets), batch_size))
            complete = False
        elif complete:
            #The policy is accepted, optimization is performed
            iter_type = 1
            old_rho = rho #Save as last accepted policy (and its trajectories)
            old_actor_params = actor_params
            old_rets = rets
            old_disc_rets = disc_rets
            old_lens = lens
            with timed('Optimizing offline'):
                rho, improvement = optimize_offline(pol, newpol, actor_params, norm_disc_rets,
                                                    normalize=normalize,
                                                    use_rmax=use_rmax,
                                                    use_renyi=use_renyi,
                                                    max_offline_ite=max_offline_iters,
                                                    max_search_ite=max_search_ite,
                                                    rmax=rmax,
                                                    delta=delta,
                                                    use_parabola=use_parabola)
                newpol.set_params(rho)
                #assert(improvement>=0.)
                #Expected performance
                promise = np.amax(newpol.eval_bound(actor_params, norm_disc_rets, pol, rmax,
                                                         normalize, use_rmax, use_renyi, delta))
        else:
            #The batch is incomplete, more data will be collected
            iter_type = 2
            if verbose: logger.log('Must collect more data (have %d/%d)' % (len(rets), batch_size))
            newpol.set_params(rho) #Policy stays the same
            
        #Save data
        if save_weights:
            logger.record_tabular('Weights', str(w_to_save))
        
        with timed('summaries after'):
            unn_iws = newpol.eval_iws(actor_params, behavioral=pol, normalize=False)
            iws = unn_iws/np.sum(unn_iws)
            ess = np.linalg.norm(unn_iws, 1) ** 2 / np.linalg.norm(unn_iws, 2) ** 2
            J, varJ = newpol.eval_performance(actor_params, norm_disc_rets, behavioral=pol)
            renyi = newpol.eval_renyi(pol)
            bound = newpol.eval_bound(actor_params, norm_disc_rets, pol, rmax,
                                                             normalize, use_rmax, use_renyi, delta)
            
            #Data regarding the whole batch
            logger.record_tabular('BatchSize', batch_size)
            logger.record_tabular('IterType', iter_type)
            logger.record_tabular('Bound', str(bound))
            #Discounted, [centered]
            logger.record_tabular('InitialReturnMean', np.mean(norm_disc_rets))
            logger.record_tabular('InitialReturnMax', np.max(norm_disc_rets))
            logger.record_tabular('InitialReturnMin', np.min(norm_disc_rets))
            logger.record_tabular('InitialReturnStd', np.std(norm_disc_rets))
            logger.record_tabular('InitialReturnMin', np.min(norm_disc_rets))
            #Discounted, uncentered
            logger.record_tabular('UncReturnMean', np.mean(disc_rets))
            logger.record_tabular('UncReturnMax', np.max(disc_rets))
            logger.record_tabular('UncReturnMin', np.min(disc_rets))
            logger.record_tabular('UncReturnStd', np.std(disc_rets))
            logger.record_tabular('UncReturnMin', np.min(disc_rets))
            #Undiscounted, uncentered
            logger.record_tabular('PlainReturnMean', np.mean(rets))
            logger.record_tabular('PlainReturnMax', np.max(rets))
            logger.record_tabular('PlainReturnMin', np.min(rets))
            logger.record_tabular('PlainReturnStd', np.std(rets))
            logger.record_tabular('PlainReturnMin', np.min(rets))
            #Iws
            logger.record_tabular('D2', str(renyi))
            logger.record_tabular('ReturnMeanIw', str(J))
            logger.record_tabular('MaxIWNorm', str(np.max(iws,0)))
            logger.record_tabular('MinIWNorm', str(np.min(iws,0)))
            logger.record_tabular('MeanIWNorm', str(np.mean(iws,0)))
            logger.record_tabular('StdIWNorm', str(np.std(iws,0)))
            logger.record_tabular('MaxIW', str(np.max(unn_iws,0)))
            logger.record_tabular('MinIW', str(np.min(unn_iws,0)))
            logger.record_tabular('MeanIW', str(np.mean(unn_iws,0)))
            logger.record_tabular('StdIW', str(np.std(unn_iws,0)))
            logger.record_tabular('ESSClassic', str(ess))
            logger.record_tabular('ESSRenyi', str(batch_size/np.exp(renyi)))
                    
        logger.dump_tabular()
        
        #Update behavioral
        pol.set_params(newpol.eval_params())
        if complete:
            #Start new batch
            actor_params, rets, disc_rets, lens = [], [], [], []

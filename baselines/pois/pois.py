import numpy as np
import warnings
import baselines.common.tf_util as U
import tensorflow as tf
import time
from baselines.common import zipsame, colorize
from contextlib import contextmanager
from collections import deque
from baselines import logger
from baselines.common.cg import cg

@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    print(colorize('done in %.3f seconds'%(time.time() - tstart), color='magenta'))

def traj_segment_generator(pi, env, n_episodes, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon * n_episodes)])
    rews = np.zeros(horizon * n_episodes, 'float32')
    vpreds = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])
    prevacs = acs.copy()
    mask = np.ones(horizon * n_episodes, 'float32')

    i = 0
    j = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        #if t > 0 and t % horizon == 0:
        if i == n_episodes:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "mask" : mask}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            mask = np.ones(horizon * n_episodes, 'float32')
            i = 0
            t = 0

        obs[t] = ob
        vpreds[t] = vpred
        news[t] = new
        acs[t] = ac
        prevacs[t] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[t] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1
        if new or j == horizon:
            new = True
            env.done = True

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()

            next_t = (i+1) * horizon

            mask[t+1:next_t] = 0.
            acs[t+1:next_t] = acs[t]
            obs[t+1:next_t] = obs[t]

            t = next_t - 1
            i += 1
            j = 0
        t += 1

def add_disc_rew(seg, gamma):
    new = np.append(seg['new'], 1)
    rew = seg['rew']

    n_ep = len(seg['ep_rets'])
    n_samp = len(rew)

    seg['ep_disc_ret'] = ep_disc_ret = np.empty(n_ep, 'float32')
    seg['disc_rew'] = disc_rew = np.empty(n_samp, 'float32')

    discounter = 0
    ret = 0.
    i = 0
    for t in range(n_samp):
        disc_rew[t] = rew[t] * gamma ** discounter
        ret += disc_rew[t]

        if new[t + 1]:
            discounter = 0
            ep_disc_ret[i] = ret
            i += 1
            ret = 0.
        else:
            discounter += 1

def update_epsilon(delta_bound, epsilon_old, max_increase=2.):
    if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
        return epsilon_old * max_increase
    else:
        return epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))

def line_search_parabola(theta_init, alpha, natural_gradient, set_parameter, evaluate_bound, delta_bound_tol=1e-4, max_line_search_ite=30):
    epsilon = 1.
    epsilon_old = 0.
    delta_bound_old = -np.inf
    bound_init = evaluate_bound()
    theta_old = theta_init

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * alpha * natural_gradient
        set_parameter(theta)

        bound = evaluate_bound()

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return theta_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon
        epsilon = update_epsilon(delta_bound, epsilon_old)
        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return theta_init, 0., 0., i+1
            else:
                return theta_old, epsilon_old, delta_bound_old, i+1

        delta_bound_old = delta_bound
        theta_old = theta

    return theta_old, epsilon_old, delta_bound_old, i+1

def line_search_binary(theta_init, alpha, natural_gradient, set_parameter, evaluate_loss, delta_bound_tol=1e-4, max_line_search_ite=30):
    low = 0.
    high = None
    bound_init = evaluate_loss()
    delta_bound_old = 0.
    theta_opt = theta_init
    i_opt = 0
    delta_bound_opt = 0.
    epsilon_opt = 0.

    epsilon = 1.

    for i in range(max_line_search_ite):

        theta = theta_init + epsilon * natural_gradient * alpha
        set_parameter(theta)

        bound = evaluate_loss()
        delta_bound = bound - bound_init

        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')

        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            theta_opt = theta
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        epsilon_old = epsilon

        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.

        if abs(epsilon_old - epsilon) < 1e-12:
            break

    return theta_opt, epsilon_opt, delta_bound_opt, i_opt+1


def optimize_offline(theta_init, set_parameter, line_search, evaluate_loss, evaluate_gradient, evaluate_natural_gradient=None, gradient_tol=1e-4, bound_tol=1e-4, max_offline_ite=100):
    theta = theta_old = theta_init
    improvement = improvement_old = 0.
    set_parameter(theta)


    '''
    bound_init = evaluate_loss()
    import scipy.optimize as opt

    def func(x):
        set_parameter(x)
        return -evaluate_loss()

    def grad(x):
        set_parameter(x)
        return -evaluate_gradient().astype(np.float64)

    theta, bound, d = opt.fmin_l_bfgs_b(func=func,
                                        fprime=grad,
                                x0=theta_init.astype(np.float64),
                                maxiter=100,
                                    )
    print(bound_init, bound)

    print(d)
    
    set_parameter(theta)
    improvement = bound_init + bound
    return theta, improvement

    '''

    fmtstr = '%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g'
    titlestr = '%6s %10s %10s %18s %18s %18s %18s'
    print(titlestr % ('iter', 'epsilon', 'step size', 'num line search', 'gradient norm', 'delta bound ite', 'delta bound tot'))

    for i in range(max_offline_ite):
        bound = evaluate_loss()
        gradient = evaluate_gradient()

        if np.any(np.isnan(gradient)):
            warnings.warn('Got NaN gradient! Stopping!')
            set_parameter(theta_old)
            return theta_old, improvement

        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            set_parameter(theta_old)
            return theta_old, improvement_old

        if evaluate_natural_gradient is not None:
            natural_gradient = evaluate_natural_gradient(gradient)
        else:
            natural_gradient = gradient

        if np.dot(gradient, natural_gradient) < 0:
            warnings.warn('NatGradient dot Gradient < 0! Using vanilla gradient')
            natural_gradient = gradient

        gradient_norm = np.sqrt(np.dot(gradient, natural_gradient))

        if gradient_norm < gradient_tol:
            print('stopping - gradient norm < gradient_tol')
            return theta, improvement

        alpha = 1. / gradient_norm ** 2

        theta_old = theta
        improvement_old = improvement
        theta, epsilon, delta_bound, num_line_search = line_search(theta, alpha, natural_gradient, set_parameter, evaluate_loss)
        set_parameter(theta)

        improvement += delta_bound
        print(fmtstr % (i+1, epsilon, alpha*epsilon, num_line_search, gradient_norm, delta_bound, improvement))

        if delta_bound < bound_tol:
            print('stopping - delta bound < bound_tol')
            return theta, improvement

    return theta, improvement

def render(env, pi, horizon):

    t = 0
    ob = env.reset()
    env.render()

    done = False
    while not done and t < horizon:
        ac, _ = pi.act(True, ob)
        ob, _, done, _ = env.step(ac)
        time.sleep(0.1)
        env.render()
        t += 1


def learn(make_env, make_policy, *,
          n_episodes,
          horizon,
          delta,
          gamma,
          max_iters,
          sampler=None,
          use_natural_gradient=False, #can be 'exact', 'approximate'
          fisher_reg=1e-2,
          iw_method='is',
          iw_norm='none',
          bound='J',
          line_search_type='parabola',
          save_weights=False,
          improvement_tol=0.,
          center_return=False,
          render_after=None,
          max_offline_iters=100,
          callback=None):

    np.set_printoptions(precision=3)
    max_samples = horizon * n_episodes

    if line_search_type == 'binary':
        line_search = line_search_binary
    elif line_search_type == 'parabola':
        line_search = line_search_parabola
    else:
        raise ValueError()

    # Building the environment
    env = make_env()
    ob_space = env.observation_space
    ac_space = env.action_space

    # Building the policy
    pi = make_policy('pi', ob_space, ac_space)
    oldpi = make_policy('oldpi', ob_space, ac_space)

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split('/')[1].startswith('pol')]

    shapes = [U.intprod(var.get_shape().as_list()) for var in var_list]
    n_parameters = sum(shapes)

    # Placeholders
    ob_ = ob = U.get_placeholder_cached(name='ob')
    ac_ = pi.pdtype.sample_placeholder([max_samples], name='ac')
    mask_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='mask')
    disc_rew_ = tf.placeholder(dtype=tf.float32, shape=(max_samples), name='disc_rew')
    gradient_ = tf.placeholder(dtype=tf.float32, shape=(n_parameters, 1), name='gradient')

    # Policy densities
    target_log_pdf = pi.pd.logp(ac_)
    behavioral_log_pdf = oldpi.pd.logp(ac_)
    log_ratio = target_log_pdf - behavioral_log_pdf
    
    # Split operations
    disc_rew_split = tf.stack(tf.split(disc_rew_ * mask_, n_episodes))
    log_ratio_split = tf.stack(tf.split(log_ratio * mask_, n_episodes))
    target_log_pdf_split = tf.stack(tf.split(target_log_pdf * mask_, n_episodes))
    mask_split = tf.stack(tf.split(mask_, n_episodes))
    
    # Renyi divergence
    emp_d2_split = tf.stack(tf.split(pi.pd.renyi(oldpi.pd, 2) * mask_, n_episodes))
    emp_d2_cum_split = tf.reduce_sum(emp_d2_split, axis=1)
    empirical_d2 = tf.reduce_mean(tf.exp(emp_d2_cum_split))

    # Return
    ep_return = tf.reduce_sum(mask_split * disc_rew_split, axis=1)
    if center_return:
        ep_return = ep_return - tf.reduce_mean(ep_return)

    return_mean = tf.reduce_mean(ep_return)
    return_std = U.reduce_std(ep_return)
    return_max = tf.reduce_max(ep_return)
    return_min = tf.reduce_min(ep_return)
    return_abs_max = tf.reduce_max(tf.abs(ep_return))
    
    if iw_method == 'pdis':
        raise NotImplementedError()
    elif iw_method == 'is':
        iw = tf.exp(tf.reduce_sum(log_ratio_split, axis=1))
        if iw_norm == 'none':
            iwn = iw / n_episodes
            w_return_mean = tf.reduce_sum(iwn * ep_return)
        elif iw_norm == 'sn':
            iwn = iw / tf.reduce_sum(iw)
            w_return_mean = tf.reduce_sum(iwn * ep_return)
        elif iw_norm == 'regression':
            iwn = iw / n_episodes
            mean_iw = tf.reduce_mean(iw)
            beta = tf.reduce_sum((iw - mean_iw) * ep_return * iw) / (tf.reduce_sum((iw - mean_iw) ** 2) + 1e-24)
            w_return_mean = tf.reduce_mean(iw * ep_return - beta * (iw - 1))
        else:
            raise NotImplementedError()
        
        ess_classic = tf.linalg.norm(iw, 1) ** 2 / tf.linalg.norm(iw, 2) ** 2
        sqrt_ess_classic = tf.linalg.norm(iw, 1) / tf.linalg.norm(iw, 2)
        ess_renyi = n_episodes / empirical_d2
    else:
        raise NotImplementedError()
    
    if bound == 'J':
        bound_ = w_return_mean
    elif bound == 'std-d2':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / (delta * ess_renyi)) * return_std
    elif bound == 'max-d2':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / (delta * ess_renyi)) * return_abs_max
    elif bound == 'max-ess':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / delta) / sqrt_ess_classic * return_abs_max
    elif bound == 'std-ess':
        bound_ = w_return_mean - tf.sqrt((1 - delta) / delta) / sqrt_ess_classic * return_std
    else:
        raise NotImplementedError()

    losses = [bound_, return_mean, return_max, return_min, return_std, empirical_d2, w_return_mean,
              tf.reduce_max(iwn), tf.reduce_min(iwn), tf.reduce_mean(iwn), U.reduce_std(iwn), tf.reduce_max(iw),
              tf.reduce_min(iw), tf.reduce_mean(iw), U.reduce_std(iw), ess_classic, ess_renyi]
    loss_names = ['Bound', 'InitialReturnMean', 'InitialReturnMax', 'InitialReturnMin', 'InitialReturnStd',
                  'EmpiricalD2', 'ReturnMeanIW', 'MaxIWNorm', 'MinIWNorm', 'MeanIWNorm', 'StdIWNorm',
                  'MaxIW', 'MinIW', 'MeanIW', 'StdIW', 'ESSClassic', 'ESSRenyi']

    if use_natural_gradient:
        p = tf.placeholder(dtype=tf.float32, shape=[None])
        target_logpdf_episode = tf.reduce_sum(target_log_pdf_split * mask_split, axis=1)
        grad_logprob = U.flatgrad(tf.stop_gradient(iwn) * target_logpdf_episode, var_list)
        dot_product = tf.reduce_sum(grad_logprob * p)
        hess_logprob = U.flatgrad(dot_product, var_list)
        compute_linear_operator = U.function([p, ob_, ac_, disc_rew_, mask_], [-hess_logprob])


    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    
    compute_lossandgrad = U.function([ob_, ac_, disc_rew_, mask_], losses + [U.flatgrad(bound_, var_list)])
    compute_grad = U.function([ob_, ac_, disc_rew_, mask_], [U.flatgrad(bound_, var_list)])
    compute_bound = U.function([ob_, ac_, disc_rew_, mask_], [bound_])
    compute_losses = U.function([ob_, ac_, disc_rew_, mask_], losses)

    set_parameter = U.SetFromFlat(var_list)
    get_parameter = U.GetFlat(var_list)

    if sampler is None:
        seg_gen = traj_segment_generator(pi, env, n_episodes, horizon, stochastic=True)
        sampler = type("SequentialSampler", (object,), {"collect": lambda self, _: seg_gen.__next__()})()

    U.initialize()
    
    # Starting optimizing
    
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=n_episodes)
    rewbuffer = deque(maxlen=n_episodes)
    
    while True:

        iters_so_far += 1

        if render_after is not None and iters_so_far % render_after == 0:
            if hasattr(env, 'render'):
                render(env, pi, horizon)

        if callback:
            callback(locals(), globals())

        if iters_so_far >= max_iters:
            print('Finised...')
            break

        logger.log('********** Iteration %i ************' % iters_so_far)
        
        theta = get_parameter()
        print(theta)
        with timed('sampling'):
            seg = sampler.collect(theta)
        
        add_disc_rew(seg, gamma)

        lens, rets = seg['ep_lens'], seg['ep_rets']
        lenbuffer.extend(lens)
        rewbuffer.extend(rets)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        args = ob, ac, disc_rew, mask = seg['ob'], seg['ac'], seg['disc_rew'], seg['mask']

        assign_old_eq_new()

        def evaluate_loss():
            loss = compute_bound(*args)
            return loss[0]

        def evaluate_gradient():
            gradient = compute_grad(*args)
            return gradient[0]

        if use_natural_gradient:
            def evaluate_fisher_vector_prod(x):
                return compute_linear_operator(x, *args)[0] + fisher_reg * x

            def evaluate_natural_gradient(g):
                return cg(evaluate_fisher_vector_prod, g, cg_iters=10, verbose=0)
        else:
            evaluate_natural_gradient = None

        with timed('summaries before'):
            logger.record_tabular("Itaration", iters_so_far)
            logger.record_tabular("InitialBound", evaluate_loss())
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        if save_weights:
            logger.record_tabular('Weights', str(get_parameter()))

        with timed("offline optimization"):

            theta, improvement = optimize_offline(theta,
                                                  set_parameter,
                                                  line_search,
                                                  evaluate_loss,
                                                  evaluate_gradient,
                                                  evaluate_natural_gradient,
                                                  max_offline_ite=max_offline_iters)

        set_parameter(theta)

        with timed('summaries after'):
            meanlosses = np.array(compute_losses(*args))
            for (lossname, lossval) in zip(loss_names, meanlosses):
                logger.record_tabular(lossname, lossval)

        logger.dump_tabular()

    env.close()

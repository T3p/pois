import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import DiagGaussianPdType, CholeskyGaussianPdType
import numpy as np
from baselines.common import set_global_seeds
import scipy.stats as sts

"""References
PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
control." International Conference on Artificial Neural Networks. Springer,
Berlin, Heidelberg, 2008.
"""

class PeMlpPolicy(object):
    """Multi-layer-perceptron policy with Gaussian parameter-based exploration"""
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            U.initialize()
            #Sample initial actor params
            tf.get_default_session().run(self._use_sampled_actor_params)

    def _init(self, ob_space, ac_space, hid_layers=[],
              deterministic=True, diagonal=True,
              use_bias=True, use_critic=False, 
              seed=None, verbose=True):
        """Params:
            ob_space: task observation space
            ac_space : task action space
            hid__layers: list with width of each hidden layer
            deterministic: whether the actor is deterministic
            diagonal: whether the higher order policy has a diagonal covariance
            matrix
            use_bias: whether to include bias in neurons
            use_critic: whether to include a critic network
            seed: optional random seed
        """
        assert isinstance(ob_space, gym.spaces.Box)
        assert len(ac_space.shape)==1
        self.diagonal = diagonal
        self.use_bias = use_bias
        batch_length = None #Accepts a sequence of episodes of arbitrary length
        self.ac_dim = ac_space.shape[0]
        self.ob_dim = ob_space.shape[0]
        self.linear = not hid_layers
        self.verbose = verbose

        if seed is not None:
            set_global_seeds(seed)

        self._ob = ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None] + list(ob_space.shape))

        #Critic (normally not used)
        if use_critic:
            with tf.variable_scope('critic'):
                last_out = ob
                for i, hid_size in enumerate(hid_layers):
                    last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        #Actor (N.B.: weight initialization is irrelevant)
        with tf.variable_scope('actor'):
            last_out = ob
            for i, hid_size in enumerate(hid_layers):
                #Mlp feature extraction
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size,
                                                      name='fc%i'%(i+1),
                                                      kernel_initializer=U.normc_initializer(1),use_bias=use_bias))
            if deterministic and isinstance(ac_space, gym.spaces.Box):
                #Determinisitc action selection
                self.actor_mean = actor_mean = tf.layers.dense(last_out, ac_space.shape[0],
                                       name='action',
                                       kernel_initializer=U.normc_initializer(0.01),
                                       use_bias=use_bias)
            else: 
                raise NotImplementedError #Currently supports only deterministic action policies

        #Higher order policy (Gaussian)
        with tf.variable_scope('actor') as scope:
            self.actor_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name)
            self.flat_actor_weights = tf.concat([tf.reshape(w, [-1]) for w in \
                                            self.actor_weights], axis=0) #flatten
            self._n_actor_weights = n_actor_weights = self.flat_actor_weights.shape[0]

        with tf.variable_scope('higher'):
            #Initial means sampled from a normal distribution N(0,1)
            higher_mean_init = tf.where(tf.not_equal(self.flat_actor_weights, tf.constant(0, dtype=tf.float32)),
                        tf.random_normal(shape=[n_actor_weights.value], stddev=0.01), tf.zeros(shape=[n_actor_weights]))
            self.higher_mean = higher_mean = tf.get_variable(name='higher_mean',
                                               initializer=higher_mean_init)
            
            if diagonal:
                #Diagonal covariance matrix; all stds initialized to 0
                self.higher_logstd = higher_logstd = tf.get_variable(name='higher_logstd',
                                                                     shape=[n_actor_weights],
                                               initializer=tf.initializers.constant(0.))
                pdparam = tf.concat([higher_mean, higher_mean * 0. + 
                                   higher_logstd], axis=0)
                self.pdtype = pdtype = DiagGaussianPdType(n_actor_weights.value) 
            else: 
                #Cholesky covariance matrix
                self.higher_logstd = higher_logstd = tf.get_variable(
                    name='higher_logstd',
                    shape=[n_actor_weights*(n_actor_weights + 1)//2],
                    initializer=tf.initializers.constant(0.))
                pdparam = tf.concat([higher_mean, 
                                    higher_logstd], axis=0)
                self.pdtype = pdtype = CholeskyGaussianPdType(
                    n_actor_weights.value) 

        #Sample actor weights
        self.pd = pdtype.pdfromflat(pdparam)
        sampled_actor_params = self.pd.sample()
        symm_sampled_actor_params = self.pd.sample_symmetric()
        self._sample_symm_actor_params = U.function(
            [],list(symm_sampled_actor_params))
        self._sample_actor_params = U.function([], [sampled_actor_params])
            
        #Assign actor weights
        with tf.variable_scope('actor') as scope:
            actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name)

            self._use_sampled_actor_params = U.assignFromFlat(actor_params,
                                                         sampled_actor_params)
            
            self._set_actor_params = U.SetFromFlat(actor_params)
            
            self._get_actor_params = U.GetFlat(actor_params)

        #Act
        self._action = action = actor_mean
        self._act = U.function([ob],[action])

        #Higher policy weights
        with tf.variable_scope('higher') as scope:
            self._higher_params = higher_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
                                         scope=scope.name) 
            self.flat_higher_params = tf.concat([tf.reshape(w, [-1]) for w in \
                                            self._higher_params], axis=0) #flatten
            self._n_higher_params = self.flat_higher_params.shape[0]
            self._get_flat_higher_params = U.GetFlat(higher_params)
            self._set_higher_params = U.SetFromFlat(self._higher_params)

        #Batch PGPE
        self._actor_params_in = actor_params_in = \
                U.get_placeholder(name='actor_params_in',
                                  dtype=tf.float32,
                                  shape=[batch_length] + [n_actor_weights])
        self._rets_in = rets_in = U.get_placeholder(name='returns_in',
                                                  dtype=tf.float32,
                                                  shape=[batch_length])
        ret_mean, ret_std = tf.nn.moments(rets_in, axes=[0])
        self._get_ret_mean = U.function([self._rets_in], [ret_mean])
        self._get_ret_std = U.function([self._rets_in], [ret_std])
        self._logprobs = logprobs = self.pd.logp(actor_params_in)
        pgpe_times_n = U.flatgrad(logprobs*rets_in, higher_params)
        self._get_pgpe_times_n = U.function([actor_params_in, rets_in],
                                            [pgpe_times_n])

        #One-episode PGPE
        #Used N times to compute the baseline -> can we do better?
        self._one_actor_param_in = one_actor_param_in = U.get_placeholder(
                                    name='one_actor_param_in',
                                    dtype=tf.float32,
                                    shape=[n_actor_weights])
        one_logprob = self.pd.logp(one_actor_param_in)
        score = U.flatgrad(one_logprob, higher_params)
        score_norm = tf.norm(score)
        self._get_score = U.function([one_actor_param_in], [score])
        self._get_score_norm = U.function([one_actor_param_in], [score_norm])

        #Batch off-policy PGPE
        self._probs = tf.exp(logprobs) 
        self._behavioral = None
        self._renyi_other = None
    
        #One episode off-PGPE 
        self._one_prob = tf.exp(one_logprob)
        
        #Renyi computation
        self._det_sigma = tf.exp(tf.reduce_sum(self.higher_logstd))

        #Fisher computation (diagonal case)
        mean_fisher_diag = tf.exp(-2*self.higher_logstd)
        cov_fisher_diag = mean_fisher_diag*0 + 2
        self._fisher_diag = tf.concat([mean_fisher_diag, cov_fisher_diag], axis=0)
        self._get_fisher_diag = U.function([], [self._fisher_diag])
        
    #Black box usage
    def act(self, ob, resample=False):
        """
        Sample weights for the actor network, then sample action(s) from the 
        resulting actor depending on state(s)
           
        Params:
               ob: current state, or a list of states
               resample: whether to resample actor params before acting
        """
        
        if resample:
            actor_param = self.resample()

        action =  self._act(np.atleast_2d(ob))[0]
        return (action, actor_param) if resample else action
    
    class _FrozenLinearActor(object):
        def __init__(self, higher_params, ob_dim, ac_dim, use_bias):
            self.higher_params = np.ravel(higher_params)
            self.ob_dim = ob_dim
            self.ac_dim = ac_dim
            self.use_bias = use_bias
            self.higher_mean = self.higher_params[:len(self.higher_params)//2]
            self.higher_cov = np.diag(np.exp(2*self.higher_params[len(self.higher_params)//2:]))
            self.resample()
        
        def resample(self):
            self.actor_params = np.random.multivariate_normal(self.higher_mean, self.higher_cov)
            return self.actor_params
        
        def act(self, ob, resample=False):
            if resample:
                self.resample()
            
            ob = np.ravel(ob)
            if self.use_bias:
                np.append(ob, 1)
            ob = ob.reshape((self.ob_dim + self.use_bias, 1))
            theta = self.actor_params.reshape((self.ac_dim, self.ob_dim + self.use_bias))
            return np.ravel(np.dot(theta, ob))
        
        def seed(self, seed):
            np.random.seed(seed)

    def freeze(self):
        if not self.linear:
            return self
            
        return self._FrozenLinearActor(self.eval_params(),
                                  self.ob_dim,
                                  self.ac_dim,
                                  self.use_bias)

    def act_with(self, ob, actor_params):
        self.set_actor_params(actor_params)
        return self.act(ob)

    def resample(self):
        """Resample actor params
        
        Returns:
            the sampled actor params
        """
        tf.get_default_session().run(self._use_sampled_actor_params)
        return self.eval_actor_params()
    
    def eval_params(self):
        """Get current params of the higher order policy"""
        return self._get_flat_higher_params()

    def set_params(self, new_higher_params):
        """Set higher order policy parameters from flat sequence"""
        self._set_higher_params(new_higher_params)

    def seed(self, seed):
        if seed is not None:
            set_global_seeds(seed)

    #Direct actor policy manipulation
    def draw_actor_params(self):
        """Sample params for an actor (without using them)"""
        sampled_actor_params = self._sample_actor_params()[0]
        return sampled_actor_params

    def draw_symmetric_actor_params(self):
        return tuple(self._sample_symm_actor_params())

    def eval_actor_params(self):
        """Get actor params as last assigned"""
        return self._get_actor_params()

    def set_actor_params(self, new_actor_params):
        """Manually set actor policy parameters from flat sequence"""
        self._set_actor_params(new_actor_params)

    #Distribution properties
    def eval_renyi(self, other, order=2):
        """Renyi divergence 
            Special case: order=1 is kl divergence
        
        Params:
            other: policy to evaluate the distance from
            order: order of the Renyi divergence
            exponentiate: if true, actually returns e^Renyi(self||other)
        """
        if other is not self._renyi_other:
            if self.verbose: print('Building graph')
            self._renyi_order = tf.placeholder(name='renyi_order', dtype=tf.float32, shape=[])
            self._renyi_other = other
            if order<1:
                raise ValueError('Order must be >= 1')
            else:   
                renyi = self.pd.renyi(other.pd, alpha=self._renyi_order) 
                self._get_renyi = U.function([self._renyi_order], [renyi])

        return self._get_renyi(order)[0]

    def eval_fisher(self):
        if not self.diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return np.ravel(self._get_fisher_diag()[0])

    def fisher_product(self, x):
        if not self.diagonal:
            raise NotImplementedError(
                'Only diagonal covariance currently supported')
        return x/self.eval_fisher()

    #Performance evaluation
    def eval_performance(self, actor_params, rets, behavioral=None):
        batch_size = len(rets)
        if behavioral is None:
            #On policy
            return self._get_ret_mean(rets)[0], self._get_ret_std(rets)[0]
        else:
            #Off policy
            if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
            return self._get_off_ret_mean(rets, actor_params)[0], self._get_off_ret_std(rets, actor_params, batch_size)[0]
            
        

    #Gradient computation
    def eval_gradient(self, actor_params, rets, use_baseline=True,
                      behavioral=None):
        """
        Compute PGPE policy gradient given a batch of episodes

        Params:
            actor_params: list of actor parameters (arrays), one per episode
            rets: flat list of total [discounted] returns, one per episode
            use_baseline: wether to employ a variance-minimizing baseline 
                (may be more efficient without)
            behavioral: higher-order policy used to collect data (off-policy
                case). If None, the present policy is assumed to be the 
                behavioral(on-policy case)

        References:
            Optimal baseline for PGPE: Zhao, Tingting, et al. "Analysis and
            improvement of policy gradient estimation." Advances in Neural
            Information Processing Systems. 2011.

        """ 
        assert rets and len(actor_params)==len(rets)
        batch_size = len(rets)
        
        if not behavioral:
            #On policy
            if not use_baseline:
                #Without baseline (more efficient)
                pgpe_times_n = np.ravel(self._get_pgpe_times_n(actor_params, rets)[0])
                return pgpe_times_n/batch_size
            else:
                #With optimal baseline
                rets = np.array(rets)
                scores = np.zeros((batch_size, self._n_higher_params))
                score_norms = np.zeros(batch_size)
                for (theta, i) in zip(actor_params, range(batch_size)):
                    scores[i] = self._get_score(theta)[0]
                    score_norms[i] = self._get_score_norm(theta)[0]
                b = np.sum(rets * score_norms**2) / np.sum(score_norms**2)
                pgpe = np.mean(((rets - b).T * scores.T).T, axis=0)
                return pgpe
        else:
            #Off-policy
            if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
            if not use_baseline:
                #Without baseline (more efficient)
                off_pgpe_times_n = np.ravel(self._get_off_pgpe_times_n(actor_params,
                                                              rets)[0])
                return off_pgpe_times_n/batch_size
            else:
                #With optimal baseline
                rets = np.array(rets)
                scores = np.zeros((batch_size, self._n_higher_params))
                score_norms = np.zeros(batch_size)
                for (theta, i) in zip(actor_params, range(batch_size)):
                    scores[i] = self._get_score(theta)[0]
                    score_norms[i] = self._get_score_norm(theta)[0]
                iws = np.ravel(self._get_iws(actor_params)[0])
                b = np.sum(rets * iws**2 * score_norms**2)/ np.sum(iws**2 *
                                                                   score_norms**2)
                pgpe = np.mean(((rets - b).T * scores.T).T, axis=0)
                return pgpe
                

    def eval_natural_gradient(self, actor_params, rets, use_baseline=True,
                      behavioral=None):
        """
        Compute PGPE policy gradient given a batch of episodes

        Params:
            actor_params: list of actor parameters (arrays), one per episode
            rets: flat list of total [discounted] returns, one per episode
            use_baseline: wether to employ a variance-minimizing baseline 
                (may be more efficient without)
            behavioral: higher-order policy used to collect data (off-policy
                case). If None, the present policy is assumed to be the 
                behavioral(on-policy case)

        References:
            Optimal baseline for PGPE: Zhao, Tingting, et al. "Analysis and
            improvement of policy gradient estimation." Advances in Neural
            Information Processing Systems. 2011.

        """ 
        assert rets and len(actor_params)==len(rets)
        batch_size = len(rets)
        fisher = self.eval_fisher() + 1e-24
        
        if not behavioral:
            #On policy
            if not use_baseline:
                #Without baseline (more efficient)
                pgpe_times_n = np.ravel(self._get_pgpe_times_n(actor_params, rets)[0])
                grad = pgpe_times_n/batch_size
                if self.diagonal:
                    return grad/fisher
                else: 
                    raise NotImplementedError #TODO: full on w/o baseline
            else:
                #With optimal baseline
                if self.diagonal:
                    rets = np.array(rets)
                    scores = np.zeros((batch_size, self._n_higher_params))
                    score_norms = np.zeros(batch_size)
                    for (theta, i) in zip(actor_params, range(batch_size)):
                        scores[i] = self._get_score(theta)[0]
                        score_norms[i] = np.linalg.norm(scores[i]/fisher)
                    b = np.sum(rets * score_norms**2) / np.sum(score_norms**2)
                    npgpe = np.mean(((rets - b).T * scores.T).T, axis=0)/fisher
                    return npgpe
                else:
                    raise NotImplementedError #TODO: full on with baseline
        else:
            #Off-policy
            if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
            if not use_baseline and self.diagonal:
                #Without baseline (more efficient)
                off_pgpe_times_n = np.ravel(self._get_off_pgpe_times_n(actor_params,
                                                              rets)[0])
                grad = off_pgpe_times_n/batch_size
                return grad/fisher
            else:
                raise NotImplementedError #TODO: full off with baseline, diagonal off with baseline
    
    def eval_iws(self, actor_params, behavioral, normalize=True):
        if behavioral is not self._behavioral:
            self._build_iw_graph(behavioral)
            self._behavioral = behavioral        
        if normalize:
            return self._get_iws(actor_params)[0]
        else:
            return self._get_unn_iws(actor_params)[0]
    
    def eval_bound(self, actor_params, rets, behavioral, rmax, normalize=True,
                   use_rmax = True, use_renyi=True, delta=0.2):
        if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
        batch_size = len(rets)
        
        #"""
        ppf = np.sqrt(1./delta - 1)
        """
        if use_rmax:
            ppf = sts.norm.ppf(1 - delta)
        else:
            ppf = sts.t.ppf(1 - delta, batch_size - 1)
        #"""
        
        index = int(str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
        bound_getter = self._get_bound[index]
        
        return bound_getter(actor_params, rets, batch_size, ppf, rmax)[0]
    
    def eval_bound_and_grad(self, actor_params, rets, behavioral, rmax, normalize=True,
                   use_rmax=True, use_renyi=True, delta=0.2):
        if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral)
                self._behavioral = behavioral
        batch_size = len(rets)
        
        #"""
        ppf = np.sqrt(1./delta - 1)
        """
        if use_rmax:
            ppf = sts.norm.ppf(1 - delta)
        else:
            ppf = sts.t.ppf(1 - delta, batch_size - 1)
        #"""
        
        index = int(str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
        bound_and_grad_getter = self._get_bound_grad[index]
        
        return bound_and_grad_getter(actor_params, rets, batch_size, ppf, rmax)
    
    def _build_iw_graph(self, behavioral):
        if self.verbose: print('Building graph')
        self._batch_size = batch_size = tf.placeholder(name='batchsize', dtype=tf.float32, shape=[])
        
        #Self-normalized importance weights
        #unn_iws = self._probs/behavioral._probs
        unn_iws = tf.exp(tf.reduce_sum(self.pd.independent_logps(self._actor_params_in) - 
                    behavioral.pd.independent_logps(self._actor_params_in), axis=-1))
        iws = unn_iws/tf.reduce_sum(unn_iws)
        self._get_unn_iws = U.function([self._actor_params_in], [unn_iws])
        self._get_iws = U.function([self._actor_params_in], [iws])
        
        #Offline performance
        ret_mean = tf.reduce_sum(self._rets_in * iws)
        unn_ret_mean = tf.reduce_mean(self._rets_in*unn_iws)
        self._get_off_ret_mean = U.function([self._rets_in, self._actor_params_in], [ret_mean])
        ret_std = tf.sqrt(tf.reduce_sum(iws ** 2 * (self._rets_in - ret_mean) ** 2) * batch_size)
        self._get_off_ret_std = U.function([self._rets_in, self._actor_params_in, self._batch_size], [ret_std])
        
        #Offline gradient
        off_pgpe_times_n = U.flatgrad((tf.stop_gradient(iws) * 
                                             self._logprobs * 
                                             self._rets_in), 
                                            self._higher_params)
                    
        self._get_off_pgpe_times_n = U.function([self._actor_params_in,
                                                self._rets_in],
                                                [off_pgpe_times_n])
        
        #Renyi
        renyi = self.pd.renyi(behavioral.pd)
        renyi = tf.cond(tf.is_nan(renyi), lambda: tf.constant(np.inf), lambda: renyi)
        renyi = tf.cond(renyi<0., lambda: tf.constant(np.inf), lambda: renyi)
        
        #Weight norm
        iws2norm = tf.norm(iws)
        
        #Return properties
        self._rmax = tf.placeholder(name='R_max', dtype=tf.float32, shape=[])
        on_ret_mean, on_ret_var = tf.nn.moments(self._rets_in, axes=[0])
        
        #Penalization coefficient
        self._ppf = tf.placeholder(name='penal_coeff', dtype=tf.float32, shape=[])
        
        #All the bounds
        bounds = []
        bounds.append(unn_ret_mean - self._ppf * tf.sqrt(on_ret_var) * iws2norm) #000
        bounds.append(unn_ret_mean - self._ppf * tf.sqrt(on_ret_var) * tf.exp(0.5*renyi)/tf.sqrt(batch_size)) #001
        bounds.append(unn_ret_mean - self._ppf * self._rmax * iws2norm) #010
        bounds.append(unn_ret_mean - self._ppf * self._rmax * tf.exp(0.5*renyi)/tf.sqrt(batch_size)) #011
        bounds.append(ret_mean - self._ppf * tf.sqrt(on_ret_var) * iws2norm) #100
        bounds.append(ret_mean - self._ppf * tf.sqrt(on_ret_var) * tf.exp(0.5*renyi)/tf.sqrt(batch_size)) #101
        bounds.append(ret_mean - self._ppf * self._rmax * iws2norm) #110
        bounds.append(ret_mean - self._ppf * self._rmax * tf.exp(0.5*renyi)/tf.sqrt(batch_size)) #111
        
        inputs = [self._actor_params_in, self._rets_in, self._batch_size, self._ppf, self._rmax]
        self._get_bound = [U.function(inputs, [bounds[i]]) for i in range(len(bounds))]
        self._get_bound_grad = [U.function(inputs, [bounds[i], 
                                                    U.flatgrad(bounds[i], self._higher_params)]) for i in range(len(bounds))]
    
    
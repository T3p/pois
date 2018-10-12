import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import MultiGaussianVectorPdType
import numpy as np
from baselines.common import set_global_seeds

"""References
PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
control." International Conference on Artificial Neural Networks. Springer,
Berlin, Heidelberg, 2008.
"""

class MultiPeMlpPolicy(object):
    """Multi-layer-perceptron policy with Gaussian parameter-based exploration"""
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self.name = name
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
        
        print('Building graph')

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
            self.layers = [v for w in self.actor_weights for v in (tf.unstack(w, axis=1) if len(w.shape)>1 else [w])]
            self.layer_lens = [w.shape[0].value for w in self.layers]
            #print('# Independent Gaussians:', len(self.layer_lens))
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
                self.pdtype = pdtype = MultiGaussianVectorPdType(n_actor_weights.value) 
            else: 
                raise NotImplementedError

        #Sample actor weights
        self.pd = pdtype.pdfromflat(pdparam, self.layer_lens)
        sampled_actor_params = self.pd.sample()
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
        self._actor_params_in = U.get_placeholder(name='actor_params_in',
                                  dtype=tf.float32,
                                  shape=[batch_length] + [n_actor_weights])
        self._rets_in = rets_in = U.get_placeholder(name='returns_in',
                                                  dtype=tf.float32,
                                                  shape=[batch_length])
        ret_mean, ret_std = tf.nn.moments(rets_in, axes=[0])
        self._get_ret_mean = U.function([self._rets_in], [ret_mean])
        self._get_ret_std = U.function([self._rets_in], [ret_std])
        
        #Renyi computation
        self._det_sigma = tf.exp(tf.reduce_sum(self.higher_logstd))

        #Fisher computation (diagonal case)
        mean_fisher_diag = tf.exp(-2*self.higher_logstd)
        cov_fisher_diag = mean_fisher_diag*0 + 2
        self._fisher_diag = tf.concat([mean_fisher_diag, cov_fisher_diag], axis=0)
        self._get_fisher_diag = U.function([], [self._fisher_diag])
        
        #Lazy initialization
        self._behavioral = None
        self._renyi_other = None
        self._get_renyi = None
        
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
                ob = np.append(ob, 1)
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

    def eval_actor_params(self):
        """Get actor params as last assigned"""
        return self._get_actor_params()

    def set_actor_params(self, new_actor_params):
        """Manually set actor policy parameters from flat sequence"""
        self._set_actor_params(new_actor_params)

    #Distribution properties
    def eval_renyi(self, other):
        """Renyi-2 divergence 
        
        Params:
            other: policy to evaluate the distance from
        """
        if self._get_renyi is None or other is not self._renyi_other or other is not self._behavioral:
            print('Building graph')
            self._renyi_other = other
            renyi = self.pd.renyi(other.pd, alpha=2) 
            self._get_renyi = U.function([], [renyi])
        return self._get_renyi()[0]

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
            return np.sum(self._get_off_ret_mean(rets, actor_params)[0]), \
                    np.sum(self._get_off_ret_std(rets, actor_params, batch_size)[0])
            
    def eval_iws(self, actor_params, behavioral, normalize=True):
        if behavioral is not self._behavioral:
            self._build_iw_graph(behavioral)
            self._behavioral = behavioral        
        if normalize:
            return self._get_iws(actor_params)[0]
        else:
            return self._get_unn_iws(actor_params)[0]
    
    def eval_bound(self, actor_params, rets, behavioral, rmax, normalize=True,
                   use_rmax = True, use_renyi=True, lamb=2.):
        index = int(str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
        if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral, index)
                self._behavioral = behavioral
        batch_size = len(rets)
        if lamb>1.:
            ppf = lamb
        else:
            ppf = np.sqrt(1./lamb -1)
        bound_getter = self._get_bound
        
        bound = bound_getter(actor_params, rets, batch_size, ppf, rmax)[0]
        return bound
    
    def eval_bound_and_grad(self, actor_params, rets, behavioral, rmax, normalize=True,
                   use_rmax=True, use_renyi=True, lamb=2.):
        index = int(str(int(normalize)) + str(int(use_rmax)) + str(int(use_renyi)), 2)
        if behavioral is not self._behavioral:
                self._build_iw_graph(behavioral, index)
                self._behavioral = behavioral
        batch_size = len(rets)
        if lamb>1.:
            ppf = lamb
        else:
            ppf = np.sqrt(1./lamb -1)
        bound_and_grad_getter = self._get_bound_grad
        
        bound = self.eval_bound(actor_params, rets, behavioral, rmax, normalize,
                   use_rmax, use_renyi, lamb)
        
        bound, grad = bound_and_grad_getter(actor_params, rets, batch_size, ppf, rmax)
        return bound, grad
    
    def _build_iw_graph(self, behavioral, bound_index):
        print('Building graph')
        self._batch_size = batch_size = tf.placeholder(name='batchsize', dtype=tf.float32, shape=[])
        
        #Self-normalized importance weights
        unn_iws = tf.exp(self.pd.logp(self._actor_params_in) - 
                    behavioral.pd.logp(self._actor_params_in))
        iws = unn_iws/tf.reduce_sum(unn_iws, axis=0)
        self._get_unn_iws = U.function([self._actor_params_in], [unn_iws])
        self._get_iws = U.function([self._actor_params_in], [iws])
        
        #Offline performance
        if bound_index>3:
            ret_mean = tf.reduce_sum(tf.expand_dims(self._rets_in, -1) * iws, axis=0)
            self._get_off_ret_mean = U.function([self._rets_in, self._actor_params_in], [ret_mean])    
            ret_std = tf.sqrt(tf.reduce_sum(iws ** 2 * (tf.expand_dims(self._rets_in, -1) - ret_mean) ** 2) * batch_size)
            self._get_off_ret_std = U.function([self._rets_in, self._actor_params_in, self._batch_size], [ret_std])
        else:
            unn_ret_mean = tf.reduce_mean(tf.expand_dims(self._rets_in, -1) * unn_iws, axis=0)
            self._get_off_ret_mean = U.function([self._rets_in, self._actor_params_in], [unn_ret_mean])
            unn_ret_std = tf.sqrt(tf.reduce_sum((unn_iws*tf.expand_dims(self._rets_in, -1) - ret_mean) ** 2) / (batch_size - 1))
            self._get_off_ret_std = U.function([self._rets_in, self._actor_params_in, self._batch_size], [unn_ret_std])
                
        #Renyi
        if bound_index % 2 !=0:
            renyi = self.pd.renyi(behavioral.pd)
            renyi = tf.where(tf.is_nan(renyi), tf.constant(np.inf, shape=renyi.shape), renyi)
            renyi = tf.where(renyi<0., tf.constant(np.inf, shape=renyi.shape), renyi)
            self._get_renyi = U.function([],[renyi])
        else:            
            #Weight norm
            self._get_renyi = None
            iws2norm = tf.norm(iws, axis=0)
            
        #Return properties
        if bound_index in [2, 3, 6, 7]:
            self._rmax = tf.placeholder(name='R_max', dtype=tf.float32, shape=[])
        else:
            on_ret_mean, on_ret_var = tf.nn.moments(self._rets_in, axes=[0])
        
        #Penalization coefficient
        self._ppf = tf.placeholder(name='penal_coeff', dtype=tf.float32, shape=[])
        
        #All the bounds
        if bound_index==0:
            bound = unn_ret_mean - self._ppf * tf.sqrt(on_ret_var) * iws2norm
        elif bound_index==1:
            bound = unn_ret_mean - self._ppf * tf.sqrt(on_ret_var) * tf.exp(0.5*renyi)/tf.sqrt(batch_size)
        elif bound_index==2:
            bound = unn_ret_mean - self._ppf * self._rmax * iws2norm
        elif bound_index==3:
            bound = unn_ret_mean - self._ppf * self._rmax * tf.exp(0.5*renyi)/tf.sqrt(batch_size)
        elif bound_index==4:
            bound = ret_mean - self._ppf * tf.sqrt(on_ret_var) * iws2norm
        elif bound_index==5:
            bound = ret_mean - self._ppf * tf.sqrt(on_ret_var) * tf.exp(0.5*renyi)/tf.sqrt(batch_size)
        elif bound_index==6:
            bound = ret_mean - self._ppf * self._rmax * iws2norm
        elif bound_index==7:
            bound = ret_mean - self._ppf * self._rmax * tf.exp(0.5*renyi)/tf.sqrt(batch_size)
        else:
            raise NotImplementedError
        
        bound_grad = U.flatgrad(bound, self._higher_params)
        
        inputs = [self._actor_params_in, self._rets_in, self._batch_size, self._ppf, self._rmax]
        self._get_bound = U.function(inputs, [bound])
        self._get_bound_grad = U.function(inputs, [bound, bound_grad])

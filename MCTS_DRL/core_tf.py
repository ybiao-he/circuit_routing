import numpy as np
import tensorflow as tf
import scipy.signal
from tf_layers import conv

class Buffer:
    """
    A buffer for storing trajectories experienced by a agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.reset()

    def reset(self):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.ptr, self.path_start_idx = 0, 0  

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.array(self.rew_buf[path_slice]+[last_val])
        vals = np.array(self.val_buf[path_slice]+[last_val])
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        obs_buf = np.array(self.obs_buf)
        act_buf = np.array(self.act_buf)
        adv_buf = np.array(self.adv_buf)
        ret_buf = np.array(self.ret_buf)
        logp_buf = np.array(self.logp_buf)
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        return [obs_buf, act_buf, adv_buf, ret_buf, logp_buf]

    def get_length(self):
        return len(self.obs_buf)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def cnn(x, hidden_sizes=(32,), activation=tf.nn.relu, output_activation=None):

    dropout = 0.25
    is_training = True
    # Data input is a 1-D vector of 900 features (30*30 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 40, 40, 2])

    # scope = 1
    for h in hidden_sizes[:-1]:
        # Convolution Layer with h filters and a kernel size of 5
        x = tf.layers.conv2d(x, h, 5, activation=activation)
        # x = activation(conv(x, "c"+str(scope), n_filters=h, filter_size=5, stride=2, init_scale=np.sqrt(2)))
        # scope += 1
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        x = tf.layers.max_pooling2d(x, 2, 2)

    # Flatten the data to a 1-D vector for the fully connected layer
    fc1 = tf.contrib.layers.flatten(x)

    # Fully connected layer (in tf contrib folder for now)
    fc1 = tf.layers.dense(fc1, 512)
    # Apply Dropout (if is_training is False, dropout is not applied)
    fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

    # Output layer, class prediction
    out = tf.layers.dense(fc1, units=hidden_sizes[-1], activation=output_activation)

    return out


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

"""
Policies
"""

def cnn_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_dim):
    
    act_dim = action_dim
    logits = cnn(x, list(hidden_sizes)+[act_dim], activation, None)
    p_all = tf.nn.softmax(logits)

    # pi is the selected action based on p_all (prob for all actions)
    pi = tf.squeeze(tf.random.categorical(tf.math.log(p_all), 1), axis=1)
    # p is the prob for the given action a
    p = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * p_all, axis=1)
    # p_pi is the prob for the selected action pi
    p_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * p_all, axis=1)
    return pi, p, p_pi, p_all

"""
Actor-Critics
"""
def actor_critic(x, a, hidden_sizes=(256, 256, 256), activation=tf.nn.relu,
                     output_activation=None, policy=None, action_dim=None):

    # default policy builder depends on action space
    policy = cnn_categorical_policy

    with tf.variable_scope('pi'):
        pi, p, p_pi, p_all = policy(x, a, hidden_sizes, activation, output_activation, action_dim)
    with tf.variable_scope('v'):
        v = tf.squeeze(cnn(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, p, p_pi, v, p_all

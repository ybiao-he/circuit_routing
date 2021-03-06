import gym
import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm

import numpy as np

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class CRPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CRPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            layer_1 = activ(conv(self.processed_obs, 'c1', n_filters=256, filter_size=5, stride=4, init_scale=np.sqrt(2), **kwargs))
            layer_2 = activ(conv(layer_1, 'c2', n_filters=256, filter_size=3, stride=2, init_scale=np.sqrt(2), **kwargs))
            layer_3 = activ(conv(layer_2, 'c3', n_filters=256, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
            layer_3 = conv_to_fc(layer_3)
            extracted_features = activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))


            # extracted_features = nature_cnn(self.processed_obs, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)

            pi_h = extracted_features
            for i, layer_size in enumerate([64, 64]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([32, 32]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            # These three self variables shold be customized, this is revising the attributes in ActorCriticPolicy class
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):

        action_mask = np.zeros(4, dtype=np.float32)
        for i in range(len(action_mask)):
            action_mask[i] = obs[0][0][i][1]
            obs[0][0][i][1] = obs[0][0][i+len(action_mask)][1]

        action_dist, value, neglogp = self.sess.run([self.policy_proba, self.value_flat, self.neglogp],
                                    {self.obs_ph: obs})
        # if np.sum(obs) < 100:
        #     print("this is a wierd obs----------------")
        #     print(np.sum(obs))
        #     np.savetxt("weird_obs.csv", obs[0,0:40,0:40,0], delimiter=',')
        print(action_dist, action_mask)
        action_dist = action_dist[0]
        # action_dist *= self.get_action_mask(obs)
        action_dist *= action_mask
        if np.sum(action_dist) == 0:
            action_dist = np.array(action_mask/sum(action_mask))
        else:
            action_dist = action_dist/sum(action_dist)

        if deterministic:
            action = np.where(action_dist == np.amax(action_dist))[0]
        else:
            action = np.random.choice(len(action_dist), 1, p=action_dist)

        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):

        action_dist = self.sess.run(self.policy_proba, {self.obs_ph: obs})

        
        action_dist = action_dist[0]
        action_dist *= self.get_action_mask(obs)
        action_dist = action_dist/sum(action_dist)

        return np.array([action_dist])

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


    # def get_action_mask(self, obs):

    #     directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    #     # print(obs)

    #     target_value = obs[0][0][0][1]

    #     board = obs[0,0:40,0:40,0]

    #     # find action node from obs/board
    #     action_node_tmp = np.where(board == np.amax(board))
    #     action_node = (action_node_tmp[0][0], action_node_tmp[1][0])

    #     # find target node for current path
    #     # value = np.amax(board)
    #     # for i in range(board.shape[0]):
    #     #     for j in range(board.shape[1]):
    #     #         if board[i,j]>1 and board[i,j]<value:
    #     #             value = board[i, j]
    #     #             target = (i, j)
    #     target = np.where(board==target_value)
    #     # print("target values are ---------------------")
    #     # print(target)

    #     action_mask = np.zeros(4, dtype=np.float32)

    #     for i in range(len(directions)):
    #         x = action_node[0] + directions[i][0]
    #         y = action_node[1] + directions[i][1]
    #         if 0 <= x < board.shape[0] and 0 <= y < board.shape[1]:
    #             if (x,y) == target or board[(x,y)] == 0:
    #                 action_mask[i] = 1

    #     return action_mask


# # Create and wrap the environment
# env = DummyVecEnv([lambda: gym.make('Breakout-v0')])

# model = A2C(CRPolicy, env, verbose=1)
# # Train the agent
# model.learn(total_timesteps=10000)
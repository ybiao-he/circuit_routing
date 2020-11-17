import numpy as np
import random
import core as core

import joblib
import os.path as osp
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PPOLoss(keras.losses.Loss):
    def __init__(self, clip_ratio=0.2, act_dim=4, name="ppo_loss"):
        super().__init__(name=name)
        self.clip_ratio = clip_ratio
        self.act_dim = act_dim

    def call(self, y_true, y_pred):
        # y_true should contain many types of data: a, adv, p_old
        a = tf.dtypes.cast(y_true[0], tf.int32)
        adv = y_true[1]
        logp_old = y_true[2]
        logp = tf.math.reduce_sum(tf.one_hot(a, depth=self.act_dim, dtype=tf.float32) * y_pred, axis=1)
        ratio = tf.math.exp(logp - logp_old)          # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv>0, (1+self.clip_ratio)*adv, (1-self.clip_ratio)*adv)
        pi_loss = -tf.math.reduce_mean(tf.minimum(ratio * adv, min_adv))
        return pi_loss

class policy(object):

    def __init__(self, env, rl_algm="vpg"):


        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n

        # Share information about action space with policy architecture
        ac_kwargs = dict()
        ac_kwargs['action_space'] = env.action_space
           
        self.p_model, self.v_model = self.actor_critic(hidden_sizes=(32,32), **ac_kwargs)

        self.algorithm = rl_algm
        if rl_algm == "vpg":
            self.vpg()
        elif rl_algm == "ppo":
            self.ppo()
        else:
            print("please select rl algorithm: vpg or ppo")


    def actor_critic(self, hidden_sizes=(64,64), activation=tf.tanh, action_space=None):

        act_dim = action_space.n
        p_model = self.mlp(list(hidden_sizes)+[act_dim], activation, None)
        p_model.layers[-1].activation = keras.activations.get('softmax')

        v_model = self.mlp(list(hidden_sizes)+[1], activation, None)

        return p_model, v_model


    def mlp(self, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):

        inputs = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(hidden_sizes[0], activation=activation)(inputs)
        for h in hidden_sizes[1:-1]:
            x = tf.keras.layers.Dense(h, activation=activation)(x)
        outputs = tf.keras.layers.Dense(hidden_sizes[-1], activation=output_activation)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def predict_probs(self, obs):
        new_shape = (1,) + obs.shape
        return self.p_model.predict(np.reshape(obs, new_shape))
        # return self.sess.run(self.logp_all, feed_dict={self.x_ph: np.reshape(obs, new_shape)})

    def get_prob_act(self, obs, act):
        new_shape = (1,) + obs.shape
        return self.p_model.predict(np.reshape(obs, new_shape))[0][act]

    def predict_value(self, obs):
        new_shape = (1,) + obs.shape
        return self.v_model.predict(np.reshape(obs, new_shape))

    def get_act(self, obs):
        new_shape = (1,) + obs.shape
        action_probs = self.p_model.predict(np.reshape(obs, new_shape))
        return np.random.choice(self.act_dim, p=np.squeeze(action_probs))


    def ppo(self, clip_ratio=0.2):

        self.p_model.compile(optimizer=keras.optimizers.Adam(), loss=PPOLoss(clip_ratio=clip_ratio))
        self.v_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())

        self.p_model.summary()

    def update(self, rl_buffer, train_pi_iters=50, train_v_iters=50, target_kl=0.01):

        # buf_len = rl_buffer.get_length()
        buffer_data = rl_buffer.get()

        X = buffer_data[0]
        y_p = [buffer_data[1], buffer_data[2], buffer_data[4]]
        y_v = buffer_data[3]
        self.p_model.fit(X, y_p, epochs=1)
        self.v_model.fit(X, y_v, epochs=1)
import numpy as np
import random
import core as core
import tensorflow as tf

import joblib
import os.path as osp
import shutil

class policy(object):

    def __init__(self, env, rl_algm="vpg"):


        self.obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.n
        
        self.sess = tf.Session()

        # Share information about action space with policy architecture
        ac_kwargs = dict()
        ac_kwargs['action_space'] = env.action_space
        
        # Inputs to computation graph
        self.x_ph, self.a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
        
        self.pi, self.logp, self.logp_pi, self.v, self.logp_all = core.mlp_actor_critic(self.x_ph, self.a_ph, hidden_sizes=(32,32), **ac_kwargs)

        self.algorithm = rl_algm
        if rl_algm == "vpg":
            self.vpg()
        elif rl_algm == "ppo":
            self.ppo()
        else:
            print("please select rl algorithm: vpg or ppo")

        # self.sess.run(tf.global_variables_initializer())
        self.reset()

    def reset(self):
        self.sess.run(tf.global_variables_initializer())

    def predict_probs(self, obs):
        new_shape = (1,) + obs.shape
        return self.sess.run(self.logp_all, feed_dict={self.x_ph: np.reshape(obs, new_shape)})

    def get_prob_act(self, obs, act):
        new_shape = (1,) + obs.shape
        new_a_shape = (-1,)
        return self.sess.run(self.logp, feed_dict={self.x_ph: np.reshape(obs, new_shape), self.a_ph: np.reshape(act, new_a_shape)})[0]

    def predict_value(self, obs):
        new_shape = (1,) + obs.shape
        return self.sess.run(self.v, feed_dict={self.x_ph: np.reshape(obs, new_shape)})[0]

    def predict_act(self, obs):
        new_shape = (1,) + obs.shape
        return self.sess.run(self.pi, feed_dict={self.x_ph: np.reshape(obs, new_shape)})

    def ppo(self, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3):

        adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)
        logp_all_old = core.placeholder(self.act_dim)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, adv_ph, ret_ph, logp_old_ph, logp_all_old]

        # PPO objectives
        ratio = tf.exp(self.logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((ret_ph - self.v)**2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(logp_old_ph - self.logp)
        # self.approx_kl = tf.reduce_sum(tf.math.multiply(tf.math.log(self.p_all) - tf.math.log(p_all_old), self.p_all))     # a sample estimate for KL-divergence, easy to compute
        # approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute

        # Optimizers
        self.train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr, name='NewAdam_pi').minimize(self.pi_loss)

        self.train_v = tf.train.AdamOptimizer(learning_rate=vf_lr, name='NewAdam_v').minimize(self.v_loss)


    def vpg(self, pi_lr=3e-4, vf_lr=1e-3):

        adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)
        logp_all_old = core.placeholder(self.act_dim)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, adv_ph, ret_ph, logp_old_ph, logp_all_old]

        # VPG objectives
        self.pi_loss = -tf.reduce_mean(self.logp * adv_ph)
        self.v_loss = tf.reduce_mean((ret_ph - self.v)**2)

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(logp_old_ph - self.logp)
        # self.approx_kl = tf.reduce_sum(tf.math.multiply(tf.math.log(self.p_all) - tf.math.log(p_all_old), self.p_all))      # a sample estimate for KL-divergence, easy to compute
        # approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute

        # Optimizers
        optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
        self.train_pi = optimizer.minimize(self.pi_loss, tf.train.get_global_step())
        # self.train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=vf_lr)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5)
        self.train_v = optimizer.minimize(self.v_loss, tf.train.get_global_step())
        # self.train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)

    def update(self, rl_buffer, train_pi_iters=50, train_v_iters=50, target_kl=0.01):

        buf_len = rl_buffer.get_length()
        buffer_data = rl_buffer.get()
        # inputs = {k:v for k,v in zip(self.all_phs, buffer.get())}
        # print(inputs)

        # pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        # train policy network
        batch = []
        loss_values = []
        kls = []
        for i in range(train_pi_iters):
            kl = 0
            start_idx = 0
            batch_size = 100
            while start_idx<len(buffer_data[0]):
                if start_idx+batch_size < len(buffer_data[0]):
                    batch = [data[start_idx:start_idx+batch_size] for data in buffer_data]
                else:
                    batch = [data[start_idx:-1] for data in buffer_data]
                start_idx += start_idx+batch_size

                inputs = {k:v for k,v in zip(self.all_phs, batch)}
                # print(inputs)
                # print(self.sess.run(self.p_all, feed_dict=inputs))
                _, kl_tem, loss_value = self.sess.run([self.train_pi, self.approx_kl, self.pi_loss], feed_dict=inputs)
                kl += kl_tem
                print(loss_value)
                loss_values.append(loss_value)
            kl /= buf_len
            kls.append(kl)
            print("kl values are:")
            print(kl)
            if self.algorithm == "vpg":
                break

            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break

        # train value network
        for i in range(train_v_iters):
            start_idx = 0
            batch_size = 100
            while start_idx<len(buffer_data[0]):
                if start_idx+batch_size < len(buffer_data[0]):
                    batch = [data[start_idx:start_idx+batch_size] for data in buffer_data]
                else:
                    batch = [data[start_idx:-1] for data in buffer_data]
                start_idx += start_idx+batch_size

                inputs = {k:v for k,v in zip(self.all_phs, batch)}
                self.sess.run(self.train_v, feed_dict=inputs)
            # print(i)

        return loss_values, kls

    def tf_save(self):

        saver = tf.train.Saver()
        # Save model weights to disk
        model_path = './save_tf1_model'
        save_path = saver.save(self.sess, model_path)
        print("Model saved in file: %s" % save_path)
        self.sess.close()

    def tf_restore(self):
        # Restore model weights from previously saved model
        saver = tf.train.Saver()
        model_path = './save_tf1_model'
        saver.restore(self.sess, model_path)
        print("Model restored from file: %s" % model_path)

    # following funcions are for MCTS
    def rollout(self, state):

        route_paths = []
        direction_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while not state.isTerminal():

            # o = state.board_embedding()
            # # print(self.predict_act(o), np.exp(self.predict_probs(o)))

            # a = self.predict_act(o)
            # action = direction_list[a[0]]
            action = self.choose_action(state)
            node = state.action_node
            node = tuple(map(sum, zip(action, node)))
            route_paths.append(node)

            # logp_all = np.exp(policy.predict_probs(o))
            # print(logp_all)
            state = state.takeAction(action)

        return state.getReward(), route_paths


    def choose_action(self, state):

        direction_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        o = state.board_embedding()
        actions_probs = np.exp(self.predict_probs(o))[0]

        for i in range(len(direction_list)):
            d = direction_list[i]
            x = state.action_node[0] + d[0]
            y = state.action_node[1] + d[1]
            if 0 <= x < state.board.shape[0] and 0 <= y < state.board.shape[1] and (state.board[(x,y)] == 0 or (x,y) == state.finish[state.pairs_idx]):
                continue
            else:
                actions_probs[i] = 0 

        if sum(actions_probs) == 0:
            actions_probs = np.ones(actions_probs.shape)
        actions_probs = actions_probs/sum(actions_probs)
        action_idx = np.random.choice(len(direction_list), p=actions_probs)
        return direction_list[action_idx]

    # def tf_simple_save(self):
    #     """
    #     Uses simple_save to save a trained model, plus info to make it easy
    #     to associated tensors to variables after restore. 
    #     """
    #     inputs={"x_ph": self.x_ph, "a_ph": self.a_ph}
    #     outputs={"pi": self.pi, "logp": self.logp, "logp_pi": self.logp_pi, "v": self.v, "logp_all": self.logp_all}

    #     self.tf_saver_elements = dict(session=self.sess, inputs=inputs, outputs=outputs)
    #     self.tf_saver_info = {'inputs': {k:v.name for k,v in inputs.items()},
    #                           'outputs': {k:v.name for k,v in outputs.items()}}

    #     fpath = 'tf1_save'
    #     fpath = osp.join('./', fpath)

    #     if osp.exists(fpath):
    #         # simple_save refuses to be useful if fpath already exists,
    #         # so just delete fpath if it's there.
    #         shutil.rmtree(fpath)
    #     tf.saved_model.simple_save(export_dir=fpath, **self.tf_saver_elements)
    #     joblib.dump(self.tf_saver_info, osp.join(fpath, 'model_info.pkl'))

    # def load_tf_graph(self, fpath):
    #     """
    #     Loads graphs saved by Logger.

    #     Will output a dictionary whose keys and values are from the 'inputs' 
    #     and 'outputs' dict you specified with logger.setup_tf_saver().

    #     Args:
    #         sess: A Tensorflow session.
    #         fpath: Filepath to save directory.

    #     Returns:
    #         A dictionary mapping from keys to tensors in the computation graph
    #         loaded from ``fpath``. 
    #     """
    #     tf.saved_model.loader.load(
    #                 self.sess,
    #                 [tf.saved_model.tag_constants.SERVING],
    #                 fpath
    #             )
    #     model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
    #     graph = tf.get_default_graph()
    #     model = dict()
    #     model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
    #     model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})

    #     return model



import numpy as np
import random

import core as core
import tensorflow as tf

import joblib
import os.path as osp
import shutil

class policies(object):

    def __init__(self, model_path):
        
        self.sess = tf.Session()
        model = self.load_tf_graph(model_path)
        self.extract_tensor_from_model(model)

        self.direction_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]

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

    def load_tf_graph(self, fpath):
        """
        Loads graphs saved by Logger.

        Will output a dictionary whose keys and values are from the 'inputs' 
        and 'outputs' dict you specified with logger.setup_tf_saver().

        Args:
            sess: A Tensorflow session.
            fpath: Filepath to save directory.

        Returns:
            A dictionary mapping from keys to tensors in the computation graph
            loaded from ``fpath``. 
        """
        tf.saved_model.loader.load(
                    self.sess,
                    [tf.saved_model.tag_constants.SERVING],
                    fpath
                )
        model_info = joblib.load(osp.join(fpath, 'model_info.pkl'))
        graph = tf.get_default_graph()
        model = dict()
        model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['inputs'].items()})
        model.update({k: graph.get_tensor_by_name(v) for k,v in model_info['outputs'].items()})

        return model

    def extract_tensor_from_model(self, model):

        self.logp_all = model['logp_all']
        self.logp = model['logp']
        self.v = model['v']
        self.pi = model['pi']
        self.x_ph = model['x_ph']
        self.a_ph = model['a_ph']

    def rollout(self, state):

        route_paths = []


        while not state.isTerminal():

            o = state.board_embedding()
            print(o)

            a = self.predict_act(o)
            action = self.direction_list[a[0]]
            node = state.action_node
            node = tuple(map(sum, zip(action, node)))
            route_paths.append(node)

            # logp_all = np.exp(policy.predict_probs(o))
            # print(logp_all)
            state = state.takeAction(action)

        return state.getReward(), route_paths

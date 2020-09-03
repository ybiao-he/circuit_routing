import numpy as np

import random

import core_tf as core

import tensorflow as tf

class policies(object):

    def __init__(self, obs_dim, act_dim):

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Share information about action space with policy architecture
        ac_kwargs = dict()
        ac_kwargs['action_dim'] = self.act_dim
        
        # Inputs to computation graph
        self.x_ph = core.placeholder(self.obs_dim)
        self.a_ph = tf.placeholder(dtype=tf.int32, shape=(None,))

        self.pi, self.p, self.p_pi, self.v, self.p_all = core.actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

        self.ppo()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.direction_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def network(self, state):
        """
        This function is for DNN-based policy without DFS. It is not used in our currecnt method
        """
        route_paths = []
        while not state.isTerminal():
            try:
                feature = np.array([state.board.ravel()])
                predict = self.policy_model.predict(feature)
                sort_action = np.flip( np.argsort(predict) )[0]
                action_set = state.getPossibleActions()

                for idx in sort_action:
                    if self.direction_list[idx] in action_set:
                        action = self.direction_list[idx]
                        break

            except IndexError:
                raise Exception("Non-terminal state has no possible actions: ")
            route_paths.append(action)
            state = state.takeAction(action)

        return state.getReward(), [route_paths]

    def indices_to_one_hot(self, data, nb_classes):
        """Convert an iterable of indices to one-hot encoded labels."""
        targets = [0]*nb_classes
        targets[data] = 1
        return np.array(targets)

    def predict_probs(self, obs):
        new_shape = (1,) + obs.shape
        return self.sess.run(self.p_all, feed_dict={self.x_ph: np.reshape(obs, new_shape)})

    def get_prob_act(self, obs, act):
        new_shape = (1,) + obs.shape
        return self.sess.run(self.p, feed_dict={self.x_ph: np.reshape(obs, new_shape), self.a_ph: self.indices_to_one_hot(act, self.act_dim)})[0]

    def predict_value(self, obs):
        new_shape = (1,) + obs.shape
        return self.sess.run(self.v, feed_dict={self.x_ph: np.reshape(obs, new_shape)})[0]

    def randomRoute(self, state):

        route_paths = []

        while not state.isTerminal():

            obs = np.copy(state.board_embedding())
            pin_idx = state.pairs_idx

            path = self.randomDFS(obs, state.action_node, state.finish[state.pairs_idx], pin_idx)

            if len(path)==1:
                # print("failed to find a path to the target node")
                return 1/(obs.shape[0]*obs.shape[1]), route_paths
            else:
                route_paths.append(path)
                path.pop(0)
                for p in path:
                    action = tuple(np.subtract(p, state.action_node))
                    state = state.takeAction(action)

        return state.getReward(), route_paths

    def randomDFS(self, obs, s, t, pin_idx):

        if len(obs.shape)==3:
            board = obs[:,:,0]
        else:
            board = obs[:,:,:,0]

        path_queue = [s]
        node = s
        pre_node = s

        # pin_max = int(np.amax(board))
        # path_num = pin_max + 1

        while t not in path_queue:

            actions = self.getPossibleActions(board, node, t)
            
            if len(actions) != 0:
                # randomly choose an action, revise it by possibilities
                # action = random.choice(actions)
                # bh = board.shape[0]
                # bw = board.shape[1]
                # obs_tem = np.reshape(obs, (1, bh, bw, 2))
                action_dist = self.predict_probs(obs)[0]
                # print(action_dist)
                action = self.get_action(actions, action_dist)

                node = tuple(map(sum, zip(action, node)))
                board[node] = pin_idx
                board[pre_node] = 1
                pre_node = node
                # path_num += 1
                path_queue.append(node)
            elif len(path_queue) > 1:
                pre_node = path_queue.pop()
                node = path_queue[-1]
                
                board[pre_node] = 1
                board[node] = pin_idx
                # path_num -= 1
            else:
                break
        return path_queue

    def getPossibleActions(self, board, node, t):

        possible_actions = []
        for d in self.direction_list:
            x = node[0] + d[0]
            y = node[1] + d[1]
            if 0 <= x < board.shape[0] and 0 <= y < board.shape[1]:
                if (x,y) == t:
                    possible_actions.append((d[0], d[1]))
                    break
                elif board[(x,y)] == 0:
                    possible_actions.append((d[0], d[1]))

        return possible_actions

    def get_action(self, possible_actions, action_dist):
        act_idx_to_poss = {}
        for act in possible_actions:
            idx_action = self.direction_list.index(act)
            act_idx_to_poss[idx_action] = action_dist[idx_action]

        # normalize the distribution of actions
        # print(act_idx_to_poss)
        poss_sum = sum(act_idx_to_poss.values())
        if poss_sum == 0:
            for a in act_idx_to_poss:
                act_idx_to_poss[a] = 1/len(act_idx_to_poss.values()) 
        else:    
            for a in act_idx_to_poss:
                act_idx_to_poss[a] /= poss_sum

        # print(act_idx_to_poss)
        # get actions according to the normalized distribution
        # print(act_idx_to_poss)
        ret_action = np.random.choice(list(act_idx_to_poss.keys()), p=list(act_idx_to_poss.values()))
        return self.direction_list[ret_action]

    def ppo(self, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3):

        adv_ph, ret_ph, p_old_ph = core.placeholders(None, None, None)

        # Need all placeholders in *this* order later (to zip with data from buffer)
        self.all_phs = [self.x_ph, self.a_ph, adv_ph, ret_ph, p_old_ph]

        # PPO objectives
        ratio = self.p / (p_old_ph + 1e-8)       # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((ret_ph - self.v)**2, name="v_loss")

        # Info (useful to watch during learning)
        self.approx_kl = tf.reduce_mean(tf.math.log(p_old_ph) - tf.math.log(self.p))      # a sample estimate for KL-divergence, easy to compute
        # approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute

        # Optimizers
        self.train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)

        self.train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(self.v_loss)


    def ppo_update(self, rl_buffer, train_pi_iters=50, train_v_iters=50, target_kl=0.05):

        buffer_data = rl_buffer.get()
        # inputs = {k:v for k,v in zip(self.all_phs, buffer.get())}
        # print(inputs)

        # pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        # train policy network
        batch = []
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
                _, kl_tem, loss_value = self.sess.run([self.train_pi, self.approx_kl, self.pi_loss], feed_dict=inputs)
                kl += kl_tem
                # print(loss_value)
            print("kl values are:")
            print(kl)
            # # print(sess.run(pi_loss, feed_dict=inputs))
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
            print(i)


        # pi_l_new, v_l_new, kl = self.sess.run([self.pi_loss, self.v_loss, self.approx_kl], feed_dict=inputs)
        # print(pi_l_new, v_l_new, kl)
import numpy as np

import random

import core_tf as core

import tensorflow as tf

class policies(object):

    def __init__(self, obs_dim, act_dim):

        # Share information about action space with policy architecture
        ac_kwargs = dict()
        ac_kwargs['action_dim'] = act_dim

        # Inputs to computation graph
        self.x_ph = core.placeholder(obs_dim)
        self.a_ph = tf.placeholder(dtype=tf.int32, shape=(None,))

        self.pi, self.p, self.p_pi, self.v, self.p_all = core.actor_critic(self.x_ph, self.a_ph, **ac_kwargs)

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

    def predict_probs(self, obs):
        return self.sess.run(self.p_all, feed_dict={self.x_ph: obs})

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
                bh = board.shape[0]
                bw = board.shape[1]
                obs_tem = np.reshape(obs, (1, bh, bw, 2))
                action_dist = self.predict_probs(obs_tem)[0]
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

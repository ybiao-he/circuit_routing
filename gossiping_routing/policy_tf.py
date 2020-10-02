import numpy as np

import random

import tensorflow as tf

import sys
# np.set_printoptions(threshold=sys.maxsize)

class policies(object):

    def __init__(self):

        self.direction_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def rollout(self, state):

        route_paths = []

        while not state.isTerminal():

            obs = np.copy(state.board)
            pin_idx = state.pairs_idx

            # path = self.randomDFS(obs, state.action_node, state.finish[state.pairs_idx], pin_idx)
            path = self.gossiping_route(obs, pin_idx)
            
            print(path)

            # if using DFS, this if-statement condition is len(path)==1
            if len(path)==0:
                # print("failed to find a path to the target node")
                return state.getReward(fail=True), route_paths
            else:
                path.append(state.finish[pin_idx])
                route_paths.append(path)
                path.pop(0)
                for p in path:
                    action = tuple(np.subtract(p, state.action_node))
                    state = state.takeAction(action)

        return state.getReward(), route_paths

    def gossiping_route(self, obs, pin_idx):

        from gossiping_search import connect

        e_param = 0.6
        paths = connect(obs, pin_idx, e_param)

        if len(paths)==0:
            return paths
        return paths[0]


    def randomDFS(self, obs, s, t, pin_idx):

        board = obs

        path_queue = [s]
        node = s
        pre_node = s

        while t not in path_queue:

            actions = self.getPossibleActions(board, node, t)
            
            if len(actions) != 0:
                # randomly choose an action, revise it by possibilities
                # action = random.choice(actions)
                # bh = board.shape[0]
                # bw = board.shape[1]
                # obs_tem = np.reshape(obs, (1, bh, bw, 2))
                action_dist = np.array([0.25,0.25,0.25,0.25])
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
        # print(path_queue)
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

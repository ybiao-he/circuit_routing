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

        # e_param = 0.6
        e_param = 0.1 * random.randint(1,9)
        paths = connect(obs, pin_idx, e_param)

        if len(paths)==0:
            return paths
        return paths[0]

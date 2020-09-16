from __future__ import division

from copy import deepcopy
from scipy.spatial import distance

import numpy as np


class CREnv(object):
    """
    This is an env for circuit routing. 
    """
    def __init__(self, board):
        super(CREnv, self).__init__()

        self.board_backup = np.copy(board)

        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # self.reset()

    def reset(self):

        self.board = np.copy(self.board_backup)

        self.max_value = np.amax(self.board)+1
        self.pairs_idx = 2

        self.path_length = 0
        self.max_pair_idx = 2

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] != 1 and self.board[i,j] != 0:
                    if self.board[i,j]<0:
                        self.finish[-self.board[i,j]] = (i,j)
                    else:
                        self.start[self.board[i,j]] = (i,j)
                    self.max_pair_idx = max(self.max_pair_idx, abs(self.board[i,j]))
                self.board[i,j] = abs(self.board[i,j])
        # initialize the action node
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = self.max_value

        self.board[self.finish[self.pairs_idx]] = self.max_value

        state = self.board_embedding()

        return state

    def board_embedding(self):

        board = self.board.reshape(40,40,1).astype(np.float32)
        net_idx_mat = np.ones(board.shape)*self.max_value
        one_sample = np.float32( np.concatenate((board, net_idx_mat), axis=2) )
        state = np.array(one_sample)/(self.max_value-1)

        return state

    def getPossibleActions(self):

        possible_actions = []
        for d in self.directions:
            x = self.action_node[0] + d[0]
            y = self.action_node[1] + d[1]
            if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
                if (x,y) == self.finish[self.pairs_idx]:
                    possible_actions = [(d[0], d[1])]
                    break
                elif self.board[(x,y)] == 0:
                    possible_actions.append((d[0], d[1]))

        return possible_actions

    def legalAction(self, action):

        action_tmp = self.directions[action]

        x = self.action_node[0]+action_tmp[0]
        y = self.action_node[1]+action_tmp[1]
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if (x,y) == self.finish[self.pairs_idx] or self.board[(x,y)] == 0:
                return True        
            else:
                return False
        return False

    def step(self, action):

        action_tmp = self.directions[action]

        # self.board[self.action_node] = 1
        
        self.action_node = (self.action_node[0]+action_tmp[0], self.action_node[1]+action_tmp[1])

        if self.board[self.action_node] == 0:
            self.board[self.action_node] = self.max_value
        elif self.action_node == self.finish[self.pairs_idx]:
            # self.board[self.action_node] = 1
            self.pairs_idx += 1
            self.action_node = self.start.get(self.pairs_idx)
            if self.action_node is not None:
                self.board[self.action_node] = self.max_value

        self.board[self.finish.get(self.pairs_idx)] = self.max_value

        state = self.board_embedding()

        # print(np.sum(state))
        self.path_length += 1

        self.max_value += 1

        reward = self.getReward()

        done = self.isTerminal()

        info = {}

        # state = np.array(self.board.reshape(40,40,1)).astype(np.float32)/(self.max_value-1.0)
        
        return state, reward, done, info

    def isTerminal(self):

        if self.action_node is None or len(self.getPossibleActions()) == 0:
            return True

        return False

    def getReward(self):

        if len(self.getPossibleActions()) == 0:
            left_dist = distance.cityblock(self.action_node, self.finish[self.pairs_idx])
            for i in range(self.pairs_idx+1, int(self.max_pair_idx+1)):
                left_dist += distance.cityblock(self.start[i], self.finish[i])
            return -left_dist*2.0
            # return -self.board.shape[0]*self.board.shape[1]/10
        # elif self.action_node is None:
        #     return -self.path_length/10
        return -1.0

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        print("-------")
        print(self.action_node)

    def close(self):
        pass
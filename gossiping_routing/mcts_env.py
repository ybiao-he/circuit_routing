# In this environment, the filled numbers for route are only 1s.
# The filled number for terminal nodes are calculated as i+1, where i=1,2,3,...
from __future__ import division

from copy import deepcopy
from copy import copy

from scipy.spatial import distance

import numpy as np

class circuitBoard():
    def __init__(self, board):

        self.board_backup = np.copy(board)

        self.board = np.copy(board)

        self.pairs_idx = 2

        self.direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

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
                # self.board[i,j] = abs(self.board[i,j])

        # initialize the action node
        self.action_node = copy(self.start[self.pairs_idx])

    def getPossibleActions(self):
        possibleActions = []
        for d in self.direction_map:

            x = self.action_node[0] + d[0]
            y = self.action_node[1] + d[1]
            if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
                if self.board[(x,y)] == 0 or (x,y) == self.finish[self.pairs_idx]:
                    possibleActions.append((d[0], d[1]))

        return possibleActions

    def takeAction(self, action):

        newState = deepcopy(self)

        newState.board[newState.action_node] = 1

        newState.action_node = (newState.action_node[0]+action[0], newState.action_node[1]+action[1])

        if newState.board[newState.action_node] == 0:
            newState.board[newState.action_node] = newState.pairs_idx
        elif newState.action_node == newState.finish[newState.pairs_idx]:
            newState.board[newState.action_node] = 1
            newState.pairs_idx += 1
            newState.action_node = newState.start.get(newState.pairs_idx)

        newState.path_length += 1
        return newState

    def isTerminal(self):

        if self.action_node is None or len(self.getPossibleActions()) == 0:
            return True

        return False

    def getReward(self, fail=False):

        if fail or len(self.getPossibleActions()) == 0:
            left_dist = distance.cityblock(self.action_node, self.finish[self.pairs_idx])
            for i in range(self.pairs_idx+1, int(self.max_pair_idx+1)):
                left_dist += distance.cityblock(self.start[i], self.finish[i])
            # return left_dist*2.0
            return 1/(left_dist*5.0+self.path_length)
        
        elif self.action_node is None:
            return 1/(self.path_length+1)

        return 0
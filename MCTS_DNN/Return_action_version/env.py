# In this environment, the filled numbers for route are only 1s.
# The filled number for terminal nodes are calculated as i+1, where i=1,2,3,...
from __future__ import division

from copy import deepcopy

import numpy as np

class circuitBoard():
    def __init__(self, board):

        self.board = np.copy(board)

        self.pairs_idx = 2

        self.direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

        self.path_length = 0

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] != 1 and self.board[i,j] != 0:
                    if self.board[i,j] in self.start:
                        self.finish[self.board[i,j]] = (i,j)
                    else:
                        self.start[self.board[i,j]] = (i,j)

        # initialize the action node
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = 1

    def getPossibleActions(self):
        possibleActions = []
        for d in self.direction_map:

            x = self.action_node[0] + d[0]
            y = self.action_node[1] + d[1]
            if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
                if self.board[(x,y)] == 0 or self.board[(x,y)] == self.pairs_idx:
                    possibleActions.append((d[0], d[1]))

        return possibleActions

    def takeAction(self, action):

        newState = deepcopy(self)
        
        newState.action_node = (newState.action_node[0]+action[0], newState.action_node[1]+action[1])

        if newState.board[newState.action_node] == 0:
            newState.board[newState.action_node] = 1
        elif newState.action_node == newState.finish[newState.pairs_idx]:
            newState.board[newState.action_node] = 1
            newState.pairs_idx += 1
            newState.action_node = newState.start.get(newState.pairs_idx)
            if newState.action_node is not None:
                newState.board[newState.action_node] = 1

        newState.path_length += 1
        return newState

    def isTerminal(self):

        if self.action_node is None or len(self.getPossibleActions()) == 0:
            return True

        return False

    def getReward(self):
        if self.action_node is None:
            return 1/(self.path_length+1)
        elif len(self.getPossibleActions()) == 0:
            return 1/(self.board.shape[0]*self.board.shape[1])
        return 0
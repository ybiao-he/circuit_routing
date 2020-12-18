# In this environment, the filled numbers for route are only 1s.
# The filled number for terminal nodes are calculated as i+1, where i=1,2,3,...
from __future__ import division

from copy import deepcopy
from copy import copy
import random
import numpy as np
from scipy.spatial import distance

class circuitBoard():
    def __init__(self, board):

        self.board_backup = np.copy(board)

        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.reset()

    def reset(self):

        self.board = np.copy(self.board_backup)

        self.head_value = 20
        self.pairs_idx = 2

        self.max_pair = int(np.amax(self.board))
        # self.max_pair = 3

        self.path_length = 0

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] != 1 and self.board[i,j] != 0:
                    if self.board[i,j]<0:
                        self.finish[-self.board[i,j]] = (i,j)
                        self.board[i,j] = abs(self.board[i,j])
                    else:
                        self.start[self.board[i,j]] = (i,j)
                # self.board[i,j] = abs(self.board[i,j])
        # initialize the action node
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = self.head_value
        # self.board[self.finish[self.pairs_idx]] = self.head_value

        return self.board_embedding()

    def board_embedding(self):

        from sklearn.decomposition import PCA

        n_pairs = self.max_pair-1
        nets_matrix = np.zeros((n_pairs,4))
        obs_matrix = []
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] != 0 and self.board[i,j] <= self.max_pair:
                    if self.board[i,j] == 1:
                        obs_matrix.append([i,j])
                    elif self.board[i,j] > 0:
                        nets_matrix[int(abs(self.board[i,j]))-2][0] = i
                        nets_matrix[int(abs(self.board[i,j]))-2][1] = j
                    else:
                        nets_matrix[int(abs(self.board[i,j]))-2][2] = i
                        nets_matrix[int(abs(self.board[i,j]))-2][3] = j

        nets_matrix = np.delete(nets_matrix, self.pairs_idx-2, 0)

        pca_nets = PCA(n_components=1)
        pca_nets.fit(nets_matrix)
        nets_vector = pca_nets.components_[0]

        pca_obs = PCA(n_components=1)
        pca_obs.fit(obs_matrix)
        obs_vector = pca_obs.components_[0]

        # print(nets_vector.tolist()+obs_vector.tolist())

        state = np.array(list(self.action_node)+list(self.finish[self.pairs_idx]))/30
        # state = np.concatenate(( state, np.array(nets_vector.tolist()+obs_vector.tolist()) ), axis=0)

        return state


    def getPossibleActions(self):
        possibleActions = []
        for d in self.directions:

            x = self.action_node[0] + d[0]
            y = self.action_node[1] + d[1]
            if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
                if self.board[(x,y)] == 0 or self.board[(x,y)] == self.pairs_idx:
                    possibleActions.append((d[0], d[1]))

        if len(possibleActions)==0:
            possibleActions.append(random.choice(self.directions))

        return possibleActions

    def takeAction(self, action):

        newState = deepcopy(self)

        action_node_pre = newState.action_node

        newState.board[newState.action_node] = 1
        
        newState.action_node = (newState.action_node[0]+action[0], newState.action_node[1]+action[1])

        x = newState.action_node[0]
        y = newState.action_node[1]
        if 0 <= x < newState.board.shape[0] and 0 <= y < newState.board.shape[1]:
            if newState.action_node == newState.finish[newState.pairs_idx] and newState.pairs_idx<newState.max_pair:
                if newState.blocking_nets_Lee():
                    newState.board[newState.action_node] = newState.head_value+10
                else:
                    newState.pairs_idx += 1
                    newState.board[newState.action_node] = 1
                    newState.action_node = newState.start[newState.pairs_idx]
                    newState.board[newState.action_node] = newState.head_value
            else:
                newState.board[newState.action_node] = newState.board[newState.action_node]*10 + newState.head_value
        else:
            newState.action_node = action_node_pre
            newState.board[newState.action_node] = newState.head_value+10

        newState.path_length += 1

        return newState

    def isTerminal(self):

        if self.board[self.action_node] > self.head_value:
            return True

        return False

    def getReward(self):
        if self.board[self.action_node] > self.head_value:
            left_dist = distance.cityblock(self.action_node, self.finish[self.pairs_idx])
            # print(self.pairs_idx, self.max_pair)
            for i in range(self.pairs_idx+1, self.max_pair):
                left_dist += distance.cityblock(self.start[i], self.finish[i])
            return -left_dist*10-self.path_length

        return 0

    def blocking_nets_Lee(self):

        print("checking this------------------------------------------------------")

        from LeeAlgm import Field
        for net_idx in range(self.pairs_idx+1, self.max_pair+1):
            barriers = []
            start = []
            finish = []
            for i in range(self.board.shape[0]):
                for j in range(self.board.shape[1]):
                    if self.board[i,j] != 0:
                        barriers.append((i,j))
                    if (i, j) == self.start[net_idx]:
                        barriers.remove((i,j))
                        start = (i,j)
                    if (i, j) == self.finish[net_idx]:
                        barriers.remove((i,j))
                        finish = (i,j)
            field = Field(len=30, start=start, finish=finish, barriers=barriers)
            field.emit()
            try:
                path = field.get_path()
                if len(path) == 0:
                    return True
            except:
                return True
        return False
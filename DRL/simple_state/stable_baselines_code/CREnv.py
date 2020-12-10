from __future__ import division

from copy import copy
from copy import deepcopy
from scipy.spatial import distance

import numpy as np
import gym
from gym import spaces
import os, random


class CREnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an env for circuit routing. 
    """
    def __init__(self, board=np.genfromtxt("./board_30.csv", delimiter=',')):
        super(CREnv, self).__init__()

        self.board_backup = np.copy(board)

        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.state_shape = (4,)

        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=1600, shape=self.state_shape, dtype=np.float32)

        self.reset()

    def reset(self):

        # self.board = deepcopy(self.board_backup)
        directory = './boards_30_30'
        filename = random.choice(os.listdir(directory))
        board_rand = os.path.join(directory, filename)
        self.board = np.genfromtxt(board_rand, delimiter=',')

        self.head_value = 20
        
        self.path_length = 0
        self.pairs_idx = 2

        self.max_pair = int(np.amax(self.board))
        # self.max_pair = 2

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] != 1 and self.board[i,j] != 0:
                    if abs(self.board[i,j])<=self.max_pair:
                        if self.board[i,j]<0:
                            self.finish[-self.board[i,j]] = (i,j)
                            self.board[i,j] = abs(self.board[i,j])
                        else:
                            self.start[self.board[i,j]] = (i,j)
                    else:
                        self.board[i,j] = 0
                # self.board[i,j] = abs(self.board[i,j])

        net_indices = list(range(self.pairs_idx, self.max_pair+1))
        random.shuffle(net_indices)

        start_tem = copy(self.start)
        finish_tem = copy(self.finish)
        for i, j in enumerate(self.start):
            self.start[j] = start_tem[net_indices[i]]
            self.finish[j] = finish_tem[net_indices[i]]
        # initialize the action node
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = self.head_value
        # self.board[self.finish[self.pairs_idx]] = self.head_value

        state = np.array(self.action_node+self.finish[self.pairs_idx])
        return state

    def step(self, action):

        action_tmp = self.directions[action]

        action_node_pre = self.action_node

        self.board[self.action_node] = 1
        
        self.action_node = (self.action_node[0]+action_tmp[0], self.action_node[1]+action_tmp[1])

        state = np.array(self.action_node+self.finish[self.pairs_idx])

        x = self.action_node[0]
        y = self.action_node[1]
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if self.action_node == self.finish[self.pairs_idx] and self.pairs_idx<self.max_pair:
                if self.blocking_nets_Lee():
                    self.board[self.action_node] = self.head_value+10
                else:
                    self.pairs_idx += 1
                    self.board[self.action_node] = 1
                    self.action_node = self.start[self.pairs_idx]
                    self.board[self.action_node] = self.head_value
                    state = np.array(self.action_node+self.finish[self.pairs_idx])
            else:
                self.board[self.action_node] = self.board[self.action_node]*10 + self.head_value
        else:
            self.action_node = action_node_pre
            self.board[self.action_node] = self.head_value+10

        # if self.board[self.action_node] == 0:
        #     self.board[self.action_node] = self.head_value
        # elif self.action_node == self.finish[self.pairs_idx]:
        #     self.board[self.action_node] = self.pairs_idx

        self.path_length += 1

        reward = self.getReward()

        done = self.isTerminal()

        info = {}

        return state, reward, done, info

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
            return float(-left_dist*10-self.path_length)

        return 0.0

    # def blocking_nets_Astar(self):

    #     print("cheking this------------------------------------------------------")

    #     from astar import astar
    #     start = []
    #     finish = []
    #     for net_idx in range(self.pairs_idx+1, self.max_pair+1):
    #         board_list = self.board.tolist()
    #         for i in range(self.board.shape[0]):
    #             for j in range(self.board.shape[1]):
    #                 if self.board[i,j] != 0:
    #                     board_list[i][j] = None
    #                 if (i, j) == self.start[net_idx]:
    #                     board_list[i][j] = 0
    #                     start = (i,j)
    #                 if (i, j) == self.finish[net_idx]:
    #                     board_list[i][j] = 0
    #                     finish = (i,j)
    #         path = astar(board_list, start, finish)
    #         if path is None:
    #             return True
    #     return False

    def blocking_nets_Lee(self):

        print("cheking this------------------------------------------------------")

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
            path = field.get_path()
            if len(path) == 0:
                return True
        return False

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        print("------")
        print(self.action_node)

    def close(self):
        pass

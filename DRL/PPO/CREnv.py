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
    def __init__(self):
        super(CREnv, self).__init__()

        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.state_shape = (4,)

        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=30, shape=self.state_shape, dtype=np.float32)

    def reset(self):

        self.board = np.genfromtxt("./board.csv", delimiter=',')
        # read board form dir
        # directory = './boards_30_30'
        # filename = random.choice(os.listdir(directory))
        # board_rand = os.path.join(directory, filename)
        # self.board = np.genfromtxt(board_rand, delimiter=',')

        self.head_value = 20
        
        self.path_length = 0

        self.max_pair = int(np.amax(self.board))
        # self.max_pair = 3
        self.connection = False
        self.collide = False

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if abs(self.board[i,j])>=2:
                    if abs(self.board[i,j])<=self.max_pair:
                        if self.board[i,j]<0:
                            self.finish[-self.board[i,j]] = (i,j)
                            self.board[i,j] = abs(self.board[i,j])
                        else:
                            self.start[self.board[i,j]] = (i,j)
                    else:
                        self.board[i,j] = 0
                # self.board[i,j] = abs(self.board[i,j])

        # initialize the action node
        self.pairs_idx = int(min(self.start.keys()))
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = self.head_value
        # print(self.start, self.finish)

        state = self.board_embedding()
        return state

    def step(self, action):

        action_tmp = self.directions[action]
        self.connection = False
        self.collide = False
        self.action_node_pre = self.action_node

        self.path_length += 1

        # pre-determine new action node
        self.action_node = (self.action_node[0]+action_tmp[0], self.action_node[1]+action_tmp[1])
        # check/adjust new action node and set its value
        x = self.action_node[0]
        y = self.action_node[1]

        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if self.action_node == self.finish[self.pairs_idx]:
                self.goto_new_net(True)
            elif self.board[self.action_node]!=0:
                self.collide = True
                self.goto_new_net(False)
            else:
                self.board[self.action_node_pre] = 1
                self.board[self.action_node] = self.head_value
        else:
            self.collide = True
            self.goto_new_net(False, out_range=True)

        reward = self.getReward()
        # while len(self.getPossibleActions())==0 and (not self.isTerminal()):
        #     self.action_node_pre = self.action_node
        #     self.goto_new_net(False)
        #     reward += self.getReward()

        state = self.board_embedding()

        done = self.isTerminal()

        info = {}

        return state, reward, done, info

    def goto_new_net(self, connection_sign, out_range=False):

        self.board[self.action_node_pre] = 1
        self.connection = connection_sign
        self.pairs_idx += 1
        if not out_range:
            self.board[self.action_node] = 1
            self.action_node_pre = self.action_node
        if self.pairs_idx<=self.max_pair:
            self.action_node = self.start[self.pairs_idx]
            self.board[self.action_node] = self.head_value

    def isTerminal(self):

        if self.pairs_idx>self.max_pair:
            return True

        return False


    def getReward(self):

        if self.connection:
            return 20
        if self.collide:
            left_dist = 5*distance.cityblock(self.action_node_pre, self.finish[self.pairs_idx-1])
            return -left_dist/10

        return -0.1

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

        ret = []
        for i in range(len(possible_actions)):
            ret.append(self.directions.index(possible_actions[i]))
        return ret

        # return possible_actions

    def board_embedding(self):

        if self.pairs_idx<=self.max_pair:
            dist_to_target = [i-j for i, j in zip(self.action_node, self.finish[self.pairs_idx])]
        else:
            dist_to_target = [i-j for i, j in zip(self.action_node, self.finish[self.pairs_idx-1])]
        # state = np.array(list(self.action_node)+list(self.finish[self.pairs_idx]))
        state = np.array(list(self.action_node)+dist_to_target)
        # state = np.concatenate(( state, np.array(nets_vector.tolist()+obs_vector.tolist()) ), axis=0)

        return state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        print("------")
        print(self.action_node)

    def close(self):
        pass

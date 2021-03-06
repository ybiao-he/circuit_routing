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

        self.board = np.genfromtxt("./board_30.csv", delimiter=',')
        # read board form dir
        # directory = './boards_30_30'
        # filename = random.choice(os.listdir(directory))
        # board_rand = os.path.join(directory, filename)
        # self.board = np.genfromtxt(board_rand, delimiter=',')

        self.head_value = 20
        
        self.path_length = 0

        self.max_pair = int(np.amax(self.board))
        # self.max_pair = 4

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

        self.pairs_idx = int(min(self.start.keys()))
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = self.head_value
        # print(self.start, self.finish)

        state = self.board_embedding()
        return state

    def step(self, action):

        action_tmp = self.directions[action]

        action_node_pre = self.action_node

        self.board[self.action_node] = 1
        
        self.action_node = (self.action_node[0]+action_tmp[0], self.action_node[1]+action_tmp[1])

        state = self.board_embedding()
        # state = np.array(self.action_node+self.finish[self.pairs_idx])

        x = self.action_node[0]
        y = self.action_node[1]
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if self.action_node == self.finish[self.pairs_idx] and self.pairs_idx<self.max_pair:
                # if self.blocking_nets_Lee():
                #     self.board[self.action_node] = self.head_value+10
                # else:
                self.pairs_idx += 1
                self.board[self.action_node] = 1
                self.action_node = self.start[self.pairs_idx]
                self.board[self.action_node] = self.head_value
                # state = np.array(self.action_node+self.finish[self.pairs_idx])
                state = self.board_embedding()
            else:
                self.board[self.action_node] = self.board[self.action_node]*10 + self.head_value

            # self.board[self.action_node] = self.board[self.action_node]*10 + self.head_value
        else:
            self.action_node = action_node_pre
            self.board[self.action_node] = self.head_value+10

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

        if self.max_pair == self.pairs_idx and self.action_node == self.finish[self.max_pair]:
            return -self.path_length

        if self.board[self.action_node] > self.head_value:
            left_dist = 5*distance.cityblock(self.action_node, self.finish[self.pairs_idx])
            for i in range(self.pairs_idx+1, self.max_pair):
                left_dist += 5*distance.cityblock(self.start[i], self.finish[i])

            return -left_dist-self.path_length

        return 0

    def board_embedding(self):

        dist_to_target = [i-j for i, j in zip(self.action_node, self.finish[self.pairs_idx])]
        # state = np.array(list(self.action_node)+list(self.finish[self.pairs_idx]))
        state = np.array(list(self.action_node)+dist_to_target)
        # state = np.concatenate(( state, np.array(nets_vector.tolist()+obs_vector.tolist()) ), axis=0)

        return state

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

    ######### MCTS #########
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
                # if newState.blocking_nets_Lee():
                #     newState.board[newState.action_node] = newState.head_value+10
                # else:
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
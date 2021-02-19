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
    def __init__(self, network_type='dense'):
        super(CREnv, self).__init__()

        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        self.network_type = network_type

        if self.network_type == 'dense':
            # Vanilla NN version
            self.state_shape = (5,)
        elif self.network_type == 'conv':
            # CNN version
            self.state_shape = (30, 30)
        else:
            assert NotImplementedError()

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

        # get pitch info (coordinates)
        # x_corners, y_corners = self.get_chips_corners()
        # self.p_x = [0]+x_corners+[self.board.shape[0]-1]
        # self.p_y = [0]+y_corners+[self.board.shape[1]-1]

        self.dye_value = 20

        self.head_value = 20

        self.ill_action = []
        
        self.path_length = 0

        self.max_pair = int(np.amax(self.board))
        # self.max_pair = 4
        self.connection = False

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

        net_start_pith_x = self.find_pin_pitch(self.start[self.pairs_idx][0], self.p_x)
        net_start_pith_y = self.find_pin_pitch(self.start[self.pairs_idx][1], self.p_y)
        
        self.op_board = self.dye_pitch((net_start_pith_x, net_start_pith_y))
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = self.head_value
        # print(self.start, self.finish)

        state = self.board_embedding()
        return state

    # def get_chips_corners(self):

    #     from functools import reduce
    #     x_corners = set()
    #     y_corners = set()
    #     for i in range(1, self.board.shape[0]-1):
    #         for j in range(1, self.board.shape[1]-1):
    #             if self.board[i,j] == 1:
    #                 # get the count of obs around an obs
    #                 obs_neigh_count = sum([int(self.board[i+d[0], j+d[1]]==1) for d in self.directions])
    #                 if obs_neigh_count==2: 
    #                     x_corners.add(i)
    #                     y_corners.add(j)

    #     return sorted(list(x_corners)), sorted(list(y_corners))

    def find_pin_pitch(self, pin_cood, cood_list):
        # This function can not guarantee to find idx equals to pin cood, just guarantee to find the right range
        # In this way, every pithc should be expressed by its lower node (smalled coods)
        low = 0
        high = len(cood_list) - 1
        mid = 0 
        while low < high-1:
            mid = (high + low) // 2
            # If x is greater, ignore left half
            if cood_list[mid] < pin_cood:
                low = mid     
            # If x is smaller, ignore right half
            elif cood_list[mid] > pin_cood:
                high = mid
            # means x is present at mid
            else:
                return mid 
        # If we reach here, then the element was not present
        return low

    def dye_pitch(self, pitch_idx):

        p_x_range = (self.p_x[pitch_idx[0]], self.p_x[pitch_idx[0]+1])
        p_y_range = (self.p_y[pitch_idx[1]], self.p_y[pitch_idx[1]+1])
        board = copy(self.board)
        board[p_x_range[0]:p_x_range[1],p_y_range[0]:p_y_range[1]] = self.dye_value
        return board

    def step(self, action):

        pass

    def isTerminal(self):

        pass

    def getReward(self):

        if self.connection:
            return 20
        if self.action_node==self.finish[self.max_pair] and self.pairs_idx==self.max_pair:
            return 20
        if len(self.getPossibleActions())==0:
            left_dist = 5*distance.cityblock(self.action_node, self.finish[self.pairs_idx])
            return -left_dist/10
        elif self.board[self.action_node]>self.head_value:
            return -0.5

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

        return possible_actions

    def board_embedding(self):


        if self.network_type == 'dense':
            dist_to_target = [i-j for i, j in zip(self.action_node, self.finish[self.pairs_idx])]
            if self.board[self.action_node]>self.head_value:
                sign = (self.board[self.action_node]-self.head_value)/5
            else:
                sign = 0
            # state = np.array(list(self.action_node)+list(self.finish[self.pairs_idx]))
            state = np.array(list(self.action_node)+dist_to_target+[sign])
            # state = np.concatenate(( state, np.array(nets_vector.tolist()+obs_vector.tolist()) ), axis=0)
        elif self.network_type == 'conv':
            # state = np.dstack((self.board,self.path_board))
            state = self.board
        else:
            assert NotImplementedError()

        return state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        print("------")
        print(self.action_node)

    def close(self):
        pass




env = CREnv()
env.reset()

# a = [1,4,6,9,12,15,19,21]
# for i in range(8):
#     print(env.find_pin_pitch(a[i], a))
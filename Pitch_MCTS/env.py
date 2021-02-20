# In this environment, the filled numbers for route are only 1s.
# The filled number for terminal nodes are calculated as i+1, where i=1,2,3,...
from __future__ import division

from copy import deepcopy
from copy import copy

import numpy as np

class CREnv():
    def __init__(self):

        self.board = np.genfromtxt("./board0.csv", delimiter=',')
        self.pairs_idx = 3
        self.action_space = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1,-1), (0,0)]
        self.max_pair = int(np.amax(self.board))

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        x_corners = set()
        y_corners = set()
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if abs(self.board[i,j])>=2:
                    if self.board[i,j]<0:
                        self.finish[-self.board[i,j]] = (i,j)
                        self.board[i,j] = abs(self.board[i,j])
                    else:
                        self.start[self.board[i,j]] = (i,j)

        x_corners, y_corners = self.get_pitch_node()
        self.p_x = [0]+x_corners+[self.board.shape[0]-1]
        self.p_y = [0]+y_corners+[self.board.shape[1]-1]
        print(self.p_x)
        print(self.p_y)

        self.path_length = 0
        self.dye_value = 20

        # initialize the action node
        self.action_node = copy(self.start[self.pairs_idx])

        self.board[self.action_node] = 1

    def get_pitch_node(self):

        from functools import reduce
        x_corners = set()
        y_corners = set()
        for i in range(self.pairs_idx+1, self.max_pair):
            x_corners.add(self.start[i][0])
            x_corners.add(self.finish[i][0])
            y_corners.add(self.start[i][1])
            y_corners.add(self.finish[i][1])

        return sorted(list(x_corners)), sorted(list(y_corners))

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

    def getPossibleActions(self):
        possibleActions = []
        if np.amax(self.board) < self.dye_value:
            # net_start_pith_x and net_start_pith_y are indices of self.p_x and self.p_y
            net_start_pith_x = self.find_pin_pitch(self.start[self.pairs_idx][0], self.p_x)
            net_start_pith_y = self.find_pin_pitch(self.start[self.pairs_idx][1], self.p_y)

            if self.p_x[net_start_pith_x]==self.start[self.pairs_idx][0] and self.p_y[net_start_pith_y]==self.start[self.pairs_idx][1]:
                for a in [0,3,4,5]:
                    region, start_node = self.extract_pitch_node(self.action_space[a], np.array([net_start_pith_x, net_start_pith_y]), self.start[self.pairs_idx])
                    if self.check_reaching(region, start_node):
                        possibleActions.append(self.action_space[a])
            elif self.p_x[net_start_pith_x]==self.start[self.pairs_idx][0] and self.p_y[net_start_pith_y]!=self.start[self.pairs_idx][1]:
                for a in [0,5]:
                    region, start_node = self.extract_pitch_node(self.action_space[a], np.array([net_start_pith_x, net_start_pith_y]), self.start[self.pairs_idx])
                    if self.check_reaching(region, start_node):
                        possibleActions.append(self.action_space[a])
            elif self.p_x[net_start_pith_x]!=self.start[self.pairs_idx][0] and self.p_y[net_start_pith_y]==self.start[self.pairs_idx][1]:
                for a in [3,5]:
                    region, start_node = self.extract_pitch_node(self.action_space[a], np.array([net_start_pith_x, net_start_pith_y]), self.start[self.pairs_idx])
                    if self.check_reaching(region, start_node):
                        possibleActions.append(self.action_space[a])
            else:
                possibleActions.append(self.action_space[5])

        # self.op_board = self.dye_pitch((net_start_pith_x, net_start_pith_y))

        return possibleActions

    def extract_pitch_node(self, action, node_idx, start_node):

        node_x = self.p_x[node_idx[0]]
        node_y = self.p_y[node_idx[1]]

        pnode_x_l = self.p_x[node_idx[0]+action[0]]
        pnode_y_l = self.p_y[node_idx[1]+action[1]]

        pnode_x_h = self.p_x[node_idx[0]+action[0]+1]
        pnode_y_h = self.p_y[node_idx[1]+action[1]+1]

        pitch = copy(self.board[pnode_x_l:pnode_x_h+1, pnode_y_l:pnode_y_h+1])
        new_node_x = start_node[0] - pnode_x_l
        new_node_y = start_node[1] - pnode_y_l

        return pitch, np.array([new_node_x, new_node_y])


    def check_reaching(self, region, node):
        # node is a numpy array with 2 elements
        if 0<node[0]<region.shape[0]-1 and 0<node[1]<region.shape[1]-1:
            return True

        new_region = copy(region)
        new_region[tuple(node)] = 1
        # get candidate new node
        directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        for d in directions:
            new_node = node + d
            if 0<=new_node[0]<region.shape[0] and 0<=new_node[1]<region.shape[1] \
                and (region[tuple(new_node)]==0 or region[tuple(new_node)]==self.pairs_idx):
                if self.check_reaching(new_region, new_node):
                    return True
        return False

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


env = CREnv()
print(env.getPossibleActions())
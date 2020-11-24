import numpy as np

from mcts_policy import policies
from mcts_env import circuitBoard
from mcts import mcts
import sys
import copy

import math

import os

class RunMCTS(object):
    """docstring for ClassName"""
    def __init__(self, para_file):

        self.paras = {}
        self.direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

        self.readParas(para_file)

        self.policy = policies('tf1_save')
    
    def readParas(self, file_path):
        with open(file_path, 'r') as f:
            paras_list = f.read().split('\n')

        for line in paras_list:
            if line is not '':
                line_split = line.split()
                self.paras[line_split[0]] = line_split[1]

        self.board_path = self.paras['board_path']
        self.rollout_times = int(self.paras['rollout_times'])
        self.mcts_reward = self.paras['mcts_reward']
        self.saved_board = self.paras['saved_boards_folder']
        self.mcts_node_select = self.paras['node_select']
        self.DNN_model_path = self.paras['DNN_model_path']

    def run(self):

        board = np.genfromtxt(self.board_path, delimiter=',')

        if not os.path.exists(self.saved_board):
            os.mkdir(self.saved_board)
        
        initialState = circuitBoard(board)

        MCTS_tem = mcts(iterationLimit=self.rollout_times, rolloutPolicy=self.policy.rollout,
                rewardType=self.mcts_reward, nodeSelect=self.mcts_node_select, 
                explorationConstant=0.5/math.sqrt(2))

       	routed_paths = MCTS_tem.search(initialState=initialState)
       	# print(routed_paths)

        self.write_to_file(routed_paths, initialState.finish.values())

    def write_to_file(self, route_paths, target_pins):

        save_path = os.path.join(self.saved_board, self.board_path)
        paths_tmp = []

        for vertex in route_paths:
            paths_tmp.append(vertex)
            if vertex in target_pins:
                paths_tmp.append([-1, -1])

        np.savetxt(save_path, np.array(paths_tmp), delimiter=',')

if __name__ == '__main__':

    run = RunMCTS(sys.argv[1])
    print(run.paras)
    run.run()
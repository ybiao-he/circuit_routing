import numpy as np

from policy import policies
from env import circuitBoard
from mcts import mcts
import sys
import copy

import math

import multiprocessing

class RunMCTS(object):
    """docstring for ClassName"""
    def __init__(self, para_file):

        self.paras = {}
        self.direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

        self.readParas(para_file)

        self.policy = policies(load_policy=True, dnn_model_path=self.DNN_model_path)
    
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
        self.mulproc = self.paras['multiprocess']
        self.saved_actions_file = self.paras['saved_actions']
        self.mcts_node_select = self.paras['node_select']
        self.DNN_model_path = self.paras['DNN_model_path']

    def multiprocess(self, state):

        MCTS_tem = mcts(iterationLimit=self.rollout_times, rolloutPolicy=self.policy.randomRoute, 
                rewardType=self.mcts_reward, nodeSelect=self.mcts_node_select, 
                explorationConstant=0.5/math.sqrt(2))
        return MCTS_tem.search(initialState=state)

    def getCompare(self, tup):
        return tup[1]

    def makeDecsion(self, states, action_rewards):

        best = max(action_rewards, key=self.getCompare)
        action = best[0]
        for i in range(len(states)):
            states[i] = states[i].takeAction(action)

        return states, action

    def run(self):

        actions_save = []
        board = np.genfromtxt(self.board_path, delimiter=',')
        idx = 0
        if self.mulproc == 'true':

            states = []
            for i in range(multiprocessing.cpu_count()):
                initialState = circuitBoard(board)
                states.append(initialState)

            pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

            while not states[0].isTerminal():

                action_rewards = pool.map(self.multiprocess, states)
                states, action = self.makeDecsion(states, action_rewards)
                actions_save.append(self.direction_map[action])
                    
                print(idx)
                idx += 1

        else:
            initialState = circuitBoard(board)
            MCTS_tem = mcts(iterationLimit=self.rollout_times, rolloutPolicy=self.policy.randomRoute,
                    rewardType=self.mcts_reward, nodeSelect=self.mcts_node_select, 
                    explorationConstant=0.5/math.sqrt(2))
            while not initialState.isTerminal():
                action, _ = MCTS_tem.search(initialState=initialState)
                actions_save.append(self.direction_map[action])
                initialState = initialState.takeAction(action)

                print(idx)
                idx += 1

        np.savetxt(self.saved_actions_file, np.array(actions_save), delimiter=',')

if __name__ == '__main__':

    run = RunMCTS(sys.argv[1])
    print(run.paras)
    run.run()




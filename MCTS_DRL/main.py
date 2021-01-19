import numpy as np

from Policy import policy
from mcts_env import circuitBoard
from CREnv import CREnv
from mcts import mcts
import sys
import copy

import core as core
import gym

import math

import os

class RunMCTS(object):
    """docstring for ClassName"""
    def __init__(self, para_file):

        self.paras = {}
        self.direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

        self.readParas(para_file)
    
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

    def RL_train(self, env, board, rl_policy):

        board_backup = copy.copy(board)

        epochs = 50
        local_steps_per_epoch = 1000

        o, ep_ret, ep_len = env.reset(board), 0, 0

        buf = core.Buffer()
  
        saved_ep_ret = []
        for epoch in range(epochs):
            ep_ret_tem = []
            for t in range(local_steps_per_epoch):
                a = rl_policy.predict_act(o)
                v_t = rl_policy.predict_value(o)
                logp_t = rl_policy.get_prob_act(o, a[0])

                action_logps = rl_policy.predict_probs(o)[0]
                print(np.exp(action_logps))

                o2, r, d, _ = env.step(a[0])
                ep_ret += r
                ep_len += 1

                # save and log
                buf.store(o, a[0], r, v_t, logp_t, 0)

                # Update obs (critical!)
                o = o2

                terminal = d
                if terminal or (t==local_steps_per_epoch-1):
                    if not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = 0 if d else rl_policy.predict_value(o)
                    buf.finish_path(last_val)
                    # print(t)
                    ep_ret_tem.append(ep_ret)
                    # print(ep_ret_tem)
                    # logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, ep_ret, ep_len = env.reset(copy.copy(board_backup)), 0, 0

            saved_ep_ret.append(sum(ep_ret_tem) / len(ep_ret_tem))
            # Perform VPG update!
            rl_policy.update(buf)
            buf.reset()

        np.savetxt("ave_rew.csv", np.array(saved_ep_ret), delimiter=',')

        return rl_policy


    def run(self):

        routes = []

        board = np.genfromtxt(self.board_path, delimiter=',')

        if not os.path.exists(self.saved_board):
            os.mkdir(self.saved_board)

        env = CREnv()
        rl_policy = policy(env, "ppo")
        # rl_policy.tf_restore()
        
        State = circuitBoard(board)
        targets = State.finish.values()

        # for i in range(3):
        while State.max_pair > 1 and State.getPossibleActions()[1]:
            # RL training
            rl_policy.reset()
            rl_policy = self.RL_train(env, board, rl_policy)
            # MCTS search
            MCTS_tem = mcts(iterationLimit=self.rollout_times, rolloutPolicy=rl_policy.rollout,
                        rewardType=self.mcts_reward, nodeSelect=self.mcts_node_select, 
                        explorationConstant=0.5/math.sqrt(2))
            path = MCTS_tem.search(initialState=State) 
            # update board with path or one action
       	    board, actions = State.place_path(path)
            # update mcts state with new board
            State.reset(board)
            print(actions, path)

            routes += actions

        self.write_to_file(routes, targets)

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
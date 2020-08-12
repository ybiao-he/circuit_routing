from __future__ import division

from copy import deepcopy
from scipy.spatial import distance

import numpy as np
import gym
from gym import spaces


class CREnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an env for circuit routing. 
    """
    def __init__(self, board=np.genfromtxt("./test_board.csv", delimiter=',')):
        super(CREnv, self).__init__()

        self.board_backup = np.copy(board)

        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # self.max_value = np.amax(board) + 1

        self.state_shape = (40, 40, 2)

        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=1600, shape=self.state_shape, dtype=np.float32)

        self.reset()

    def reset(self):

        self.board = np.copy(self.board_backup)

        self.max_value = np.amax(self.board)+1
        self.pairs_idx = 2

        self.path_length = 0
        self.max_pair_idx = 2

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] != 1 and self.board[i,j] != 0:
                    if self.board[i,j]<0:
                        self.finish[-self.board[i,j]] = (i,j)
                    else:
                        self.start[self.board[i,j]] = (i,j)
                    self.max_pair_idx = max(self.max_pair_idx, abs(self.board[i,j]))
                self.board[i,j] = abs(self.board[i,j])
        # initialize the action node
        print(self.start)
        self.action_node = self.start[self.pairs_idx]

        self.board[self.action_node] = self.max_value
        # self.board[self.finish[self.pairs_idx]] = self.max_value
         
        # state embedding 1
        # board = self.board.reshape(40,40,1).astype(np.float32)
        # net_idx_mat = np.ones(board.shape)*self.pairs_idx
        # one_sample = np.concatenate((board, net_idx_mat), axis=2)
        # state = np.array(one_sample/(self.max_value-1.0))

        # state embedding 2
        action_mask = self.getActionProbMask(self.getPossibleActions())

        board = self.board.reshape(40,40,1).astype(np.float32)
        net_idx_mat = np.ones(board.shape)*self.pairs_idx
        one_sample = np.float32( np.concatenate((board, net_idx_mat), axis=2) )
        for i in range(len(action_mask)):
            one_sample[0][i][1] = action_mask[i]
        state = np.array(one_sample)


        # print(state)

        return state


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

    def getActionProbMask(self, possible_actions):

        act_prob_mask = np.zeros(len(self.directions), dtype=np.float32)
        for act in possible_actions:
            idx_action = self.directions.index(act)
            act_prob_mask[idx_action] = 1.0

        return np.array(act_prob_mask)


    def step(self, action):


        action_tmp = self.directions[action]

        # self.board[self.action_node] = 1
        
        self.action_node = (self.action_node[0]+action_tmp[0], self.action_node[1]+action_tmp[1])

        print(self.action_node)

        if self.board[self.action_node] == 0:
            self.board[self.action_node] = self.max_value
        elif self.action_node == self.finish[self.pairs_idx]:
            # self.board[self.action_node] = 1
            self.pairs_idx += 1
            self.action_node = self.start.get(self.pairs_idx)
            if self.action_node is not None:
                self.board[self.action_node] = self.max_value

        # self.board[self.finish.get(self.pairs_idx)] = self.max_value
        # print(self.finish.get(self.pairs_idx))

        # # compute state (embedding 1)
        # board = self.board.reshape(40,40,1).astype(np.float32)
        # net_idx_mat = np.ones(board.shape)*self.pairs_idx
        # one_sample = np.concatenate((board, net_idx_mat), axis=2)
        # state = np.array(one_sample/(self.max_value-1.0))

        # compute state (embedding 2)
        action_mask = self.getActionProbMask(self.getPossibleActions())

        board = self.board.reshape(40,40,1).astype(np.float32)
        net_idx_mat = np.ones(board.shape)*self.pairs_idx
        one_sample = np.float32( np.concatenate((board, net_idx_mat), axis=2) )
        for i in range(len(action_mask)):
            one_sample[0][i][1] = action_mask[i]
        state = np.array(one_sample)

        print(np.sum(state))
        self.path_length += 1

        self.max_value += 1

        reward = self.getReward()

        done = self.isTerminal()

        info = {}

        return state, reward, done, info


    def isTerminal(self):

        if self.action_node is None or len(self.getPossibleActions()) == 0:
            return True

        return False

    def getReward(self):

        if len(self.getPossibleActions()) == 0:
            left_dist = distance.cityblock(self.action_node, self.finish[self.pairs_idx])
            for i in range(self.pairs_idx+1, int(self.max_pair_idx+1)):
                left_dist += distance.cityblock(self.start[i], self.finish[i])
            return -left_dist*2.0
            # return -self.board.shape[0]*self.board.shape[1]/10
        # elif self.action_node is None:
        #     return -self.path_length/10
        return -1.0

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        print("------")
        print(self.action_node)

    def close(self):
        pass

from stable_baselines.common.env_checker import check_env

env = CREnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)


# Testing the environment
env = CREnv()

obs = env.reset()
env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

import random
# Hardcoded best agent: always go left!
n_steps = 20
for step in range(n_steps):
    action = random.randint(0, 3)
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(action)
    print('obs=', obs.shape, 'reward=', reward, 'done=', done)
    env.render()
    if done:
        print("Goal reached!", "reward=", reward)
        break


# Try it with Stable-Baselines
from stable_baselines import DQN, PPO2, A2C, ACKTR, ACER, DDPG, TRPO, PPO1
from stable_baselines.common.cmd_util import make_vec_env
from CRPolicy import CRPolicy
import os
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
# Instantiate the env
env = CREnv()
# wrap it
# env = make_vec_env(lambda: env, n_envs=1)

env = Monitor(env, log_dir)

env = make_vec_env(lambda: env, n_envs=1)

# env_test = ACER(CRPolicy, env, verbose=1).get_env()
# print(env_test.board)
# Train the agent
model = PPO2(CRPolicy, env, verbose=1)
# model = PPO2(policy=CRPolicy, env=env, n_steps=1600, nminibatches=4,
#              lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
#              learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
# model = ACER("CnnPolicy", env, verbose=1).learn(1000)

# Train the agent
time_steps = 100000
model.learn(total_timesteps=int(time_steps))

save_path = os.path.join(log_dir, 'best_model')
model.save(save_path)

import matplotlib.pyplot as plt

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO2 for routing")
plt.show()


# Test the trained agent
obs = env.reset()
n_steps = 200
print("let's start to predict")
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs.shape, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
from __future__ import division

import time
import math
import random
import numpy as np
import os

from copy import deepcopy

import core_tf as core

from functools import reduce

def randomPolicy(state):
    while not state.isTerminal():
        try:
            action = random.choice(state.getPossibleActions())
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()

class treeNode():
    def __init__(self, state, parent, rewardType):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        if rewardType=="ave":
            self.totalReward = 0
        else:
            self.totalReward = float("-inf")
        self.children = {}

class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 Policy=randomPolicy, rewardType="ave", nodeSelect="best"):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant

        self.policy = Policy
        self.rollout = self.policy.randomRoute

        # self.RL_actor = self.policy.policy_model
        # self.RL_critic = self.policy.value_model

        self.rewardType = rewardType
        self.nodeSelect = nodeSelect

        self.route_paths_saved = []

        self.buf = core.Buffer()

    def search(self, initialState):
        self.root = treeNode(initialState, None, self.rewardType)

        from CREnv import CREnv

        self.cr_env = CREnv(board=self.root.state.board_backup)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRoundByIters()
                self.policy.ppo_update(self.buf)
                self.buf.reset()
            # res = self.buf.get()
            # [print(r.shape) for r in res]

        # bestChild = self.getBestChildBasedonReward(self.root)
        # return self.getAction(self.root, bestChild), bestChild.totalReward
        return self.route_paths_saved

    def executeRound(self):
        node = self.selectNode(self.root)
        reward, _ = self.rollout(node.state)
        self.backpropogate(node, reward)

    def executeRoundByIters(self):
        # selection and expansion
        node, select_by_node = self.selectNode(self.root)
        # rollout
        reward, route_paths = self.rollout(node.state)
        # update the best paths
        reward_total = 1/(len(select_by_node)+1/reward)
        if len(route_paths)==0:
            select_by_node.append([-1, -1])
            route_paths = select_by_node
        else:
            route_paths = reduce(lambda x,y: x+y, route_paths)
            route_paths = select_by_node + route_paths

        self.store_to_buf(route_paths)

        if reward_total>self.root.totalReward:
            self.route_paths_saved = route_paths
        # backpropagation
        self.backpropogate(node, reward)

    def selectNode(self, node):
        route_by_select = []
        while not node.isTerminal:
            if node.isFullyExpanded:
                if self.nodeSelect == "best":
                    node = self.getBestChild(node, self.explorationConstant)
                    # node = self.getBestChildBasedonReward(node)
                else:
                    node = random.choice( list(node.children.values()) )
                route_by_select.append(node.state.action_node)
            else:
                ret_node = self.expand(node)
                route_by_select.append(ret_node.state.action_node)
                return ret_node, route_by_select
        return node, route_by_select

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                newNode = treeNode(node.state.takeAction(action), node, self.rewardType)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode
        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        back_node = 0
        while node is not None:
            # the backpropagation for the node on the tree is revised,
            # the path length from root to the selected node is counted
            reward_node = 1/(back_node+1/reward)
            node.numVisits += 1

            if self.rewardType == "ave":
                node.totalReward += reward
            else:
                node.totalReward = max(reward_node, node.totalReward)
            node = node.parent
            back_node += 1

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():

            if self.rewardType == "ave":
                nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)
            else:
                nodeValue = child.totalReward + explorationValue * math.sqrt(
                    2 * math.log(node.numVisits) / child.numVisits)

            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    # def getBestChildBasedonReward(self, node):

    #     bestValue = float("-inf")
    #     bestNodes = []
    #     for child in node.children.values():

    #         nodeValue = child.totalReward

    #         if nodeValue > bestValue:
    #             bestValue = nodeValue
    #             bestNodes = [child]
    #         elif nodeValue == bestValue:
    #             bestNodes.append(child)
    #     return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action

    def store_to_buf(self, routed_path):

        # from CREnv import CREnv

        # cr_env = CREnv(board=self.root.state.board_backup)
        from view import view

        render = view()

        pre_state = self.cr_env.reset()
        
        num_iters = 0

        for vertex in routed_path:
            action = tuple(map(lambda i, j: i - j, vertex, self.cr_env.action_node))
            action_idx = self.cr_env.directions.index(action)

            render.display(self.cr_env.board)

            current_state, reward, done, info = self.cr_env.step(action_idx)
            p = self.policy.get_prob_act(pre_state, action_idx)
            value = self.policy.predict_value(pre_state)
            print(np.sum(pre_state))
            print(self.policy.predict_probs(pre_state))
            # print(logp, value)

            self.buf.store(pre_state, action_idx, reward, value, p)

            pre_state = current_state

            num_iters += 1
        self.buf.finish_path()
        # print(num_iters)
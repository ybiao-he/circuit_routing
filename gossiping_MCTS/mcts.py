from __future__ import division

import time
import math
import random
import numpy as np
import os

from copy import deepcopy

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
        self.rollout = self.policy.rollout

        self.rewardType = rewardType
        self.nodeSelect = nodeSelect

        self.route_paths_saved = []

        if not os.path.exists('improved_paths'):
            os.makedirs('improved_paths')
        self.save_board_folder = "./improved_paths"
        self.save_iter_idices = []

    def search(self, initialState):
        self.root = treeNode(initialState, None, self.rewardType)

        self.save_board_idx = 0

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRoundByIters(i)
            save_idices_path = os.path.join(self.save_board_folder, "saved_iters_idices.csv")
            np.savetxt(save_idices_path, np.array(self.save_iter_idices), delimiter=',')

        return self.route_paths_saved

    def executeRound(self):
        node = self.selectNode(self.root)
        reward, _ = self.rollout(node.state)
        self.backpropogate(node, reward)

    def executeRoundByIters(self, i):
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

        if reward_total>self.root.totalReward:
            self.route_paths_saved = route_paths
            self.write_to_file(route_paths)
            self.save_iter_idices.append(i)
            self.save_board_idx += 1
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


    def write_to_file(self, route_paths):

        target_pins = self.root.state.finish.values()

        save_path = os.path.join(self.save_board_folder, "improved_board"+str(self.save_board_idx)+".csv")
        paths_tmp = []

        for vertex in route_paths:
            paths_tmp.append(vertex)
            if vertex in target_pins:
                paths_tmp.append([-1, -1])

        np.savetxt(save_path, np.array(paths_tmp), delimiter=',')
from __future__ import division

import time
import math
import random
import numpy as np
import os

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
                 rolloutPolicy=randomPolicy, rewardType="ave", nodeSelect="best"):
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
        self.rollout = rolloutPolicy

        self.rewardType = rewardType
        self.nodeSelect = nodeSelect

        self.route_paths_saved = []

    def search(self, initialState):
        self.root = treeNode(initialState, None, self.rewardType)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRoundByIters()

        # bestChild = self.getBestChildBasedonReward(self.root)
        # return self.getAction(self.root, bestChild), bestChild.totalReward
        return self.route_paths_saved

    def executeRound(self):
        node = self.selectNode(self.root)
        reward, _ = self.rollout(node.state)
        self.backpropogate(node, reward)

    def executeRoundByIters(self):
        node, select_by_node = self.selectNode(self.root)
        reward, route_paths = self.rollout(node.state)

        if len(route_paths)==0:
            # select_by_node.append([-1, -1])
            route_paths = select_by_node
        else:
            # route_paths = reduce(lambda x,y: x+y, route_paths)
            route_paths = select_by_node + route_paths
        # print(route_paths)
        if reward>self.root.totalReward:
        	self.route_paths_saved = route_paths
            # self.write_to_file(route_paths)
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

                route_node = self.get_route_node(node)   
                route_by_select.append(route_node)
            else:
                ret_node = self.expand(node)
                route_node = self.get_route_node(ret_node) 
                route_by_select.append(route_node)
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
        while node is not None:
            node.numVisits += 1

            if self.rewardType == "ave":
                node.totalReward += reward
            else:
                node.totalReward = max(reward, node.totalReward)
            node = node.parent

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

    def get_route_node(self, node):
        pre_pair_idx = node.state.pairs_idx-1

        if pre_pair_idx in node.state.start and node.state.action_node==node.state.start[node.state.pairs_idx]:
            return node.state.finish[pre_pair_idx]
        else:
            return node.state.action_node

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action
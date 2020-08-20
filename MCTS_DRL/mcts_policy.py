import numpy as np

import random

from keras.models import load_model

class policies(object):

    def __init__(self, load_policy, dnn_model_path):
    
        if load_policy:
            self.policy_model = load_model(dnn_model_path)
        else:
            from keras.models import Sequential
            from keras.layers import Dense, Conv2D, Flatten
            self.policy_model = Sequential()
            self.policy_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(40,40,1)))
            self.policy_model.add(Conv2D(32, kernel_size=3, activation='relu'))
            self.policy_model.add(Flatten())
            self.policy_model.add(Dense(4, activation='softmax'))

        self.direction_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def network(self, state):
        """
        This function is for DNN-based policy without DFS. It is not used in our currecnt method
        """
        route_paths = []
        while not state.isTerminal():
            try:
                feature = np.array([state.board.ravel()])
                predict = self.policy_model.predict(feature)
                sort_action = np.flip( np.argsort(predict) )[0]
                action_set = state.getPossibleActions()

                for idx in sort_action:
                    if self.direction_list[idx] in action_set:
                        action = self.direction_list[idx]
                        break

            except IndexError:
                raise Exception("Non-terminal state has no possible actions: ")
            route_paths.append(action)
            state = state.takeAction(action)

        return state.getReward(), [route_paths]

    def randomRoute(self, state):

        route_paths = []

        while not state.isTerminal():

            board = np.copy(state.board)
            pin_idx = state.pairs_idx

            path = self.randomDFS(board, state.action_node, state.finish[state.pairs_idx], pin_idx)

            if len(path)==1:
                # print("failed to find a path to the target node")
                return 1/(board.shape[0]*board.shape[1]), route_paths
            else:
                route_paths.append(path)
                path.pop(0)
                for p in path:
                    action = tuple(np.subtract(p, state.action_node))
                    state = state.takeAction(action)

        return state.getReward(), route_paths


    def randomDFS(self, board, s, t, pin_idx):

        path_queue = [s]
        node = s
        pre_node = s

        # pin_max = int(np.amax(board))
        # path_num = pin_max + 1

        while t not in path_queue:

            actions = self.getPossibleActions(board, node, t)
            
            if len(actions) != 0:
                # randomly choose an action, revise it by possibilities
                # action = random.choice(actions)
                bh = board.shape[0]
                bw = board.shape[1]
                board_tem = np.reshape(board, (1, bh, bw, 1))
                action_dist = self.policy_model.predict(board_tem)[0]
                # print(action_dist)
                action = self.get_action(actions, action_dist)

                node = tuple(map(sum, zip(action, node)))
                board[node] = pin_idx
                board[pre_node] = 1
                pre_node = node
                # path_num += 1
                path_queue.append(node)
            elif len(path_queue) > 1:
                pre_node = path_queue.pop()
                node = path_queue[-1]
                
                board[pre_node] = 1
                board[node] = pin_idx
                # path_num -= 1
            else:
                break
        return path_queue

    def getPossibleActions(self, board, node, t):

        possible_actions = []
        for d in self.direction_list:
            x = node[0] + d[0]
            y = node[1] + d[1]
            if 0 <= x < board.shape[0] and 0 <= y < board.shape[1]:
                if (x,y) == t:
                    possible_actions.append((d[0], d[1]))
                    break
                elif board[(x,y)] == 0:
                    possible_actions.append((d[0], d[1]))

        return possible_actions

    def get_action(self, possible_actions, action_dist):
        act_idx_to_poss = {}
        for act in possible_actions:
            idx_action = self.direction_list.index(act)
            act_idx_to_poss[idx_action] = action_dist[idx_action]

        # normalize the distribution of actions
        # print(act_idx_to_poss)
        poss_sum = sum(act_idx_to_poss.values())
        print(act_idx_to_poss.values())
        if poss_sum == 0:
            for a in act_idx_to_poss:
                act_idx_to_poss[a] = 1/len(act_idx_to_poss.values()) 
        else:    
            for a in act_idx_to_poss:
                act_idx_to_poss[a] /= poss_sum

        # print(act_idx_to_poss)
        # get actions according to the normalized distribution
        # print(act_idx_to_poss)
        ret_action = np.random.choice(list(act_idx_to_poss.keys()), p=list(act_idx_to_poss.values()))
        return self.direction_list[ret_action]

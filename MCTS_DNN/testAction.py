import numpy as np

from view import view

import sys

class test_policy():

    def __init__(self, board):

        self.board = board

        self.pin_max = int(np.amax(self.board))
        self.path_num = self.pin_max + 1

        self.start = {}
        self.finish = {}

        self.pairs_idx = 2

        self.direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

        self.parseBoard()

        self.path_x = []
        self.path_y = []

        self.path_length = 0

    def parseBoard(self):

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] != 1 and self.board[i,j] != 0:
                    if self.board[i,j] in self.start:
                        self.finish[self.board[i,j]] = (i,j)
                    else:
                        self.start[self.board[i,j]] = (i,j)

        self.action_node = self.start[self.pairs_idx]

        self.path_x = self.action_node[0]
        self.path_y = self.action_node[1]

        self.board[self.action_node] = self.path_num

    def takeAction(self, action):
        
        self.action_node = (self.action_node[0]+action[0], self.action_node[1]+action[1])

        if self.board[self.action_node] == 0:
            self.board[self.action_node] = self.path_num
            self.path_num += 1
            self.path_x.append(self.action_node[0])
            self.path_y.append(self.action_node[1])
        elif self.action_node == self.finish[self.pairs_idx]:
            self.board[self.action_node] = 1
            self.board[(self.path_x, self.path_y)] = 1
            self.pairs_idx += 1
            self.action_node = self.start.get(self.pairs_idx)
            self.path_num = self.pin_max + 1
            if self.action_node is not None:
                self.board[self.action_node] = self.path_num
                self.path_x = [self.action_node[0]]
                self.path_y = [self.action_node[1]]

        self.path_length += 1


if __name__ == '__main__':

    direction_list = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    filename = '../GenPlacement/boards/board7.csv'
    board = np.genfromtxt(filename, delimiter=',')

    visual = view()

    initialState = test_policy(board)

    print(initialState.finish)

    actions = np.genfromtxt(sys.argv[1], delimiter=',')

    for action_idx in actions:

        action = direction_list[int(action_idx)]

        initialState.takeAction(action)

        figure = visual.display(initialState.board)
        
        print(initialState.action_node)

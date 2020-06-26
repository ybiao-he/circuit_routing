from LeeAlgm import Field
import numpy as np

import os
import sys
import random
from view import view

class MazeRouting():

    def __init__(self):

        self.board = np.zeros((30,30))

        self.barriers = []
        self.start = {}
        self.finish = {}

        self.data_idx = 0

        self.pin_max = int(np.amax(self.board))

        self.labels = []

        self.filename = ''

        self.direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}

        self.display = view()

    def parseBoard(self):
        ''' Get the initial barriers from the board including terminals,
            and also get the start and end of the routing path (terminals).
        '''

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i,j] == 1:
                    self.barriers.append((i,j))
                elif self.board[i,j] != 0:
                    if self.board[i,j] in self.start:
                        self.finish[self.board[i,j]] = (i,j)
                    else:
                        self.start[self.board[i,j]] = (i,j)
                    self.barriers.append((i,j))

    def getRoutingPath(self, pin_idx):

        path = []
        self.barriers.remove(self.start[pin_idx])
        self.barriers.remove(self.finish[pin_idx])

        field = Field(len=30, start=self.start[pin_idx], finish=self.finish[pin_idx],
                      barriers=self.barriers)
        field.emit()

        try:
            path = field.get_path()
        except:
            print("No path left between the pin pairs "+str(self.start[pin_idx])+" and "+str(self.finish[pin_idx]))
        return list(path)

    def GenRoutingData(self, path, pin_idx):
        ''' Given the routing path by maze algorithm of one pair of terminals,
            generate and save the data for RL: the number in the path increase as the path gets longer, \
            and other blocks on the board are 1s
        '''
        path_x = []
        path_y = []

        sample_one_path = []
        pre_p = self.start[pin_idx]
        for p in range(len(path)-1):
            # generate board
            self.barriers.append(path[p])

            self.board[pre_p] = 1
            self.board[path[p]] = pin_idx
            pre_p = path[p]
            path_x.append(path[p][0])
            path_y.append(path[p][1])

            # self.display.display(self.board)
            
            # generate label
            action = self.direction_map[(path[p+1][0]-path[p][0], path[p+1][1]-path[p][1])]

            # save the data to csv file
            one_sample = np.append(self.board.ravel(), action)
            sample_one_path.append(one_sample)

            self.data_idx += 1

        self.board[(path_x, path_y)] = 1

        return sample_one_path

    def run(self):

        self.parseBoard()

        data_one_board = []

        for pin_idx in range(2, self.pin_max+1):

            path = self.getRoutingPath(pin_idx)

            if len(path)==0:
                print(path)
                break

            samples_one_path = self.GenRoutingData(path, pin_idx)
            data_one_board.append(random.choice(samples_one_path))

        return data_one_board


    def reset(self, board, filename):

        self.board = board

        self.barriers = []
        self.start = {}
        self.finish = {}

        self.data_idx = 0

        self.pin_max = int(np.amax(board))

        self.labels = []

        self.filename = filename[:-4]


if __name__ == '__main__':

    foldername = sys.argv[1]

    routing = MazeRouting()

    save_data = []

    for filename in os.listdir(foldername):

        print(filename)

        Bpath = foldername + filename
        board = np.genfromtxt(Bpath, delimiter=',')

        routing.reset(board, filename)

        data_one_board = routing.run()

        save_data = save_data+data_one_board

    save_data = np.array(save_data)
    np.savetxt('training_data.csv', save_data, delimiter=',')
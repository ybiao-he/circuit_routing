import matplotlib.pyplot as plt
import numpy as np

class view():
    def __init__(self):
        self.x_axis = []
        self.y_axis = []

    def display(self, board):

        height, width = board.shape
        
        plt.grid()

        plt.xlim(0, width-1)
        plt.ylim(0, height-1)

        plt.xticks(np.arange(0, width-1, 1.0))
        plt.yticks(np.arange(0, height-1, 1.0))

        self.x_axis = []
        self.y_axis = []
        for x in range(width):
            for y in range(height):
                if board[x, y]!=0:
                    self.x_axis.append(y)
                    self.y_axis.append(x)

        plt.scatter(self.x_axis, self.y_axis, marker='s')
        plt.draw()
        plt.pause(1e-17)
        # time.sleep(0.1)
        plt.clf()
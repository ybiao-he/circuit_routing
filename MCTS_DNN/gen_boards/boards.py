# This file ims at generate the PCB board with the placement for all nets
# The parameters to be considered includes Grid size (30*30), #Obstacle, #Nets 
import numpy as np
import random

import sys
import os

def add_obstacles(board, num_obstacles):

    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 0 and random.randint(1,901)<=num_obstacles:
                board[i,j] = 1

    return board
def add_pins(board, pattern):
    height, width = board.shape
    max_num = height * width
    blocks = pattern['blocks']
    pairs = pattern['pairs']
    for b in blocks:
        board[b[0]:b[1], b[2]:b[3]] = 1
    pair_code = 2
    for p in pairs:
        board[p[0]] = pair_code 
        board[p[1]] = pair_code
        pair_code += 1
    return board

def gen_pattern():
    patterns_set = []
    blocks1 = [[10, 16, 4, 8], [6, 12, 15, 22]]
    pairs1 = [[(9,4),(5,18)], [(9,7),(7,14)], [(13,8),(10,14)], [(12,16),(16,7)], [(16,4),(12,19)]]
    pattern1 = {'blocks': blocks1, 'pairs': pairs1}
    patterns_set.append(pattern1)
    blocks2 = [[4, 6, 5, 10], [10, 14, 18, 25], [20, 22, 9, 12]]
    pairs2 = [[(3,6),(9,21)], [(5,4),(21,8)], [(3,9),(9,19)], [(6,8),(11,17)], [(20,12),(14,24)]]
    pattern2 = {'blocks': blocks2, 'pairs': pairs2}
    patterns_set.append(pattern2)

    return patterns_set

if __name__== "__main__":
    height = 30
    width = 30
    num_obstacles = 90
    num_nets = 5 # manually added to the csv file

    patterns_set = gen_pattern()

    board_num = int(sys.argv[1])

    board_idx = 0

    if not os.path.exists('./boards'):
    	os.mkdir('./boards')

    for i in range(board_num):

        for pattern in patterns_set:

            board = np.zeros((height, width))

            board = add_pins(board, pattern)
            board = add_obstacles(board, num_obstacles)
            
            np.savetxt('./boards/board'+str(board_idx)+'.csv', board, delimiter=',')

            board_idx += 1
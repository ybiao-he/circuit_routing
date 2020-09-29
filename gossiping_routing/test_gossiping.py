# test gossiping routing
import numpy as np
import random
from copy import copy
import sys
sys.setrecursionlimit(10**6) 

def cos_value(B, A, D):
    '''
    B, A, D should be numpy arrays
    return cos(BAD)
    '''
    AB_vec = B-A
    AD_vec = D-A
    dot_product = np.dot(AB_vec, AD_vec)
    AB = np.linalg.norm(AB_vec)
    AD = np.linalg.norm(AD_vec)
    return dot_product/(AB*AD)

def Score1(B, A, D):
    return cos_value(B,A,D)

def Score2(B,A,D):
    return 1 + cos_value(B,A,D)

def Score3(B,A,D):
    e = 0.9
    return (1-e)/(1-e*cos_value(B,A,D))

def find_node(board, number):
    '''
    return a numpy array with 2 elements
    '''
    return np.asarray(np.where(board == number)).reshape(-1)

def rollout(board, route_boards):
    
    net_number = 2
    current_node = find_node(board, net_number)
    neighbors = find_neighbors(board, current_node)
    
    destination = find_node(board, -net_number)
    if tuple(destination) in neighbors:
        route_boards.append(board)
        print("find one!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return 1
    elif len(neighbors)==0:
        return 0
    probs = calculate_probs(neighbors, current_node, destination)
    # print(probs)
    
    for neighbor in neighbors:
        prob = probs[neighbor]
        if random.random()<=prob:
            board_next = copy(board)
            board_next[neighbor] = net_number
            board_next[tuple(current_node)] = 1
            rollout(board_next, route_boards)

def find_neighbors(board, current_node):
    '''
    current_node is a numpy array
    return:
    neighbors a list of tuples
    '''
    neighbors = []
    directions = np.array([[0,1], [0,-1], [1,0], [-1,0]])
    for d in directions:
        candidate_neighbor = tuple(current_node + d)
        if np.amax(candidate_neighbor)<board.shape[0] and np.amin(candidate_neighbor)>=0:
            if board[candidate_neighbor]!=1:
                neighbors.append(candidate_neighbor)
    return neighbors

def calculate_probs(neighbors, current_node, destination):
    '''
    return a dict mapping the tuple-based neighbors to probs 
    '''
    scores = []
    probs = dict()
    for neighbor in neighbors:
        scores.append(Score2(np.asarray(neighbor), current_node, destination))
        
    max_score = max(scores)
    for neighbor in neighbors:
        p = Score2(np.asarray(neighbor), current_node, destination)/max_score
        probs[neighbor] = p
    return probs

board = np.loadtxt("sim_board.csv", delimiter=',')
route_boards = []
rollout(board, route_boards)

print(len(route_boards))
print(route_boards[0])
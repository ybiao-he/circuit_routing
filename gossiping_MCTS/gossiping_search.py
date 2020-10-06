# test gossiping routing
import numpy as np
import random
from copy import copy

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

def Score3(B,A,D,e):
    # e = 0.3
    return (1-e)/(1-e*cos_value(B,A,D))

def find_node(board, number):
    '''
    return a numpy array with 2 elements
    '''
    return np.asarray(np.where(board == number)).reshape(-1)

def connect(board, net_number, e):

    breadth_limit = 100
    final_paths_len = 2

    boards = np.array([copy(board)])
#     current_node = find_node(board)
    paths = [[]]
    final_paths = []
    max_iters = board.shape[0]+board.shape[1]
    for j in range(max_iters):
    # while len(final_paths)<2:
        boards_tem = []
        paths_tem = []
        for i in range(len(boards)):
            b = boards[i]
            p = paths[i]
            bs_tem, ps_tem, path = take_actions(b,p, net_number, e)
            if path is not None:
                final_paths.append(path)
            boards_tem += bs_tem
            paths_tem += ps_tem
        if len(boards_tem)>breadth_limit:
            random_index = random.sample(range(1, len(boards_tem)), breadth_limit)
            boards = np.array(boards_tem)[random_index]
            paths = [paths_tem[i] for i in random_index]
        else:
            boards = np.array(boards_tem)
            paths = paths_tem

        if len(final_paths)>=final_paths_len:
            break

    return final_paths

def take_actions(board, path, net_number, e):
    '''
    return a list of 2d numpy array (board)
    '''
    # net_number = 2
    boards_next = []
    paths_next = []
    current_node = find_node(board, net_number)
    # print(net_number)
    neighbors = find_neighbors(board, current_node, net_number)
    
    if len(neighbors)==0:
        return [], [], None
    
    destination = find_node(board, -net_number)
    probs = calculate_probs(neighbors, current_node, destination, e)
    
    if tuple(destination) in neighbors:
        return boards_next, paths_next, path
        
    for neighbor in neighbors:
        prob = probs[neighbor]
        if random.random()<=prob:
            board_next = copy(board)
            path_next = copy(path)
            board_next[neighbor] = net_number
            board_next[tuple(current_node)] = 1
            path_next.append(neighbor)
            boards_next.append(board_next)
            paths_next.append(path_next)
            
    return boards_next, paths_next, None

def find_neighbors(board, current_node, net_number):
    '''
    current_node is a numpy array
    return:
    neighbors a list of tuples
    '''
    # print("--------------")
    neighbors = []
    directions = np.array([[0,1], [0,-1], [1,0], [-1,0]])
    for d in directions:
        # print(current_node)
        candidate_neighbor = tuple(current_node + d)
        if np.amax(candidate_neighbor)<board.shape[0] and np.amin(candidate_neighbor)>=0:
            if board[candidate_neighbor]==0 or board[candidate_neighbor]==-net_number:
                neighbors.append(candidate_neighbor)
    return neighbors

def calculate_probs(neighbors, current_node, destination, e):
    '''
    return a dict mapping the tuple-based neighbors to probs 
    '''
    probs = dict()
    for neighbor in neighbors:
        probs[neighbor] = Score3(np.asarray(neighbor), current_node, destination, e)
        
    max_score = max(probs.values())
    for neighbor in neighbors:
        probs[neighbor] /= max_score
    return probs

# board = np.loadtxt("sim_board.csv", delimiter=',')
# routes = connect(board, 2, 0.9)
# [print(len(r)) for r in routes]
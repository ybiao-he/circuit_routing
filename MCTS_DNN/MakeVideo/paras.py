import os
import sys
import re
if __name__ == '__main__':

    board_path = "./test_boards/"
    dnn_model_path = "./actor.h5"
    rollout_times = int(sys.argv[1])
    multiprocess = "false"
    mcts_reward = "best"
    node_select = "best"

    os.mkdir(mcts_reward+str(rollout_times))
    os.mkdir("./video_boards/")

    for bfile in os.listdir(board_path):
        num = re.findall(r'\d+', bfile)[0]
        saved_boards_path = "./video_boards/video_board_"+num
        os.mkdir(saved_boards_path)
        with open(mcts_reward+str(rollout_times)+"/paras_"+mcts_reward+num+".txt", 'w') as f:
            f.write("board_path "+board_path+bfile+"\n")
            f.write("DNN_model_path "+dnn_model_path+"\n")
            f.write("rollout_times "+str(rollout_times)+"\n")
            f.write("saved_boards_folder "+saved_boards_path+"\n")
            f.write("mcts_reward "+mcts_reward+"\n")
            f.write("node_select "+node_select)

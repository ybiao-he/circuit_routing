import numpy as np
import os
import cv2
import shutil

def draw_board(paths_x, paths_y, board, save_name):
    
    import matplotlib.pyplot as plt
    width, height = board.shape

    # create a 8" x 8" board
    fig = plt.figure(figsize=[8,8])
    # fig.patch.set_facecolor((1,1,.8))

    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(30):
        ax.plot([x, x], [0,29], color=(0.5,0.5,0.5,1))
    for y in range(30):
        ax.plot([0, 29], [y,y], color=(0.5,0.5,0.5,1))

    # draw paths
    for p in range(len(paths_x)):

        ph = plt.subplot()
        ph.plot(paths_y[p], paths_x[p], linewidth=5, color='black')

    # draw obstacles
    x_axis = []
    y_axis = []
    nets = dict()
    for x in range(width):
        for y in range(height):
            if board[x, y]!=0:
                x_axis.append(y)
                y_axis.append(x)
                if board[x, y]!=1:
                    nets[(x,y)] = board[x, y]

    ax.scatter(x_axis, y_axis, marker='s', s=250, c='k')

    for xy in nets:
        ax.text(xy[1], xy[0], str(int(nets[xy])-1), fontsize=18, color='w',
                horizontalalignment='center', verticalalignment='center')

    # scale the axis area to fill the whole figure
    ax.set_position([0,0,1,1])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()

    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(0,29)
    ax.set_ylim(0,29)
    
    fig.savefig(save_name, bbox_inches='tight')

def parse_board(board):
    
    start = {}
    finish = {}
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i,j] != 1 and board[i,j] != 0:
                if board[i,j] in start:
                    finish[board[i,j]] = (i,j)
                else:
                    start[board[i,j]] = (i,j)
    return start

def make_videos(num_boards):

    for board_idx in range(num_boards):
        # load board
        filename = "board"+str(board_idx)+".csv"
        foldername = './test_boards'
        Bpath = os.path.join(foldername, filename)
        board = np.loadtxt(Bpath, delimiter=',')

        start = parse_board(board)
        net_indices = sorted(list(start.keys()))
        print(start)
        # load routing from csv files
        if not os.path.exists('./routing_images'):
            os.mkdir("./routing_images")
            
        for iters_idx in range(15):
            try:
                route_folder = './video_boards/video_board_'+str(board_idx)
                route_file_name = 'board_video_'+str(iters_idx)+'.csv'
                route_file_path = os.path.join(route_folder, route_file_name)
                routes = np.loadtxt(route_file_path, delimiter=',')

                # drawing paths
                path_x_tmp = [start[net_indices[0]][0]]
                path_y_tmp = [start[net_indices[0]][1]]
                paths_x = [path_x_tmp]
                paths_y = [path_y_tmp]
                p_idx = 1
                frame = 0
                for coord in routes:
                    if coord[0]==-1:
                        if p_idx in range(len(net_indices)):
                            path_x_tmp = [start[net_indices[p_idx]][0]]
                            path_y_tmp = [start[net_indices[p_idx]][1]]
                            paths_x.append(path_x_tmp)
                            paths_y.append(path_y_tmp)
                        p_idx += 1
                    else:
                        paths_x[-1].append(coord[0])
                        paths_y[-1].append(coord[1])
                        draw_board(paths_x, paths_y, board, "./routing_images/frame_"+str(iters_idx)+"_"+str(frame)+".png")
                        frame += 1
            except:
                break
        make_video_from_images(board_idx)
        shutil.rmtree("./routing_images/")

def make_video_from_images(board_idx):

    img_array = []
    iters_idx = 0

    for iters_idx in range(10):
        for frm_idx in range(900):
            frame_name = "./routing_images/frame_"+str(iters_idx)+"_"+str(frm_idx)+".png"
            try:
                print(frame_name)
                img = cv2.imread(frame_name)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
            except:
                break

    out = cv2.VideoWriter('board'+str(board_idx)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == "__main__":

    make_videos(30)
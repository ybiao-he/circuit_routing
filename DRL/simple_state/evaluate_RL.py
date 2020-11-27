# Off-policy RL training
from tf_model_load import policy
import numpy as np
import core as core
import gym
from CREnv import CREnv

def get_paths(env_test, rl_policy):

    d = False
    o, ep_ret, ep_len = env_test.reset(), 0, 0
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    paths_x = []
    paths_y = []

    pathx = [o[0]]
    pathy = [o[1]]
    total_ep_ret = ep_ret
    while not d:
        a = rl_policy.predict_act(o)
        target = (o[2], o[3])
        o, r, d, _ = env_test.step(a[0])
        total_ep_ret += r

        head = (o[0], o[1])
        if head in env_test.start.values() or d:
            if head in env_test.start.values():
                pathx.append(target[0])
                pathy.append(target[1])
            paths_x.append(pathx)
            paths_y.append(pathy)
            pathx = [o[0]]
            pathy = [o[1]]
        else:
            pathx.append(o[0])
            pathy.append(o[1])

    return paths_x, paths_y, total_ep_ret

def draw_board(paths_x, paths_y, board, save_name):
    
    import matplotlib.pyplot as plt
    width, height = board.shape

    # create a 8" x 8" board
    fig = plt.figure(figsize=[8,8])
    # fig.patch.set_facecolor((1,1,.8))

    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(40):
        ax.plot([x, x], [0,39], color=(0.5,0.5,0.5,1))
    for y in range(40):
        ax.plot([0, 39], [y,y], color=(0.5,0.5,0.5,1))

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
    ax.set_xlim(0,39)
    ax.set_ylim(0,39)
    
    fig.savefig(save_name, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':

    epochs = 100

    test_board = np.genfromtxt("./test_board.csv", delimiter=',')
    env = CREnv(board=test_board)

    rl_policy = policy("tf1_save")

    for ep in range(epochs):
        paths_x, paths_y, total_ep_ret = get_paths(env, rl_policy)

        abs_board = test_board
        for i in range(abs_board.shape[0]):
            for j in range(abs_board.shape[1]):
                abs_board[i][j] = abs(abs_board[i][j])
        draw_board(paths_x, paths_y, abs_board, "./test_res/test_"+str(ep)+".jpg")


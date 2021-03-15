import numpy as np
import tensorflow as tf
from config import Params
from wrapper_env import env_wrapper
import gym

from nn_architecures import network_builder
from models import CategoricalModel, GaussianModel
from policy import Policy
from CREnv import CREnv
from copy import copy
import os

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

def rollout(env, model, res_idx):

    res_folder_name = "route_results"
    if not os.path.isdir(res_folder_name):
        os.mkdir(res_folder_name)
    saved_fig_name = os.path.join(res_folder_name, "route_board_{}.png".format(res_idx))

    obs_vec, obs_vis, rew, done = env.reset()
    board = np.absolute(np.genfromtxt("./board.csv", delimiter=','))
    ep_len = 0
    ep_rew = 0

    paths_x = []
    paths_y = []
    path_x = []
    path_y = []
    while not done:
        path_x.append(obs_vec[0][0])
        path_y.append(obs_vec[0][1])
        mask = env.env.compute_mask()
        action_t, logp_t, value_t = model.get_action_logp_value({"vec_obs": obs_vec, "vis_obs": obs_vis}, mask=mask)

        # print(obs_vec)
        obs_vec, obs_vis, rew, done, info = env.step(action_t)
        ep_rew += rew
        ep_len += 1

        if env.env.pairs_idx-2>len(paths_x):
            paths_x.append(path_x)
            paths_y.append(path_y)
            path_x = []
            path_y = []
    draw_board(paths_x, paths_y, board, saved_fig_name)
    return ep_rew, ep_len

if __name__ == "__main__":

    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    params = Params()          # Get Configuration | HORIZON = Steps per epoch

    tf.random.set_seed(params.env.seed)                                     # Set Random Seeds for np and tf
    np.random.seed(params.env.seed)

    # env = create_batched_env(params.env.num_envs, params.env)               # Create Environment in multiprocessing mode
    # env = gym.make('CartPole-v0')
    env = CREnv()
    env = env_wrapper(env, params.env)

    network = network_builder(params.trainer.nn_architecure) \
        (hidden_sizes=params.policy.hidden_sizes, env_info=env.env_info)    # Build Neural Network with Forward Pass

    model = CategoricalModel if env.env_info.is_discrete else GaussianModel
    model = model(network=network, env_info=env.env_info)                   # Build Model for Discrete or Continuous Spaces

    # if params.trainer.load_model:
    print('Loading Model ...')
    model.load_weights("./saved_model/oneNet_vec_oneHit")           # Load model if load_model flag set to true

    for i in range(50):
        ep_rew, ep_len = rollout(env, model, i)
        print(ep_rew, ep_len)

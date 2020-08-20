#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import keras.backend as K
import tensorflow as tf
import time
import core_tf as core
import matplotlib.pyplot as plt

from env import circuitBoard

from logx import EpochLogger

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, select_act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.select_act_buf = np.zeros(core.combined_shape(size, select_act_dim), dtype=np.float32)

    def store(self, obs, act, select_actor, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.select_act_buf[self.ptr] = select_actor
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.select_act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = [0]*nb_classes
    targets[data[0]] = 1
    return targets


def dyn_rew_plot(data, epoch ,epochs):

    xdata = []
    ydata = []
     
    # plt.show()
     
    axes = plt.gca()
    axes.set_xlim(0, epochs+1)
    axes.set_ylim(-10, 0)
    line, = axes.plot(xdata, ydata, 'r-')
    
    xdata = list(range(epoch+1))
    ydata = list(data)
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    plt.draw()
    plt.pause(1e-17)

def save_rew_fig(data, epochs):
    
    xdata = []
    ydata = []

    fig = plt.figure()
    axes = plt.gca()
    axes.set_xlim(0, epochs+1)
    axes.set_ylim(-10, 0)
    line, = axes.plot(xdata, ydata, 'r-')
    
    xdata = list(range(epochs))
    ydata = list(data)
    line.set_xdata(xdata)
    line.set_ydata(ydata)
    fig.savefig('rew_figure.svg')

"""

Proximal Policy Optimization 

"""

def ppo(env_class, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env = env_class

    obs_dim = 900
    act_dim = 4

    # Share information about action space with policy architecture
    ac_kwargs['action_dim'] = act_dim

    # Inputs to computation graph
    x_ph = core.placeholder(obs_dim)
    a_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
    select_actor_ph = tf.placeholder(dtype=tf.float32, shape=(None, act_dim))

    adv_ph, ret_ph, p_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, p, p_pi, v, p_all = actor_critic(x_ph, a_ph, select_actor_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, select_actor_ph, adv_ph, ret_ph, p_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, p_pi, p_all]

    # Experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPOBuffer(obs_dim, None, act_dim, local_steps_per_epoch, gamma, lam)

    # PPO objectives
    ratio = p / (p_old_ph + 1e-8)       # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - v)**2, name="v_loss")

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(tf.math.log(p_old_ph) - tf.math.log(p))      # a sample estimate for KL-divergence, easy to compute
    # approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute

    # Optimizers
    train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    # gradients, variables = zip(*optimizer.compute_gradients(pi_loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    # train_pi = optimizer.apply_gradients(zip(gradients, variables))
    train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'actor_mask': select_actor_ph}, outputs={'pi': pi, 'v': v})

    def update():

        inputs = {k:v for k,v in zip(all_phs, buf.get())}
        # print(inputs)

        # pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)
        
        for i in range(train_pi_iters):
            _, kl, loss_value = sess.run([train_pi, approx_kl, pi_loss], feed_dict=inputs)
            print(loss_value)
            # kl = mpi_avg(kl)
            print("kl values are:")
            print(kl)
            # # print(sess.run(pi_loss, feed_dict=inputs))
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
        # saver.save(sess, 'actor') # save actor model
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)
        # saver.save(sess, 'critic') # save critic model

        pi_l_new, v_l_new, kl = sess.run([pi_loss, v_loss, approx_kl], feed_dict=inputs)
        print(pi_l_new, v_l_new, kl)


    env.reset()
    obs, r, d, ep_ret, ep_len = env.board.ravel(), 0, False, 0, 0

    results = []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            prob_all = []

            o = obs
            act_mask = env.getActionProbMask(env.getPossibleActions())
            # print(act_mask)

            a, v_t, p_t, p_all_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1), select_actor_ph: act_mask.reshape(1,-1)})
            a = a[0]
            # print(p_all_t)

            env = env.takeAction(a)
            from view import display
            # if epoch>95:
            #     display(env.board)

            obs2, r, d, a_m = env.board.ravel(), env.getReward(), env.isTerminal(), act_mask

            ep_ret += r
            ep_len += 1

            buf.store(obs, a, a_m, r, v_t, p_t)

            # Update obs (critical!)
            obs = obs2

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                # else:
                #     buf.last_terminal_idx = buf.ptr
                last_val = 0 if d else sess.run(v, feed_dict={x_ph: obs.reshape(1,-1)})
                buf.finish_path(last_val)

                env.reset()
                obs, r, d, ep_ret, ep_len = env.board.ravel(), 0, False, 0, 0

        # Perform PPO update: train network
        update()
        results.append(np.mean(buf.rew_buf))
        # print(results)
        dyn_rew_plot(np.array(results), epoch, epochs)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

    save_rew_fig(np.array(results), epochs)
    np.savetxt("ave_rew_ppo.csv", np.array(results), delimiter=',')

if __name__ == '__main__':
    # parse arguments

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scenario', default='simple_spread.py', help='Path of the scenario Python script.')

    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    board = np.genfromtxt("test_board.csv", delimiter=',')

    env = circuitBoard(board)
    logger_kwargs = dict(output_dir="./Model_save")

    ppo(env, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs)

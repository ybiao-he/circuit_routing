import numpy as np
import tensorflow as tf

EPS = 1e-8

class ReplayBuffer:
    """
    A buffer for storing trajectories experienced by a agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.reset()

    def reset(self):
        self.obs_buf = []
        self.act_buf = []
        self.adv_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.p_all_buf = []
        self.ptr, self.path_start_idx = 0, 0  

    def store(self, obs, act, rew, val, logp, p_all):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)
        self.p_all_buf.append(p_all)
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
        rews = np.array(self.rew_buf[path_slice]+[last_val])
        vals = np.array(self.val_buf[path_slice]+[last_val])
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        obs_buf = np.array(self.obs_buf, dtype='float32')
        act_buf = np.array(self.act_buf, dtype='float32')
        adv_buf = np.array(self.adv_buf, dtype='float32')
        ret_buf = np.array(self.ret_buf, dtype='float32')
        logp_buf = np.array(self.logp_buf, dtype='float32')
        p_all_buf = np.array(self.p_all_buf, dtype='float32')
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(adv_buf), np.std(adv_buf)
        adv_buf = (adv_buf - adv_mean) / adv_std
        return [obs_buf, act_buf, adv_buf, ret_buf, logp_buf]

    def get_length(self):
        return len(self.obs_buf)



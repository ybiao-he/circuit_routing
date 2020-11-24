# Off-policy RL training
from tf_model_load import policy
import numpy as np
import core as core
import gym
from CREnv import CREnv

def test_policy(env, policy):

    o, ep_ret, ep_len = env.reset(), 0, 0
    saved_ep_ret = []
    local_steps_per_epoch = 1000

    # Main loop: collect experience in env and update/log each epoch
    ep_ret_tem = []
    for t in range(local_steps_per_epoch):
        a = policy.predict_act(o)

        logp_all = np.exp(policy.predict_probs(o))
        print(logp_all)
        o2, r, d, _ = env.step(a[0])
        ep_ret += r
        # Update obs (critical!)
        o = o2

        terminal = d
        if terminal or (t==local_steps_per_epoch-1):
            if not(terminal):
                print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
            ep_ret_tem.append(ep_ret)
            # print(ep_ret_tem)
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

    return ep_ret_tem

if __name__ == '__main__':

    env = CREnv()

    rl_policy = policy('tf1_save')
    ret = test_policy(env, rl_policy)

    print(ret)
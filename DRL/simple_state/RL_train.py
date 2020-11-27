# Off-policy RL training
from RL_Policy import policy
import numpy as np
import core as core
import gym
from CREnv import CREnv

def test_policy(env, policy):

    o, ep_ret, ep_len = env.reset(), 0, 0
    saved_ep_ret = []

    # Main loop: collect experience in env and update/log each epoch
    ep_ret_tem = []
    for t in range(local_steps_per_epoch):
        a = policy.predict_act(o)

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

    epochs = 100
    local_steps_per_epoch = 1500

    env = CREnv()

    buf = core.Buffer()
    rl_policy = policy(env, "ppo")

    o, ep_ret, ep_len = env.reset(), 0, 0
    saved_ep_ret = []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        ep_ret_tem = []
        for t in range(local_steps_per_epoch):
            a = rl_policy.predict_act(o)
            v_t = rl_policy.predict_value(o)
            logp_t = rl_policy.get_prob_act(o, a[0])

            action_logps = rl_policy.predict_probs(o)[0]
            print(action_logps, v_t)

            o2, r, d, _ = env.step(a[0])
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a[0], r, v_t, logp_t, 0)

            # Update obs (critical!)
            o = o2

            terminal = d
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = 0 if d else rl_policy.predict_value(o)
                buf.finish_path(last_val)
                # print(t)
                ep_ret_tem.append(ep_ret)
                # print(ep_ret_tem)
                # logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        saved_ep_ret.append(sum(ep_ret_tem) / len(ep_ret_tem))
        # Perform VPG update!
        rl_policy.update(buf)
        buf.reset()

    np.savetxt("ave_rew.csv", np.array(saved_ep_ret), delimiter=',')

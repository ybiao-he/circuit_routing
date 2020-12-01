from __future__ import division

import numpy as np
from stable_baselines.common.noise import AdaptiveParamNoiseSpec


from CREnv import CREnv

from stable_baselines.common.env_checker import check_env


def test_env(env):

    # If the environment don't follow the interface, an error will be thrown
    check_env(env, warn=True)

    # Testing the environment

    obs = env.reset()
    env.render()

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())

    import random
    # Hardcoded best agent: always go left!
    n_steps = 20
    for step in range(n_steps):
        action = random.randint(0, 3)
        print("Step {}".format(step + 1))
        obs, reward, done, info = env.step(action)
        print('obs=', obs.shape, 'reward=', reward, 'done=', done)
        env.render()
        if done:
            print("Goal reached!", "reward=", reward)
            break

if __name__ == '__main__':

    # Try it with Stable-Baselines
    from stable_baselines import DQN, PPO2, A2C, ACKTR, ACER, DDPG, TRPO, PPO1
    from stable_baselines.common.cmd_util import make_vec_env
    import os
    from stable_baselines import results_plotter
    from stable_baselines.bench import Monitor

    import tensorflow as tf

    from stable_baselines.deepq.policies import CnnPolicy

    from CRPolicy import CRPolicy

    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Create and wrap the environment
    # Instantiate the env
    env = CREnv()
    # wrap it
    # env = make_vec_env(lambda: env, n_envs=1)

    env = Monitor(env, log_dir)

    env = make_vec_env(lambda: env, n_envs=1)

    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[64, 64])

    # Train the agent
    model = PPO2("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, cliprange=0.2)

    # model = PPO1(CRPolicy, env, verbose=1)

    # model = DQN(CnnPolicy, env, verbose=1)

    # Train the agent
    time_steps = 5000
    model.learn(total_timesteps=int(time_steps))

    save_path = os.path.join(log_dir, 'best_model')
    model.save(save_path)

    del model

    model = PPO2.load(save_path, env)
    model.learn(total_timesteps=int(time_steps))

    # import matplotlib.pyplot as plt

    # results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO1 for routing")
    # plt.savefig('RL_result1.png')

    # # Test the trained agent
    # obs = env.reset()
    # n_steps = 200
    # for step in range(n_steps):
    #     action, _ = model.predict(obs, deterministic=True)
    #     print("Step {}".format(step + 1))
    #     print("Action: ", action)
    #     obs, reward, done, info = env.step(action)
    #     print('obs=', obs.shape, 'reward=', reward, 'done=', done)
    #     env.render(mode='console')
    #     if done:
    #         # Note that the VecEnv resets automatically
    #         # when a done signal is encountered
    #         print("Goal reached!", "reward=", reward)
    #         break

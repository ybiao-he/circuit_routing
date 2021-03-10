
import numpy as np
import tensorflow as tf
from roller import Roller
from config import Params
from wrapper_env import env_wrapper
import gym

from nn_architecures import network_builder
from models import CategoricalModel, GaussianModel
from policy import Policy
from CREnv import CREnv

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

    roller = Roller(env, model, params.trainer.steps_per_epoch,
                    params.trainer.gamma, params.trainer.lam)               # Define Roller for genrating rollouts for training

    ppo = Policy(model=model)            # Define PPO Policy with combined loss
    # rollouts, infos = roller.rollout()
    # print(rollouts) 

    tensorboard_dir = './tensorboard'
    writer = tf.summary.create_file_writer(tensorboard_dir)
    with writer.as_default():

        for epoch in range(params.trainer.epochs):                              # Main training loop for n epochs
            rollouts, infos = roller.rollout()                                  # Get Rollout and infos
            outs = ppo.update(rollouts)                                         # Push rollout in ppo and update policy accordingly
            
            tf.summary.scalar('pi_loss', np.mean(outs['pi_loss']), epoch)
            tf.summary.scalar('v_loss', np.mean(outs['v_loss']), epoch)
            tf.summary.scalar('entropy_loss', np.mean(outs['entropy_loss']), epoch)
            tf.summary.scalar('approx_ent', np.mean(outs['approx_ent']), epoch)
            tf.summary.scalar('approx_kl', np.mean(outs['approx_kl']), epoch)
            tf.summary.scalar('eps_reward', np.mean(infos['ep_rews']), epoch)
            tf.summary.scalar('pi_loss', outs['pi_loss'], epoch)
            writer.flush()
    model.save_weights( 'saved_model/oneNet_vec_oneHit', 
                        save_format='tf')
    # env.close()                                                             # Dont forget closing the environment

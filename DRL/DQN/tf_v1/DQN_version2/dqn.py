import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from gym.spaces import Box, Discrete

from memory import ReplayBuffer
from networks import dense_nn, conv2d_net


class DqnPolicy(object):
    def __init__(self, env, name,
                 training=True,
                 gamma=0.99,
                 batch_size=32,
                 model_type='conv',
                 model_params=None,
                 step_size=1,  # only > 1 if model_type is 'lstm'.
                 layer_sizes=[32, 32]):
        """
        model_params: 'layer_sizes', 'step_size', 'lstm_layers', 'lstm_size'
        """

        self.env = env
        self.name = name

        self.gamma = gamma
        self.batch_size = batch_size
        self.training = training
        self.model_type = model_type
        self.model_params = model_params or {}
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.sess = tf.Session()

    @property
    def state_dim(self):
        # Returns: A list
        if self.model_type == 'dense':
            return [np.prod(list(self.env.observation_space.shape))]
        elif self.model_type in ('conv', 'lstm'):
            return list(self.env.observation_space.shape)
        else:
            assert NotImplementedError()

    def init_target_q_net(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def scope_vars(self, scope, only_trainable=True):
        collection = tf_v1.GraphKeys.TRAINABLE_VARIABLES if only_trainable else tf.GraphKeys.VARIABLES
        variables = tf_v1.get_collection(collection, scope=scope)
        assert len(variables) > 0
        print(f"Variables in scope '{scope}':")
        for v in variables:
            print("\t" + str(v))
        return variables 

    def create_q_networks(self):
        # The first dimension should have batch_size * step_size
        self.states = tf.placeholder(tf.float32, shape=(None, *self.state_dim), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(None, *self.state_dim),
                                          name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.actions_next = tf.placeholder(tf.int32, shape=(None,), name='action_next')
        self.rewards = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done_flags = tf.placeholder(tf.float32, shape=(None,), name='done')

        # The output is a probability distribution over all the actions.

        # net_class, net_params = dense_nn, {}
        net_class, net_params = conv2d_net, {}

        self.q = net_class(self.states, self.layer_sizes + [self.act_size], name='Q_primary',
                           **net_params)
        self.q_target = net_class(self.states_next, self.layer_sizes + [self.act_size],
                                  name='Q_target', **net_params)

        # The primary and target Q networks should match.
        self.q_vars = self.scope_vars('Q_primary')
        self.q_target_vars = self.scope_vars('Q_target')
        assert len(self.q_vars) == len(self.q_target_vars), "Two Q-networks are not same."

    def build(self):
        self.create_q_networks()

        self.actions_selected_by_q = tf.argmax(self.q, axis=-1, name='action_selected')
        action_one_hot = tf.one_hot(self.actions, self.act_size, 1.0, 0.0, name='action_one_hot')
        pred = tf.reduce_sum(self.q * action_one_hot, reduction_indices=-1, name='q_acted')

        max_q_next_target = tf.reduce_max(self.q_target, axis=-1)

        y = self.rewards + (1. - self.done_flags) * self.gamma * max_q_next_target

        self.learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
        self.loss = tf.reduce_mean(tf.square(pred - tf.stop_gradient(y)), name="loss_mse_train")
        self.optimizer = tf_v1.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss, name="adam_optim")

        with tf.variable_scope('summary'):
            q_summ = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.act_size):
                q_summ.append(tf_v1.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summ = tf_v1.summary.merge(q_summ, 'q_summary')

            self.q_y_summ = tf_v1.summary.histogram("batch/y", y)
            self.q_pred_summ = tf_v1.summary.histogram("batch/pred", pred)
            self.loss_summ = tf_v1.summary.scalar("loss", self.loss)

            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.ep_reward_summ = tf_v1.summary.scalar('episode_reward', self.ep_reward)

            self.merged_summary = tf_v1.summary.merge_all(key=tf_v1.GraphKeys.SUMMARIES)

        self.sess.run(tf_v1.global_variables_initializer())
        self.init_target_q_net()

    def update_target_q_net(self):
        self.sess.run([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def act(self, state, epsilon=0.1):
        if self.training and np.random.random() < epsilon:
            return self.env.action_space.sample()

        with self.sess.as_default():
            return self.actions_selected_by_q.eval({self.states: [state]})[-1]

    @property
    def act_size(self):
        # number of options of an action; this only makes sense for discrete actions.
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        else:
            return None

    ##########


    def train(self):

        lr = 0.001
        lr_decay = 1.0
        epsilon = 1.0
        epsilon_final = 0.01
        memory_capacity = 100000
        target_update_every_step = 100
        n_episodes = 500
        warmup_episodes = 450
        log_every_episode = 10

        buff = ReplayBuffer(capacity=memory_capacity)

        reward = 0.
        reward_history = []
        reward_averaged = []

        max_reward = -10000
        # best_traj = []

        eps = epsilon
        annealing_episodes = warmup_episodes or n_episodes
        eps_drop = (epsilon - epsilon_final) / annealing_episodes
        print("eps_drop:", eps_drop)
        step = 0

        for n_episode in range(n_episodes):
            ob = self.env.reset()
            done = False
            traj = []
            num_step = 0
            while not done:
                num_step += 1
                a = self.act(ob, eps)
                new_ob, r, done, info = self.env.step(a)
                step += 1
                reward += r

                traj.append([ob, a, r, new_ob, done])
                ob = new_ob

                # No enough samples in the buffer yet.
                if buff.size < self.batch_size:
                    continue

                # Training with a mini batch of samples!
                batch_data = buff.sample(self.batch_size)
                # print(batch_data)
                feed_dict = {
                    self.learning_rate: lr,
                    self.states: batch_data['s'],
                    self.actions: batch_data['a'],
                    self.rewards: batch_data['r'],
                    self.states_next: batch_data['s_next'],
                    self.done_flags: batch_data['done'],
                    self.ep_reward: reward_history[-1],
                }

                _, q_val, q_target_val, loss, summ_str = self.sess.run(
                    [self.optimizer, self.q, self.q_target, self.loss, self.merged_summary],
                    feed_dict
                )
                # self.writer.add_summary(summ_str, step)
                if step % target_update_every_step:
                    self.update_target_q_net()

            # Add all the transitions of one trajectory into the replay memory.
            print(num_step)
            buff.add(traj)

            # if reward>max_reward:
            #     best_traj = [a_t[1] for a_t in traj]

            # One episode is complete.
            reward_history.append(reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            reward = 0.

            # Annealing the learning and exploration rate after every episode.
            lr *= lr_decay
            if eps > epsilon_final:
                eps = max(eps - eps_drop, epsilon_final)

            if reward_history and log_every_episode and n_episode % log_every_episode == 0:
                # Report the performance every `every_step` steps
                print(
                    "[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lr:{:.4f}, eps:{:.4f}".format(
                        n_episode, step, np.max(reward_history),
                        np.mean(reward_history[-10:]), reward_history[-5:],
                        lr, eps, buff.size
                    ))
                # self.save_checkpoint(step=step)

        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        np.savetxt("reward_history_revise_reward.csv", np.array(reward_history), delimiter=",")
        # np.savetxt("best_traj.csv", np.array(best_traj), delimiter=",")

        import matplotlib.pyplot as plt
        plt.plot(reward_history)
        plt.savefig("training_plot_revise_reward.png")

if __name__ == '__main__':

    from CREnv import CREnv
    env = CREnv()
    env.reset()
    DQN = DqnPolicy(env=env, name='dqn')
    DQN.build()
    DQN.train()

    # import gym
    # env = gym.make('CartPole-v1')
    # env.reset()
    # DQN = DqnPolicy(env=env, name='dqn')
    # DQN.build()
    # DQN.train()
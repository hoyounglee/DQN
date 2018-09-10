"""
https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
"""
from DQN import DQN
from memory import Memory
from collections import deque
import tensorflow as tf
import random

import numpy as np

class DQNAgent:
    def __init__(self, sess,env, window_size, input_shape, gamma, batch_size, update_freq,
                 is_duel, is_double, is_per, is_distributional,  num_step, is_noisy, learning_rate, train_step):

        self.sess = sess
        self.env = env
        self.per = is_per
        self.noisy = is_noisy
        self.dist = is_distributional
        self.duel = is_duel
        self.double = is_double
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_step = 500000
        self.beta_start = 0.4
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_steps = num_step
        self.update_freq = update_freq
        self.learning_rate = learning_rate
        self.mem_size = 1000000
        self.num_actions = env.num_action.n
        self.train_step = train_step
        self.input_shape = input_shape
        self.window_size =window_size
        self.history = None#np.zeros(shape=(1, self.input_shape[0], self.input_shape[1], self.window_size), dtype=np.uint8)
        self.state = None
        self.update_network = False
        if self.dist:
            self.num_atoms = 51
        else:
            self.num_atoms = 1

        #self, sess, window_size, input_shape, name='dqn',double=True, duel=False, dist=False, noisy=False, trainable=True
        self.predict_network = DQN(self.sess, window_size, input_shape, self.num_actions, self.num_atoms, name='pred_net', double =self.double, duel=self.duel, dist=self.dist, noisy=self.noisy, trainable=True )
        self.target_network = DQN(self.sess, window_size, input_shape, self.num_actions, self.num_atoms, name='target_net', double =self.double, duel=self.duel, dist=self.dist, noisy=self.noisy, trainable=True )
        self.target_network.create_copy_op(self.predict_network)

        if self.per == 1 :
            self.memory = Memory(self.mem_size, self.n_steps, self.gamma)
        else:
            self.memory = deque()

        with tf.variable_scope('optimizer'):
            self.targets = tf.placeholder('float32', [None], name='target_q')
            self.actions = tf.placeholder('int64', [None], name='action')
            actions_onehot = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name='action_onehot')
            pred_q = tf.reduce_sum(self.predict_network.outputs * actions_onehot, reduction_indices=1, name='q_acted')

            self.importance_weights = tf.placeholder('float32', [None], name='importance_weights')

            if self.per:
                # use importance sampling
                self.delta = tf.square(self.targets - pred_q, name='squared_error')
            else:
                # use huber loss
                td_error = self.targets - pred_q
                self.delta = tf.where(tf.abs(td_error) < 1.0,
                                      0.5 * tf.square(td_error),
                                      tf.abs(td_error) - 0.5, name='clipped_error')

            self.loss = tf.reduce_mean(tf.multiply(self.importance_weights, self.delta), name='loss')

            optimizer = tf.train.AdamOptimizer(self.learning_rate, epsilon=1.5e-4)
            self.optim = optimizer.minimize(self.loss, global_step=self.train_step)

    def reset(self):
        state = self.env.reset()
        self.history = np.stack((state, state, state, state), axis=2)
        self.history = np.reshape([self.history],(self.input_shape[0], self.input_shape[1], self.window_size))

    def train(self, episode):

        self.cnt = self.sess.run(self.train_step)

        if self.per == 1:
            beta = min(1.0, self.beta_start + (1 - self.beta_start) * float(self.cnt) / float(self.eps_step))
            samples, weights = self.memory.sample(self.batch_size, beta)
        else:
            # random.sample activates in dic or list
            samples = random.sample(list(self.memory), self.batch_size)
            weights = np.ones(self.batch_size)

        batch_s = [] # state
        batch_r = [] # reward
        batch_a = [] # action
        batch_n = [] # next state
        batch_t = [] # terminal flag

        if self.per:
            for i in range(len(samples)):
                batch_s.append(samples[i][1][0])
                batch_r.append(samples[i][1][1])
                batch_a.append(samples[i][1][2])
                batch_n.append(samples[i][1][3])
                batch_t.append(samples[i][1][4])
        else:
            for i in samples:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])

        batch_s = np.array(batch_s)
        batch_r = np.array(batch_r)
        batch_a = np.array(batch_a)
        batch_n = np.array(batch_n)
        batch_t = np.array(batch_t)

        batch_n = np.float32(batch_n / 255.0)
        batch_s = np.float32(batch_s / 255.0)



        if self.double:
            pred_next_max_action = self.predict_network.calc_actions(batch_n)
            target_next_qmax = self.target_network.calc_outputs_with_idx(batch_n,[[idx, pred_a] for idx, pred_a in enumerate(pred_next_max_action)])
            target_q = (1. - batch_t) * self.gamma * target_next_qmax + batch_r
            # print(batch_r)
        else:
            target_next_qmax = self.target_network.calc_max_outputs(batch_n)
            target_q = (1. - batch_t) * self.gamma * target_next_qmax + batch_r


        _, q_t, loss, step = self.sess.run(
            [self.optim, self.predict_network.outputs, self.loss, self.train_step],
            {self.targets: target_q,
             self.actions: batch_a,
             self.predict_network.inputs: batch_s,
             self.importance_weights: weights})

        if self.per:
            for i in range(len(batch_a)):
                error = abs(target_q[i] - q_t[i][int(batch_a[i])])
                self.memory.update(samples[i][0], error)


        if step % self.update_freq == 0:
            print(episode, " episode, ", step, "th steps update target network")
            self.update_network = True
            self.target_network.run_copy()


    def get_action(self, history, Training=True):

        if Training == True:
            self.cnt = self.sess.run(self.train_step)
            eps = max(self.eps_end, self.eps_start - float(self.cnt)/float(self.eps_step))
            # print('epsilon : ', eps)
            if np.random.rand() < eps and not self.noisy:
                # exploration
                move = np.random.randint(0, self.num_actions)

                max_q_pred = None
            else:

                ob = np.float32(history / 255.0)
                ob = np.reshape(ob, (1, self.input_shape[0], self.input_shape[1], self.window_size))
                move = self.predict_network.calc_actions(ob)[0]
                max_q_pred = max(self.predict_network.calc_outputs(ob)[0])

        else :
            ob = np.float32(history / 255.0)
            ob = np.reshape(ob, (1, self.input_shape[0], self.input_shape[1], self.window_size))
            move = self.predict_network.calc_actions(ob)[0]
            max_q_pred = max(self.predict_network.calc_outputs(ob)[0])


        return move, max_q_pred


    def step(self, num_steps, training=True):

        cumulative_reward = 0
        terminal = 0
        last_history = self.history
        last_action = 0
        for _ in range(num_steps):


            action, q_value =self.get_action(self.history, Training=training)
            next_state, reward, terminal = self.env.step(action, Training=training)
            #print("reward:", reward)
            if training:
                reward = np.clip(reward, -1., 1.)

            self.state = next_state
            cumulative_reward += reward
            last_action = action

            s1 = np.reshape(next_state, (self.input_shape[0], self.input_shape[1], 1))
            next_history = np.append(self.history[:, :, 1:], s1, axis=2)


            self.history = next_history
            if terminal == True:
                break


        return last_history, last_action, cumulative_reward, self.history, q_value, terminal

    def evaluate(self, num_episode):

        rewards_list = []

        for _ in range(num_episode):

            cumulative_reward = 0

            self.reset()

            while True :

                action, q_value= self.get_action(self.history, Training=False)
                next_state, reward, terminal = self.env.step(action, Training = False)
                cumulative_reward += reward

                s1 = np.reshape(next_state, (self.input_shape[0], self.input_shape[1], 1))
                next_history = np.append(self.history[:, :, 1:], s1, axis=2)
                self.history = next_history

                if terminal == True:
                    break
            rewards_list.append(cumulative_reward)

        return np.mean(rewards_list), np.std(rewards_list)



    #experience:(old_state, reward, action, new_state, Done)
    def append(self, experience):

        if self.per == 1:
            old_state  = experience[0]
            reward = experience[1]
            action = experience[2]
            new_state = experience[3]
            done = experience[4]

            if self.double:
                ob = np.float32(new_state / 255.0 )

                observation = np.reshape(ob, (1, self.input_shape[0], self.input_shape[1], self.window_size))
                pred_next_max_action = self.predict_network.calc_actions(observation)

                target_next_qmax = self.target_network.calc_outputs_with_idx(observation,
                                                                            [[idx, pred_a] for idx, pred_a in enumerate(pred_next_max_action)])
                target_q = (1. - done) * self.gamma * target_next_qmax + float(reward)

            else:
                ob = np.float32(new_state / 255.0)

                observation = np.reshape(ob, (1, self.input_shape[0], self.input_shape[1], self.window_size))
                target_next_qmax = self.target_network.calc_max_outputs(observation)
                target_q = (1. - done) * self.gamma * target_next_qmax + float(reward)

            ob_last = np.float32(old_state / 255.0 )
            last_observation = np.reshape(ob_last, (1, self.input_shape[0], self.input_shape[1], self.window_size))
            pred_q = self.predict_network.calc_outputs_with_idx(last_observation,[[0, action]])

            error = abs(target_q - pred_q)

            self.memory.add(error[0], experience)
        else:
            self.memory.append(experience)
            if len(self.memory) > self.mem_size:
                self.memory.popleft()
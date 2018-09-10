import argparse
import gym
import numpy as np
import random
import tensorflow as tf
import PIL
import os
from defines import NUM_EVALUATE_EPSIODE
from agent import DQNAgent
from environment import Environment

def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument('--seed', default=10703, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=(84, 84), help='Input shape')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--num_process', default=3, help='Number of process')
    parser.add_argument('--epsilon', default=0.01, help='Exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.0000625, help='Training learning rate.')
    parser.add_argument('--window_size', default=4, type=int, help='Number of frames to feed to the Q-network')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size of the training part')
    parser.add_argument('--num_iteration', default=10000000, type=int, help='number of iterations to train')
    parser.add_argument('--eval_freq', default=20, type=int, help='evaluation_frequency.')
    parser.add_argument('--update_freq', default=10000, type=int, help='update_frequency.')
    parser.add_argument('--is_duel', default=0, type=int, help='Whether use duel DQN, 0 means no, 1 means yes.')
    parser.add_argument('--is_double', default=0, type=int, help='Whether use double DQN, 0 means no, 1 means yes.')
    parser.add_argument('--is_per', default=0, type=int, help='Whether use PriorityExperienceReplay, 0 means no, 1 means yes.')
    parser.add_argument('--is_distributional', default=0, type=int, help='Whether use distributional DQN, 0 means no, 1 means yes.')
    parser.add_argument('--is_noisy', default=0, type=int, help='Whether use NoisyNet, 0 means no, 1 means yes.')
    parser.add_argument('--is_bootstrap', default=0, type=int, help='Whether use bootstrap, 0 means no, 1 means yes.')
    parser.add_argument('--tarin_step', default=100000, type=int, help='number of steps to train start')
    parser.add_argument('--num_step', default=1, type=int, help='Num Step for multi-step DQN')
    parser.add_argument('--is_finetune', default=0, type=int, help='Whether use finetune, 0 means no, 1 means yes. ')
    parser.add_argument('--finetune_model', default='./model/model-380000.cptk', type=str, help='finetune model file')
    parser.add_argument('--summary_path', default='./summary/dqn', type=str, help='result name of summary')


    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)
    print('Environment: %s.' % (args.env,))

    model_path = './model'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)

    env = Environment(args.env, args.window_size, args.input_shape, display=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    train_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False) # steps during training

    summary_writer = tf.summary.FileWriter(args.summary_path)

    total_steps = 0

    with sess.as_default():


        agent = DQNAgent(sess, env,
                         args.window_size,
                         args.input_shape,
                         args.gamma,
                         args.batch_size,
                         args.update_freq,
                         args.is_duel,
                         args.is_double,
                         args.is_per,
                         args.is_distributional,
                         args.num_step,
                         args.is_noisy,
                         args.learning_rate,
                         train_step)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=50)

        if args.is_finetune == 1:
            print(args.finetune_model)
            saver = tf.train.import_meta_graph(args.finetune_model + ".meta")
            saver.restore(sess, args.finetune_model)

        sess.run(global_episodes)

        max_mean_reward = 0

        rewards_list = []

        # episode
        for i in range (args.num_iteration):

            #print(i,"th episode ", total_steps, " steps")
            episode_steps = 0
            cumulative_reward = 0
            agent.reset()
            episode_reward = []
            episode_q_values = []
            num_q_value = 0

            while True:
                # 1 episode

                old_state, action, reward, new_state, q_value, Done = agent.step(args.num_step, training=True)

                agent.append((old_state, reward, action, new_state, Done))

                cumulative_reward += reward

                if total_steps >= args.tarin_step:
                    agent.train(i)

                episode_steps += 1
                total_steps += 1
                episode_reward.append(reward)
                if q_value != None:
                    num_q_value += 1
                    episode_q_values.append(q_value)
                if Done == True:
                    break
                if episode_steps > 5000:
                    break


            print(i, 'th episode, current steps:', episode_steps, 'total_steps:', total_steps,
                  " returns:", cumulative_reward, end=" ")
            if num_q_value >0 :
                print("mean_q_value", round(float(np.mean(episode_q_values)), 4), end=" ")

            print("")
            rewards_list.append(cumulative_reward)

            if i % args.eval_freq == 0:
                reward_mean = float(np.mean(rewards_list))

                summary = tf.Summary()
                summary.value.add(tag='Perf/return', simple_value=float(np.mean(rewards_list)))
                summary_writer.add_summary(summary, i)
                summary_writer.flush()

                #reward_mean, reward_var = agent.evaluate(1)
                print("evalute: %dth episode %f" % (i, float(np.mean(rewards_list))))

                if max_mean_reward < reward_mean and i > 0:
                    max_mean_reward = reward_mean
                    saver.save(sess, model_path + '/model-' + str(i) + 'mean_' + str(max_mean_reward) + '.cptk')

                rewards_list = []

            sess.run(global_episodes.assign_add(1))

if __name__ == '__main__':
    main()

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
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--seed', default=10703, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=(84, 84), help='Input shape')
    parser.add_argument('--gamma', default=0.998, help='Discount factor')
    parser.add_argument('--epsilon', default=0.01, help='Exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.0000625, help='Training learning rate.')
    parser.add_argument('--window_size', default=4, type=int, help='Number of frames to feed to the Q-network')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size of the training part')
    parser.add_argument('--num_iteration', default=100, type=int, help='number of iterations to train')
    parser.add_argument('--is_duel', default=1, type=int, help='Whether use duel DQN, 0 means no, 1 means yes.')
    parser.add_argument('--is_double', default=1, type=int, help='Whether use double DQN, 0 means no, 1 means yes.')
    parser.add_argument('--is_per', default=1, type=int, help='Whether use PriorityExperienceReplay, 0 means no, 1 means yes.')
    parser.add_argument('--is_distributional', default=0, type=int, help='Whether use distributional DQN, 0 means no, 1 means yes.')
    parser.add_argument('--is_noisy', default=0, type=int, help='Whether use NoisyNet, 0 means no, 1 means yes.')
    parser.add_argument('--is_bootstrap', default=0, type=int, help='Whether use bootstrap, 0 means no, 1 means yes.')
    parser.add_argument('--num_step', default=1, type=int, help='Num Step for multi-step DQN')
    parser.add_argument('--trained_model', default='./summary_spaceinvader/model/model-4380mean_13.45.cptk', type=str, help='finetune model file')

    args = parser.parse_args()
    args.input_shape = tuple(args.input_shape)
    print('Environment: %s.' % (args.env,))



    env = Environment(args.env, args.window_size, args.input_shape, display=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    train_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)  # steps during training

    with sess.as_default():


        agent = DQNAgent(sess, env,
                         args.window_size,
                         args.input_shape,
                         args.gamma,
                         args.batch_size,
                            0,
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

        
        saver = tf.train.import_meta_graph(args.trained_model + ".meta")
        saver.restore(sess, args.trained_model)
        graph = tf.get_default_graph()
        
        max_mean_reward = 0

        rewards_list = []
        total_steps = 0
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

                old_state, action, reward, new_state, q_value, Done = agent.step(args.num_step, training=False)

                cumulative_reward += reward

            
                episode_steps += 1
                episode_reward.append(reward)
                total_steps += 1
                if q_value != None:
                    num_q_value += 1
                    episode_q_values.append(q_value)
                if Done == True:
                    break


            print(i, 'th episode, current steps:', episode_steps, 'total_steps:', total_steps,
                  " returns:", cumulative_reward, end=" ")
            if num_q_value >0 :
                print("mean_q_value", round(float(np.mean(episode_q_values)), 4), end=" ")

            print("")
            rewards_list.append(cumulative_reward)

            

if __name__ == '__main__':
    main()

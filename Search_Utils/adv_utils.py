import copy
from tqdm import tqdm
import numpy as np
import torch
import collections
import random

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_adv(env, agent, num_episodes, sample_size):
    return_list = []
    capture_time_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                capture_time_tot = 0
                transition_dict = {'observations': [],'actions': [], 'action_nums': [], 'next_observations': [],
                                   'next_action_nums': [], 'rewards': [], 'dones': []}
                for _ in range(sample_size):
                    info_dict = env.reset()
                    observation = info_dict["observation"]
                    action_num = info_dict["action_num"]
                    done = info_dict["done"]
                    remain_prob = 1.0
                    time_step = 0
                    while not done:
                        action = agent.take_action(observation, action_num)
                        info_dict = env.step(action)
                        transition_dict['observations'].append(observation)
                        transition_dict['actions'].append(action)
                        transition_dict['action_nums'].append(action_num)

                        observation = info_dict['observation']
                        reward = info_dict["reward"]
                        done = info_dict["done"]
                        action_num = info_dict["action_num"]
                        transition_dict['next_observations'].append(observation)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        transition_dict['next_action_nums'].append(action_num)

                        remain_prob += reward
                        time_step += 1
                        episode_return -= reward
                        capture_time_cur_step = time_step * (-reward)
                        capture_time_tot += capture_time_cur_step
                    capture_time_remain = (time_step * 2) * remain_prob
                    capture_time_tot += capture_time_remain
                return_list.append(episode_return / sample_size)
                capture_time_list.append(capture_time_tot / sample_size)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'total_return': '%.3f' % np.mean(return_list[-10:]),
                                      'mean_capture_time': '%.3f' % np.mean(capture_time_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:", env.target_trajectory)
    return return_list, capture_time_list

def train_on_policy_adv2(env, agent, num_episodes, sample_size):
    return_list = []
    capture_time_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                capture_time_tot = 0
                transition_dict = {'observations': [],'actions': [], 'action_nums': [], 'next_observations': [],
                                   'next_action_nums': [], 'rewards': [], 'explore_rewards': [], 'dones': []}
                for _ in range(sample_size):
                    info_dict = env.reset()
                    observation = info_dict["observation"]
                    action_num = info_dict["action_num"]
                    done = info_dict["done"]
                    remain_prob = 1.0
                    time_step = 0
                    while not done:
                        action = agent.take_action(observation, action_num)
                        info_dict = env.step(action)
                        transition_dict['observations'].append(observation)
                        transition_dict['actions'].append(action)
                        transition_dict['action_nums'].append(action_num)

                        observation = info_dict['observation']
                        reward = info_dict["reward"]
                        explore_reward = info_dict['explore_reward']
                        done = info_dict["done"]
                        action_num = info_dict["action_num"]
                        transition_dict['next_observations'].append(observation)
                        transition_dict['rewards'].append(reward)
                        transition_dict['explore_rewards'].append(explore_reward)
                        transition_dict['dones'].append(done)
                        transition_dict['next_action_nums'].append(action_num)

                        remain_prob += reward
                        time_step += 1
                        episode_return -= reward
                        capture_time_cur_step = time_step * (-reward)
                        capture_time_tot += capture_time_cur_step
                    capture_time_remain = (time_step * 2) * remain_prob
                    capture_time_tot += capture_time_remain
                return_list.append(episode_return / sample_size)
                capture_time_list.append(capture_time_tot / sample_size)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'total_return': '%.3f' % np.mean(return_list[-10:]),
                                      'mean_capture_time': '%.3f' % np.mean(capture_time_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:", env.target_trajectory)
    return return_list, capture_time_list
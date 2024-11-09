# compare to previous version add one
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

def update_episode_return(episode_return, reward):
    # Check if the reward is a float
    if isinstance(reward, float):
        episode_return += reward
    # Check if the reward is a list
    elif isinstance(reward, list):
        # Calculate the sum of the list and add it to episode_return
        episode_return += sum(reward)
    return episode_return

def train_on_policy_mures1(env, agents, num_episodes, sample_size):
    return_list = []
    capture_time_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                capture_time_tot = 0
                # individual: observations, actions, next_observations, action_num
                # team: remain_time, next_remain_time, rewards, capture_flag, dones
                transition_dict = {'observations': [],'cur_positions': [], 'possible_next_positions': [], 'actions': [],
                                   'action_nums': [] , 'next_observations': [], 'next_action_nums': [], 'rewards': [],
                                   'team_rewards': [], 'dones': []}
                if getattr(agents, "alg_name", None) == "SMC":
                    __ = agents.signal_random_generate()
                for _ in range(sample_size):
                    observations, info = env.reset()
                    action_nums = info['action_nums']
                    cur_positions = info['current_positions']
                    possible_next_positions = info['next_all_positions']
                    done = False
                    remain_prob = 1.0
                    time_step = 0
                    while not done:
                        actions, probs = agents.take_actions(observations, action_nums)
                        next_observations, reward, done, info = env.step(actions)
                        transition_dict['observations'].append(observations)
                        transition_dict['cur_positions'].append(cur_positions)
                        transition_dict['possible_next_positions'].append(possible_next_positions)
                        transition_dict['actions'].append(actions)
                        transition_dict['action_nums'].append(action_nums)
                        transition_dict['next_observations'].append(next_observations)
                        transition_dict['rewards'].append(reward)
                        transition_dict['team_rewards'].append(sum(reward))
                        transition_dict['dones'].append(done)
                        observations = next_observations
                        action_nums = info['action_nums']
                        transition_dict['next_action_nums'].append(action_nums)
                        cur_positions = info['current_positions']
                        possible_next_positions = info['next_all_positions']
                        episode_return = update_episode_return(episode_return, reward)
                        # *********** prob related calculate ***********:
                        capture_prob_cur_step = update_episode_return(0., reward)
                        remain_prob -= capture_prob_cur_step
                        time_step += 1
                        capture_time_cur_step = time_step * capture_prob_cur_step
                        capture_time_tot = update_episode_return(capture_time_tot, capture_time_cur_step)
                    # capture_time_remain = (time_step + 1) * remain_prob
                    capture_time_remain = (time_step *2) * remain_prob
                    capture_time_tot = update_episode_return(capture_time_tot, capture_time_remain)
                    # ***********************************************
                return_list.append(episode_return/sample_size)
                capture_time_list.append(capture_time_tot/sample_size)
                agents.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'total_return': '%.3f' % np.mean(return_list[-10:]),
                                      'mean_capture_time': '%.3f' % np.mean(capture_time_list[-10:])})
                pbar.update(1)
    return return_list, capture_time_list

class ReplayBufferMuRESI:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.current_trajectory = []  # Temporary storage for the current trajectory

    def start_trajectory(self):
        self.current_trajectory = []  # Reset the current trajectory

    def add(self, observations, actions, action_nums, reward, team_reward, next_observations, next_action_nums, done):
        self.current_trajectory.append((observations, actions, action_nums, reward, team_reward, next_observations,
                                        next_action_nums, done))
        if done:
            self.buffer.append(list(self.current_trajectory))  # Store as a list to keep the trajectory together
            self.start_trajectory()  # Prepare for a new trajectory

    def sample(self, num_trajectories):
        num_trajectories = min(num_trajectories, len(self.buffer))
        sampled_trajectories = random.sample(self.buffer, num_trajectories)
        flattened_samples = [experience for trajectory in sampled_trajectories for experience in trajectory]
        observations, actions, action_nums, reward, team_reward, next_observations, \
        next_action_nums, done = zip(*flattened_samples)
        return observations, actions, action_nums, reward, team_reward, next_observations, next_action_nums, done

    def size(self):
        return len(self.buffer)

def train_off_policy_mures1(env, agents, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    capture_time_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                time_step = 0
                remain_prob = 1.0
                capture_time_tot = 0
                episode_return = 0
                observations, info = env.reset()
                action_nums = info['action_nums']
                done = False
                while not done:
                    actions = agents.take_actions(observations, action_nums)
                    next_observations, reward, done, info = env.step(actions)
                    team_reward = sum(reward)
                    next_action_nums = info['action_nums']
                    replay_buffer.add(observations, actions, action_nums, reward, team_reward, next_observations,
                                      next_action_nums, done)
                    observations = next_observations
                    action_nums = next_action_nums
                    episode_return = update_episode_return(episode_return, reward)
                    # *********** prob related calculate ***********:
                    capture_prob_cur_step = update_episode_return(0., reward)
                    remain_prob -= capture_prob_cur_step
                    time_step += 1
                    capture_time_cur_step = time_step * capture_prob_cur_step
                    capture_time_tot = update_episode_return(capture_time_tot, capture_time_cur_step)

                capture_time_remain = (time_step * 2) * remain_prob
                capture_time_tot = update_episode_return(capture_time_tot, capture_time_remain)

                if replay_buffer.size() >= minimal_size:
                    b_o, b_a, b_an, b_r, b_tr, b_no, b_nan, b_d= replay_buffer.sample(batch_size)
                    transition_dict = {'observations': b_o, 'actions': b_a, 'action_nums': b_an, 'rewards': b_r,
                                       'team_rewards':b_tr,'next_observations': b_no, 'next_action_nums': b_nan, 'dones': b_d}
                    agents.update(transition_dict)
                return_list.append(episode_return)
                capture_time_list.append(capture_time_tot)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'total_return': '%.3f' % np.mean(return_list[-10:]),
                                      'mean_capture_time': '%.3f' % np.mean(capture_time_list[-10:])})
                pbar.update(1)
    return return_list, capture_time_list

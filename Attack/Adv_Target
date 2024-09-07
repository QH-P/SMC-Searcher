from Search_Env.escape_env import escape_sim as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Search_Utils import adv_utils
import copy
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import time
import math

class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, action_mask):
        lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.device)
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(2).float()
        x = x[torch.arange(x.size(0)), lengths - 1]
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        logits_masked = logits.masked_fill(action_mask, float('-inf'))
        return F.softmax(logits_masked, dim=1)

class Vnet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super(Vnet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.device)
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(2).float()
        x = x[torch.arange(x.size(0)), lengths - 1]
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        # delta = 1e-6
        # x = torch.clamp(x, min=delta-1, max=-delta)
        return x

class ActorCritic:
    def __init__(self, obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, device):
        self.action_dim = action_dim
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        self.critic = Vnet(obs_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.device = device

    def take_action(self, obs, action_num):
        obs = [obs]
        action_mask = self.create_action_mask(self.action_dim, action_num)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        probs = self.actor(obs, action_mask)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def create_action_mask(self, total_actions, valid_actions):
        action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
        return action_mask

    def create_action_masks(self, total_actions, valid_actions_list, device):
        action_masks = []
        for valid_actions in valid_actions_list:
            action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
            action_masks.append(action_mask)
        action_masks_tensor = torch.tensor(action_masks, dtype=torch.bool).to(device)
        return action_masks_tensor

class Adv_Target:
    def __init__(self, obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.ac = ActorCritic(obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, device)
        self.gamma = gamma
        self.device = device
        self.counter = 0

    def take_action(self, observations, action_nums):
        agent = self.ac
        obs = observations
        action_num = action_nums
        action = agent.take_action(obs, action_num)
        return action

    def update(self, transitions):
        self.counter += 1
        if self.counter % 1000 == 0:
            print("counter:", self.counter)
        trajec_length = len(transitions['observations'])
        agent = self.ac
        agent.actor_optimizer.zero_grad()
        agent.critic_optimizer.zero_grad()

        dones = transitions['dones']
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
        observations = transitions['observations']
        observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in observations]
        actions = transitions['actions']
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = transitions['rewards']
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        ex_rewards = transitions['explore_rewards']
        ex_rewards = torch.tensor(ex_rewards, dtype=torch.float).view(-1, 1).to(self.device)
        rewards = rewards + ex_rewards
        next_observations = transitions['next_observations']
        next_observations = [torch.tensor(next_obs, dtype=torch.float).to(self.device) for next_obs in
                             next_observations]
        action_nums = transitions['action_nums']
        action_masks = agent.create_action_masks(agent.action_dim, action_nums, self.device)
        next_action_nums = transitions['next_action_nums']
        next_action_masks = agent.create_action_masks(agent.action_dim, next_action_nums, self.device)
        # The difference between policy gradient to actor-critic
        td_target = rewards + self.gamma * agent.critic(next_observations) * (1 - dones)
        td_delta = td_target - agent.critic(observations)  # 时序差分误差
        critic_loss = torch.mean(F.mse_loss(agent.critic(observations), td_target.detach()))
        critic_loss.backward()  # Backpropagate through the critic
        agent.critic_optimizer.step()  # Update critic parameters
        probs = agent.actor(observations, action_masks)
        prob = probs.gather(1, actions)
        action_log_probs = torch.log(prob)
        actor_loss = -(td_delta.detach() * action_log_probs).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()


    def _cal_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)
        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + (1 - dones[i])* self.gamma * G
            returns[i] = G
        return returns

def add_component_to_csv(file_path, key, mean_values, variance_values, mean_objective1=None, variance_objective1=None):
    new_component = pd.DataFrame({
        'Key': [key],
        'MeanValues': [mean_values],
        'VarianceValues': [variance_values],
        'MeanObjectiveI': [mean_objective1],
        'VarianceObjectiveI': [variance_objective1]
    })

    # Check if the file exists
    if os.path.exists(file_path):
        # File exists, append the new component without header
        existing_data = pd.read_csv(file_path)
        if key in existing_data['Key'].values:
            # Key exists, replace the row
            existing_data = existing_data[existing_data['Key'] != key]  # Remove old row
            existing_data = pd.concat([existing_data, new_component], ignore_index=True)  # Add new row
        else:
            existing_data = pd.concat([existing_data, new_component], ignore_index=True)
        existing_data.to_csv(file_path, index=False)
    else:
        # File does not exist, create it and add the component with header
        new_component.to_csv(file_path, mode='w', header=True, index=False)

def save_policy_network(policy_net, net_file_path):
    torch.save(policy_net.state_dict(), net_file_path)
    print(f"Policy network saved to {net_file_path}")

if __name__ == "__main__":
    Algorithm = "SMC"
    robot_num = 2
    env_name = "MUSEUM"

    print("Algorithm:{} with robot_num {} in Env:{}".format(Algorithm, robot_num, env_name))
    actor_lr = 1e-4
    critic_lr = 1e-3
    num_episodes = 20000
    hidden_dim = 16
    gamma = 0.95
    sample_size = 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = gym(env_name)
    env.setup(Algorithm, robot_num)
    obs_dim = env.position_embed
    action_dim = env.action_dim

    return_list_set = []
    capture_time_list_set = []

    agent = None
    for p in range(5):
        agent = Adv_Target(obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
        return_list, capture_time_list = adv_utils.train_on_policy_adv2(env, agent, num_episodes, sample_size)
        return_list_set.append(copy.copy(return_list))
        capture_time_list_set.append(copy.copy(capture_time_list))
    script_dir = os.path.dirname(os.path.realpath(__file__))
    net_file_p = os.path.join(script_dir, "Target_Net/{}_{}R{}.pth")
    save_policy_network(agent.ac.actor, net_file_p.format(env_name, Algorithm, robot_num))

    return_list = [sum(col) / len(col) for col in zip(*return_list_set)]
    capture_time_list = [sum(col) / len(col) for col in zip(*capture_time_list_set)]
    return_variance = [sum((xi - mu) ** 2 for xi in col) / len(col) for col, mu in
                       zip(zip(*return_list_set), return_list)]
    capture_time_variance = [sum((xi - mu) ** 2 for xi in col) / len(col) for col, mu in
                             zip(zip(*capture_time_list_set), capture_time_list)]
    episodes_list = list(range(len(return_list)))

    fig, ax1 = plt.subplots()
    ax1.plot(episodes_list, return_list, 'g-')  # 'g-' means green line
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Detection Probability')
    ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis
    ax2.plot(episodes_list, capture_time_list, 'b-')  # 'b-' means blue line
    ax2.set_ylabel('Capture Time')
    plt.title('{} on {} with R = {}'.format(Algorithm, env_name, robot_num))
    plt.show()

    mv_return = adv_utils.moving_average(return_list, 99)
    mv_capture_time = adv_utils.moving_average(capture_time_list, 99)
    fig1, ax1 = plt.subplots()
    ax1.plot(episodes_list, mv_return, 'g-')  # 'g-' means green line
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Detection Probability')
    ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis
    ax2.plot(episodes_list, mv_capture_time, 'b-')  # 'b-' means blue line
    ax2.set_ylabel('Capture Time')
    plt.title('{} on {} with R = {}'.format(Algorithm, env_name, robot_num))
    plt.show()

    selected_mv_return = mv_return.tolist()[::9]
    selected_return_variance = return_variance[::9]
    selected_mv_capture_time = mv_capture_time.tolist()[::9]
    selected_capture_time_variance = capture_time_variance[::9]
    selected_episodes_list = episodes_list[::9]

    file_path = os.path.join(script_dir, "Data/{}_R{}.csv")
    file_path = file_path.format(env_name, robot_num)
    add_component_to_csv(file_path, Algorithm, selected_mv_capture_time, selected_capture_time_variance,
                         selected_mv_return, selected_return_variance)
    # add_or_update_column_based_on_key(file_path, Algorithm, training_duration)
    # save_to_csv(selected_mv_return, Algorithm, file_path)
    fig2, ax1 = plt.subplots()
    ax1.plot(selected_episodes_list, selected_mv_return, 'g-')  # 'g-' means green line
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Detection Probability')
    ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis
    ax2.plot(selected_episodes_list, selected_mv_capture_time, 'b-')  # 'b-' means blue line
    ax2.set_ylabel('Capture Time')
    plt.title('{} on {} with R = {}'.format(Algorithm, env_name, robot_num))
    plt.show()

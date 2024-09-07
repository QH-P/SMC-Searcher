
from Search_Env.classic_env import classic_sim as gym
from Search_Utils.Embedding import EmbeddingLayer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Search_Utils import sample_utils
import copy
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import time
import math
import numpy as np

class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, sig, action_mask):
        lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.device)
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(2).float()
        x = x[torch.arange(x.size(0)), lengths - 1]
        x_cat = torch.cat((x,sig),dim=1)
        x = F.relu(self.fc1(x_cat))
        logits = self.fc2(x)
        logits_masked = logits.masked_fill(action_mask, float('-inf'))
        return F.softmax(logits_masked, dim=1)

class AC_Exe:
    def __init__(self, obs_dim, hidden_dim, action_dim, device):
        self.action_dim = action_dim
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        self.device = device

    def take_action(self, obs, sig, action_num):
        obs = [obs]
        sig = sig.unsqueeze(0)
        action_mask = self.create_action_mask(self.action_dim, action_num)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        probs = self.actor(obs, sig, action_mask)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        prob = probs[0][action].item()
        return action.item(), prob

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

class SMC_Exe:
    def __init__(self, num_agents, signal_tot_num, obs_dim, hidden_dim, action_dim, device):
        self.alg_name = "SMC"
        self.acs = [AC_Exe(obs_dim, hidden_dim, action_dim, device) for _ in
                    range(num_agents)]
        self.num_agents = num_agents
        self.signal_tot_num = signal_tot_num
        self.signal_embedding = EmbeddingLayer(self.signal_tot_num + 1, hidden_dim, 0)
        self.current_signal_num = 0
        self.current_signal_embed = self.signal_embedding(torch.tensor(self.current_signal_num))

    def policy_net_load(self, net_file_p):
        for agent_id, agent in enumerate(self.acs):
            agent.actor.load_state_dict(torch.load(net_file_p.format(env_name, Algorithm, robot_num, agent_id)))
            agent.actor.eval()
            for param in agent.actor.parameters():
                param.requires_grad = False

    def signal_random_generate(self):
        self.current_signal_num = np.random.randint(0, self.signal_tot_num)
        self.current_signal_embed = self.signal_embedding(torch.tensor(self.current_signal_num))
        return self.current_signal_embed

    def take_actions(self, observations, action_nums):
        actions = []
        probs = []
        sig = self.current_signal_embed
        for agent_id, agent in enumerate(self.acs):
            obs = observations[agent_id]
            action_num = action_nums[agent_id]

            action, prob = agent.take_action(obs, sig, action_num)
            actions.append(action)
            probs.append(prob)
        return actions

if __name__ == "__main__":
    Algorithm = "SMC"
    signal_tot_num = 3
    robot_num = 2
    env_name = "MUSEUM"

    sample_num = 10000

    hidden_dim = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env = gym(env_name, robot_num)
    env.seed = 0
    env.robot_num = robot_num
    torch.manual_seed(0)
    obs_dim = env.position_embed
    action_dim = env.action_dim

    agents = SMC_Exe(robot_num, signal_tot_num, obs_dim, hidden_dim, action_dim, device)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    net_file_path_base = os.path.join(script_dir, "Robot_Net/{}_{}T{}R{}.pth")
    # net_file_path_base = "/Users/pqh/PycharmProjects/SMC_Searcher/One_shot_MuRES/Robot_Data/Robot_net/{}_{}T{}R{}.pth"
    agents.policy_net_load(net_file_path_base)

    robots_trajectories_list = sample_utils.sample_from_trained_algorithm(env,agents,sample_num)
    trajectories_file_path_base = os.path.join(script_dir, "Robot_Trajectory/{}_{}R{}.npy")
    trajectories_file_path = trajectories_file_path_base.format(env_name, Algorithm, robot_num)
    np.save(trajectories_file_path, np.array(robots_trajectories_list))
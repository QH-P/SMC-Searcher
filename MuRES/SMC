from Search_Env.classic_env import classic_sim as gym
from Search_Utils.Embedding import EmbeddingLayer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Search_Utils import marl_utils
import copy
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import time
import math
import numpy as np
import networkx as nx
# from Path_Display import draw_graph_multi

# ce-pg is designed for MuRES I, so remain capture time is not needed
class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        # # Freeze the GRU layer
        # for param in self.gru.parameters():
        #     param.requires_grad = False
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

class Vnet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, num_agents):
        super(Vnet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim * num_agents, hidden_size=hidden_dim, batch_first=True)
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
        x = F.sigmoid(self.fc2(x))
        delta = 1e-6
        x = torch.clamp(x, min=delta, max=1 - delta)
        return x

class ActorCritic:
    def __init__(self, obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, device, num_agents):
        self.action_dim = action_dim
        self.actor = PolicyNet(obs_dim, hidden_dim, action_dim).to(device)
        self.critic = Vnet(obs_dim, hidden_dim, action_dim, num_agents).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.device = device

    def take_action(self, obs, sig, action_num):
        obs = [obs]
        sig = sig.unsqueeze(0)
        action_mask = self.create_action_mask(self.action_dim, action_num)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        # print("action_num:",action_num)
        # print("obs:",obs,", remain_time:",remain_time, ", action_mask:", action_mask)
        probs = self.actor(obs, sig, action_mask)
        # print("action_mask:",action_mask)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        prob = probs[0][action].item()
        return action.item(), prob

    def get_prob(self, obs, sig, action, action_num):
        obs = [obs]
        sig = sig.unsqueeze(0)
        action_mask = self.create_action_mask(self.action_dim, action_num)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        probs = self.actor(obs, sig, action_mask)
        print("probs:", probs, ", type:", type(probs))
        prob = probs[action]
        print("prob:", prob, ", type:", type(prob))
        return prob

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

class SMC:
    def __init__(self, env_name, num_agents, signal_tot_num, obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, beta1, beta2, gamma, device):
        self.alg_name = "SMC"
        self.acs = [ActorCritic(obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, device, num_agents) for _ in range(num_agents)]
        self.target_acs = [ActorCritic(obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, device, num_agents) for _ in
                    range(num_agents)]
        for q in range(len(self.acs)):
            ac = self.acs[q]
            target_ac = self.target_acs[q]
            target_ac.critic.load_state_dict(ac.critic.state_dict())
            target_ac.actor.load_state_dict(ac.actor.state_dict())
        self.tau = 0.05

        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.num_agents = num_agents
        self.device = device
        self.counter = 0
        self.signal_tot_num = signal_tot_num
        self.signal_embedding = EmbeddingLayer(self.signal_tot_num + 1, hidden_dim, 0)
        self.current_signal_num = 0
        self.current_signal_embed = self.signal_embedding(torch.tensor(self.current_signal_num))
        self.signal_repo = {}
        self.judge_cross_length = 10
        self.cross_prob_bound = 1.0

        # graph-related variables
        self.G = None
        self.core_number = None
        self.degree_number = None
        self.G_2core = None
        self.c2G_degree_number = None
        self._graph_initial(env_name)

    def _graph_initial(self, graph_name):
        script_dir1 = os.path.dirname(os.path.realpath(__file__))
        script_dir1 = os.path.dirname(script_dir1)
        graph_path = os.path.join(script_dir1, f"Search_Env/Map_Info/{graph_name}.csv")
        graph = pd.read_csv(graph_path)
        self.G = nx.from_pandas_edgelist(graph, 'Room', 'Connected_Room')
        self.core_number = nx.core_number(self.G)
        nodes_to_keep = [node for node, core in self.core_number.items() if core >= 2]
        self.G_2core = self.G.subgraph(nodes_to_keep).copy()
        c2G_degree_number = nx.degree(self.G_2core)
        self.c2G_degree_number = dict(c2G_degree_number)
        degree_number = nx.degree(self.G)
        self.degree_number = dict(degree_number)

    def signal_random_generate(self):
        self.current_signal_num = np.random.randint(0, self.signal_tot_num)
        self.current_signal_embed = self.signal_embedding(torch.tensor(self.current_signal_num))
        return self.current_signal_embed

    def _signal_repo_update(self, transitions):
        if self.current_signal_num in self.signal_repo:
            self.signal_repo[self.current_signal_num].update(transitions)
        else:
            self.signal_repo[self.current_signal_num] = transitions

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

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
        return actions, probs

    def _cal_cross_helper(self, cur_trajectory, cross_trajectory, dones, cross_possible_next_positions):
        cross_actions = [-1 for _ in range(len(dones))]
        cross_positions = [-1 for _ in range(len(dones))]
        cur_path = []
        for i in range(0, len(cur_trajectory)):
            if dones[i]:
                cur_path = []
                continue
            cur_position = cur_trajectory[i]
            # *****************conditions that don't need cross entropy *****************
            if self.core_number[cur_position] == 1 or self.degree_number[cur_position] <= 2:
                continue
            if self.c2G_degree_number[cur_position] == 2 and not (len(cur_path) == 0
                                                                  or cur_path[-1] not in self.c2G_degree_number):
                continue
            # ***************************************************************************
            cur_path.append(cur_position)
            next_position = cur_trajectory[i+1]
            for j in range(i, max(-1,i-self.judge_cross_length), -1):
                if dones[j]:
                    break
                elif cross_trajectory[j] == cur_position and next_position in cross_possible_next_positions[j]:
                    cross_positions[i] = j
                    cross_actions[i] = cross_possible_next_positions[j].index(next_position)
                    break
        return cross_positions, cross_actions

    def _cal_cross_prob(self, transitions):
        trajectories_list = []
        possible_next_positions_list = []
        trajec_length = len(transitions['observations'])
        dones = transitions['dones']
        sigs = transitions['signals']
        # sigs = torch.stack(sigs).to(self.device)
        # print("sigs:", sigs.size())
        for agent_id, agent in enumerate(self.acs):
            trajectory = [transitions['cur_positions'][_][agent_id] for _ in range(trajec_length)]
            possible_next_positions = [transitions['possible_next_positions'][_][agent_id] for _ in range(trajec_length)]
            trajectories_list.append(trajectory)
            possible_next_positions_list.append(possible_next_positions)
        cross_prob_list = [[0. for _ in range(len(trajectories_list[0]))] for __ in range(len(trajectories_list))]
        for agent_id in range(1, self.num_agents):
            cur_trajectory = trajectories_list[agent_id]
            for cross_id in range(0, agent_id):
                agent = self.acs[cross_id]
                observations = [transitions['observations'][_][cross_id] for _ in range(trajec_length)]
                action_nums = [transitions['action_nums'][_][cross_id] for _ in range(trajec_length)]
                cross_trajectory = trajectories_list[cross_id]
                cross_possible_next_positions = possible_next_positions_list[cross_id]
                cross_positions, cross_actions = self._cal_cross_helper(cur_trajectory, cross_trajectory, dones,
                          cross_possible_next_positions)
                # by now, we got the cross positions and corresponding action,
                # when cross_position ==0, it means not cross
                # abbreviate first, and then expand
                observations_abb = []
                action_nums_abb = []
                sigs_abb = []
                actions = []
                for po in range(len(cross_positions)):
                    if cross_positions[po] != -1:
                        observations_abb.append(observations[cross_positions[po]])
                        action_nums_abb.append(action_nums[cross_positions[po]])
                        sigs_abb.append(sigs[cross_positions[po]])
                        actions.append(cross_actions[po])
                if len(observations_abb) == 0:
                    continue
                observations_abb = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in observations_abb]
                action_masks_abb = agent.create_action_masks(agent.action_dim, action_nums_abb, self.device)
                # print("observations_abb:",observations_abb.size())
                sigs_abb = torch.stack(sigs_abb).to(self.device)
                probs = agent.actor(observations_abb, sigs_abb, action_masks_abb)
                actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device).detach()
                prob = probs.gather(1, actions)
                q = 0
                cross_prob = cross_prob_list[agent_id]
                for po in range(len(cross_positions)):
                    if cross_positions[po] != -1:
                        cross_prob[po] += prob[q]
                        q += 1
        prob_tensor = torch.tensor(cross_prob_list)
        clipped_tensor = torch.clamp(prob_tensor, max=1.0)
        cross_prob_list = clipped_tensor.tolist()
        return cross_prob_list

    def _cal_signal_cross_prob(self, transitions):
        trajectories_list = []
        possible_next_positions_list = []
        trajec_length = len(transitions['observations'])
        dones = transitions['dones']
        sigs = transitions['signals']
        transitions_repo = self.signal_repo
        sig_flag = 0
        for sig_num in range(0,self.current_signal_num):
            if sig_num not in transitions_repo:
                continue
            else:
                sig_flag += 1
        for agent_id, agent in enumerate(self.acs):
            trajectory = [transitions['cur_positions'][_][agent_id] for _ in range(trajec_length)]
            possible_next_positions = [transitions['possible_next_positions'][_][agent_id] for _ in range(trajec_length)]
            trajectories_list.append(trajectory)
            possible_next_positions_list.append(possible_next_positions)
        sig_cross_prob_list = [[0. for _ in range(len(trajectories_list[0]))] for __ in range(len(trajectories_list))]
        if sig_flag == 0:
            return sig_cross_prob_list
        for agent_id in range(0, self.num_agents):
            cur_trajectory = trajectories_list[agent_id]
            agent = self.acs[agent_id]
            for sig_num in range(0,self.current_signal_num):
                if sig_num not in transitions_repo:
                    continue
                for cross_id in range(self.num_agents):
                    sig_cross_transitions = transitions_repo[sig_num]
                    cross_sigs = sig_cross_transitions['signals']
                    sig_observations = [sig_cross_transitions['observations'][_][cross_id] for _ in range(trajec_length)]
                    sig_action_nums = [sig_cross_transitions['action_nums'][_][cross_id] for _ in range(trajec_length)]
                    sig_cross_trajectory = [sig_cross_transitions['cur_positions'][_][cross_id] for _ in range(trajec_length)]
                    sig_cross_possible_next_positions = [sig_cross_transitions['possible_next_positions'][_][cross_id] for _ in range(trajec_length)]
                    sig_cross_positions, sig_cross_actions = self._cal_cross_helper(cur_trajectory, sig_cross_trajectory, dones,
                                                                            sig_cross_possible_next_positions)
                    sig_observations_abb = []
                    sig_action_nums_abb = []
                    sig_sigs_abb = []
                    sig_actions = []
                    for po in range(len(sig_cross_positions)):
                        if sig_cross_positions[po] != -1:
                            sig_observations_abb.append(sig_observations[sig_cross_positions[po]])
                            sig_action_nums_abb.append(sig_action_nums[sig_cross_positions[po]])
                            sig_sigs_abb.append(cross_sigs[sig_cross_positions[po]])
                            sig_actions.append(sig_cross_actions[po])
                    if len(sig_observations_abb) == 0:
                        continue
                    sig_observations_abb = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in sig_observations_abb]
                    sig_action_masks_abb = agent.create_action_masks(agent.action_dim, sig_action_nums_abb, self.device)
                    sig_sigs_abb = torch.stack(sig_sigs_abb).to(self.device)
                    probs = agent.actor(sig_observations_abb, sig_sigs_abb, sig_action_masks_abb)
                    sig_actions = torch.tensor(sig_actions, dtype=torch.long).view(-1, 1).to(self.device).detach()
                    prob = probs.gather(1, sig_actions)
                    q = 0
                    sig_cross_prob = sig_cross_prob_list[agent_id]
                    for po in range(len(sig_cross_positions)):
                        if sig_cross_positions[po] != -1:
                            sig_cross_prob[po] += prob[q]
                            q += 1
        # clamp the sig_cross_prob_list less than 1.0:
        prob_tensor = torch.tensor(sig_cross_prob_list)
        clipped_tensor = torch.clamp(prob_tensor, max=1.0)
        sig_cross_prob_list = clipped_tensor.tolist()
        # *************************************************************
        return sig_cross_prob_list

    def _cal_self_cross_helper(self, cur_trajectory, dones, cross_possible_next_positions):
        cross_actions = [-1 for _ in range(len(dones))]
        cross_positions = [-1 for _ in range(len(dones))]
        cur_path = []
        for i in range(1, len(cur_trajectory)):
            if dones[i]:
                i+=1
                cur_path = []
                continue
            cur_position = cur_trajectory[i]
            # *****************conditions that don't need cross entropy *****************
            if self.core_number[cur_position] == 1 or self.degree_number[cur_position] <= 2:
                continue
            if self.c2G_degree_number[cur_position] == 2 and not (len(cur_path) == 0
                                                                  or cur_path[-1] not in self.c2G_degree_number):
                continue
            # ***************************************************************************
            cur_path.append(cur_position)
            next_position = cur_trajectory[i+1]
            for j in range(i-1, max(-1,i-self.judge_cross_length), -1):
                if dones[j]:
                    break
                elif cur_trajectory[j] == cur_position and next_position in cross_possible_next_positions[j]:
                    cross_positions[i] = j
                    cross_actions[i] = cross_possible_next_positions[j].index(next_position)
                    break
        return cross_positions, cross_actions

    def _cal_self_cross_prob(self, transitions):
        trajectories_list = []
        possible_next_positions_list = []
        trajec_length = len(transitions['observations'])
        dones = transitions['dones']
        sigs = transitions['signals']
        for agent_id, agent in enumerate(self.acs):
            trajectory = [transitions['cur_positions'][_][agent_id] for _ in range(trajec_length)]
            possible_next_positions = [transitions['possible_next_positions'][_][agent_id] for _ in range(trajec_length)]
            trajectories_list.append(trajectory)
            possible_next_positions_list.append(possible_next_positions)
        self_prob_list = [[0. for _ in range(len(trajectories_list[0]))] for __ in range(len(trajectories_list))]
        for agent_id in range(1, self.num_agents):
            cur_trajectory = trajectories_list[agent_id]
            agent = self.acs[agent_id]
            observations = [transitions['observations'][_][agent_id] for _ in range(trajec_length)]
            action_nums = [transitions['action_nums'][_][agent_id] for _ in range(trajec_length)]
            self_possible_next_positions = possible_next_positions_list[agent_id]
            self_cross_positions, self_cross_actions = self._cal_self_cross_helper(cur_trajectory, dones, self_possible_next_positions)
            observations_abb = []
            action_nums_abb = []
            sigs_abb = []
            actions = []
            for po in range(len(self_cross_positions)):
                if self_cross_positions[po] != -1:
                    observations_abb.append(observations[self_cross_positions[po]])
                    action_nums_abb.append(action_nums[self_cross_positions[po]])
                    sigs_abb.append(sigs[self_cross_positions[po]])
                    actions.append(self_cross_actions[po])
            if len(observations_abb) == 0:
                continue
            observations_abb = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in observations_abb]
            action_masks_abb = agent.create_action_masks(agent.action_dim, action_nums_abb, self.device)
            # print("observations_abb:",observations_abb.size())
            sigs_abb = torch.stack(sigs_abb).to(self.device)
            probs = agent.actor(observations_abb, sigs_abb, action_masks_abb)
            actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device).detach()
            prob = probs.gather(1, actions)
            q = 0
            self_cross_prob = self_prob_list[agent_id]
            for po in range(len(self_cross_positions)):
                if self_cross_positions[po] != -1:
                    self_cross_prob[po] += prob[q]
                    q += 1
        return self_prob_list

    def _cal_returns(self, rewards, dones):
        returns = torch.zeros_like(rewards)  # Tensor to hold the calculated returns
        G = 0  # Variable to keep track of the accumulated return
        # Iterate through rewards and dones in reverse
        for i in reversed(range(len(rewards))):
            # Reset the accumulator if the episode ends at this step
            G = rewards[i] + (1 - dones[i])* self.gamma * G
            returns[i] = G
        return returns

    def update(self, transitions):
        self.counter += 1
        if self.counter % 1000 == 0:
            print("counter:", self.counter)
        transitions['signals'] = [self.current_signal_embed for _ in range(len(transitions['dones']))]
        self._signal_repo_update(transitions)
        cross_prob_list = self._cal_cross_prob(transitions)
        sig_cross_prob_list = self._cal_signal_cross_prob(transitions)
        self_cross_prob_list = self._cal_self_cross_prob(transitions)
        trajec_length = len(transitions['observations'])
        dones = transitions['dones']
        sigs = transitions['signals']
        sigs = torch.stack(sigs).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        observation_lists = []
        next_observation_lists = []
        for agent_id in range(self.num_agents):
            observations = [transitions['observations'][_][agent_id] for _ in range(trajec_length)]
            next_observations = [transitions['next_observations'][_][agent_id] for _ in range(trajec_length)]
            observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in observations]
            next_observations = [torch.tensor(next_obs, dtype=torch.float).to(self.device) for next_obs in
                                 next_observations]
            observation_lists.append(observations)
            next_observation_lists.append(next_observations)
        global_observations = [torch.cat(tensors, dim=1) for tensors in zip(*observation_lists)]
        global_next_observations = [torch.cat(tensors, dim=1) for tensors in zip(*next_observation_lists)]

        for agent_id, agent in enumerate(self.acs):
            target_ac = self.target_acs[agent_id]
            target_actor = target_ac.actor
            target_critic = target_ac.critic

            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            cross_prob = cross_prob_list[agent_id]
            cross_prob = torch.tensor(cross_prob, dtype=torch.float).view(-1, 1).to(self.device)
            sig_cross_prob = sig_cross_prob_list[agent_id]
            sig_cross_prob = torch.tensor(sig_cross_prob, dtype=torch.float).view(-1, 1).to(self.device)
            self_cross_prob = self_cross_prob_list[agent_id]
            self_cross_prob = torch.tensor(self_cross_prob, dtype=torch.float).view(-1, 1).to(self.device)

            observations = [transitions['observations'][_][agent_id] for _ in range(trajec_length)]
            actions = [transitions['actions'][_][agent_id] for _ in range(trajec_length)]
            rewards = [transitions['rewards'][_][agent_id] for _ in range(trajec_length)]
            next_observations = [transitions['next_observations'][_][agent_id] for _ in range(trajec_length)]
            action_nums = [transitions['action_nums'][_][agent_id] for _ in range(trajec_length)]
            next_action_nums = [transitions['next_action_nums'][_][agent_id] for _ in range(trajec_length)]
            observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in observations]
            next_observations = [torch.tensor(next_obs, dtype=torch.float).to(self.device) for next_obs in next_observations]
            actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)

            td_target = rewards + self.gamma * target_critic(global_next_observations) * (1 -
                                                                           dones)
            td_delta = td_target - target_critic(global_observations)  # 时序差分误差

            # returns = self._cal_returns(rewards, dones)
            action_masks = agent.create_action_masks(agent.action_dim, action_nums, self.device)
            next_action_masks = agent.create_action_masks(agent.action_dim, next_action_nums, self.device)
            probs = agent.actor(observations, sigs, action_masks)
            prob = probs.gather(1, actions)
            action_log_probs = torch.log(prob)
            actor_loss =  -(((1-self.beta1-self.beta2) * td_delta - self.beta1 * cross_prob -
                             self.beta2 * sig_cross_prob - self.beta2 * self_cross_prob) * action_log_probs).mean()
            critic_loss = torch.mean(F.mse_loss(agent.critic(global_observations), td_target.detach()))
            actor_loss.backward()
            critic_loss.backward()
            agent.actor_optimizer.step()
            agent.critic_optimizer.step()

        for agent_id, agent in enumerate(self.acs):
            target_ac = self.target_acs[agent_id]
            target_actor = target_ac.actor
            target_critic = target_ac.critic
            self.soft_update(agent.actor, target_actor)
            self.soft_update(agent.critic, target_critic)

def paths_list_record(trained_agents, tot_signal_num, tot_robot_num):
    agent = trained_agents
    path_list_list = []
    for sig_num in range(tot_signal_num):
        path_list = []
        transitions = agent.signal_repo[sig_num]
        dones = transitions['dones']
        trajec_length = len(transitions['observations'])
        for robot_id in range(tot_robot_num):
            trajectory = [transitions['cur_positions'][_][robot_id] for _ in range(trajec_length)]
            for po in range(len(dones)):
                if dones[po]:
                    trajectory = trajectory[:(po+1)]
                    break
            path_list.append(trajectory)
        path_list_list.append(path_list)
    return copy.copy(path_list_list)

def save_raw_data(raw_file_path, key, raw_data, raw_data_obj1 = None):
    new_component = pd.DataFrame({
        'Key': [key],
        'RawData': [raw_data],
        'RawDataObj1': [raw_data_obj1]
    })
    if os.path.exists(raw_file_path):
        existing_data = pd.read_csv(raw_file_path)
        if key in existing_data['Key'].values:
            # Key exists, replace the row
            existing_data = existing_data[existing_data['Key'] != key]  # Remove old row
            existing_data = pd.concat([existing_data, new_component], ignore_index=True)  # Add new row
        else:
            existing_data = pd.concat([existing_data, new_component], ignore_index=True)
        existing_data.to_csv(raw_file_path, index=False)

    else:
        new_component.to_csv(raw_file_path, mode='w', header=True, index=False)

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


def add_or_update_column_based_on_key(file_path, key, training_time):
    #log time
    training_time *= 10
    training_time = math.log10(training_time)
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    # Load existing data
    existing_data = pd.read_csv(file_path)

    # Check if the 'Training Time/log(s)' column exists, if not add it with default values
    if 'Training Time/log(s)' not in existing_data.columns:
        existing_data['Training Time/log(s)'] = None

    # Check if the key exists in the DataFrame
    if key in existing_data['Key'].values:
        # Key exists, update the column for the specific key
        existing_data.loc[existing_data['Key'] == key, 'Training Time/log(s)'] = training_time
    else:
        # Optionally, handle the case where the key does not exist
        print(f"Key '{key}' not found. No data updated.")

    # Save the updated data back to the CSV file
    existing_data.to_csv(file_path, index=False)


def save_policy_network(policy_net, net_file_path):
    torch.save(policy_net.state_dict(), net_file_path)
    print(f"Policy network saved to {net_file_path}")

if __name__ == "__main__":
    Algorithm = "SMC"
    signal_tot_num = 3
    robot_num = 4
    env_name = "OFFICE"

    actor_lr = 1e-4
    critic_lr = 1e-3
    beta1 = 0.1
    beta2 = 0.1
    print("{} test beta1 = {}, and beta2 = {}:".format(Algorithm, beta1, beta2))
    num_episodes = 40000
    hidden_dim = 16
    gamma = 0.95
    sample_size = 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env = gym(env_name, robot_num)
    env.seed = 0
    env.robot_num = robot_num
    torch.manual_seed(0)
    obs_dim = env.position_embed
    action_dim = env.action_dim
    return_list_set = []
    capture_time_list_set = []
    # replay_buffer = m_marl_utils.ReplayBuffer(buffer_size)
    # replay_buffer = m_marl_utils.ImportanceSamplingReplayBuffer(buffer_size)
    agents = None
    for p in range(1):
        agents = SMC(env_name, robot_num, signal_tot_num, obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, beta1, beta2,
                            gamma, device)
        return_list, capture_time_list = marl_utils.train_on_policy_mures1(env, agents, num_episodes, sample_size)
        return_list_set.append(copy.copy(return_list))
        capture_time_list_set.append(copy.copy(capture_time_list))

    script_dir = os.path.dirname(os.path.realpath(__file__))
    # script_dir = os.path.join(script_dir, f"Search_Env/Map_Info/{graph_name}.csv")
    net_file_p = os.path.join(script_dir, "Robot_Net/{}_{}T{}R{}.pth")
    # net_file_p = "/Users/pqh/PycharmProjects/SMC_Searcher/One_shot_MuRES/Robot_Data/Robot_net/{}_{}T{}R{}.pth"
    for agent_id, agent in enumerate(agents.acs):
        save_policy_network(agent.actor, net_file_p.format(env_name,Algorithm,robot_num,agent_id))

    # draw paths first:
    # paths_list = paths_list_record(agents, signal_tot_num, robot_num)
    # figure_name = "{} with beta1 = {}, beta2 = {}".format(Algorithm,beta1,beta2)
    # draw_graph_multi.draw_paths(env_name, paths_list, figure_name)
    # *******************
    # raw_file_path = '/Users/pqh/PycharmProjects/Prob-VDN-AC/Classical_Sim/Data_1/Raw/{}_R{}_Raw.csv'
    # raw_file_path = raw_file_path.format(env_name, robot_num)
    # save_raw_data(raw_file_path, Algorithm, compare_list_set, return_list_set)

    return_list = [sum(col) / len(col) for col in zip(*return_list_set)]
    capture_time_list = [sum(col) / len(col) for col in zip(*capture_time_list_set)]
    return_variance = [sum((xi - mu) ** 2 for xi in col) / len(col) for col, mu in
                       zip(zip(*return_list_set), return_list)]
    capture_time_variance = [sum((xi - mu) ** 2 for xi in col) / len(col) for col, mu in
                             zip(zip(*capture_time_list_set), capture_time_list)]
    episodes_list = list(range(len(return_list)))

    # Create the plot
    fig, ax1 = plt.subplots()
    ax1.plot(episodes_list, return_list, 'g-')  # 'g-' means green line
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Detection Probability')
    ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis
    ax2.plot(episodes_list, capture_time_list, 'b-')  # 'b-' means blue line
    ax2.set_ylabel('Capture Time')
    plt.title('{} on {} with R = {}'.format(Algorithm,env_name,robot_num))
    plt.show()

    mv_return = marl_utils.moving_average(return_list, 9)
    mv_capture_time = marl_utils.moving_average(capture_time_list, 9)
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
    add_component_to_csv(file_path, Algorithm, selected_mv_capture_time, selected_capture_time_variance, selected_mv_return, selected_return_variance)
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

# used to quick start;
# Note that we adopt "one-shot" adversarial adaptation scheme to ensure the fair comparison between
# MuRAS and MuRES solutions as state in the manuscript.
from MuRES.SMC import SMC
from MuRES.SMC_Sample import SMC_Exe
from Attack.Adv_Target import Adv_Target
import torch
import os
import numpy as np
from Search_Env.classic_env import classic_sim as gym
from Search_Env.escape_env import escape_sim as adv_gym
from Search_Utils import marl_utils, sample_utils, adv_utils
import copy

if __name__ == "__main__":
    # Common settings
    Algorithm = "SMC"
    signal_tot_num = 3
    robot_num = 2
    env_name = "MUSEUM"
    sample_num = 10000
    hidden_dim = 16
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Step 1: Train the policy network using SMC
    actor_lr = 1e-4
    critic_lr = 1e-3
    beta1 = 0.1
    beta2 = 0.1
    gamma = 0.95
    num_episodes = 40000
    sample_size = 8

    env = gym(env_name, robot_num)
    env.seed = 0
    env.robot_num = robot_num
    torch.manual_seed(0)
    obs_dim = env.position_embed
    action_dim = env.action_dim

    agents = SMC(env_name, robot_num, signal_tot_num, obs_dim, hidden_dim, action_dim, actor_lr, critic_lr, beta1, beta2, gamma, device)
    print("Training SMC Agents...")
    return_list, capture_time_list = marl_utils.train_on_policy_mures1(env, agents, num_episodes, sample_size)

    # Save the trained policy networks
    script_dir = os.path.dirname(os.path.realpath(__file__))
    net_file_path_base = os.path.join(script_dir, "../MuRES/Robot_Net/{}_{}T{}R{}.pth")
    for agent_id, agent in enumerate(agents.acs):
        torch.save(agent.actor.state_dict(), net_file_path_base.format(env_name, Algorithm, robot_num, agent_id))

    # Step 2: Sample trajectories using SMC_Sample
    print("Sampling Trajectories from Trained SMC Agents...")
    sample_agents = SMC_Exe(robot_num, signal_tot_num, obs_dim, hidden_dim, action_dim, device)
    sample_agents.policy_net_load(net_file_path_base)
    trajectories_file_path_base = os.path.join(script_dir, "../MuRES/Robot_Trajectory/{}_{}R{}.npy")
    trajectories_file_path = trajectories_file_path_base.format(env_name, Algorithm, robot_num)
    robots_trajectories_list = sample_utils.sample_from_trained_algorithm(env, sample_agents, sample_num)
    np.save(trajectories_file_path, np.array(robots_trajectories_list))

    # Step 3: Train adversarial target using Adv_Target
    adv_actor_lr = 1e-4
    adv_critic_lr = 1e-3
    adv_num_episodes = 20000

    print("Training Adversarial Target...")
    adv_env = adv_gym(env_name)
    adv_env.setup(Algorithm, robot_num)
    adv_agent = Adv_Target(obs_dim, hidden_dim, action_dim, adv_actor_lr, adv_critic_lr, gamma, device)
    adv_return_list, adv_capture_time_list = adv_utils.train_on_policy_adv2(adv_env, adv_agent, adv_num_episodes, sample_size)

    # Save the adversarial target policy network
    adv_net_file_path_base = os.path.join(script_dir, "../Attack/Target_Net/{}_{}R{}.pth")
    torch.save(adv_agent.ac.actor.state_dict(), adv_net_file_path_base.format(env_name, Algorithm, robot_num))

    print("Demo completed.")

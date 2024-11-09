
from Search_Utils.Embedding import EmbeddingLayer
import random
import torch
import copy
import numpy as np
from Search_Env.classic_map import Map
class escape_sim:
    def __init__(self, env_name):
        self.env_name = env_name
        self.map = Map(env_name)

        self.robot_trajectories_num = 1
        self.explore_base = 0.1 / self.robot_trajectories_num

        self.total_position = self.map.map_position_num
        self.position_embed = 4
        self.action_dim = self.map.map_action_num
        self.seed = 0
        self.embedding_layer = EmbeddingLayer(self.total_position + 1, self.position_embed, 0)

        # self.start_position = 5 if env_name == "MUSEUM" else 16 if env_name == "OFFICE" else 46 if env_name == "Grid_10" \
        #     else 171 if env_name == "Grid_20" else 190 if env_name == "Grid_R" else None
        self.target_start_positions = [31, 45, 55, 65, 69] if env_name == "MUSEUM" \
            else [1, 14, 30, 43, 52] if env_name == "OFFICE" \
            else [4, 8, 28, 56, 78, 80, 94] if env_name == "Grid_10" \
            else [7, 11, 47, 97, 175, 177, 339] if env_name == "Grid_20" \
            else [12, 83, 97, 243, 257] if env_name == "Grid_R" else None
        self.target_start_probability = [0.2, 0.2, 0.2, 0.2, 0.2] if env_name == "MUSEUM" \
            else [0.2,0.2,0.2,0.2,0.2] if env_name == "OFFICE" \
            else [0.05, 0.15, 0.05, 0.05, 0.05, 0.3, 0.35] if env_name == "Grid_10" \
            else [0.3, 0.05, 0.1, 0.1, 0.05, 0.1, 0.3] if env_name == "Grid_20" \
            else [0.4, 0.25, 0.15, 0.1, 0.1] if env_name == "Grid_R" else None
        self.done = False
        self.robot_trajectories_list = None
        self.robot_trajectory_file_path = None
        self.robot_trajectories_repo = None
        self.capture_flag_list = None
        self.target_trajectory = None
        self.observation = None
        self.reward = None
        self.explore_reward = None
        self.action_num = None
        self.next_position_list = None
        self.remain_time_init = min(int(self.total_position / 2), 25) -1
        self.remain_time = None

        self.time_step = 0
        self.direct_from_list = True
        self.sensor_range = 1
        self.target_back_length = 5
        self.trajectory_clip_length = 10
        self.info_dict = {}

    def _prepare_robots_repo(self):
        print("load robot trajectory file")
        loaded_array = np.load(self.robot_trajectory_file_path)
        self.robot_trajectories_repo = loaded_array.tolist()
        print("end load robot trajectory file")

    def setup(self, Algorithm, robot_num):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        script_dir = os.path.dirname(script_dir)
        self.robot_trajectory_file_path = os.path.join(script_dir, f"MuRES/Robot_Trajectory/{self.env_name}_{Algorithm}R{robot_num}.npy")
        self._prepare_robots_repo()

    def _sample_robots_trajectories(self):
        self.robot_trajectories_list = random.choices(self.robot_trajectories_repo, k = self.robot_trajectories_num)
        # for i in range(len(self.robot_trajectories_list)):
        #     print("robot_trajectory {}:".format(i), self.robot_trajectories_list[i])

    def reset(self):
        self.explore_base = 0.1 / self.robot_trajectories_num
        self.remain_time = self.remain_time_init
        self.done = False
        self._sample_robots_trajectories()
        self.time_step = 0
        self.target_trajectory = random.choices(self.target_start_positions, self.target_start_probability)
        self._trajectory_to_observation()
        self.reward = 0
        self.capture_flag_list = [False for _ in range(self.robot_trajectories_num)]
        self.explore_reward = self.explore_base * (len(self.capture_flag_list) - sum(self.capture_flag_list))
        self._actionNum()
        self._info_dict_prepare()
        return self.info_dict

    def step(self, action):
        next_position = self.next_position_list[action]
        self.target_trajectory.append(next_position)
        self._trajectory_to_observation()
        self._actionNum()
        self._judge_capture()
        self.remain_time -= 1
        if self.remain_time == 0:
            self.done = True
        self._info_dict_prepare()
        return self.info_dict

    def _trajectory_to_observation(self):
        clipped_trajectory = copy.copy(self.target_trajectory[-self.trajectory_clip_length:])
        self.observation = self.embedding_layer(torch.tensor(clipped_trajectory))

    def _judge_capture(self):
        t_position = self.target_trajectory[-1]
        t_p_position = self.target_trajectory[-2]
        time_step = len(self.target_trajectory)
        reward = 0.
        for i in range(len(self.capture_flag_list)):
            if self.capture_flag_list[i]:
                continue
            robot_trajectories = self.robot_trajectories_list[i]
            for robot_id in range(len(robot_trajectories)):
                robot_trajectory = robot_trajectories[robot_id]
                r_position = robot_trajectory[time_step]
                r_p_position = robot_trajectory[time_step-1]
                if t_position == r_position or (t_position == r_p_position and t_p_position == r_position):
                    reward -= 1./self.robot_trajectories_num
                    self.capture_flag_list[i] = True
                    break
        self.explore_reward = 0
        if t_position not in self.target_trajectory[:-1]:
            self.explore_reward = self.explore_base * (len(self.capture_flag_list) - sum(self.capture_flag_list))
        self.reward = reward

    def _actionNum(self):
        # need trajectory to judge how many position
        position = self.target_trajectory[-1]
        self.next_position_list = self.map.next_total_position_list(position)
        for i in range(len(self.target_trajectory)-1,max(-1,len(self.target_trajectory)-self.target_back_length),-1):
            if len(self.next_position_list) == 1:
                break
            if self.target_trajectory[i] in self.next_position_list:
                self.next_position_list.remove(self.target_trajectory[i])
        self.action_num = len(self.next_position_list)

    def _info_dict_prepare(self):
        self.info_dict["target_trajectory"] = copy.deepcopy(self.target_trajectory)
        self.info_dict["observation"] = copy.deepcopy(self.observation)
        self.info_dict["reward"] = self.reward
        self.info_dict["action_num"] = self.action_num
        self.info_dict["done"] = self.done
        self.info_dict["explore_reward"] = self.explore_reward

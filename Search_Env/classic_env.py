# have different thoughts about target, target still random walk, however, it will not steady and not come back
from Search_Utils.Embedding import EmbeddingLayer
import random
import torch
import copy
from Search_Env.classic_map import Map
class classic_sim:
    def __init__(self, env_name, robot_num):
        self.env_name = env_name
        self.robot_num = robot_num
        self.map = Map(env_name)
        self.target_num = 1000
        self.target_repo_num = 100 * self.target_num
        # self.target_repo_num = self.target_num
        self.total_position = self.map.map_position_num
        self.max_time = min(self.total_position,100)
        self.position_embed = 4
        # self.time_embed = 8
        self.action_dim = self.map.map_action_num
        self.seed = 0
        self.embedding_layer = EmbeddingLayer(self.total_position + 1, self.position_embed, 0)
        self.start_position = 5 if env_name == "MUSEUM" else 49 if env_name == "OFFICE" else 46 if env_name == "Grid_10" \
            else 171 if env_name == "Grid_20" else 190 if env_name == "Grid_R" else None
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
        self.robot_trajectories = None
        self.observations = None
        self.done = False
        self.team_reward = None
        self.robot_reward = None
        self.action_nums = None
        self.remain_time_init = min(int(self.total_position/2),25)
        self.remain_time = self.remain_time_init
        self.target_position_list = None
        self.target_last_position_list = None
        self.capture_flag_list = None
        self.target_trajectories_repo = None
        self.sampled_target_trajectories = None
        self.robot_positions = None
        self.next_all_positions = None
        self.info_dict = {}
        self.target_back_length = 5
        self._prepare_target_repo()
        self.sensor_range = 1
        self.trajectory_clip_length = 10

    def _prepare_target_repo(self):
        print("start preparation of target trajectories sample buffer")
        if self.target_start_probability == None:
            self.target_position_list = random.choices(self.target_start_positions, k=self.target_repo_num)
        else:
            self.target_position_list = random.choices(self.target_start_positions,
                                                       weights=self.target_start_probability,
                                                       k=self.target_repo_num)
        self.target_trajectories_repo = [[ele] for ele in self.target_position_list]
        self._target_ramdom_move_prepare()
        print("target trajectories finish")

    def _target_ramdom_move_prepare(self):
        counter = self.remain_time
        while counter >= 0:
            buffer_list = copy.copy(self.target_position_list)
            for i in range(len(self.target_position_list)):
                target_cur_position = buffer_list[i]
                # this is the old version of random walk
                # action_num = self._actionNum_position(target_cur_position)
                # target_action = random.randint(0, action_num)
                # target_next_position = self.map.step(target_cur_position, target_action)
                # this is new version: random walk + exploration:
                # target_action_list = [p for p in range(action_num)]
                target_possible_position_list = self.map.next_total_position_list(target_cur_position)
                target_possible_position_list = target_possible_position_list[: -1]
                target_next_possible_sample_list = copy.copy(target_possible_position_list)
                cur_target_trajectory = self.target_trajectories_repo[i]
                for j in range(len(cur_target_trajectory)-1, max(-1,len(cur_target_trajectory)-self.target_back_length),-1):
                    if len(target_next_possible_sample_list) == 1:
                        break
                    elif cur_target_trajectory[j] in target_next_possible_sample_list:
                        target_next_possible_sample_list.remove(cur_target_trajectory[j])
                target_next_position = random.choice(target_next_possible_sample_list)
                # ************************************************
                buffer_list[i] = target_next_position
            self.target_position_list = buffer_list
            for i in range(len(self.target_position_list)):
                self.target_trajectories_repo[i].append(self.target_position_list[i])
            counter -= 1

    def _sample_target_trajectories(self):
        self.sampled_target_trajectories = random.choices(self.target_trajectories_repo, k = self.target_num)
        self.target_position_list = []
        self.target_last_position_list = []
        for i in range(len(self.sampled_target_trajectories)):
            self.target_position_list.append(self.sampled_target_trajectories[i][0])
            self.target_last_position_list.append(self.sampled_target_trajectories[i][0])

    def reset(self):
        self.robot_trajectories = [[self.start_position] for _ in range(self.robot_num)]
        self.action_nums = [self._actionNum_position(self.start_position) for _ in range(self.robot_num)]
        self.robot_positions = [self.start_position for _ in range(self.robot_num)]
        self.next_all_positions = [self._next_all_positions(self.start_position) for _ in range(self.robot_num)]
        self._trajectories_to_observations()
        self.done = False
        self.capture_flag_list = False
        self.team_reward = 0.
        self.robot_reward = [0. for _ in range(self.robot_num)]
        # self.remain_time = random.randint(1, self.max_time)
        self.remain_time = self.remain_time_init
        self.info_dict['action_nums'] = self.action_nums
        # self.info_dict['remain_time'] = self.time_embedding(torch.tensor(self.remain_time))
        self.info_dict['current_positions'] = self.robot_positions
        self.info_dict['next_all_positions'] = self.next_all_positions
        # self.target_position_list = random.choices(self.target_start_positions, k=self.target_num)
        self._sample_target_trajectories()

        self.capture_flag_list = [False for _ in range(self.target_num)]
        return copy.deepcopy(self.observations), copy.deepcopy(self.info_dict)

    def step(self, actions):
        for robot_label in range(self.robot_num):
            action = actions[robot_label]
            self._take_action(robot_label,action)
        self.remain_time -= 1
        self.info_dict['current_positions'] = self.robot_positions
        self.info_dict['next_all_positions'] = self.next_all_positions
        # self.info_dict['remain_time'] = self.time_embedding(torch.tensor(self.remain_time))
        self.team_reward = 0.
        self.robot_reward = [0. for _ in range(self.robot_num)]
        self._target_ramdom_move()
        for robot_label in range(self.robot_num):
            self._judge_capture(robot_label)
        self.info_dict['action_nums'] = self.action_nums
        self._trajectories_to_observations()
        return copy.deepcopy(self.observations), self.robot_reward, self.done, copy.deepcopy(self.info_dict)

    def _trajectories_to_observations(self):
        self.observations = []
        for robot_label in range(self.robot_num):
            observation_embed = self._trajectory_to_observation(robot_label)
            self.observations.append(observation_embed)

    def _trajectory_to_observation(self, robot_label):
        clipped_trajectory = copy.copy(self.robot_trajectories[robot_label][-self.trajectory_clip_length:])
        return self.embedding_layer(torch.tensor(clipped_trajectory))

    def _judge_capture(self,robot_label):
        robot_cur_position = self.robot_trajectories[robot_label][-1]
        for h in range(self.target_num):
            if self.sensor_range == 1:
                capture_flag = self.capture_flag_list[h]
                if not capture_flag:
                    target_position = self.target_position_list[h]
                    #node capture
                    if target_position == robot_cur_position:
                        self.team_reward += (1./self.target_num)
                        self.robot_reward[robot_label] += (1./self.target_num)
                        self.capture_flag_list[h] = True
                    #edge capture
                capture_flag = self.capture_flag_list[h]
                if not capture_flag and len(self.robot_trajectories[robot_label]) >= 2:
                    target_cur_position = self.target_position_list[h]
                    target_last_position = self.target_last_position_list[h]
                    robot_last_position = self.robot_trajectories[robot_label][-2]
                    if target_cur_position == robot_last_position and target_last_position == robot_cur_position:
                        self.team_reward += (1. / self.target_num)
                        self.robot_reward[robot_label] += (1./ self.target_num)
                        self.capture_flag_list[h] = True
            else:
                capture_flag = self.capture_flag_list[h]
                if not capture_flag:
                    target_position = self.target_position_list[h]
                    robot_sensor_range = self.map.search_range(robot_cur_position,self.sensor_range)
                    #node capture
                    if target_position in robot_sensor_range:
                        # self.capture_one_step += 1
                        self.team_reward += (1./self.target_num)
                        self.robot_reward[robot_label] += (1. / self.target_num)
                        self.capture_flag_list[h] = True
            #***********
        if self.done == False and self.remain_time == 0 and robot_label == self.robot_num - 1:
            self.done = True

    def _take_action(self, robot_label, action):
        robot_cur_position = self.robot_trajectories[robot_label][-1]
        real_action = self._action_to_real_action(robot_label, action)
        robot_next_position = self.map.step(robot_cur_position, real_action)
        self.robot_trajectories[robot_label].append(robot_next_position)
        self.robot_positions[robot_label] = robot_next_position
        self.action_nums[robot_label] = self._actionNum_position(robot_next_position)

        all_next_positions = self._next_all_positions(robot_next_position)

        # eliminate the previous position:
        if self.action_nums[robot_label] > 1:
            self.action_nums[robot_label] -= 1
            all_next_positions.remove(robot_cur_position)
            self.next_all_positions[robot_label] = all_next_positions
        # **********************************************

    def _actionNum_position(self, position):
        action_num = self.map.next_total_action(position)
        # can not stay at same position:
        action_num -= 1
        # **********************************************
        return action_num

    def _next_all_positions(self, position):
        next_position_list = self.map.next_total_position_list(position)
        next_position_list = next_position_list[:-1]
        return copy.copy(next_position_list)

    def _action_to_real_action(self, robot_label, action):
        real_action = action
        robot_cur_position = self.robot_trajectories[robot_label][-1]
        if self._actionNum_position(robot_cur_position) == 1:
            return real_action
        elif len(self.robot_trajectories[robot_label]) >= 2:
            robot_pre_position = self.robot_trajectories[robot_label][-2]
            robot_next_position = self.map.step(robot_cur_position, action)
            if robot_next_position >= robot_pre_position:
                real_action += 1
        return real_action

    def _target_ramdom_move(self):
        time_step = len(self.robot_trajectories[0])
        self.target_last_position_list = copy.copy(self.target_position_list)
        buffer_list = copy.copy(self.target_position_list)
        for i in range(len(self.capture_flag_list)):
            if not self.capture_flag_list[i]:
                target_next_position = self.sampled_target_trajectories[i][time_step]
                buffer_list[i] = target_next_position
        self.target_position_list = buffer_list
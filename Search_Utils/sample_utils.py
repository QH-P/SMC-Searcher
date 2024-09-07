import copy
from tqdm import tqdm

def sample_from_trained_algorithm(env, agents, sample_num):
    robots_trajectories_list = []
    for i in range(10):
        with tqdm(total=int(sample_num / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(sample_num / 10)):
                if getattr(agents, "alg_name", None) == "SMC":
                    __ = agents.signal_random_generate()
                    # print("indeed sample")
                observations, info = env.reset()
                action_nums = info['action_nums']
                done = False
                while not done:
                    actions = agents.take_actions(observations, action_nums)
                    next_observations, reward, done, info = env.step(actions)
                    observations = next_observations
                    action_nums = info['action_nums']
                # ***********************************************
                robots_trajectories_list.append(copy.deepcopy(env.robot_trajectories))
                # if getattr(agents, "alg_name", None) == "SMC":
                #     agents.trajectory_dict[agents.current_signal_num] = env.robot_trajectories

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (sample_num / 10 * i + i_episode + 1)})
                pbar.update(1)
                # if i_episode % (int(sample_num / 1000)) == 0 and i_episode != 0:
                #     print("trajectories:",env.robot_trajectories)
    return robots_trajectories_list
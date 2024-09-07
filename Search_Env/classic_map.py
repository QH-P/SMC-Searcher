
import pandas as pd
import os
import copy
class Map:
    def __init__(self, map_name):
        self.map_name = map_name
        script_dir = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(script_dir, f"Map_Info/{self.map_name}.csv")
        self.map = pd.read_csv(file_path)
        self.map_position_num = self._max_room_number_df()
        self.map_action_num = self._max_connected_rooms_df()
        self.degree_of_one_core_graph = self._degree_of_one_core_graph()
        self.obstacles = self._get_obstacle()
        self.grid_size = self.map_position_num

    def step(self, position, action):
        next_possible_position = self._find_connected_rooms(position)
        # print("position:",position,"action:",action,"next:",next_possible_position)
        action %= len(next_possible_position)
        return next_possible_position[action]

    def step_list(self, position_list, action_list):
        next_position_list = []
        for position,action in zip(position_list,action_list):
            next_position_list.append(self.step(position,action))
        return next_position_list

    def next_total_action(self, position):
        next_possible_position = self._find_connected_rooms(position)
        return len(next_possible_position)

    def next_total_position_list(self, position):
        next_possible_position = self._find_connected_rooms(position)
        return copy.copy(next_possible_position)

    def _find_connected_rooms(self, room):
        connected_room = self.map[self.map["Room"] == room]["Connected_Room"].tolist()
        connected_room.append(room)
        return connected_room

    def _max_connected_rooms_df(self):
        connected_max = self.map.groupby("Room").size().max()
        return connected_max + 1

    def _max_room_number_df(self):
        return max(self.map["Room"].max(), self.map["Connected_Room"].max())

    def _degree_of_one_core_graph(self):
        # Create a copy of the DataFrame
        map_copy = self.map.copy()

        # Iterate until no changes in the DataFrame
        while True:
            # Calculate the degree of each node
            degree_dict = map_copy.groupby('Room').size().to_dict()
            map_copy['Connected_Room_Degree'] = map_copy['Connected_Room'].map(degree_dict)

            # Remove nodes with degree 1
            one_degree_nodes = map_copy[map_copy['Connected_Room_Degree'] == 1]['Connected_Room'].unique()
            map_copy = map_copy[~map_copy['Room'].isin(one_degree_nodes)]
            map_copy = map_copy[~map_copy['Connected_Room'].isin(one_degree_nodes)]

            # Recalculate the degree of each node
            new_degree_dict = map_copy.groupby('Room').size().to_dict()

            # Break the loop if the DataFrame does not change
            if degree_dict == new_degree_dict:
                break

        # Return the degree of the 1-core graph
        return new_degree_dict

    def _get_obstacle(self):
        # Get unique nodes from both 'Room' and 'Connected_Room' columns
        connected_nodes = set(self.map['Room'].unique()).union(set(self.map['Connected_Room'].unique()))

        # Generate the full set of nodes based on the total number of nodes expected
        all_nodes = set(range(1,  self.map_position_num + 1))

        # Find obstacles by determining which nodes are not in the connected nodes set
        obstacles = all_nodes - connected_nodes
        return list(obstacles)

    def search_range(self, node, square_size = 1):
        obstacles = self.obstacles
        grid_size = self.map_position_num
        neighbors = []
        row_size = int(grid_size ** 0.5)  # Calculate grid dimensions (assuming square grid)
        row = (node - 1) // row_size
        col = (node - 1) % row_size
        half_square = square_size // 2
        # Iterate over the square area around the node to find neighbors
        for i in range(-half_square, half_square + 1):
            for j in range(-half_square, half_square + 1):
                new_row = row + i
                new_col = col + j
                # Check if within grid bounds
                if 0 <= new_row < row_size and 0 <= new_col < row_size:
                    neighbor = new_row * row_size + new_col + 1
                    # Exclude the node itself and any obstacles
                    if neighbor in obstacles:
                        continue
                    # Check for partial occlusion
                    if not self.is_occluded(row, col, new_row, new_col, row_size, obstacles):
                        neighbors.append(neighbor)
        return neighbors

    def is_occluded(self, start_row, start_col, target_row, target_col, row_size, obstacles):
        # Determine step increments
        row_diff = target_row - start_row
        col_diff = target_col - start_col
        num_steps = max(abs(row_diff), abs(col_diff))
        # Check each step for obstacles
        for step in range(1, num_steps + 1):
            intermediate_row = start_row + (step * row_diff) // num_steps
            intermediate_col = start_col + (step * col_diff) // num_steps
            # Calculate node index
            intermediate_node = intermediate_row * row_size + intermediate_col + 1
            if intermediate_node in obstacles:
                return True  # Path is blocked by an obstacle
        return False
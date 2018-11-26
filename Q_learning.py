import numpy as np
import random
from maze_utilities import MazeState, MazePlayer, MazeBeast


class QLearning:
    def __init__(self, simulation_environment, number_episodes, p, lmb, player_possible_actions, police_possible_actions):
        self.player_possible_actions = player_possible_actions
        self.police_possible_actions = police_possible_actions
        self.simulation_environment = simulation_environment
        self.number_episodes = number_episodes
        self.memory = {'state_agent': [], 'state_police': [], 'reward': [], 'action': []}
        self. T_list = np.random.geometric(p=p, size=number_episodes)
        self.lmb = lmb
        self.q_function = None

    def generate_random_policy(self, player):
        for x_p in range(self.simulation_environment.maze.length):
            for y_p in range(self.simulation_environment.maze.height):
                for x_b in range(self.simulation_environment.maze.length):
                    for y_b in range(self.simulation_environment.maze.height):
                        player_state = MazeState(x_p, y_p)
                        feasible_actions = []
                        for action in player.possible_actions:
                            if self.simulation_environment.maze.check_player_action(player_state, action):
                                feasible_actions.append(action)
                        player.policies[(x_p, y_p, x_b, y_b)] = random.choice(feasible_actions)

    @staticmethod
    def reward_function(self, player_state, police_state, goal_state):
        if player_state.x == police_state.x and player_state.y == police_state.y:
            return -10

        if player_state.x == goal_state.x and player_state.y == goal_state.y:
            return 1

        return 0

    def simulate_episode(self, length_episode):
        # Initialize states
        player = MazePlayer(0, 0, self.player_possible_actions)
        police = MazeBeast(3, 3, self.police_possible_actions)
        self.simulation_environment.player = player
        self.simulation_environment.police = police

        # Initialize simulation
        self.simulation_environment.step = 0
        self.generate_random_policy(player)

        # Initialize the memory for the episode
        memory = {'state_agent': [], 'state_police': [], 'reward': [], 'action': []}
        memory['state_agent'].append(player.state)
        memory['state_police'].append(police.state)
        memory['reward'].append(self.reward_function(player_state=player.state, police_state=police.state,
                                                          goal_state=self.simulation_environment.maze.goal_state))

        # Simulate for length episode steps
        for i in range(length_episode):
            self.simulation_environment.step_simulation()
            memory = {'state_agent': [], 'state_police': [], 'reward': [], 'action': []}
            memory['state_agent'].append(player.state)
            memory['state_police'].append(police.state)
            memory['reward'].append(self.reward_function(player_state=player.state, police_state=police.state,
                                                              goal_state=self.simulation_environment.maze.goal_state))

        return memory

    def init_q_function(self):
        seq = []
        for x_p in range(self.simulation_environment.maze.length):
            for y_p in range(self.simulation_environment.maze.height):
                for x_b in range(self.simulation_environment.maze.length):
                    for y_b in range(self.simulation_environment.maze.height):
                        for action in self.player_possible_actions:
                            seq.append((x_p, y_p, x_b, y_b, action))
        # Store in the dictionary, the value of the Q function and the number of updates
        self.q_function = dict.fromkeys(seq, (0, 0))

    def q_learning_experience(self):
        self.init_q_function()
        for i in range(self.number_episodes):
            memory = self.simulate_episode(self.T_list[i])
            self.update_q_function(memory)

    def update_q_function(self, memory):
        pass


    def step_size(self, n):
        return n


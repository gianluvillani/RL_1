import numpy as np
import random
import copy
from maze_utilities import MazeState, MazePlayer, MazeBeast, Maze, QMazeSimulation
import matplotlib.pyplot as plt
from tqdm import tqdm
DEBUG = False


def log(s):
    if DEBUG:
        print(s)

class QLearning:
    def __init__(self, simulation_environment, num_iterations, lmb, player_possible_actions, police_possible_actions):
        self.player_possible_actions = player_possible_actions
        self.police_possible_actions = police_possible_actions
        self.simulation_environment = simulation_environment
        self.lmb = lmb
        self.q_function = None
        self.num_iterations = num_iterations
        self.q_initial_state_history = []
        self.init_q_function()

    @staticmethod
    def reward_function(player_state, police_state, goal_state):
        if player_state.x == police_state.x and player_state.y == police_state.y:
            return -10

        if player_state.x == goal_state.x and player_state.y == goal_state.y:
            return 1

        return 0

    def simulate_and_learn(self):
        # Initialize states
        # Notice that the player is also a beast since it can choose randomly. Needed for exploration purposes
        player_init_state = MazeState(0, 0)
        player = MazeBeast(init_state=player_init_state, maze = self.simulation_environment.maze,
                           possible_actions=self.player_possible_actions)
        police_init_state = MazeState(3, 3)
        police = MazeBeast(init_state=police_init_state, maze = self.simulation_environment.maze,
                           possible_actions=self.player_possible_actions)

        # Plug the player inside the simulation environment
        self.simulation_environment.player = player
        self.simulation_environment.beast = police

        # Initialize simulation
        self.simulation_environment.step = 0
        # Initialize the memory for the simulation_step
        memory = {'state_agent_0': None, 'state_police_0': None, 'reward': None, 'action': None, 'state_agent_1': None,
                  'state_police_1': None}

        for i in tqdm(range(self.num_iterations)):
            log('')
            log('Learning step: ' + str(i))
            log('     Player state: ' + str(player.state.x) + ', ' + str(player.state.y))
            log('     Police state: ' + str(police.state.x) + ', ' + str(police.state.y))
            memory['state_agent_0'] = copy.deepcopy(player.state)
            memory['state_police_0'] = copy.deepcopy(police.state)
            memory['reward'] = self.reward_function(player_state=player.state, police_state=police.state,
                                                              goal_state=self.simulation_environment.maze.goal_state)
            log('            Reward collected: ' + str(memory['reward']))
            p_action, _ = self.simulation_environment.step_simulation()
            memory['state_agent_1'] = copy.deepcopy(player.state)
            memory['state_police_1'] = copy.deepcopy(police.state)
            memory['action'] = p_action
            log('            Action chosen: ' + str(memory['action']))
            q_initial_state_t = self.update_q_function(memory)
            self.q_initial_state_history.append(q_initial_state_t)

    def init_q_function(self):
        seq = []
        for x_p in range(self.simulation_environment.maze.length):
            for y_p in range(self.simulation_environment.maze.height):
                for x_b in range(self.simulation_environment.maze.length):
                    for y_b in range(self.simulation_environment.maze.height):
                        for action in self.player_possible_actions:
                            seq.append((x_p, y_p, x_b, y_b, action))
        # Store in the dictionary, the value of the Q function and the number of updates
        self.q_function = dict.fromkeys(seq)

        for x_p in range(self.simulation_environment.maze.length):
            for y_p in range(self.simulation_environment.maze.height):
                for x_b in range(self.simulation_environment.maze.length):
                    for y_b in range(self.simulation_environment.maze.height):
                        for action in self.player_possible_actions:
                            self.q_function[(x_p, y_p, x_b, y_b, action)] = [0.1, 0]


    def update_q_function(self, memory):
        state_player = memory['state_agent_0']
        state_police = memory['state_police_0']
        reward = memory['reward']
        next_state_player = memory['state_agent_1']
        next_state_police = memory['state_police_1']
        action = memory['action']
        current_state_action = (state_player.x, state_player.y, state_police.x, state_police.y, action)

        log('    Updating Q in: ' + '(' + str(state_player.x) + str(', ') + str(state_player.y) + ')')
        log('    Possible Q to be picked, given future state: ' +
            '(' + str(next_state_player.x) + str(', ') + str(next_state_player.y) + ')')
        possible_future_q = []

        for p_action in self.player_possible_actions:
            if self.simulation_environment.maze.check_player_action(action=p_action, state=state_player):
                future_state = (next_state_player.x, next_state_player.y, next_state_police.x, next_state_police.y, p_action)
                log('        '+str(self.q_function[future_state][0]))
                possible_future_q.append(self.q_function[future_state][0])

        # update n_t
        self.q_function[current_state_action][1] = self.q_function[current_state_action][1] + 1
        n = self.q_function[current_state_action][1]

        # update q(s, a)
        self.q_function[current_state_action][0] = \
            self.q_function[current_state_action][0] + \
            self.step_size(n) * (reward + self.lmb * max(possible_future_q) - self.q_function[current_state_action][0])

        q_initial_state = []
        for p_action in ['stay', 'right', 'down']:
            q_initial_state.append(self.q_function[(0, 0, 3, 3, p_action)][0])

        return max(q_initial_state)


    @staticmethod
    def step_size(n):
        return 1/(n**(3/3))


def main():
    bank_location = MazeState(1, 1)
    bank_maze = Maze(4, 4, goal_state=bank_location)
    player_actions = ['up', 'down', 'left', 'right', 'stay']
    police_actions = ['up', 'down', 'left', 'right']

    simulation_environment = QMazeSimulation(maze=bank_maze)
    num_iterations = 1000000
    lmb = 0.8

    q_learning_experience = QLearning(simulation_environment=simulation_environment,
                                      num_iterations=num_iterations,
                                      lmb=lmb,
                                      player_possible_actions=player_actions,
                                      police_possible_actions=police_actions)
    q_learning_experience.simulate_and_learn()
    q_function = q_learning_experience.q_function
    print(q_learning_experience.q_initial_state_history[-10:])
    plt.plot(q_learning_experience.q_initial_state_history)
    plt.show()

if __name__ == '__main__':
    main()


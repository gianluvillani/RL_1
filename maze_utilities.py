import numpy as np
import copy
from abc import ABCMeta, abstractmethod
import random

def cprint(cstring):
    cstring = '{0: <3}'.format(cstring)
    print(cstring.format(), end="", flush=True)


class MazeAgent:
    __metaclass__ = ABCMeta

    def __init__(self, init_state, maze, possible_actions):
        self.state = init_state
        self.maze = maze
        self.possible_actions = possible_actions

    def update_state(self, action):
        self.state.update(action)

    @abstractmethod
    def choose_action(self):
        pass


class MazePlayer(MazeAgent):
    def choose_action(self, beast_state, time_step, action):
        x_p = self.state.x
        y_p = self.state.y
        x_b = beast_state.x
        y_b = beast_state.y
        if action is not None:
            policy = self.policies[(x_p, y_p, x_b, y_b)]
            return policy[-(time_step+1)][0]
        else:
            return action

    def collect_reward(self):
        pass

    def update_policy(self):
        pass

    def init_policies(self):
        # Build all the possible states
        seq = []
        for x_p in range(self.maze.length):
            for y_p in range(self.maze.height):
                for x_b in range(self.maze.length):
                    for y_b in range(self.maze.height):
                        seq.append((x_p, y_p, x_b, y_b))
        self.policies = dict.fromkeys(seq)
        self.maze_dim = self.maze.length * self.maze.height


class MazeBeast(MazeAgent):
    def choose_action(self):
        return random.choice(self.possible_actions)


class Maze:
    """
    (0,0) top left ->
    """
    def __init__(self, length, height, goal_state):
        self.length = length
        self.height = height
        self.vert_walls = []
        self.horiz_walls = []
        self.outer_horiz_walls = []
        self.outer_vert_walls = []

        self.add_vert_wall(-1, 0, height-1, outer=True)
        self.add_vert_wall(length-1, 0, height-1, outer=True)
        self.add_horiz_wall(-1, 0, length-1, outer=True)
        self.add_horiz_wall(height-1, 0, length-1, outer=True)

        self.goal_state = goal_state

    def add_vert_wall(self, x, y_min, y_max, outer=False):
        self.vert_walls.append({'x': x, 'y_min': y_min, 'y_max': y_max})
        if outer:
            self.outer_vert_walls.append({'x': x, 'y_min': y_min, 'y_max': y_max})

    def add_horiz_wall(self, y, x_min, x_max, outer = False):
        self.horiz_walls.append({'y': y, 'x_min': x_min, 'x_max': x_max})
        if outer:
            self.outer_horiz_walls.append({'y': y, 'x_min': x_min, 'x_max': x_max})

    def check_player_action(self, action, state):
        if action == 'stay':
            return True

        if action == 'down':
            for h_wall in self.horiz_walls:
                if h_wall['x_min'] <= state.x <= h_wall['x_max'] and state.y == h_wall['y']:
                    return False
            return True

        if action == 'up':
            for h_wall in self.horiz_walls:
                if h_wall['x_min'] <= state.x <= h_wall['x_max'] and state.y == h_wall['y']+1:
                    return False
            return True

        if action == 'right':
            for v_wall in self.vert_walls:
                if v_wall['y_min'] <= state.y <= v_wall['y_max'] and state.x == v_wall['x']:
                    return False
            return True

        if action == 'left':
            for v_wall in self.vert_walls:
                if v_wall['y_min'] <= state.y <= v_wall['y_max'] and state.x == v_wall['x']+1:
                    return False
            return True

    def check_beast_action(self, action, state):
        if action == 'stay':
            return True

        if action == 'down':
            for h_wall in self.outer_horiz_walls:
                if h_wall['x_min'] <= state.x <= h_wall['x_max'] and state.y == h_wall['y']:
                    return False
            return True

        if action == 'up':
            for h_wall in self.outer_horiz_walls:
                if h_wall['x_min'] <= state.x <= h_wall['x_max'] and state.y == h_wall['y'] + 1:
                    return False
            return True

        if action == 'right':
            for v_wall in self.outer_vert_walls:
                if v_wall['y_min'] <= state.y <= v_wall['y_max'] and state.x == v_wall['x']:
                    return False
            return True

        if action == 'left':
            for v_wall in self.outer_vert_walls:
                if v_wall['y_min'] <= state.y <= v_wall['y_max'] and state.x == v_wall['x'] + 1:
                    return False
            return True

    def compute_feasible_player_future_states(self, allowed_actions, player_state):
        # TODO: adjust in the init
        feasible_actions = []
        for action in allowed_actions:
            if self.check_player_action(action, player_state):
                feasible_actions.append(action)
        return feasible_actions

    def compute_feasible_beast_future_states(self, allowed_actions, beast_state):
        # TODO: adjust in the init
        feasible_actions = []
        for action in allowed_actions:
            if self.check_beast_action(action, beast_state):
                feasible_actions.append(action)
        return feasible_actions


class MazeState:
    def __init__(self, x_0, y_0):
        self.x = x_0
        self.y = y_0

    def update(self, action):
        if action == 'stay':
            pass

        if action == 'down':
            self.y += 1

        if action == 'up':
            self.y -= 1

        if action == 'right':
            self.x += 1

        if action == 'left':
            self.x -= 1


class QMazeSimulation():
    def __init__(self,  maze, player=None, beast=None):
        self.player = player
        self.beast = beast
        self.maze = maze
        self.simulation_time = 0

    def get_player_action(self, action=None):
        if action is None:
            action = self.player.choose_action()
        return action

    def get_beast_action(self, action=None):
        if action is None:
            action = self.beast.choose_action()
        return action

    def step_simulation(self):
        assert self.player is not None,  "You have to initialize the player in the simulations. No simulation possible"
        assert self.beast is not None,  "You have to initialize the police in the simulations. No simulation possible"

        player_action_feasible = False
        while not player_action_feasible:
                p_action = self.get_player_action()
                player_action_feasible = self.maze.check_beast_action(action=p_action, state=self.player.state)

        beast_action_feasible = False
        while not beast_action_feasible:
            b_action = self.get_beast_action()
            beast_action_feasible = self.maze.check_beast_action(action=b_action, state=self.beast.state)

        self.player.update_state(p_action)
        self.beast.update_state(b_action)
        self.simulation_time += 1
        return p_action, b_action


class MazeGame():
    def __init__(self, player, beast, maze):
        self.player = player
        self.beast = beast
        self.maze = maze
        self.sim_time = 0

    def step_simulation(self):

        p_action = self.player.choose_action(time_step=self.sim_time, beast_state=self.beast.state)
        beast_action_feasible = False
        while not beast_action_feasible:
            b_action = self.beast.choose_action()
            beast_action_feasible = self.maze.check_beast_action(action=b_action, state=self.beast.state)
        if self.player.state.x == self.beast.state.x and self.player.state.y == self.beast.state.y:
            b_action = 'stay'
            p_action = 'stay'

        self.player.update_state(p_action)
        self.beast.update_state(b_action)
        self.sim_time += 1

    def plot_game(self):
        print('Map at step: '+str(self.sim_time))
        x_p = self.player.state.x
        y_p = self.player.state.y
        x_b = self.beast.state.x
        y_b = self.beast.state.y

        for y in range(self.maze.height):
            print()
            for x in range(self.maze.length):
                if x_p == x_b and y_p == y_b:
                    if x_p == x and y_p == y:
                        cprint('O')
                else:
                    if x_p == x and y_p == y:
                        cprint('P')
                    elif x_b == x and y_b == y:
                        cprint('B')
                    else:
                        cprint(' ')
                cprint('|')
        print()

from maze_utilities import *
import copy
import operator
import time
import random

DEBUG = False

def log(s):
    if DEBUG:
        print(s)


class BellmansOptimizer():

    def __init__(self, player, maze, player_actions, beast_actions, T=15):
        self.player = player
        self.maze = maze
        self.T = T
        self.player_actions = player_actions
        self.beast_actions = beast_actions
        print('Bellman built')

    def solve_bellman(self):
        t = self.T
        i = 0
        while t >= 0:
            log('Bellmans step: ' +str(t))
            current_step = self.T - t -1
            for x_p in range(self.player.maze.length):
                for y_p in range(self.player.maze.height):
                    for x_b in range(self.player.maze.length):
                        for y_b in range(self.player.maze.height):
                            if self.player.policies[(x_p, y_p, x_b, y_b)] == None:
                                self.player.policies[(x_p, y_p, x_b, y_b)] = []
                            log('* -------------------------------------------------------------------- *')
                            log('State analyzed: '+str(x_p)+', '+str(y_p)+', '+str(x_b)+', '+str(y_b)+ ', step: '+str(t))
                            dead_end = False
                            player_state = MazeState(x_p, y_p)
                            beast_state = MazeState(x_b, y_b)
                            action = None

                            if t == self.T:
                                r_T = compute_reward(t, maze=self.maze, player_state=player_state, beast_state=beast_state, action=action, final=True)
                                dead_end = True
                                self.player.policies[(x_p, y_p, x_b, y_b)].append(('stay', r_T))
                                log('     ' + 'u_T = ' + str(r_T))

                            else:
                                r_t = compute_reward(t, self.maze, player_state, beast_state, action, False)
                                log('     ' + 'u_t = ' + str(r_t))

                            if not dead_end:
                                beast_feasible_actions = []
                                beast_future_states = []
                                for action in self.beast_actions:
                                    if self.maze.check_beast_action(action, beast_state):
                                        beast_feasible_actions.append(action)
                                        new_beast_state = copy.deepcopy(beast_state)
                                        new_beast_state.update(action)
                                        beast_future_states.append(new_beast_state)

                                log('Players feasible actions: ')
                                player_feasible_actions = self.player_actions

                                if player_state.x == self.maze.goal_state.x and player_state.y == self.maze.goal_state.y:
                                    player_feasible_actions = ['stay']
                                    beast_future_states = [copy.deepcopy(beast_state)]

                                if player_state.x == beast_state.x and player_state.y == beast_state.y:
                                    player_feasible_actions = ['stay']
                                    beast_future_states = [copy.deepcopy(beast_state)]

                                seq = player_actions
                                possible_policies_rewards = dict.fromkeys(seq, -1000)
                                for action in player_feasible_actions:
                                    u_t = 0
                                    if self.maze.check_player_action(action, player_state):
                                        log(action)
                                        for future_state_beast in beast_future_states:
                                            future_state_player = copy.deepcopy(player_state)
                                            future_state_player.update(action)
                                            x_f_p = future_state_player.x
                                            y_f_p = future_state_player.y
                                            x_f_b = future_state_beast.x
                                            y_f_b = future_state_beast.y
                                            u_star_next = self.player.policies[(x_f_p, y_f_p, x_f_b, y_f_b)][current_step][1]
                                            u_t += 1 / len(beast_future_states) * u_star_next
                                        u_t += r_t
                                        possible_policies_rewards[action] = u_t
                                        log('     '+ 'u_t = ' + str(u_t))
                                #optimal_policy = max(possible_policies_rewards)
                                optimal_policy = max(possible_policies_rewards.items(), key=operator.itemgetter(1))[0]
                                optimal_future_cost = possible_policies_rewards[optimal_policy]
                                log('Policy chosen: ')
                                self.player.policies[(x_p, y_p, x_b, y_b)].append((optimal_policy, optimal_future_cost))
                                log(self.player.policies[(x_p, y_p, x_b, y_b)])
            t -= 1








def compute_reward(t, maze, beast_state, player_state, action, final):
    if player_state.x == maze.goal_state.x and player_state.y == maze.goal_state.y:
        if final:
            return 100
        else:
            return 0
    if player_state.x == beast_state.x and player_state.y == beast_state.y:
        return 0
    return 0





if __name__ == '__main__':
    goal_state = MazeState(4, 4)
    our_maze = Maze(6, 5, goal_state)
    our_maze.add_horiz_wall(3, 1, 4)
    our_maze.add_horiz_wall(1, 4, 5)
    our_maze.add_vert_wall(1, 0, 2)
    our_maze.add_vert_wall(3, 1, 2)
    our_maze.add_vert_wall(3, 4, 4)

    init_p_state = MazeState(0, 0)
    init_b_state = MazeState(4, 4)
    player_actions = ['up', 'down', 'left', 'right', 'stay']
    beast_actions = ['up', 'down', 'left', 'right', 'stay']

    p1 = MazePlayer(init_p_state, our_maze, possible_actions=player_actions)
    b1 = MazeBeast(init_b_state, our_maze, possible_actions=beast_actions)
    p1.init_policies()


    # Solve the optimal control problem

    T = 15


    b_opt = BellmansOptimizer(p1, our_maze, player_actions, beast_actions, T)
    print(b_opt.maze.goal_state.x)
    print(b_opt.maze.goal_state.y)
    start = time.time()
    b_opt.solve_bellman()
    end = time.time()
    print('COMPUTATION TIME: ')
    print(end - start)

    state = (0, 0, 4, 4)
    policies = p1.policies[state]

    print('Optimal policy for the state: ', str(state))
    for i in range(T+1):
        print(policies[-(i+1)])
    my_game = MazeGame(player=p1, beast=b1, maze=our_maze)
    print("Player policies computed.")
    print("Starting the game")
    for j in range(0):
        print('**********************************************')
        print('final game: '+str(j))
        del my_game
        my_game = MazeGame(player=p1, beast=b1, maze=our_maze)
        for i in range(T):
            my_game.plot_game()
            my_game.step_simulation()
        my_game.plot_game()

        my_game.plot_game()






import numpy as np
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque,namedtuple
import random
import math
import DQNagent
import maze as mz
import DQNparameters as dp


class Player:
    def __init__(self,i,j):
        self.i = i
        self.j = j

class Maze:
    def __init__(self,maze,charparams:dp.CharacterParams) -> None:
        self.env = np.array(maze)
        self.original = np.array(maze)
        nr,nc = self.env.shape
        self.rows = nr
        self.columns = nc
        self.characterparams = charparams
        self.agent_rep = charparams.agent
        self.space_rep = charparams.space
        self.enemy_rep = charparams.enemy
        self.point_rep = charparams.goal
        self.wall_rep = charparams.wall
        self.player = None
        self.old_action = None
        self.new_action = None
        self.num_points_left = None
        self.number_of_steps = 0
        self.spawn_player()
        #self.print_state()



    def specific_spawn_player(self,i,j):
        self.player = Player(i,j)

    def random_spawn_goal(self):
        pass


    def spawn_player(self):
        e = self.env
        zero_indices = np.argwhere(e == self.space_rep)
        if len(zero_indices) == 0:
            point_indices = np.argwhere(e == self.point_rep)
            point_indices= np.array(point_indices) #why
            random_point_index = np.random.choice(point_indices.shape[0])
            random_point_index_tuple = tuple(point_indices[random_point_index])
            self.specific_spawn_player(random_point_index_tuple[0],random_point_index_tuple[1])
            # This is spawning player in place of point, must remove point from here
        else:
            random_index = np.random.choice(zero_indices.shape[0])
            random_zero_index = tuple(zero_indices[random_index])
            self.specific_spawn_player(random_zero_index[0], random_zero_index[1])
        
        self.update_state()
        
        
        #print(self.env)
    
    def maze_reset(self,m):
        # self.env= self.original.copy()
        # self.spawn_player()
        # self.old_action= None
        # self.new_action = None
        # self.number_of_steps = 0
        new_maze = Maze(m,self.characterparams)
        return new_maze

    
    def calculate_points_left(self):
        e = self.env
        points_left = np.count_nonzero(e == 1)
        return points_left
    
    def update_points_left(self):
        self.num_points_left = self.calculate_points_left()

    def update_pellets(self):
        e = self.env
        p = self.player
        if e[p.i,p.j] == self.point_rep:
            e[p.i,p.j] = self.space_rep

        self.update_points_left()

    
    def update_state(self):
        self.update_pellets()

    def force_end(self):
        return self.number_of_steps >= 100


        

    def return_state(self):

        e = self.env.copy()
        e[self.player.i,self.player.j] = self.agent_rep

        return e
    
    def return_flat_state(self):

        e = self.env.copy()
        e[self.player.i,self.player.j] = self.agent_rep

        return e.flatten()
    
    def print_state(self):

        e = self.env.copy()
        e[self.player.i,self.player.j] = self.agent_rep
        print(e)

    
    def what_is_in_next_block(self,action):
        p_i = self.player.i
        p_j = self.player.j
        if action == dp.UP:
            p_i = p_i-1
        if action == dp.RIGHT:
            p_j = p_j+1
        if action == dp.DOWN:
            p_i = p_i+1
        if action == dp.LEFT:
            p_j = p_j-1
        e = self.env.copy()
        if self.check_in_bounds(p_i,p_j,self.rows,self.columns):
            next_block = e[p_i,p_j]
        else:
            next_block = -2

        return next_block
    
    def check_in_bounds(self,i,j,nr,nc):
        return i >= 0 and i < nr and j >= 0 and j < nc 
    
    def return_reward(self,next_block):
        is_wall = False
        is_enemy = False
        reward = -0.1
        if next_block == self.point_rep:
            reward += 2
        elif next_block == self.space_rep:
            reward+=0
        elif next_block == self.enemy_rep:
            #reward+=-5
            is_enemy = True
        elif next_block == self.wall_rep:
            #reward+=-2
            is_wall= True
        else:
            print("There is something wrong or DQN chose invalid move")
        return reward,is_wall,is_enemy
    

    def send_to_agent(self):
        pass
    
    def move_player(self,action):
        e = self.env.copy()
        next_block = self.what_is_in_next_block(action)
    
        self.change_action(action)
        if next_block != self.wall_rep:

            if (action == dp.UP) and (self.player.i-1 >= 0 ) and (self.player.i-1 < self.rows) :
                self.player.i = self.player.i-1

            elif action == dp.RIGHT and (self.player.j+1 >= 0 ) and (self.player.j+1 < self.columns):
                self.player.j = self.player.j+1

            elif action == dp.DOWN and (self.player.i+1 >= 0 ) and (self.player.i+1 < self.rows) :
                self.player.i = self.player.i+1

            elif action == dp.LEFT and (self.player.j-1 >= 0 ) and (self.player.j-1 < self.columns):
                self.player.j = self.player.j-1
            else:
                print("Invalid action")
        
        self.number_of_steps+=1

        reward,is_wall, is_enemy  = self.return_reward(next_block)
        done = self.game_ended(is_enemy)

        self.update_state()
        #self.print_state()
        new_state = self.return_state()

        return reward,done,new_state


    def has_won(self):
        return self.num_points_left == 0
    
    def game_ended(self, is_enemy):
        return (is_enemy) or (self.has_won()) or (self.force_end())

    #find a way to get the reward, new_state, done to agent when requested
    

    def change_action(self,action):
        self.old_action = self.new_action
        self.new_action = action

    



def main():
    character_params = dp.CharacterParams(dp.WALL,dp.PATH,dp.AGENT,dp.ENEMY,dp.GOAL)
    m = Maze(mz.bernard_maze(),character_params)
    m.spawn_player()
    m.print_state()
    reward,done,new_state = m.move_player(dp.LEFT)
    m.print_state()
    print(f"reward: {reward} done: {done}")
    print(new_state)





if __name__ == '__main__':
    main()








# self.change_action(action)
# if action == UP :
#     if e[self.player.i-1,self.player.j] != self.wall_rep:
#         self.player.i = self.player.i-1

# elif action == RIGHT :
#     if e[self.player.i,self.player.j+1] != self.wall_rep:
#         self.player.j = self.player.j+1

# elif action == DOWN :
#     if e[self.player.i+1,self.player.j] != self.wall_rep:
#                 self.player.i = self.player.i+1


# elif action == LEFT :
#     if e[self.player.i,self.player.j-1] != self.wall_rep:
#         self.player.j = self.player.j-1


# else:
#     print("Invalid action")
# self.update_state()
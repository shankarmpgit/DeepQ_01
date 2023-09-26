
import maze as mz
import torch
from torch import nn
from collections import deque
import itertools
import numpy as np
import random
from tkinter import *
import time
import DQNparameters as dp
import DQNagent
import ENV

colors = {
    dp.WALL: "black", 
    dp.PATH: "white",  
    dp.GOAL: "green", 
    dp.AGENT: "orange",    
    dp.ENEMY: "red"
}


def read_in_parameters(maze:ENV.Maze, agent:DQNagent.DQNAgent,path):
    
    test_net  = DQNagent.DQN(maze.rows,maze.columns,4)
    test_net.load_state_dict(torch.load(path))
    return test_net

def visualise(m:ENV.Maze, game_canvas:Canvas,root:Tk):

    e = m.return_state()      
    nr,nc = e.shape

    cell_width = 1000//nc
    cell_height = 1000//nr
    for row in range(nr):
        game_canvas.grid_rowconfigure(row, weight=1, minsize=cell_height)
        for column in range(nc):
            game_canvas.grid_columnconfigure(column, weight=1, minsize=cell_width)
            x1 = column * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            game_canvas.create_rectangle(x1, y1, x2, y2, fill=colors[e[row, column]])
    
    time.sleep(0.1)
    root.update()

def create_visual():

    root = Tk()
    root.configure(bg="black")
    root.geometry(f'{1000}x{1000}')
    root.title('DQN maze')
    root.resizable(False,False)
    game_canvas = Canvas(
        root,
        width = 1000,
        height = 1000
        )
    game_canvas.pack()
    return root, game_canvas 

def testing(maze:ENV.Maze, agent:DQNagent.DQNAgent,path):
    test_net = read_in_parameters(maze,agent,path)
    root, game_canvas = create_visual()
    visualise(maze,game_canvas,root)
    time.sleep(5)

    maze = maze.maze_reset(mz.medium_maze(7,7,3))
    for step in itertools.count():
        visualise(maze,game_canvas,root)
        st = maze.return_state()
        at = test_net.act(st.flatten())
        _,done,_ = maze.move_player(at)
        if done:
             maze = maze.maze_reset(mz.medium_maze(7,7,3))


    root.mainloop()


if __name__ == '__main__':
    character_params = dp.CharacterParams(dp.WALL,dp.PATH,dp.AGENT,dp.ENEMY,dp.GOAL)
    deepQ_params = dp.DeepQParams(dp.GAMMA,dp.BATCH_SIZE,dp.BUFFER_SIZE,dp.MIN_REPLAY_SIZE,dp.TARGET_UPDATE_FREQUENCY,dp.OPTIMIZER_LR)
    training_params = dp.TrainingParams(dp.EPSILON_START,dp.EPSILON_DECAY,dp.EPSILON_END,dp.EPISODES)
    m = ENV.Maze(mz.medium_maze(7,7,3),character_params)
    a = DQNagent.DQNAgent(character_params,deepQ_params)
    path = '/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/state_dic_policy_net/test_v10.pth'

    testing(m,a,path)
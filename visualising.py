import DeepQ_Classes as DQC
import DeepQ_Training
import maze as mz
import torch
from torch import nn
from collections import deque
import itertools
import numpy as np
import random
import  maze as mz
from tkinter import *
import time

NUM_ACTIONS = 4
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE =  50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 100000
TARGET_UPDATE_FREQUENCY = 1000
OPTIMIZER_LR = 5e-4
EPISODES = 100 
WALL = -1
GOAL = 1
AGENT = 6
PATH = 0
ENEMY = 3
ROWS =4
COLUMNS = 4

colors = {
    WALL: "black", 
    PATH: "white",  
    GOAL: "green", 
    AGENT: "orange",    
    ENEMY: "red"
}


def read_in_parameters():
    
    training_params = DQC.TrainingParams(EPSILON_START,EPSILON_DECAY,EPSILON_END,EPISODES)
    deepQ_params = DQC.DeepQParams(GAMMA,BATCH_SIZE,BUFFER_SIZE,MIN_REPLAY_SIZE,TARGET_UPDATE_FREQUENCY,OPTIMIZER_LR)
    character_params = DQC.CharacterParams(WALL,PATH,AGENT,ENEMY,GOAL)
    m = DQC.Maze(mz.simple_maze(ROWS,COLUMNS),NUM_ACTIONS,character_params)
    test_net  = DQC.DQN(m)
    test_net.load_state_dict(torch.load('/home/shankar/Documents/Github/DeepQlearning/state_dic_policy_net/test_v3.pth'))
    return test_net

def visualise(m:DQC.Maze, game_canvas:Canvas,root:Tk):

    e = m.env.copy()
    print(e)
    a = m.agent
   # g = m.goal
    e[a.i,a.j] = m.agent_rep
    #e[g.i,g.j] = para.GOAL       
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

def main():
    test_net = read_in_parameters()
    root, game_canvas = create_visual()
    training_params = DQC.TrainingParams(EPSILON_START,EPSILON_DECAY,EPSILON_END,EPISODES)
    deepQ_params = DQC.DeepQParams(GAMMA,BATCH_SIZE,BUFFER_SIZE,MIN_REPLAY_SIZE,TARGET_UPDATE_FREQUENCY,OPTIMIZER_LR)
    character_params = DQC.CharacterParams(WALL,PATH,AGENT,ENEMY,GOAL)
    m = DQC.Maze(mz.simple_maze(ROWS,COLUMNS),NUM_ACTIONS,character_params)
    visualise(m,game_canvas,root)
    time.sleep(5)

    for step in itertools.count():
        visualise(m,game_canvas,root)
        st = m.flattened_state()
        moves = m.all_actions
        at = test_net.act(st)
        print(at)
        move = moves[at]
        rt = m.do_a_move(move)
        if m.has_won():
            m = DQC.Maze(mz.simple_maze(ROWS,COLUMNS),NUM_ACTIONS,character_params)


    root.mainloop()


if __name__ == '__main__':
    main()
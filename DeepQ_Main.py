import numpy as np
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque,namedtuple
import random
import DeepQ_Classes as DQC
import math
import DeepQ_Training as DQT
import maze as mz

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

colors = {
    WALL: "black", 
    PATH: "white",  
    GOAL: "green", 
    AGENT: "orange",    
    ENEMY: "red"
}


def main():
    training_params = DQC.TrainingParams(EPSILON_START,EPSILON_DECAY,EPSILON_END,EPISODES)
    deepQ_params = DQC.DeepQParams(GAMMA,BATCH_SIZE,BUFFER_SIZE,MIN_REPLAY_SIZE,TARGET_UPDATE_FREQUENCY,OPTIMIZER_LR)
    character_params = DQC.CharacterParams(WALL,PATH,AGENT,ENEMY,GOAL)
    m = DQC.Maze(mz.simple_maze(5,5),NUM_ACTIONS,character_params)

    policy_net = DQC.DQN(m)
    target_net = DQC.DQN(m)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.AdamW(policy_net.parameters(),lr=deepQ_params.optimizer_lr,amsgrad=True)
    rb  = DQC.ReplayBuffer(deepQ_params)

    total_steps = 0
    episode_steps = 0
    episode_count = 0
    avg_reward = 0.0

    DQT.Populate_ReplayBuffer(m,rb)

    for i in range(training_params.training_episodes):
        episode_count+=1
        m = DQC.Maze(mz.simple_maze(5,5),NUM_ACTIONS,character_params)
        done = 0
        while not done:
            total_steps+=1
            episode_steps+=1
            done,reward = DQT.Training(m,rb,training_params,policy_net,total_steps)
            DQT.optimize_model(rb,deepQ_params,policy_net,target_net,optimizer)
            avg_reward+= reward
            if total_steps % deepQ_params.target_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if m.should_force_end():
                break

        
        print()
        print(f'Steps in episode: {episode_steps}')
        print(f'Score for episode: {avg_reward}')
        avg_reward = 0.0
        episode_steps = 0

        
        




if __name__ == '__main__':
    main()
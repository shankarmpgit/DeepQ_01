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


def random_action(maze:DQC.Maze):
   moves = maze.compute_possible_moves()
   random.shuffle(moves)
   move_idx,move = moves[0]
   return move_idx,move

def get_action(maze:DQC.Maze,policy_net:DQC.DQN):
    state = maze.flattened_state()
    action = policy_net.act(state)
    return action


def specific_action(maze:DQC.Maze,action):
    move_idx = action
    moves = maze.all_actions
    move = moves[move_idx]
    return move_idx, move

def get_reward(maze:DQC.Maze,move):
    reward = maze.do_a_move(move)
    return reward

def goal_reached(maze:DQC.Maze):
    done  = maze.has_won()
    return done

def do_transition(maze:DQC.Maze,move_idx, move):
    state = maze.flattened_state()
    reward = get_reward(maze,move)
    new_state = maze.flattened_state()
    done = goal_reached(maze)
    return state,move_idx,reward,new_state,done


def Populate_ReplayBuffer(maze:DQC.Maze,rb:DQC.ReplayBuffer):
    env = maze.env
    action_space = maze.action_space
    characterParams = maze.characterParams
    buffer_size = rb.buffer_size
    num_random_samples = round(0.2*buffer_size)
    for i in range(num_random_samples):
        move_idx,move = random_action(maze)
        state,action,reward,new_state,done = do_transition(maze,move_idx,move)
        transition = DQC.Sarsd(state,action,reward,new_state,done)
        rb.push(transition)
        if done:
            maze = DQC.Maze(env,action_space,characterParams)

def Training(maze:DQC.Maze,rb:DQC.ReplayBuffer,trainingparam:DQC.TrainingParams,policy_net:DQC.DQN,total_steps):
    eps_end = trainingparam.epsilon_end
    eps_start = trainingparam.epsilon_start
    eps_decay = trainingparam.epsilon_decay

    sample = random.random()

    eps_threshold = eps_end + (eps_start-eps_end) * math.exp(-1 * total_steps/eps_decay)

    if sample > eps_threshold:
        with torch.no_grad():
            act = get_action(maze,policy_net)
            move_idx,move = specific_action(maze,act)
    else:
        move_idx, move = random_action(maze)

    state,action,reward,new_state,done = do_transition(maze,move_idx,move)
    transition = DQC.Sarsd(state,action,reward,new_state,done)
    rb.push(transition)
    return done,reward


    
            

def optimize_model(rb:DQC.ReplayBuffer,deepQparam:DQC.DeepQParams,policy_net:DQC.DQN,target_net:DQC.DQN,optimizer:torch.optim.AdamW):
    batch_size = deepQparam.batch_size
    gamma = deepQparam.gamma
    if len(rb) < batch_size:
        return
    
    transitions = rb.sample()
    # for transition in transitions:
    #       print(transition)

    state_batch = torch.stack([torch.Tensor(s.state) for s in transitions])
    action_batch = torch.stack([torch.Tensor([s.action]) for s in transitions])
    reward_batch = torch.stack([torch.Tensor([s.reward]) for s in transitions])
    next_state_batch = torch.stack([torch.Tensor(s.next_state) for s in transitions])
    done_batch = torch.stack([torch.Tensor([s.done]) for s in transitions])


    
    
    # batch = DQC.Transition(*zip(*transitions))
    # state_batch = torch.cat(tuple(torch.as_tensor(s, dtype=torch.float32) for s in batch.state))
    # action_batch = torch.cat(tuple(torch.tensor(a, dtype=torch.int32) for a in batch.action))
    # reward_batch = torch.cat(tuple(torch.as_tensor(s, dtype=torch.float32) for s in batch.reward))
    # next_state_batch = torch.cat(tuple(torch.as_tensor(s, dtype=torch.float32) for s in batch.next_state))
    # done_batch = torch.cat(tuple(torch.tensor(a, dtype=torch.int32) for a in batch.done))


 
    state_action_values = policy_net(state_batch).gather(1,action_batch.long())
    with torch.no_grad():
        target_values = target_net(next_state_batch)

    next_state_action_values = target_values.max(dim=1,keepdim=True)[0]

    expected_state_action_values = reward_batch + gamma * (1 - done_batch) * next_state_action_values
    
    loss = nn.functional.smooth_l1_loss(state_action_values,expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
    optimizer.step()

    
















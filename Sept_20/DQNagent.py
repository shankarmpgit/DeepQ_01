import numpy as np
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
from collections import namedtuple
import DQNparameters as dp




class DQN(nn.Module):
    def __init__(self,nr,nc,actions) -> None: # env is of class Maze
        super(DQN,self).__init__()
        
        in_features = int(nr * nc)
        self.layer1 = nn.Linear(in_features,128)
        self.layer2 = nn.Linear(128,64)
        self.layer3 = nn.Linear(64,actions)


    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

    #Need to check this action
    #Why do you only have to put self instead of self.forward

    def act(self,state):
        state = np.array(state)
        state = state.flatten()
        state_t = torch.as_tensor(state,dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0)) # Adding a batch dimension add doing a forward pass
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action
    
class ReplayBuffer(object):
    def __init__(self,DeepQParam:dp.DeepQParams) -> None:
        self.buffer_size = DeepQParam.buffer_size
        self.batch_size = DeepQParam.batch_size
        self.memory = deque([],maxlen=self.buffer_size)

    def push(self, transition:dp.Sarsd):
        self.memory.append(transition)
    
    def sample(self):
        return random.sample(self.memory,self.batch_size)
    
    def __len__(self):
        return len(self.memory)



class DQNAgent:
    def __init__(self,charparams:dp.CharacterParams,deepQparams:dp.DeepQParams) -> None:
        self.agent_rep = charparams.agent
        self.space_rep = charparams.space
        self.enemy_rep = charparams.enemy
        self.point_rep = charparams.goal
        self.i = None
        self.j = None
        self.rows = None
        self.columns = None
        self.policy_net = None
        self.target_net = None
        self.rb = None
        self.optimizer = None
        self.batch_size = deepQparams.batch_size
        self.gamma = deepQparams.gamma
        self.lr = deepQparams.optimizer_lr
        self.target_update_hz = deepQparams.target_update_frequency

    
    def assign_policy_net(self,policy_net:DQN):
        self.policy_net = policy_net

    def assign_target_net(self,target_net:DQN):
        self.target_net = target_net
    
    def assign_replay_buffer(self,rb:ReplayBuffer):
        self.rb = rb

    def assign_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(),lr=self.lr,amsgrad=True)


    def optimize_model(self):

        if len(self.rb) < self.batch_size:
            print("Not big enough replay buffer")
            return


        transitions= self.rb.sample()
        state_batch = torch.stack([torch.Tensor(s.state) for s in transitions])
        action_batch = torch.stack([torch.Tensor([s.action]) for s in transitions])
        reward_batch = torch.stack([torch.Tensor([s.reward]) for s in transitions])
        next_state_batch = torch.stack([torch.Tensor(s.next_state) for s in transitions])
        done_batch = torch.stack([torch.Tensor([s.done]) for s in transitions])

        state_action_values = self.policy_net(state_batch).gather(1,action_batch.long())
        
        with torch.no_grad():
            target_values = self.target_net(next_state_batch)

        next_state_action_values = target_values.max(dim=1,keepdim=True)[0]

        expected_state_action_values = reward_batch + self.gamma * (1 - done_batch) * next_state_action_values
        
        loss = nn.functional.smooth_l1_loss(state_action_values,expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(),100)
        self.optimizer.step()


    

    def get_valid_actions(self,state):
        valid_moves = []
        #print(state)
        self.get_agent_location(state)
        nr,nc = state.shape
        #Check up
        if self.check_in_bounds(self.i-1,self.j,nr,nc):
            valid_moves.append(dp.UP)
        #Check right
        if self.check_in_bounds(self.i,self.j+1,nr,nc):
            valid_moves.append(dp.RIGHT)
        #Check down
        if self.check_in_bounds(self.i+1,self.j,nr,nc):
            valid_moves.append(dp.DOWN)   
        #Check right
        if self.check_in_bounds(self.i,self.j-1,nr,nc):
            valid_moves.append(dp.LEFT)

        #print(valid_moves)
        return valid_moves
    
    def check_in_bounds(self,i,j,nr,nc):
        return i >= 0 and i < nr and j >= 0 and j < nc 

    def pick_random_action(self,state):
        valid_actions = self.get_valid_actions(state)
        random.shuffle(valid_actions)
        action = valid_actions[0]
        return action
    
    def pick_dqn_action(self,state):
        state = np.array(state)
        state = state.flatten()
        action = self.policy_net.act(state)
        return action
    
    def get_action(self,state):
        #random action at the moment
        #action = self.pick_random_action(state)
        #DQN action
        action = self.pick_dqn_action(state)

    
        return action



    def get_agent_location(self,state):
        location = np.where(state == self.agent_rep)
        self.i,self.j = location

        #print(f"i:{self.i}, j:{self.j}")


    # Need to do DQN action
    # Need to do training


def main():
    character_params = dp.CharacterParams(dp.WALL,dp.PATH,dp.AGENT,dp.ENEMY,dp.GOAL)
    deepQ_params = dp.DeepQParams(dp.GAMMA,dp.BATCH_SIZE,dp.BUFFER_SIZE,dp.MIN_REPLAY_SIZE,dp.TARGET_UPDATE_FREQUENCY,dp.OPTIMIZER_LR)
    a = DQNAgent(character_params,deepQ_params)
    state = np.zeros((4,4))
    state[2,2] = character_params.agent
    print(state)
    a.pick_random_action(state)

if __name__ == '__main__':
    main()


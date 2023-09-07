import numpy as np
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state','done'))

class Agent:
    def __init__(self,i,j):
        self.i = i
        self.j = j
        self.maze = None
        self.current_state= None

    @property
    def loc(self):
        return(self.i,self.j)

    def vmove(self,direction):
        direction = 1 if direction >0 else -1
        return Agent(self.i+direction,self.j)
    
    def hmove(self,direction):
        direction = 1 if direction >0 else -1
        return Agent(self.i,self.j+direction)
    
    @property
    def all_actions(self):
        return [
            self.vmove(1),
            self.vmove(-1),
            self.hmove(1),
            self.hmove(-1)
        ]

    
    def query_possible_actions(self):
        actions = self.all_actions
        possible_actions = self.maze.compute_possible_moves(actions)
        return possible_actions
    
    def do_random_action(self):
        pass

    def query_maze(self,m):
        pass



@dataclass
class DeepQParams:
    gamma:  float
    batch_size: int
    buffer_size: int
    min_replay_size: int
    target_update_frequency: int
    optimizer_lr: float


@dataclass
class TrainingParams:
    epsilon_start: int
    epsilon_decay: int
    epsilon_end: int
    training_episodes: int

@dataclass 
class Sarsd:
    state:Any
    action:int
    reward:float
    next_state:Any
    done:int


@dataclass
class CharacterParams:
    wall: int
    space:  int
    agent: int
    enemy: int
    goal: int


   
class Maze:
    def __init__(self,maze,action_space,CharacterParam:CharacterParams):
        self.env = np.array(maze)
        nr,nc = self.env.shape
        self.rows = nr
        self.columns = nc
        self.agent = None
        self.enemy = None
        self.action_space = action_space
        self.characterParams = CharacterParam
        self.agent_rep = CharacterParam.agent
        self.wall_rep = CharacterParam.wall
        self.space_rep = CharacterParam.space
        self.goal_rep  = CharacterParam.goal
        self.enemy_rep = CharacterParam.enemy
        self.num_points_left = None
        self.steps_since_last_point = 0
        self.update_points_left()
        self.random_spawn_agent()


    def calculate_points_left(self):
        e = self.env
        points_left = np.count_nonzero(e == 1)
        return points_left
    
    def update_points_left(self):
        self.num_points_left = self.calculate_points_left()

    def specific_spawn_agent(self,i,j):
        self.agent = Agent(i,j)
    
    def random_spawn_agent(self): #Creates randomly spawned agent 
        e = self.env.copy()
        zero_indices = np.argwhere(e == self.space_rep) 
        if len(zero_indices) == 0:
            randrow = random.randint(0,self.rows-1)
            randcol = random.randint(0,self.columns-1)
            self.specific_spawn_agent(randrow,randcol)
        else:
            random_index = np.random.choice(zero_indices.shape[0])
            random_zero_index = tuple(zero_indices[random_index])
            self.specific_spawn_agent(random_zero_index[0], random_zero_index[1])
    


    @property
    def all_actions(self):
        a = self.agent
        return  [
            a.vmove(1),
            a.vmove(-1),
            a.hmove(-1),
            a.hmove(1)
        ]
        
    #CHEATING  can still use if agent is inbounds
    def in_bounds(self,i,j):
        nr,nc = self.env.shape
        return i >= 0 and i < nr and j >= 0 and j < nc 
    
    def agent_in_bounds(self,a):
        return self.in_bounds(a.i,a.j)
    
    def thats_not_a_wall(self,a):
        return not self.env[a.i,a.j] == self.wall_rep
    
    def is_valid_new_agent(self,a):
        return self.agent_in_bounds(a) 

    def compute_possible_moves(self,actions):
        return [(ii,m) for ii,m in enumerate(actions) if self.is_valid_new_agent(m)]
    
    #CHEATING

    def update_agent(self,move:Agent):

        is_done = False
        cur_state = self.return_state()
        next_block = self.what_is_in_next_block(move)
        reward,is_wall,is_enemy = self.return_reward(next_block)


        #Decides whether the agent moves into the next block
        if (is_wall == False) and (self.is_valid_new_agent(move)):
            self.agent = move
        else:
            self.agent =self.agent
        
        #Upates pellet counts
        self.update_pellets()
        new_state = self.return_state()

        #Finds out if the game is done
        if (is_enemy == True) or (self.has_won()):
            is_done = True

        return cur_state,reward,new_state,is_done
        
        
        
    def update_pellets(self):
        e = self.env
        a = self.agent
        if e[a.i,a.j] == self.goal_rep:
            e[a.i,a.j] = self.space_rep

        self.update_points_left()


    
    def what_is_in_next_block(self,move:Agent):
        future_agent = move
        e = self.env
        next_block_occupant = e[future_agent.i,future_agent.j]
        return next_block_occupant


    def return_state(self):
        e = self.env.copy()
        a = self.agent
        e[a.i,a.j] = self.agent_rep
        en = self.enemy
        e[en.i,en.j] = self.enemy_rep
        return e
    
    def return_reward(self,next_block):
        e = self.env
        is_wall = False
        is_enemy = False
        reward = -1
        if next_block == self.goal_rep:
            reward += 3
        elif next_block == self.space_rep:
            reward+=-1
        elif next_block == self.enemy:
            reward+=-5
            is_enemy = True
        elif next_block == self.wall_rep:
            reward+=-2
            is_wall= True
        else:
            print("There is something wrong")
        return reward,is_wall,is_enemy

     
    
    def print_maze(self):
        e = self.env.copy()
        a = self.agent
        e[a.i,a.j] = self.agent_rep
        print(e)

class ReplayBuffer(object):
    def __init__(self,DeepQParam:DeepQParams) -> None:
        self.buffer_size = DeepQParam.buffer_size
        self.batch_size = DeepQParam.batch_size
        self.memory = deque([],maxlen=self.buffer_size)

    def push(self, transition:Sarsd):
        self.memory.append(transition)
    
    def sample(self):
        return random.sample(self.memory,self.batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self,maze:Maze) -> None: # env is of class Maze
        super(DQN,self).__init__()
        
        in_features = int(maze.rows * maze.columns)
        self.layer1 = nn.Linear(in_features,128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128,maze.action_space)


    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def act(self,state):
        state_t = torch.as_tensor(state,dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0)) # Adding a batch dimension add doing a forward pass
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action
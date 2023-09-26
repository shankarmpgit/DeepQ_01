from dataclasses import dataclass
from typing import Any

@dataclass
class CharacterParams:
    wall: int
    space:  int
    agent: int
    enemy: int
    goal: int

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

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NUM_ACTIONS = 4
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE =  50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 200000
TARGET_UPDATE_FREQUENCY = 1000
OPTIMIZER_LR = 5e-4
EPISODES = 20000
RUN_DURATION = 12 #Hours 

WALL = -1
GOAL = 1
AGENT = 6
PATH = 0
ENEMY = 3

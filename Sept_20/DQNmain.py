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
import ENV
import DQNparameters as dp
import DQNtesting as dt
import pickle
import itertools
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from dotenv import load_dotenv




def populate_replay_buffer(maze:ENV.Maze, agent: DQNagent.DQNAgent, training_params:dp.TrainingParams):
    num_random_samples = round(0.2 * agent.rb.buffer_size)
    maze = maze.maze_reset(mz.medium_maze(7,7,3))
    state = maze.return_state()
    for i in range(num_random_samples):
        action = agent.pick_random_action(state)
        reward,done,new_state = maze.move_player(action)
        transition = dp.Sarsd(state.flatten(),action,reward,new_state.flatten(),done)
        agent.rb.push(transition)

        #We ideally only want to query the Maze once, hence the new state becomes the state for next iteration
        state = new_state
        if done :
            print('done')
            maze = maze.maze_reset(mz.medium_maze(7,7,3))
            state = maze.return_state()



# Will get rid of this later
def training(maze:ENV.Maze, agent: DQNagent.DQNAgent, training_params:dp.TrainingParams):
    eps_end = training_params.epsilon_end
    eps_start = training_params.epsilon_start
    eps_decay = training_params.epsilon_decay
    episodes = training_params.training_episodes
    total_steps = 0
    episode_reward = 0.0
    print(episodes)
    for i in range(episodes):
        maze = maze.maze_reset(mz.medium_maze(7,7,3))
        state = maze.return_state()
        done = False
        while not done:
            sample = random.random()
            eps_threshold = eps_end + (eps_start-eps_end) * math.exp(-1 * total_steps/eps_decay)
            if sample > eps_threshold:
                action =  agent.pick_dqn_action(state)
            else:
                action = agent.pick_random_action(state)

            reward,done,new_state = maze.move_player(action)
            transition = dp.Sarsd(state.flatten(),action,reward,new_state.flatten(),done)
            agent.rb.push(transition)
            agent.optimize_model()
            state = new_state
            episode_reward+= reward
            total_steps+=1
        if total_steps % agent.target_update_hz == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            
        print(f'Episode : {i}')
        print(f'Score for episode: {episode_reward}')
        print(f'Episode threshold: {eps_threshold}')
        print(f'total steps: {total_steps}')
        episode_reward = 0.0

    torch.save(agent.policy_net.state_dict(),'/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/state_dic_policy_net/test_v10.pth' )
    torch.save(agent.target_net.state_dict(),'/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/state_dic_target_net/test_v10_.pth' )
    path = '/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/state_dic_policy_net/test_v10.pth'
    filepath = '/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/replay_buffer/rb_v10.pkl'
    save_replay_buffer(agent.rb,filepath)
    #dt.testing(maze,agent,path)

def inf_training(maze:ENV.Maze, agent: DQNagent.DQNAgent, training_params:dp.TrainingParams):
    eps_end = training_params.epsilon_end
    eps_start = training_params.epsilon_start
    eps_decay = training_params.epsilon_decay
    total_steps = 0
    episode_reward = 0.0
    run_duration_seconds =  60 # hours * 60 minutes * 60 seconds
    start_time = time.time()

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= run_duration_seconds:
            print(elapsed_time)
            send_email()
            break


        maze = maze.maze_reset(mz.medium_maze(7,7,3))
        state = maze.return_state()
        done = False
        while not done:
            sample = random.random()
            eps_threshold = eps_end + (eps_start-eps_end) * math.exp(-1 * total_steps/eps_decay)
            if sample > eps_threshold:
                action =  agent.pick_dqn_action(state)
            else:
                action = agent.pick_random_action(state)

            reward,done,new_state = maze.move_player(action)
            transition = dp.Sarsd(state.flatten(),action,reward,new_state.flatten(),done)
            agent.rb.push(transition)
            agent.optimize_model()
            state = new_state
            episode_reward+= reward
            total_steps+=1
        if total_steps % agent.target_update_hz == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
            
 
        print(f'Score for episode: {episode_reward}')
        print(f'Episode threshold: {eps_threshold}')
        print(f'total steps: {total_steps}')
        episode_reward = 0.0

    # torch.save(agent.policy_net.state_dict(),'/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/state_dic_policy_net/test_v10.pth' )
    # torch.save(agent.target_net.state_dict(),'/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/state_dic_target_net/test_v10_.pth' )
    # path = '/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/state_dic_policy_net/test_v10.pth'
    # filepath = '/home/shankar/Documents/Github/DeepQlearning/DeepQ_01/replay_buffer/rb_v10.pkl'
    # save_replay_buffer(agent.rb,filepath)
    #dt.testing(maze,agent,path)

def save_replay_buffer(rb:DQNagent.ReplayBuffer,filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(rb,file)

def load_replay_buffer(filepath):
    with open(filepath, 'rb') as file:
        rb = pickle.load(file)

    return rb



def send_email():
    PORT = 587
    EMAIL_SERVER = "smtp-mail.outlook.com"
    sender_email = "shanks.notify@outlook.com"
    password_email = "P@ssmes123"
    reciever_email = "shanks.notify@gmail.com"
    current_time = datetime.now().time()
    
    msg = EmailMessage()
    msg["Subject"] = "The run you have started is complete!"
    msg["From"] = formataddr(("Run Complete",f"{sender_email}"))
    msg["To"] = reciever_email
    msg["BCC"] = sender_email

    msg.set_content(
        f"The run that you have started has completed at {current_time}"
    )

    with smtplib.SMTP(EMAIL_SERVER,PORT) as server:
        server.starttls()
        server.login(sender_email,password_email)
        server.sendmail(sender_email,"shanks.notify@gmail.com", msg.as_string())




        
#Develop a way to get the data required and save it so that I can start the training again
#Need some other training parameters
#Need to create graphs to visualise results
#Also need to alter my code to do GPU computation 




        

    

    




def main():
    character_params = dp.CharacterParams(dp.WALL,dp.PATH,dp.AGENT,dp.ENEMY,dp.GOAL)
    deepQ_params = dp.DeepQParams(dp.GAMMA,dp.BATCH_SIZE,dp.BUFFER_SIZE,dp.MIN_REPLAY_SIZE,dp.TARGET_UPDATE_FREQUENCY,dp.OPTIMIZER_LR)
    training_params = dp.TrainingParams(dp.EPSILON_START,dp.EPSILON_DECAY,dp.EPSILON_END,dp.EPISODES)
    m = ENV.Maze(mz.medium_maze(7,7,3),character_params)
    a = DQNagent.DQNAgent(character_params,deepQ_params)
    policy_net = DQNagent.DQN(m.rows,m.columns,4)
    target_net = DQNagent.DQN(m.rows,m.columns,4)
    target_net.load_state_dict(policy_net.state_dict())
    rb  = DQNagent.ReplayBuffer(deepQ_params)
    a.assign_policy_net(policy_net)
    a.assign_target_net(target_net)
    a.assign_replay_buffer(rb)
    a.assign_optimizer()

    populate_replay_buffer(m,a,training_params)
    inf_training(m,a,training_params)
    

    # for _ in range(22):
    #     state = m.return_state()
    #     action = a.get_action(state)
    #     reward,done = m.move_player(action)
    #     print(f"reward: {reward} is done: {done}")






if __name__ == '__main__':
    main()
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from agent import Graph_agent
from graph_env import Graph_env

#file_name = "E3SM_cice_dot_inter_module.txt"
file_name = "ten_node_test2.txt"
env = Graph_env('merge_cyclomatic', file_name)

total_episodes = 1000        # Total episodes
total_test_episodes = 200    # Total test episodes
test_steps = 5               # Max steps per episode
num_iterations = 50

learning_rate = 0.0001        # Learning rate
gamma = 0.8                  # Discounting rate

# Exploration parameters
epsilon = 0.9                # Exploration rate
max_epsilon = 1.0            # Exploration probability at start
min_epsilon = 0.01           # Minimum exploration probability 
decay_rate = 0.01            # Exponential decay rate for exploration prob

# game_info = [game-name, #input, #hidden, #output, eps, lr, gamma]
game_info = ['merge_clustering_cv', 156, 50, 12, \
             epsilon, learning_rate, gamma]
#game_info = ['merge_clustering_cv', 69432, 500, 263, epsilon, learning_rate, gamma]
game_info

agent = Graph_agent(game_info)

test_rewards = []
# random test before training:
for episode in range(5):
    
    # print("done with getting original reward")
    done = False
    test_total_rewards = 0
    
    env.reset()
    state = env.graph.b_graph.state
    print(state)
    
    for step in range(test_steps):
            
        #print("prepare to select node")
        action = agent.test(state, step)
        next_state, reward, done = env.step(action)
        print(action)
        print(reward)
        state = next_state
        test_total_rewards += reward
        
        if done:
            print("it is done")
            break 
            
    test_rewards.append(test_total_rewards)
    #print(total_rewards)
    #print("average score: " + str(sum(rewards)/(episodes+1))

print("average is: " + str(sum(test_rewards)/5) + " at " + str(episode))

rewards = []
averages = []
old_average = 0
success = 0

rewards_window = deque(maxlen=10)
rewards_windows = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    
    # print("done with getting original reward")
    done = False
    total_rewards = 0
    
    env.reset()
    state = env.graph.b_graph.state
    
    for step in range(test_steps):
            
        #print("prepare to select node")
        action = agent.act(state, step)
        next_state, reward, done = env.step(action)
        agent.train(next_state, reward, action)
        state = next_state
        total_rewards += reward
        
        if done:
            break 
            
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    rewards_window.append(total_rewards)
    rewards.append(total_rewards)
    #print(total_rewards)
    #print("average score: " + str(sum(rewards)/(episodes+1))
    if(episode % 10 == 0):
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, \
              np.mean(rewards_window)))
        rewards_windows.append(np.mean(rewards_window))
    
    #print("average is: " + str(sum(rewards)/(episode+1)) + " at " + str(episode))
    #averages.append(sum(rewards)/(episode+1))
    
agent.save_model()

plt.figure(figsize=(15, 10))
plt.plot(averages)
plt.show()

test_rewards = []
agent.load_model()

# test after training:
for episode in range(5):
    
    # print("done with getting original reward")
    done = False
    test_total_rewards = 0
    
    env.reset()
    state = env.graph.b_graph.state
    
    for step in range(test_steps):
            
        #print("prepare to select nodnp.mean(rewards_window)e")
        action = agent.test(state, step)
        next_state, reward, done = env.step(action)
        print(action)
        print(reward)
        state = next_state
        test_total_rewards += reward
        
        if done:
            break 
            
    test_rewards.append(test_total_rewards)
    #print(total_rewards)
    #print("average score: " + str(sum(rewards)/(episodes+1))
print("average is: " + str(sum(test_rewards)/5) + " at " + str(episode))

agent.close()

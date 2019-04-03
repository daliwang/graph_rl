#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:46:45 2019

graph_env.py: This is graph environment class, called by main function

How to use this class: 
Gym : env = gym.make('LunarLander-v2')
Our framework : env = Graph_env('merge_cyclomatic', 'file_name')

@author: weijianzheng
"""

# coding: utf-8

import numpy as np
from graph import Graph

class Graph_env:
    
    def __init__(self, env_type, graph_file_name):
        
        self.type = env_type
        self.file_name = graph_file_name
        
        self.graph = Graph(env_type, self.file_name)
              
    # provide an action, this function will return the corresponding 
    # new_state, done and reward. 
    def step(self, action):
        
        done = self.graph.update_state(action)
        
        reward = self.graph.get_reward()

        return(self.graph.b_graph.state, reward, done)
        
    def reset(self):
        
        self.graph.reload()
        
    def render(self):
        return True
        # TODO: finish the code to represent the graph
        
    def close(self):
        #TODO: close the graph
        return True
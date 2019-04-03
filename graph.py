#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:06:44 2019

@author: weijianzheng

graph.py: This class is used to represent the graph, 
          it works as the interface between graph environment 
              and several different types of the graph object 
              (e.g., Basic_graph, networkx graph)

"""

import networkx as nx

from basic_graph import Basic_graph

class Graph:
    
    def __init__(self, env_type, file_name):
        
        self.env_type  = env_type
        self.file_name = file_name          
        
        # TODO: we need to initialize different types of the graph 
        #   based on env_type         
        if(self.env_type == 'merge_cyclomatic'): 
            # this game just need basic graph
            self.b_graph = Basic_graph(self.file_name, False)
            
        if(self.env_type == 'merge_clustering_cv'): 
            # this game use networkx 
            self.b_graph = Basic_graph(self.file_name, False)
            
            #print(self.file_name)
            self.edgelist_file_name = self.file_name.split(".")[0]
            self.edgelist_file_name = self.edgelist_file_name + ".edgelist"
            
            #print(self.edgelist_file_name)
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())
            
            self.old_cv =  nx.average_clustering(self.temp_G.to_undirected())
            
            print("The original cv is " + str(self.old_cv))
            
    def update_state(self, action):
        
        if(self.env_type == 'merge_cyclomatic'): 
            # this game just need basic graph
            self.done = self.b_graph.get_new_state(action[1], action[2])
   
        if(self.env_type == 'merge_clustering_cv'): 
            # this game need both basic and networkx graph
            self.done = self.b_graph.get_new_state(action[1], action[2])
            
            first_node = self.b_graph.function_list[action[1]]
            second_node = self.b_graph.function_list[action[2]]
            
            temp_GG = nx.contracted_nodes(self.temp_G, first_node, \
                                          second_node)
            self.temp_G.clear()
            self.temp_G = temp_GG.copy()
            
        return self.done
        

    def get_reward(self):
        
        if(self.env_type == 'merge_cyclomatic'): 
            # this game just need basic graph
            self.reward = self.b_graph.get_complexity_reward()
        
        if(self.env_type == 'merge_clustering_cv'): 
            
            self.new_cv = nx.average_clustering(self.temp_G.to_undirected())
            
            self.reward = self.new_cv - self.old_cv
            self.reward *= 5
            
            self.old_cv = self.new_cv
            
        return self.reward
            
            
    def reload(self):
        
        if(self.env_type == 'merge_cyclomatic'):
            self.b_graph = Basic_graph(self.file_name, True)
            
        if(self.env_type == 'merge_clustering_cv'):
            self.b_graph = Basic_graph(self.file_name, True) 
            
            self.temp_G = nx.read_edgelist(self.edgelist_file_name, \
                                           create_using=nx.DiGraph())    
            self.old_cv =  nx.average_clustering(self.temp_G.to_undirected())
            
            
            
            
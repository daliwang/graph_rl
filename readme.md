This is a copy of Weijian's graph RL code, that contains following python code:

__init__.py    setup python environment
agent.py       define a DQN RL agant that contains basic actions: init, test, train, step, act
agent_helper.py  define how an agent act on specific task (e.g., "graph")
graph.py       define the interface between "graph" environment and "graph" object
graph_env.py   define the open gym environment for "graph" model
model.py       define the NN structure for DQN agent

graph_RL_diagram shows the relationship between these files

merge_clustering_cv.py and merge_cyclomatic.py are two examples on how to write a main python drive for RL. 

Two extra files ten_node_test1.edgelist and ten_node_test2.txt are the edge list and node list of a simple graph. 

This code requires networkx for basic graph operation and calculation. 

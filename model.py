#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:49:09 2019

@author: weijianzheng
"""

import tensorflow as tf

class Model:
    
    def __init__(self, model_info):
        
        # first need to parse the application info and define nn
        self.game_name = model_info[0]
        self.in_num_units = model_info[1]
        self.hid_num_units = model_info[2]
        self.out_num_units = model_info[3]
        
        # define placeholders
        self.x = tf.placeholder(shape=[1,self.in_num_units], dtype=tf.float32)
        self.y = tf.placeholder(shape=[1,self.out_num_units],dtype=tf.float32)

        # set remaining variables
        self.epochs = 100
        self.batch_size = 128
        self.learning_rate = 0.01

        ### define weights and biases of the neural network 
        self.weights = {
            'hidden': tf.Variable(tf.random_normal([self.in_num_units, \
                                                    self.hid_num_units \
                                                    ], seed=1)),
            'output': tf.Variable(tf.random_normal([self.hid_num_units, \
                                                    self.out_num_units \
                                                    ], seed=1))
        }

        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.hid_num_units], \
                                                   seed=1)),
            'output': tf.Variable(tf.random_normal([self.out_num_units], \
                                                   seed=1))
        }

        self.hidden_layer = tf.add(tf.matmul(self.x, self.weights['hidden']),\
                                   self.biases['hidden'])
        self.hidden_layer = tf.nn.relu(self.hidden_layer)
        self.output_layer = tf.matmul(self.hidden_layer, \
                                      self.weights['output']) \
                                      + self.biases['output']
        # self.output_layer = tf.nn.relu(self.output_layer)

        self.cost = tf.reduce_sum(tf.square(self.output_layer - self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(\
                                               self.cost)
        self.saver = tf.train.Saver()

        self.init = tf.initialize_all_variables()    
        self.sess = tf.Session()
        
        self.sess.run(self.init)
        
    def forward(self, state):
        self.A2 = self.sess.run([self.output_layer],feed_dict={self.x:state.\
                                reshape((1, self.in_num_units))})
        
            
    def backward(self, state, label):
        for i in range(0, self.epochs):
            _,c = self.sess.run([self.optimizer, self.cost], feed_dict = {\
                                self.x:state.reshape((1, self.in_num_units)),\
                                self.y:label.reshape((1, self.out_num_units))})
            
    def save(self):
        
        self.save_path = self.saver.save(self.sess, \
                                         "../model_saved/model.ckpt")
        print("Model saved in path: %s" % self.save_path)
        
        #self.sess.close()
        
        return True
        
    def load(self):
        
        self.sess = tf.Session()
        
        self.saver.restore(self.sess, "../model_saved/model.ckpt")
        print("Model restored.")
        
        #self.sess.close()
        
        return True  
import numpy as np
import tensorflow as tf
from collections import OrderedDict
import pandas as pd
import random
import matplotlib.pyplot as plt

class DeepQNetwork:
	def __init__(
		self,
		n_actions,
		n_features,
		learnig_rate = 0.01,
		reward_decay = 0.9,
		e_greedy = 0.9,
		memory_size = 500,
		replace_target_iter = 300,
		epsilon_increment = None,
		batch_size = 32,
		):
		self.n_actions = n_actions;
		self.n_features = n_features;
		self.learnig_rate = learnig_rate;
		self.gamma = reward_decay;
		self.epsilon_max = e_greedy;
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.memory = np.array((self.memory_size,n_features*2+2))
		self.replace_target_iter = replace_target_iter

		self.epsilon_increment = epsilon_increment
		self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max


	###create a session

		self.learn_step_counter = 0
		self.cost_hist = list()
		self.sess = tf.Session()

		self._build_net()
		e_params = tf.get_collection('eval_net_params')
		t_params = tf.get_collection('target_net_params')
		self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]
	
		self.sess.run(tf.global_variables_initializer())
	def _build_net(self):
		###eval
		self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s')
		self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name='Q_target')
		with tf.variable_scope('eval_net'):
			c_names = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES]
			n_l1 = 10
			w_initializer = tf.random_normal_initializer(0.0,0.3)
			b_initializer = tf.constant_initialzier(0.1)

			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
				b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s,w1)+b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[self.n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
				b2 = tf.get_variable('b1',[1,self.n_actions],initializer=b_initializer,collections=c_names)
				self.q_eval = tf.matmul(l1,w2) + b2

			with tf.variable_scope('loss'):
				self.loss = tf.reduce_mean(tf.squared_difference(self.q_eval,self.q_target))
			with tf.variable_scope('train'):
				self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s')

		with tf.variable_scope('target_net'):
			c_naemes = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]


			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
				b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_,w1)+b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[self.n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
				b2 = tf.get_variable('b1',[1,self.n_actions],initializer=b_initializer,collections=c_names)
				self.q_next = tf.matmul(l1,w2) + b2

	def store_transition(self,s,a,s_,r):
		if not hasattr(self,'memory_counter'):
			self.memory_counter = 0

		index = self.memory_counter % self.memory.memory_size
		self.memory[index,:] = np.array(s,a,r,s_)
		self.memory_counter += 1

	def choose_action(self,state):
		q_val = self.sess.run(self.q_eval,feed_dict={self.s:state})
		a_max = np.argmax(q_val)
		if np.random.uniform() > e_greedy:
			a_max = np.random.randint(0,self.n_actions)
		return a_max

	def learn(self):
		if (self.learn_step_counter % self.replace_target_iter == 0):
			self.sess.run(self.replace_target_op)
			print("\ntaget params updated \n")
		if (self.memory_counter > self.memory_size):
			batch_index = np.random.choice(self.memory_counter,self.batch_size)
		else:
			batch_index = np.random.choice(self.memory_size,self.batch_size)

		batch_memory = self.memory[batch_index,:]
		q_next = self.sess.run(self.q_next,feed_dict={self.s_:batch_memory[:,-n_features:]})
		q_eval = self.sess_run(self.q_eval,fedd_dict={self.s:batch_memory[:,:n_features]})


		q_target = q_eval.copy()
		eval_act_index = batch_memory[:,self.n_features].astype(int)
		reward = batch_memory[:,self.n_features+1]
		indexs = np.arange(batch_size,dtype=np.int32)
		q_target[indexs,eval_act_index] = reward + self.gamma*np.max(q_next,axis=1)

		###
		_,self.cost = self.sess.run(self._train_op,self.loss,feed_dict={self.s:batch_memory[:,n_features:],self.q_target:q_target})

		self.cost_hist.append(his)
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1

	def plot_lost(self):
		plt.plot(self.cost_hist)
		plt.show()







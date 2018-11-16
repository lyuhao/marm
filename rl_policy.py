import numpy as np
import tensorflow as tf
from collections import OrderedDict
import pandas as pd
import random
import matplotlib.pyplot as plt


np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
	def __init__(
		self,
		n_actions,
		n_features,
		name,
		learning_rate = 0.001,
		reward_decay = 0.99,
		output_graph = False,
		num_traj = 50.0
		):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.ep_obs, self.ep_as, self.ep_rs = [], [], []
		self.reward_hist = []
		self.cost_hist = []
		self.name = name
		self.num_traj = num_traj
		self.Vs  = dict() ## a book keeping structure for keeping track of baseline term 

	## build net
		self._build_net()
		self.sess = tf.Session()

		self.sess.run(tf.global_variables_initializer())

	def _build_net(self):
		with tf.name_scope(self.name+"_"+"inputs"):
			self.tf_obs = tf.placeholder(tf.float32,[None,self.n_features],name='observations')
			self.tf_acts = tf.placeholder(tf.int32,[None,],name='actions_num')
			self.tf_vt = tf.placeholder(tf.float32,[None,],name='actions_value')

		self.layer = tf.layers.dense(inputs=self.tf_obs,
			units=self.n_features/2,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),
			name=self.name+"_"+'fc1')
		# layers = tf.layers.conv1d(inputs=self.tf_obs,kernel_size=2,strides=1,filters=1,activation=tf.nn.tanh,kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
		# 	name=self.name+"_"+'conv1')

		self.all_act = tf.layers.dense(inputs=self.layer,units=self.n_actions,activation=None,kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.1),name=self.name+"_"+'fc2')

		self.all_act_prob = tf.nn.softmax(self.all_act, name=self.name+"_"+'act_prob')

		with tf.name_scope(self.name+"_"+'loss'):
			neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts)
			self.mean_log = tf.reduce_mean(neg_log_prob)
			self.loss = tf.reduce_sum(neg_log_prob * self.tf_vt)

		with tf.name_scope(self.name+"_"+'train'):
			self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

 	def choose_action(self, observation): 
		prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
		#print(prob_weights)
		action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
 		return action
 	def store_transition(self, s, a, r):
 		#print("aaa")
 		self.ep_obs.append(s)
 		self.ep_as.append(a)
 		self.ep_rs.append(r)
 		
 	def learn(self):
 		if(len(self.ep_obs) == 0):
 			return
 		discounted_ep_rs_norm,mean = self._discount_and_norm_rewards()
 		#print(self.ep_obs)
 		#print("____")
 		#print("~~~")
 		#for ep in self.ep_obs:
 		#	print(len(ep))
 		_ ,loss,mean_log = self.sess.run([self.train_op,self.loss,self.mean_log],feed_dict={
 			self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
 			self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
 			self.tf_vt: discounted_ep_rs_norm+mean,  # shape=[None, ]
 		})
 		#print(loss)
 		self.cost_hist.append(loss)
 		

 		self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
 		return discounted_ep_rs_norm+mean

 	def _discount_and_norm_rewards(self):
 		#print(self.ep_rs)
 		discounted_ep_rs = np.zeros_like(self.ep_rs,dtype=float)
 		running_add = 0
 		#print(self.ep_rs)
 		for t in reversed(range(0, len(self.ep_rs))):
 			running_add = running_add * self.gamma + self.ep_rs[t]
 			discounted_ep_rs[t] = running_add
 			obs = tuple(list(self.ep_obs[t])+[t])
 			if(obs not in self.Vs):
 				self.Vs[obs] = [discounted_ep_rs[t],1]
 			else:

 				self.Vs[obs][0] = (self.Vs[obs][0] * self.Vs[obs][1] + discounted_ep_rs[t])/(self.Vs[obs][1] + 1)
 				self.Vs[obs][1] += 1
 				discounted_ep_rs[t] -= self.Vs[obs][0]
 				#discounted_ep_rs[t] = max(discounted_ep_rs[t],0)
 			#discounted_ep_rs[t]/= self.num_traj
 		#print(len(discounted_ep_rs))

 		self.reward_hist.append(discounted_ep_rs[-1])
 		#discounted_ep_rs_norm = discounted_ep_rs
 		val = np.mean(discounted_ep_rs)
 		discounted_ep_rs -= np.mean(discounted_ep_rs)
 		#print("hhhhpython")
 		#print(discounted_ep_rs)
 		#print(np.std(discounted_ep_rs))
 		#if(np.std(discounted_ep_rs) != 0):
 		#	discounted_ep_rs /= np.std(discounted_ep_rs)

 		#print(self)
 		return discounted_ep_rs,val

 	def get_variable(self,name):
 		for val in tf.global_variables():
 			if val.name.startswith(name):
 				return val
 	def get_weights(self):
 		layer1 = self.get_variable(self.name+"_"+"fc1")
 		layer2 = self.get_variable(self.name+"_"+"fc2")
 		return layer1.eval(session=self.sess),layer2.eval(session=self.sess)
 	def getVs(self):
 		return self.Vs
 	def plot_lost(self):
 		#print(self.cost_hist)
 		#plt.plot(self.cost_hist)
 		#plt.plot(self.reward_hist)
 		return self.cost_hist
 		#plt.show()

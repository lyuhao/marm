#!/usr/bin/python
from rl_network import DeepQNetwork
from rl_policy import PolicyGradient
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import tensorflow as tf


TaskQueue_Size = 5
TaskQueue = list()
ExecutionList = list()

## re[0] cpu re[1] mem
resource_pool = np.zeros(2)
resource_pool[0] = 1.0
resource_pool[1] = 1.0


short_task = 0
long_task = 0

num_of_traj = 10
global_id = 0
Pgt = PolicyGradient(n_actions = TaskQueue_Size+1,
					 n_features = TaskQueue_Size*2+2,
					 learning_rate = 0.001,num_traj = num_of_traj,name="cloudlet")

DQN = DeepQNetwork(n_actions = TaskQueue_Size+1, n_features = TaskQueue_Size*2+2,learning_rate = 1e-3)

class Task:
	def __init__(self,
				execution_Time,
				cpu_demand,
				memory_demand,
				response_time,
				deadline,
				id):
		self.execution_Time = execution_Time
		self.cpu_demand = cpu_demand
		self.memory_demand = memory_demand
		self.deadline =deadline
		self.response_time = response_time
		self.id = id

def genshortTask(id):
	et = np.random.choice([1,2,3])
	cpu_demand = np.random.uniform(0.01,0.1)
	task = Task(execution_Time = et, cpu_demand=cpu_demand,memory_demand=cpu_demand,response_time = 1, deadline=100,id=id)
	return task

def genlongTask(id):
	et = np.random.choice([5,6,7,8])
	cpu_demand = np.random.uniform(0.15,0.2)
	task = Task(execution_Time = et, cpu_demand=cpu_demand,memory_demand=cpu_demand,response_time = 1, deadline=150,id=id)
	return task

def genTask(l):
	global global_id
	#s = int(np.random.poisson(l,1))
	s = 0
	for i in range(l):
		s += np.random.choice([0,1])
	#print(s)
	for i in range(s):
		if(np.random.uniform() < 0.8):
			task = genshortTask(global_id)
		else:
			task = genlongTask(global_id)
		#if(len(TaskQueue) < TaskQueue_Size):
		TaskQueue.append(task)
		global_id +=1

def rewardsignal(EQ):
	count = len(TaskQueue)#+len(ExecutionList)
	Sum = 0.0
	for T in TaskQueue:
		Sum +=  T.response_time
	#for task in ExecutionList:
	#	Sum += task.response_time
	#for et in EQ:
	#	if et.execution_Time == 0:
	#		print("removing!!!")
	#		EQ.remove(et)
	#		resource_pool[0] += et.cpu_demand
	#		resource_pool[1] += et.memory_demand
	#		count += 1
	if(count == 0.0):
		count = 0.5
	return -Sum, 1.0/count

def progress(TQ,EQ):
	total_response = 0.0
	count = 0.0
	for pd in TQ:
		pd.response_time += 1
	for et in EQ:
		#print et.execution_Time
		assert et.execution_Time >= 0
		et.response_time += 1
		et.execution_Time -= 1
		if(et.execution_Time == 0):
			EQ.remove(et)
			resource_pool[0] += et.cpu_demand
			resource_pool[1] += et.memory_demand
			total_response+=et.response_time
			count+=1
	return total_response,count

def execution(action):
	if(action >= min(len(TaskQueue),TaskQueue_Size)):
		return 0


	task = TaskQueue[action]
	if task.memory_demand > resource_pool[1]:
		return 0
	#if task.cpu_demand <= resource_pool[0] and task.memory_demand <= resource_pool[1]:
		
	TaskQueue.remove(task)
	resource_pool[0] -= task.cpu_demand
	resource_pool[1] -= task.memory_demand
	ExecutionList.append(task)
	global short_task
	global long_task
	if(task.execution_Time == 1):
		short_task += 1
	else:
		long_task += 1
	return 1
	#return False
def getObservation():
	observation = list()
	executable_task = list()
	#for task in TaskQueue:
		#if task.cpu_demand <= resource_pool[0] and task.memory_demand <= resource_pool[1]:
	#	if task.memory_demand <= resource_pool[1]:
	#		executable_task.append(task)

	for i in range(min(len(TaskQueue),TaskQueue_Size)):
		observation.append(TaskQueue[i].memory_demand)
		observation.append(TaskQueue[i].execution_Time/10.0)

	size = len(TaskQueue)
	for i in range(max(TaskQueue_Size - size,0)):
		for j in range(2):
			observation.append(0.0)
	left_num = max(len(TaskQueue)-TaskQueue_Size,0)
	#print(left_num)
	##observation.append(left_num)
	#observation.append(resource_pool[0])
	observation.append(resource_pool[1])
	observation.append(min(left_num,15)/15.0)
	return np.array(observation)

#rw = []
#ct = []
def train():

	global num_of_traj
	obs = [list()]*num_of_traj
	acts = [list()]*num_of_traj
	rws = [list()]*num_of_traj
	
	np.random.seed(1)
	for num in range(num_of_traj):

		iteration = 50
		avgs = []
		counts = []
		actions = [0]*TaskQueue_Size
		rewards = [0]*TaskQueue_Size
		global resource_pool
		global TaskQueue
		global ExecutionList
		resource_pool[0] = 1.0
		resource_pool[1] = 1.0
		TaskQueue = list()
		ExecutionList = list()
			#np.random.seed(ID)
		resp = 0.0
		count = 0.0
		for times in range(iteration):
			resp_new,count_new = progress(TaskQueue,ExecutionList)
			resp += resp_new
			count += count_new
			genTask(2)
			#print("here!!!!")
			#if(len(TaskQueue) > 15):
				#print(times,"break")
			#	break
			while(True):
				observation = getObservation()

				action = Pgt.choose_action(observation)
				#for i in range(min(len(exec_list),10)):
				#	print(exec_list[i].execution_Time)
				#print(len(exec_list),action,resource_pool[0],resource_pool[1])
				#print(action,len(TaskQueue),resource_pool[0],resource_pool[1])
			#	action = 1
				#print(action)
				actions[min(action,TaskQueue_Size-1)] += 1
				result = execution(action) 


				reward,average = rewardsignal(ExecutionList)
				avgs.append(average)

				counts.append(average)

				reward = average - 0.1
				if(reward < 0):
					reward = reward*10
				rewards[min(action,TaskQueue_Size-1)] += reward



				obs[num].append(observation)
				acts[num].append(action)
				rws[num].append(reward)
				if(result == 0):
					break

		#print(resp,count)
	rewards = []
	for num in range(num_of_traj):
		for i in range(len(obs[num])):
			Pgt.store_transition(obs[num][i],acts[num][i],rws[num][i])
		#print("begin learn")
		#l1,l2 = Pgt.get_weights()
		#print(l2)
		Vt = Pgt.learn()
		#print("after learn")
		#l1,l2 = Pgt.get_weights()
		#print(l2)
	rewards.append(Vt[0])
	#rw.append(sum(avgs)/len(avgs))

	return actions,rewards



def eval_policy():
	iteration = 2
	counts = []
	actions = [0]*(TaskQueue_Size+1)
	for iter in range(iteration):
		iteration = 300
		np.random.seed(10)
		global resource_pool
		global TaskQueue
		global ExecutionList
		resource_pool[0] = 1.0
		resource_pool[1] = 1.0
		TaskQueue = list()
		ExecutionList = list()

		#cts = []
		resp = 0.0
		count = 0.0
		for times in range(iteration):
			resp_new,cnt_new = progress(TaskQueue,ExecutionList)
			resp += resp_new
			count += cnt_new
			genTask(2)
			while(True):
				observation = getObservation()
				action = Pgt.choose_action(observation)
				actions[action]+=1	
				#print(action)
				#print(times,action)
				result = execution(action) 
				reward,average = rewardsignal(ExecutionList)
				
				if(result == 0):
						#print(times,action)
						#cts.append(1.0/average)
						break
		#print resp,count
		counts.append(resp/count)

		#print(counts)
	return sum(counts)/len(counts),actions	
def run():
	iteration = 500
	rw = []
	counts = []
	for ite in range(iteration):
		#np.random.seed(7)
		actions,rewards = train()
		l1,l2 = Pgt.get_weights()
		#print(l1)
		#print("---------")
		#print(l2)

		#rewards = [rewards[x]/max(actions[x],1) for x in range(len(actions))]
		ma = zip(actions,rewards)
		#print(ite,ma)
		#rw.append(reward)
		average_count,actions = eval_policy()
		print(ite,sum(rewards)/len(rewards),average_count,actions)
		print("\n")
		counts.append(average_count)


	for i in range(300):
		obs = np.random.random_sample(12)*16
		action = Pgt.choose_action(obs)
		print(action)
	traj = []
	global resource_pool
	global TaskQueue
	global ExecutionList
	resource_pool[0] = 1.0
	resource_pool[1] = 1.0
	TaskQueue = list()
	ExecutionList = list()
	# np.random.seed(10)
	# for times in range(300):
	# 	progress(TaskQueue,ExecutionList)
	# 	genTask(2)
	# 	for task in TaskQueue:
	# 		if(task.memory_demand <= resource_pool[1]):
	# 				TaskQueue.remove(task)
	# 				resource_pool[0] -= task.cpu_demand
	# 				resource_pool[1] -= task.memory_demand
	# 				ExecutionList.append(task)
	# 	string = ""
	# 	for task in TaskQueue:
	# 		string+=' '+str(task.memory_demand)
	# 	print(string)
	# 	traj.append(len(TaskQueue))
	#layer1,layer2 = Pgt.get_weights()
	#print(layer1)
	#print("-----------")
	#print(layer2)


	# plt.plot(traj)
	#cost = Pgt.plot_lost()
	#plt.plot(cost)
	# plt.show()


def run_train():
	rw = []
	ct = []
	iteration = 5000
	for times in range(iteration):
		progress(TaskQueue,ExecutionList)
		genTask(2)
		#print("here!!!!")
		avgs = []
		counts = []
		while(True):
			exec_list,observation = getObservation()

			action = Pgt.choose_action(observation)
			#for i in range(min(len(exec_list),10)):
			#	print(exec_list[i].execution_Time)
			#print(len(exec_list),action,resource_pool[0],resource_pool[1])
			#print(action,len(TaskQueue),resource_pool[0],resource_pool[1])
		#	action = 1
			result = execution(action,exec_list) 

			#if not result: 
			#	action = 0
			reward,average = rewardsignal(ExecutionList)
			avgs.append(average)
			#if(average == 0):
			#	average = 1
			counts.append(average)
			#print(reward)
			#for task in ExecutionList:
			#	print (task.execution_Time)
			#print(action,reward)
			if(len(exec_list) == 0):
				action = 1
			Pgt.store_transition(observation,action,result)
			if(result == 0):
				#print("break!")
				break

		if((times+1) % 20 == 0):
			#print(reward)
			#print("time: "+str(time))
			#start = time.time()
			#print(sum(avgs)/len(avgs))
			rw.append(sum(avgs)/len(avgs))
			ct.append(sum(counts)/len(counts))
			print(str((times+1)/20)+' '+str(sum(counts)/len(counts))+' '+str(short_task)+' '+str(long_task))
			avgs = []
			counts = []
			vt = Pgt.learn()
			string=""
			for task in TaskQueue:
				string += str(task.id)+' '
			#print(string)
			#print(str((times+1)/30)+" "+str(vt))
			#print(time.time() - start)

	#print("here")
	##DQN.plot_lost()
	#Pgt.plot_lost()
	#plt.plot(rw)
	#plt.plot(ct,color='red')
	cost = Pgt.plot_lost()
	plt.plot(cost)
	plt.show()



if __name__ == "__main__":
	run()
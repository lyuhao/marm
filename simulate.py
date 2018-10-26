#!/usr/bin/python
from rl_network import DeepQNetwork
from rl_policy import PolicyGradient
import numpy as np
import random
import matplotlib.pyplot as plt
import time

TaskQueue_Size = 10
TaskQueue = list()
ExecutionList = list()

## re[0] cpu re[1] mem
resource_pool = np.zeros(2)
resource_pool[0] = 10
resource_pool[1] = 30


global_id = 0
Pgt = PolicyGradient(n_actions = TaskQueue_Size+1,
					 n_features = TaskQueue_Size*4+2,
					 learning_rate = 0.01)

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
	task = Task(execution_Time = 1, cpu_demand=1,memory_demand=2,response_time = 1, deadline=100,id=id)
	return task

def genlongTask(id):
	task = Task(execution_Time = 3, cpu_demand=2,memory_demand=4,response_time = 1, deadline=150,id=id)
	return task

def genTask(l):
	global global_id
	s = int(np.random.poisson(l,1))
	for i in range(s):
		if(np.random.uniform() < 0.5):
			task = genshortTask(global_id)
		else:
			task = genlongTask(global_id)
		if(len(TaskQueue) < TaskQueue_Size):
			TaskQueue.append(task)
			global_id +=1

def rewardsignal(EQ):
	count = len(TaskQueue)
	Sum = 0.0
	for T in TaskQueue:
		Sum +=  T.response_time
	#for et in EQ:
	#	if et.execution_Time == 0:
	#		print("removing!!!")
	#		EQ.remove(et)
	#		resource_pool[0] += et.cpu_demand
	#		resource_pool[1] += et.memory_demand
	#		count += 1
	return -Sum, Sum/count

def progress(TQ,EQ):
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

def execution(action):
	if(action > len(TaskQueue) or action == 0):
		return False

	task = TaskQueue[action-1]
	if task.cpu_demand < resource_pool[0] and task.memory_demand < resource_pool[1]:
		TaskQueue.remove(task)
		resource_pool[0] -= task.cpu_demand
		resource_pool[1] -= task.memory_demand
		ExecutionList.append(task)
		return True
	return False
def getObservation():
	observation = list()

	for task in TaskQueue:
		observation.append(task.execution_Time)
		observation.append(task.cpu_demand)
		observation.append(task.memory_demand)
		observation.append(task.response_time)
	size = len(TaskQueue)
	for i in range(TaskQueue_Size-size):
		for j in range(4):
			observation.append(0)
	observation.append(resource_pool[0])
	observation.append(resource_pool[1])
	return np.array(observation)
def run():
	rw = []
	iteration = 30000
	for times in range(iteration):
		progress(TaskQueue,ExecutionList)
		genTask(5)
		#print("here!!!!")
		avgs = []

		while(True):
			observation = getObservation()
			action = Pgt.choose_action(observation)
			#print(action,len(TaskQueue),resource_pool[0],resource_pool[1])
			result = execution(action) 
			if not result:
				action = 0
			reward,average = rewardsignal(ExecutionList)
			avgs.append(average)
			#print(reward)
			#for task in ExecutionList:
			#	print (task.execution_Time)
			#print(action)
			if(action == 0):
				break
			Pgt.store_transition(observation,action,reward)

		if((times+1) % 30 == 0):
			#print(reward)
			#print("time: "+str(time))
			#start = time.time()
			#print(sum(avgs)/len(avgs))
			rw.append(sum(avgs)/len(avgs))
			avgs = []
			print(str((times+1)/30))
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
	plt.plot(rw)
	plt.show()



if __name__ == "__main__":
	run()
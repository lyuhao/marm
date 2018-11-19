from rl_network import DeepQNetwork
from rl_policy import PolicyGradient
import numpy as np
import random
import matplotlib.pyplot as plt
import time
global_id = 0


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
	TaskQueue = list()
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
	return TaskQueue

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
		self.iteration = 0
		self.time = 0


class Cloudlet:
	def __init__(self,cpu_capacity,mem_capacity,TaskQueue_Size,neighbours,id):
		self.cpu_capacity = cpu_capacity
		self.mem_capacity = mem_capacity
		self.resource_pool = np.zeros(2)
		self.resource_pool[0] = self.cpu_capacity
		self.resource_pool[1] = self.mem_capacity
		self.TaskQueue_Size =TaskQueue_Size
		self.TaskQueue = list()
		self.routedTasks = list()
		self.ExecutionList = list()
		self.rwlist = list()
		self.avgs = list()
		self.neighbours = neighbours
		self.id = id
		self.obs_local = []
		self.acts_local = []
		self.rws_local = []
		self.obs_route = []
		self.acts_route = []
		self.rws_route = []
		#print(len(neighbours))
		self.Pgt = PolicyGradient(n_actions = TaskQueue_Size+1,
					 n_features = TaskQueue_Size*2+2,
					  learning_rate = 0.001,name='local'+str(self.id))
		self.Pgt_route = PolicyGradient(n_actions = len(self.neighbours)+1,n_features = (TaskQueue_Size*2+2)*len(self.neighbours)+2, learning_rate = 0.001,name='route'+str(self.id))

		self.global_id = 0
		self.iteration = 0

	def init_neighbours(self,cloudlets):
		self.neighbours = [cloudlets[i] for i in self.neighbours]

	def acceptTask(self,task):
		task.response_time += 5
		if(len(self.routedTasks) < self.TaskQueue_Size):
			self.routedTasks.append(task)
			return True
		return False

	def remoutereward(self):
		count = len(self.routedTasks)
		Sum = 0.0
		for T in self.routedTasks:
			Sum += T.response_time

		return -Sum,count
	def rewardsignal(self):
		count = len(self.TaskQueue)
		Sum = 0.0
		for T in (self.TaskQueue + self.ExecutionList):
			Sum +=  T.response_time
		#for et in EQ:
		#	if et.execution_Time == 0:
		#		print("removing!!!")
		#		EQ.remove(et)
		#		resource_pool[0] += et.cpu_demand
		#		resource_pool[1] += et.memory_demand
		#		count += 1
		if(count == 0.0):
			count = 0.5
		return -Sum,1.0/count

	def progress(self):
		total_response = 0.0
		count = 0.0
		for pd in self.TaskQueue:
			pd.response_time += 1
		for t in self.routedTasks:
			t.response_time += 1
		for et in self.ExecutionList:
			#print et.execution_Time
			assert et.execution_Time >= 0
			et.response_time += 1
			et.execution_Time -= 1
			if(et.execution_Time == 0):
				self.ExecutionList.remove(et)
				self.resource_pool[0] += et.cpu_demand
				self.resource_pool[1] += et.memory_demand
				total_response+=et.response_time
				count+=1
			return total_response,count

	def execution(self,action):
		if(action >= min(len(self.TaskQueue)+len(self.routedTasks),self.TaskQueue_Size)):
			return 0

		
		if(action >= len(self.TaskQueue)):
			action -= len(self.TaskQueue)
			task = self.routedTasks[action]
			if  task.memory_demand > self.resource_pool[1]:
				return 0 
			self.routedTasks.remove(task)
			self.resource_pool[0] -= task.cpu_demand
			self.resource_pool[1] -= task.memory_demand
			self.ExecutionList.append(task)
			return 1
		task = self.TaskQueue[action]
		if  task.memory_demand > self.resource_pool[1]:
			return 0 
		self.TaskQueue.remove(task)
		self.resource_pool[0] -= task.cpu_demand
		self.resource_pool[1] -= task.memory_demand
		self.ExecutionList.append(task)
		

		return 1



	def getlocalobs(self):
		observation = list()


		##print(len(self.TaskQueue))
		for k in range(min(len(self.TaskQueue),self.TaskQueue_Size)):
			task = self.TaskQueue[k]
			observation.append(task.execution_Time)
			#observation.append(task.cpu_demand)
			observation.append(task.memory_demand)
			#observation.append(task.response_time)

		#print("------")
		size = len(self.TaskQueue)
		size_routed = len(self.routedTasks)


		#print(size)
		for i in range(min(size_routed,self.TaskQueue_Size-size)):
			task = self.routedTasks[i]
			observation.append(task.execution_Time)
			observation.append(task.memory_demand)

		for i in range(max(self.TaskQueue_Size- size - size_routed,0)):
			for j in range(2):
				observation.append(0.0)
		left_num = max(len(self.TaskQueue)+len(self.routedTasks)-self.TaskQueue_Size,0)
		#observation.append(self.resource_pool[0])
		observation.append(self.resource_pool[1])
		observation.append(min(left_num,15)/15.0)
		return np.array(observation)

	def get_rw(self):
		return self.rws_local[0],self.rws_route[0]

	def train(self):
		#print("train")
		for num in range(len(self.obs_local)):
			#print(num)
			for i in range(len(self.obs_local[num])):
				self.Pgt.store_transition(self.obs_local[num][i],self.acts_local[num][i],self.rws_local[num][i])
			self.Pgt.learn()
			for j in range(len(self.obs_route[num])):
				self.Pgt_route.store_transition(self.obs_route[num][j],self.acts_route[num][j],self.rws_route[num][j])
			self.Pgt_route.learn()

		self.obs_local = []
		self.acts_local = []
		self.rws_local = []
		self.obs_route = []
		self.acts_route = []
		self.rws_route = []
	#def run_onestep(self,id_traj):
	def reset(self):
		self.resource_pool[0] = self.cpu_capacity
		self.resource_pool[1] = self.mem_capacity
	##	self.TaskQueue_Size =TaskQueue_Size
		self.TaskQueue = list()
		self.routedTasks = list()
		self.ExecutionList = list()
		self.rwlist = list()
		self.avgs = list()
		#self.obs_local = []
		#self.acts_local = []
		#self.rws_local = []
		#self.obs_route = []
		#self.acts_route = []
		#self.rws_route = []

	def run_onestep(self,id_traj):
		#print(id_traj)
		if id_traj >= len(self.obs_local):
			self.obs_local.append(list())
			self.rws_local.append(list())
			self.acts_local.append(list())
			self.obs_route.append(list())
			self.acts_route.append(list())
			self.rws_route.append(list())

		self.progress()
		tasklist = genTask(2)
		for task in tasklist:
			self.TaskQueue.append(task)
		### find the tasks execute locally 
		while(True):
			observation = self.getlocalobs()
			action = self.Pgt.choose_action(observation)
			result = self.execution(action) 
			if not result:
				action = 0
			Sum,average = self.rewardsignal()
			#if count == 0:
			#	count = 1
			#self.avgs.append(reward/count)
			#print(reward)
			#for task in ExecutionList:
			#	print (task.execution_Time)
			#print(action)
			reward = average - 0.1
			if(reward < 0):
				reward = reward*10

			#print(id_traj)
			#print(len(self.obs_local))
			self.obs_local[id_traj].append(observation)
			self.acts_local[id_traj].append(action)
			self.rws_local[id_traj].append(reward)
			if(result == 0):
				break
			#self.Pgt.store_transition(observation,action,reward/count)

		#### find the tasks to route
		nb_obs = np.array([])
		for neighbour in self.neighbours:
			##impleentation
			observation = neighbour.getlocalobs()
			nb_obs = np.concatenate((nb_obs,observation),axis=0)


		for task in (self.routedTasks):
			task_obs = list()
			task_obs.append(task.execution_Time)
			#task_obs.append(task.cpu_demand)
			task_obs.append(task.memory_demand) 
			#task_obs.append(task.response_time)
			task_obs = np.array(task_obs)
			total_obs = np.concatenate((nb_obs,task_obs),axis=0)
			action = self.Pgt_route.choose_action(total_obs)

			if(action == 0):
				continue
			neighbour = self.neighbours[action-1]
			flag = neighbour.acceptTask(task)
			if not flag:
				continue
			self.routedTasks.remove(task)
			reward = 0
			count = 0
			for neighbour in self.neighbours:
				r,c = neighbour.remoutereward()
				reward += r
				count += c

			reward = reward / count
			self.obs_route[id_traj].append(total_obs)
			self.acts_route[id_traj].append(action)
			self.rws_route[id_traj].append(reward)
			#self.Pgt_route.store_transition(total_obs,action,reward)



		###only route task
		for task in (self.TaskQueue):
			task_obs = list()
			task_obs.append(task.execution_Time)
			#task_obs.append(task.cpu_demand)
			task_obs.append(task.memory_demand) 
			#task_obs.append(task.response_time)
			task_obs = np.array(task_obs)
			total_obs = np.concatenate((nb_obs,task_obs),axis=0)
			action = self.Pgt_route.choose_action(total_obs)

			if(action == 0):
				continue
			neighbour = self.neighbours[action-1]
			neighbour.acceptTask(task)
			self.TaskQueue.remove(task)
			reward = 0
			count = 0
			for neighbour in self.neighbours:
				r,c = neighbour.remoutereward()
				reward += r
				count += c

			reward = reward / count
			self.obs_route[id_traj].append(total_obs)
			self.acts_route[id_traj].append(action)
			self.rws_route[id_traj].append(reward)
			#self.Pgt_route.store_transition(total_obs,action,reward)
			#self.iteration += 1
			#if(self.iteration%30 == 0):
			#	self.Pgt_route.learn()

			#self.time += 1
			#print(self.time)
		# if((ite+1) % 30 == 0):
		# 	self.Pgt.learn()
		# 	self.rwlist.append(sum(self.avgs)/len(self.avgs))
		# 	self.avgs = []
			#print(reward)
			#print("time: "+str(time))
			#start = time.time()
			#print(sum(avgs)/len(avgs))
			#vt = Pgt.learn()
			#print(string)
				#print(str((times+1)/30)+" "+str(vt))
				#print(time.time() - start)

		
		#plt.plot(rw)
		#plt.show()
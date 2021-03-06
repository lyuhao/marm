from rl_network import DeepQNetwork
from rl_policy import PolicyGradient
import numpy as np
import random
import matplotlib.pyplot as plt
import time
global_id = 0


def genshortTask(id,cloudlet):
    et = np.random.choice([1,2])
    cpu_demand = np.random.uniform(0.01,0.1)
    task = Task(execution_Time = et, cpu_demand=cpu_demand,memory_demand=cpu_demand,response_time = 1, deadline=100,id=id,cloudlet=cloudlet)
    return task

def genlongTask(id,cloudlet):
    et = np.random.choice([5,6])
    cpu_demand = np.random.uniform(0.15,0.2)
    task = Task(execution_Time = et, cpu_demand=cpu_demand,memory_demand=cpu_demand,response_time = 1, deadline=150,id=id,cloudlet=cloudlet)
    return task

def genTask(l,cloudlet):
    global global_id
    #s = int(np.random.poisson(l,1))
    s = 0
    TaskQueue = list()
    #print(l)
    for i in range(3):
        if np.random.uniform() < l:
            s += 1
    #print(s)
    for i in range(s):
        if(np.random.uniform() < 0.8):
            task = genshortTask(global_id,cloudlet)
        else:
            task = genlongTask(global_id,cloudlet)
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
                id,cloudlet):
        self.execution_Time = execution_Time
        self.cpu_demand = cpu_demand
        self.memory_demand = memory_demand
        self.deadline =deadline
        self.response_time = response_time
        self.id = id
        self.iteration = 0
        self.time = 0
        self.origin = cloudlet


class Cloudlet:
    def __init__(self,cpu_capacity,mem_capacity,TaskQueue_Size,neighbours,id,load):
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.resource_pool = np.zeros(2)
        self.resource_pool[0] = self.cpu_capacity
        self.resource_pool[1] = self.mem_capacity
        self.TaskQueue_Size = int(TaskQueue_Size)
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
        self.pending_task = {}
        self.resp = []
        self.load = load
        #print(len(neighbours))
        self.Pgt = PolicyGradient(n_actions = TaskQueue_Size+1,
                     n_features = TaskQueue_Size*2+2,
                      learning_rate = 0.001,name='local'+str(self.id))
        self.Pgt_route = PolicyGradient(n_actions = len(self.neighbours)+1,reward_decay=0, n_features = (TaskQueue_Size*2+2)*len(self.neighbours)+2, learning_rate = 0.001,name='route'+str(self.id))

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
    def getreward(self,task):
        task_id = task.id
        tu = self.pending_task.pop(task_id)
        response_time = task.response_time*1.0
       #print(response_time)
        reward = 1 / response_time  - 1.0/(4.0*self.load*10)
        self.obs_route.append(tu[0])
        self.acts_route.append(tu[1])
        self.rws_route.append(reward)

    def remoutereward(self):
        count = len(self.routedTasks)
        Sum = 0.0
        for T in self.routedTasks:
            Sum += T.response_time

        return -Sum,1.0/count
    def rewardsignal(self):
        count = len(self.TaskQueue)+len(self.routedTasks)
        Sum = 0.0
        for T in (self.TaskQueue):
            Sum +=  T.response_time
        #for et in EQ:
        #   if et.execution_Time == 0:
        #       print("removing!!!")
        #       EQ.remove(et)
        #       resource_pool[0] += et.cpu_demand
        #       resource_pool[1] += et.memory_demand
        #       count += 1
        if(count == 0.0):
            count = 0.5
        return float(Sum),float(Sum)/max(len(self.TaskQueue),1.0)

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
                self.resp.append(et.response_time)
                count+=1
                if(et.origin != self):
                    et.origin.getreward(et)
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


    def isaccept(self):
        return self.routedTasks < TaskQueue
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

        for key in self.pending_task.keys():
            tu = self.pending_task.pop(key)
            self.rws_route.append(- 1.0/(4.0*self.load*10))
        #print(len(self.rws_local[0]),len(self.rws_route))
        a,b,c = self.rws_local[0],self.rws_route,self.resp
        #print(len(self.TaskQueue))
        #print(len(self.routedTasks))
        for et in self.TaskQueue:
            self.resp.append(et.response_time)
        for et in self.routedTasks:
            self.resp.append(et.response_time)
        self.obs_local = []
        self.acts_local = []
        self.rws_local = []
        self.obs_route = []
        self.acts_route = []
        self.rws_route = []
        self.resp = []
        return a,b,c

    def train(self):
        #print("train")
        for num in range(len(self.obs_local)):
            #print(num)
            for i in range(len(self.obs_local[num])):
                self.Pgt.store_transition(self.obs_local[num][i],self.acts_local[num][i],self.rws_local[num][i])
            self.Pgt.learn()
        for j in range(len(self.obs_route)):
            self.Pgt_route.store_transition(self.obs_route[j],self.acts_route[j],self.rws_route[j])
        for key in self.pending_task.keys():
            tu = self.pending_task.pop(key)
            self.Pgt_route.store_transition(tu[0],tu[1],-1.0)
        self.Pgt_route.learn()

        self.obs_local = []
        self.acts_local = []
        self.rws_local = []
        self.obs_route = []
        self.acts_route = []
        self.rws_route = []
        self.resp = []
    #def run_onestep(self,id_traj):
    def reset(self):
        self.resource_pool[0] = self.cpu_capacity
        self.resource_pool[1] = self.mem_capacity
    ##  self.TaskQueue_Size =TaskQueue_Size
        #print(len(self.TaskQueue))
        self.TaskQueue = list()
        self.routedTasks = list()
        self.ExecutionList = list()
        self.rwlist = list()
        self.avgs = list()
        self.pending_task = {}


        #self.Pgt_route.store_transition(tu[0],tu[1],-1.0)
        #self.obs_local = []
        #self.acts_local = []
        #self.rws_local = []
        #self.obs_route = []
        #self.acts_route = []
        #self.rws_route = []

    def run_onestep(self,id_traj,ifgen):

        if id_traj >= len(self.obs_local):
            self.obs_local.append(list())
            self.rws_local.append(list())
            self.acts_local.append(list())

        #print(len(self.TaskQueue))
        self.progress()
        if ifgen:
        	tasklist = genTask(self.load,self)
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

            reward = 1.0/max(average,1.0) - 1.0/(4.0*self.load*10)
            if(reward < 0):
                reward = reward * 10


            self.obs_local[id_traj].append(observation)
            self.acts_local[id_traj].append(action)
            self.rws_local[id_traj].append(reward)
            if(result == 0):
                break


        #### find the tasks to route
        nb_obs = np.array([])
        for neighbour in self.neighbours:
            ##impleentation
            observation = neighbour.getlocalobs()
            nb_obs = np.concatenate((nb_obs,observation),axis=0)





        ###only route task

        neis = []
        for neighbour in self.neighbours:
            if(neighbour.isaccept):
                neis.append(neighbour)
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
            flag = neighbour.acceptTask(task)
            if(flag):
                self.TaskQueue.remove(task)
                reward = 0
                count = 0
                self.pending_task[task.id] = (total_obs,action)


        return self.TaskQueue_Size < 15
        #plt.plot(rw)
        #plt.show()

import numpy as np
import random
import matplotlib.pyplot as plt
import time

global_id = 0


def genshortTask(id):
    et = np.random.choice([1,2])
    cpu_demand = np.random.uniform(0.01,0.1)
    task = Task(execution_Time = et, cpu_demand=cpu_demand,memory_demand=cpu_demand,response_time = 1, deadline=100,id=id)
    return task

def genlongTask(id):
    et = np.random.choice([5,6])
    cpu_demand = np.random.uniform(0.15,0.2)
    task = Task(execution_Time = et, cpu_demand=cpu_demand,memory_demand=cpu_demand,response_time = 1, deadline=150,id=id)
    return task

def genTask(l):
    global global_id
    #s = int(np.random.poisson(l,1))
    s = 0
    TaskQueue = list()
    for i in range(3):
        if np.random.uniform() < l:
            s += 1
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
    def __init__(self,cpu_capacity,mem_capacity,TaskQueue_Size,neighbours,id,load):
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
        self.resp = []
        #print(len(neighbours))

        self.global_id = 0
        self.iteration = 0
        self.load = load

    def init_neighbours(self,cloudlets):
    
        self.neighbours = [cloudlets[i] for i in self.neighbours]

    def acceptTask(self,task):
        #self.routedTasks = 
        task.response_time += 5
        if(len(self.routedTasks) < self.TaskQueue_Size):
            self.routedTasks.append(task)
            return True
        return False

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
            return total_response,count

    def execution(self,q,action):
        if(action >= min(len(self.TaskQueue)+len(self.routedTasks),self.TaskQueue_Size)):
            return 0

        task = q[action]


        if  task.memory_demand > self.resource_pool[1]:
                return 0 
        if task in self.routedTasks:
            self.routedTasks.remove(task)
        else:
            self.TaskQueue.remove(task)
        q.remove(task)
        self.resource_pool[0] -= task.cpu_demand
        self.resource_pool[1] -= task.memory_demand
        self.ExecutionList.append(task)
        return 1

        #return 1



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




    #def run_onestep(self,id_traj):
    def reset(self):
        self.resource_pool[0] = self.cpu_capacity
        self.resource_pool[1] = self.mem_capacity
    ##  self.TaskQueue_Size =TaskQueue_Size
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
        

####################################################
    def run_onestepSJF(self,ifgen):


        self.progress()
        if ifgen:
            tasklist = genTask(self.load)
            for task in tasklist:
                self.TaskQueue.append(task)
                        
         
        #key1= self.TaskQueue.execution_Time
        q = self.TaskQueue+self.routedTasks    
        #self.TaskQueue.sort(key=lambda x: x.response_time,reverse=True)
        #self.routedTasks.sort(key=lambda x: x.response_time,reverse=True)
        q.sort(key=lambda x:x.response_time,reverse=True)
        action = 0
        ### find the tasks execute locally 
        while(True):
            #observation = self.getlocalobs()
            result = self.execution(q,action)

                        #if not result:
                            #action = 0
                        
            #self.obs_local[id_traj].append(observation)
            #self.acts_local[id_traj].append(action)
            if(result == 0):
                break
            
                        
        #### find the tasks to route
        self.TaskQueue.sort(key=lambda x:x.execution_Time)
        for task in (self.TaskQueue):
                 
            flag = False
           # print(self.neighbours)
           # print("hhhhh")
            random.shuffle(self.neighbours)
            for neighbour in self.neighbours:
                flag = neighbour.acceptTask(task)
                if(flag):
                    break
            #flag = neighbour.acceptTask(task)
            if not flag:
                continue
            self.TaskQueue.remove(task)
            #self.obs_route[id_traj].append(total_obs)
            #self.acts_route[id_traj].append(action)
             #   action -=1
    def get_rw(self):


        #print(len(self.rws_local[0]),len(self.rws_route))
        # for task in self.ExecutionList:
        #     self.resp.append(task.response_time)
        # for task in self.TaskQueue:
        #     self.resp.append(task.response_time)
        a = self.resp
        self.resp = []
        return a
##################################################################
##################################################################
    def run_onestepFCFS(self,ifgen):

        self.progress()
        if ifgen:
            tasklist = genTask(self.load)
            for task in tasklist:
                self.TaskQueue.append(task)
                        
        #print(self.TaskQueue) 
        #key1= self.TaskQueue.execution_Time
                
        #self.TaskQueue.sort(key=lambda x: x.execution_Time)
                
        q = self.routedTasks+self.TaskQueue           
        action = 0
        ### find the tasks execute locally 
        while(True):
            #observation = self.getlocalobs()
            result = self.execution(q,action)
                        #if not result:
                            #action = 0               
            #self.obs_local[id_traj].append(observation)
            #self.acts_local[id_traj].append(action)
            if(result == 0):
                break
            
                        
        #### find the tasks to route
            for task in self.TaskQueue[::-1]:
                flag = False
                random.shuffle(self.neighbours)
                for neighbour in self.neighbours:
                    flag = neighbour.acceptTask(task)
                    if(flag):
                        break
             #flag = neighbour.acceptTask(task)
                if not flag:
                    continue
                self.TaskQueue.remove(task)
            #self.obs_route[id_traj].append(total_obs)
            #self.acts_route[id_traj].append(action)
             #   action -=1


##################################################################
        

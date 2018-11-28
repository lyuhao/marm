#from rl_network import DeepQNetwork
#from rl_policy import PolicyGradient
#!/usr/bin/python3
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import sys
import  optparse
from util import Cloudlet





parser = optparse.OptionParser()
parser.add_option('-f', '--file',dest="file",
    help="adjacent matrix", default="adjacent.txt")
parser.add_option('-c', '--config',dest="config",
    help="config matrix", default="config.txt")
#parser.add_option('-n','--num',dest="num",help="number of cloudlets",default="4")

options, args = parser.parse_args()


cloudlet_file_name = options.file
cloudlet_config_name = options.config
cloudlet_file = open(cloudlet_file_name,'r')
config_file = open(cloudlet_config_name,'r')
configs = list()
neighbours = list()
for line in cloudlet_file.readlines():
	neighbours.append([int(x) for x in line.split(' ')])

cloudlet_file.close()


for config in config_file.readlines():
	configs.append([float(x) for x in config.split(' ')])

config_file.close()

number_of_cloudlet = len(neighbours)

cloudlet_list = list()

for i in range(number_of_cloudlet):
	config = configs[i]
	cloudlet_list.append(Cloudlet(1.0,1.0,config[0],neighbours[i],id=i,load=config[1]))


max_iter = 100
for i in range(number_of_cloudlet):
	cloudlet_list[i].init_neighbours(cloudlet_list)

training_iter = 500
num_of_traj = 5
for j in range(training_iter):
	#print("iter "+str(j))
	for traj in range(num_of_traj):
		np.random.seed(2)
		#print("traj "+str(traj))
		for i in range(max_iter):
			flag = True
			#random.shuffle(cloudlet_list)
			for cloudlet in cloudlet_list:
				flag_temp = cloudlet.run_onestep(traj,True)
				flag = flag and flag_temp
			if not flag:
				break
		for cloudlet in cloudlet_list:
			cloudlet.reset()
	for cloudlet in cloudlet_list:
		cloudlet.train()

## eval 
	np.random.seed(1)
	for i in range(200):
		#random.shuffle(cloudlet_list)
		for cloudlet in cloudlet_list:
			cloudlet.run_onestep(0,True)
	for i in range(50):
		#random.shuffle(cloudlet_list)
		for cloudlet in cloudlet_list:
			cloudlet.run_onestep(0,False)
	for cloudlet in cloudlet_list:
		cloudlet.reset()
	rw0,rw0_,rsp0 = cloudlet_list[0].get_rw()
	rw1,rw1_,rsp1 = cloudlet_list[1].get_rw()
	rsp = rsp0+rsp1
	#print(rsp0,rsp1)
	print(j,sum(rw0)/len(rw0),sum(rw0_)/max(len(rw0_),1),sum(rw1)/len(rw1),sum(rw1_)/max(len(rw1_),1),sum(rsp0)*1.0/max(len(rsp0),1),sum(rsp1)*1.0/max(len(rsp1),1))

#rw0 = cloudlet_list[0].get()
#rw1 = cloudlet_list[1].get()
#plt.plot(rw0,color='red')
#plt.plot(rw1,color='blue')
#plt.show()
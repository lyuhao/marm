#from rl_network import DeepQNetwork
#from rl_policy import PolicyGradient
#!/usr/bin/python3
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import sys
import  optparse
from util_comp import Cloudlet





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


for i in range(number_of_cloudlet):
	cloudlet_list[i].init_neighbours(cloudlet_list)


## eval 
np.random.seed(1)
for i in range(200):
	#random.shuffle(cloudlet_list)
	cloudlet_temp = cloudlet_list[0]
	cloudlet_list = cloudlet_list[1:]
	cloudlet_list.append(cloudlet_temp)
	for cloudlet in cloudlet_list:
		cloudlet.run_onestepSJF(True)
for i in range(200):
	#random.shuffle(cloudlet_list)
	for cloudlet in cloudlet_list:
		cloudlet.run_onestepSJF(False)

rsp_list = list()

for cloudlet in cloudlet_list:
	rsp_list.append(cloudlet.get_rw())
#rsp0 = cloudlet_list[0].get_rw()
#rsp1 = cloudlet_list[1].get_rw()	
for cloudlet in cloudlet_list:
	cloudlet.reset()

print("End simulation")
for index,rsp in list(enumerate(rsp_list)):
	print("avg response time for cloudlet "+str(index)+" :" + str(float(sum(rsp))/max(len(rsp),0)))

#rw0 = cloudlet_list[0].get()
#rw1 = cloudlet_list[1].get()
#plt.plot(rw0,color='red')
#plt.plot(rw1,color='blue')
#plt.show()
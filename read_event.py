from os import listdir
from os.path import isfile, join
import gzip
class taskreader:

        def __init__(self,path):
                self.path = path
                self.filelist = [f for f in listdir(self.path) if isfile(join(self.path, f))]
                self.filelist = filter(lambda x: x.split(".")[-1]=="gz",self.filelist)
                self.curfile = 1
                self.curidx = 0
                with gzip.open(join(self.path,self.filelist[self.curfile])) as f:
                        self.curlines = f.readlines()
                        self.curlines = [x.strip() for x in self.curlines] 

        def isnext(self):
                if(self.curidx < len(self.curlines)):
                        return True
                elif (self.curfile < len(self.filelist)):
                        self.curidx = 0
                        with gzip.open(join(self.path,self.filelist[self.curfile])) as f:
                                self.curlines = f.readlines()
                                self.curlines = [x.strip() for x in self.curlines]
                        self.curfile += 1
                else:
                        return False
        def getnext(self):
                line = self.curlines[self.curidx]
                self.curidx += 1
                line = line.split(",")
                cpu_req = float(line[9])
                mem_req = float(line[10])
                return(cpu_req,mem_req)






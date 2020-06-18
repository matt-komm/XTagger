import logging
import os
import ROOT

class InputFiles():
    def __init__(self, maxFiles=-1):
        self.maxFiles = maxFiles
        self.fileList = []
        
    
    def addFileList(self,path):
        f = open(path)
        for line in f:
            basepath = path.rsplit('/',1)[0]
            fileName = line.strip()
            self.addFile(os.path.join(basepath,fileName))
        f.close()
        
    def addFile(self,path):
        if os.path.exists(path):
            self.fileList.append(path)
            logging.debug("Adding file: '"+path+"'")
        else:
            logging.warning("file '"+path+"' does not exists -> skip!")
        
    def nFiles(self):
        if self.maxFiles>0:
            return min(self.maxFiles,len(self.fileList))
        else:
            return len(self.fileList)
            
    def nJets(self):
        chain = ROOT.TChain("jets")
        for f in self.getFileList():
            chain.AddFile(f)
        return chain.GetEntries()
        
    def getFileList(self):
        return self.fileList[0:self.nFiles()]
            
            

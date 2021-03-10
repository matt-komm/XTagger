import logging
import os
import ROOT

class InputFiles():
    def __init__(self, maxFiles=-1,percentage=1.0):
        self.maxFiles = maxFiles
        self.percentage = percentage
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
            rootFile = ROOT.TFile(path)
            if rootFile.IsZombie():
                logging.error("Found zombie input '"+path+"' -> skip")
            else:
                self.fileList.append(path)
                logging.debug("Adding file: '"+path+"'")
            rootFile.Close()
        else:
            logging.warning("file '"+path+"' does not exists -> skip!")
        
    def nFiles(self):
        return min(
            int(round(self.percentage*len(self.fileList))),
            self.maxFiles if self.maxFiles>0 else len(self.fileList)
        )
            
    def nJets(self):
        chain = ROOT.TChain("jets")
        for f in self.getFileList():
            chain.AddFile(f)
        return chain.GetEntries()
        
    def getFileList(self):
        return self.fileList[0:self.nFiles()]
            
            

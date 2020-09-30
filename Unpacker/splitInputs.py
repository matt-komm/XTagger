import os
import sys
import re
import shutil

inputPath = "/vols/build/cms/mkomm/HNL/XTagger/Unpacker/inputs/2018"


outputTrainPath = "/vols/build/cms/mkomm/HNL/XTagger/Unpacker/training_mixed/2018"
outputTestPath = "/vols/build/cms/mkomm/HNL/XTagger/Unpacker/testing_mixed/2018"

configs = [
    {
        "pattern":"TTJets\S+",
        "maxFiles": -1,
        "trainFraction": 1,
        "header":[
            #"#cap 2000000",
            "#cap 5000000",
            "#select ((isG>0.5)*(0.5+0.2/(1+exp(4*(log(pt)-log(30))))<rand)) + (isG<0.5)",
            "#select ((isUD>0.5)*(0.5/(1+exp(4*(log(pt)-log(30))))<rand)) + (isUD<0.5)",
            "#select ((isPU>0.5)*(0.95/(1+exp(4*(log(pt)-log(20))))<rand)) + (isPU<0.5)",
        ],
        "output":"TTJets.txt"
    },
    {
        "pattern":"QCD_Pt\S+",
        "maxFiles": -1,
        "trainFraction": 1,
        "header":[
            #"#cap 3000000",
            "#cap 5000000",
            "#select ((isG>0.5)*(0.5+0.2/(1+exp(4*(log(pt)-log(30))))<rand)) + (isG<0.5)",
            "#select ((isUD>0.5)*(0.5/(1+exp(4*(log(pt)-log(30))))<rand)) + (isUD<0.5)",
            "#select ((isPU>0.5)*(0.95/(1+exp(4*(log(pt)-log(20))))<rand)) + (isPU<0.5)",
        ],
        "output":"QCD.txt"
    },
    {
        "pattern":"WJets\S+",
        "maxFiles": 200,
        "trainFraction": 1,
        "header":[
            #"#cap 5000000",
            "#cap 100000000",
            "#select ((isG>0.5)*(0.5+0.2/(1+exp(4*(log(pt)-log(30))))<rand)) + (isG<0.5)",
            "#select ((isUD>0.5)*(0.5/(1+exp(4*(log(pt)-log(30))))<rand)) + (isUD<0.5)",
            "#select ((isPU>0.5)*(0.95/(1+exp(4*(log(pt)-log(20))))<rand)) + (isPU<0.5)",
        ],
        "output":"WJets.txt"
    },
    {
        "pattern":"HNL_\S+_all_ctau1p0e04\S+",
        "maxFiles": 4,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNL_all_ctau1p0e04.txt"
    },
    {
        "pattern":"HNL_\S+_all_ctau1p0e03\S+",
        "maxFiles": 4,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNL_all_ctau1p0e03.txt"
    },
    {
        "pattern":"HNL_\S+_all_ctau1p0e02\S+",
        "maxFiles": 4,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNL_all_ctau1p0e02.txt"
    },
    {
        "pattern":"HNL_\S+_all_ctau1p0e01\S+",
        "maxFiles": 4,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNL_all_ctau1p0e01.txt"
    },
    {
        "pattern":"HNL_\S+_all_ctau1p0e00\S+",
        "maxFiles": 4,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNL_all_ctau1p0e00.txt"
    },
    {
        "pattern":"HNL_\S+_all_ctau1p0e-01\S+",
        "maxFiles": 4,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNL_all_ctau1p0e-01.txt"
    },
    {
        "pattern":"HNL_\S+_all_ctau1p0e-02\S+",
        "maxFiles": 4,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNL_all_ctau1p0e-02.txt"
    },
    {
        "pattern":"LLPGun\S+",
        "maxFiles":-1,
        "trainFraction": 1,
        "header":[
            "#select ((isLLP_ANY)>0.5)"
        ],
        "output":"HNLGun.txt"
    },
]


try:
    shutil.rmtree(os.path.join(outputTrainPath))
except Exception,e:
    print e
    
os.makedirs(os.path.join(outputTrainPath))
        
        
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
        
for f in os.listdir(inputPath):
    if f.endswith(".txt"):
        inputs = []
        for l in open(os.path.join(inputPath,f)):
            if l.find(".root")>0:
                inputs.append(l.replace('\r','').replace('\n','').replace(' ',''))
        if len(inputs)==1:
            print "Cannot split only 1 file in ",f
            continue
        elif len(inputs)==0:
            print "File is empty ",f
            continue

        inputs.sort(key=natural_keys)

        cfg = None
        for config in configs:
            if re.match(config['pattern'],f):
                cfg = config
                break
        if cfg==None:
            print "No matching pattern - skip ",f
            continue


        if cfg['trainFraction']<0.999:
            index = int(round(len(inputs)*cfg['trainFraction']))
            if index==0:
                index=1
            elif index==len(inputs):
                index = len(inputs)-1
        else:
            index=len(inputs)

        outputExists=False
        if os.path.exists(os.path.join(outputTrainPath,cfg['output'])):
            outputExists=True

        outTrain = open(os.path.join(outputTrainPath,cfg['output']),'aw+')

        if not outputExists:
            for h in cfg['header']:
                outTrain.write(h+'\n')
                
        if cfg['maxFiles']>0:
            index = min(index,cfg['maxFiles'])
                
        for i in range(0,index):
            outTrain.write(inputs[i]+'\n')
        outTrain.close()

        if index<len(inputs):
            outTest = open(os.path.join(outputTestPath,f),'w')
            for i in range(index,len(inputs)):
                outTest.write(inputs[i]+'\n')
            outTest.close()
            
            

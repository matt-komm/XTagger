import sys
import signal
import datetime
import numpy as np
import copy
import tensorflow as tf
import ROOT
import keras
from keras import backend as K
import sklearn.metrics
import logging
import math
import xtools
import os
import copy
import argparse
import time
import shutil
import imp
import random
import inspect

from feature_dict import featureDict

xtools.setupLogging(level=logging.INFO)
# tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest="trainFiles", default=[], action='append', help='input file list for training jet class')
parser.add_argument('--test', dest="testFiles", default=[], action='append', help='input file list for testing jet class')
parser.add_argument('--trainDA', dest="trainFilesDA", default=[], action='append', help='input file list for training jet domain')
parser.add_argument('--testDA', dest="testFilesDA", default=[], action='append', help='input file list for testing jet domain')
parser.add_argument('--perf', dest="perfList", default=[], action='append', help='input file list for performance test')
parser.add_argument('-o', '--output', help='job name', dest='outputFolder', required=True)
parser.add_argument('-n', action='store', type=int, dest='maxFiles',
                    help='number of files to be processed')
parser.add_argument('--gpu', action='store_true', dest='forceGPU',
                    help='try to use gpu', default=False)
parser.add_argument('-b', '--batch', action='store', type=int,
                    help='batchSize', dest='batchSize', default=10000)
parser.add_argument('-c', action='store_true',
                    help='achieve class balance', default=False)
parser.add_argument('-e', '--epoch', action='store', type=int, dest='nepochs',
                    help='number of epochs', default=60)
parser.add_argument('-f', '--force', action='store_true',
                    dest='overwriteFlag',
                    help='overwrite output folder', default=False)
parser.add_argument('-p', '--parametric', action='store_true',
                    dest='parametric',
                    help='train a parametric model', default=False)
parser.add_argument('--noda', action='store_true',
                    dest='noda',
                    help='deactivate DA', default=False)
parser.add_argument('--seed', dest='seed', default=int(time.time()), type=int,
                    help='Random seed')
parser.add_argument('-m', '--model', action='store', help='model file',
                    default='xtools/NominalNetwork.py')
parser.add_argument('-r', '--resume', type=int, help='resume training at given epoch',
                    default=0, dest='resume')
parser.add_argument('--lambda', type=float,help='domain loss weight',
                    default=30.,dest='lambda0')
parser.add_argument('--lr', type=float,help='initial learning rate',
                    default=0.01,dest='lr')  
parser.add_argument('--kappa', type=float,help='learning rate decay val',
                    default=0.01,dest='kappa')

args = parser.parse_args()


outputFolder = args.outputFolder
if (os.path.exists(outputFolder) and args.overwriteFlag):
    logging.warning( "Overwriting output folder!")
else:
    logging.info( "Creating output folder '%s'!" % outputFolder)
    os.makedirs(outputFolder)

#from xtools import AttentionNetwork as Network
#from xtools import NominalNetwork as Network

logging.info("Python: %s (%s)"%(sys.version_info,sys.executable))
logging.info("Keras: %s (%s)"%(keras.__version__,os.path.dirname(keras.__file__)))
logging.info("TensorFlow: %s (%s)"%(tf.__version__,os.path.dirname(tf.__file__)))
logging.info("Network: "+args.model)

shutil.copyfile(args.model,os.path.join(args.outputFolder,"Network.py"))




devices = xtools.Devices()
if args.forceGPU and devices.nGPU==0:
    logging.critical("Enforcing GPU usage but no GPU found!")
    sys.exit(1)

logging.info("Output folder: %s"%args.outputFolder)
logging.info("Epochs: %i"%args.nepochs)
logging.info("Batch size: %i"%args.batchSize)
logging.info("Random seed: %i"%args.seed)
logging.info("Learning rate: %.3e"%args.lr)
logging.info("Learning rate decay: %.3e"%args.kappa)

useDA = len(args.trainFilesDA)>0 and len(args.testFilesDA)>0
logging.info("Use DA: "+"True" if useDA else "False")
logging.info("Lambda0: %.1f"%(args.lambda0))

random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

#TODO: make maxFiles a percentage

trainInputs = xtools.InputFiles(maxFiles=args.maxFiles)
for f in args.trainFiles:
    trainInputs.addFileList(f)
testInputs = xtools.InputFiles(maxFiles=args.maxFiles)
for f in args.testFiles:
    testInputs.addFileList(f)
    
logging.info("Training files %i"%trainInputs.nFiles())
logging.info("Testing files %i"%testInputs.nFiles())
    
if useDA:
    trainDAInputs = xtools.InputFiles(maxFiles=args.maxFiles)
    for f in args.trainFilesDA:
        trainDAInputs.addFileList(f)
    testDAInputs = xtools.InputFiles(maxFiles=args.maxFiles)
    for f in args.testFilesDA:
        testDAInputs.addFileList(f)

    logging.info("Training DA files %i"%trainDAInputs.nFiles())
    logging.info("Testing DA files %i"%testDAInputs.nFiles())


#TODO make dict and read title/name from file saved as comments
perfInputList = []
for perfFile in args.perfList:
    perfInputs = xtools.InputFiles(maxFiles=args.maxFiles)
    perfInputs.addFileList(perfFile)
    perfInputList.append(perfInputs)
    logging.info("Perf files %i"%perfInputs.nFiles())



resampleWeights = xtools.ResampleWeights(
    trainInputs.getFileList(),
    featureDict['truth']['names'],
    featureDict['truth']['weights'],
    targetWeight='jetorigin_isLLP_MU||jetorigin_isLLP_QMU||jetorigin_isLLP_QQMU' \
            +'||jetorigin_isLLP_E||jetorigin_isLLP_QE||jetorigin_isLLP_QQE' \
            +'||jetorigin_isLLP_TAU||jetorigin_isLLP_QTAU||jetorigin_isLLP_QQTAU',
    ptBinning=np.concatenate([[10.],np.logspace(1.2,2.1,22)]),#np.array([10., 12.5, 15., 17.5, 20., 22.5, 25.,27.5 30.,3 35., 40., 50., 60., 70., 80., 100., 120.]),
    etaBinning=np.linspace(-2.4,2.4,6),
    paramBinning=np.linspace(-3,3,10)
)

resampleWeights.plot(outputFolder)
weights = resampleWeights.reweight(classBalance=True,oversampling=2)
weights.plot(os.path.join(outputFolder,"weights.pdf"))
weights.save(os.path.join(outputFolder,"weights.root"))


pipelineTrain = xtools.Pipeline(
    trainInputs.getFileList(),
    featureDict,
    batchSize=args.batchSize,
    resample=True,
    weightFile=os.path.join(outputFolder,"weights.root"),
    labelNameList=resampleWeights.getLabelNameList(),
)

pipelineTest = xtools.Pipeline(
    testInputs.getFileList(),
    featureDict,
    batchSize=args.batchSize,
    resample=True,
    weightFile=os.path.join(outputFolder,"weights.root"),
    labelNameList=resampleWeights.getLabelNameList(),
)

if useDA:
    featureDictDA = copy.deepcopy(featureDict)
    del featureDictDA["gen"]
    featureDictDA["truth"] = {
        "weights":[
            "1",
        ],
        "branches":[
            "isData",
        ]
    }
    featureDictDA['xsecweight'] = {
        "branches":[
            "xsecweight",
        ]
    }
    
    pipelineTrainDA = xtools.Pipeline(
        trainDAInputs.getFileList(),
        featureDictDA,
        batchSize=args.batchSize,
        repeat=None, #unlimited
    )

    pipelineTestDA = xtools.Pipeline(
        testDAInputs.getFileList(),
        featureDictDA,
        batchSize=args.batchSize,
        repeat=None, #unlimited
    )

perfPipelines = []
for perfInputs in perfInputList:
    perfPipelines.append(xtools.Pipeline(
        perfInputs.getFileList(),
        featureDict,
        batchSize=min(args.batchSize,max(1000,int(round(perfInputs.nJets()/100.)))),
    ))


coord = None
def resetSession():
    if coord:
        logging.info("Stopping threads and resetting session")
        coord.request_stop()
        coord.join(threads)
        K.clear_session()

signal.signal(signal.SIGINT, lambda signum, frame: [resetSession(),sys.exit(1)])


for epoch in range(args.resume, args.nepochs):
    start_time_epoch = time.time()

    Network = imp.load_source('Network', os.path.join(args.outputFolder,"Network.py")).network

    network = Network(featureDict)
    modelClassTrain = network.makeClassModelWithSmearing()
    modelClass = network.makeClassModel()
    
    lrDecayDelay = 10

    learningRateDecay = 1./(1.+args.kappa*max(0,epoch-lrDecayDelay)**1.5)
    learningRate = args.lr*learningRateDecay
    optClass = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999)
    
    modelClassTrain.compile(
        optClass,
        loss=[keras.losses.categorical_crossentropy],
        metrics=[keras.metrics.categorical_accuracy],
        loss_weights=[1.]
    )
    
    if useDA:
        daWeight = args.lambda0*(2./(1+math.exp(-0.1*max(0,epoch-1)))-1.)
        optDA = keras.optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999)
        modelFullTrain = network.makeFullModelWithSmearing()

        modelFullTrain.compile(
            optDA,
            loss=[keras.losses.categorical_crossentropy,keras.losses.binary_crossentropy],
            metrics=[keras.metrics.categorical_accuracy,keras.metrics.binary_accuracy],
            loss_weights=[1.,daWeight]
        )
    
    modelClass.compile(
        optClass,
        loss=[keras.losses.categorical_crossentropy],
        metrics=[keras.metrics.categorical_accuracy],
        loss_weights=[1.]
    )

    if epoch==0:
        modelClass.summary()
    
    firstLLPIdx = featureDict['truth']['firstLLPIdx']
    train_batch = pipelineTrain.init(isLLPFct = lambda batch: tf.reduce_sum(batch["truth"][:, firstLLPIdx:],axis=1) > 0.5)
    test_batch = pipelineTest.init(isLLPFct = lambda batch: tf.reduce_sum(batch["truth"][:, firstLLPIdx:],axis=1) > 0.5)
    perf_batches = []
    for perfPipeline in perfPipelines:
        perf_batches.append(perfPipeline.init())

    if useDA:
        train_da_batch = pipelineTrainDA.init()
        test_da_batch = pipelineTestDA.init()


    if epoch==0:
        distributions = resampleWeights.makeDistribution()
        
        daHistPlotter = xtools.HistPlotter('da',bins=np.linspace(10,250,51),xaxis="Jet pT (GeV)",yaxis="#Jets")
        daHistPlotter.makeGroup("mc",label="MC",lineColor=ROOT.kAzure-4,drawOptions="HIST")
        daHistPlotter.makeGroup("data",label="Data",markerStyle=20,markerSize=1.3,drawOptions="PE")

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    sess = K.get_session()
    sess.run(init_op)

    if epoch>0:
        networkWeightFile = os.path.join(outputFolder,'weight_%i.hdf5'%(epoch-1))
        if os.path.exists(networkWeightFile):
            logging.info("loading weights from "+networkWeightFile)
            modelClass.load_weights(networkWeightFile)
        else:
            logging.critical("No weights from previous epoch found")
            sys.exit(1)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_loss = 0
    time_train = time.time()
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            train_batch_value = sess.run(train_batch)
            
            for k,elem in train_batch_value.iteritems():
                train_batch_value[k] = np.nan_to_num(train_batch_value[k])
                if k=='num':
                    continue
                if not np.isfinite(elem).all():
                    logging.error("Found non finite training value in "+k+" ("+str(elem.shape)+")\nindices:"+str(np.isfinite(elem).nonzero()))
                if np.any(elem>1e8) or np.any(elem<-1e8):
                    logging.error("Found large training value (>1e8 or <-1e8) in "+k+" ("+str(elem.shape)+")\nindices:"+str(np.nonzero(np.logical_or(elem>1e8,elem<-1e8))))
                
            if useDA:
                train_da_batch_value = sess.run(train_da_batch)
                for k,elem in train_da_batch_value.iteritems():
                    train_da_batch_value[k] = np.nan_to_num(train_da_batch_value[k])
                    if k=='num':
                        continue
                    if not np.isfinite(elem).all():
                        logging.error("Found non finite training da value in "+k+" ("+str(elem.shape)+")\nindices:"+str(np.isfinite(elem).nonzero()))
                    if np.any(elem>1e8) or np.any(elem<-1e8):
                        logging.error("Found large training da value (>1e8 or <-1e8) in "+k+" ("+str(elem.shape)+")\nindices:"+str(np.nonzero(np.logical_or(elem>1e8,elem<-1e8))))
                    

            if epoch==0:
                #featurePlotter.fill(train_batch_value)

                distributions.fill(
                    train_batch_value['truth'],
                    train_batch_value['globalvars'][:,0],
                    train_batch_value['globalvars'][:,1],
                    train_batch_value['gen'][:,0],
                )
                
            
            smearBkgGen = np.expand_dims((np.sum(train_batch_value['truth'][:,:firstLLPIdx],axis=1)>0.5)*np.random.normal(0.,1.,size=train_batch_value['truth'].shape[0]),axis=1)

            train_inputs_class = [
                train_batch_value['gen'],
                train_batch_value['gen']+smearBkgGen,
                train_batch_value['globalvars'],
                train_batch_value['cpf'],
                train_batch_value['npf'],
                train_batch_value['sv'],
                train_batch_value['muon'],
                train_batch_value['electron'],
            ]
            
            if useDA:
                train_inputs_domain = [
                    train_da_batch_value['globalvars'],
                    train_da_batch_value['cpf'],
                    train_da_batch_value['npf'],
                    train_da_batch_value['sv'],
                    train_da_batch_value['muon'],
                    train_da_batch_value['electron'],
                ]
                train_outputs = modelFullTrain.train_on_batch(
                    train_inputs_class+train_inputs_domain,
                    [train_batch_value['truth'],train_da_batch_value['truth']],
                    sample_weight=[
                        np.ones(train_batch_value['truth'].shape[0]),
                        train_da_batch_value['xsecweight'][:,0],
                    ]
                )
                #print train_outputs
                train_loss+=train_outputs[0]
                acc = train_outputs[3]
                
            else:
                train_outputs = modelClassTrain.train_on_batch(
                    train_inputs_class,
                    train_batch_value['truth']
                )
                train_loss+=train_outputs[0]
                acc = train_outputs[1]
                

            if step%10==0:
                logging.info("Training step %i-%i: loss=%.4f, accuracy=%.2f%%"%(epoch,step,train_outputs[0],100.*acc))


    except tf.errors.OutOfRangeError:
        pass


    train_loss = train_loss/step
    time_train = (time.time()-time_train)/step
    logging.info('Done training for %i steps of epoch %i and learning rate %.4f: loss=%.3e, %.1fms/step'%(step,epoch,learningRate,train_loss,time_train*1000.))

    if epoch==0:
        #featurePlotter.plot(outputFolder)
        distributions.plot(os.path.join(outputFolder,"resampled.pdf"))
    modelClass.save_weights(os.path.join(outputFolder,'weight_%i.hdf5'%epoch))


    if epoch%5==0:
        testMonitor = xtools.PerformanceMonitor(featureDict)

    test_loss = 0
    time_test = time.time()
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            test_batch_value = sess.run(test_batch)

            badValue = False
            for k,elem in test_batch_value.iteritems():
                test_batch_value[k] = np.nan_to_num(test_batch_value[k])
                if k=='num':
                    continue
                if not np.isfinite(elem).all():
                    logging.error("Found non finite testing value in "+k+" ("+str(elem.shape)+")\nindices:"+str(np.isfinite(elem).nonzero()))
                    badValue = True
                if np.any(elem>1e8) or np.any(elem<-1e8):
                    logging.error("Found large testing value (>1e8 or <-1e8) in "+k+" ("+str(elem.shape)+")\nindices:"+str(np.nonzero(np.logical_or(elem>1e8,elem<-1e8))))
                    badValue = True
                
            #print train_batch_value
            test_inputs_class = [
                test_batch_value['gen'],
                test_batch_value['globalvars'],
                test_batch_value['cpf'],
                test_batch_value['npf'],
                test_batch_value['sv'],
                test_batch_value['muon'],
                test_batch_value['electron'],
                
            ]
            
            test_outputs = modelClass.test_on_batch(test_inputs_class,test_batch_value['truth'])
            test_batch_prediction = modelClass.predict_on_batch(test_inputs_class)
            
            if epoch%5==0:
                testMonitor.analyze_batch(test_batch_value,test_batch_prediction)
                
            test_loss+=test_outputs[0]
            acc = test_outputs[1]
            if step%10==0:
                logging.info("Testing step %i-%i: loss=%.4f, accuracy=%.2f%%"%(epoch,step,test_outputs[0],100.*acc))


    except tf.errors.OutOfRangeError:
        pass
        
    test_loss = test_loss/step
    time_test = (time.time()-time_test)/step
    logging.info('Done testing for %i steps of epoch %i: loss=%.4f, %.1fms/step'%(step,epoch,test_loss,time_test*1000.))

    logging.info("Epoch duration: %.1fmin"%((time.time() - start_time_epoch)/60.))

        
    if epoch%5==0:
        #testMonitor.save(os.path.join(outputFolder,'test_%i.hdf5'%(epoch)))
        testMonitor.plot(featureDict['truth']['names'],os.path.join(outputFolder,'test_%i'%(epoch)))
            
    
    f = open(os.path.join(outputFolder, "model_epoch.stat"), "a")
    f.write("%i;%.3e;%.3e;%.3e\n"%(
        epoch,
        learningRate,
        train_loss,
        test_loss,
    ))
    f.close()

    if epoch%5==0:
        for iperf,perf_batch in enumerate(perf_batches):
            perfMonitor = xtools.PerformanceMonitor(featureDict)
            try:
                step = 0
                while not coord.should_stop():
                    step += 1
                    perf_batch_value = sess.run(perf_batch)
                    for k,elem in perf_batch_value.iteritems():
                        perf_batch_value[k] = np.nan_to_num(perf_batch_value[k])
                        
                    selectSignal = np.sum(perf_batch_value['truth'][:,8:],axis=1)>0.5
                    if np.sum(selectSignal)>0:
                        meanParameter = min(max(np.mean(perf_batch_value['gen'][selectSignal]),-3),3)
                        stdParameter = max(1e-2,np.std(perf_batch_value['gen'][selectSignal]))
                    else:
                        meanParameter = 0.0
                        stdParameter = 0.2
                    #print meanParameter,stdParameter
                    
                    perf_batch_value['gen'] = np.random.normal(meanParameter,stdParameter,size=perf_batch_value['gen'].shape)
                    perf_inputs_class = [
                        perf_batch_value['gen'],
                        perf_batch_value['globalvars'],
                        perf_batch_value['cpf'],
                        perf_batch_value['npf'],
                        perf_batch_value['sv'],
                        perf_batch_value['muon'],
                        perf_batch_value['electron'],
                        
                    ]
                    perf_batch_prediction = modelClass.predict_on_batch(perf_inputs_class)
                    perfMonitor.analyze_batch(perf_batch_value,perf_batch_prediction)

                    if step%10==0:
                        logging.info("Perf %i step %i-%i"%(iperf,epoch,step))


            except tf.errors.OutOfRangeError:
                pass

            #auc = perfMonitor.auc()
            #logging.info("Done perf %i for %i steps of epoch %i: auc=%.2f%%"%(iperf,step,epoch,100.*auc))
            
            #perfMonitor.save(os.path.join(outputFolder,'perf_%i_%i.hdf5'%(iperf,epoch)))
            perfMonitor.plot(featureDict['truth']['names'],os.path.join(outputFolder,'perf_%i_%i'%(iperf,epoch)))
    resetSession()
    

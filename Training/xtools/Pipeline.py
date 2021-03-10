import tensorflow as tf
import logging
import xtagger
import os

class Pipeline():
    def __init__(
        self,
        files, 
        features, 
        batchSize=1000,
        resample=False,
        weightFile=None,
        weightParamFile=None,
        labelNameList=None,
        repeat=1,
        bagging=1.,
        maxThreads = 6
    ):
        self.files = files
        self.features = features
        self.labelNameList = labelNameList
        self.weightFile = weightFile
        self.weightParamFile = weightParamFile
        self.batchSize = batchSize
        self.resample = resample
        self.repeat = repeat
        self.bagging = bagging
        self.maxThreads = maxThreads
    

    def init(self,isLLPFct=None):
        with tf.device('/cpu:0'):
            if self.bagging>0. and self.bagging<1.:
                inputFileList = random.sample(self.files,int(max(1,round(len(self.files)*self.bagging))))
            else:
                inputFileList = self.files
            fileListQueue = tf.train.string_input_producer(
                    inputFileList, num_epochs=self.repeat, shuffle=True)

            rootreader_op = []
            resamplers = []
            OMP_NUM_THREADS = -1
            if os.environ.has_key('OMP_NUM_THREADS'):
                try:
                    OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
                except Exception:
                    pass
            if OMP_NUM_THREADS>0:
                self.maxThreads = min(OMP_NUM_THREADS,self.maxThreads)
            
            for _ in range(min(1+int(len(inputFileList)/2.), self.maxThreads)):
                reader_batch = max(10,int(self.batchSize/(2.5*self.maxThreads)))
                reader = xtagger.root_reader(fileListQueue, self.features, "jets", batch=reader_batch).batch()
                rootreader_op.append(reader)
                if self.resample:
                    result = reader
                    if self.weightParamFile!=None:
                        lxyweight = xtagger.lxy_weights(
                            result["truth"],
                            result["gen"],
                            self.weightParamFile,
                            self.labelNameList,
                            [0]
                        )
                        result['lxyweight'] = lxyweight
                        result = xtagger.resampler(
                            lxyweight,
                            result
                        ).resample()
                    
                    weight = xtagger.classification_weights(
                        result["truth"],
                        result["globalvars"],
                        self.weightFile,
                        self.labelNameList,
                        [0, 1]
                    )
                    result = xtagger.resampler(
                        weight,
                        result
                    ).resample()
                    
                    

                    resamplers.append(result)

            minAfterDequeue = self.batchSize * 2
            capacity = minAfterDequeue + 3*self.batchSize
            batch = tf.train.shuffle_batch_join(
                resamplers if self.resample else rootreader_op,
                batch_size=self.batchSize,
                capacity=capacity,
                min_after_dequeue=minAfterDequeue,
                enqueue_many=True  # requires to read examples in batches!
            )
            if self.resample:
                isSignal = isLLPFct(batch)
                batch["gen"] = xtagger.fake_background(batch["gen"], isSignal, feature_index=0, rstart=-2.5, rend=3.5)

            return batch



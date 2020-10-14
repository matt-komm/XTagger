import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
import xtools
    
class AttentionMergedNetworkSigmoid(xtools.NominalNetwork):
    def __init__(self,featureDict):
        xtools.NominalNetwork.__init__(self,featureDict)
        
        
        #### CAND ####
        self.cand_conv = []
        for i,filters in enumerate([64,32,32,24]):
            self.cand_conv.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name='cand_conv'+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name='cand_activation'+str(i+1)),
                keras.layers.Dropout(0.1,name='cand_dropout'+str(i+1)),
            ])
            
        self.cand_attention = []
        for i,filters in enumerate([64,32,32,15]):
            self.cand_attention.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name='cand_attention'+str(i+1)
                ),
            ])
            if i<3:
                self.cand_attention.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name='cand_attention_activation'+str(i+1)),
                    keras.layers.Dropout(0.1,name='cand_attention_dropout'+str(i+1)),
                ])
            else:
                self.cand_attention.extend([
                    keras.layers.Activation('sigmoid',name="cand_attention_activation"+str(i+1)),
                    keras.layers.Dropout(0.1,name='cand_attention_dropout'+str(i+1)),
                ])

        
        #### Global ####
        self.global_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["globalvars"]["branches"],
                self.featureDict["globalvars"]["preprocessing"]
            ), name="global_preproc")
        
        
        #### Features ####
        self.full_features = [keras.layers.Concatenate()]
        for i,nodes in enumerate([200]):
            self.full_features.extend([
                keras.layers.Dense(
                    nodes,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="features_dense"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="features_activation"+str(i+1))
            ])
            
            
        #### Class prediction ####
        self.class_prediction = []
        for i,nodes in enumerate([100,100]):
            self.class_prediction.extend([
                keras.layers.Dense(
                    nodes,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="class_dense"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="class_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="class_dropout"+str(i+1)),
            ])
        self.class_prediction.extend([
            keras.layers.Dense(
                self.nclasses,
                kernel_initializer='lecun_normal',
                bias_initializer='zeros',
                kernel_regularizer=keras.regularizers.l1(1e-6),
                name="class_nclasses"
            ),
        ])
        
    def mergeCandidates(self,candidates):
        def merge(candList):
            candFillList = []
            for icand in range(len(candList)):
                nFillLeft = 0
                nFillRight = 0
                for i in range(len(candList)):
                    if i<icand:
                        nFillLeft+=candList[i].shape.as_list()[2]
                    elif i>icand:
                        nFillRight+=candList[i].shape.as_list()[2]
                candFillList.append(tf.pad(
                    candList[icand],
                    [[0,0],[0,0],[nFillLeft,nFillRight]],
                    mode='CONSTANT',
                    constant_values=0.
                ))
            return tf.concat(candFillList,axis=1)
            
        return keras.layers.Lambda(merge)(candidates)
        
        
    def applyAttention(self,features,attention):
        result = keras.layers.Lambda(lambda x: tf.matmul(tf.transpose(x[0],[0,2,1]),x[1]))([attention,features])
        return keras.layers.Flatten()(result)
        #result = tf.constant(0,dtype=tf.float32,shape=[tf.getShape(features)[0],attention.shape[1],features.shape[2]
            
    def addToConvFeatures(self,conv,features):
        def tileFeatures(x):
            x = tf.reshape(tf.tile(features,[1,conv.shape[1]]),[-1,conv.shape[1],x.shape[1]])
            return x
            
        tiled = keras.layers.Lambda(tileFeatures)(features)
        return keras.layers.Concatenate(axis=2)([conv,tiled])
 
    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen=None):
        globalvars_preproc = self.global_preproc(globalvars)
        
        cand = self.mergeCandidates([
            self.cpf_preproc(cpf),
            self.npf_preproc(npf),
            self.muon_preproc(muon),
            self.electron_preproc(electron)
        ])
        
        cand_conv = self.applyLayers(cand,self.cand_conv)
        cand_attention = self.applyLayers(cand,self.cand_attention)
        
        cand_tensor = self.applyAttention(cand_conv,cand_attention)
             
        sv_conv = self.applyLayers(self.sv_preproc(sv),self.sv_conv)   
        
        full_features = self.applyLayers([globalvars_preproc,cand_tensor,sv_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,gen], self.full_features)
        
        return full_features
    
network = AttentionMergedNetworkSigmoid
    

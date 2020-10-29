import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
import xtools

class AttentionNetwork(xtools.NominalNetwork):
    def __init__(self,featureDict):
        xtools.NominalNetwork.__init__(self,featureDict)


        #### CPF ####
        self.cpf_conv = []
        for i,filters in enumerate([64,32,32,8]):
            self.cpf_conv.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l2(1e-6),
                    name='cpf_conv'+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name='cpf_activation'+str(i+1)),
                keras.layers.Dropout(0.1,name='cpf_dropout'+str(i+1)),
            ])

        self.cpf_attention = []#keras.layers.Lambda(lambda x: tf.transpose(x,[0,2,1]))]
        for i,filters in enumerate([64,32,32,10]):
            self.cpf_attention.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l2(1e-6),
                    name='cpf_attention'+str(i+1)
                ),
            ])
            if i<3:
                self.cpf_attention.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name='cpf_attention_activation'+str(i+1)),
                    keras.layers.Dropout(0.1,name='cpf_attention_dropout'+str(i+1)),
                ])
            else:
                self.cpf_attention.extend([
                    keras.layers.Activation('sigmoid',name="cpf_attention_activation"+str(i+1)),
                    keras.layers.Dropout(0.1,name='cpf_attention_dropout'+str(i+1)),
                ])

        #### NPF ####
        self.npf_conv = []
        for i,filters in enumerate([32,16,16,4]):
            self.npf_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l2(1e-6),
                    name="npf_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="npf_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="npf_droupout"+str(i+1)),
            ])

        self.npf_attention = []#keras.layers.Lambda(lambda x: tf.transpose(x,[0,2,1]))]
        for i,filters in enumerate([64,32,32,10]):
            self.npf_attention.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l2(1e-6),
                    name="npf_attention"+str(i+1)
                ),
            ])
            if i<3:
                self.npf_attention.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name="npf_attention_activation"+str(i+1)),
                    keras.layers.Dropout(0.1,name="npf_attention_droupout"+str(i+1)),
                ])
            else:
                self.npf_attention.extend([
                    keras.layers.Activation('sigmoid',name="npf_attention_activation"+str(i+1)),
                    keras.layers.Dropout(0.1,name="npf_attention_droupout"+str(i+1)),
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
                    kernel_regularizer=keras.regularizers.l2(1e-6),
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
                    kernel_regularizer=keras.regularizers.l2(1e-6),
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
                kernel_regularizer=keras.regularizers.l2(1e-6),
                name="class_nclasses"
            ),
            keras.layers.Softmax(name="class_softmax")
        ])

    def returnsLogits(self):
        return False
        
    def addToConvFeatures(self,conv,features):
        def tileFeatures(x):
            x = tf.reshape(tf.tile(features,[1,conv.shape[1]]),[-1,conv.shape[1],x.shape[1]])
            return x
            
        tiled = keras.layers.Lambda(tileFeatures)(features)
        return keras.layers.Concatenate(axis=2)([conv,tiled])

    def applyAttention(self,features,attention):
        result = keras.layers.Lambda(lambda x: tf.matmul(tf.transpose(x[0],[0,2,1]),x[1]))([attention,features])
        return keras.layers.Flatten()(result)
        #result = tf.constant(0,dtype=tf.float32,shape=[tf.getShape(features)[0],attention.shape[1],features.shape[2]


    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen=None):
        globalvars_preproc = self.global_preproc(globalvars)
        
        global_features = keras.layers.Concatenate(axis=1)([globalvars_preproc,gen])

        cpf_features = self.addToConvFeatures(self.cpf_preproc(cpf),global_features)
        npf_features = self.addToConvFeatures(self.npf_preproc(npf),global_features)
        sv_features = self.addToConvFeatures(self.sv_preproc(sv),global_features)
        muon_features = self.addToConvFeatures(self.muon_preproc(muon),global_features)
        electron_features = self.addToConvFeatures(self.electron_preproc(electron),global_features)

        cpf_conv = self.applyLayers(cpf_features,self.cpf_conv)
        npf_conv = self.applyLayers(npf_features,self.npf_conv)
        sv_conv = self.applyLayers(sv_features,self.sv_conv)
        muon_conv = self.applyLayers(muon_features,self.muon_conv)
        electron_conv = self.applyLayers(electron_features,self.electron_conv)

        cpf_attention = self.applyLayers(cpf_features,self.cpf_attention)
        npf_attention = self.applyLayers(npf_features,self.npf_attention)

        cpf_tensor = self.applyAttention(cpf_conv,cpf_attention)
        npf_tensor = self.applyAttention(npf_conv,npf_attention)

        full_features = self.applyLayers([globalvars_preproc,cpf_tensor,npf_tensor,sv_conv,muon_conv,electron_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_tensor,npf_tensor,sv_conv,gen], self.full_features)

        return full_features

network = AttentionNetwork

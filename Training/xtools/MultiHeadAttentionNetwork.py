import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
import xtools

        
class QKAttentionLayer:
    def __init__(self,prefix,ncomb=[64,10],nqueries=[64,16],nkeys=[64,16],nheads=2):
        self.prefix = prefix
    
        if nqueries[-1]!=nkeys[-1]:
            raise Exception("Query size (%i) needs to match key size (%i)"%(ncomb[-1],nqueries[-1]))
            
        if nqueries[-1] % nheads != 0:
            raise Exception("Number of heads (%i) needs to be a multiple of key and query size (%i)"%(nheads,nqueries[-1]))
        
        self.ncomb = ncomb
        self.nqueries = nqueries
        self.nkeys = nkeys
        self.nheads = nheads
        
        self.combLayers = []
        for i,filters in enumerate(ncomb):
            self.combLayers.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=i<(len(ncomb)-1), #no bias in last layer
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name=prefix+'_comb_'+str(i+1)
                )
            ])
            if i<(len(ncomb)-1):
                self.combLayers.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name=prefix+'_comb_activation_'+str(i+1)),
                    keras.layers.Dropout(0.1,name=prefix+'_comb_dropout_'+str(i+1)),
                ])
    
    
        self.queryLayers = []
        for i,filters in enumerate(nqueries):
            self.queryLayers.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=i<(len(nqueries)-1), #no bias in last layer
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name=prefix+'_query_'+str(i+1)
                )
            ])
            if i<(len(nqueries)-1):
                self.queryLayers.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name=prefix+'_query_activation_'+str(i+1)),
                    keras.layers.Dropout(0.1,name=prefix+'_query_dropout_'+str(i+1)),
                ])
                
        self.keyLayers = []
        for i,filters in enumerate(nkeys):
            self.keyLayers.extend([
                keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=i<(len(nkeys)-1), #no bias in last layer
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name=prefix+'_key_'+str(i+1)
                )
            ])
            if i<(len(nkeys)-1):
                self.keyLayers.extend([
                    keras.layers.LeakyReLU(alpha=0.1,name=prefix+'_key_activation_'+str(i+1)),
                    keras.layers.Dropout(0.1,name=prefix+'_key_dropout_'+str(i+1)),
                ])
                

    def _combineCandidates(self,x,w):
        # x: features [batch,ncandidates,nfeatures]
        # w: weights [batch,ncandidates,ncomb]
        
        #w = tf.nn.softmax(w,axis=1) # softmax over candidates
        w = tf.nn.sigmoid(w) # sigmoid
        return tf.matmul(w,x, transpose_a = True) # returns [batch, ncomb, nfeatures]
        
    def _tileFeatures(self,x):
        # repeat feature ncomb[-1] times 
        x = tf.reshape(tf.tile(x,[1,self.ncomb[-1]]),[-1,self.ncomb[-1],x.shape[1]])
        return x     
    

    def _splitHeads(self,x):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x,[-1,x.shape[1],self.nheads,x.shape[2]/self.nheads])
        return tf.transpose(x,[0,2,1,3])
        
    def _mergeHeads(self,x):
        # [batch,heads,ncomb,ncandidates] -> [batch,ncomb,heads*ncandidates] (feature consists of attention from multiple heads)
        x = tf.transpose(x,[0,2,1,3])
        return tf.reshape(x,[-1,x.shape[1],x.shape[2]*x.shape[3]])
        
    def _applyAttention(self,features,weights):
        return keras.layers.Lambda(lambda x: tf.matmul(x[0],x[1]))([weights,features])
        
    def _queryKey(self,q,k):
        """
        Args:
            q: query [batch, heads, ncomb, querysize/heads]
            k: key [batch, heads, ncandidates, querysize/heads]
          Returns:
            attention_weights
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # [batch, heads, ncomb, ncandidates]
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)


        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # [batch, heads, ncomb, ncandidates]

        return attention_weights 
        
        
    def applyAttentionToCandidates(self,candidates,attentionWeights):
        if candidates.shape[-1] % self.nheads != 0:
            raise Exception("Candidate features (%i) need to be a multiple of number of heads (%i)"%(candidates.shape.as_list()[-1],self.nheads))
        candidates = keras.layers.Lambda(self._splitHeads)(candidates)
        candidates = keras.layers.Lambda(
            lambda x: self._applyAttention(x[0],x[1])
        )([candidates,attentionWeights])
        candidates = keras.layers.Lambda(self._mergeHeads)(candidates)  # expand features by nheads
        return candidates
        
    def applyAttentionToP4(self,p4,attentionWeights):
        # no good way to deal with multiple heads in case of lorentz vectors; just tile batch and take the first head afterwards (alternatively take average) or sum over combinations
        p4 = keras.layers.Lambda(
            lambda x: tf.reshape(tf.tile(x,[self.nheads,1,1]),[-1,self.nheads,x.shape[1],x.shape[2]])
        )(p4)
        '''
        candidates = keras.layers.Lambda(
            lambda x: self._applyAttention(x[0],x[1])[:,0,:,:]
        )([p4,attentionWeights])
        '''
        '''
        candidates = keras.layers.Lambda(
            lambda x: tf.reduce_mean(
                self._applyAttention(x[0],x[1]),
                axis=1
            )
        )([p4,attentionWeights])
        '''
        
        candidates = keras.layers.Lambda(
            lambda x: tf.reduce_sum(
                self._applyAttention(x[0],x[1]),
                axis=2
            )
        )([p4,attentionWeights])
        
        return candidates

                
    def getAttentionWeights(self,candidates, globalFeatures=None):
        combWeights = self.combLayers[0](candidates)
        for layer in self.combLayers[1:]:
            combWeights = layer(combWeights)
             
        combCandidates = keras.layers.Lambda(lambda x: self._combineCandidates(x[0],x[1]))([candidates,combWeights])
        
        if globalFeatures!=None:
            globalFeatures = keras.layers.Lambda(lambda x: self._tileFeatures(x))(globalFeatures)
            combCandidates = keras.layers.Concatenate(axis=2)([combCandidates,globalFeatures])
            
    
        queryResult = self.queryLayers[0](combCandidates)
        for layer in self.queryLayers[1:]:
            queryResult = layer(queryResult)
            
        keyResult = self.keyLayers[0](candidates)
        for layer in self.keyLayers[1:]:
            keyResult = layer(keyResult)
            
        splitHeadsLayer = keras.layers.Lambda(self._splitHeads)
            
        queryResult = splitHeadsLayer(queryResult) #[batch,heads,ncomb,querysize]
        keyResult = splitHeadsLayer(keyResult) #[batch,heads,ncandidates,querysize]
        
        attention = keras.layers.Lambda(lambda x: self._queryKey(x[0],x[1]))([queryResult,keyResult]) #[batch, heads, ncomb, ncandidates]

        return attention #[batch, heads, ncomb, ncandidates]
        
       
        

class MultiHeadAttentionNetwork(xtools.NominalNetwork):
    def __init__(self,featureDict):
        xtools.NominalNetwork.__init__(self,featureDict)

        self.input_gen = keras.layers.Input(
            shape=(len(self.featureDict["gen"]["branches"]),),
            name="input_gen"
        )
        self.input_globalvars = keras.layers.Input(
            shape=(len(self.featureDict["globalvars"]["branches"]),),
            name="input_global"
        )
        self.input_cpf = keras.layers.Input(
            shape=(self.featureDict["cpf"]["max"], len(self.featureDict["cpf"]["branches"])),
            name="input_cpf"
        )
        self.input_cpf_p4 = keras.layers.Input(
            shape=(self.featureDict["cpf_p4"]["max"], len(self.featureDict["cpf_p4"]["branches"])),
            name="input_cpf_p4"
        )
        self.input_npf = keras.layers.Input(
            shape=(self.featureDict["npf"]["max"], len(self.featureDict["npf"]["branches"])),
            name="input_npf"
        )
        self.input_npf_p4 = keras.layers.Input(
            shape=(self.featureDict["npf_p4"]["max"], len(self.featureDict["npf_p4"]["branches"])),
            name="input_npf_p4"
        )
        self.input_sv = keras.layers.Input(
            shape=(self.featureDict["sv"]["max"], len(self.featureDict["sv"]["branches"])),
            name="input_sv"
        )
        self.input_muon = keras.layers.Input(
            shape=(self.featureDict["muon"]["max"], len(self.featureDict["muon"]["branches"])),
            name="input_muon"
        )
        self.input_muon_p4 = keras.layers.Input(
            shape=(self.featureDict["muon_p4"]["max"], len(self.featureDict["muon_p4"]["branches"])),
            name="input_muon_p4"
        )
        self.input_electron = keras.layers.Input(
            shape=(self.featureDict["electron"]["max"], len(self.featureDict["electron"]["branches"])),
            name="input_electron"
        )
        self.input_electron_p4 = keras.layers.Input(
            shape=(self.featureDict["electron_p4"]["max"], len(self.featureDict["electron_p4"]["branches"])),
            name="input_electron_p4"
        )
        
        '''
        self.global_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["globalvars"]["branches"],
                self.featureDict["globalvars"]["preprocessing"]
            ), name="global_preproc")
            
        self.cpf_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["cpf"]["branches"],
                self.featureDict["cpf"]["preprocessing"]
            ),name='cpf_preproc')
            
        self.npf_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["npf"]["branches"],
                self.featureDict["npf"]["preprocessing"]
            ),name='npf_preproc')
            
        self.sv_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["sv"]["branches"],
                self.featureDict["sv"]["preprocessing"]
            ),name='sv_preproc')
            
        self.muon_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["muon"]["branches"],
                self.featureDict["muon"]["preprocessing"]
            ),name="muon_preproc")
            
        self.electron_preproc = \
            keras.layers.Lambda(self.preprocessingFct(
                self.featureDict["electron"]["branches"],
                self.featureDict["electron"]["preprocessing"]
            ),name="electron_preproc")
        '''

        self.cand_conv = []
        for i,filters in enumerate([64,32,32,16]):
            self.cand_conv.extend([
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

            
        self.cand_qkAttention = QKAttentionLayer("qk",ncomb=[16],nqueries=[64,32,32,16],nkeys=[16],nheads=4)
        


        self.sv_conv = []
        for i,filters in enumerate([32,16,16,12]):
            self.sv_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="sv_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="sv_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="sv_dropout"+str(i+1)),
            ])
 
        self.muon_conv = []
        for i,filters in enumerate([64,32,32,16]):
            self.muon_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="muon_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="muon_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="muon_dropout"+str(i+1)),
            ])
            
        self.electron_conv = []
        for i,filters in enumerate([64,32,32,16]):
            self.electron_conv.extend([keras.layers.Conv1D(
                    filters,1,
                    strides=1,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='lecun_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=keras.regularizers.l1(1e-6),
                    name="electron_conv"+str(i+1)
                ),
                keras.layers.LeakyReLU(alpha=0.1,name="electron_activation"+str(i+1)),
                keras.layers.Dropout(0.1,name="electron_dropout"+str(i+1)),
            ])

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
        
    def mergeCandidatesSlim(self,candidates):
        def merge(candList):
            maxFeatureSize = max(map(lambda x: x.shape.as_list()[2],candList))
            
            candFillList = []
            for icand in range(len(candList)):
                nFillLeft = 0
                nFillRight = 0
                
                #pad right to max features
                x = tf.pad(
                    candList[icand],
                    [[0,0],[0,0],[0,maxFeatureSize-candList[icand].shape.as_list()[2]]],
                    mode='CONSTANT',
                    constant_values=0.
                )
                # add index on the left
                x = tf.pad(
                    x,
                    [[0,0],[0,0],[1,0]],
                    mode='CONSTANT',
                    constant_values=(icand+1.)/len(candList)
                )
                
                candFillList.append(x)
                
            return tf.concat(candFillList,axis=1)
            
        return keras.layers.Lambda(merge)(candidates)
        
    def addE(self,cand):
        #need to assign a relatively large minimum mass here; otherwise mass calculation leads to nan
        return keras.layers.Lambda(lambda x: 
            tf.stack([tf.sqrt(tf.square(x[:,:,0])+tf.square(x[:,:,1])+tf.square(x[:,:,2])+0.01),x[:,:,0],x[:,:,1],x[:,:,2]],axis=2)
        )(cand)
        
    def sumComb(self,cand):
        return keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1,keep_dims=True))(cand)

    def calcM(self,cand):
        return keras.layers.Lambda(lambda x:
            tf.sqrt(tf.nn.relu(tf.square(x[:,:,0])-tf.square(x[:,:,1])-tf.square(x[:,:,2])-tf.square(x[:,:,3]))+0.01)
        )(cand)
        
    def calcPt(self,cand):
        return keras.layers.Lambda(lambda x:
            tf.sqrt(tf.square(x[:,:,1])+tf.square(x[:,:,2])+0.01)
        )(cand)
        
    def calcRap(self,cand):    
        return keras.layers.Lambda(lambda x:
            0.5*tf.abs(tf.log((x[:,:,0]+x[:,:,3]))/(x[:,:,0]-x[:,:,3]))
        )(cand)

    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen,cpf_p4,npf_p4,muon_p4,electron_p4):
        globalvars = self.global_preproc(globalvars)
        cpf = self.cpf_preproc(cpf)
        npf = self.npf_preproc(npf)
        sv = self.sv_preproc(sv)
        muon = self.muon_preproc(muon)
        electron = self.electron_preproc(electron)
        
        global_features = keras.layers.Concatenate(axis=1)([globalvars,gen])
        
        
        cand_features = self.mergeCandidatesSlim([
            cpf,
            npf,
            muon,
            electron
        ])
        
        cand_p4 = keras.layers.Concatenate(axis=1)([
            self.addE(cpf_p4),
            self.addE(npf_p4),
            self.addE(muon_p4),
            self.addE(electron_p4) 
        ])
        
        cand_values = self.applyLayers(cand_features,self.cand_conv)
        cand_attention = self.cand_qkAttention.getAttentionWeights(cand_features,global_features)
        
        cand_values = self.cand_qkAttention.applyAttentionToCandidates(cand_values,cand_attention)
        cand_p4 = self.cand_qkAttention.applyAttentionToP4(cand_p4,cand_attention)
        cand_tensor = keras.layers.Flatten()(cand_values)
        
        #cand_p4 = self.sumComb(cand_p4)
        
        cand_m = self.calcM(cand_p4)
        cand_pt = self.calcPt(cand_p4)
        cand_rap = self.calcRap(cand_p4)
        
        sv_features = self.applyLayers(sv,self.sv_conv)
        sv_features = keras.layers.Flatten()(sv_features)
        
        muon_features = self.applyLayers(muon,self.muon_conv)
        muon_features = keras.layers.Flatten()(muon_features)
        
        electron_features = self.applyLayers(electron,self.electron_conv)
        electron_features = keras.layers.Flatten()(electron_features)

        full_features = self.applyLayers([cand_m,cand_pt,cand_rap,cand_tensor,global_features,sv_features,muon_features,electron_features], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_tensor,npf_tensor,sv_conv,gen], self.full_features)

        return full_features
        
    def predictClass(self,globalvars,cpf,npf,sv,muon,electron,gen,cpf_p4,npf_p4,muon_p4,electron_p4):
        full_features = self.extractFeatures(globalvars,cpf,npf,sv,muon,electron,gen,cpf_p4,npf_p4,muon_p4,electron_p4)
        class_prediction = self.applyLayers(full_features,self.class_prediction)
        return class_prediction
        
        
    def makeClassModel(self):
        predictedClass = self.predictClass(
            self.input_globalvars,
            self.input_cpf,
            self.input_npf,
            self.input_sv,
            self.input_muon,
            self.input_electron,
            self.input_gen,
            self.input_cpf_p4,
            self.input_npf_p4,
            self.input_muon_p4,
            self.input_electron_p4
        )
        model = keras.models.Model(
            inputs=[
                self.input_gen,
                self.input_globalvars,
                self.input_cpf,
                self.input_npf,
                self.input_sv,
                self.input_muon,
                self.input_electron,
                self.input_cpf_p4,
                self.input_npf_p4,
                self.input_muon_p4,
                self.input_electron_p4
            ],
            outputs=[
                predictedClass
            ]
        )
        return model

network = MultiHeadAttentionNetwork


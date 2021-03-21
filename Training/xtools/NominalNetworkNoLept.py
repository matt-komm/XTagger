import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
import xtools

class NominalNetworkNoLept(xtools.NominalNetwork):
    def __init__(self,featureDict,wasserstein=False):
        xtools.NominalNetwork.__init__(self,featureDict,wasserstein)


    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen):
        globalvars = self.global_preproc(globalvars)
        cpf = self.cpf_preproc(cpf)
        npf = self.npf_preproc(npf)
        sv = self.sv_preproc(sv)
        muon = self.muon_preproc(muon)
        electron = self.electron_preproc(electron)
        
        #global_features = keras.layers.Concatenate(axis=1)([globalvars,gen])
        
        cpf = self.addToConvFeatures(cpf,gen)
        npf = self.addToConvFeatures(npf,gen)
        sv = self.addToConvFeatures(sv,gen)
        muon = self.addToConvFeatures(muon,gen)
        electron = self.addToConvFeatures(electron,gen)

        cpf_conv = self.applyLayers(cpf,self.cpf_conv)
        npf_conv = self.applyLayers(npf,self.npf_conv)
        sv_conv = self.applyLayers(sv,self.sv_conv)
        muon_conv = self.applyLayers(muon,self.muon_conv)
        electron_conv = self.applyLayers(electron,self.electron_conv)

        full_features = self.applyLayers([globalvars,cpf_conv,npf_conv,sv_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,gen], self.full_features)

        return full_features


network = NominalNetworkNoLept

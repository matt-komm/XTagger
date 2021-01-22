import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
import xtools

class NominalNetworkOnlyGlobal(xtools.NominalNetwork):
    def __init__(self,featureDict):
        xtools.NominalNetwork.__init__(self,featureDict)


    def returnsLogits(self):
        return False

    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen,cpf_p4,npf_p4,muon_p4,electron_p4):
        globalvars_preproc = self.global_preproc(globalvars)


        full_features = self.applyLayers([globalvars_preproc,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,gen], self.full_features)

        return full_features

network = NominalNetworkOnlyGlobal

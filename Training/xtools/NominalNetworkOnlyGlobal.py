import keras
import tensorflow as tf
from keras import backend as K
import os
import sys
import logging
import xtools

class NominalNetworkOnlyGlobal(xtools.NominalNetwork):
    def __init__(self,featureDict,wasserstein=False):
        xtools.NominalNetwork.__init__(self,featureDict,wasserstein)

    def extractFeatures(self,globalvars,cpf,npf,sv,muon,electron,gen):
        globalvars_preproc = self.global_preproc(globalvars)


        full_features = self.applyLayers([globalvars_preproc,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,muon_conv,gen], self.full_features)
        #full_features = self.applyLayers([globalvars_preproc,cpf_conv,npf_conv,sv_conv,gen], self.full_features)

        return full_features

network = NominalNetworkOnlyGlobal

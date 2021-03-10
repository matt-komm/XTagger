import argparse
import os
import sys
import tensorflow as tf
import keras
import numpy
import imp
from keras import backend as K
from feature_dict import featureDict
import xtools

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parametric', action='store_true',
                    dest='parametric',
                    help='train a parametric model', default=False)
parser.add_argument('networkfile', metavar='networkfile', type=str, nargs=1,
                    help='a network py file')
parser.add_argument('weightfile', metavar='weightfile', type=str, nargs=1,
                    help='a weight file from keras')
arguments = parser.parse_args()


Network = imp.load_source('Network', arguments.networkfile[0]).network

if not os.path.exists(arguments.weightfile[0]):
    print "Error - weight file '",arguments.weightfile[0],"' does not exists"
    sys.exit(1)
if not arguments.weightfile[0].endswith(".hdf5"):
    print "Error - file '",parser.weightfile[0],"' is not a hdf5 file"
    sys.exit(1)

sess = K.get_session()

tf_cpf = tf.placeholder('float32',shape=(None,featureDict["cpf"]["max"],len(featureDict["cpf"]["branches"])),name="cpf")
cpf = keras.layers.Input(tensor=tf_cpf)
#tf_cpf_p4 = tf.placeholder('float32',shape=(None,featureDict["cpf_p4"]["max"],len(featureDict["cpf_p4"]["branches"])),name="cpf_p4")
#cpf_p4 = keras.layers.Input(tensor=tf_cpf_p4)
tf_npf = tf.placeholder('float32',shape=(None,featureDict["npf"]["max"],len(featureDict["npf"]["branches"])),name="npf")
npf = keras.layers.Input(tensor=tf_npf)
#tf_npf_p4 = tf.placeholder('float32',shape=(None,featureDict["npf_p4"]["max"],len(featureDict["npf_p4"]["branches"])),name="npf_p4")
#npf_p4 = keras.layers.Input(tensor=tf_npf_p4)
tf_sv = tf.placeholder('float32',shape=(None,featureDict["sv"]["max"],len(featureDict["sv"]["branches"])),name="sv")
sv = keras.layers.Input(tensor=tf_sv)
tf_muon = tf.placeholder('float32',shape=(None,featureDict["muon"]["max"],len(featureDict["muon"]["branches"])),name="muon")
muon = keras.layers.Input(tensor=tf_muon)
#tf_muon_p4 = tf.placeholder('float32',shape=(None,featureDict["muon_p4"]["max"],len(featureDict["muon_p4"]["branches"])),name="muon_p4")
#muon_p4 = keras.layers.Input(tensor=tf_muon_p4)
tf_electron = tf.placeholder('float32',shape=(None,featureDict["electron"]["max"],len(featureDict["electron"]["branches"])),name="electron")
electron = keras.layers.Input(tensor=tf_electron)
#tf_electron_p4 = tf.placeholder('float32',shape=(None,featureDict["electron_p4"]["max"],len(featureDict["electron_p4"]["branches"])),name="electron_p4")
#electron_p4 = keras.layers.Input(tensor=tf_electron_p4)
tf_globalvars = tf.placeholder('float32',shape=(None,len(featureDict["globalvars"]["branches"])),name="globalvars")
globalvars = keras.layers.Input(tensor=tf_globalvars)

tf_gen = tf.placeholder('float32',shape=(None,1),name="gen")
gen = keras.layers.Input(tensor=tf_gen)


print "cpf shape: ",cpf.shape.as_list()
#print "cpf p4 shape: ",cpf_p4.shape.as_list()
print "npf shape: ",npf.shape.as_list()
#print "npf p4 shape: ",npf_p4.shape.as_list()
print "sv shape: ",sv.shape.as_list()
print "muon shape: ",muon.shape.as_list()
#print "muon p4 shape: ",muon_p4.shape.as_list()
print "electron shape: ",electron.shape.as_list()
#print "electron p4 shape: ",electron_p4.shape.as_list()
print "globalvars shape: ",globalvars.shape.as_list()
print "gen shape: ",gen.shape.as_list()

network = Network(featureDict)
print "learning phase: ",sess.run(keras.backend.learning_phase())

class_prediction = network.predictClass(globalvars,cpf,npf,sv,muon,electron,gen)
    
prediction = tf.identity(class_prediction,name="prediction")

model = keras.Model(inputs=[gen, globalvars, cpf, npf, sv, muon, electron], outputs=class_prediction)

train_writer = tf.summary.FileWriter("graph",sess.graph)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
train_writer.close()

def shape(tf_tensor):
    dims = tf_tensor.shape.as_list()
    dims[0] = 1
    return dims

model.load_weights(arguments.weightfile[0])


#test if graph can be executed
feed_dict={
    tf_gen:numpy.zeros(shape(tf_gen)),
    tf_globalvars:numpy.zeros(shape(tf_globalvars)),
    tf_cpf:numpy.zeros(shape(tf_cpf)),
    tf_npf:numpy.zeros(shape(tf_npf)),
    tf_sv:numpy.zeros(shape(tf_sv)),
    tf_muon:numpy.zeros(shape(tf_muon)),
    tf_electron:numpy.zeros(shape(tf_electron)),
    #tf_cpf_p4:numpy.zeros(shape(tf_cpf_p4)),
    #tf_npf_p4:numpy.zeros(shape(tf_npf_p4)),
    #tf_muon_p4:numpy.zeros(shape(tf_muon_p4)),
    #tf_electron_p4:numpy.zeros(shape(tf_electron_p4)),
}

prediction_val = sess.run(
    prediction,
    feed_dict=feed_dict
)

#print [n.name for n in sess.graph.as_graph_def().node]

const_graph = tf.graph_util.convert_variables_to_constants(
    sess,
    sess.graph.as_graph_def(),
    ["prediction"]
)

tf.train.write_graph(const_graph,"",arguments.weightfile[0].replace("hdf5","pb"),as_text=False)

print "Sucessfully saved graph and weights into '%s'"%arguments.weightfile[0].replace("hdf5","pb")


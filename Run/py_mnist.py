#imports and add module path to sys
import sys
import os
sys.path.insert(0, os.getcwd()+"/../build")


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import binding
import numpy as np
import binding as bd;
from train_functions import *


# load data
from load_mnist import load_data_wrapper
train_data,val_data,test_data=load_data_wrapper()

# set up training parameters
epochs=10
batchsize=50
input_shape=(batchsize,train_data[0].shape[1])

# set up NETWORK
flag_host=False;
learning_rate=0.1;

nn2=bd.neural_network(learning_rate,flag_host);
nn2.add_layer(bd.linear("lin1",input_shape[1],64));
nn2.add_layer(bd.relu("relu1"));
nn2.add_layer(bd.linear("linear2",64,10));
nn2.add_layer(bd.softmax("softmax1"));
nn2.set_cost(bd.cce_cost());


#set up mnist batches
batches=[]
n_batches=int(train_data[0].shape[0]/batchsize)

for batch in range(n_batches):
    input=train_data[0][batch*batchsize:(batch+1)*batchsize]
    output=train_data[1][batch*batchsize:(batch+1)*batchsize]
    val_in=val_data[0][:batchsize]
    val_out=val_data[1][:batchsize]
    batches+=[(input,output,val_in,val_out)]

if(flag_host):
    hardware='Host'
else:
    hardware='Device'

#train and get accuracy
print('Starting Training on ',hardware)
loss=train(nn2,epochs,batches,flag_host)
print("Accuracy: ",get_test_accuracy(nn2,test_data[0][:batchsize],test_data[1][:batchsize],flag_host))

# see output for 10 samples
print('______________________________________________________________')
print('Output for 10 samples: \n')

for i in range(10):
   print(i)
   print('Label',np.array(batches[i][1])[0])
   m_input=bd.matrix(batches[i][0])
   if not flag_host:
       m_input.copy_host_to_device()
   m_output=nn2.prop_forward(m_input)
   m_output.copy_device_to_host()
   print('NN', np.array(m_output)[0])
   print('\n')


# Kotrollimplementierung keras
print("Keras Kontrollimplementierung")
print('______________________________________________________________')
inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
outputs = layers.Dense(10,activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
    metrics=[keras.metrics.CategoricalCrossentropy(),"accuracy"],
)

hist=[]
for e in range(epochs):
    for batch in batches:
        hist+=[model.train_on_batch(batch[0],batch[1])]

print('Keras Accuracy: ', np.array(hist)[-10:,2].mean())

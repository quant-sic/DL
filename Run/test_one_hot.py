# imports and add module path to sys
import sys
import os
sys.path.insert(0, os.getcwd()+"/../build")

# import binding +tests
import numpy as np
import binding as bd;

from train_functions import *


# load data
from load_mnist import load_data_wrapper
train_data,val_data,test_data=load_data_wrapper()


# Try out easy example
# set up NETWORK
flag_host=False;
learning_rate=0.1;
nn2=bd.neural_network(learning_rate,flag_host);
nn2.add_layer(bd.linear("lin1",10,10));
nn2.add_layer(bd.sigmoid("relu"));
nn2.add_layer(bd.linear("lin1",10,10));
nn2.add_layer(bd.sigmoid("sigmoid"));
nn2.set_cost(bd.rms_cost());



#set up training parameters
epochs=150
n_batches=100
batchsize=30

# set up batches
batches=[]
for batch in range(n_batches):
    choices=np.random.randint(10,size=batchsize)
    input=np.vstack([np.eye(1,10,choices[i]) for i in range(batchsize)])
    batches+=[(input,input,input,input)]

#train the network
loss=train(nn2,epochs,batches,flag_host)

# must of course be smaller than epochs
average_over=10
achieved_loss=np.array(loss)[-average_over:].mean()

print("On Device Network achieved a loss of: ",achieved_loss," on Dummy example")

print("End")

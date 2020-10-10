import sys
import os
sys.path.insert(0, os.getcwd()+"/../build")

# import binding +tests
import numpy as np
import binding as bd;


# load data
from load_mnist import load_data_wrapper
train_data,val_data,test_data=load_data_wrapper()

epochs=3
batchsize=250
input_shape=(batchsize,train_data[0].shape[1])

# Try out easy example
# set up NETWORK
flag_host=False;
learning_rate=0.4;

nn2=bd.neural_network(learning_rate,flag_host);
nn2.add_layer(bd.linear("lin1",input_shape[1],500));
nn2.add_layer(bd.sigmoid("sigmoid1"));
nn2.add_layer(bd.linear("lin2",500,300));
nn2.add_layer(bd.sigmoid("sigmoid2"));
nn2.add_layer(bd.linear("lin3",300,10));
nn2.set_cost(bd.cce_soft_cost());


# train method
def train(net,epochs,batches,flag_host):
    costs=[]
    for e in range(epochs):
        av_cost=0
        for batch in batches:
            train_on_batch(net,batch[0],batch[1],flag_host);

        costs+=[validate_net(net,batch[2],batch[3],flag_host)];
        print("Epoch: ",e,"; Cost: ",costs[e])

    return costs


# validate method
def validate_net(net,val_input,val_labels,flag_host):

    if not flag_host:
        val_input.copy_host_to_device()
        val_labels.copy_host_to_device()

    m_output=net.prop_forward(val_input)
    return net.get_cost().cost(m_output,val_labels,flag_host)


#train on batch
def train_on_batch(net,batch_input,batch_labels,flag_host):

    if not flag_host:
        batch_input.copy_host_to_device()
        batch_labels.copy_host_to_device()

    m_output=net.prop_forward(batch_input)
    net.prop_backward(m_output,batch_labels)

# calculate accuracy
def get_test_accuracy(net,test_input,test_labels,flag_host):
    m_input=bd.matrix(test_input)
    m_labels=bd.matrix(test_labels)

    if not flag_host:
        m_input.copy_host_to_device()

    m_output=net.prop_forward(m_input)
    m_output.copy_device_to_host()
    output=np.array(m_output)
    print((np.argmax(output,axis=1)==np.argmax(test_labels,axis=1)).astype(int).sum())
    return float((np.argmax(output,axis=1)==np.argmax(test_labels,axis=1)).astype(int).sum())/(test_input.shape[0])


#set up examples
batches=[]
stride=int(train_data[0].shape[0]/batchsize)
for batch in range(stride):
    batches+=[(bd.matrix(train_data[0][batch*stride:(batch+1)*stride]),\
                bd.matrix(train_data[1][batch*stride:(batch+1)*stride]),\
                bd.matrix(val_data[0][:stride]),\
                bd.matrix(val_data[1][:stride]))]


#train and get accuracy
if(flag_host):
    hardware='Host'
else:
    hardware='Device'

print('Starting Training on ',hardware)
loss=train(nn2,epochs,batches,flag_host)
print("Accuracy: ",get_test_accuracy(nn2,test_data[0][:stride],test_data[1][:stride],flag_host))

#keep this to know programme has finished before Segmentation fault
print("End")

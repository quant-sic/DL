import sys
import os
sys.path.insert(0, os.getcwd()+"/../build")

import numpy as np
import binding as bd;

# def get_matrix(array,flag_host):
#     # get c++ object
#     matrix=bd.matrix(array)
#     # copy data onto device
#     if flag_host:
#         matrix.copy_host_to_device()
#
#     return matrix


# train method
def train(net,epochs,batches,flag_host,print_flag=True):
    costs=[]
    for e in range(epochs):
        av_cost=0
        for batch in batches:
            train_on_batch(net,batch[0],batch[1],flag_host);

        costs+=[validate_net(net,batch[2],batch[3],flag_host)];

        if print_flag:
            print("Epoch: ",e,"; Cost: ",costs[e])

    return costs


# validate method
def validate_net(net,val_input,val_labels,flag_host):
    m_val_input=bd.matrix(val_input)
    m_val_labels=bd.matrix(val_labels)

    if not flag_host:
        m_val_input.copy_host_to_device()
        m_val_labels.copy_host_to_device()

    m_output=net.prop_forward(m_val_input)
    return net.get_cost().cost(m_output,m_val_labels,flag_host)


# train on batch
def train_on_batch(net,batch_input,batch_labels,flag_host):
    m_batch_input=bd.matrix(batch_input)
    m_batch_labels=bd.matrix(batch_labels)

    if not flag_host:
        m_batch_input.copy_host_to_device()
        m_batch_labels.copy_host_to_device()

    m_output=net.prop_forward(m_batch_input)
    net.prop_backward(m_output,m_batch_labels)

# #     #train on batch
# def train_on_batch(net,batch_input,batch_labels,flag_host):
#     m_batch_input=get_matrix(batch_input,flag_host)
#     m_batch_labels=get_matrix(batch_labels,flag_host)
#
#     m_output=net.prop_forward(m_batch_input)
#     net.prop_backward(m_output,m_batch_labels)

# calculate accuracy
def get_test_accuracy(net,test_input,test_labels,flag_host):
    m_input=bd.matrix(test_input)
    m_labels=bd.matrix(test_labels)

    if not flag_host:
        m_input.copy_host_to_device()

    m_output=net.prop_forward(m_input)
    m_output.copy_device_to_host()
    output=np.array(m_output)
    return float((np.argmax(output,axis=1)==np.argmax(test_labels,axis=1)).astype(int).sum())/(test_input.shape[0])

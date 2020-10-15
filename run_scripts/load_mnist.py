import pickle
import gzip
import numpy as np
import os
import sys

mnist_data_location=os.getcwd()+'../data/mnist.pkl.gz';

def load_data():

    f = gzip.open(mnist_data_location, 'rb')

    if sys.version_info[0]>=3:
        train_data, val_data, test_data = pickle.load(f,encoding='bytes')
    else:
        train_data, val_data, test_data = pickle.load(f)

    f.close()
    return (train_data, val_data, test_data)


def load_data_wrapper():

    train_data, val_data, test_data=load_data();

    train_input=train_data[0]
    train_labels=np.array([np.eye(1,10,x).reshape(-1) for x in train_data[1]])

    val_input=val_data[0]
    val_labels=np.array([np.eye(1,10,x).reshape(-1) for x in val_data[1]])

    test_input=test_data[0]
    test_labels=np.array([np.eye(1,10,x).reshape(-1) for x in test_data[1]])

    return (train_input,train_labels),(val_input,val_labels),(test_input,test_labels)

# imports and add module path to sys
import sys
import os
sys.path.insert(0, os.getcwd()+"/../build")

# import binding +tests
import numpy as np
import binding as bd;
import itertools
from train_functions import *

# import keras and tensorflow
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras


for i in range(100):
    input=np.random.random(size=(np.random.randint(10,1000),np.random.randint(10,1000)));
    target=np.random.random(size=(np.random.randint(10,1000),np.random.randint(10,1000)));

    m_target=bd.matrix(target);
    m_input=bd.matrix(input);

    if(not np.allclose(input,np.array(m_input))):
        print('Matrix hand over does not work!')

print('Done')

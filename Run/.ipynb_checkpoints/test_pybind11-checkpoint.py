import sys
import os
sys.path.insert(0, os.getcwd()+"/../build")

# import binding +tests
import numpy as np
import binding as bd;

input=np.random.random(size=(10,10));
target=np.random.random(size=(10,10));

m_target=bd.matrix(target);
m_input=bd.matrix(input);

print('Matrix hand over works ',np.allclose(input,np.array(m_input)))


# load data
from load_mnist import load_data_wrapper
train,val,test=load_data_wrapper()


# Try out easy example
# set up NETWORK
flag_host=True;
learning_rate=0.1;
nn2=bd.neural_network(learning_rate,flag_host);
nn2.add_layer(bd.linear("lin1",10,10));
nn2.add_layer(bd.sigmoid("sigmoid"));
nn2.set_cost(bd.rms_cost());

def train(net,epochs,batches,flag_host):
    costs=[]
    for e in range(epochs):
        av_cost=0
        for batch in batches:
            av_cost+=train_on_batch(net,batch[0],batch[1],flag_host);
        costs+=[av_cost/len(batches)]
    return costs

def train_on_batch(net,batch_input,batch_labels,flag_host):
    m_input=bd.matrix(batch_input)
    m_labels=bd.matrix(batch_labels)
    m_output=net.prop_forward(m_input)
    net.prop_backward(m_output,m_labels)
    cost=net.get_cost().cost(m_output,m_labels,flag_host)
    return cost

epochs=100
n_batches=100
batchsize=30

batches=[]
for batch in range(n_batches):
    choices=np.random.randint(10,size=batchsize)
    input=np.vstack([np.eye(1,10,choices[i]) for i in range(batchsize)])
    batches+=[(input,input)]

loss=train(nn2,epochs,batches,flag_host)
print(loss)

print("End")

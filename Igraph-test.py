
# coding: utf-8

# In[19]:

import igraph as ig
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[7]:

erd = ig.Graph.Erdos_Renyi(100, m=400)
adj = np.array(erd.get_adjacency())


# In[23]:

ks = np.array(erd.degree())
print(ks)
degnn = np.zeros((2, ks.max()+1), dtype='float')
for i in range(100):
    nns = erd.neighbors(i)
    avg_k = np.array([ks[m] for m in nns])
    degnn[0, ks[i]] += avg_k.mean()
    degnn[1, ks[i]] += 1 
there = np.where(degnn[1] == 0)[0]
degnn[1, there] = 1
degnn[0] /= degnn[1]

plt.scatter(range(ks.max()+1), degnn[0])


# In[25]:

def NNdeg(graph, Plot=False):
    '''
    creates an array with the mean degree of the neighbors of nodes with the same degree
    '''
    ks = np.array(graph.degree())
    degnn = np.zeros((2, ks.max()+1), dtype='float')
    for i in range(100):
        nns = graph.neighbors(i)
        avg_k = np.array([ks[m] for m in nns])
        degnn[0, ks[i]] += avg_k.mean()
        degnn[1, ks[i]] += 1 
    there = np.where(degnn[1] == 0)[0]
    degnn[1, there] = 1
    degnn[0] /= degnn[1]
    if Plot:
        plt.scatter(range(ks.max()+1), degnn[0])
    return degnn[0]


# In[26]:

red = NNdeg(erd, Plot=True)


# In[18]:

nns = erd.neighbors(0)
avg_k = [ks[m] for m in nns]
print(nns)
print(avg_k)
np.array(avg_k).mean()


# In[ ]:




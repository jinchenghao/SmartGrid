# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

c=2
T=24
m=0#因子为零表示不考虑同伴效应的作用
n=20
x_sum=[]

def getResults(k1,k2):
    list_a = pd.read_csv("..//data//a%s-%s.csv"%(k1,k2),header=-1)
    list_b = pd.read_csv("..//data//b%s-%s.csv"%(k1,k2),header=-1)
    
    for i in range(0,list_a.shape[0]):
#        print(list_a.iloc[i,:])
        vector_a = np.array(list_a.iloc[i,:])
        vector_b = np.array(list_b.iloc[i,:])
        matrix_b = np.matrix(np.multiply(2*vector_b,np.identity(n-1)))
        #计算p
        h = np.dot(np.dot(np.ones(n-1),matrix_b.I),vector_a)
        l = np.dot(np.dot(np.ones(n-1),matrix_b.I),np.ones(n-1))
        p = (h*(1+2*c*l))/(2*l*(1+c*l))
         #计算x_k
        x_k = np.dot(matrix_b.I,(vector_a-p*np.ones(n-1)).reshape(-1, 1))
        x_sum.append(np.dot(np.ones(n-1),np.array(x_k)))

getResults(1,9)
getResults(9,16)
getResults(16,17)
getResults(17,21);
getResults(21,24)
getResults(24,25)

print(x_sum)

fig = plt.figure(figsize=(18, 8))  
ax = plt.subplot(111)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='red',marker='o',label='Electicity Load')
plt.legend(fontsize='xx-large')
plt.ylabel("Total consumption",fontsize=20)
plt.xlabel("Time slot",fontsize=20)
plt.ylim(0,5)
plt.savefig("p1.pdf")
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

p=8

c=2
T=24
m=0.05#因子为零表示不考虑同伴效应的作用
n=20
x_sum=[]
x_sum_2=[]
x_sum_4=[]
t = np.matrix(np.multiply(m,np.identity(n)))

w = np.matrix(np.multiply(1/(n),np.ones((n,n)))) - np.matrix(np.multiply(1/(n),np.identity(n)))

#第三个图
def getResults(k1,k2):
    list_a = pd.read_csv("..//data//a%s-%s.csv"%(k1,k2),header=-1)
    list_b = pd.read_csv("..//data//b%s-%s.csv"%(k1,k2),header=-1)
    
    for i in range(0,list_a.shape[0]):
#        print(list_a.iloc[i,:])
        vector_a = np.array(list_a.iloc[i,:])
        vector_b = np.array(list_b.iloc[i,:])
        matrix_b = np.matrix(np.multiply(2*vector_b,np.identity(n)))
        matrix_b_1 = matrix_b+2*t-np.dot(t,w)
        #计算p
        h = np.dot(np.dot(np.ones(n),matrix_b_1.I),vector_a)
        l = np.dot(np.dot(np.ones(n),matrix_b_1.I),np.ones(n))
        p = (h*(1+2*c*l))/(2*l*(1+c*l))
         #计算x_k
        x_k = np.dot(matrix_b_1.I,(vector_a-p*np.ones(n)).reshape(-1, 1))
        x_sum.append(np.dot(np.ones(n),np.array(x_k)))
#第二个图     
def getResults1(k1,k2):
    list_a = pd.read_csv("..//data//a%s-%s.csv"%(k1,k2),header=-1)
    list_b = pd.read_csv("..//data//b%s-%s.csv"%(k1,k2),header=-1)
    
    for i in range(0,list_a.shape[0]):
#        print(list_a.iloc[i,:])
        vector_a = np.array(list_a.iloc[i,:])
        vector_b = np.array(list_b.iloc[i,:])
        matrix_b = np.matrix(np.multiply(2*vector_b,np.identity(n)))
        #计算p
        h = np.dot(np.dot(np.ones(n),matrix_b.I),vector_a)
        l = np.dot(np.dot(np.ones(n),matrix_b.I),np.ones(n))
        p = (h*(1+2*c*l))/(2*l*(1+c*l))
         #计算x_k
        x_k = np.dot(matrix_b.I,(vector_a-p*np.ones(n)).reshape(-1, 1))
        x_sum_2.append(np.dot(np.ones(n),np.array(x_k)))

#第一个图       
def getResults4(k1,k2):
    list_a = pd.read_csv("..//data//a%s-%s.csv"%(k1,k2),header=-1)
    list_b = pd.read_csv("..//data//b%s-%s.csv"%(k1,k2),header=-1)
    for i in range(0,list_a.shape[0]):
#        print(list_a.iloc[i,:])
        sum_=0
        for j in range(0,list_a.shape[1]):
            a = list_a.iloc[i,j]
            b = list_b.iloc[i,j]
            sum_ = sum_+(a-p)/(2*b)
        x_sum_4.append(sum_)
#第三个图
getResults(1,9)
getResults(9,16)
getResults(16,17)
getResults(17,21);
getResults(21,24)
getResults(24,25)
#第二个图
getResults1(1,9)
getResults1(9,16)
getResults1(16,17)
getResults1(17,21);
getResults1(21,24)
getResults1(24,25)
#第一个图


fig = plt.figure(figsize=(18, 8))  
ax = plt.subplot(111)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='red',marker='o',label='Electicity Load')
line1,=ax.plot(x_sum_2, ls='-',lw=2.8,color='blue',marker='o',label='Electicity Load')
line1,=ax.plot(x_sum_4, ls='-',lw=2.8,color='black',marker='o',label='Electicity Load')
plt.legend(fontsize='xx-large')
plt.ylabel("Total consumption",fontsize=20)
plt.xlabel("Time slot",fontsize=20)
#plt.ylim(0,5)
plt.savefig("p1.pdf")
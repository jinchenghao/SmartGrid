# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

c=2
T=24
m=0.5#因子为零表示不考虑同伴效应的作用
n=20
x_sum=[]
#构造t矩阵
t = np.matrix(np.multiply(m,np.identity(n-1)))
#构造w矩阵，密集矩阵
w0 = np.matrix(np.multiply(1/(n-1),np.ones((n-1,n-1)))) - np.matrix(np.multiply(1/(n-1),np.identity(n-1)))
#构造w矩阵，稀疏矩阵
w1 = np.zeros((n-1,n-1))
vector_w = (1/(n-1))*np.ones(n-1)
#哪一行赋值，此时为第5行
w1[5,:] = vector_w
w1[5,5] = 0

def getResults(k1,k2,w):
    list_a = pd.read_csv("..//data//a%s-%s.csv"%(k1,k2),header=-1)
    list_b = pd.read_csv("..//data//b%s-%s.csv"%(k1,k2),header=-1)
    
    for i in range(0,list_a.shape[0]):
        #计算向量a和矩阵b
        vector_a = np.array(list_a.iloc[i,:])
        vector_b = np.array(list_b.iloc[i,:])
        matrix_b = np.matrix(np.multiply(2*vector_b,np.identity(n-1)))
        #使用w和t重新计算矩阵b
        matrix_b_1 = matrix_b+2*t-t.dot(w)
        #计算p
        h = np.dot(np.dot(np.ones(n-1),matrix_b_1.I),vector_a)
        l = np.dot(np.dot(np.ones(n-1),matrix_b_1.I),np.ones(n-1))
        p = (h*(1+2*c*l))/(2*l*(1+c*l))
         #计算x_k
        x_k = np.dot(matrix_b_1.I,(vector_a-p*np.ones(n-1)).reshape(-1, 1))
        x_sum.append(np.dot(np.ones(n-1),np.array(x_k)))

#画密集矩阵w0的图
getResults(1,9,w0)
getResults(9,16,w0)
getResults(16,17,w0)
getResults(17,21,w0);
getResults(21,24,w0)
getResults(24,25,w0)
print(x_sum)
print("--------------------------------------")
fig = plt.figure(figsize=(18, 8))  
ax = plt.subplot(111)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='red',marker='o',label='Electicity Load')

#画稀疏矩阵w1的图
x_sum = []
getResults(1,9,w1)
getResults(9,16,w1)
getResults(16,17,w1)
getResults(17,21,w1);
getResults(21,24,w1)
getResults(24,25,w1)
print(x_sum)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='blue',marker='o',label='Electicity Load')
plt.legend(fontsize='xx-large')
plt.ylabel("Total consumption",fontsize=20)
plt.xlabel("Time slot",fontsize=20)
plt.ylim(0,5)
plt.savefig("p1.pdf")
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

c=2
T=24
gamma=1#因子为零表示不考虑同伴效应的作用
n=20
lamda = 0.3
#构造t矩阵
gamma_m = np.matrix(np.multiply(gamma,np.identity(n)))
#构造w矩阵，密集矩阵
w0 = np.matrix(np.multiply(1/(n),np.ones((n,n)))) - np.matrix(np.multiply(1/(n),np.identity(n)))
#构造w矩阵，稀疏矩阵
w1 = np.zeros((n,n))
vector_w = (1/(n))*np.ones(n)
#哪一行赋值，此时为第5行
w1[2,:] = vector_w
w1[2,2] = 0
#构造lamda
lamda_m = np.matrix(np.multiply(lamda,np.identity(n)))


x_sum=[]
def getResults(k1,k2,w,p=0):
    list_a = pd.read_csv("..//data//a%s-%s.csv"%(k1,k2),header=-1)
    list_b = pd.read_csv("..//data//b%s-%s.csv"%(k1,k2),header=-1)
    list_d = pd.read_csv("..//data//d.csv",header=-1)
    vector_d = np.array(list_d)
    #print(vector_d)
    
    M = lamda_m+2*gamma_m-gamma_m*w
    x_k=np.zeros((1,n))
    y = np.zeros((1,n))
    for k in range(0,list_a.shape[0]):
        #计算向量a和矩阵b
        vector_a = np.array(list_a.iloc[k,:])
        vector_b = np.array(list_b.iloc[k,:])
        matrix_b = np.matrix(np.multiply(2*vector_b,np.identity(n)))
        #使用w和t重新计算矩阵b
        H = (matrix_b+M).I
        y = y+x_k
        #计算p
        l = np.dot(np.dot(np.ones(n),H),(vector_a[:,None]+np.dot(((k/T)*lamda_m),vector_d)-np.dot(M,y.reshape(-1,1))))
        #l = np.dot(np.dot(np.ones(n),H),np.dot((vector_a+((k/T)*lamda_m),(vector_d))))
        h = np.dot(np.dot(np.ones(n),H),np.ones(n))
        if p!=3:
            p = (l+2*c*l*h)/(2*h+2*c*h*h)
         #计算x_k
        #print((p*np.ones(n)).reshape(-1,1))
        x_k = (np.dot((matrix_b+M).I,(vector_a[:,None]-(p*np.ones(n)).reshape(-1,1)+np.dot(((k/T)*lamda_m),vector_d)-np.dot(M,y.reshape(-1,1))))).reshape(1,-1)
        print(x_k)
        
        x_sum.append(np.dot(np.ones(n),np.array(x_k.reshape(-1,1))))

#试验3
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

#试验2
c=2
T=24
gamma=0#因子为零表示不考虑同伴效应的作用
#构造t矩阵
gamma_m = np.matrix(np.multiply(gamma,np.identity(n)))
#构造w矩阵，密集矩阵
w0 = np.matrix(np.multiply(1/(n),np.ones((n,n)))) - np.matrix(np.multiply(1/(n),np.identity(n)))
#构造w矩阵，稀疏矩阵
w1 = np.zeros((n,n))
vector_w = (1/(n))*np.ones(n)
#哪一行赋值，此时为第5行
w1[2,:] = vector_w
w1[2,2] = 0
x_sum = []
getResults(1,9,w0)
getResults(9,16,w0)
getResults(16,17,w0)
getResults(17,21,w0);
getResults(21,24,w0)
getResults(24,25,w0)
print(x_sum)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='blue',marker='o',label='Electicity Load')
plt.legend(fontsize='xx-large')
plt.ylabel("Total consumption",fontsize=20)
plt.xlabel("Time slot",fontsize=20)
plt.savefig("p1.pdf")

c=2
T=24
gamma=0#因子为零表示不考虑同伴效应的作用

#构造t矩阵
gamma_m = np.matrix(np.multiply(gamma,np.identity(n)))
#构造w矩阵，密集矩阵
w0 = np.matrix(np.multiply(1/(n),np.ones((n,n)))) - np.matrix(np.multiply(1/(n),np.identity(n)))
#构造w矩阵，稀疏矩阵
w1 = np.zeros((n,n))
vector_w = (1/(n))*np.ones(n)
#哪一行赋值，此时为第5行
w1[2,:] = vector_w
w1[2,2] = 0
x_sum = []
getResults(1,9,w0,3)
getResults(9,16,w0,3)
getResults(16,17,w0,3)
getResults(17,21,w0,3)
getResults(21,24,w0,3)
getResults(24,25,w0,3)
print(x_sum)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='black',marker='o',label='Electicity Load')
plt.legend(fontsize='xx-large')
plt.ylabel("Total consumption",fontsize=20)
plt.xlabel("Time slot",fontsize=20)
plt.savefig("p1.pdf")


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# 此程序生成符合要求的a和b的数据，一般不要再运行否则会覆盖原数据内容
# 本程序也会画图，以观察数据是否符合要求
# =============================================================================

c=2
T=24
m=0#因子为零表示不考虑同伴效应的作用
n=20
x_sum=[]
def getResults(k1,k2,loc_a,loc_b,scale=0.1):
    #建立两个容器保存生成的数据
    collection_a = []
    collection_b = []
    #将之前的for循环改为了while循环，k否则循环计数
    k=k1
    while k<k2:
        #将每次循环生成的20个a和b的值先暂存进容器中，后面借助这两个容器，构造向量a和矩阵b
        array_a = []
        array_b = []
        for i in range(1,n):
            a = np.random.normal(loc_a,scale)
            b = np.random.normal(loc_b,scale)
            array_a.append(a)
            array_b.append(b)
        #构造向量a
        vector_a = np.array(array_a)
        #构造对角矩阵b
        vector_b = np.array(array_b)
        matrix_b = np.matrix(np.multiply(2*vector_b,np.identity(n-1)))
        #计算p
        h = np.dot(np.dot(np.ones(n-1),matrix_b.I),vector_a)
        l = np.dot(np.dot(np.ones(n-1),matrix_b.I),np.ones(n-1))
        p = (h*(1+2*c*l))/(2*l*(1+c*l))
        #判断向量a和矩阵b是否满足要求
        flag = True
        for a_i, b_i in zip(array_a,array_b):
            if a_i<p or b_i<=0:
                flag = False
                break
        #如果不满足要求，则重新计算
        if flag==False:
            continue
        #如果满足要求，计算x_k和需要的点，并使计数值k+1接着计算下次的值
        x_k = np.dot(matrix_b.I,(vector_a-p*np.ones(n-1)).reshape(-1, 1))
        x_sum.append(np.dot(np.ones(n-1),np.array(x_k)))
        k = k+1
        #将本次循环得到的合理的a和b的值存入容器中，后面将容器内容输出到文件
        collection_a.append(array_a)
        collection_b.append(array_b)
    #循环结束后，24小时所有合理的a和b都已经在容器collection_a和b中了，建立文件并输出到文件中，以备后续使用
    file11 = pd.DataFrame(collection_a)
    file12 = pd.DataFrame(collection_b)
    file11.to_csv("..//data//a%s-%s.csv"%(k1,k2),header=None,index=0)
    file12.to_csv("..//data//b%s-%s.csv"%(k1,k2),header=None,index=0)

getResults(1,9,8.5,0.5,0.15)
getResults(9,16,9.6,0.8,0.25)
getResults(16,17,11.6,1.2,0.4)
getResults(17,21,14,1.5,0.5);
getResults(21,24,11.6,1.2,0.4)
getResults(24,25,9.6,0.8,0.25)


fig = plt.figure(figsize=(18, 8))  
ax = plt.subplot(111)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='red',marker='o',label='Electicity Load')
plt.legend(fontsize='xx-large')
plt.ylabel("Total consumption",fontsize=20)
plt.xlabel("Time slot",fontsize=20)
plt.ylim(0,5)
plt.savefig("p1.pdf")
        


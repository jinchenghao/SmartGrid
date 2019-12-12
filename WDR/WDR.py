# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

c=2
T=24
p=8 #%统一定价无波动，不参与博弈
n=20
x_sum=[]
def getResults(k1,k2):
    #读取文件到二维表中
    list_a = pd.read_csv("..//data//a%s-%s.csv"%(k1,k2),header=-1)
    list_b = pd.read_csv("..//data//b%s-%s.csv"%(k1,k2),header=-1)
# =============================================================================
#     根据文件内容循环计算每个点的值
#     list_a.shape[0]是返回行数
#     list_a.shape[1]是返回列数
# =============================================================================
    for i in range(0,list_a.shape[0]):
        sum_=0
        for j in range(0,list_a.shape[1]):
            a = list_a.iloc[i,j]#第[i,j]个元素
            b = list_b.iloc[i,j]
            sum_ = sum_+(a-p)/(2*b)
        x_sum.append(sum_)

getResults(1,9)
getResults(9,16)
getResults(16,17)
getResults(17,21);
getResults(21,24)
getResults(24,25)


fig = plt.figure(figsize=(18, 8))  
ax = plt.subplot(111)
line1,=ax.plot(x_sum, ls='-',lw=2.8,color='red',marker='o',label='Electicity Load')
plt.legend(fontsize='xx-large')
plt.ylabel("Total consumption",fontsize=20)
plt.xlabel("Time slot",fontsize=20)
#plt.ylim(0,1000)
plt.savefig("p1.pdf")
        


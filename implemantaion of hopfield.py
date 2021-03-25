# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:41:48 2019

@author: sakin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:47:18 2019

@author: sakin
"""

import numpy as np

# =============================================================================
# define 10 samples number one to 10 as 8*8 matrix
# =============================================================================
num0=np.array([[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]])
num1=np.array([[-1,-1,1,1,1,-1,-1,-1],[-1,-1,1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1]])
# =============================================================================
# convert nums to 64*1
# =============================================================================
s0=np.reshape(num0,(64,1))
s1=np.reshape(num1,(64,1))


s0t=np.reshape(s0,(1,64))
s1t=np.reshape(s1,(1,64))

w=s0*s0t +s1*s1t 

for i in range(64):
    for j in range(64):
        if i==j:
            w[i][j]=0
print(w)
#verify network with weight matrix w
out0=np.dot(s0t,w)
out1=np.dot(s1t,w)
#apply activation function
for i in range(64):
    if out0[0,i]>0:
        out0[0,i]=1
    else:
        out0[0,i]=-1
    if out1[0,i]>0:
        out1[0,i]=1
    else:
        out1[0,i]=-1
out0=np.reshape(out0,(64,1))
out1=np.reshape(out1,(64,1))
#if a=b=1 it means that network is correct
a=np.array_equal(out0,s0)
b=np.array_equal(out1,s1)
# =============================================================================
# after initializate weight matrix, start hopfield algorithm 
# =============================================================================
# add 20% noise for 64*1 input = disturbe 13 number of 64
num0_noisy=np.array([[1,1,-1,1,-1,1,1,1],[1,-1,1,1,1,-1,1,1],[1,-1,-1,-1,1,-1,-1,1],[1,1,1,-1,-1,1,1,1],[1,-1,1,-1,1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,1,1,-1,1,1,1],[1,1,1,1,1,1,1,1]])
num1_noisy=np.array([[-1,1,1,-1,1,-1,-1,-1],[-1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,1,1,-1,-1,-1],[-1,1,-1,1,1,-1,1,-1],[-1,-1,-1,-1,1,-1,1,-1],[-1,1,-1,1,-1,-1,-1,-1],[-1,-1,-1,-1,1,1,-1,-1],[-1,-1,1,1,1,-1,-1,-1]])

s0_noisy=np.reshape(num0_noisy,(64,1))
s1_noisy=np.reshape(num1_noisy,(64,1))

s0_noisyt=np.reshape(num0_noisy,(1,64))
s1_noisyt=np.reshape(num1_noisy,(1,64))
# =============================================================================
# apply hopfield algorithm, k+1=#of epochs that algorithm be applyed
# =============================================================================
yin0=np.ones(64*1)
yin0=np.reshape(yin0,(1,64))
y0=s0_noisyt
for k in range(1):
    for i in range(64):
        sum=0
        for j in range(64):
            sum=sum+(y0[0][j]*w[j][i])
        yin0[0][i]=y0[0][i]+sum
        if yin0[0][i]>0:
            y0[0][i]=1
        if yin0[0][i]==0:
            y0[0][i]=yin0[0][i]
        if yin0[0][i]<0:
            y0[0][i]=-1  

if (np.array_equal(y0,s0t)):
    print("0 stop")
    print("epoch 0:",k+1)
error0=y0-s0t

yin1=np.ones(64*1)
yin1=np.reshape(yin0,(1,64))
y1=s1_noisyt
for k in range(1):
    for i in range(64):
        sum=0
        for j in range(64):
            sum=sum+(y1[0][j]*w[j][i])
        yin1[0][i]=s1_noisyt[0][i]+sum
        if yin1[0][i]>0:
            y1[0][i]=1
        if yin1[0][i]==0:
            y1[0][i]=yin1[0][i]
        if yin1[0][i]<0:
            y1[0][i]=-1  

if (np.array_equal(y1,s1t)):
    print("1 stop")
    print("epoch 1:",k+1)
error=y1-s1t
print('error 1 :',error)
print('error 0:',error0)


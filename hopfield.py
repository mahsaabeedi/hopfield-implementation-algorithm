# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:47:18 2019

@author: sakin
"""

import numpy as np
import random
# =============================================================================
# define 10 samples number one to 10 as 8*8 matrix
# =============================================================================
num0=np.array([[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]])
num1=np.array([[-1,-1,1,1,1,-1,-1,-1],[-1,-1,1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1],[-1,-1,-1,1,1,-1,-1,-1]])
num2=np.array([[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,-1,-1,-1,-1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,-1,-1,-1,-1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1]])
num3=np.array([[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,-1,-1,-1,-1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,-1,-1,-1,-1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1]])
num4=np.array([[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1,1,1],[-1,-1,-1,-1,-1,-1,1,1],[-1,-1,-1,-1,-1,-1,1,1]])
num5=np.array([[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,-1,-1,-1,-1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,-1,-1,-1,-1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1]])
num6=np.array([[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,-1,-1,-1,-1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,-1,-1,1,1,-1],[-1,1,1,-1,-1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1]])
num7=np.array([[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,1],[-1,-1,-1,-1,-1,1,1,1],[-1,-1,-1,-1,1,1,1,-1],[-1,-1,-1,1,1,1,-1,-1],[-1,-1,1,1,1,-1,-1,-1],[-1,1,1,1,-1,-1,-1,-1],[-1,1,1,-1,-1,-1,-1,-1]])
num8=np.array([[1,1,1,1,1,1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,1,1,1,1,1,1]])
num9=np.array([[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,-1,-1,1,1,-1],[-1,1,1,-1,-1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,-1,-1,-1,-1,1,1,-1],[-1,1,1,1,1,1,1,-1],[-1,1,1,1,1,1,1,-1]])
# =============================================================================
# convert nums to 64*1
# =============================================================================
s0=np.reshape(num0,(64,1))
s1=np.reshape(num1,(64,1))
s2=np.reshape(num2,(64,1))
s3=np.reshape(num3,(64,1))
s4=np.reshape(num4,(64,1))
s5=np.reshape(num5,(64,1))
s6=np.reshape(num6,(64,1))
s7=np.reshape(num7,(64,1))
s8=np.reshape(num8,(64,1))
s9=np.reshape(num9,(64,1))
s0_noisy=s0
s1_noisy=s1

s0t=np.reshape(s0,(1,64))
s1t=np.reshape(s1,(1,64))
s2t=np.reshape(s2,(1,64))
s3t=np.reshape(s3,(1,64))
s4t=np.reshape(s4,(1,64))
s5t=np.reshape(s5,(1,64))
s6t=np.reshape(s6,(1,64))
s7t=np.reshape(s7,(1,64))
s8t=np.reshape(s8,(1,64))
s9t=np.reshape(s9,(1,64))

#w=(s0*s0t +s1*s1t +s2*s2t + s3*s3t +s4*s4t + s5*s5t +s6*s6t +s7*s7t +s8*s8t +s9*s9t)/128

w=s0*s0t +s1*s1t 

for i in range(64):
    for j in range(64):
        if i==j:
            w[i][j]=0

out0=np.dot(s0t,w)
out1=np.dot(s1t,w)
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
a=np.array_equal(out0,s0)
b=np.array_equal(out1,s1)
# =============================================================================
# after initializate weight matrix, start hopfield algorithm 
# =============================================================================
# add 20% noise for 64*1 input = disturbe 13 number of 64
s0_noisy=np.array([[1,1,-1,1,-1,1,1,1],[1,-1,1,1,1,-1,1,1],[1,-1,-1,-1,1,-1,-1,1],[1,1,1,-1,-1,1,1,1],[1,-1,1,-1,1,-1,1,1],[1,1,-1,-1,-1,-1,1,1],[1,1,1,1,-1,1,1,1],[1,1,1,1,1,1,1,1]])
s1_noisy=np.array([[-1,1,1,-1,1,-1,-1,-1],[-1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,1,1,-1,-1,-1],[-1,1,-1,1,1,-1,1,-1],[-1,-1,-1,-1,1,-1,1,-1],[-1,1,-1,1,-1,-1,-1,-1],[-1,-1,-1,-1,1,1,-1,-1],[-1,-1,1,1,1,-1,-1,-1]])
s0_noisy=np.reshape(s0_noisy,(1,64))
s1_noisy=np.reshape(s1_noisy,(1,64))
#y=np.ones(64*1)
#y=np.reshape(y,(1,64))
yin=np.ones(64*1)
yin=np.reshape(yin,(1,64))
y=s0_noisy
#w=w.astype('float64')
for k in range(500):
    for i in range(64):
        sum=0
        for j in range(64):
            sum=sum+(y[0][j]*w[j][i])
        yin[0][i]=s0_noisy[0][i]+sum
        if yin[0][i]>0:
            y[0][i]=1
        if yin[0][i]==0:
            y[0][i]=yin[0][i]
        if yin[0][i]<0:
            y[0][i]=-1  

if (np.array_equal(y,s0t)):
    print("stop")
    
   
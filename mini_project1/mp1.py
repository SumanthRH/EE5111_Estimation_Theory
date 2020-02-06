import numpy as np
import os 
import argparse
import cmath
j= complex(0,1)
Y_SIZE=512
H_SIZE=32
SIG = 0.005
LAMBDA = 0.2

## Constructing X and F
x = np.random.choice([1+1j,1-1j,-1+1j,-1-1j],size=Y_SIZE)
X = np.diag(x,k=0)
F = np.empty((Y_SIZE,H_SIZE),dtype=np.csingle)
for i in range(F.shape[0]):
    for k in range(F.shape[1]):
        F[i][k] = cmath.exp(1j*(2*cmath.pi*i*k/512))
n = np.random.normal(scale=SIG**2,size=Y_SIZE)

## Constructing h
p = np.asarray([np.exp(-1*(LAMBDA)*i) for i in range(H_SIZE)])
p = p/np.sum(p**2)
a = np.random.normal(scale=0.5,size=H_SIZE)
b = np.random.normal(scale=0.5,size=H_SIZE)

h = np.multiply(p,a+b*j)

## Observations 
A = np.matmul(X,F)

y = np.matmul(A,h) + n

## LSE

h_,d,r,s = np.linalg.lstsq(A,y)
# d = np.sqrt(np.sum((h-h_)**2)/np.sum(h**2))

print(np.matmul(A,h)[:4])
print("h vector was : ",h)
print("h_ vector is :",h_)
print("Normalized difference :",d)


### Sparse h 
h2 = h.copy()
for i in np.random.randint(low=0,high=H_SIZE,size=H_SIZE-6):
    h2[i] = 0

y2 = np.matmul(A,h2) + n

h_,d,r,s = np.linalg.lstsq(A,y2)

print("Sparse h vector was : ",h2)
print(" h_ vector is :",h_)



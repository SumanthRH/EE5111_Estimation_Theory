
# coding: utf-8

import numpy as np
import os 
import argparse
import cmath
import random
from numpy.linalg import norm
from numpy.linalg import inv
from numpy import matmul, transpose
j = complex(0,1)

Y_SIZE = 512
H_SIZE = 32
SIG1 = np.sqrt(0.001)
SIG2 = np.sqrt(0.1)
LAMBDA = 0.2


# Constructing X and F
x = np.random.choice([1+1j,1-1j,-1+1j,-1-1j],size=Y_SIZE)
X = np.diag(x,k=0)
F = np.empty((Y_SIZE,H_SIZE),dtype=np.csingle)
for i in range(F.shape[0]):
    for k in range(F.shape[1]):
        F[i][k] = cmath.exp(1j*(2*cmath.pi*i*k/512))        

# Noise vectors of different variances
# n1 = np.random.normal(loc=0, scale=SIG1,size=(Y_SIZE,2)).view(np.complex128)
# n2 = np.random.normal(loc=0, scale=SIG2,size=(Y_SIZE,2)).view(np.complex128)

# Constructing h
p = np.asarray([np.exp(-1*(LAMBDA)*i) for i in range(H_SIZE)])
p = p/np.sum(p**2)
a = np.random.normal(scale=0.5,size=H_SIZE)
b = np.random.normal(scale=0.5,size=H_SIZE)

# Reshape added else y was taking size 512*512
h = np.multiply(p,a+b*j).reshape(-1,1)

# Calculate A, where A = XF and y = XFh + n
A = matmul(X,F)


def pinv(A, alpha=0):
    Ah = np.asmatrix(A).getH()
    return  matmul(inv(matmul(Ah,A)+alpha*np.eye(np.shape(Ah)[0])),Ah)


def normalized_difference(y_original, y_hat):
    return norm(y_original-y_hat)/norm(y_original)


# Q1
def estimate_vanilla_h(A=A,h=h,sig=SIG1):
    n = np.random.normal(loc=0, scale=sig,size=(Y_SIZE,2)).view(np.complex128)
    y = matmul(A,h) + n
    h_hat = matmul(pinv(A), y)
    d = normalized_difference(h,h_hat)
    print(f"[Vanilla] Normalized difference for variance {np.round(sig**2, decimals=3)} : {d}")
    return h_hat,d
    
estimate_vanilla_h(sig=SIG1)
estimate_vanilla_h(sig=SIG2)


### Q2
### Sparse h
# h_sparse = h.copy()

# sparsity_points = [i for i in random.sample(range(0,H_SIZE),H_SIZE-6)]
# for i in sparsity_points:
#     h_sparse[i] = 0

# y1_sparse = np.matmul(A,h_sparse) + n1
# y2_sparse = np.matmul(A,h_sparse) + n2

# constraints_matrix = np.zeros((H_SIZE-6, H_SIZE))
# for i in range(len(sparsity_points)):
#     constraints_matrix[i,sparsity_points[i]]=1

# h_hat1_unconstrained = matmul(pinv(A), y1_sparse)
# Ah = np.asmatrix(A).getH()
# lambda_1 = matmul(2*inv(matmul(matmul(constraints_matrix, inv(matmul(Ah, A))),transpose(constraints_matrix))), matmul(constraints_matrix, h_hat1_unconstrained))
# h_hat1_constrained = h_hat1_unconstrained - 0.5*matmul(matmul(inv(matmul(Ah,A)), transpose(constraints_matrix)), lambda_1)
# d1_constrained = normalized_difference(h_sparse,h_hat1_constrained)

# h_hat2_unconstrained = matmul(pinv(A), y2_sparse)
# Ah = np.asmatrix(A).getH()
# lambda_2 = matmul(2*inv(matmul(matmul(constraints_matrix, inv(matmul(Ah, A))),transpose(constraints_matrix))), matmul(constraints_matrix, h_hat2_unconstrained))
# h_hat2_constrained = h_hat2_unconstrained - 0.5*matmul(matmul(inv(matmul(Ah,A)), transpose(constraints_matrix)), lambda_2)
# d2_constrained = normalized_difference(h_sparse,h_hat2_constrained)

# print(f"Normalized difference for variance {np.round(SIG1**2, decimals=3)} : {d1_constrained}")
# print(f"Normalized difference for variance {np.round(SIG2**2, decimals=1)} : {d2_constrained}")


# Q2
def estimate_sparse_h(A=A,h=h,sig=SIG1,sparsity_points=None):
    h_sparse = h.copy()
    
    if sparsity_points is None:
        sparsity_points = [i for i in random.sample(range(0,H_SIZE),H_SIZE-6)]
    for i in sparsity_points:
        h_sparse[i] = 0
    n = np.random.normal(loc=0, scale=sig,size=(Y_SIZE,2)).view(np.complex128)
    y_sparse = np.matmul(A,h_sparse) + n
    
    constraints_matrix = np.zeros((H_SIZE-6, H_SIZE))
    for i in range(len(sparsity_points)):
        constraints_matrix[i,sparsity_points[i]]=1

    h_hat_unconstrained = matmul(pinv(A), y_sparse)
    Ah = np.asmatrix(A).getH()
    
    lambda_ = matmul(2*inv(matmul(matmul(constraints_matrix, inv(matmul(Ah, A))),                                   transpose(constraints_matrix))), matmul(constraints_matrix, h_hat_unconstrained))
    h_hat_constrained = h_hat_unconstrained - 0.5*matmul(matmul(inv(matmul(Ah,A)),                                                                 transpose(constraints_matrix)), lambda_)
    d_constrained = normalized_difference(h_sparse,h_hat_constrained)

    print(f"[Sparse] Normalized difference for variance {np.round(sig**2, decimals=3)} : {d_constrained}")
    return h_hat_constrained,d_constrained

estimate_sparse_h(sig=SIG1)
estimate_sparse_h(sig=SIG2)


# Q3

# # Set guard bands of width 180 on either side
# X_ = X.copy()
# X_[:180] = 0
# X_[-180:] = 0
# A_ = matmul(X_,F)

# y1_ = matmul(A_,h) + n1
# y2_ = matmul(A_,h) + n2

# # define alpha values
# alpha1 = 1
# alpha2 = 1

# ## LSE
# h_hat1_ = matmul(pinv(A_, alpha1), y1_)
# h_hat2_ = matmul(pinv(A_, alpha2), y2_)
# d1_ = normalized_difference(h,h_hat1_)
# d2_ = normalized_difference(h,h_hat2_)

# # print("h vector was : ",h)
# # print("h_hat1 vector is :",h_hat1)
# # print("h_hat2 vector is :",h_hat2)
# print(f"Normalized difference for variance {np.round(SIG1**2, decimals=3)} and alpha {alpha1} : {d1}")
# print(f"Normalized difference for variance {np.round(SIG2**2, decimals=1)} and alpha {alpha2} : {d2}")


# Q3a
X_ = X.copy()
X_[:180] = 0
X_[-180:] = 0
A_ = matmul(X_,F)

def estimate_guard_h_no_reg(A=A_,h=h,sig=SIG1):
    h_vanilla,d_vanilla = estimate_vanilla_h(A=A,sig=sig)
    h_sparse,d_sparse = estimate_sparse_h(A=A,sig=sig)
    return h_vanilla, h_sparse, d_vanilla, d_sparse

estimate_guard_h_no_reg(sig=SIG1)
estimate_guard_h_no_reg(sig=SIG2)

# Q3b
def estimate_guard_h_reg(A=A_,h=h,alpha=1,sig=SIG1):
    n = np.random.normal(loc=0, scale=sig,size=(Y_SIZE,2)).view(np.complex128)
    y_ = matmul(A,h) + n
    
    h_hat_ = matmul(pinv(A_, alpha), y_)
    d_ = normalized_difference(h,h_hat_)
    
    print(f"[Regularized] Normalized difference for variance {np.round(sig**2, decimals=3)} and alpha {alpha} : {d_}")
    return h_hat_,d_
    
estimate_guard_h_reg(alpha=0.01,sig=SIG1)
estimate_guard_h_reg(alpha=1,sig=SIG1)


# Q4

# Building constraints matrix
number_of_constraints = 3
constraints_matrix = np.zeros((number_of_constraints, H_SIZE))
constraints_matrix[0,0]=1
constraints_matrix[0,1]=-1
constraints_matrix[1,2]=1
constraints_matrix[1,3]=-1
constraints_matrix[2,4]=1
constraints_matrix[2,5]=-1

def estimate_constrained_h(A=A,h=h,sig=SIG1):
    h_constrained = h.copy()
    h_constrained[0] = h_constrained[1]
    h_constrained[2] = h_constrained[3]
    h_constrained[4] = h_constrained[5]
    
    n = np.random.normal(loc=0, scale=sig,size=(Y_SIZE,2)).view(np.complex128)
    y_constrained = np.matmul(A,h_constrained) + n

    h_hat_unconstrained = matmul(pinv(A), y_constrained)
    Ah = np.asmatrix(A).getH()
    lambda_ = matmul(2*inv(matmul(matmul(constraints_matrix, inv(matmul(Ah, A))),                                  transpose(constraints_matrix))), matmul(constraints_matrix, h_hat_unconstrained))
    h_hat_constrained = h_hat_unconstrained - 0.5*matmul(matmul(inv(matmul(Ah,A)),                                                                transpose(constraints_matrix)), lambda_)
    d_constrained = normalized_difference(h_constrained,h_hat_constrained)
    print(f"[Constrained] Normalized difference for variance {np.round(sig**2, decimals=3)} : {d_constrained}")
    return h_hat_constrained,d_constrained

estimate_constrained_h(sig=SIG1)
estimate_constrained_h(sig=SIG2)


# Q5
Somp = []
n = np.random.normal(loc=0, scale=SIG2,size=(Y_SIZE,2)).view(np.complex128)
y = matmul(A,h) + n
r = y.copy()
ko = 6
Ah = np.asmatrix(A).getH()
for k in range(ko):
    t_ = np.matmul(Ah,r)
    t = np.argmax(np.abs(np.matmul(Ah,r)))
    Somp.append(int(t))
    p = np.matmul(A[:,Somp],pinv(A[:,Somp]))
    r = np.matmul((np.eye(Y_SIZE) - p),y) 

sparsity_points = [x for x in range(H_SIZE) if x not in Somp]
estimate_sparse_h(sig=SIG2,sparsity_points=sparsity_points)


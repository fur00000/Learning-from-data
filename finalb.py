#final problems 11-12

import numpy as np
import cvxopt
cvxopt.solvers.options['show_progress'] = False

#load x, y
x = np.array([[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]])
y = np.array([-1, -1, -1, 1, 1, 1, 1])

#define kernel
def K(a, b):
    return (1 + np.inner(a, b))**2
    
#do SVM
#Assign parameters
N = y.size
P = np.empty([N,N], dtype=float)
for j in range(N): #for each row
    for k in range(N): #for each column
        P[j,k] = y[j]*y[k]*K(x[j,:],x[k,:])
q = np.array([-1]*N)
G = -1*np.eye(N)
h = [0]*N
#A = y
#b = 0.0
#convert to matrices
P = cvxopt.matrix(P, tc='d')
q = cvxopt.matrix(q, tc='d')
G = cvxopt.matrix(G, tc='d')
h = cvxopt.matrix(h, tc='d')
A = cvxopt.matrix(y, (1,N), tc='d')
b = cvxopt.matrix(0.0, tc='d')
#solve
sol = cvxopt.solvers.qp(P,q,G,h,A,b)
alpha = np.array(sol['x'])
print alpha

#homework 2: nonlinear transformation, problems 8-10

import random
import numpy as np

random.seed()
runs = 1000
#N = int(raw_input("Number of training points: "))
N = 1000
transform = raw_input("Nonlinear transform? (y/n): ")
iterations = [0]*runs
Ein = [0]*runs
Eout = [0]*runs
if transform == 'y':
    cols = 6
else:
    cols = 3
weights = np.empty((runs,cols), dtype = np.float)
for i in range(runs):
    #the target function is f(x1, x2) = sign(x1^2 + x2^2 -0.6)
    
    #choose x_n, the training points; load y (noiseless)
    x = np.ones([N,3], dtype=np.float64) # first column is 1
    y = np.array([0]*N, dtype = int)
    for j in range(N):
        x[j,1] = random.uniform(-1,1)
        x[j,2] = random.uniform(-1,1)
        y[j] = np.sign(x[j,1]**2 + x[j,2]**2 - 0.6)
    
    #add noise to y
    noisy_indices = random.sample(xrange(N),N/10)
    for j in noisy_indices:
        y[j] *= -1
    
    #debug
    #print x
    #print y
    
    #transform (1, x1, x2) -> (1, x1, x2, x1x2, x1**2, x2**2)
    if transform == 'y':
        xp = np.ones([N,6], dtype=np.float64)
        xp[:,0:3] = x
        for j in range(N):
            xp[j,3] = xp[j,1]*xp[j,2]
            xp[j,4] = xp[j,1]**2
            xp[j,5] = xp[j,2]**2
    else:
        xp = x
    
    #do linear regression: w = Xdagger y where Xdagger = (XT X)^-1 XT
    #print xp
    xt = xp.transpose()
    xtx = np.dot(xt, xp)
    xtx_inv = np.linalg.inv(xtx)
    xdag = np.dot(xtx_inv,xt)
    w = np.dot(xdag, y)
    
    #validate w
    gy = np.array([0]*N, dtype = int)
    for j in range(N):
        gy[j] = np.sign(np.inner(w,xp[j,:]))
    Ey = map(abs,(gy - y)/2)
    Ein[i] = np.mean(Ey)
   
    #choose test_x_n, the testing points; load test_y
    if transform == 'y':
        test_x = np.ones([runs,6], dtype=np.float64) # first column is 1
    else:
        test_x = np.ones([runs,3], dtype=np.float64)
    test_fy = np.array([0]*runs, dtype=int)
    test_gy = np.array([0]*runs, dtype=int)
    for j in range(runs):
        test_x[j,1] = random.uniform(-1,1)
        test_x[j,2] = random.uniform(-1,1)
        if transform == 'y':
            test_x[j,3] = test_x[j,1]*test_x[j,2]
            test_x[j,4] = test_x[j,1]**2
            test_x[j,5] = test_x[j,2]**2
        test_fy[j] = np.sign(test_x[j,1]**2 + test_x[j,2]**2 - 0.6)
        test_gy[j] = np.sign(np.inner(w,test_x[j,:]))
    #add noise to test_fy
    noisy_indices = random.sample(xrange(runs),runs/10)
    for k in noisy_indices:
        test_fy[k] *= -1
    Egy = map(abs,(test_gy - test_fy)/2)
    Eout[i] = np.mean(Egy)
    weights[i] = w
    
#print results    
print float(sum(Ein))/len(Ein)
print float(sum(Eout))/len(Eout)
print np.mean(weights, axis=0)
#homework 2: linear regression, problems 5-7

import random
import numpy as np

random.seed()
runs = 1000
N = int(raw_input("Number of training points: "))
pla = raw_input("Perceptron learning algorithm (y/n)? ")
iterations = [0]*runs
Ein = [0]*runs
Eout = [0]*runs
for i in range(runs):
    #choose f, the target function
    f1 = [random.uniform(-1,1),random.uniform(-1,1)]
    f2 = [random.uniform(-1,1),random.uniform(-1,1)]
    while f2[:] == f1[:]:
        f2 = [random.uniform(-1,1),random.uniform(-1,1)]
    
    fm = (f2[1]-f1[1])/(f2[0]-f1[0])
    fbv = [f1[1]-fm*f1[0],f2[1]-fm*f2[0]]
    fb = (fbv[0]+fbv[1])/2
    
    #debug
    #print f1
    #print f2
    #print fm
    #print fb
    #print [f1[0],fm*f1[0]+fb]
    #print [f2[0],fm*f2[0]+fb]
    #print
    
    #choose x_n, the training points; load y
    x = np.ones([N,3], dtype=np.float64) # first column is 1
    y = []
    for j in range(N):
        x[j,1] = random.uniform(-1,1)
        x[j,2] = random.uniform(-1,1)
        if x[j,2] > fm*x[j,1]+fb:
            y.append(1)
        elif x[j,2] < fm*x[j,1]+fb:
            y.append(-1)
        else:
            y.append(0)
    y = np.array(y, dtype = int)
    
    #do linear regression: w = Xdagger y where Xdagger = (XT X)^-1 XT
    xt = x.transpose()
    xtx = np.dot(xt, x)
    xtx_inv = np.linalg.inv(xtx)
    xdag = np.dot(xtx_inv,xt)
    w = np.dot(xdag, y)
    
    #validate w
    gy = [0]*N
    for j in range(N):
        if np.inner(w,x[j,:]) > 0:
            gy[j] = 1
        elif np.inner(w,x[j,:]) < 0:
            gy[j] = -1
    gy = np.array(gy, dtype = int)
    Ey = map(abs,(gy - y)/2)
    
    #do PLA if specified
    if pla == 'y':       
        # build lm, the list of mismatched indices, i.e. for which Ey != 0. len(lm) <= N always. 
        lm = []
        for j in range(N):
            if Ey[j] > 0:
                lm.append(j)
        
        while len(lm) > 0:
            #choose an index in lm
            test_index = random.choice(lm)
            
            #update weights
            if Ey[test_index] > 0:
                w += x[test_index,:]
            else:
                w -= x[test_index,:]
            
            #update gy, Ey, and lm
            lm = []
            for j in range(N):
                if np.inner(w,x[j,:]) > 0:
                    gy[j] = 1
                elif np.inner(w,x[j,:]) < 0:
                    gy[j] = -1
                Ey[j] = y[j] - gy[j]
                if Ey[j] != 0:
                    lm.append(j)
              
            #increase iteration count
            iterations[i] += 1
    
    #print Ey
    #print np.mean(Ey)
    Ein[i] = np.mean(Ey)
    
    #choose test_x_n, the training points; load test_y
    test_x = np.ones([runs,3], dtype=np.float64) # first column is 1
    test_fy = [0]*runs
    test_gy = [0]*runs
    test_agree = [0]*runs
    for j in range(runs):
        test_x[j,1] = random.uniform(-1,1)
        test_x[j,2] = random.uniform(-1,1)
        if test_x[j,2] > fm*test_x[j,1]+fb:
            test_fy[j] = 1
        elif test_x[j,2] < fm*test_x[j,1]+fb:
            test_fy[j] = -1
        else:
            test_fy[j] = 0
        if np.inner(w,test_x[j,:]) > 0:
            test_gy[j] = 1
        elif np.inner(w,test_x[j,:]) < 0:
            test_gy[j] = -1
        else:
            test_gy[j] = 0
        if test_fy[j] ==  test_gy[j]:
            test_agree[j] = 1
    Eout[i] = 1 - sum(test_agree)/float(len(test_agree))
print float(sum(Ein))/len(Ein)
print float(sum(Eout))/len(Eout)
print sum(iterations)/float(len(iterations))
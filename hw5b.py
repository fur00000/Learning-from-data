#homework 5: logistic regression, problems 8-9

import random
import numpy as np
import math

'''
Individual error: err = ln(1+exp(-y*w.x)) = ln(1+exp(-y*(w1*x1+w2*x2+w3*x3)))
Gradient: (d/dwn)err = [1/(1+exp(-y*w.x))][(-y*xn)*exp(-y*w.x)] 
'''

def errf(w, x, y):
    wx = np.inner(w,x)
    eywx = math.exp(-1*y*wx)
    answer = math.log(1+eywx)

    return answer
    
def graderr(w,x,y): #w and x are 3-d vectors, y is scalar
    answer = np.array([0,0,0], dtype=np.float)
    wx = np.inner(w,x)
    eywx = math.exp(-1*y*wx)
    answer[0] = (-1*y*x[0]*eywx)/(1+eywx)
    answer[1] = (-1*y*x[1]*eywx)/(1+eywx)
    answer[2] = (-1*y*x[2]*eywx)/(1+eywx)
    return answer

random.seed()
runs = 100
#N = int(raw_input("Number of training points: "))
N = 100
test_points = 1000
iterations = [0]*runs
#Ein = [0]*runs
Eout = [0]*runs
#weights = np.empty((runs,3), dtype = np.float)
for i in range(runs):
    #choose f, the target function
    f1 = [random.uniform(-1,1),random.uniform(-1,1)]
    f2 = [random.uniform(-1,1),random.uniform(-1,1)]
    while f2[:] == f1[:]:
        f2 = [random.uniform(-1,1),random.uniform(-1,1)]
    
    fm = (f2[1]-f1[1])/(f2[0]-f1[0])
    fbv = [f1[1]-fm*f1[0],f2[1]-fm*f2[0]]
    fb = (fbv[0]+fbv[1])/2
    
    #choose x_n, the training points; load y
    x = np.ones([N,3], dtype=np.float64) # first column is 1
    y = np.array([0]*N, dtype = int)
    for j in range(N):
        x[j,1] = random.uniform(-1,1)
        x[j,2] = random.uniform(-1,1)
        y[j] = np.sign(x[j,2] - fm*x[j,1] - fb)
        
    w = np.array([0,0,0], dtype = np.float64) #initialize weights as 0
    #Ein[i] = errf(y, yt) #Ein[i] is the in-sample error for run i
    test_index = range(N)
    random.shuffle(test_index)
    eta = 0.01
    wnext = w
    for j in test_index:
        wnext = wnext - eta*graderr(wnext,x[j,:],y[j])
    
    while np.linalg.norm(wnext-w) >= 0.01:
        #update w
        w = wnext
        
        #choose an new index
        random.shuffle(test_index)
        
        #update weights
        for j in test_index:
            wnext = wnext - eta*graderr(wnext,x[j,:],y[j])
          
        #increase iteration count
        iterations[i] += 1

    #update w
    w = wnext
    
    #choose test_x_n, the testing points; load test_y
    test_x = np.ones([test_points,3], dtype=np.float64)
    test_fy = np.array([0]*test_points, dtype=int)
    #test_gy = np.array([0]*runs, dtype=int)
    Egy = [0]*test_points
    for j in range(test_points):
        test_x[j,1] = random.uniform(-1,1)
        test_x[j,2] = random.uniform(-1,1)

        test_fy[j] = np.sign(test_x[j,2] - fm*test_x[j,1] - fb)
        #test_gy[j] = np.sign(np.inner(w,test_x[j,:]))
        Egy[j] = errf(w,test_x[j,:],test_fy[j])
    
    Eout[i] = float(sum(Egy))/test_points
    print i

avg_eout = sum(Eout)/runs 
avg_iter = sum(iterations)/runs
   
print "Average Eout: %.3f" % avg_eout   
print "Average iterations: %f" % avg_iter
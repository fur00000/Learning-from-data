#homework 1: perceptron learning algorithm

import random
import numpy as np
#import scipy.integrate as integrate

def integrandA(x, f, g):
#integrandA(f,g) = min(f,1)-max(g,-1)
    fx = f[0]*x + f[1]
    gx = g[0]*x + g[1]
    A = min(fx,1)
    B = max(gx,-1)
    return A-B
    
def integrandB(x, f, g):
#integrandB(f,g) = (min(g,1)-max(f,-1))
    fx = f[0]*x + f[1]
    gx = g[0]*x + g[1]
    A = min(gx,1)
    B = max(fx,-1)
    return A-B    

random.seed()
runs = 1000
N = int(raw_input("Number of training points: "))
iterations = [0]*runs
disagreements = [0]*runs
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
    
    #debug    
    #print x
    #print
    
    #run perceptron learning algorithm
    w = np.array([0,0,0], dtype = np.float64)
    yd = np.array(y, dtype = int) # yd is the difference in values between y and the test yt. For the initial case (w = 0) yd = y.
    yt = np.zeros(y.shape, dtype = int)
    
    lm = range(N) # lm is the list of mismatched indices, i.e. for which yd != 0. lm <= N always. In the first case, yd != 0 for all cases so lm is every index from 0 to N-1.
    
    while len(lm) > 0:
        #choose an index in lm
        test_index = random.choice(lm)
        
        #update weights
        if yd[test_index] > 0:
            w += x[test_index,:]
        else:
            w -= x[test_index,:]
        
        #update yt, yd, and lm
        lm = []
        for j in range(N):
            if np.inner(w,x[j,:]) > 0:
                yt[j] = 1
            elif np.inner(w,x[j,:]) < 0:
                yt[j] = -1
            yd[j] = y[j] - yt[j]
            if yd[j] != 0:
                lm.append(j)
          
        #increase iteration count
        iterations[i] += 1

    #normalize w
    w /= w[2]
    '''
    #find disagreements
    #Formula if slope of w (-w[1]) > fm: INT(-1, X)[min(f,1)-max(g,-1)]dx + INT(X, 1)[min(g,1)-max(f,-1)]dx
    #Formula if slope of w (-w[1]) < fm: INT(-1, X)[min(g,1)-max(f,-1)]dx + INT(X, 1)[min(f,1)-max(g,-1)]dx
    #X is where fm*X + fb = -w[1]*X - w[0], i.e. X = -(fb + w[0])/(fm + w[1])
    #integrandA(f,g) = min(f,1)-max(g,-1); integrandB(f,g) = (min(g,1)-max(f,-1))
    xfg = -(fb + w[0])/(fm + w[1])
    #yf = fm*xfg + fb
    #yg = -w[1]*xfg - w[0]
    f = [fm, fb]
    g = [-w[1], -w[0]]
    
    if f[0] < g[0]:
        disagreements[i] = integrate.quad(integrandA, -1, xfg, args = (f,g)) + integrate.quad(integrandB, xfg, 1, args = (f,g))[0]
    elif f[0] > g[0]:
        disagreements[i] = integrate.quad(integrandB, -1, xfg, args = (f,g)) + integrate.quad(integrandA, xfg, 1, args = (f,g))[0]
    else:
        disagreements[i] = 0

    #debug
    print "slope is %f, we found %f" % (fm, -w[1])
    print "intercept is %f, we found %f" % (fb, -w[0])
    print "%d iterations to converge" % iterations[i]
    

    #print "Common X point is %f, Y point is %f by f and %f by g" % (xfg, yf, yg)
    
    #print "Probability of error: %f" % disagreements[i]
    print disagreements[i]
    '''
    
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
    disagreements[i] = 1 - sum(test_agree)/float(len(test_agree))
print sum(iterations)/float(len(iterations))
print sum(disagreements)/float(len(disagreements))
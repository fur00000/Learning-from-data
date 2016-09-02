#homework 4: bias variance tradeoff, problems 4-7

import random
import numpy as np
import math
import scipy.integrate as integrate

random.seed()

print "Choose hypothesis set:"
print "1. h(x) = b"
print "2. h(x) = ax"
print "3. h(x) = ax+b"
print "4. h(x) = ax**2"
print "5. h(x) = ax**2+b"
hypothesis = int(raw_input("Hypothesis set: "))
runs = int(raw_input("Number of runs: "))
testpoints = int(raw_input("Number of testing points: "))
#Ein = np.array([0]*runs, dtype = np.float)
Eout = np.array([0]*runs, dtype = np.float)
avalues = np.array([0]*runs, dtype = np.float)
bvalues = np.array([0]*runs, dtype = np.float)

def mse(a, x):
    dif = a*x - math.sin(math.pi*x)
    return 0.5*(dif**2)

for i in range(runs):
    #the target function is f(x) = sin(pi x) on [-1,1]
    
    #choose xa, xb, the training points; load y
    xa = random.uniform(-1,1)
    xb = random.uniform(-1,1)
    ya = math.sin(xa*math.pi)
    yb = math.sin(xb*math.pi)
    
    #hypothesis minimizes square error 
    
    #Let h(x)=b; if g=h then b = (ya + yb)/2
    if hypothesis == 1:
        bvalues[i] = (ya+yb)/2
    
    #Let h(x)=ax; if g=h then a = (xaya + xbyb)/(xa^2 + xb^2)
    elif hypothesis == 2:
        num = xa*ya + xb*yb
        den = xa**2 + xb**2
        avalues[i] = num/den
        
    #Let h(x)=ax+b; if g=h then a = (ya-yb)/(xa-xb) and b = ya - a*xa = yb - a*xb
    elif hypothesis == 3:
        avalues[i] = (ya-yb)/(xa-xb)
        fbv = [ya-avalues[i]*xa,yb-avalues[i]*xb]
        bvalues[i] = (fbv[0]+fbv[1])/2
        
    #Let h(x)=ax**2; if g=h then a = (ya-yb)/(xa-xb) and b = ya - a*xa = yb - a*xb
    elif hypothesis == 4:
        num = xa*xa*ya + xb*xb*yb
        den = xa**4 + xb**4
        avalues[i] = num/den
    
    #Let h(x)=ax**2+b; if g=h then a = (ya-yb)/(xa**2-xb**2) and b = ya - a*xa**2 = yb - a*xb**2
    elif hypothesis == 5:
        avalues[i] = (ya-yb)/(xa**2-xb**2)
        fbv = [ya-avalues[i]*(xa**2),yb-avalues[i]*(xb**2)]
        bvalues[i] = (fbv[0]+fbv[1])/2
    
    #choose test_x_n, the testing points; load test_y
    test_x = np.array([0]*testpoints, dtype=float)
    test_fy = np.array([0]*testpoints, dtype=float)
    test_gy = np.array([0]*testpoints, dtype=float)
    E_gy = np.array([0]*testpoints, dtype=float)
    for j in range(testpoints):
        test_x[j] = random.uniform(-1,1)
        test_fy[j] = math.sin(math.pi*test_x[j])
        if hypothesis == 1:
            test_gy[j] = bvalues[i]
        elif hypothesis == 2:
            test_gy[j] = test_x[j]*avalues[i]
        elif hypothesis == 3:
            test_gy[j] = test_x[j]*avalues[i] + bvalues[i]
        elif hypothesis == 4:
            test_gy[j] = test_x[j]*test_x[j]*avalues[i]
        elif hypothesis == 5:
            test_gy[j] = test_x[j]*test_x[j]*avalues[i] + bvalues[i]
        E_gy[j] = (test_fy[j] - test_gy[j])**2
    
    Eout[i] = np.mean(E_gy)
    
#mean g(x) = ahat*x    
ahat = np.mean(avalues) 
print "A-hat is %.2f" % ahat

#bias is integral from -1 to 1 of (ax-sin(pi x))**2 dx times 1/2 (uniform distribution)
bias = integrate.quad(lambda x: mse(ahat, x), -1, 1)
print "Bias is %.2f" % bias[0]

#variance is Eout - bias
var = np.mean(Eout).item() - bias[0]
print "Variance is %.2f" % var
print "Out of sample error is %.2f" % np.mean(Eout).item()
import numpy as np
import math

reg = raw_input("Regularize? (y/n): ")
if reg == 'y':
    ell = 10**int(raw_input("Value of k? "))
else:
    ell = 0
    
#load data
raw_data = []
file = open('hw6in.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
N = len(raw_data)
train_x = np.ones((N,8))
train_y = [0]*N
for i in range(N):
    train_x[i,1] = raw_data[i][0]
    train_x[i,2] = raw_data[i][1]
    train_x[i,3] = raw_data[i][0]**2
    train_x[i,4] = raw_data[i][1]**2
    train_x[i,5] = raw_data[i][0]*raw_data[i][1]
    train_x[i,6] = abs(raw_data[i][0]-raw_data[i][1])
    train_x[i,7] = abs(raw_data[i][0]+raw_data[i][1])
    train_y[i] = int(raw_data[i][2])
train_y = np.array(train_y)
#print train_x
#print train_y

#do linear regression: w = Xdagger y where Xdagger = (XT X + ell*I)^-1 XT
xt = train_x.transpose()
xtx = np.dot(xt, train_x)
xtx_inv = np.linalg.inv(xtx+ell*np.identity(8))
xdag = np.dot(xtx_inv,xt)
w = np.dot(xdag, train_y)

#validate on training set to find Ein
Ein = [0]*N
for i in range(N):
    if train_y[i]*np.inner(w,train_x[i,:]) > 0:
        Ein[i] = 0
    else:
        Ein[i] = 1
Ein_tot = float(sum(Ein))/N
print Ein_tot

#load testing data
raw_data = []
file = open('hw6out.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
Nout = len(raw_data)
test_x = np.ones((Nout,8))
test_y = [0]*Nout
for i in range(Nout):
    test_x[i,1] = raw_data[i][0]
    test_x[i,2] = raw_data[i][1]
    test_x[i,3] = raw_data[i][0]**2
    test_x[i,4] = raw_data[i][1]**2
    test_x[i,5] = raw_data[i][0]*raw_data[i][1]
    test_x[i,6] = abs(raw_data[i][0]-raw_data[i][1])
    test_x[i,7] = abs(raw_data[i][0]+raw_data[i][1])
    test_y[i] = int(raw_data[i][2])
test_y = np.array(test_y)
#print test_x
#print test_y

#validate on testing set to fine Eout
Eout = [0]*Nout
for i in range(Nout):
    if test_y[i]*np.inner(w,test_x[i,:]) > 0:
        Eout[i] = 0
    else:
        Eout[i] = 1
Eout_tot = float(sum(Eout))/Nout
print Eout_tot
print w
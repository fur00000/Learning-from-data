#homework 7, problems 1-5
import numpy as np
import math

print "Choose split (a/b):"
print "a. 25 training points, 10 validation points"
print "b. 10 training points, 25 validation points"
split = raw_input("Choice (a/b): ")
first_N = 25
ell = 0
k_values = [3,4,5,6,7]
    
#load data
raw_data = []
file = open('hw6in.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
N = len(raw_data)
raw_x = np.ones((N,8))
raw_y = np.array([0]*N, dtype=int)
for i in range(N):
    raw_x[i,1] = raw_data[i][0]
    raw_x[i,2] = raw_data[i][1]
    raw_x[i,3] = raw_data[i][0]**2
    raw_x[i,4] = raw_data[i][1]**2
    raw_x[i,5] = raw_data[i][0]*raw_data[i][1]
    raw_x[i,6] = abs(raw_data[i][0]-raw_data[i][1])
    raw_x[i,7] = abs(raw_data[i][0]+raw_data[i][1])
    raw_y[i] = int(raw_data[i][2])
#print raw_x
#print raw_y

#split into training and validation sets
if split == 'a':
    train_x = np.array(raw_x[0:first_N,:], dtype=np.float)
    train_y = np.array(raw_y[0:first_N], dtype=np.float)
    validate_x = np.array(raw_x[first_N:,:], dtype=np.float)
    validate_y = np.array(raw_y[first_N:], dtype=np.float)
elif split == 'b':
    validate_x = np.array(raw_x[0:first_N,:], dtype=np.float)
    validate_y = np.array(raw_y[0:first_N], dtype=np.float)
    train_x = np.array(raw_x[first_N:,:], dtype=np.float)
    train_y = np.array(raw_y[first_N:], dtype=np.float)
else:
    exit(0)
'''
print "Training x"
print train_x
print "Training y"
print train_y
print "Validation x"
print validate_x
print "Validation y"
print validate_y
'''

#load testing data
raw_data = []
file = open('hw6out.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
Nout = len(raw_data)
test_x = np.ones((Nout,8))
test_y = np.array([0]*Nout, dtype=int)
for i in range(Nout):
    test_x[i,1] = raw_data[i][0]
    test_x[i,2] = raw_data[i][1]
    test_x[i,3] = raw_data[i][0]**2
    test_x[i,4] = raw_data[i][1]**2
    test_x[i,5] = raw_data[i][0]*raw_data[i][1]
    test_x[i,6] = abs(raw_data[i][0]-raw_data[i][1])
    test_x[i,7] = abs(raw_data[i][0]+raw_data[i][1])
    test_y[i] = int(raw_data[i][2])
#print test_x
#print test_y

for k in k_values:
    #do linear regression: w = Xdagger y where Xdagger = (XT X + ell*I)^-1 XT
    train_x_k = np.array(train_x[:,:k+1], dtype=np.float)
    xt = train_x_k.transpose()
    xtx = np.dot(xt, train_x_k)
    xtx_inv = np.linalg.inv(xtx+ell*np.identity(k+1))
    xdag = np.dot(xtx_inv,xt)
    w = np.dot(xdag, train_y)
    
    #validate on the validation set to find Eval
    if split == 'a':
        validate_N = N - first_N
    else:
        validate_N = first_N
    Eval = [0]*validate_N
    for i in range(validate_N):
        if validate_y[i]*np.inner(w,validate_x[i,:k+1]) > 0:
            Eval[i] = 0
        else:
            Eval[i] = 1
    Eval_tot = float(sum(Eval))/N
    
    #validate on testing set to fine Eout
    Eout = [0]*Nout
    for i in range(Nout):
        if test_y[i]*np.inner(w,test_x[i,:k+1]) > 0:
            Eout[i] = 0
        else:
            Eout[i] = 1
    Eout_tot = float(sum(Eout))/Nout
    
    print "k = %d:" % k
    print "Eval = %.3f; Eout = %.3f" % (Eval_tot, Eout_tot)
    print 
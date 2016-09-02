#Final problems 7-10: regularized linear regression

import numpy as np

#load training set
raw_data = []
file = open('hw8in.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
Nin = len(raw_data)
train_x = np.ones((Nin,6)) #first column is 1, second is intensity, third is symmetry, 4-6 are nonlinear transforms
train_digit = np.array([0]*Nin, dtype=int) #the actual digits
for i in range(Nin):
    train_x[i,1] = raw_data[i][1]
    train_x[i,2] = raw_data[i][2]
    train_x[i,3] = raw_data[i][1]*raw_data[i][2]
    train_x[i,4] = raw_data[i][1]*raw_data[i][1]
    train_x[i,5] = raw_data[i][2]*raw_data[i][2]
    train_digit[i] = int(raw_data[i][0])

#load testing data
raw_data = []
file = open('hw8out.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
Nout = len(raw_data)
test_x = np.ones((Nout,6)) #first column is 1, second is intensity, third is symmetry, 4-6 are nonlinear transforms
test_digit = np.array([0]*Nout, dtype=int) #the actual digits
for i in range(Nout):
    test_x[i,1] = raw_data[i][1]
    test_x[i,2] = raw_data[i][2]
    test_x[i,3] = raw_data[i][1]*raw_data[i][2]
    test_x[i,4] = raw_data[i][1]*raw_data[i][1]
    test_x[i,5] = raw_data[i][2]*raw_data[i][2]
    test_digit[i] = int(raw_data[i][0])
    
#get train_y and test_y
print "Input digits to classify. First digit is classified +1 and second -1."
print "If there is no second digit, then classification is considered to be one-versus-all."
selected_digits = map(int, raw_input("Input (n [m]): ").split())
if len(selected_digits) == 1 and 0 <= selected_digits[0] <= 9: #one-versus-all classification
    train_y = np.array([0]*Nin, dtype=int)
    test_y = np.array([0]*Nout, dtype=int)
    for i in range(Nin):
        if train_digit[i] == selected_digits[0]:
            train_y[i] = 1
        else:
            train_y[i] = -1
    for i in range(Nout):
        if test_digit[i] == selected_digits[0]:
            test_y[i] = 1
        else:
            test_y[i] = -1
    #print train_y.tolist()[0:20]
elif len(selected_digits) > 1 and 0 <= selected_digits[0] <= 9 and 0 <= selected_digits[1] <= 9 and selected_digits[0] != selected_digits[1]: #one-versus-one classification
    train_y = []
    test_y = []
    indices_to_delete = []
    for i in range(Nin):
        if train_digit[i] == selected_digits[0]:
            train_y.append(1)
        elif train_digit[i] == selected_digits[1]:
            train_y.append(-1)
        else:
            indices_to_delete.append(i)
    train_x = np.delete(train_x, indices_to_delete, axis=0) #delete all other digits from training set
    train_y = np.array(train_y, dtype=int)
    indices_to_delete = []
    for i in range(Nout):
        if test_digit[i] == selected_digits[0]:
            test_y.append(1)
        elif test_digit[i] == selected_digits[1]:
            test_y.append(-1)
        else:
            indices_to_delete.append(i)
    test_x = np.delete(test_x, indices_to_delete, axis=0) #delete all other digits from testing set
    test_y = np.array(test_y, dtype=int)
else:
    print "Invalid input!"
    exit(0)

transform = raw_input("2nd order transform? (y/n): ")
if transform == 'y':
    d = 6
elif transform == 'n':
    d = 3
    train_x = train_x[:,:3]
    test_x = test_x[:,:3]
else:
    print "Invalid input!"
    exit(0)
    
#linear transformation
ell = float(raw_input("Regularization factor lambda: ")) #lambda
xt = train_x.transpose()
xtx = np.dot(xt, train_x)
xtx_inv = np.linalg.inv(xtx+ell*np.identity(d))
xdag = np.dot(xtx_inv,xt)
w = np.dot(xdag, train_y)

#Get Ein, Eout
train_gy = map(np.sign, np.dot(train_x,w))
test_gy = map(np.sign, np.dot(test_x,w))

Ein = map(abs, (train_gy - train_y)/2)
Eout = map(abs, (test_gy - test_y)/2)
print "In-sample error is %.5f" % np.mean(Ein)
print "Out-of-sample error is %.5f" % np.mean(Eout)
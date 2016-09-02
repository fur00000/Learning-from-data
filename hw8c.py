#Homework 8 problems 9-10

import numpy as np
from sklearn import svm

#load training set
raw_data = []
file = open('hw8in.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
Nin = len(raw_data)
train_x = np.ones((Nin,2)) #first column is intensity, second is symmetry
train_digit = np.array([0]*Nin, dtype=int) #the actual digits
for i in range(Nin):
    train_x[i,0] = raw_data[i][1]
    train_x[i,1] = raw_data[i][2]
    train_digit[i] = int(raw_data[i][0])
    
#load testing data
raw_data = []
file = open('hw8out.txt', 'r')
for line in file:
    x = line.split()
    raw_data.append(map(float,x))
file.close()
Nout = len(raw_data)
test_x = np.ones((Nout,2)) #first column is intensity, second is symmetry
test_digit = np.array([0]*Nout, dtype=int) #the actual digits
for i in range(Nout):
    test_x[i,0] = raw_data[i][1]
    test_x[i,1] = raw_data[i][2]
    test_digit[i] = int(raw_data[i][0])
    
#get train_y and test_y
print "Input digits to classify. First digit is classified +1 and second -1."
selected_digits = map(int, raw_input("Input (n m): ").split())
if len(selected_digits) > 1 and 0 <= selected_digits[0] <= 9 and 0 <= selected_digits[1] <= 9 and selected_digits[0] != selected_digits[1]: #one-versus-one classification
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
    
#do svm
print "Enter C, the upper bound of each support vector parameter."
input_params = raw_input("Input (C1 [C2 ...]): ").split()
C_values = map(float, input_params)
    
for C_in in C_values:
    clf = svm.SVC(C=C_in, kernel='rbf', gamma=1.)
    clf.fit(train_x,train_y)
    train_gy = clf.predict(train_x)
    test_gy = clf.predict(test_x)
    alphas = map(abs, clf.dual_coef_)
    print "Alphas range from %.5f to %f" % (np.amin(alphas), np.amax(alphas))
    Ein = map(abs, (train_gy - train_y)/2)
    Eout = map(abs, (test_gy - test_y)/2)
    print "C = %g: In-sample error is %.5f" % (C_in, np.mean(Ein))
    print "C = %g: Out-of-sample error is %.5f" % (C_in, np.mean(Eout))
    print "C = %g: Number of support vectors: %d" % (C_in, sum(clf.n_support_))
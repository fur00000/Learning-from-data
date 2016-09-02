#Homework 8 problems 2-6

'''
import sys
sys.path.append('libsvm-3.21/python')
import svm
'''
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
    #print np.concatenate((train_x[:20,:], np.array([train_y[:20]]).T),axis=1).tolist()
    #print np.concatenate((test_x[:20,:], np.array([test_y[:20]]).T),axis=1).tolist()
else:
    print "Invalid input!"
    exit(0)

#normalize data
normalize = raw_input("Normalize data? (y/n): ")
means = [np.mean(train_x[:,0]), np.mean(train_x[:,1])]
stds = [np.std(train_x[:,0]), np.std(train_x[:,1])] 
train_x_norm = np.empty_like(train_x)
for i in range (len(train_x_norm[:,0])):
    train_x_norm[i,:] = (train_x[i,:]-means)/stds

test_x_norm = np.empty_like(test_x)
for i in range (len(test_x_norm[:,0])):
    test_x_norm[i,:] = (test_x[i,:]-means)/stds
    
#do svm
print "Input Q and C. Q is the order of the polynomial kernel and C is the upper bound of each support vector parameter."
try:
    input_params = raw_input("Input (Q C): ").split()
    Q_in = int(input_params[0])
    C_in = float(input_params[1])
except:
    print "Invalid input!"
    raise    
    
clf = svm.SVC(C=C_in, kernel='poly', degree=Q_in, gamma=1., coef0=1.)
if normalize == 'y':
    clf.fit(train_x_norm,train_y)
    train_gy = clf.predict(train_x_norm)
    test_gy = clf.predict(test_x_norm)
elif normalize == 'n':
    clf.fit(train_x,train_y)
    train_gy = clf.predict(train_x)
    test_gy = clf.predict(test_x)
else:
    exit(0)
alphas = map(abs, clf.dual_coef_)
print "Alphas range from %.5f to %f" % (np.amin(alphas), np.amax(alphas))
#print alphas
#print alphas.shape
Ein = map(abs, (train_gy - train_y)/2)
Eout = map(abs, (test_gy - test_y)/2)
print "In-sample error is %.5f" % np.mean(Ein)
print "Out-of-sample error is %.5f" % np.mean(Eout)
print "Number of support vectors: %d" % sum(clf.n_support_)
#print clf.n_support_
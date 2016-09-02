#Homework 8 problems 7-8

import random
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
    
#get train_y
print "Input digits to classify. First digit is classified +1 and second -1."
selected_digits = map(int, raw_input("Input (n m): ").split())
if len(selected_digits) > 1 and 0 <= selected_digits[0] <= 9 and 0 <= selected_digits[1] <= 9 and selected_digits[0] != selected_digits[1]: #one-versus-one classification
    train_y = []
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
else:
    print "Invalid input!"
    exit(0)
    
Q_in = int(raw_input("Input Q, the order of the polynomial kernel: "))

#do 10-fold cross-validation
runs = int(raw_input("Now input number of runs for cross-validation: "))
Ecv = np.zeros((runs, 5), dtype=float) #Ecv[a, b] is the cross validation error for run a using the bth value of C_in
C_values = [0.0001, 0.001, 0.01, 0.1, 1.]
C_choice = [0]*runs
#print train_x.shape
train_y = np.array([train_y])
#print train_y.shape
for i in range(runs):
    Nin = train_y.size #update Nin
    nin = int(Nin/10) #size of each partition
    shuffled_indices = range(Nin)
    train_xy = np.concatenate((train_x,train_y.T), axis=1)
    #print train_xy.shape
    #print np.array([train_xy[0,:]]).shape
    partitions = [0]*10 #each element of partitions has nin entries of entire set except for last element, which has the remainder
    np.random.shuffle(shuffled_indices)
    for j in range(Nin):
        index = shuffled_indices[j]
        partition_number = int(j/nin) # partition_number = 0 for j from 0 to nin-1, 1 from nin to 2nin-1, etc.
        if partition_number > 9:
            partition_number = 9 # put last elements in the final partition
        if isinstance(partitions[partition_number], int): # if partition is empty
            partitions[partition_number] = np.array([train_xy[index,:]]) # add the data point to it
        else: #otherwise add data point to end
            partitions[partition_number] = np.concatenate((partitions[partition_number], np.array([train_xy[index,:]])), axis=0)
        #print partitions[partition_number].shape
    for j in range(10):
        #print partitions[j]
        temp_cv_x = partitions[j][:,:2]
        temp_cv_y = partitions[j][:,-1]
        temp_train_xy = partitions[:]
        del temp_train_xy[j]
        temp_train_xy = np.concatenate(temp_train_xy, axis=0)
        temp_train_x = temp_train_xy[:,:2]
        temp_train_y = temp_train_xy[:,-1]
        temp_Ecv = [0]*5
        for k in range(5):
            C_in = C_values[k]
            temp_clf = svm.SVC(C=C_in, kernel='poly', degree=Q_in, gamma=1., coef0=1.)
            temp_clf.fit(temp_train_x,temp_train_y)
            temp_cv_gy = temp_clf.predict(temp_cv_x)
            temp_Ecv[k] = map(abs, (temp_cv_gy - temp_cv_y)/2)
            Ecv[i, k] += 0.1*np.mean(temp_Ecv[k]) # 0.1 factor since we are averaging over 10 partitions
    #Now, for Ecv[i,:] let's find the column with the lowest value and put that into C_choice[i]
    C_choice[i] = C_values[np.argmin(Ecv[i,:])]

print "C=0.0001 count: %d" % C_choice.count(0.0001)
print "C=0.001 count: %d" % C_choice.count(0.001)
print "C=0.01 count: %d" % C_choice.count(0.01)
print "C=0.1 count: %d" % C_choice.count(0.1)
print "C=1 count: %d" % C_choice.count(1)
#print "Cross-validation error is %.5f" % np.mean(Ecv)
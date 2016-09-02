#Final problems 13-17

import random
import numpy as np
import math
from sklearn import svm
#import matplotlib.pyplot as plt

runs = int(raw_input("Number of runs: "))
K = int(raw_input("Number of centers for clustering: "))
Gamma = float(raw_input("Gamma: "))
N = 100 #number of training points
reg_rbf_failed = [] #list of runs in which regular rbf failed. Regular rbf fails if one of the S_k clusters becomes empty.
E_reg_in = [0]*runs
E_reg_out = [0]*runs
svm_rbf_failed = [] #list of runs in which svm rbf failed. SVM rbf fails if data is not linearly separable in Z-space, i.e. if E_in > 0.
E_svm_in = [0]*runs
E_svm_out = [0]*runs
svm_wins = [0]*runs

print "------------------------------------------------"

#target function
def signal(x):
    temp = math.sin(math.pi*x[0])
    return x[1]-x[0]+0.25*temp

#returns true if matrix is big
def isbig(m):
    sum = 0
    threshold = 0
    for i in range(m.shape[0]):
        sum += np.linalg.norm(m[i,:])   
    if sum > threshold:
        return True
    else:
        return False
'''    
# Plot of target function
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
t = np.arange(-1., 1., 0.02)
plt.plot(t, map(lambda x: x-0.25*math.sin(math.pi*x), t))
plt.axis((-1,1,-1,1))
plt.ion()
plt.show()
plt.pause(0.001)    
'''    
for i in range(runs):
    #choose x_n, the training points; load y
    x = np.empty([N,2], dtype=np.float64)
    y = np.array([0]*N, dtype=int)
    for j in range(N):
        x[j,0] = random.uniform(-1,1)
        x[j,1] = random.uniform(-1,1)
        y[j] = np.sign(signal(x[j,:]))
    '''    
    # Plotting the training points
    plt.scatter(
        x[:,0],
        x[:,1],
        c=y,
        cmap='bwr',
        alpha=1,
        s=10,
        edgecolors='k'
        )    
    plt.show()
    '''    
    #Generate testing points as well
    test_N = 200
    test_x = np.empty([test_N,2], dtype=np.float64)
    test_y = np.array([0]*test_N, dtype=int)
    for j in range(test_N):
        test_x[j,0] = random.uniform(-1,1)
        test_x[j,1] = random.uniform(-1,1)
        test_y[j] = np.sign(signal(test_x[j,:]))
        
    #Do regular RBF model with Lloyd's algorithm
    
    #Initialize mu_k
    mu = np.random.rand(K,2)*2 - 1
    lloyd_cont = True
    escape = False
    while lloyd_cont == True and escape == False:
        #Update S_k
        S = [0]*K #each element S[k] is a list of indices [n] such that x[n,:] is closest to mu[k,:]
        for j in range(N):
            distances = [0]*K
            for k in range(K):
                distances[k] = np.linalg.norm(x[j,:]-mu[k,:])
            k_choice = np.argmin(distances)
            if isinstance(S[k_choice], int):
                S[k_choice] = [j]
            else:
                S[k_choice].append(j)
        
        #update mu to be center of each S_k
        mu_new = np.zeros((K,2),dtype=float)
        for k in range(K):
            if isinstance(S[k], int):
                escape = True
            else:
                for j in S[k]:
                    mu_new[k,:] += x[j,:]
                mu_new[k,:] /= len(S[k])
        
        #find how much mu changed        
        mu_change = mu_new - mu 
        lloyd_cont = isbig(mu_change) #if mu_change is false then lloyd_cont also becomes false
        mu = mu_new    
    
    #Check that all clusters are nonempty
    lloyd_cont = True #now this boolean refers to if we should continue the run
    for k in S:
        if isinstance(k, int):
            lloyd_cont = False
        elif len(k) == 0: 
            lloyd_cont = False
        
    if lloyd_cont == True and escape == False: 
        #Generate Phi matrix    
        Phi = np.ones((N, K+1), dtype=float) #K+1 columns since we have a bias term
        for j in range(N):
            for k in range(K):
                Phi[j,k+1] = math.exp(-1*Gamma*np.inner((x[j,:]-mu[k,:]),(x[j,:]-mu[k,:])))

        #do linear regression: w = Phi_dagger y where Phi_dagger = (PhiT Phi)^-1 PhiT
        Phi_T = Phi.transpose()
        Phi_T_Phi = np.dot(Phi_T, Phi)
        Phi_T_Phi_inv = np.linalg.inv(Phi_T_Phi)
        Phidag = np.dot(Phi_T_Phi_inv,Phi_T)
        w_reg = np.dot(Phidag, y)              
        #print Phi
        
        #Find Ein
        gy = map(np.sign, np.dot(Phi,w_reg))
        Ey = map(abs,(gy - y)/2)
        E_reg_in[i] = np.mean(Ey)
        
        #Generate testing Phi matrix    
        test_Phi = np.ones((test_N, K+1), dtype=float) #K+1 columns since we have a bias term
        for j in range(test_N):
            for k in range(K):
                test_Phi[j,k+1] = math.exp(-1*Gamma*np.inner((test_x[j,:]-mu[k,:]),(test_x[j,:]-mu[k,:])))
                
        #Find Eout
        test_gy = map(np.sign, np.dot(test_Phi,w_reg))
        test_Ey = map(abs,(test_gy - test_y)/2)
        E_reg_out[i] = np.mean(test_Ey)
    else:
        reg_rbf_failed.append(i)
        
    #Do SVM with RBF kernel 

    clf = svm.SVC(C=10**10, kernel='rbf', gamma=Gamma)
    clf.fit(x,y)
    gy_svm = map(np.sign, clf.predict(x))
    test_gy_svm = map(np.sign, clf.predict(test_x))
    alphas = map(abs, clf.dual_coef_)
    #print "Alphas range from %.5f to %f" % (np.amin(alphas), np.amax(alphas))
    E_svm_in[i] = np.mean(map(abs, (gy_svm - y)/2))
    if E_svm_in[i] != 0:
        svm_rbf_failed.append(i)
    E_svm_out[i] = np.mean(map(abs, (test_gy_svm - test_y)/2))
    #print "C = %g: Number of support vectors: %d" % (C_in, sum(clf.n_support_))
    
    if E_svm_out[i] < E_reg_out[i]:
        svm_wins[i] = 1
    else:
        svm_wins[i] = 0
        
print "Regular RBF model data"
print "------------------------"
percent_rbf_failed = float(len(reg_rbf_failed))/runs
print "Percent of runs with an empty cluster: %g" % percent_rbf_failed
if len(reg_rbf_failed) > 0:
    for i in reversed(range(len(reg_rbf_failed))):
        del E_reg_in[reg_rbf_failed[i]]
        del E_reg_out[reg_rbf_failed[i]]
print "In-sample error: %g" % np.mean(E_reg_in)
print "Out-of-sample error: %g" % np.mean(E_reg_out)
print
print "SVM with RBF kernel data"
print "------------------------"
#print svm_rbf_failed
#print E_svm_in
#print E_svm_out
percent_svm_failed = float(len(svm_rbf_failed))/runs
print "Percent of runs with non-separable data in Z-space: %g" % percent_svm_failed
if len(svm_rbf_failed) > 0:
    for i in reversed(range(len(svm_rbf_failed))):
        del E_svm_in[svm_rbf_failed[i]]
        del E_svm_out[svm_rbf_failed[i]]
print "In-sample error: %g" % np.mean(E_svm_in)
print "Out-of-sample error: %g" % np.mean(E_svm_out)
print "------------------------------------------------"
runs_to_remove = list((set(reg_rbf_failed) | set(svm_rbf_failed)))
runs_to_remove.sort()
#print reg_rbf_failed
#print runs_to_remove
#print svm_wins
if len(runs_to_remove) > 0:
    for i in reversed(range(len(runs_to_remove))):
        del svm_wins[runs_to_remove[i]]
#print svm_wins
print "SVM is better %g percent of the time." % np.mean(svm_wins)

#homework 7: PLA vs SVM, problems 8-10

import random
import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False
random.seed()
N = int(raw_input("Number of training points: "))
runs = int(raw_input("Number of runs: "))
iterations = [0]*runs
Ein = [0]*runs
Eout_pla = [0]*runs
Eout_svm = [0]*runs
svm_better = 0
support_vectors = [0]*runs
bad_run_indices = []
for i in range(runs):
    #choose f, the target function
    f1 = [random.uniform(-1,1),random.uniform(-1,1)]
    f2 = [random.uniform(-1,1),random.uniform(-1,1)]
    while f2[:] == f1[:]: # redo if they are same point
        f2 = [random.uniform(-1,1),random.uniform(-1,1)]
    
    fm = (f2[1]-f1[1])/(f2[0]-f1[0])
    fbv = [f1[1]-fm*f1[0],f2[1]-fm*f2[0]]
    fb = (fbv[0]+fbv[1])/2
    
    #choose x_n, the training points; load y
    x = np.ones([N,3], dtype=np.float64) # first column is 1
    y = [1]*N
    while abs(sum(y)) == len(y): #redo if all points lie on one side
        for j in range(N):
            x[j,1] = random.uniform(-1,1)
            x[j,2] = random.uniform(-1,1)
            if x[j,2] > fm*x[j,1]+fb:
                y[j]=1
            elif x[j,2] < fm*x[j,1]+fb:
                y[j]=-1
            else:
                y[j]=0
    y = np.array(y, dtype = int)
    
    #do PLA
    w_pla = np.array([0,0,0], dtype=float)      
    # build lm, the list of mismatched indices, i.e. for which Ey != 0. len(lm) <= N always. All indicies at the beginning
    lm = range(N)
    # build Ey, the error list, which is all one at the beginning
    Ey = [1]*N
    gy = [0]*N
    
    while len(lm) > 0:
        #choose an index in lm
        test_index = random.choice(lm)
        
        #update weights
        if Ey[test_index] > 0:
            w_pla += x[test_index,:]
        else:
            w_pla -= x[test_index,:]
        
        #update gy, Ey, and lm
        lm = []
        for j in range(N):
            gy[j] = np.sign(np.inner(w_pla,x[j,:]))
            Ey[j] = y[j] - gy[j]
            if y[j] != gy[j]:
                lm.append(j)
          
        #increase iteration count
        iterations[i] += 1
        #print np.linalg.norm(np.array(Ey))
        #print lm
    
    #print Ey
    #print np.mean(Ey)
    Ein[i] = np.mean(Ey)
    
    #do SVM
    #Assign constraints
    P = np.empty([N,N], dtype=float)
    for j in range(N): #for each row
        for k in range(N): #for each column
            P[j,k] = y[j]*y[k]*np.inner(x[j,1:],x[k,1:])
    q = np.array([-1]*N)
    G = np.vstack([-1*np.eye(N),np.eye(N)])
    h = np.hstack([[0]*N,[10**6]*N])
    #A = y
    #b = 0.0
    #convert to matrices
    P = cvxopt.matrix(P, tc='d')
    q = cvxopt.matrix(q, tc='d')
    G = cvxopt.matrix(G, tc='d')
    h = cvxopt.matrix(h, tc='d')
    A = cvxopt.matrix(y, (1,N), tc='d')
    b = cvxopt.matrix(0.0, tc='d')
    #solve
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    alpha = np.array(sol['x'])
    #calculate w_svm
    w2_svm = np.array([0,0], dtype=float)
    w_svm = np.array([0,0,0], dtype=float)
    support_vectors_indices = []
    for j in range(N):
        if alpha[j] > 10**-3:
            support_vectors[i]+=1
            support_vectors_indices.append(j)
        w2_svm += alpha[j]*y[j]*x[j,1:]
    try:
        w_svm[0] = (1./y[support_vectors_indices[0]]) - np.inner(x[support_vectors_indices[0],1:],w2_svm)
    except IndexError:
        bad_run_indices.append(i)
        print "bad run: %d" % i
        print y
    '''
    #validation: all elements of b_vec represent b and should be identical
    b_vec = []
    for index in support_vectors_indices:
        b_vec.append(y[index] - np.inner(x[index,1:],w2_svm))
    print b_vec 
    '''
    w_svm[1] = w2_svm[0]
    w_svm[2] = w2_svm[1]
    #print w_svm
    #validate w_svm: Ein should be 0
    y_svm = np.array([0]*N, dtype=int)
    Ein[i] = 0
    for j in range(N):
        #y_svm[j] = np.sign(np.inner(x[j,:],w_svm))
        y_svm[j] = np.sign(np.inner(x[j,1:],w2_svm)+w_svm[0])
        if y_svm[j] != y[j]:
            Ein[i] += 1./N    
    
    #choose test_x_n, the training points; load test_y
    test_x = np.ones([runs,3], dtype=np.float64) # first column is 1
    test_fy = [0]*runs
    test_gy_pla = [0]*runs
    test_agree_pla = [0]*runs
    test_gy_svm = [0]*runs
    test_agree_svm = [0]*runs
    for j in range(runs):
        test_x[j,1] = random.uniform(-1,1)
        test_x[j,2] = random.uniform(-1,1)
        if test_x[j,2] > fm*test_x[j,1]+fb:
            test_fy[j] = 1
        elif test_x[j,2] < fm*test_x[j,1]+fb:
            test_fy[j] = -1
        else:
            test_fy[j] = 0
        test_gy_pla[j] = np.sign(np.inner(w_pla,test_x[j,:]))
        test_gy_svm[j] = np.sign(np.inner(w_svm,test_x[j,:]))
        if test_fy[j] ==  test_gy_pla[j]:
            test_agree_pla[j] = 1
        if test_fy[j] ==  test_gy_svm[j]:
            test_agree_svm[j] = 1
    Eout_pla[i] = 1 - sum(test_agree_pla)/float(len(test_agree_pla))
    Eout_svm[i] = 1 - sum(test_agree_svm)/float(len(test_agree_svm))
    if Eout_svm[i] <= Eout_pla[i]:
        svm_better += 1
    '''
    #Alternative: measure Eout by integration-- not implemented since it's buggy and doesn't appear to offer speed improvement over stochastic method above
    k = 3
    points = 10**k
    dx = 2./points
    test_x1 = np.arange(-1,1,dx)
    #test_fx2 = [0]*points
    #test_gx2 = [0]*points
    left_area = [0]*points
    right_area = [0]*points
    gm = -w_pla[1]/w_pla[2]
    gb = -1./w_pla[2]
    for j in range(points):
        test_fx2 = fm*test_x1[j] + fb
        test_gx2 = gm*test_x1[j] + gb
        if test_fx2 < test_gx2:
            if test_gx2 > -1 and test_fx2 < 1:
                left_area[j] = dx*(min(1,test_gx2)-max(test_fx2,-1))
            else:
                left_area[j] = 0
        elif test_fx2 > test_gx2:
            if test_fx2 > -1 and test_gx2 < 1:
                left_area[j] = dx*(min(1,test_fx2)-max(test_gx2,-1))
            else:
                left_area[j] = 0
        else:
            left_area[j] = 0
            
        test_fx2 = fm*(test_x1[j]+dx) + fb
        test_gx2 = gm*(test_x1[j]+dx) + gb
        if test_fx2 < test_gx2:
            if test_gx2 > -1 and test_fx2 < 1:
                right_area[j] = dx*(min(1,test_gx2)-max(test_fx2,-1))
            else:
                right_area[j] = 0
        elif test_fx2 > test_gx2:
            if test_fx2 > -1 and test_gx2 < 1:
                right_area[j] = dx*(min(1,test_fx2)-max(test_gx2,-1))
            else:
                right_area[j] = 0
        else:
            right_area[j] = 0
    Eout[i] = float(sum(left_area) + sum(right_area))/8 # because total area of [-1,1]x[-1,1] square is 4
    print (np.array(left_area)-np.array(right_area)).sum
    '''
    
    if i*100 % runs == 0:
        print 'X ',
print
for index in bad_run_indices:
    del Ein[index]
    del Eout_pla[index]
    del Eout_svm[index]
    del support_vectors[index]
Ein_avg = float(sum(Ein))/len(Ein)
print 'Ein is %.3f' % Ein_avg
Eout_pla_avg = float(sum(Eout_pla))/len(Eout_pla)
print 'Average Eout for PLA is %.3f' % Eout_pla_avg
Eout_svm_avg = float(sum(Eout_svm))/len(Eout_svm)
print 'Average Eout for SVM is %.3f' % Eout_svm_avg
svm_better_percentage = float(svm_better)/(runs - len(bad_run_indices))
print "SVM is better %.3f of the time." % svm_better_percentage
support_vectors_avg = float(sum(support_vectors))/len(support_vectors)
print 'Average number of support vectors in SVM: %.3f' % support_vectors_avg
#print sum(iterations)/float(len(iterations))
#print bad_run_indices
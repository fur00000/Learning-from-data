#for homework 2 problems 1-2

import random
import numpy as np

experiments = int(raw_input("Number of experiments: ")) #100000
num_coins = 1000
num_flips = 10

nu_one = [0]*experiments
nu_rand = [0]*experiments
nu_min = [0]*experiments
counter = 0

for i in range(experiments):
    x = np.random.choice(2,size=(num_coins,num_flips))
    c_one = x[0]
    c_rand = x[random.randrange(num_coins)]
    
    c_min_index = 0
    for j in range(1,num_coins):
        if np.sum(x[j]) < np.sum(x[c_min_index]):
            c_min_index = j
    
    c_min = x[c_min_index]
    
    nu_one[i] = np.sum(c_one, dtype=np.float)/num_flips
    nu_rand[i] = np.sum(c_rand, dtype=np.float)/num_flips
    nu_min[i] = np.sum(c_min, dtype=np.float)/num_flips
    if i*100/experiments > counter:
        counter = i*100/experiments
        print '%d percent complete' % counter
    
nu_one_avg = sum(nu_one)/len(nu_one)
nu_rand_avg = sum(nu_rand)/len(nu_rand)
nu_min_avg = sum(nu_min)/len(nu_min)

print "nu_one_avg is %f, nu_rand_avg is %f, and nu_min_avg is %f" % (nu_one_avg, nu_rand_avg, nu_min_avg)
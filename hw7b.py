#answer to homework 7 problem 6

import random

runs = int(raw_input("Nmber of runs: "))

e1 = [0]*runs
e2 = [0]*runs
e = [0]*runs
for i in range(runs):
    e1[i] = random.uniform(0,1)
    e2[i] = random.uniform(0,1)
    if e1[i] < e2[i]:
        e[i] = e1[i]
    else:
        e[i] = e2[i]
exp_e1 = sum(e1)/runs
exp_e2 = sum(e2)/runs
exp_e = sum(e)/runs

print exp_e1 #expected answer 0.5
print exp_e2 #expected answer 0.5
print exp_e #expected answer 0.333
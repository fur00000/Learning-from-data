#homework 5: gradient descent, problems 4-7

import math

coordinate_descent = raw_input("Coordinate Descent? (y/n): ")
#Error function is E(u,v) = (ue**v-2ve**-u)**2
def errf(u, v):
    a = u*math.exp(v)
    b = 2*v*math.exp(-u)
    answer = (a-b)**2
    return answer
    
#Error function partial derivative: dE/du = 2(e**v+2ve**-u)(ue**v-2ve**-u)
def errf_du(u,v):
    a = u*math.exp(v)
    b = 2*v*math.exp(-u)
    answer = 2*(a-b)
    answer *= (b+math.exp(v))
    return answer
    
#Error function partial derivative: dE/dv = 2(ue**v-2e**-u)(ue**v-2ve**-u)
def errf_dv(u,v):
    a = u*math.exp(v)
    b = 2*v*math.exp(-u)
    answer = 2*(a-b)
    answer *= (a-2*math.exp(-u))
    return answer
    
#Start at u,v = (1,1), eta=0.1 is the learning rate
u = 1
v = 1
eta = 0.1
curr_err = errf(u,v)
threshold = float(10)**(-14)
iterations = 0

while curr_err >= threshold:
    print "Iterations: %d, Err: %.3g, (u,v): (%.3g,%.3g)" % (iterations, curr_err, u, v)
    temp_u = u - eta*errf_du(u,v)
    if coordinate_descent == 'n':
        temp_v = v - eta*errf_dv(u,v)
    elif coordinate_descent == 'y':
        temp_v = v - eta*errf_dv(temp_u,v)
    u = temp_u
    v = temp_v
    curr_err = errf(u,v)
    iterations += 1

    raw_input("")

print "Iterations: %d, Err: %.3g, (u,v): (%.3g,%.3g)" % (iterations, curr_err, u, v)
raw_input("")
print "Done!"
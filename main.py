import numpy
import numpy.random
import pandas as pd
import xcsr
import csv
import struct
from parameters import Parameters

"""
    An implementation of an 6-bit multiplexer problem for XCSR
"""
#The maximum reward
rmax = 1000

#The number of steps we learn for
learning_steps = 20000

'''
#Load CSV dataset
file = './dataset/Mux-6.csv' 
with open(file,'r') as f: 
    reader = csv.reader(f)
    data = [low for low in reader]
'''

"""
    Returns a random state of the multiplexer
"""
def state(): 
    return [numpy.random.rand() for i in range(6)]

"""
    The 6bit multiplexer is a single step problem, and thus always is at the end of the problem
"""
def eop():
    return True

"""
    Calculates the reward for performing the action in the given state
"""
def reward(state, action):
    str_state = ''
    for i in range(len(state)):
        if state[i] > 0.5:
            str_state += '1'
        else:
            str_state += '0'
  
    address = str_state[:2]
    data = str_state[2:]

    #Check the action
    if str(action) == data[int(address, 2)]:
        return rmax
    else:
        return 0
 

"""
    Here is the main function. We'll train and validate XCSR!!
"""

#Set parameters
parameters = xcsr.Parameters()
print("[ XCSR General Parameters ]")
print(" Learning Steps =", learning_steps)
print("              N =", parameters.N)
print("           beta =", parameters.beta)
print("          alpha =", parameters.alpha)
print("      epsilon_0 =", parameters.e0)
print("             nu =", parameters.nu)
print("          gamma =", parameters.gamma)
print("       theta_GA =", parameters.theta_GA)
print("            chi =", parameters.chi)
print("             mu =", parameters.mu)
print("      theta_del =", parameters.theta_del)
print("          delta =", parameters.delta)
print("      theta_sub =", parameters.theta_sub)
print("            s_0 =", parameters.s0)
print("              m =", parameters.m)
print("            p_I =", parameters.p_I)
print("      epsilon_I =", parameters.e_I)
print("            F_I =", parameters.F_I)
print("        p_explr =", parameters.p_explr)
print("doGAsubsumption =", parameters.do_GA_subsumption)
print("doASSubsumption =", parameters.do_action_set_subsumption)
print("crossoverMethod = two-point\n")

print("[ XCSR Optional Settings]")
print("            tau =", parameters.tau)

#Construct an XCSR instance
my_xcsr = xcsr.XCSR(parameters, state, reward, eop)

#Make lists to generate CSV
rewardList = [[0] for i in range(learning_steps)]
classifierList = [[0] * 11]
accuracyList = [[0] for i in range(learning_steps - 1000)]

#Begin learning
this_correct = 0
print("\n Iteration     Reward")
print("========== ==========")
for j in range(learning_steps):
    my_xcsr.run_experiment()

    rand_state = state()
    this_correct = this_correct + reward(rand_state, my_xcsr.classify(rand_state))

    if j % 1000 == 0 and j != 0:
        if j < 10000:
            print("     ", j, "  ", round(this_correct / (j+1), 3))
        else:
            print("    ", j, "  ", round(this_correct / (j+1), 3))

    rewardList[j][0]  = reward(rand_state,my_xcsr.classify(rand_state))
    if j == learning_steps - 1:
        classifierList[0][0] = "Classifier"
        classifierList[0][1] = "Condition(Center)"
        classifierList[0][2] = "Condition(Spread)"
        classifierList[0][3] = "Action"
        classifierList[0][4] = "Fitness"
        classifierList[0][5] = "Prediction"
        classifierList[0][6] = "Error"
        classifierList[0][7] = "Experience"
        classifierList[0][8] = "Time Stamp"
        classifierList[0][9] = "Action Set Size"
        classifierList[0][10] = "Numerosity"
        for clas in my_xcsr.population:
            classifierList.append([clas.id, clas.condition_c, clas.condition_s, clas.action, clas.fitness, clas.prediction, clas.error, clas.experience, clas.time_stamp, clas.action_set_size, clas.numerosity])

print("ALL Performance " + ": " + str((this_correct / learning_steps / rmax) * 100) + "%");
print("The whole process is finished. After this, please check reward.csv, classifier.csv, and accuracy.csv files in 'result' folder. Thank you.")

#Make accuracy list (Percentage of correct answers per 1000 iterations)
ini_k = 0
for ini_k in range(learning_steps - 1000):
    sum_1000 = 0
    for k in range(ini_k, 1000 + ini_k):
        sum_1000 = sum_1000 + rewardList[k][0]
    accuracyList[ini_k][0] = sum_1000/1000
    #print(accuracyList)

#Make CSV files
with open('./result/reward.csv','w') as f:
    dataWriter = csv.writer(f, lineterminator='\n')
    dataWriter.writerows(rewardList)

with open('./result/classifier.csv', 'w') as f:
    dataWriter = csv.writer(f, lineterminator='\n')
    dataWriter.writerows(classifierList)

with open('./result/accuracy.csv', 'w') as f:
    dataWriter = csv.writer(f, lineterminator='\n')
    dataWriter.writerows(accuracyList)

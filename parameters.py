"""
A class that represents the parameters of an XCSR system
"""
class Parameters:
    def __init__(self):
        self.bit          = 2
        self.state_length   = self.bit + 2 ** self.bit               #The number of bits in the state
        self.num_actions    = 2               #The number of actions in this system
        self.learning_steps = 20000           #The number of steps we learn for

        self.N         = 1000                 #The maximum size of the population in micro-classifiers
        self.beta      = 0.2                  #The learning rate for the prediction, prediction error, fitness and action set size
        self.alpha     = 0.1                  #The accuracy gap
        self.e0        = 10                   #The minimum error value
        self.nu        = 5                    #The prediction error threhold
        self.gamma     = 0.71                 #The discount factor
        self.theta_GA  = 25                   #The GA threshold
        self.chi       = 0.8                  #The probability of applying crossover in the GA
        self.mu        = 0.04                 #The probability of mutating an allele in the offspring
        self.theta_del = 20                   #The deletion threshold
        self.delta     = 0.1                  #The multiplier for the deletion vote of a classifier
        self.theta_sub = 20                   #The subsumption threshold
        self.p_I       = 0.01                 #The initial prediction value in new classifiers
        self.e_I       = 0.01                 #The initial error value in classifiers
        self.F_I       = 0.01                 #The initial fitness value in classifiers
        self.p_explr   = 1.                   #The probability during action selection ofchoosing the action uniform randomly
        self.theta_mna = 2                    #The minimal number of actions that must be present in a match set, or else covering will occur

        self.tau       = 0.4                  #The probability for Tournament selection in Select Offspring function

        self.m         = 0.1                  #The maximum change in mutation
        self.s0        = 1                    #The maximum spread in covering
        
        self.do_GA_subsumption         = True  #A boolean parameter whether do GA subsumption or not
        self.do_action_set_subsumption = True  #A boolean parameter whether do action set subsumption or not
        self.do_condensation_approach  = False #A boolean parameter whether do Wilson's rule condensation approach or not
        
        

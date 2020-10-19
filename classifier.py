import numpy
import numpy.random
from parameters import Parameters

"""
A condition of a classifier in CSR expression
(OBR and UBR expression have not been supported.)
"""
class Condition:
    """
        Initializes an instance of Classifier's condition
        @param center - The center of the interval
        @param spread - The spread of the interval
    """
    def __init__(self, center, spread):
        self.c = center
        self.s = spread
    """
    Returns the lower limit of the interval
    """
    def get_lower_bound(self):
        return self.c - self.s

    """
    Returns the upper limit of the interval
    """
    def get_upper_bound(self):
        return self.c + self.s

    def __eq__(self, other):
        return self.c == other.c and self.s == other.s

"""
A classifier in XCSR
"""
class Classifier:
    """
        Initializes an instance of Classifier
        @param parameters - A parameters instance (See parameters.py), containing the parameters for this classifier
        @param state - The state of the system to generate classifier
    """
    global_id = 0 #A Globally unique identifier
    def __init__(self, parameters, state = None):
        self.id = Classifier.global_id
        Classifier.global_id = Classifier.global_id + 1
        self.action = numpy.random.randint(0, parameters.num_actions)
        self.prediction = parameters.p_I
        self.error = parameters.e_I
        self.fitness = parameters.F_I
        self.experience = 0
        self.time_stamp = 0
        self.action_set_size = 1
        self.numerosity = 1
        self.condition = []
        self.condition_c = []
        self.condition_s = []
        self.m = parameters.m

        for i in range(parameters.state_length):
            cl_condition = Condition(state[i], numpy.random.rand() * parameters.s0)
            self.condition.append(cl_condition)
            self.condition_c.append(cl_condition.c)
            self.condition_s.append(cl_condition.s)
        #print(Classifier.global_id,self.condition_c,self.condition_s)

    def __str__(self):
        return "Classifier " + str(self.id) + ": " + self.condition + " = " + str(self.action) + " Fitness: " + str(self.fitness) + " Prediction: " + str(self.prediction) + " Error: " + str(self.error) + " Experience: " + str(self.experience)

    """
    APPLY MUTATION (XCSR Version)
       Mutates this classifier, changing the condition and action
       @param state - The state of the system to mutate around
       @param mu - The probability with which to mutate
       @param num_actions - The number of actions in the system
    """
    """
    def _apply_mutation(self, state, mu, num_actions):
        for i in range(len(self.condition)):
            if numpy.random.rand() < mu:
                self.condition[i].c = self.condition[i].c + 2 * self.m * numpy.random.rand() - self.m
                if self.condition[i].c < 0:
                    self.condition[i].c = 0
                elif self.condition[i].c > 1:
                    self.condition[i].c = 1
            if numpy.random.rand() < mu:
                self.condition[i].s = self.condition[i].s + 2 * self.m * numpy.random.rand() - self.m
                if self.condition[i].s < 0:
                    self.condition[i].s = 0
        
        if numpy.random.rand() < mu:
            self.action = numpy.random.randint(0, num_actions)
    """
    def _apply_mutation(self, state, mu, num_actions):
        for i in range(len(self.condition)):
            if numpy.random.rand() < mu:
                if numpy.random.rand() < 0.5:
                    self.condition[i].c += 2 * self.m * numpy.random.rand() - self.m
                    self.condition[i].c = min(max(0.0, self.condition[i].c), 1.0)
                else:
                    self.condition[i].s += 2 * self.m * numpy.random.rand() - self.m
                    self.condition[i].s = max(0.0, self.condition[i].s)

        if numpy.random.rand() < mu:
            used_actions = [self.action]
            available_actions = list(set(range(num_actions)) - set(used_actions))
            self.action = numpy.random.choice(available_actions)

    """
    DELETION VOTE (3.11 Deletion from the population ~The deletion vote~)
       Calculates the deletion vote for this classifier, that is, how much it thinks it should be deleted
       @param average_fitness - The average fitness in the current action set
       @param theta_del - See parameters.py
       @param delta - See parameters.py
    """
    def _deletion_vote(self, average_fitness, theta_del, delta):
        vote = self.action_set_size * self.numerosity
        if self.experience > theta_del and self.fitness / self.numerosity < delta * average_fitness:
            vote = vote * average_fitness / (self.fitness / self.numerosity)
        
        return vote

    """
    COULD SUBSUME (3.12 Subsumption ~Subsumption of a classifier~)
        Returns whether this classifier can subsume others
        @param theta_sub - See parameters.py
        @param e0 - See parameters.py
    """
    def _could_subsume(self, theta_sub, e0):
        return self.experience > theta_sub and self.error < e0

    """
    IS MORE GENERAL (XCSR Verision)
        Returns whether this classifier is more general than another
        @param spec - the classifier to check against
    """
    def _is_more_general(self, spec):
        k = 0
        for i in range(len(self.condition)):
            l_gen = self.condition[i].get_lower_bound()
            u_gen = self.condition[i].get_upper_bound()
            l_spec = spec.condition[i].get_lower_bound()
            u_spec = spec.condition[i].get_upper_bound()
            if l_spec < l_gen or u_gen < u_spec:
                return False
            if l_spec == l_gen and u_gen == u_spec:
                k = k + 1
        if k == len(self.condition):
            return False
        return True

    """
    DOES SUBSUME (3.12 Subsumption ~Subsumption of a classifier~)
        Returns whether this classifier subsumes another
        @param tos - the classifier to check against
        @param theta_sub - See parameters.py
        @param e0 - See parameters.py
    """
    def _does_subsume(self, tos, theta_sub, e0):
        return self.action == tos.action and self._could_subsume(theta_sub, e0) and self._is_more_general(tos)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id
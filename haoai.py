from random import random, randint, sample
import matplotlib.pyplot as plt

# Function to roll a weighted die. Returns True with probability p.
# else False.
def rolldie (p):
    '''Returns True with probability p.'''
    return(random() <= p)

# Our infection model is quite simple (see Carrat et al, 2008). People
# are exposed for E days (the incubation period), then infected for I
# additional days (the symptomatic period). Individuals are infectious
# as either E or I.  Carrat et al. (2008) indicate E~2 and I~7 for
# influenza.
#
# Recall status[] starts at E+I and counts down to REC=0.
#
# If I=7, E=2:
#   SUS REC                   I    I+E
#     |  |                    |     | 
#    -1  0  1  2  3  4  5  6  7  8  9
#          |===================||====|
#
# If I=7, E=2, Q=3:
#   SUS REC         I-Q       I    I+E
#     |  |           |        |     | 
#    -1  0  1  2  3  4  5  6  7  8  9
#          |==========||-------||====|

# Disease model. Each disease has a name, transmissivity coefficient,
# recovery coefficient, and exposure and infection times.
class Disease():
    def __init__(self, name='influenza', t=0.95, E=2, I=7, r=0.0): 
        self.name=name
        self.t=t         # Transmissivity: how easy is it to pass on?
        self.E=E         # Length of exposure (in days)
        self.I=I         # Length of infection (in days)
        self.Q=0	 # Length of quarantine (in days)
        self.r=r         # Probability of lifelong immunity at recovery

    def __repr__(self):
        '''Fancy printed representation for Disease objects.'''
        return("<{}: {},{},{}>".format(self.name.title(), self.I, self.E, self.Q))

    def quarantine(self, Q):
        '''Establish quarantine of Q days for this disease object.'''
        self.Q = min(Q, self.I)

# Agent model. Each agent has a susceptibility value, a vaccination
# state, and a counter that is used to model their current E, I, R or
# S status.
class Agent():
    def __init__(self, g=0, cp=[1.0], s=0.99, q=0.9):
        self.g = g	 # Group identifier
        self.cp = cp	 # Contact probability vector
        self.s = s       # Susceptibility: how frail is my immune system?
        self.q = q	 # Quarantine compliance probability
        self.Q = []	 # Current quarantine status (by disease)
        self.c = []      # Current state S=-1, R=0, E,I > 0 (by disease)
        self.D = []	 # Disease vector (by disease)
        self.v = []      # Vaccination state (by disease)

    def __repr__(self):
        '''Fancy printed representation for Agent objects.'''
        return("<Agent_{}: {}>".format(self.g, self.c))

    # Return index for disease in internal data structures. If disease
    # isn't found, add it.
    def index(self, disease):
        '''Return internal index of disease in self.D list; add new if necessary.'''
        try:
            # Got it; return the index.
            return(self.D.index(disease))
        except:
            # New disease; add it.
            self.D.append(disease)
            self.Q.append(False)
            self.c.append(-1)
            self.v.append(1.0)
            return(len(self.D) - 1)

    # Return vector of True/False indicating state of agent (E,I,Q,S) with respect to disease.
    def state(self, disease):
        '''Returns (E,I,Q,S) truth vector for agent wrt disease.'''
        d = self.index(disease)
        if self.c[d] == 0:
            # Recovered from disease d.
            return((False, False, False, False))
        elif self.c[d] < 0:
            # Susceptible to disease d.
            return((False, False, False, True))
        elif self.c[d] > disease.I:
            # Exposed to disease d.
            return((True, False, False, False))
        elif max(self.Q):
            # Under quarantine for ANY disease, not just disease
            # d. Recall self.Q is a list of True/False values, so if
            # max is True, then some quarantine is underway.
            return((False, False, True, False))
        elif self.c[d] > 0:
            # Infectious with disease d.
            return((False, True, False, False))

    # Set the agent's vaccination value to whatever value you give for
    # the specified disease.
    def vaccinate(self, disease, v):
        '''Models vaccination; v=0 denotes full immunity; v=1 denotes no immunity.'''
        d = self.index(disease)
        self.v[d] = v

    # Susceptible: if other is infected, roll the dice and update your
    # state. No need to check other.state() here anymore, since it is
    # checked prior to invoking the method: we do, however, need to
    # access other's group identity. Note also that I add 1 to I+E,
    # because my first step in run() is to update state: your code may
    # differ. Finally, it's important to "remember" which disease you
    # have so that you can handle recovery and susceptibility
    # correctly when the disease finally runs its course.
    def infect(self, other, disease):
        '''Try to infect self with disease.'''
        d = self.index(disease)
        if self.c[d] < 0 and rolldie(self.s*self.v[d]*self.cp[other.g]*disease.t):
            self.c[d] = disease.E + disease.I + 1
            return(True)
        return(False)

    # Update the status of the agent. Returns infection status: True
    # if you are infectious and False otherwise. This involves
    # decrementing your internal counter if you are actively
    # infected. When you get to 0, you need to flip a (weighted) coin
    # to decide if the agent goes to state R (c=0) or back to state S
    # (c=-1). You also need to handle quarantine: deciding whether to
    # honor it and then making sure you return False while in ANY
    # quarantine.
    def update(self, disease):
        '''Daily status update.'''
        d = self.index(disease)
        if self.c[d] <= 0:
            return(False)
        elif self.c[d] == 1:
            if not rolldie(disease.r):
                # Revert to susceptible, c=-1.
                self.c[d] = -1
            else:
                # Lifelong immunity at recovery, c=0.
                self.c[d] = 0
        elif self.c[d] == disease.I + 1 and disease.Q > 0 and rolldie(self.q):
            # Agent elects to honor quarantine.
            self.c[d] = self.c[d] - 1
            #print('Opting for {} quarantine [{},{},{}]!'.format(disease.name, self.c[d], disease.I, disease.Q))
            self.Q[d] = True
            return(False)
        elif self.Q[d]:
            # Agent is currently in quarantine.
            self.c[d] = self.c[d] - 1
            if self.c[d] == disease.I - disease.Q:
                #print('Expiring {} quarantine [{},{}.{}]!'.format(disease.name, self.c[d], disease.I, disease.Q))
                self.Q[d] = False
                return(True)
            return(False)
        else:
            # One day closer to recovery.
            self.c[d] = self.c[d] - 1
            return(True)
        return(False)

# Simulation model. Each simulation runs for at most a certain
# duration, D, expressed in terms of days. Assumes 1 compartment.
class Simulation():
    def __init__(self, D=500, m=0.001, cmatrix=[[1.0]]):
        self.steps = D		# Maximum number of timesteps
        self.m = m	        # Mixing parameter for this simulation
        self.cmatrix = cmatrix	# Matrix of contact probabilities
        self.agents = []        # List of agents in the simulation
        self.D = []		# Diseases being simulated
        self.history = []       # History of (E, I, R, V) tuples
        self.events = []	# Dictionary of delayed events, keyed by day

    def __repr__(self):
        '''Fancy printed representation for Simulation objects.'''
        return("<Simulation_{}: {}>".format(len(self.agents), self.D))

    # Populates the simulation with n agents from group g.
    def populate(self, n, g = 0):
        '''Populate simulation with n agents from group g.'''
        for i in range(n):
            self.join(Agent(g, self.cmatrix[g]))

    # Add agent to current simulation.
    def join(self, agent):
        '''Add specified agent to current simulation.'''
        self.agents.append(agent)

    # Add disease to current simulation. 
    def introduce(self, disease):
        '''Add specified disease to current simulation.'''
        self.D.append(disease)

    # Seed the simulation with k agents having the specified disease.
    def seed(self, disease, k=1):
        '''Seed a certain number of agents with a particular disease.'''
        # I+E+1, because my first step in run() is to update state.
        for agent in sample(self.agents, k):
            d = agent.index(disease)
            agent.c[d] = disease.E + disease.I + 1

    # This is where the simulation actually happens. The run() method
    # performs at most self.steps iterations, where each iteration
    # updates the agents, counts how many are in E, I, Q and S states,
    # checks if there is an early termination (i.e., no contagious
    # agents left for any disease) and then propagates the infection
    # as per the mixing parameter, m.
    def run(self):
        '''Run the simulation.'''
        for i in range(self.steps):
            # Run any events queued up for today.
            for event in self.events:
                if event[0] == i:
                    if event[1] == 'quarantine':
                        # Set the quarantine length for the specified
                        # disease. Everything else should happen
                        # naturally.
                        event[2].Q = event[3]
                        print('{}: Establishing {} quarantine.'.format(i, event[2].name))
                    elif event[1] == 'vaccinate':
                        # Step through and vaccinate each agent with
                        # probability event[3] and vaccine
                        # effectiveness 1-event[4]. 
                        for agent in self.agents:
                            if rolldie(event[3]):
                                agent.vaccinate(event[2], 1.0-event[4])
                        print('{}: Vaccinating for {}.'.format(i, event[2].name))
                    elif event[1] == 'seed':
                        # Infect event[3] agents with the specified
                        # disease.
                        self.seed(event[2], event[3])
                        print('{}: Seeding {} agents with {}.'.format(i, event[3], event[2].name))

            # Keep track of each disease vector for inclusion in history.
            states = []   
            # Keep track of early termination when no contagion is
            # left. Assume none is left to start with, then change
            # this to True when you encounter an infected agent.
            contagion = False
            # Update each disease. If there aren't any left, exit
            # early.  Note that disease is the outer loop and agent
            # contacts are the inner loop. A better solution might
            # well reverse these, so that all the diseases being run
            # in the same simulation would play out over exactly the
            # same agent contact pattern. Would require a fair bit of
            # rewriting.
            for disease in self.D:
                # First, update agents wrt to this disease.
                for agent in self.agents:
                    agent.update(disease)

                # Next, create a state vector for this disease to
                # drive this cycle. Determining who is infected first
                # avoids letting the infection infect a friend's
                # friend in one pass. Each entry is (E, I, Q, S).
                state = [ a.state(disease) for a in self.agents ]
                # Append (E, IvQ, Q, S) to history.
                states.append((len([ s for s in state if s[0] ]),           # E
                               len([ s for s in state if (s[1] or s[2]) ]), # I or Q
                               len([ s for s in state if s[2] ]),           # Q
                               len([ s for s in state if s[3] ])))          # S
                # If there is anyone still sick, keep going.
                if sum([ sum(x[:3]) for x in state ]) > 0:
                    contagion = True
                # Let each infectious agent try to pass on this infection.
                for j in range(len(self.agents)):
                    if state[j][0] or state[j][1]:
                        # Agent j is infectious and not under quarantine.
                        for k in range(len(self.agents)):
                            if state[k][3]:
                                # Agent k is susceptible.
                                if rolldie(self.m):
                                    self.agents[k].infect(self.agents[j], disease)
            # Append counts to history.
            self.history.append(states)
            # Terminate early if no contagion this iteration and there are no remaining
            # events on the schedule.
            if not contagion and not [ True for event in self.events if event[0] > i ]:
                break
        # Done.
        return(self.history)

    # This method plots the pandemic curve from the self.history variable.
    def plot(self, disease):
        '''Produce a pandemic curve for the simulation.'''
        d = self.D.index(disease)
        plt.title('{}'.format(disease.name.title()))
        plt.axis( [0, len(self.history), 0, len(self.agents)] )
        plt.xlabel('Days')
        plt.ylabel('N')
        plt.plot( [ i for i in range(len(self.history)) ], [ s[d][3] for s in self.history ], 'g-', label='Susceptible' )
        plt.plot( [ i for i in range(len(self.history)) ], [ s[d][0] for s in self.history ], 'y-', label='Exposed' )
        plt.plot( [ i for i in range(len(self.history)) ], [ s[d][1] for s in self.history ], 'r-', label='Infected' )
        plt.plot( [ i for i in range(len(self.history)) ], [ s[d][2] for s in self.history ], 'b-', label='Quarantine' )
        plt.legend(prop={'size':'small'})
        plt.show()

    # Institute a quarantine order for disease at specified time.
    def order(self, time, disease, Q):
        '''Put a quarantine order in place.'''
        self.events.append((time, 'quarantine', disease, Q))

    # Start a vaccination campaign for disease at specified time.
    def campaign(self, time, disease, coverage, v):
        '''Institute a vaccination campaign.'''
        self.events.append((time, 'vaccinate', disease, coverage, v))

    # Introduce some infecteds with disease at specified time.
    def infect(self, time, disease, k):
        '''Introduce some infecteds.'''
        self.events.append((time, 'seed', disease, k))

    # This method is used by the interactive simulation function as
    # well as the configuration file reader.
    def process(self, cmd):
        if cmd[0] == 'add':
            # add 100 0
            self.populate(int(cmd[1]), int(cmd[2]))
        elif cmd[0] == 'disease':
            # disease influenza 0.95 2 7 0
            self.introduce(Disease(cmd[1], float(cmd[2]), int(cmd[3]), int(cmd[4]), float(cmd[5])))
        elif cmd[0] == 'seed':
            # seed 10 influenza 1
            self.infect(int(cmd[1]), [ d for d in self.D if d.name==cmd[2] ][0], int(cmd[3]))
        #elif cmd[0] == 'infect':
        #    # infect 13 influenza 20
        #    self.infect(int(cmd[1]), [ d for d in self.D if d.name==cmd[2] ][0], int(cmd[3]))
        elif cmd[0] == 'quarantine':
            # order 25 influenza Q
            self.order(int(cmd[1]), [ d for d in self.D if d.name==cmd[2] ][0], int(cmd[3]))
        #elif cmd[0] == 'order':
        #    # order 25 influenza Q
        #    self.order(int(cmd[1]), [ d for d in self.D if d.name==cmd[2] ][0], int(cmd[3]))
        elif cmd[0] == 'campaign':
            # campaign 100 influenza coverage v
            self.campaign(int(cmd[1]), [ d for d in self.D if d.name==cmd[2] ][0], float(cmd[3]), float(cmd[4]))
        elif cmd[0] == 'plot':
            # plot influenza
            self.plot([ d for d in self.D if d.name==cmd[1] ][0])
        elif cmd[0] == 'run':
            self.run()

    # This method reads interactive simulation commands from a config
    # file instead.
    def config(self, filename):
        try:
            file = open(filename, 'r')
            # Read in a command
            for line in file:
                self.process(line.split())
        except:
            print('Error in configuration file.')

def simulate():
    # Current simulation object
    S = None
    # Read in a command
    cmd = []
    while not cmd:
        cmd = input('sim> ').split()
    # Keep going while input is not 'bye'.
    while cmd[0] != 'bye':
        if cmd[0] == 'new':
            # Careful: need to "undo" any inadvertent splitting of the
            # contact matrix, hence the join().
            S = Simulation(int(cmd[1]), float(cmd[2]), eval(''.join(cmd[3:])))
        elif S is None:
            print('No simulation object: try "new" first.')
        else:
            S.process(cmd)
        # Read in next command
        cmd = []
        while not cmd:
            cmd = input('sim> ').split()
    return(S)

# A few simple tests
def test0():
    s = Simulation()
    s.populate(3)
    d = Disease('flu', 0.1, 2, 7, 1)
    s.introduce(d)
    s.infect(1, d, 1)
    s.order(1, d, 6)
    s.run()
    s.plot(d)
    return(s)
# No quarantine
def test1():
    # new 500 0.001 [[1.0]]
    s = Simulation()
    # add 1000 0
    s.populate(1000)
    # disease influenza 0.95 2 7 0
    d1 = Disease('influenza', 0.95, 2, 7, 0.9)
    s.introduce(d1)
    # seed 0 influenza 3
    s.infect(0, d1, 3)
    s.run()
    # plot influenza
    s.plot(d1)
    return(s)
# Early but short quarantine
def test2():
    # new 500 0.001 [[1.0]]
    s = Simulation()
    # add 1000 0
    s.populate(1000)
    # disease influenza 0.95 2 7 0
    d1 = Disease('influenza', 0.95, 2, 7, 0.9)
    s.introduce(d1)
    # seed 0 influenza 3
    s.infect(0, d1, 3)
    # order influenza 7
    s.order(0, d1, 3)
    s.run()
    # plot influenza
    s.plot(d1)
    return(s)
# Early and longer quarantine
def test3():
    # new 500 0.001 [[1.0]]
    s = Simulation()
    # add 1000 0
    s.populate(1000)
    # disease influenza 0.95 2 7 0
    d1 = Disease('influenza', 0.95, 2, 7, 0.9)
    s.introduce(d1)
    # seed 0 influenza 3
    s.infect(0, d1, 3)
    # order influenza 7
    s.order(0, d1, 7)
    s.run()
    # plot influenza
    s.plot(d1)
    return(s)
# Late but longer quarantine
def test4():
    # new 500 0.001 [[1.0]]
    s = Simulation()
    # add 1000 0
    s.populate(1000)
    # disease influenza 0.95 2 7 0
    d1 = Disease('influenza', 0.95, 2, 7, 0.9)
    s.introduce(d1)
    # seed 0 influenza 3
    s.infect(0, d1, 3)
    # order influenza 7
    s.order(25, d1, 7)
    s.run()
    # plot influenza
    s.plot(d1)
    return(s)
# Multiple groups
def test5():
    # new 500 0.001 [[1.0,0.5,0.5],[0.5,1.0,0.5],[0.5,0.5,1.0]]
    s = Simulation(500, 0.001, [[1.0,0.5,0.5],[0.5,1.0,0.5],[0.5,0.5,1.0]])
    # add 100 0
    s.populate(100, 0)
    # add 50 1
    s.populate(50, 1)
    # add 200 2
    s.populate(200, 2)
    # disease influenza 0.95 2 7 0
    d1 = Disease('influenza', 0.95, 2, 7, 0)
    s.introduce(d1)
    # disease mumps 0.99 17 10 0.99
    d2 = Disease('mumps', 0.99, 17, 10, 0.99)
    s.introduce(d2)
    # seed 0 influenza 3
    s.infect(0, d1, 3)
    # seed 24 mumps 10
    s.infect(100, d2, 10)
    # order mumps 10
    s.order(118, d2, 10)
    # run
    s.run()
    # plot influenza
    s.plot(d1)
    # plot mumps
    s.plot(d2)
    return(s)

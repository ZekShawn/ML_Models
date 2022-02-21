import numpy as np

class Learner:
    # Initialing...
    def __init__(self, Q, gamma = 0.99, lr=0.1):
        self.gamma = gamma
        self.lr = lr
        self.Q = Q
    
    # Search the max value
    def greedy(self, s, action):
        opChoice = []
        Qmax = -np.inf
        for a in action:
            if self.Q.at[s, str(a)] >= Qmax:
                opChoice.append(a)
                Qmax = self.Q.at[s,str(a)]
        return np.random.choice(opChoice)

    # Search the random and max value
    def e_greedy(self, s, action = [1, 2, 3, 4], epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.random.choice(action)
        else:
            return self.greedy(s,action)
    
    # Calculate the Q-value function
    def step(self, s, a, r, s_next):
        Qmax_s_next = max(self.Q.loc[s_next,:])
        self.Q.at[s,str(a)] += self.lr * (r + self.gamma * Qmax_s_next - self.Q.at[s,str(a)])
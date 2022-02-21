
class ENV:
    def __init__(self) -> None:

        # current point where the agent in
        self.state = [0.,0.]
        self.reward = 0.

    def render(self):
        pass

    def reset(self):

        self.__init__()
        return self.state

    def step(self,action):

        # Calculate the random time_steps and distance
        self.state = [self.state[i] + action[i] for i in range(len(action))]
        done, reward = self.isEND()
        return self.state, reward, done, None

    def isEND(self):

        if (abs(self.state[0]) > 0.5) or (abs(self.state[1]) > 0.5):
            return True, -100
        elif (self.state[0] > 0.4 and self.state[1] > 0.4):
            return True, 100
        else:
            return False, -1
    
    def close(self):

        self.reset()
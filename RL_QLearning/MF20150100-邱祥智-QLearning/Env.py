import numpy as np

class Env:
    # Initialing...
    def __init__(self,rewardForm,state):
        self.state = state
        self.rewardForm = rewardForm

    # New game begins
    def reset(self,rewardForm,state):
        self.__init__(rewardForm=rewardForm,state=state)

    # Return the current reward
    def reward(self, s):
        temp_state = s
        temp_state = temp_state.split(',')
        temp_state = list(map(int,temp_state))
        return self.rewardForm.at[temp_state[1],str(temp_state[0])]

    # To be a new state
    def step(self,state,a,terminal,trap1=-10, trap2=-100):
        # To a List which is 'int list'
        temp_state = state
        temp_state = temp_state.split(',')
        temp_state = list(map(int,temp_state))

        # Up
        if a == 1:
            temp_state[1] += 1
        # Down
        elif a == 2:
            temp_state[1] -= 1
        # Left
        elif a == 3:
            temp_state[0] -= 1
        # Right
        elif a == 4:
            temp_state[0] += 1

        temp_state = ",".join('%s' %id for id in temp_state)

        # Return state
        if temp_state == terminal:
            return temp_state, temp_state, self.reward(temp_state), True
        elif self.reward(temp_state) == trap1 :
            return temp_state, state, self.reward(temp_state), False
        elif self.reward(temp_state) == trap2 :
            return temp_state, state, self.reward(temp_state), True
        else:
            return temp_state, temp_state, self.reward(temp_state), False
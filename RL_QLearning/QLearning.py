import turtle
import pandas as pd
import numpy as np
from Learner import Learner
from Env import Env
from Show import Show
from other import Q_func
from other import action

# Switch
train = 2
trainMode = True
picShow = False

# key arguments
gamma = 0.1
lr = 0.1
epsilon = 0.1

if train == 1:
    # Environment value setting
    start = '6,2'
    terminal = '2,3'
    max_epochs = 10000
    max_steps = 100
    # File load
    saveFile = 'data\\train1.csv'
    rewardFormFile = "data\\env1.csv"
    # Draw Arguments
    Q_Len = 12  # Maze Length
    Q_Wid = 12  # Maze Width
    trap1 = -10 # trap as Wall
    trap2 = -100
    deviation = 6   # the Maze's primmary loc
    size = 42   # the single Squre size
    fontsize = 20   # Code number fontsize
    Rterminal = 20  # reward number
    startList = [6,2]
    terminalList = [2,3]

elif train == 2:
    # Environment value setting
    start = '2,2'
    terminal = '20,21'
    max_epochs = 10000
    max_steps = 1000
    # File load
    saveFile = 'data/train2.csv'
    rewardFormFile = "data/env2.csv"
    # Draw Arguments
    Q_Len = 22  # Maze Length
    Q_Wid = 22  # Maze Width
    trap1 = -10 # trap as Wall
    trap2 = -100
    deviation = 11   # the Maze's primmary loc
    size = 12   # the single Squre size
    fontsize = 10   # Code number fontsize
    Rterminal = 300  # reward number
    startList = [2,2]
    terminalList = [20,21]

# Read data csv
if trainMode:
    Q = Q_func(Len=Q_Len,Wid=Q_Wid)
else:
    Q = pd.read_csv(saveFile,index_col=0,header=0)
rewardForm = pd.read_csv(rewardFormFile,header=0,index_col=0)
rewardForm = rewardForm.fillna(0)

# Create Class
learner = Learner(Q,gamma=gamma,lr=lr)
env = Env(rewardForm,start)
if picShow:
    show = Show(turtle,rewardForm)

# Show the details in terminal with words
steps = np.zeros(max_epochs)
minStep = []
loc = []
if picShow:
    startList = [(a-deviation)*size for a in startList]
    terminalList = [(a-deviation)*size for a in terminalList]
    forPosi = startList
    show.draw(title = 1,size=size,deviation=deviation,trap = trap2,dangerTrap = trap1,terminal = Rterminal,fontsize=fontsize)

for epoch in range(max_epochs):
    # Initialing 
    if picShow:
        title = f"NO.{epoch+1} Let\'s see the foolish robot find the way to green squre..."
        show.reset(forPosi=forPosi,startList = startList,terminalList = terminalList,title=title,size=size)
    env.reset(rewardForm,start)
    tempMinStep = []
    minLoc = []
    print(f"第 {epoch+1} 次寻找路径...")

    for step in range(max_steps):

        a = learner.e_greedy(env.state,epsilon=epsilon)
        s_next, s_new, r, done = env.step(env.state, a, terminal, trap1 = trap1, trap2 = trap2)

        # Show data record
        trap = False
        if env.reward(env.state) == trap2:
            trap = True
        if picShow:
            forPosi = show.move(s_new, env.state, title = epoch+1,size=size,deviation=deviation,trap = trap)

        # Data append
        tempMinStep.append(a)
        minLoc.append(s_new)

        # Learner step
        learner.step(env.state, a, r, s_next)
        if done: break
        env.state = s_new

    print(f"第 {epoch+1} 次寻找路径结束，共尝试 {step+1} 步。")
    steps[epoch] = step+1
    if len(minStep) == 0:
        minStep = tempMinStep
        loc = minLoc
    elif len(minStep) > len(tempMinStep):
        minStep = tempMinStep
        loc = minLoc

steps = pd.DataFrame(steps)

# show
if picShow:
    show.t.done()
if trainMode:
    learner.Q.to_csv(saveFile)
steps.to_csv(f"Problem{train}-gamma{gamma}-lr{lr}-epsilon{epsilon}-{trainMode}.csv")
print(f"最小步数为：{len(minStep)}，其决策为 {action(minStep)}！")
print(f"坐标变化：{loc}，以供参考位置变化是否正确。")
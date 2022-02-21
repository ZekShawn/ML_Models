import turtle
import numpy as np

class Show:
    def __init__(self, t, drawTable):
        self.t = t
        self.drawTable = drawTable

    def draw(self, title = 1, size = 42, deviation = 6, start = '6,2', dangerTrap = -10, trap = -100, terminal = 20,fontsize = 10):
        title = f"NO.{title} Let\'s see the foolish robot find the way to green squre..."
        mapArray = np.array(self.drawTable)
        start = start.split(',')
        start = list(map(int,start))
        for i in range(mapArray.shape[0]):
            for j in range(mapArray.shape[1]):
                color = None
                if mapArray[mapArray.shape[0]-i-1][j] == dangerTrap:
                    color = 'gray'
                elif mapArray[mapArray.shape[0]-i-1][j] == terminal:
                    color = 'green'
                elif mapArray.shape[0]-i == start[0] and j + 1 == start[1]:
                    color = 'yellow'
                elif mapArray[mapArray.shape[0]-i-1][j]:
                    color = 'red'
                else:
                    color = 'white'
                self.drawSqure([(j-deviation + 1) * size,(i-deviation + 1)*size],color = color,title = title,size=size-2)

        self.write(position=[(1-deviation)*size+10, (-deviation)*size+5], 
        lenWid = mapArray.shape[0], lenLen = mapArray.shape[1], size=size, fontSize = fontsize)
    
    def drawSqure(self, position = [0,0], size = 40, color = 'white', title = 'None'):
        self.t.speed(0)
        self.t.delay(0)
        self.t.title(title)
        self.t.penup()
        self.t.goto(position[0],position[1])
        self.t.pendown()
        self.t.begin_fill()
        self.t.fillcolor(color)
        self.t.goto(position[0]+size,position[1])
        self.t.goto(position[0]+size,position[1]-size)
        self.t.goto(position[0],position[1]-size)
        self.t.goto(position[0],position[1])
        self.t.end_fill()
        self.t.hideturtle()

    def write(self, position=[0,0], size = 42, fontSize = 20, lenWid = 12, lenLen = 12):
        for i in range(lenWid):
            self.t.penup()
            self.t.goto(position[0]+i*size, position[1])
            self.t.pendown()
            self.t.write(i+1,font=("Times",fontSize))
            self.t.penup()
            self.t.goto(position[0], position[1]+i*size)
            self.t.pendown()
            self.t.write(i+1,font=("Times",fontSize))

    def move(self, forPosi, backPosi, size = 42, deviation = 6, title = 'None', trap = False):
        title = f"NO.{title} Let\'s see the foolish robot find the way to green squre..."
        forPosi = forPosi.split(',')
        backPosi = backPosi.split(',')
        forPosi = list(map(int,forPosi))
        backPosi = list(map(int,backPosi))
        forPosi = [(a-deviation)*size for a in forPosi]
        backPosi = [(a-deviation)*size for a in backPosi]
        self.drawSqure(position = forPosi, color = 'yellow', title = title, size = size-2)
        if trap:
            self.drawSqure(position = backPosi, color = 'red', title = title, size = size-2)
        else:
            self.drawSqure(position = backPosi, color = 'white', title = title, size = size-2)

        return forPosi

    def reset(self,forPosi,startList,terminalList,title,size):
        self.drawSqure(position = forPosi, color = 'white', title = title, size = size-2)
        self.drawSqure(position = startList, color = 'yellow', title = title, size = size-2)
        self.drawSqure(position = terminalList, color = 'green', title = title, size = size-2)
        self.__init__(turtle,self.drawTable)

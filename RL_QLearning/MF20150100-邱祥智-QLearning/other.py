import pandas as pd

# To show results directly
def action(actList):
    Result = []
    for i in actList:
        if i == 1:
            Result.append('上')
        elif i == 2:
            Result.append('下')
        elif i == 3:
            Result.append('左')
        elif i == 4:
            Result.append('右')
    return Result

def Q_func(Len=12, Wid=12, act = 4):
        index = []
        Q = pd.DataFrame(columns = [str(a+1) for a in range(act)])
        for i in range(Len):
            for j in range(Wid):
                temp_index = ''
                temp_index += str(i+1)
                temp_index += ','
                temp_index += str(j+1)
                index.append(temp_index)
        for i in range(len(index)):
            Q.loc[i] = [0 for a in range(act)]
        Q.index = pd.Series(index)
        return Q
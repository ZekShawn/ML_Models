import numpy as np
import matplotlib.pyplot as plt

def Show(dataList1, dataList2 = None, dataList3 = None, Name1 = None, Name2 = None, Name3 = None,
                double = False, three = False, fileName = 'None',title = 'None'):
    x1_axis = np.arange(1, len(dataList1)+1, 1)
    x1_axis = [i * 12 for i in x1_axis]
    if three:
        x2_axis,x3_axis = x1_axis,x1_axis
        plt.plot(x1_axis,dataList1,'r',x2_axis,dataList2,'g',x3_axis,dataList3,'b')
        label = [Name1,Name2,Name3]
    elif double:
        x2_axis = np.arange(1, len(dataList2)+1, 1)
        plt.plot(x1_axis,dataList1,'r',x2_axis,dataList2,'b')
        label = [Name1, Name2]
    else:
        plt.plot(x1_axis,dataList1,'r')
        label = Name1
    plt.legend(label, loc = 'upper left')
    plt.title(title)
    plt.savefig(fileName,dpi=500)
    # plt.show()
TRAIN = ""
TEST = ""
TARGET = ""

LABEL_LIST = [5, 10, 20]

CATEGORIES = [

]

NUMERICS = [

]


def label_func(x):
    """
    对 x 进行打标签
    :param x:
    :return:
    """
    for i in range(len(LABEL_LIST)):
        if x <= LABEL_LIST[i]:
            return i
        elif x > LABEL_LIST[i] and i == len(LABEL_LIST) - 1:
            return i+1
    return 0

from math import exp

def sigmoid(x):
    try:
        return 1/(1 + exp(-1 * x))
    except:
        raise ValueError(x)

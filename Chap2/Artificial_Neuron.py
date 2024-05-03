import math
import numpy as np

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def activate(x,w):
    res = x.dot(w)
    return sigmoid(res)

if __name__ == "__main__":
    inputs = np.array([.5,.3,.2])
    weights = np.array([.4,.7,.2])
    output= activate(inputs,weights)
    print(output)

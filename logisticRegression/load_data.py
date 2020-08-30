import pandas as pd


def load():
    data = pd.read_csv('ex2data1.txt', sep=',', header=None, names=['x1', 'x2', 'y'])
    return data


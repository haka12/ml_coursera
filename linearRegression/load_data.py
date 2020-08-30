import pandas as pd


def load():
    data = pd.read_csv('ex1data1.txt', sep=',', header=None, names=['X', 'y'])
    return data

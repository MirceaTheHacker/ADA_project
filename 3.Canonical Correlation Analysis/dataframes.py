import pandas as pd
import numpy as np


d1 = {"col1":[1, 2, 3], "col2":[4, 5, 6]}
df1 = pd.DataFrame(d1, index=range(1, 4), dtype=str)
print(df1.values)
print(df1.shape[1])


def Random(low=None, high=None, size=None):
    return ((high - low) * np.random.rand(size) + low)

d2 = Random(-30, 30, 12)
print(d2)
print(d2.shape)
print(d2.itemsize)

npa = np.ndarray(shape=(2, 2), dtype=float, buffer= d2, order='C')
print("Numpy ndarray: \n", npa)

npa1 = np.ndarray(shape=(5, 2), dtype=float, buffer=d2, offset=d2.itemsize*2, order='C')
print(npa1)
print(npa1.shape)

df2 = pd.DataFrame(npa1, index=('L'+str(i+1) for i in range(npa1.shape[0])),
                   columns=('C'+str(j+1) for j in range(npa1.shape[1])))
print(df2)

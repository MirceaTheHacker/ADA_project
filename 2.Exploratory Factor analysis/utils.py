import numpy as np

# Replace NA (not available, not applicable) or NaN (not a number) cells
# with column (variable) mean
# for a pandas DataFrame
def replace_NA(X):
    avgs = np.nanmean(X, axis=0)
    pos = np.where(np.isnan(X))
    print(pos[:])
    X[pos] = avgs[pos[1]]
    return X

# Standardize the column (variable) values
# for a pandas DataFrame
def standize(X):
    avgs = np.mean(X, axis = 0 )
    stds = np.std(X, axis = 0)
    Xstd = (X - avgs) / stds
    return Xstd
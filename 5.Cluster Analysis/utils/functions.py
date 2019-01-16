import numpy as np
import pandas as pd
import scipy.stats as sts
import pandas.api.types as pdt

#Introduction to Cluster Analysis:
#While we often think of statistics as giving definitive answers to well-posed questions, there are some statistical
#  techniques that are used simply to gain further insight into a group of observations. One such technique (which
#  encompasses lots of different methods) is cluster analysis. The idea of cluster analysis is that we have a set of
#  observations, on which we have available several measurements. Using these measurements, we want to find out if the
# observations naturally group together in some predictable way. For example, we may have recorded physical measurements
#  on many animals, and we want to know if there's a natural grouping (based, perhaps on species) that distinquishes
#  the animals from another. (This use of cluster analysis is sometimes called "numerical taxonomy"). As another
# example, suppose we have information on the demographics and buying habits of many consumers. We could use cluster
# analysis on the data to see if there are distinct groups of consumers with similar demographics and buying habits
# (market segmentation).

#There are two very important decisions that need to be made whenever you are carrying
#  out a cluster analysis. The first regards the relative scales of the variables being measured.
#  We'll see that the available cluster analysis algorithms all depend on the concept of measuring the
# distance (or some other measure of similarity) between the different observations we're trying to cluster.
#  If one of the variables is measured on a much larger scale than the other variables, then whatever measure we use
#  will be overly influenced by that variable.
def standardize(X):
    avgs = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    Xstd = (X - avgs) / stds
    return Xstd

def inverse(t, y=None):
    if type(t) is pd.DataFrame:
        for c in t.columns:
            minim = t[c].min();
            maxim = t[c].max()
            if abs(minim) > abs(maxim):
                t[c] = -t[c]
                if y is not None:
                    k = t.columns.get_loc(c)
                    y[:, k] = -y[:, k]
    else:
        for i in range(np.shape(t)[1]):
            minim = np.min(t[:, i]);
            maxim = np.max(t[:, i])
            if np.abs(minim) > np.abs(maxim):
                t[:, i] = -t[:, i]

# Replace NA by mean/mode
def replace_na_df(t):
    for c in t.columns:
        if pdt.is_numeric_dtype(t[c]):
            if t[c].isna().any():
                medie = t[c].mean()
                t[c] = t[c].fillna(medie)
        else:
            if t[c].isna().any():
                modul = t[c].mode()
                t[c] = t[c].fillna(modul[0])

def replace_na(X):
    avgs = np.nanmean(X, axis=0)
    k_nan = np.where(np.isnan(X))
    X[k_nan] = avgs[k_nan[1]]

def tabling(X, col_name=None, obs_name=None, table=None):
    X_tab = pd.DataFrame(X)
    if col_name is not None:
        X_tab.columns = col_name
    if obs_name is not None:
        X_tab.index = obs_name
    if table is None:
        X_tab.to_csv("table.csv")
    else:
        X_tab.to_csv(table)
    return X_tab

def partition(h, k):
    n = np.shape(h)[0] + 1
    g = np.arange(0, n)
    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g == k1] = n + i
        g[g == k2] = n + i
    clusters = ['c'+str(i) for i in pd.Categorical(g).codes]
    return clusters

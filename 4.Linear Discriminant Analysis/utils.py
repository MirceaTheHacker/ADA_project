import numpy as np
import pandas as pd
import scipy.stats as sts
import pandas.api.types as pdt


def standardize(X):
    avg = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    Xstd = (X - avg) / std
    return Xstd

def inv(t, y=None):
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

def pca(X):
    R = np.corrcoef(X, rowvar=False)

    # Compute eigenvalues and eigenvectors
    valp, vecp = np.linalg.eig(R)

    # Sort eigenvalues and eigenvectors in descending order
    k_inv = [k for k in reversed(np.argsort(valp))]
    alpha = valp[k_inv]
    a = vecp[:, k_inv]
    inv(a)

    # Compute factor correlations
    Rxc = a * np.sqrt(alpha)

    # Compute principal components on standardized X matrix
    avg = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    Xstd = (X - avg) / std
    C = Xstd @ a
    return R, alpha, a, Rxc, C

# Replace NA, NaN, through mean/mode on pandas.DataFrame
def replace_na_df(t):
    for c in t.columns:
        if pdt.is_numeric_dtype(t[c]):
            if t[c].isna().any():
                avg = t[c].mean()
                t[c] = t[c].fillna(avg)
        else:
            if t[c].isna().any():
                mod = t[c].mode()
                t[c] = t[c].fillna(mod[0])

def repalce_na(X):
    avg = np.nanmean(X, axis=0)
    k_nan = np.where(np.isnan(X))
    X[k_nan] = avg[k_nan[1]]

def variance_table(alpha):
    m = len(alpha)
    var_cum = np.cumsum(alpha)
    var_per = alpha * 100 / m
    cum_per = np.cumsum(var_per)
    tabel_varianta = pd.DataFrame(data={"Variance": alpha,
                                        "Cumulated variance": var_cum,
                                        "Percent variance": var_per,
                                        "Cumulated percent": cum_per
                                        })
    tabel_varianta.to_csv("variance.csv")


def toTable(X, col_name=None, index_name=None, tabel=None):
    X_tab = pd.DataFrame(X)
    if col_name is not None:
        X_tab.columns = col_name
    if index_name is not None:
        X_tab.index = index_name
    if tabel is None:
        X_tab.to_csv("tabel.csv")
    else:
        X_tab.to_csv(tabel)
    return X_tab

def evaluate(C, alpha, R):
    n = np.shape(C)[0]
    # Comute scores
    S = C / np.sqrt(alpha)

    # Compute cosines
    C2 = C * C
    suml = np.sum(C2, axis=1)
    q = np.transpose(np.transpose(C2) / suml)

    # Compute distributions
    beta = C2 / (alpha * n)

    # Compute commonalities
    R2 = R * R
    common = np.cumsum(R2, axis=1)
    return S, q, beta, common

def bartlett_test(n, l, x, e):
    m, q = np.shape(l)
    v = np.corrcoef(x, rowvar=False)
    psi = np.diag(e)
    v_ = l @ np.transpose(l) + psi
    I_ = np.linalg.inv(v_) @ v
    det_v_ = np.linalg.det(I_)
    trace_I = np.trace(I_)
    chi2_computed = (n - 1 - (2 * m + 4 * q - 5) / 2) * (trace_I - np.log(det_v_) - m)
    g_lib = ((m - q) * (m - q) - m - q) / 2
    chi2_evaluated = sts.chi2_computed.cdf(chi2_computed, g_lib)
    return chi2_computed, chi2_evaluated

def bartlett_factor(x):
    n, m = np.shape(x)
    r = np.corrcoef(x, rowvar=False)
    chi2_computed = -(n - 1 - (2 * m + 5) / 6) * np.log(np.linalg.det(r))
    g_lib = m * (m - 1) / 2
    chi2_evaluated = 1 - sts.chi2_computed.cdf(chi2_computed, g_lib)
    return chi2_computed, chi2_evaluated

def coding(t, vars):
    for v in vars:
        t[v] = pd.Categorical(t[v]).codes
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.cluster.hierarchy as hiclu

def correlogram(t, title=None, valmin=-1, valmax=1):
    f = plt.figure(title, figsize=(8, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    sb.heatmap(np.round(t,2), cmap='bwr', vmin=valmin, vmax=valmax, annot=True)

def variance(alpha, title='Variance Plot'):
    n = len(alpha)
    f = plt.figure(title, figsize=(10,7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    f1.set_xticks(np.arange(1, n+1))
    f1.set_xlabel('Component', fontsize=12, color='r', verticalalignment='top')
    f1.set_ylabel('Variance', fontsize=12, color='r', verticalalignment='bottom')
    f1.plot(np.arange(1, n + 1), alpha, 'ro-')
    f1.axhline(1, c='g')
    j_Kaiser = np.where(alpha < 1)[0][0]
    eps = alpha[:n - 1] - alpha[1:]
    d = eps[:n-2]-eps[1:]
    j_Cattel = np.where(d<0)[0][0]
    f1.axhline(alpha[j_Cattel + 1], c='m')
    return j_Cattel+2, j_Kaiser

def scatter(x, y, label=None, tx="", ty="", title='Scatterplot'):
    f = plt.figure(title, figsize=(10, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    f1.set_xlabel(tx, fontsize=12, color='r', verticalalignment='top')
    f1.set_ylabel(ty, fontsize=12, color='r', verticalalignment='bottom')
    f1.scatter(x=x, y=y, c='r')
    if label is not None:
        n = len(label)
        for i in range(n):
            f1.text(x[i], y[i], label[i])

def t_scatter(x, y, x1, y1, label=None, label1=None, tx="", ty="", title='Scatterplot - Test Dataset'):
    f = plt.figure(title, figsize=(10, 7))
    f1 = f.add_subplot(1, 1, 1)
    f1.set_title(title, fontsize=16, color='b', verticalalignment='bottom')
    f1.set_xlabel(tx, fontsize=12, color='r', verticalalignment='top')
    f1.set_ylabel(ty, fontsize=12, color='r', verticalalignment='bottom')
    f1.scatter(x=x, y=y, c='r')
    f1.scatter(x=x1, y=y1, c='b')
    if label is not None:
        n = len(label)
        p = len(label1)
        for i in range(n):
            f1.text(x[i],y[i],label[i],color='k')
        for i in range(p):
            f1.text(x1[i],y1[i],label1[i],color='k')

def dendrogram(h, labels, title='Hierarchical classification', threshold=None):
    f = plt.figure(figsize=(12, 7))
    axis = f.add_subplot(1, 1, 1)
    axis.set_title(title, fontsize=16, color='b')
    hiclu.dendrogram(h, labels=labels, leaf_rotation=30, ax=axis, color_threshold=threshold)

def show():
    plt.show()

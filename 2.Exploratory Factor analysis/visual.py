import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


def correlogram(t, title=None, valmin=-1, valmax=1):
    plt.figure(title, figsize=(11, 6))
    sb.heatmap(data=np.round(t,2), vmin=valmin, vmax=valmax, cmap='bwr', annot=True)
    plt.title(title)
    plt.show()

def eighenValues(alpha):
    plt.figure("Eigenvalues - Variance of the Components", figsize=(11, 6))

    # one line, one column, index=1
    # f1 = f.add_subplot(1, 1, 1, title="Eigenvalues - Variance of the Components",
    #                    xlabel="Components", ylabel="Eigenvalues")

    plt.title("Eigenvalues - Variance of the Components")
    plt.xlabel("Components")
    plt.ylabel("Eigenvalues")
    plt.plot(alpha, 'bo-')
    plt.xticks([k for k in range(len(alpha))])
    plt.axhline(1, color='r')
    plt.show()

def corrCircle(R, k1, k2, title="The Correlation Circles"):
    plt.figure(title, figsize=(6, 6))
    plt.title(title, fontsize=16, color='b', verticalalignment='bottom')
    T =[t for t in np.arange(0, np.math.pi*2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(0,color='g')
    plt.axvline(0,color='g')
    plt.scatter(R.iloc[:, k1], R.iloc[:, k2], c='r')
    plt.xlabel(R.columns[k1], fontsize=12, color='r', verticalalignment='top')
    plt.ylabel(R.columns[k2], fontsize=12, color='r', verticalalignment='bottom')
    for i in range(len(R)):
        plt.text(R.iloc[i, k1], R.iloc[i, k2], R.index[i])
    plt.show()
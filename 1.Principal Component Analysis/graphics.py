import matplotlib.pyplot as plt
import seaborn as sb


def correlogram(t, title="Correlogram"):
    plt.figure(title, figsize=(10, 7))
    sb.heatmap(data=t, vmin=-1, vmax=1, cmap='bwr')
    plt.title(title)
    plt.show()


def eighenValues(alpha):
    f = plt.figure("Eigenvalues", figsize=(10, 7))

    #one line, oune column, index=1
    f1 = f.add_subplot(1, 1, 1, title="Eigenvalues - Variance of the Components",
                       xlabel="Component", ylabel="Eigenvalue")

    f1.plot(alpha, 'ro-')
    f1.set_xticks([k for k in range(len(alpha))])
    f1.axhline(1, c='g')
    plt.show()


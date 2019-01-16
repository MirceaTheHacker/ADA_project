import numpy as np

def pca(X):
    R = np.corrcoef(X, rowvar=False)

    #compute eigenvectors and eigenvalues
    eigenVal, eigenVect = np.linalg.eig(R)

    #sort the eigenvalues and the corresponding eigenvectors in descending order
    k_reverse = [k for k in reversed(np.argsort(eigenVal))]
    alpha = eigenVal[k_reverse]
    a = eigenVect[:, k_reverse]

    #compute correlation factors
    Rxc = a * np.sqrt(alpha)

    #compute the principal components on standardize X
    avg_var = np.mean(X, axis=0)
    std_deviation = np.std(X, axis=0)
    Xstd = (X - avg_var) / std_deviation
    C = Xstd @ a

    return R, alpha, a, Rxc, C



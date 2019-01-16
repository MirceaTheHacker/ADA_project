import numpy as np


def explore(C, alpha, R):
    n = np.shape(C)[0]

    # Compute scores
    S = C / np.sqrt(alpha)

    # Compute cosines
    C2 = C * C
    sumObs = np.sum(C2, axis=1)
    q = np.transpose(np.transpose(C2) / sumObs)

    # Compute contributions
    beta = C2 / (alpha * n)

    # Compute commonalities
    R2 = R * R
    common = np.cumsum(R2, axis=1)
    return S, q, beta, common

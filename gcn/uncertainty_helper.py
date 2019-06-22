
import numpy as np


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def average_divergence(softmax):
    average_KL = 0
    k = softmax.shape[0]
    for i in range(k):
        for j in range(k):
            average_KL += kl(softmax[i], softmax[j])
    return average_KL / (k*(k-1))

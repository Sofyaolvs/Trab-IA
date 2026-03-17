import numpy as np


def train_test_split_manual(X, y, train_ratio=0.8):
    N = X.shape[0]
    indices = np.random.permutation(N)
    n_train = int(N * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)


def one_hot_encode(labels, C):
    N = len(labels)
    Y = np.zeros((N, C))
    for i, l in enumerate(labels):
        Y[i, l - 1] = 1
    return Y

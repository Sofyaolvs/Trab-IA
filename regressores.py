import numpy as np


def mqo_tradicional(X_train, y_train):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return beta


def mqo_regularizado(X_train, y_train, lambd):
    p = X_train.shape[1]
    I = np.eye(p)
    I[0, 0] = 0
    beta = np.linalg.pinv(X_train.T @ X_train + lambd * I) @ X_train.T @ y_train
    return beta


def modelo_media(y_train):
    return np.mean(y_train)


def predizer(X, beta):
    return X @ beta

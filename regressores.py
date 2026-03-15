import numpy as np


def mqo_tradicional(X_train, y_train):
    """MQO Tradicional: beta = (X^T X)^{-1} X^T y"""
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return beta


def mqo_regularizado(X_train, y_train, lambd):
    """MQO Regularizado (Tikhonov): beta = (X^T X + lambda*I)^{-1} X^T y"""
    p = X_train.shape[1]
    I = np.eye(p)
    I[0, 0] = 0  # Não regularizar o intercepto
    beta = np.linalg.pinv(X_train.T @ X_train + lambd * I) @ X_train.T @ y_train
    return beta


def modelo_media(y_train):
    """Modelo de média da variável dependente."""
    return np.mean(y_train)


def predizer(X, beta):
    """Faz predição: y_hat = X @ beta"""
    return X @ beta

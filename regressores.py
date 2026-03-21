import numpy as np


# MQO classico: acha os pesos que minimizam o erro quadratico
def mqo_tradicional(X_train, y_train):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
    return beta


# MQO com regularizacao: adiciona penalidade pra evitar overfitting
def mqo_regularizado(X_train, y_train, lambd):
    p = X_train.shape[1]
    I = np.eye(p)
    I[0, 0] = 0  # nao penaliza o bias
    beta = np.linalg.pinv(X_train.T @ X_train + lambd * I) @ X_train.T @ y_train
    return beta


# modelo mais simples possivel: so retorna a media do y de treino
def modelo_media(y_train):
    return np.mean(y_train)


# multiplica X pelos pesos pra gerar a predicao
def predizer(X, beta):
    return X @ beta

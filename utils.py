import numpy as np


# separa os dados em treino e teste de forma aleatoria
def train_test_split_manual(X, y, train_ratio=0.8):
    N = X.shape[0]
    indices = np.random.permutation(N)
    n_train = int(N * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# calcula o erro quadratico medio entre o valor real e o previsto
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# calcula o R2 (quanto mais perto de 1, melhor o modelo)
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)  # soma dos residuos
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # variacao total
    return 1 - (ss_res / ss_tot)


# retorna a porcentagem de acertos
def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)


# transforma os rotulos em one-hot (ex: classe 2 vira [0,1,0,0,0])
def one_hot_encode(labels, C):
    N = len(labels)
    Y = np.zeros((N, C))
    for i, l in enumerate(labels):
        Y[i, l - 1] = 1
    return Y

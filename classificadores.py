import numpy as np

from utils import acuracia


# --- Classificador MQO ---

def classificador_mqo_treino(X_train, Y_train_oh):
    """Treina classificador MQO: B = (X^T X)^{-1} X^T Y"""
    B = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train_oh
    return B


def classificador_mqo_predizer(X_test, B):
    """Prediz classes via MQO."""
    scores = X_test @ B
    return np.argmax(scores, axis=1) + 1  # +1 pois classes são 1-5


# --- Classificador Gaussiano Tradicional ---

def estimar_parametros_gaussiano(X, labels, C):
    """Estima média, covariância e prior para cada classe."""
    p = X.shape[0]  # X é p x N
    medias = {}
    covariancias = {}
    priors = {}
    N_total = X.shape[1]

    for c in range(1, C + 1):
        idx = np.where(labels == c)[0]
        Xc = X[:, idx]  # p x Nc
        Nc = Xc.shape[1]

        medias[c] = np.mean(Xc, axis=1, keepdims=True)  # p x 1
        diff = Xc - medias[c]  # p x Nc
        covariancias[c] = (diff @ diff.T) / Nc  # p x p
        priors[c] = Nc / N_total

    return medias, covariancias, priors


def discriminante_gaussiano(x, media, cov, prior):
    """Calcula o discriminante gaussiano para uma amostra."""
    p = len(media)
    diff = x - media

    # Regularização mínima para evitar singularidade
    cov_reg = cov + 1e-6 * np.eye(p)

    try:
        sign, logdet = np.linalg.slogdet(cov_reg)
        if sign <= 0:
            logdet = -1e10
        inv_cov = np.linalg.inv(cov_reg)
    except:
        return -np.inf

    g = -0.5 * logdet - 0.5 * (diff.T @ inv_cov @ diff).item() + np.log(prior + 1e-300)
    return g


def classificador_gaussiano_predizer(X_test, medias, covariancias, priors, C):
    """Prediz classes usando classificador gaussiano."""
    N = X_test.shape[1]
    predicoes = np.zeros(N, dtype=int)

    for i in range(N):
        x = X_test[:, i:i+1]  # p x 1
        melhor_g = -np.inf
        melhor_c = 1

        for c in range(1, C + 1):
            g = discriminante_gaussiano(x, medias[c], covariancias[c], priors[c])
            if g > melhor_g:
                melhor_g = g
                melhor_c = c

        predicoes[i] = melhor_c

    return predicoes


# --- Classificador Gaussiano com Covariâncias Iguais (pooled de todo treino) ---

def estimar_cov_pooled_total(X, labels, C):
    """Estima covariância pooled usando todo o conjunto de treino."""
    p = X.shape[0]
    N = X.shape[1]
    media_global = np.mean(X, axis=1, keepdims=True)
    diff = X - media_global
    cov_pooled = (diff @ diff.T) / N
    return cov_pooled


# --- Classificador Gaussiano com Matriz Agregada ---

def estimar_cov_agregada(covariancias, priors, C):
    """Estima covariância agregada (weighted average das covariâncias de classe)."""
    p = list(covariancias.values())[0].shape[0]
    cov_agregada = np.zeros((p, p))

    for c in range(1, C + 1):
        cov_agregada += priors[c] * covariancias[c]

    return cov_agregada


# --- Classificador de Bayes Ingênuo (Naive Bayes) ---

def classificador_naive_bayes_predizer(X_test, medias, covariancias, priors, C):
    """Naive Bayes: assume independência entre features (covariância diagonal)."""
    N = X_test.shape[1]
    predicoes = np.zeros(N, dtype=int)

    # Criar covariâncias diagonais
    cov_diag = {}
    for c in range(1, C + 1):
        cov_diag[c] = np.diag(np.diag(covariancias[c]))

    for i in range(N):
        x = X_test[:, i:i+1]
        melhor_g = -np.inf
        melhor_c = 1

        for c in range(1, C + 1):
            g = discriminante_gaussiano(x, medias[c], cov_diag[c], priors[c])
            if g > melhor_g:
                melhor_g = g
                melhor_c = c

        predicoes[i] = melhor_c

    return predicoes


# --- Classificador Gaussiano Regularizado (Friedman) ---

def cov_regularizada_friedman(cov_classe, cov_agregada, lambd):
    """Σ_c(λ) = (1-λ) * Σ_c + λ * Σ_agregada"""
    return (1 - lambd) * cov_classe + lambd * cov_agregada


def kfold_cross_validation_lambda(X, labels, C, lambdas_list, k=5):
    """
    K-Fold Cross Validation para encontrar o melhor lambda
    do classificador Gaussiano Regularizado (Friedman).
    X: p x N
    labels: (N,)
    """
    N = X.shape[1]
    indices = np.random.permutation(N)
    fold_size = N // k

    acuracias_por_lambda = {l: [] for l in lambdas_list}

    for fold in range(k):
        # Definir índices de teste e treino
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k - 1 else N
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        X_train = X[:, train_idx]
        X_test = X[:, test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Estimar parâmetros com dados de treino
        medias, covariancias, priors = estimar_parametros_gaussiano(X_train, y_train, C)
        cov_agregada = estimar_cov_agregada(covariancias, priors, C)

        for lambd in lambdas_list:
            # Criar covariâncias regularizadas
            cov_reg = {}
            for c in range(1, C + 1):
                cov_reg[c] = cov_regularizada_friedman(covariancias[c], cov_agregada, lambd)

            # Predizer
            preds = classificador_gaussiano_predizer(X_test, medias, cov_reg, priors, C)
            acc = acuracia(y_test, preds)
            acuracias_por_lambda[lambd].append(acc)

    # Calcular média das acurácias por lambda
    media_acc = {l: np.mean(accs) for l, accs in acuracias_por_lambda.items()}
    melhor_lambda = max(media_acc, key=media_acc.get)

    return melhor_lambda, media_acc

import numpy as np
import matplotlib.pyplot as plt

from utils import acuracia, one_hot_encode
from classificadores import (
    classificador_mqo_treino,
    classificador_mqo_predizer,
    estimar_parametros_gaussiano,
    classificador_gaussiano_predizer,
    estimar_cov_pooled_total,
    estimar_cov_agregada,
    classificador_naive_bayes_predizer,
    cov_regularizada_friedman,
    kfold_cross_validation_lambda,
)


def carregar_dados_classificacao(filepath="EMGsDataset.csv"):
    """Carrega o dataset EMG."""
    data = np.genfromtxt(filepath, delimiter=',')

    if data.shape[0] == 3:
        x1 = data[0, :]
        x2 = data[1, :]
        labels = data[2, :].astype(int)
    elif data.shape[1] == 3:
        x1 = data[:, 0]
        x2 = data[:, 1]
        labels = data[:, 2].astype(int)
    else:
        if data.shape[1] > data.shape[0]:
            x1 = data[0, :]
            x2 = data[1, :]
            labels = data[2, :].astype(int)
        else:
            x1 = data[:, 0]
            x2 = data[:, 1]
            labels = data[:, 2].astype(int)

    return x1, x2, labels


def executar_classificacao():
    """Executa toda a tarefa de classificação."""
    print("\n\n" + "=" * 70)
    print("PARTE 2: TAREFA DE CLASSIFICAÇÃO")
    print("=" * 70)

    # --- 1. Carregar e organizar dados ---
    x1, x2, labels = carregar_dados_classificacao("EMGsDataset.csv")
    N = len(labels)
    C = 5
    p = 2

    print(f"\nDados carregados: {N} amostras, {p} características, {C} classes")
    classes_nomes = {1: 'Neutro', 2: 'Sorriso', 3: 'Sobrancelhas', 4: 'Surpreso', 5: 'Rabugento'}
    for c in range(1, C + 1):
        print(f"  Classe {c} ({classes_nomes[c]}): {np.sum(labels == c)} amostras")

    # Organizar dados nos dois formatos
    X_mqo = np.column_stack([np.ones(N), x1, x2])  # N x 3 (com intercepto)
    Y_mqo = one_hot_encode(labels, C)               # N x C

    X_gauss = np.vstack([x1, x2])  # p x N

    print(f"\nFormato MQO:  X={X_mqo.shape}, Y={Y_mqo.shape}")
    print(f"Formato Gauss: X={X_gauss.shape}")

    # --- 2. Visualização ---
    plt.figure(figsize=(10, 7))
    cores = ['blue', 'green', 'orange', 'red', 'purple']
    for c in range(1, C + 1):
        idx = labels == c
        plt.scatter(x1[idx], x2[idx], alpha=0.3, s=5, c=cores[c-1], label=classes_nomes[c])
    plt.xlabel("Sensor 1 - Corrugador do Supercílio", fontsize=12)
    plt.ylabel("Sensor 2 - Zigomático Maior", fontsize=12)
    plt.title("Gráfico de Espalhamento - EMG Dataset", fontsize=14)
    plt.legend(fontsize=10, markerscale=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/05_espalhamento_classificacao.png", dpi=150)
    plt.close()
    print("Gráfico de espalhamento salvo.")

    # --- 4. K-Fold Cross Validation para lambda ---
    print("\nExecutando K-Fold Cross Validation para encontrar λ ideal...")
    lambdas_list = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    melhor_lambda, media_acc_lambdas = kfold_cross_validation_lambda(
        X_gauss, labels, C, lambdas_list, k=5
    )

    print(f"\nAcurácias por λ:")
    for l, acc in sorted(media_acc_lambdas.items()):
        marcador = " <-- MELHOR" if l == melhor_lambda else ""
        print(f"  λ = {l:.3f}: Acurácia = {acc:.4f}{marcador}")

    print(f"\nλ ideal = {melhor_lambda}")

    # Gráfico lambda vs acurácia
    plt.figure(figsize=(10, 5))
    lvals = sorted(media_acc_lambdas.keys())
    avals = [media_acc_lambdas[l] for l in lvals]
    plt.plot(lvals, avals, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=melhor_lambda, color='red', linestyle='--', label=f'λ ideal = {melhor_lambda}')
    plt.xlabel("λ", fontsize=12)
    plt.ylabel("Acurácia Média (K-Fold)", fontsize=12)
    plt.title("Seleção de Hiperparâmetro λ - Classificador Regularizado (Friedman)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/06_lambda_selecao.png", dpi=150)
    plt.close()
    print("Gráfico de seleção de λ salvo.")

    # --- 5. Simulações de Monte Carlo ---
    R = 500
    nomes_classif = [
        "MQO Tradicional",
        "Gaussiano Tradicional",
        "Gaussiano (Cov. Total Treino)",
        "Gaussiano (Cov. Agregada)",
        "Naive Bayes",
        f"Gaussiano Regularizado (λ={melhor_lambda})"
    ]
    n_classif = len(nomes_classif)
    resultados_acc = np.zeros((R, n_classif))

    print(f"\nExecutando Monte Carlo (R={R} rodadas)...")

    for r in range(R):
        if (r + 1) % 50 == 0:
            print(f"  Rodada {r+1}/{R}...")

        # Particionar dados
        indices = np.random.permutation(N)
        n_train = int(N * 0.8)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # --- Classificador 0: MQO ---
        X_mqo_train = X_mqo[train_idx]
        X_mqo_test = X_mqo[test_idx]
        Y_mqo_train = Y_mqo[train_idx]

        B = classificador_mqo_treino(X_mqo_train, Y_mqo_train)
        preds_mqo = classificador_mqo_predizer(X_mqo_test, B)
        resultados_acc[r, 0] = acuracia(y_test, preds_mqo)

        # --- Dados para classificadores gaussianos ---
        X_g_train = X_gauss[:, train_idx]
        X_g_test = X_gauss[:, test_idx]

        # Estimar parâmetros
        medias, covariancias, priors = estimar_parametros_gaussiano(X_g_train, y_train, C)

        # --- Classificador 1: Gaussiano Tradicional ---
        preds_gt = classificador_gaussiano_predizer(X_g_test, medias, covariancias, priors, C)
        resultados_acc[r, 1] = acuracia(y_test, preds_gt)

        # --- Classificador 2: Gaussiano (Cov. de todo cj. treino) ---
        cov_pooled = estimar_cov_pooled_total(X_g_train, y_train, C)
        cov_iguais = {c: cov_pooled for c in range(1, C + 1)}
        preds_gp = classificador_gaussiano_predizer(X_g_test, medias, cov_iguais, priors, C)
        resultados_acc[r, 2] = acuracia(y_test, preds_gp)

        # --- Classificador 3: Gaussiano (Cov. Agregada) ---
        cov_agregada = estimar_cov_agregada(covariancias, priors, C)
        cov_agregadas = {c: cov_agregada for c in range(1, C + 1)}
        preds_ga = classificador_gaussiano_predizer(X_g_test, medias, cov_agregadas, priors, C)
        resultados_acc[r, 3] = acuracia(y_test, preds_ga)

        # --- Classificador 4: Naive Bayes ---
        preds_nb = classificador_naive_bayes_predizer(X_g_test, medias, covariancias, priors, C)
        resultados_acc[r, 4] = acuracia(y_test, preds_nb)

        # --- Classificador 5: Gaussiano Regularizado (Friedman) ---
        cov_reg = {}
        for c in range(1, C + 1):
            cov_reg[c] = cov_regularizada_friedman(covariancias[c], cov_agregada, melhor_lambda)
        preds_gr = classificador_gaussiano_predizer(X_g_test, medias, cov_reg, priors, C)
        resultados_acc[r, 5] = acuracia(y_test, preds_gr)

    # --- 6. Tabela de resultados ---
    print("\n" + "=" * 90)
    print("RESULTADOS DA CLASSIFICAÇÃO - ACURÁCIA")
    print("=" * 90)
    print(f"{'Modelo':<45} {'Média':>10} {'Desvio':>10} {'Maior':>10} {'Menor':>10}")
    print("-" * 90)

    tabela_class = []
    for i, nome in enumerate(nomes_classif):
        media = np.mean(resultados_acc[:, i])
        std = np.std(resultados_acc[:, i])
        maior = np.max(resultados_acc[:, i])
        menor = np.min(resultados_acc[:, i])
        print(f"{nome:<45} {media:>10.4f} {std:>10.4f} {maior:>10.4f} {menor:>10.4f}")
        tabela_class.append([nome, media, std, maior, menor])

    # --- Gráficos ---

    # Boxplot de acurácias
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot([resultados_acc[:, i] for i in range(n_classif)],
                      labels=[f"C{i}" for i in range(n_classif)],
                      patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, n_classif))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.title("Distribuição da Acurácia por Classificador", fontsize=14)
    plt.ylabel("Acurácia", fontsize=12)
    plt.grid(True, alpha=0.3)

    legenda_labels = [f"C{i}: {nomes_classif[i]}" for i in range(n_classif)]
    plt.figtext(0.5, -0.03, "\n".join(legenda_labels), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig("resultados/07_boxplots_classificacao.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Barras de acurácia
    plt.figure(figsize=(12, 6))
    medias_acc = [np.mean(resultados_acc[:, i]) for i in range(n_classif)]
    stds_acc = [np.std(resultados_acc[:, i]) for i in range(n_classif)]
    x_pos = np.arange(n_classif)

    plt.bar(x_pos, medias_acc, yerr=stds_acc, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    plt.xticks(x_pos, [f"C{i}" for i in range(n_classif)])
    plt.title("Acurácia Média (± Desvio-Padrão) por Classificador", fontsize=14)
    plt.ylabel("Acurácia", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(bottom=min(medias_acc) - 0.05)

    plt.figtext(0.5, -0.03, "\n".join(legenda_labels), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig("resultados/08_barras_classificacao.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\nGráficos de classificação salvos.")

    return tabela_class, melhor_lambda

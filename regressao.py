import numpy as np
import matplotlib.pyplot as plt

from utils import train_test_split_manual, mse, r2_score
from regressores import mqo_tradicional, mqo_regularizado, modelo_media, predizer


def carregar_dados_regressao(filepath="aerogerador.dat"):
    """Carrega o dataset do aerogerador."""
    try:
        data = np.loadtxt(filepath)
    except:
        try:
            data = np.loadtxt(filepath, delimiter=',')
        except:
            data = np.loadtxt(filepath, delimiter='\t')

    x = data[:, 0]  # velocidade do vento
    y = data[:, 1]  # potência gerada
    return x, y


def executar_regressao():
    """Executa toda a tarefa de regressão."""
    print("=" * 70)
    print("PARTE 1: TAREFA DE REGRESSÃO")
    print("=" * 70)

    # --- 1. Carregar e visualizar dados ---
    x_raw, y_raw = carregar_dados_regressao("aerogerador.dat")
    N = len(x_raw)
    print(f"\nDados carregados: {N} amostras")
    print(f"Velocidade do vento: min={x_raw.min():.2f}, max={x_raw.max():.2f}")
    print(f"Potência gerada: min={y_raw.min():.2f}, max={y_raw.max():.2f}")

    # Gráfico de espalhamento
    plt.figure(figsize=(10, 6))
    plt.scatter(x_raw, y_raw, alpha=0.5, s=15, color='steelblue')
    plt.xlabel("Velocidade do Vento", fontsize=12)
    plt.ylabel("Potência Gerada", fontsize=12)
    plt.title("Gráfico de Espalhamento - Aerogerador", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/01_espalhamento_regressao.png", dpi=150)
    plt.close()
    print("Gráfico de espalhamento salvo.")

    # --- 2. Organizar dados (adicionar coluna de 1s para intercepto) ---
    X = np.column_stack([np.ones(N), x_raw])  # N x (p+1), com intercepto
    y = y_raw.reshape(-1, 1)                   # N x 1
    print(f"\nMatriz X: {X.shape}")
    print(f"Vetor y: {y.shape}")

    # --- 3 e 4. Definir modelos ---
    lambdas = [0, 0.25, 0.5, 0.75, 1.0]
    nomes_modelos = ["Média da var. dependente", "MQO Tradicional"]
    nomes_modelos += [f"MQO Regularizado (λ={l})" for l in lambdas]

    n_modelos = len(nomes_modelos)  # 7 modelos no total

    # --- 5. Random Subsampling Validation ---
    R = 500
    resultados_mse = np.zeros((R, n_modelos))
    resultados_r2 = np.zeros((R, n_modelos))

    print(f"\nExecutando Random Subsampling Validation (R={R} rodadas)...")

    for r in range(R):
        X_train, X_test, y_train, y_test = train_test_split_manual(X, y)

        # Modelo 0: Média da variável dependente
        media_val = modelo_media(y_train)
        y_pred_media = np.full_like(y_test, media_val)
        resultados_mse[r, 0] = mse(y_test, y_pred_media)
        resultados_r2[r, 0] = r2_score(y_test, y_pred_media)

        # Modelo 1: MQO Tradicional
        beta_mqo = mqo_tradicional(X_train, y_train)
        y_pred_mqo = predizer(X_test, beta_mqo)
        resultados_mse[r, 1] = mse(y_test, y_pred_mqo)
        resultados_r2[r, 1] = r2_score(y_test, y_pred_mqo)

        # Modelos 2-6: MQO Regularizado com diferentes lambdas
        for i, lambd in enumerate(lambdas):
            beta_reg = mqo_regularizado(X_train, y_train, lambd)
            y_pred_reg = predizer(X_test, beta_reg)
            resultados_mse[r, 2 + i] = mse(y_test, y_pred_reg)
            resultados_r2[r, 2 + i] = r2_score(y_test, y_pred_reg)

    # --- 6. Tabela de resultados ---
    print("\n" + "=" * 90)
    print("RESULTADOS DA REGRESSÃO - MSE")
    print("=" * 90)
    print(f"{'Modelo':<35} {'Média':>12} {'Desvio-Padrão':>15} {'Maior':>12} {'Menor':>12}")
    print("-" * 90)

    tabela_mse = []
    for i, nome in enumerate(nomes_modelos):
        media = np.mean(resultados_mse[:, i])
        std = np.std(resultados_mse[:, i])
        maior = np.max(resultados_mse[:, i])
        menor = np.min(resultados_mse[:, i])
        print(f"{nome:<35} {media:>12.4f} {std:>15.4f} {maior:>12.4f} {menor:>12.4f}")
        tabela_mse.append([nome, media, std, maior, menor])

    print("\n" + "=" * 90)
    print("RESULTADOS DA REGRESSÃO - R²")
    print("=" * 90)
    print(f"{'Modelo':<35} {'Média':>12} {'Desvio-Padrão':>15} {'Maior':>12} {'Menor':>12}")
    print("-" * 90)

    tabela_r2 = []
    for i, nome in enumerate(nomes_modelos):
        media = np.mean(resultados_r2[:, i])
        std = np.std(resultados_r2[:, i])
        maior = np.max(resultados_r2[:, i])
        menor = np.min(resultados_r2[:, i])
        print(f"{nome:<35} {media:>12.4f} {std:>15.4f} {maior:>12.4f} {menor:>12.4f}")
        tabela_r2.append([nome, media, std, maior, menor])

    # --- Gráficos complementares ---

    # Boxplot MSE
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bp1 = axes[0].boxplot([resultados_mse[:, i] for i in range(n_modelos)],
                           labels=[f"M{i}" for i in range(n_modelos)],
                           patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, n_modelos))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].set_title("Distribuição do MSE por Modelo", fontsize=13)
    axes[0].set_ylabel("MSE", fontsize=11)
    axes[0].set_xlabel("Modelo", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Boxplot R²
    bp2 = axes[1].boxplot([resultados_r2[:, i] for i in range(n_modelos)],
                           labels=[f"M{i}" for i in range(n_modelos)],
                           patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    axes[1].set_title("Distribuição do R² por Modelo", fontsize=13)
    axes[1].set_ylabel("R²", fontsize=11)
    axes[1].set_xlabel("Modelo", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Legenda
    legenda_labels = [f"M{i}: {nomes_modelos[i]}" for i in range(n_modelos)]
    fig.text(0.5, -0.02, "  |  ".join(legenda_labels), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig("resultados/02_boxplots_regressao.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nBoxplots salvos.")

    # Gráfico de barras com média e desvio-padrão
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    medias_mse = [np.mean(resultados_mse[:, i]) for i in range(n_modelos)]
    stds_mse = [np.std(resultados_mse[:, i]) for i in range(n_modelos)]
    medias_r2 = [np.mean(resultados_r2[:, i]) for i in range(n_modelos)]
    stds_r2 = [np.std(resultados_r2[:, i]) for i in range(n_modelos)]

    x_pos = np.arange(n_modelos)

    axes[0].bar(x_pos, medias_mse, yerr=stds_mse, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([f"M{i}" for i in range(n_modelos)])
    axes[0].set_title("MSE Médio (± Desvio-Padrão)", fontsize=13)
    axes[0].set_ylabel("MSE", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x_pos, medias_r2, yerr=stds_r2, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"M{i}" for i in range(n_modelos)])
    axes[1].set_title("R² Médio (± Desvio-Padrão)", fontsize=13)
    axes[1].set_ylabel("R²", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')

    fig.text(0.5, -0.02, "  |  ".join(legenda_labels), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig("resultados/03_barras_regressao.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Gráficos de barras salvos.")

    # Visualização da reta de regressão (último treino)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_raw, y_raw, alpha=0.4, s=10, color='steelblue', label='Dados')

    x_plot = np.linspace(x_raw.min(), x_raw.max(), 200)
    X_plot = np.column_stack([np.ones(200), x_plot])

    beta_final = mqo_tradicional(X, y)
    y_plot = X_plot @ beta_final
    plt.plot(x_plot, y_plot.flatten(), 'r-', linewidth=2, label='MQO Tradicional')

    plt.axhline(y=np.mean(y_raw), color='green', linestyle='--', linewidth=2, label='Média')

    plt.xlabel("Velocidade do Vento", fontsize=12)
    plt.ylabel("Potência Gerada", fontsize=12)
    plt.title("Modelos de Regressão - Ajuste aos Dados", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultados/04_ajuste_regressao.png", dpi=150)
    plt.close()
    print("Gráfico de ajuste salvo.")

    return tabela_mse, tabela_r2

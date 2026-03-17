import os
import warnings
warnings.filterwarnings('ignore')

from regressao import executar_regressao
from classificacao import executar_classificacao

os.makedirs("resultados", exist_ok=True)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRABALHO COMPUTACIONAL - REGRESSAO E CLASSIFICACAO")
    print("UNIFOR - Inteligencia Artificial Computacional")
    print("=" * 70)

    arquivo_regressao = "aerogerador.dat"
    arquivo_classificacao = "EMGsDataset.csv"

    if not os.path.exists(arquivo_regressao):
        print(f"\nATENCAO: Arquivo '{arquivo_regressao}' nao encontrado!")
        print(f"   Coloque o arquivo na pasta: {os.getcwd()}")
        print("   Pulando tarefa de regressao...\n")
        tabela_mse, tabela_r2 = None, None
    else:
        tabela_mse, tabela_r2 = executar_regressao()

    if not os.path.exists(arquivo_classificacao):
        print(f"\nATENCAO: Arquivo '{arquivo_classificacao}' nao encontrado!")
        print(f"   Coloque o arquivo na pasta: {os.getcwd()}")
        print("   Pulando tarefa de classificacao...\n")
        tabela_class = None
    else:
        tabela_class, melhor_lambda = executar_classificacao()

    print("\n" + "=" * 70)
    print("EXECUCAO FINALIZADA!")
    print("=" * 70)
    print(f"\nGraficos salvos na pasta: {os.path.abspath('resultados')}/")
    print("\nArquivos gerados:")
    if os.path.exists("resultados"):
        for f in sorted(os.listdir("resultados")):
            print(f"  - resultados/{f}")

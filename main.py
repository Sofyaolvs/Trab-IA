import os
import warnings
warnings.filterwarnings('ignore')

from regressao import executar_regressao
from classificacao import executar_classificacao

# Criar pasta de resultados
os.makedirs("resultados", exist_ok=True)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TRABALHO COMPUTACIONAL - REGRESSÃO E CLASSIFICAÇÃO")
    print("UNIFOR - Inteligência Artificial Computacional")
    print("=" * 70)

    # Verificar se arquivos existem
    arquivo_regressao = "aerogerador.dat"
    arquivo_classificacao = "EMGDataset.csv"

    if not os.path.exists(arquivo_regressao):
        print(f"\n⚠️  ATENÇÃO: Arquivo '{arquivo_regressao}' não encontrado!")
        print(f"   Coloque o arquivo na pasta: {os.getcwd()}")
        print("   Pulando tarefa de regressão...\n")
        tabela_mse, tabela_r2 = None, None
    else:
        tabela_mse, tabela_r2 = executar_regressao()

    if not os.path.exists(arquivo_classificacao):
        print(f"\n⚠️  ATENÇÃO: Arquivo '{arquivo_classificacao}' não encontrado!")
        print(f"   Coloque o arquivo na pasta: {os.getcwd()}")
        print("   Pulando tarefa de classificação...\n")
        tabela_class = None
    else:
        tabela_class, melhor_lambda = executar_classificacao()

    print("\n" + "=" * 70)
    print("EXECUÇÃO FINALIZADA!")
    print("=" * 70)
    print(f"\nGráficos salvos na pasta: {os.path.abspath('resultados')}/")
    print("\nArquivos gerados:")
    if os.path.exists("resultados"):
        for f in sorted(os.listdir("resultados")):
            print(f"  - resultados/{f}")

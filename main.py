"""
main.py - Script principal
Integra todos os componentes do sistema
"""

import numpy as np
from sklearn.model_selection import train_test_split

from data_manager import DataManager
from predictor import PredictorChurn
from nlp_analyzer import AnaliseFeedback
from recommender import RecomendadorProdutos


def main():
    print("\n" + "="*70)
    print(" 🛍️  MOTOR DE INTELIGÊNCIA DE VAREJO")
    print("="*70 + "\n")
    
    # ================================================
    # ETAPA 1: CARREGAR E LIMPAR DADOS
    # ================================================
    print("\n📊 ETAPA 1: Carregamento de Dados\n")
    
    dm = DataManager()
    df = dm.gerar_dados_mock(num_linhas=50000, nome='vendas')
    
    # Mostrar informações
    info = dm.info_dados('vendas')
    print(f"Dados carregados:")
    print(f"  • {info['linhas']} linhas")
    print(f"  • {info['colunas']} colunas")
    print(f"  • Colunas: {', '.join(info['nomes_colunas'][:5])}...\n")
    
    # Limpar dados
    df_limpo = dm.limpar_dados('vendas')
    
    # ================================================
    # ETAPA 2: MODELO PREDITIVO (CHURN/CONVERSÃO)
    # ================================================
    print("\n🌳 ETAPA 2: Modelo Preditivo - Churn vs Conversão\n")
    
    # Preparar features
    features = ['tempo_navegacao_min', 'valor_carrinho', 'idade_cliente']
    X = df_limpo[features].values
    y = df_limpo['compra_concluida'].values
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dados de treinamento: {len(X_train)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras\n")
    
    # Treinar modelo
    modelo = PredictorChurn()
    modelo.treinar(X_train, y_train)
    
    # Fazer predições
    y_pred = modelo.prever(X_test)
    
    # Avaliar
    acuracia, matriz_conf = modelo.avaliar(y_test, y_pred)
    print(f"Acurácia do modelo: {acuracia:.4f}\n")
    
    # Mostrar matriz de confusão
    print("Matriz de Confusão:")
    print(f"  Verdadeiro Negativo: {matriz_conf[0][0]}")
    print(f"  Falso Positivo: {matriz_conf[0][1]}")
    print(f"  Falso Negativo: {matriz_conf[1][0]}")
    print(f"  Verdadeiro Positivo: {matriz_conf[1][1]}\n")
    
    # Plotar matriz de confusão
    modelo.plotar_matriz_confusao(y_test, y_pred, salvar=True)
    
    # Mostrar complexidade
    print(modelo.complexidade_info())
    
    # ================================================
    # ETAPA 3: ANÁLISE DE FEEDBACK (NLP)
    # ================================================
    print("\n📝 ETAPA 3: Análise de Feedback com NLP\n")
    
    comentarios = df_limpo['comentario'].dropna().tolist()
    print(f"Analisando {len(comentarios)} comentários...\n")
    
    analise = AnaliseFeedback()
    analise.treinar(comentarios)
    
    # Mostrar termos frequentes
    analise.exibir_termos_frequentes(15)
    
    # Gerar Word Cloud
    analise.gerar_wordcloud(salvar=True)
    
    # ================================================
    # ETAPA 4: MOTOR DE RECOMENDAÇÃO
    # ================================================
    print("\n🔗 ETAPA 4: Motor de Recomendação\n")
    
    recomendador = RecomendadorProdutos()
    recomendador.treinar(df_limpo)
    
    # Mostrar info da rede
    info_rede = recomendador.info_rede()
    print(f"Informações da Rede:")
    print(f"  • Produtos: {info_rede['num_produtos']}")
    print(f"  • Conexões: {info_rede['num_conexoes']}")
    print(f"  • Densidade: {info_rede['densidade']:.4f}\n")
    
    # Fazer algumas recomendações
    print("Exemplos de Recomendações:\n")
    produtos_teste = df_limpo['product_id'].value_counts().head(3).index
    
    for prod in produtos_teste:
        recomendacoes = recomendador.recomendar(prod, top_k=3)
        if recomendacoes:
            print(f"  Se cliente comprou {prod}:")
            print(f"    → Recomende: {', '.join(recomendacoes)}\n")
    
    # Plotar grafo
    recomendador.plotar_grafo(max_nos=20, salvar=True)
    
    # ================================================
    # RESUMO FINAL
    # ================================================
    print("\n" + "="*70)
    print(" ✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*70)
    print("\n📁 Arquivos gerados:")
    print("   • matriz_confusao.png")
    print("   • wordcloud.png")
    print("   • grafo_produtos.png")
    print("\n")


if __name__ == "__main__":
    main()

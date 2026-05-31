"""
predictor.py - Modelo preditivo para Churn/Conversão
Usa Árvore de Decisão para prever se cliente vai abandonar ou comprar
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from models import ModeloAnalitico


class PredictorChurn(ModeloAnalitico):
    """
    Preditor de Churn usando Árvore de Decisão
    
    Complexidade de Tempo:
    - Treinamento: O(n * m * log(n))
      * n = número de amostras
      * m = número de features
      * log(n) = profundidade da árvore
    
    - Predição: O(log(n)) por amostra
    """
    
    def __init__(self):
        super().__init__("Preditor Churn - Árvore Decisão")
        self.modelo = DecisionTreeClassifier(max_depth=10, random_state=42)
        self.scaler = StandardScaler()
        self.X_treino_escalado = None
        self.metricas = {}
    
    def treinar(self, X_treino, y_treino):
        """Treina o modelo com os dados"""
        print("🌳 Treinando Árvore de Decisão...")
        
        # Normalizar dados
        X_treino_escalado = self.scaler.fit_transform(X_treino)
        
        # Treinar
        self.modelo.fit(X_treino_escalado, y_treino)
        print("✓ Modelo treinado com sucesso\n")
    
    def prever(self, X_teste):
        """Faz predições"""
        X_teste_escalado = self.scaler.transform(X_teste)
        return self.modelo.predict(X_teste_escalado)
    
    def avaliar(self, y_verdadeiro, y_predito):
        """Calcula métricas de desempenho"""
        acuracia = accuracy_score(y_verdadeiro, y_predito)
        conf_matrix = confusion_matrix(y_verdadeiro, y_predito)
        relatorio = classification_report(y_verdadeiro, y_predito, 
                                         output_dict=True)
        
        self.metricas = {
            'acuracia': acuracia,
            'matriz_confusao': conf_matrix,
            'relatorio': relatorio
        }
        
        return acuracia, conf_matrix
    
    def exibir_metricas(self):
        """Mostra as métricas do modelo"""
        if not self.metricas:
            print("Modelo ainda não foi avaliado")
            return
        
        print("📊 Métricas do Modelo:")
        print(f"   Acurácia: {self.metricas['acuracia']:.4f}")
        print(f"\nMatriz de Confusão:")
        print(self.metricas['matriz_confusao'])
    
    def plotar_matriz_confusao(self, y_verdadeiro, y_predito, salvar=False):
        """Plota a matriz de confusão"""
        cm = confusion_matrix(y_verdadeiro, y_predito)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Matriz de Confusão - Churn vs Conversão')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        
        if salvar:
            plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
            print("✓ Matriz salva em 'matriz_confusao.png'")
        
        plt.show()
    
    def complexidade_info(self):
        """Retorna informação sobre complexidade"""
        info = """
╔═══════════════════════════════════════════════════════════════╗
║        ANÁLISE DE COMPLEXIDADE - ÁRVORE DE DECISÃO           ║
╚═══════════════════════════════════════════════════════════════╝

⏱️  TREINAMENTO:
    Complexidade: O(n * m * log(n))
    - n = número de amostras
    - m = número de features  
    - log(n) = profundidade média da árvore
    
    Explicação: O algoritmo testa cada feature em cada nó,
    criando uma árvore com altura logarítmica.

⏱️  PREDIÇÃO:
    Complexidade: O(log(n)) por amostra
    - Navegação até uma folha da árvore
    - Máximo log(n) comparações

✅ VANTAGENS:
    • Rápido e fácil de implementar
    • Interpretável e visualizável
    • Funciona bem com não-linearidades

⚠️  DESVANTAGENS:
    • Propenso a overfitting
    • Sensível a pequenas mudanças
        """
        return info

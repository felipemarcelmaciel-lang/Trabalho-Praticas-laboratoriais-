# 🛍️ Motor de Inteligência de Varejo

**Projeto Final:** Prova de Conceito (PoC) para backend analítico de e-commerce

## 📋 Descrição

Sistema em Python que integra:
- **Gestão de Dados**: Carregamento e limpeza com Pandas
- **Modelo Preditivo**: Árvore de Decisão para prever Churn vs Conversão
- **Análise de Feedback**: NLP com geração de Word Cloud
- **Motor de Recomendação**: Rede de produtos (Grafo)

## 🏗️ Estrutura do Projeto

```
├── models.py              # Classe base abstrata
├── data_manager.py        # Gerenciador de dados
├── predictor.py           # Modelo preditivo
├── nlp_analyzer.py        # Análise de feedback
├── recommender.py         # Motor de recomendação
├── main.py                # Script principal
├── requirements.txt       # Dependências
└── README.md             # Este arquivo
```

## 🚀 Como Usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o programa
```bash
python main.py
```

### 3. Saídas geradas
- `matriz_confusao.png` - Matriz de confusão do modelo
- `wordcloud.png` - Nuvem de palavras dos comentários
- `grafo_produtos.png` - Grafo de recomendações

## 📊 Funcionalidades

### Etapa 1: Dados
- Geração de dados mock (50k linhas)
- Limpeza de valores nulos e inválidos
- Organização em estruturas reutilizáveis

### Etapa 2: Modelo Preditivo
- Árvore de Decisão para classificação
- Normalização de dados
- Cálculo de acurácia e matriz de confusão
- Visualização dos resultados

### Etapa 3: Análise de Feedback
- Extração de termos relevantes
- Remoção de stopwords
- Geração de Word Cloud
- Identificação de sentimentos básicos

### Etapa 4: Recomendação
- Construção de grafo de produtos
- Busca de produtos relacionados
- Visualização da rede
- Métricas de análise de rede

## 📚 Conceitos Usados

**Orientação a Objetos:**
- Classes abstratas e herança
- Encapsulamento de dados
- Polimorfismo

**Algoritmos:**
- Árvore de Decisão (O(n*m*log(n)))
- Busca em Grafo (O(V+E))
- Processamento de texto

**Bibliotecas:**
- `pandas`: Manipulação de dados
- `scikit-learn`: Machine Learning
- `networkx`: Análise de redes
- `wordcloud`: Visualização
- `matplotlib/seaborn`: Gráficos

## 📝 Exemplo de Uso

```python
from data_manager import DataManager
from predictor import PredictorChurn

# Carregar dados
dm = DataManager()
df = dm.gerar_dados_mock(50000)
df_limpo = dm.limpar_dados()

# Treinar modelo
modelo = PredictorChurn()
modelo.treinar(X_train, y_train)

# Fazer predições
predictions = modelo.prever(X_test)
```

## 👨‍💻 Autor
Felipe Marcel Maciel

## 📅 Data
2026

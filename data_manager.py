"""
data_manager.py - Gerenciador simples de dados
Carrega, limpa e organiza os dados para o modelo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


class DataManager:
    """Classe para gerenciar carregamento e limpeza de dados"""
    
    def __init__(self):
        self.dados = {}
    
    def carregar_csv(self, caminho, nome='default'):
        """Carrega um arquivo CSV"""
        try:
            df = pd.read_csv(caminho)
            self.dados[nome] = df
            print(f"✓ Arquivo carregado: {len(df)} linhas")
            return df
        except Exception as e:
            print(f"✗ Erro ao carregar: {e}")
            return None
    
    def gerar_dados_mock(self, num_linhas=50000, nome='vendas'):
        """
        Gera dados simulados para teste
        Simula vendas de um e-commerce
        """
        print(f"📊 Gerando {num_linhas} linhas de dados fictícios...")
        
        categorias = ["Eletrônicos", "Roupas", "Livros", "Móveis", "Esportes", "Beleza"]
        estados = ["SP", "RJ", "MG", "BA", "RS", "PE", "CE", "PA", "SC", "GO"]
        comentarios = [
            "Produto excelente! Recomendo.",
            "Entrega rápida e bem embalado.",
            "Qualidade ruim, não gostei.",
            "Produto chegou quebrado.",
            "Perfeito, muito satisfeito!",
            "Decepcionante, não volto a comprar.",
            "Ótimo custo-benefício.",
            "Atendimento excelente!",
            "Produto com defeito.",
            "Superou minhas expectativas!"
        ]
        
        dados = []
        for i in range(num_linhas):
            dados.append({
                'order_id': f'ord_{i:06d}',
                'product_id': f'prod_{random.randint(1, 500):04d}',
                'categoria': random.choice(categorias),
                'preco': round(random.uniform(10, 2000), 2),
                'tempo_navegacao_min': random.randint(1, 120),
                'valor_carrinho': round(random.uniform(0, 3000), 2),
                'idade_cliente': random.randint(18, 75),
                'estado': random.choice(estados),
                'review_score': random.randint(1, 5),
                'comentario': random.choice(comentarios),
                'compra_concluida': random.choice([0, 1])
            })
        
        df = pd.DataFrame(dados)
        self.dados[nome] = df
        print(f"✓ Dados gerados: {len(df)} linhas\n")
        return df
    
    def limpar_dados(self, nome='vendas'):
        """Remove valores nulos e duplicados"""
        if nome not in self.dados:
            print(f"Dados '{nome}' não encontrados")
            return None
        
        df = self.dados[nome].copy()
        linhas_antes = len(df)
        
        # Remove duplicadas
        df = df.drop_duplicates()
        
        # Remove NaN
        df = df.dropna()
        
        # Remove valores inválidos
        if 'preco' in df.columns:
            df = df[df['preco'] > 0]
        if 'idade_cliente' in df.columns:
            df = df[(df['idade_cliente'] >= 18) & (df['idade_cliente'] <= 100)]
        if 'review_score' in df.columns:
            df = df[(df['review_score'] >= 1) & (df['review_score'] <= 5)]
        
        linhas_depois = len(df)
        print(f"✓ Limpeza: {linhas_antes} → {linhas_depois} linhas")
        
        self.dados[nome] = df
        return df
    
    def obter_dados(self, nome='vendas'):
        """Retorna os dados armazenados"""
        return self.dados.get(nome, None)
    
    def info_dados(self, nome='vendas'):
        """Mostra informações sobre os dados"""
        if nome not in self.dados:
            return None
        
        df = self.dados[nome]
        return {
            'linhas': len(df),
            'colunas': len(df.columns),
            'nomes_colunas': list(df.columns),
            'tipos': df.dtypes.to_dict()
        }

"""
recommender.py - Motor de recomendação baseado em grafo
Recomenda produtos que frequentemente são comprados juntos
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

from models import ModeloAnalitico


class RecomendadorProdutos(ModeloAnalitico):
    """
    Recomenda produtos usando grafo de compras conjuntas
    
    Complexidade:
    - Construção: O(n * m) - n pedidos, m produtos/pedido
    - Recomendação: O(vizinhos) - busca entre vizinhos
    """
    
    def __init__(self):
        super().__init__("Recomendador Produtos")
        self.grafo = nx.Graph()
        self.frequencia_produtos = defaultdict(int)
        self.peso_arestas = defaultdict(int)
    
    def treinar(self, df, col_pedido='order_id', col_produto='product_id'):
        """Constrói o grafo a partir dos dados de vendas"""
        print("🔗 Construindo grafo de produtos...")
        
        # Agrupar produtos por pedido
        pedidos = df.groupby(col_pedido)[col_produto].apply(list)
        
        # Criar conexões entre produtos comprados juntos
        for produtos in pedidos:
            produtos_unicos = list(set(produtos))
            
            # Adicionar nós
            for prod in produtos_unicos:
                if prod not in self.grafo:
                    self.grafo.add_node(prod)
                self.frequencia_produtos[prod] += 1
            
            # Conectar produtos
            for i in range(len(produtos_unicos)):
                for j in range(i + 1, len(produtos_unicos)):
                    prod1, prod2 = produtos_unicos[i], produtos_unicos[j]
                    
                    if self.grafo.has_edge(prod1, prod2):
                        self.grafo[prod1][prod2]['weight'] += 1
                    else:
                        self.grafo.add_edge(prod1, prod2, weight=1)
        
        print(f"✓ Grafo construído: {self.grafo.number_of_nodes()} produtos, "
              f"{self.grafo.number_of_edges()} conexões\n")
    
    def prever(self, produto_id, top_k=5):
        """Recomenda produtos para um dado produto"""
        return self.recomendar(produto_id, top_k)
    
    def recomendar(self, produto_id, top_k=5):
        """Retorna top_k produtos relacionados"""
        if produto_id not in self.grafo:
            return []
        
        # Pegar vizinhos
        vizinhos = self.grafo[produto_id]
        
        # Ordenar por peso (frequência de compra conjunta)
        vizinhos_ordenados = sorted(
            vizinhos.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )
        
        # Retornar top_k
        return [v[0] for v in vizinhos_ordenados[:top_k]]
    
    def plotar_grafo(self, max_nos=20, salvar=False):
        """Plota o grafo de recomendações"""
        print("🎨 Plotando grafo...")
        
        # Pegar produtos mais populares
        top_produtos = sorted(
            self.frequencia_produtos.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_nos]
        
        ids_top = [p[0] for p in top_produtos]
        subgrafo = self.grafo.subgraph(ids_top)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgrafo, k=0.5, iterations=50)
        
        # Tamanho dos nós
        tamanhos = [self.frequencia_produtos[n] * 20 for n in subgrafo.nodes()]
        
        # Largura das arestas
        larguras = [subgrafo[u][v]['weight'] for u, v in subgrafo.edges()]
        
        nx.draw_networkx_nodes(subgrafo, pos, node_size=tamanhos, 
                              node_color='lightblue', edgecolors='black')
        nx.draw_networkx_edges(subgrafo, pos, width=larguras, alpha=0.6)
        nx.draw_networkx_labels(subgrafo, pos, font_size=8)
        
        plt.title('Rede de Produtos Comprados Juntos', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if salvar:
            plt.savefig('grafo_produtos.png', dpi=300, bbox_inches='tight')
            print("✓ Grafo salvo em 'grafo_produtos.png'")
        
        plt.show()
    
    def info_rede(self):
        """Retorna informações sobre a rede"""
        return {
            'num_produtos': self.grafo.number_of_nodes(),
            'num_conexoes': self.grafo.number_of_edges(),
            'densidade': nx.density(self.grafo)
        }

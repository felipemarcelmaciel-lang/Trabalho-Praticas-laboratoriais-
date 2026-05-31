"""
models.py - Classes base para modelos analíticos
Define a estrutura base que outros modelos vão herdar
"""

from abc import ABC, abstractmethod


class ModeloAnalitico(ABC):
    """
    Classe abstrata que define a interface para todos os modelos.
    Todo modelo deve implementar os métodos obrigatórios.
    """
    
    def __init__(self, nome):
        self.nome = nome
        self.modelo = None
        self.metricas = {}
    
    @abstractmethod
    def treinar(self, X, y):
        """Método abstrato para treinar o modelo"""
        pass
    
    @abstractmethod
    def prever(self, X):
        """Método abstrato para fazer predições"""
        pass
    
    def __str__(self):
        return f"Modelo: {self.nome}"

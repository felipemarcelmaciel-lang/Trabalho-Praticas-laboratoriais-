"""
models.py - Classes base abstratas para modelos analíticos
Implementa herança, polimorfismo e encapsulamento para proteger dados sensíveis.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import pickle
import os
from datetime import datetime


class ModeloAnalitico(ABC):
    """
    Classe base abstrata para todos os modelos analíticos do sistema.
    Define a interface obrigatória para todos os modelos filhos.
    
    Atributos privados:
        __dataset: Dados protegidos em memória
        __modelo: Modelo treinado encapsulado
        __metricas: Métricas de desempenho
    """
    
    def __init__(self, nome_modelo: str):
        """
        Inicializa o modelo analítico.
        
        Args:
            nome_modelo: Identificador único do modelo
            
        Raises:
            ValueError: Se nome_modelo estiver vazio
        """
        if not nome_modelo or not isinstance(nome_modelo, str):
            raise ValueError("nome_modelo deve ser uma string não-vazia")
        
        self.__nome = nome_modelo
        self.__dataset = None
        self.__modelo = None
        self.__metricas: Dict[str, Any] = {}
        self.__data_treinamento: Optional[datetime] = None
        
    @property
    def nome_modelo(self) -> str:
        """Retorna o nome do modelo (somente leitura)"""
        return self.__nome
    
    @property
    def dataset(self) -> Optional[Any]:
        """Retorna o dataset encapsulado (protegido)"""
        return self.__dataset
    
    @dataset.setter
    def dataset(self, dados: Any) -> None:
        """Define o dataset com validação"""
        if dados is None:
            raise ValueError("Dataset não pode ser None")
        self.__dataset = dados
    
    @property
    def metricas(self) -> Dict[str, Any]:
        """Retorna as métricas de desempenho"""
        return self.__metricas.copy()
    
    @property
    def modelo_treinado(self) -> Optional[Any]:
        """Retorna o modelo treinado (se existir)"""
        return self.__modelo
    
    def registrar_metricas(self, metricas: Dict[str, Any]) -> None:
        """Registra métricas de desempenho do modelo"""
        if not isinstance(metricas, dict):
            raise TypeError("Métricas devem ser um dicionário")
        self.__metricas.update(metricas)
        self.__data_treinamento = datetime.now()
    
    def _salvar_modelo(self, modelo: Any, caminho: str) -> None:
        """
        Método protegido para salvar modelo em disco.
        
        Args:
            modelo: Objeto do modelo a ser salvo
            caminho: Caminho do arquivo
            
        Raises:
            IOError: Se houver erro ao salvar
        """
        try:
            os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
            with open(caminho, 'wb') as f:
                pickle.dump(modelo, f)
            self.__modelo = modelo
        except Exception as e:
            raise IOError(f"Erro ao salvar modelo em {caminho}: {str(e)}")
    
    def _carregar_modelo(self, caminho: str) -> Any:
        """
        Método protegido para carregar modelo do disco.
        
        Args:
            caminho: Caminho do arquivo do modelo
            
        Returns:
            Modelo carregado
            
        Raises:
            FileNotFoundError: Se arquivo não existir
            IOError: Se houver erro ao carregar
        """
        try:
            if not os.path.exists(caminho):
                raise FileNotFoundError(f"Modelo não encontrado em {caminho}")
            
            with open(caminho, 'rb') as f:
                self.__modelo = pickle.load(f)
            return self.__modelo
        except Exception as e:
            raise IOError(f"Erro ao carregar modelo de {caminho}: {str(e)}")
    
    @abstractmethod
    def treinar_modelo(self, X_treino: Any, y_treino: Any) -> None:
        """
        Método abstrato obrigatório para treinar o modelo.
        Cada classe filha deve implementar sua própria lógica de treinamento.
        
        Args:
            X_treino: Features para treinamento
            y_treino: Labels para treinamento
            
        Raises:
            NotImplementedError: Se não implementado pela classe filha
        """
        raise NotImplementedError("treinar_modelo() deve ser implementado")
    
    @abstractmethod
    def gerar_predicao(self, X_teste: Any) -> Any:
        """
        Método abstrato obrigatório para gerar predições.
        Cada classe filha deve implementar sua própria lógica de predição.
        
        Args:
            X_teste: Features para predição
            
        Returns:
            Predições geradas pelo modelo
            
        Raises:
            NotImplementedError: Se não implementado pela classe filha
        """
        raise NotImplementedError("gerar_predicao() deve ser implementado")
    
    @abstractmethod
    def obter_complexidade(self) -> str:
        """
        Retorna a complexidade de tempo do algoritmo.
        
        Returns:
            String descrevendo a complexidade (ex: O(n log n))
        """
        raise NotImplementedError("obter_complexidade() deve ser implementado")
    
    def __repr__(self) -> str:
        """Representação em string do modelo"""
        return f"<{self.__class__.__name__}: {self.__nome}>"
    
    def __str__(self) -> str:
        """Descrição legível do modelo"""
        status = "Treinado" if self.__modelo else "Não treinado"
        return f"{self.__class__.__name__} ({self.__nome}) - Status: {status}"


class ModeloPreditivo(ModeloAnalitico):
    """Classe base para modelos preditivos (Churn/Conversão)"""
    
    def __init__(self, nome_modelo: str):
        super().__init__(nome_modelo)
        self._modelo_interno = None
    
    def salvar_predicoes(self, predicoes: Any, caminho: str) -> None:
        """Salva predições em arquivo"""
        try:
            import json
            if hasattr(predicoes, 'tolist'):
                predicoes = predicoes.tolist()
            with open(caminho, 'w') as f:
                json.dump(predicoes, f)
        except Exception as e:
            raise IOError(f"Erro ao salvar predições: {str(e)}")


class ModeloRecomendacao(ModeloAnalitico):
    """Classe base para modelos de recomendação"""
    
    def __init__(self, nome_modelo: str):
        super().__init__(nome_modelo)
        self._rede_produtos = None
    
    @abstractmethod
    def recomendar_produtos(self, produto_id: str, top_k: int = 5) -> list:
        """
        Recomenda produtos relacionados.
        
        Args:
            produto_id: ID do produto referência
            top_k: Número de recomendações
            
        Returns:
            Lista de produtos recomendados
        """
        raise NotImplementedError()

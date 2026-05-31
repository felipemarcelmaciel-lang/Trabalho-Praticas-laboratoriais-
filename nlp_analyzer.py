"""
nlp_analyzer.py - Análise de comentários e Word Cloud
Extrai termos relevantes dos comentários de clientes
"""

import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

from models import ModeloAnalitico

# Palavras comuns a remover
STOPWORDS = {'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
             'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais',
             'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das'}


class AnaliseFeedback(ModeloAnalitico):
    """Analisa comentários de clientes"""
    
    def __init__(self):
        super().__init__("Análise Feedback")
        self.comentarios = []
        self.termos = Counter()
    
    def treinar(self, comentarios, y=None):
        """Carrega e processa comentários"""
        print("📝 Processando comentários...")
        self.comentarios = comentarios
        self._extrair_termos()
        print(f"✓ {len(comentarios)} comentários analisados\n")
    
    def prever(self, texto):
        """Analisa sentimento básico de um texto"""
        texto_limpo = self._limpar_texto(texto)
        
        palavras_pos = {'excelente', 'ótimo', 'adorei', 'perfeito', 
                       'recomendo', 'maravilhoso', 'gostei'}
        palavras_neg = {'ruim', 'horrível', 'péssimo', 'problema',
                       'defeito', 'odeio', 'decepcionante'}
        
        pos = sum(1 for t in texto_limpo if t in palavras_pos)
        neg = sum(1 for t in texto_limpo if t in palavras_neg)
        
        if pos > neg:
            return "positivo"
        elif neg > pos:
            return "negativo"
        else:
            return "neutro"
    
    def _limpar_texto(self, texto):
        """Remove pontuação e normaliza"""
        texto = texto.lower()
        texto = re.sub(r'[^a-záéíóúâêôãõç\s]', '', texto)
        palavras = texto.split()
        palavras = [p for p in palavras if p not in STOPWORDS and len(p) > 2]
        return palavras
    
    def _extrair_termos(self):
        """Extrai termos mais frequentes"""
        for comentario in self.comentarios:
            if pd.isna(comentario):
                continue
            termos_limpos = self._limpar_texto(str(comentario))
            self.termos.update(termos_limpos)
    
    def termos_frequentes(self, top_n=15):
        """Retorna os termos mais frequentes"""
        return self.termos.most_common(top_n)
    
    def gerar_wordcloud(self, comentarios=None, salvar=False):
        """Gera Word Cloud dos comentários"""
        print("☁️  Gerando Word Cloud...")
        
        if comentarios is None:
            comentarios = self.comentarios
        
        # Juntar todos os comentários
        texto = ' '.join([str(c) for c in comentarios if pd.notna(c)])
        
        # Limpar
        palavras = []
        for c in comentarios:
            if pd.notna(c):
                palavras.extend(self._limpar_texto(str(c)))
        
        texto_final = ' '.join(palavras)
        
        # Gerar nuvem
        wc = WordCloud(width=1000, height=600, background_color='white',
                      colormap='viridis').generate(texto_final)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Análise de Comentários', fontsize=14, fontweight='bold')
        
        if salvar:
            plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
            print("✓ Word Cloud salva em 'wordcloud.png'")
        
        plt.show()
    
    def exibir_termos_frequentes(self, top_n=15):
        """Exibe os termos mais frequentes"""
        print(f"\n🔤 Top {top_n} Termos Mais Frequentes:")
        for i, (termo, freq) in enumerate(self.termos_frequentes(top_n), 1):
            print(f"   {i:2d}. {termo:15s} ({freq:3d}x)")

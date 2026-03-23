'''Implementar uma classe capaz de normalizar dados conforme os métodos citados em sala:

MimMax Scaler
Label Encoding
One Hot Encodig
Deve-se implementar os métodos de reversão também

A classe deve ser reaproveitável

Utilizar o arquivo dados_normalizar.csv (na pasta da aula 2) para testar seu código.'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# a classe precisa aprender os parâmentros da ida para poder voltar dps

class Normalizador:
    def norm_MinMax(dados):
        # transforma valores para uma escala de 0 a 1 | fórmula: x_norm = (x - min) / (max - min)
        # para voltar = x_original = x_norm * (max - min) + min
        # p reverter tem que guardar o mínimo e máx de cada coluna
        # n é adequado para a coluan 'sexo'
        scaler = MinMaxScaler(feature_range=(0,1))
        return scaler.fit_transform(dados)

    def norm_rotulo_LE():
        # serve p transformar categorias em números int
        # p reverter tem q guardar o mapeamento e o inverso
        # ideal p coluna 'sexo'
        pass

    def norm_HE():
        # tbm é bom p coluna categórica
        # cria uma coluna p cada categoria
        # p voltar precisa saber quais colunas pertencem ao grupo original
       pass


dados = pd.read_csv('dados_normalizar.csv')


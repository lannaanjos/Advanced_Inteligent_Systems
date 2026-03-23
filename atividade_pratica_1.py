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

class Humano:
    def normalizaMinMax(dados):
        scaler = MinMaxScaler(feature_range=(0,1))
        return scaler.fit_transform(dados)

    def normalizaRotulo():
        pass

    def pontoQuente():
       pass


dados = pd.read_csv('dados_normalizar.csv')


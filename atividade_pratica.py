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

class Humano:
    def normalizaMinMax():
        pass

    def normalizaRotulo():
        pass

    def pontoQuente():
       pass


dados = pd.read_csv('dados_normalizar.csv')
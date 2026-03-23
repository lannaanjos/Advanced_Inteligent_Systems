'''Implementar uma classe capaz de normalizar dados conforme os métodos citados em sala:

MimMax Scaler
Label Encoding
One Hot Encodig
Deve-se implementar os métodos de reversão também

A classe deve ser reaproveitável

Utilizar o arquivo dados_normalizar.csv (na pasta da aula 2) para testar seu código.'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# a classe precisa aprender os parâmentros da ida para poder voltar dps

class Normalizador:
    # fazer transform e inverso_transform
    
    def __init__(self):
        self.parametros_minmax = {} # p reverter tem que guardar o --> mínimo e máx <-- de cada coluna
        self.mapeamento_labels = {} # p reverter tem q guardar o mapeamento e o inverso, --> dicionario de categorias <--
        self.categorias_onehot = {} # p voltar precisa saber quais colunas pertencem ao grupo origina # guarda a --> lista de categorias <--
        
    # /\/\/\/\ min max /\/\/\/\
    
    def norm_MinMax(self, df, colunas):
        # transforma valores para uma escala de 0 a 1         
        for coluna in colunas:
            minimo = df[coluna].min()
            maximo = df[coluna].max()
            self.parametros_minmax[colunas] = {"min": minimo, "max":maximo}
            
    def tranforma_MinMax(self, df, colunas):
        # fórmula: x_norm = (x - min) / (max - min)
         # n é adequado para a coluan 'sexo'
        resultado = df.copy()
        
        for coluna in colunas:
            minimo = self.parametros_minmax[coluna]["min"]
            maximo = self.parametros_minmax[coluna]["max"]
            
            if maximo == minimo: # se for vazio
                resultado[coluna] = 0
            else:
                resultado[coluna] = (resultado[coluna] - minimo) / (maximo - minimo)
                
            return resultado
    
    def inverte_MinMax(self, df, colunas):
        # fórmula p voltar: x_original = x_norm * (max - min) + min
        resultado = df.copy()
        
        for coluna in colunas:
            minimo = self.parametros_minmax[coluna]["min"]
            maximo = self.parametros_minmax[coluna]["max"]
            
            resultado[coluna] = resultado[coluna] * (minimo * maximo) + minimo
            
            return resultado
        
    # /\/\/\/\/\ label encoding /\/\/\/\/\

    def norm_rotulos(self, df, coluna):
        # serve p transformar categorias em números int
        rotulos = sorted(df[coluna].unique())
        mapeamento = {rotulo: i for i, rotulo in enumerate(rotulos)}
        self.mapeamento_labels[coluna] = mapeamento

    def transfroma_rotulo_LE(self, df, coluna):
        # ideal p coluna 'sexo'
        pass
    
    def inverte_LE(self, df, coluna):
        pass

    # /\/\/\/\ one hot encoding /\/\/\/\

    def norm_OHE():
        # tbm é bom p coluna categórica
        # cria uma coluna p cada categoria
        
       pass
   
    def transforma_OHE():
       pass
   
    def inverte_OHE():
       pass


dados = pd.read_csv('dados_normalizar.csv')


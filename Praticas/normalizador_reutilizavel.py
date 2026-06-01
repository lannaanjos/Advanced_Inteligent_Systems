'''Implementar uma classe capaz de normalizar dados conforme os métodos citados em sala:

MimMax Scaler
Label Encoding
One Hot Encodig
Deve-se implementar os métodos de reversão também

A classe deve ser reaproveitável

Utilizar o arquivo dados_normalizar.csv (na pasta da aula 2) para testar seu código.'''
import pandas as pd

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
            self.parametros_minmax[coluna] = {"min": minimo, "max":maximo}
            
    def transforma_MinMax(self, df, colunas):
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
            
            resultado[coluna] = resultado[coluna] * (maximo - minimo) + minimo
            
        return resultado
        
    # /\/\/\/\/\ label encoding /\/\/\/\/\
    # funciona criando um dicionário de tradução entre rotulo e numero

    def norm_rotulos(self, df, coluna):
        # serve p transformar categorias em números int
        rotulos = sorted(df[coluna].unique()) # olha coluna e pega valores possiveis sem repetir de forma organizada
        mapeamento = {rotulo: i for i, rotulo in enumerate(rotulos)} # cria um 'codigo' numerico p cada rotulo, ex: F = 1, M = 0
        self.mapeamento_labels[coluna] = mapeamento # guarda mapa na classe

    def transfroma_rotulo_LE(self, df, coluna):
        # ideal p coluna 'sexo'
        resultado = df.copy()
        
        mapa = self.mapeamento_labels[coluna]        
        resultado[coluna] = resultado[coluna].map(mapa) # aplica o dicionario em cada valor
        
        return resultado
    
    def inverte_LE(self, df, coluna):
        resultado = df.copy()
        
        mapa = self.mapeamento_labels[coluna]
        mapa_invertido = {valor: chave for chave, valor in mapa.items()} # mapeia inversamente (valores viram chaves)
        
        resultado[coluna] = resultado[coluna].map(mapa_invertido) # volta ao estado original
        
        return resultado

    # /\/\/\/\ one hot encoding /\/\/\/\
    # categoria ganha prórpia coluna
    def norm_OHE(self, df, coluna):
        # tbm é bom p coluna categórica
        # cria uma coluna p cada categoria
        categorias = sorted(df[coluna].unique()) # olha coluna p descobrir qnts valores tem nela e ordena
        self.categorias_onehot[coluna] = categorias
   
    def transforma_OHE(self, df, coluna):
        resultado = df.copy()        
        categorias = self.categorias_onehot[coluna] # recupera oq foi aprendido no norm_OHE
        # ex: humano é homem -> sex_f = 0, sex_m = 1
        
        for categoria in categorias: # cada categoria cria uma nova coluna
            nome_col = f"{coluna}_{categoria}"
            resultado[nome_col] = (resultado[coluna] == categoria).astype(int)
            # ex: categoria = 'F'
            # resultado['sexo_F'] = (resultado["sexo"] == 'F').astype(int)
            # se col. sexo for: M F F M, fica: 0, 1, 1, 0
            
        resultado = resultado.drop(columns=[coluna]) # se liva da "coluna mae" q já n é necessária pq temos as colunas booleanas
        return resultado
        
   
    def inverte_OHE(self, df, coluna):
        resultado = df.copy()
        categorias = self.categorias_onehot[coluna] # pega as categorias salvas
        
        colunas_OHE = [f"{coluna}_{categoria}" for categoria in categorias]
        
        def recupera_cat(linha): # recupera colunas como valores de cmapos
            for categoria in categorias:
                nome_coluna = f"{coluna}_{categoria}"
                if linha[nome_coluna] == 1:
                    return categoria
            return None
        
        resultado[coluna] = resultado.apply(recupera_cat, axis=1)
        resultado = resultado.drop(columns=colunas_OHE)
        
        return resultado   
    
    ######## ex 2 2 2 2 2 2
    def estrutura_nova_instancia_OHE(self, df, coluna):
        resultado = df.copy()
        
        categorias = self.categorias_onehot[coluna]
        valor = str(df[coluna]).lower()
        
        resultado = {}
        
        for categoria in categorias:
            nome_coluna = f"{coluna}_{categoria}"
            resultado[nome_coluna] = 1 if valor == str(categoria).lower() else 0
            # se o valor da nova instancia for igual a categoria atual é um senão eh 0
            # ex: cor = azul, logo cor_azul = 1, cor_vemerlho = 0, cor_verde = 0
            
        return resultado       


if __name__ == "__main__":
    dados = pd.read_csv('dados_normalizar.csv', sep=';')

    for col in ['idade', 'altura', 'peso']:
        dados[col] = dados[col].astype(str).str.replace(',', '.', regex=False)
        dados[col] = pd.to_numeric(dados[col])

    # teste MinMax
    normalizador = Normalizador()

    print(f"Dados Originais:\n{dados}")
    print()

    colunas_numericas = ["idade", "altura", "peso"]

    normalizador.norm_MinMax(dados, colunas_numericas)
    dados_minmax = normalizador.transforma_MinMax(dados, colunas_numericas)

    print(f"MinMax:\n{dados_minmax}")
    print()

    dados_minxmax_revertido = normalizador.inverte_MinMax(dados_minmax, colunas_numericas)

    print(f"Min Max Invertido:\n{dados_minxmax_revertido}")
    print()

    # teste label encoding

    normalizador.norm_rotulos(dados, 'sexo')
    dados_rotulos = normalizador.transfroma_rotulo_LE(dados, "sexo")

    print(f"Label Encoding:\n{dados_rotulos}")
    print()

    dados_rotulos_invertidos = normalizador.inverte_LE(dados_rotulos, "sexo")

    print(f"LE Invertido:\n{dados_rotulos_invertidos}")
    print()

    # teste one hot encoding

    normalizador.norm_OHE(dados, 'sexo')
    dados_OHE = normalizador.transforma_OHE(dados, "sexo")

    print(f"One Hot Encoding:\n{dados_OHE}")
    print()

    dados_OHE_invertido = normalizador.inverte_OHE(dados_OHE, "sexo")

    print(f"OHE Invertido:\n{dados_OHE_invertido}")
    print()



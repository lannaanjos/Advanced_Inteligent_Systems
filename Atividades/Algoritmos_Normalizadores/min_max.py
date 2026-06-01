class MinMax:
    def __init__(self) -> None:
        self.parametros_minmax = {} # p reverter tem que guardar o --> mínimo e máx <-- de cada coluna
    
    def set_parametros_MinMax(self, df, colunas):
        # setta os parametros      
        for coluna in colunas:
            minimo = df[coluna].min()
            maximo = df[coluna].max()
            self.parametros_minmax[coluna] = {"min": minimo, "max":maximo}
            
    def encode_MinMax(self, df, colunas):
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
    
    def decode_MinMax(self, df, colunas):
        # fórmula p voltar: x_original = x_norm * (max - min) + min
        resultado = df.copy()
        
        for coluna in colunas:
            minimo = self.parametros_minmax[coluna]["min"]
            maximo = self.parametros_minmax[coluna]["max"]
            
            resultado[coluna] = resultado[coluna] * (maximo - minimo) + minimo
            
        return resultado
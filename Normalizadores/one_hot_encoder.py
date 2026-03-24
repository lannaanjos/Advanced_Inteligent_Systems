class One_Hot:
    def __init__(self) -> None:
        self.categorias_onehot = {} # p voltar precisa saber quais colunas pertencem ao grupo origina # guarda a --> lista de categorias <--

     # /\/\/\/\ one hot encoding /\/\/\/\
    # categoria ganha prórpia coluna
    def set_parametros_OHE(self, df, coluna):
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
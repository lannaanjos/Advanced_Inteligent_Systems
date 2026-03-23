class Label_Encoder:
    def __init__(self) -> None:
        self.mapeamento_labels = {} # p reverter tem q guardar o mapeamento e o inverso, --> dicionario de categorias <--r
    
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
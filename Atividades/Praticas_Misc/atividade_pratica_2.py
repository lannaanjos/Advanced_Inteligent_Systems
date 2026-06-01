'''Você deve considerar o seguinte cenário

    Após normalizar os dados categóricos nominais, a tabela de dados passou a ter as colunas [cor_azul, cor_verde, cor_vermelho]. Essa estrutura será, então, utilizada para treinamento do modelo de IA;
    Quando uma nova instância é recebida, ela terá o atributo cor=Azul.
    Ocorre que essa nova instancia precisa ser alterada, de forma a obedecer a estrutura dos dados normalizados: [cor_azul, cor_verde, cor_vermelho]

Pede-se:

    Implemente um método que recebe os dados da nova instância e altera sua estrutura de acordo com os dados normalizados com one hot encoder'''
    
from Atividades.Praticas_Misc.normalizador_reutilizavel import Normalizador
import pandas as pd

dados = pd.DataFrame({
    'cor': ["preto", "branco", "azul", "verde"]
})

normalizador = Normalizador()

normalizador.norm_OHE(dados, 'cor') # aprende a estrutura

novo_dado = pd.DataFrame({
    'cor': ['amarelo']
})

resultado = normalizador.estrutura_nova_instancia_OHE(novo_dado, 'cor')

print(f"Novo dado original:\n{novo_dado}")
print()

print(f"Novo dado ajustado:\n{resultado}")

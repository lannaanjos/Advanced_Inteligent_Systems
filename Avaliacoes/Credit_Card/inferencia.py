# Modulo de inferência - Escolhido: RANDOM FOREST
# results:
'''Melhores parâmtros Random Forest:
{'n_estimators': 70, 'min_samples_split': 2, 'max_features': 'log2', 'max_depth': 62, 'criterion': 'gini'}
Acurácia RF:  0.8728867964904772
Especificidade RF:  0.9083488472003437
Sensibilidade RF:  0.8376918703808982'''

import pickle
import pandas as pd
from pathlib import Path
 
PATH_SPLITS     = Path(__file__).parent
PATH_CREDITO_RF = PATH_SPLITS / "credito_rf.pkl"
PATH_COLUNAS    = PATH_SPLITS / "colunas.pkl"
 
# /\/\ CARREGANDO MODELO E COLUNAS DO TREINO
modelo  = pickle.load(open(PATH_CREDITO_RF, "rb"))
colunas = pickle.load(open(PATH_COLUNAS,    "rb"))
 
# /\/\ DADOS DO CLIENTE
# Preencha com os dados do cliente a ser avaliado
cliente = {
    'LIMIT_BAL': 50000,
    'SEX':       'M',
    'EDUCATION': 'Middle School',
    'MARRIAGE':  'Married',
    'AGE':       30,
    'PAY_0':     0,
    'PAY_2':     0,
    'PAY_3':     0,
    'PAY_4':     0,
    'PAY_5':     0,
    'PAY_6':     0,
    'BILL_AMT1': 10000,
    'BILL_AMT2': 9000,
    'BILL_AMT3': 8000,
    'BILL_AMT4': 7000,
    'BILL_AMT5': 6000,
    'BILL_AMT6': 5000,
    'PAY_AMT1':  1000,
    'PAY_AMT2':  1000,
    'PAY_AMT3':  1000,
    'PAY_AMT4':  1000,
    'PAY_AMT5':  1000,
    'PAY_AMT6':  1000,
}
 
# /\/\ PRÉ-PROCESSAMENTO
# Converte para DataFrame, aplica get_dummies e realinha colunas com as do treino
df_cliente = pd.DataFrame([cliente])
 
df_cliente = pd.get_dummies(df_cliente, columns=['SEX', 'EDUCATION', 'MARRIAGE'])
 
# Garante que as colunas são exatamente as mesmas do treino
# Colunas ausentes (ex: SEX_F quando o cliente é M) viram 0
df_cliente = df_cliente.reindex(columns=colunas, fill_value=0)
 
# /\/\ INFERÊNCIA
classe_predita = modelo.predict(df_cliente)[0]
probabilidades = modelo.predict_proba(df_cliente)[0]
 
# /\/\ RESULTADO
print("RESULTADO DA ANÁLISE DE CRÉDITO")
print()
 
if classe_predita == 1:
    print("Predição: INADIMPLENTE")
else:
    print("Predição: ADIMPLENTE")
 
print()
print("Distribuição de probabilidade:")
print(f"  Adimplente:   {probabilidades[0] * 100:.2f}%")
print(f"  Inadimplente: {probabilidades[1] * 100:.2f}%")

'''RESULTADO DA ANÁLISE DE CRÉDITO

Predição: INADIMPLENTE

Distribuição de probabilidade:
  Adimplente:   31.43%
  Inadimplente: 68.57%
'''
 


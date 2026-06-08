# O Banco AgoraVai precisa detereminar se os clientes do seu cartão de crédito se tornarão inadimplementes ou não (em ingles Default).
# 
# Para tanto, você deve implementar um sistema inteligente que determine se um cliente tem risco e qual é o risco (score) de se tornar inadimplente usando o arquivo de dados default_of_credit_card_clients.
# 
# Critérios de avaliação:
#
#    Pipeline completo
#    Seleção do melhor modelo
#    Módulo de inferência
#    Resultado acompanhado da distribuição de probabilidade
#
#
# Entregue o link para o seu repositório de código.

# TARGET = default payment next month
# segundo o kaggle não tem nenhum colunas com dados missing
# prof falou que é bom olhar o dataset mas ele parece estar ok
# aparentemente há dados q parecem num mas são cat

# UTIL
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from numpy import asarray

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# VISU
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PATH_CM_RF = Path(__file__).parent / "confusion_rf.png" # random forest
PATH_CM_DT = Path(__file__).parent / "confusion_dt.png" # decision tree

# /\/\ CARREGANDO ARQUIVO
DADOS_PATH = Path(__file__).parent.parent.parent / "Data" / "default_of_credit_card_clients.csv"
dados = asarray(pd.read_csv(DADOS_PATH, sep=';'))
dados = dados.fillna(dados.mean())

print(dados)

# /\/\ NORMALIZAÇÃO DOS DADOS
# Separar categoricos de numericos

colunas_categoricas = ['SEX', 'EDUCATION', 'MARRIAGE']
colunas_numericas = dados.drop(columns = [colunas_categoricas])

dados_cat = pd.get_dummies(data[colunas_categoricas])

encoder = OneHotEncoder(sparse=false)
onehot = encoder.fit_transform(dados)

# /\/\ BALANCEAMENTO DOS DADOS

# SEPARAÇÃO DADOS DE TREINO E DADOS DE TESTE

# TREINAR E SALVAR MODEL

# Modelos e inferência em arquivos a parte
# Seria melhor deixar tudo modular mas não deu tempo :P




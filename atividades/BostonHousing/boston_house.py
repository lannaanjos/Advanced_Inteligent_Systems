'''
Instruções

Utilize o arquivo HousingData.csv disponível na pasta da Aula6
A descrição do arquivo pode ser encontrada em https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset
Treine um modelo de clusters conforme nossas aulas
Implemente um módulo de descrição dos segmentos obtidos durante o treinamento
Implemente um módulo de inferência que receba os dados de um imóvel desconhecido e informe a qual cluster tal imóvel pertence
'''

import pickle
import pandas as pd 
from sklearn.preprocessing import MinMax

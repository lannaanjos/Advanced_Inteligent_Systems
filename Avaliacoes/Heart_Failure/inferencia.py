import pickle
import pandas as pd
from pathlib import Path

NORM_PATH = Path(__file__).parent / "norm_heart_failure.pkl"
CLUSTER_PATH = Path(__file__).parent / "cluster_heart_failure.pkl"
COLS_PATH = Path(__file__).parent / "nomes_cols.pkl"

COLUNAS_CONT = ["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium", "time"]
COLUNAS_BIN = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

# inferencia nova
novo_paciente_cont = pd.DataFrame([[60, 250, 35, 200000, 1.2, 137, 180]],
                                   columns=COLUNAS_CONT)

novo_paciente_bin = pd.DataFrame([[0, 1, 0, 1, 0]],
                                  columns=COLUNAS_BIN)

# norm só as contínuas com o scaler do treino
normalizador = pickle.load(open(NORM_PATH, "rb"))
cont_norm = normalizador.transform(novo_paciente_cont)
cont_norm = pd.DataFrame(cont_norm, columns=COLUNAS_CONT)

# junta
paciente_pronto = cont_norm.join(novo_paciente_bin)

# Predicao
modelo = pickle.load(open(CLUSTER_PATH, "rb"))
cluster_predito = modelo.predict(paciente_pronto)

print(f"Cluster predito: {cluster_predito[0]}")

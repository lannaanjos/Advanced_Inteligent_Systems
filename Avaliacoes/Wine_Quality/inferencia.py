# Carrega o melhor modelo treinado (Random Forest) e prediz a qualidade de um novo vinho.
import pickle
import pandas as pd
from pathlib import Path

BASE_PATH = Path(__file__).parent

NORM_PATH    = BASE_PATH / "norm_wq.pkl"
COLUNAS_PATH = BASE_PATH / "colunas_wq.pkl"
RF_PATH      = BASE_PATH / "rf_wq.pkl"

# /\/\ CARREGANDO MODELO
scaler = pickle.load(open(NORM_PATH,    "rb"))
colunas = pickle.load(open(COLUNAS_PATH, "rb"))
modelo = pickle.load(open(RF_PATH,      "rb"))

# /\/\ DADOS DO VINHO
vinho = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "wine_type": "red",
}

# /\/\ PRE-PROCESSAMENTO
df = pd.DataFrame([vinho])

dados_cat = pd.get_dummies(df[["wine_type"]], prefix_sep="_", dtype=int)
dados_num = df.drop(columns=["wine_type"])

dados_num_norm = scaler.transform(dados_num)
dados_num_norm = pd.DataFrame(dados_num_norm, columns=dados_num.columns)

atributos = dados_num_norm.join(dados_cat.reset_index(drop=True))
atributos = atributos.reindex(columns=colunas, fill_value=0)

# /\/\ INFERENCIA
qualidade = modelo.predict(atributos)[0]
probabilidades = modelo.predict_proba(atributos)[0]
classes = modelo.classes_
confianca = probabilidades.max() * 100

print("RESULTADO DA ANÁLISE DE QUALIDADE DO VINHO")
print()
print(f"  Qualidade predita: {qualidade}")
print(f"  Confiança:         {confianca:.2f}%")
print()
print("  Distribuição de probabilidade:")
for c, p in sorted(zip(classes, probabilidades), key=lambda x: -x[1]):
    print(f"    Nota {c}   {p*100:.2f}%")

# Qualidade predita: 5
# Confiança:         98.57%


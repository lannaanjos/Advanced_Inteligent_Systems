# HIPERPARAMETRIZAÇÃO RANDOM FOREST
# Definição dos domínios para os hiperparametros + criação da grade de valores
rf_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
    # Criterion é  afunção usada p/ medir a qualidade de uma divisão na árvore interna da RF
    # - Gini: mede impureza, ou seja, o quão misturadas as classes estão
    # - Entropy: mede ganho de informação, ou seja, o quanto uma divisão reduz a incerteza 
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [int(x) for x in np.linspace(start=2, stop=10, num=2)],
    'max_depth': [int(x) for x in np.linspace(start=10, stop=100, num=20)],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)

rf_search = RandomizedSearchCV(
    estimator = rf,
    param_distributions = rf_grid,
    n_iter = 10,
    cv = 3,
    verbose = 2,
    n_jobs = -1,
    random_state = 42
)

rf_search.fit(att_treino, classe_treino)

# Mostra resultado da hiperparametrização
print("Melhores parâmtros Random Forest:")
print(rf_search.best_params_)

# Avaliação
melhor_rf = rf_search.best_estimator_ # pega modelo campeão da busca antes de avaliar
preditos_rf = melhor_rf.predict(att_teste)

acuracia_rf = accuracy_score(classe_teste, preditos_rf)
tn, fp, fn, tp = confusion_matrix(classe_teste, preditos_rf).ravel()
especificidade_rf = tn / (tn + fp)
sensibilidade_rf = tp / (tp + fn)

print("Acurácia RF: ", acuracia_rf)
print("Especificidade RF: ", especificidade_rf)
print("Sensibilidade RF: ", sensibilidade_rf)

ConfusionMatrixDisplay.from_estimator(melhor_rf, att_teste, classe_teste)
plt.title("Random Forest")
plt.savefig(PATH_CM_RF)
plt.close()


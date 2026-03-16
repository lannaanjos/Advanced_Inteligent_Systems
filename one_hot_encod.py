import pandas as pd

df = pd.DataFrame({
    'cor': ['Vemelho', 'Roxo', 'Preto', 'Amarelo']
})

dumnics = pd.get_dummies(df['cor'], prefix='cor')

print('Dummies:')
print(dumnics)
import pandas as pd

def reverter_cor(linha_dummy: pd.Series) -> str:
    # recebe linha com colunas one hot e retorna categoria original

    # converte em dataframe de uma linha
    df_linha = linha_dummy.to_frame().T

    # reverte usndo pandas.from_dummies
    df_original = pd.from_dummies(df_linha, sep='_')

    return df_original.iloc[0,0]

df = pd.DataFrame({
    'cor': ['Vemelho', 'Roxo', 'Preto', 'Amarelo']
})

dumnics = pd.get_dummies(df['cor'], prefix='cor')

print()

linha_exemplo = dumnics.iloc[1]

cor_original = reverter_cor(linha_exemplo)

print(f"Linha dummy:\n{linha_exemplo}\n\n")
print(f'Linha original:\n{cor_original}')
import pandas as pd

df = pd.read_csv('../data/raw/registros-base-10-08-2025.csv')
df.replace(['NULL', 'null', ''], pd.NA, inplace=True)
df = df.drop(columns=['id'])
df = df.dropna(subset=[
    'temperatura', 
    'umidade', 
    'pressao', 
    'velocidade_vento', 
    'precipitacao_chuva',
    'tempo_chuva',
    'intensidade_chuva',
    'ponto_orvalho',
    'historico_alagamento',
    'precipitacao_acumulada'
])
colunas_numericas = [
    'temperatura', 'umidade', 'pressao', 'velocidade_vento',
    'precipitacao_chuva', 'tempo_chuva', 'ponto_orvalho',
    'precipitacao_acumulada'
]
df[colunas_numericas] = df[colunas_numericas].apply(pd.to_numeric, errors='coerce')

df['historico_alagamento'] = df['historico_alagamento'].astype(int)

df['intensidade_chuva'] = df['intensidade_chuva'].str.lower()

df['bairro'] = (
    df['bairro']
    .str.lower()
    .str.replace(r'\s+', '', regex=True)
)

df['data_hora'] = pd.to_datetime(df['data_hora'].str.replace('T', ' '))

inconsistentes = df[(df['tempo_chuva'] == 0) & (df['precipitacao_chuva'] > 0)]
print(f"{len(inconsistentes)} registros inconsistentes encontrados.")

df.to_csv('../data/processed/dados_processados.csv', index=False)

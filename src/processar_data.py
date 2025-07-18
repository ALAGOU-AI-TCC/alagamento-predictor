import pandas as pd

df = pd.read_csv('../data/raw/registros-base-17-07-2025.csv')
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

df['intensidade_chuva'] = df['intensidade_chuva'].str.lower()

df['data_hora'] = pd.to_datetime(df['data_hora'].str.replace('T', ' '))

df.to_csv('../data/processed/dados_processados.csv', index=False)

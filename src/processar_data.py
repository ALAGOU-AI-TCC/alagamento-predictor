import pandas as pd
import numpy as np  # NOVO (pra chuva_media_h)

# --- ARQUIVO COM DECLIVE ---
df = pd.read_csv('../data/raw/testdeclive.csv')  # NOVO (antes era registros-base-10-08-2025.csv)

# normaliza valores nulos
df.replace(['NULL', 'null', ''], pd.NA, inplace=True)

# remove colunas inúteis se existirem
df = df.drop(columns=['id'], errors='ignore')

# renomeia a coluna do declive, caso tenha vindo como declive_graus1/slope...
for c in df.columns:  # NOVO
    if c.lower().startswith('declive_graus') or 'slope' in c.lower():
        df = df.rename(columns={c: 'declive_graus'})
        break

# SE X/Y vieram do QGIS, normaliza (opcional, seguro)
if 'X' in df.columns and 'longitude' not in df.columns:  # NOVO
    df = df.rename(columns={'X': 'longitude'})
if 'Y' in df.columns and 'latitude' not in df.columns:   # NOVO
    df = df.rename(columns={'Y': 'latitude'})

# --- DROPNA: só o essencial (REMOVI velocidade_vento) ---
df = df.dropna(subset=[
    'temperatura',
    'umidade',
    'pressao',
    'precipitacao_chuva',
    'tempo_chuva',
    'intensidade_chuva',
    'ponto_orvalho',
    'historico_alagamento',
    'precipitacao_acumulada',
    'solo_elevacao',     # NOVO
    'declive_graus'      # NOVO
])

# --- CONVERSÕES NUMÉRICAS (REMOVI velocidade_vento) ---
colunas_numericas = [
    'temperatura', 'umidade', 'pressao',
    'precipitacao_chuva', 'tempo_chuva', 'ponto_orvalho',
    'precipitacao_acumulada', 'solo_elevacao', 'declive_graus'  # NOVO
]
df[colunas_numericas] = df[colunas_numericas].apply(pd.to_numeric, errors='coerce')

# higiene numérica leve
df['tempo_chuva'] = df['tempo_chuva'].clip(0, 4)                   
df['precipitacao_chuva'] = df['precipitacao_chuva'].clip(lower=0)    
df['precipitacao_acumulada'] = df['precipitacao_acumulada'].clip(lower=0)  
df['declive_graus'] = df['declive_graus'].clip(lower=0)             

# alvo
df['historico_alagamento'] = df['historico_alagamento'].astype(int)

# categóricas / texto (corrigido: use .str)
df['intensidade_chuva'] = df['intensidade_chuva'].astype(str).str.lower().str.strip() 

df['bairro'] = (
    df['bairro']
      .astype(str)
      .str.lower()
      .str.replace(r'\s+', '', regex=True)
)

# datas
df['data_hora'] = pd.to_datetime(df['data_hora'].astype(str).str.replace('T', ' '), errors='coerce')

# feature auxiliar (opcional, mas útil no treino)
df['chuva_media_h'] = df['precipitacao_acumulada'] / np.maximum(df['tempo_chuva'], 1) 

# checagem simples
inconsistentes = df[(df['tempo_chuva'] == 0) & (df['precipitacao_chuva'] > 0)]
print(f"{len(inconsistentes)} registros inconsistentes encontrados.")

# salva processado
df.to_csv('../data/processed/dados_processados.csv', index=False)
print("OK: ../data/processed/dados_processados.csv salvo com declive_graus.")

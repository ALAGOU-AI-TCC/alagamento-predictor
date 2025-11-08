# -*- coding: utf-8 -*-
import os
import pandas as pd

INPUT_CSV  = '../data/raw/dados_comdeclive.csv'
OUTPUT_DIR = '../data/processed'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'dados_processados_01-11-2025.csv')

NA_MARKERS = ['', ' ', '  ', 'NULL', 'null', 'NaN', 'nan', 'None', 'N/A']

# 1) Ler CSV tratando nulos e espaços após vírgula
df = pd.read_csv(
    INPUT_CSV,
    na_values=NA_MARKERS,
    keep_default_na=True,
    skipinitialspace=True,
    encoding='utf-8',
    on_bad_lines='warn'
)
print(f"[INFO] Linhas lidas: {len(df)}, Colunas: {len(df.columns)}")

# 2) Trim em todas as strings + strings vazias -> NaN
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]):
        df[col] = df[col].astype(str).str.strip()
        df[col].replace('', pd.NA, inplace=True)

# 3) Remove colunas indesejadas
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# >>> REMOVER velocidade_vento do dataset <<<
if 'velocidade_vento' in df.columns:
    df = df.drop(columns=['velocidade_vento'])
    print("[INFO] Coluna 'velocidade_vento' removida do dataset.")

# 4) Tratar vírgula decimal nas numéricas ANTES do to_numeric
# (sem 'velocidade_vento')
numericas = [
    'temperatura', 'umidade', 'pressao',
    'precipitacao_chuva', 'tempo_chuva', 'ponto_orvalho',
    'precipitacao_acumulada'
]
for col in numericas:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

# 5) Converter numéricas
for col in numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 6) Normalizações específicas
if 'historico_alagamento' in df.columns:
    df['historico_alagamento'] = pd.to_numeric(df['historico_alagamento'], errors='coerce').astype('Int64')

if 'intensidade_chuva' in df.columns:
    df['intensidade_chuva'] = (
        df['intensidade_chuva']
        .astype('string')
        .str.replace(r'\s+', ' ', regex=True)
        .str.lower()
    )

if 'bairro' in df.columns:
    df['bairro'] = (
        df['bairro']
        .astype('string')
        .str.lower()
        .str.replace(r'\s+', '', regex=True)
    )

if 'data_hora' in df.columns:
    df['data_hora'] = pd.to_datetime(
        df['data_hora'].astype(str).str.replace('T', ' ', regex=False),
        errors='coerce',
        infer_datetime_format=True
    )

# 7) Dropna apenas nas essenciais (sem velocidade_vento)
subset = [
    'temperatura', 'umidade', 'pressao',
    'precipitacao_chuva', 'tempo_chuva', 'intensidade_chuva',
    'ponto_orvalho', 'historico_alagamento', 'precipitacao_acumulada'
]
faltantes_cols = [c for c in subset if c not in df.columns]
if faltantes_cols:
    raise SystemExit(f"[ERRO] Colunas obrigatórias ausentes no CSV: {faltantes_cols}")

print("\n[DIAG] NaNs por coluna essencial antes do dropna:")
print(df[subset].isna().sum().sort_values(ascending=False))

before = len(df)
df = df.dropna(subset=subset)
after = len(df)
print(f"[INFO] Linhas antes do dropna: {before} | depois: {after}")

if after == 0:
    raise SystemExit("[ABORTADO] Todas as linhas foram removidas. Revise o CSV de origem.")

# 8) Checagem simples
if {'tempo_chuva','precipitacao_chuva'}.issubset(df.columns):
    inconsistentes = df[(df['tempo_chuva'] == 0) & (df['precipitacao_chuva'] > 0)]
    print(f"[INFO] Inconsistentes (tempo_chuva=0 & precipitação>0): {len(inconsistentes)}")

# 9) Salvar
os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[OK] Arquivo salvo em: {OUTPUT_CSV} com {len(df)} linhas.")

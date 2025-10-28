# -*- coding: utf-8 -*-
# Regressão Logística (com declive_graus, slope_plano, declive_bin)
# - StandardScaler em numéricos
# - OneHotEncoder em categóricos
# - clipping de chuvas (>=0) e declive (>=0)
# - cria declive_bin e slope_plano (se necessário)
# - CV estratificado (k=10) — sem CV por mês

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_predict
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    make_scorer, fbeta_score
)

# =======================
# 0) Paths e setup
# =======================
DATA_PATH = '../data/processed/dados_processados.csv'
REPORTS_DIR = Path('../reports'); REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR  = Path('../models');  MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =======================
# 1) Carrega dados
# =======================
df = pd.read_csv(DATA_PATH)

# =======================
# 2) Garantias/derivações (iguais ao XGB)
# =======================
# Clippings básicos
if 'tempo_chuva' in df.columns:
    df['tempo_chuva'] = df['tempo_chuva'].clip(0, 4).astype('int8')

for col in ['precipitacao_chuva', 'precipitacao_acumulada']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').clip(lower=0)

if 'declive_graus' in df.columns:
    df['declive_graus'] = pd.to_numeric(df['declive_graus'], errors='coerce').clip(lower=0)

# Remove qualquer resquício de 'chuva_media_h' se existir
if 'chuva_media_h' in df.columns:
    df = df.drop(columns=['chuva_media_h'])

# Bins de declive (categoria) e flag de plano
DECLIVE_BINS = [0, 2, 4, 6, 8, 10, 15, 60]
DECLIVE_LABELS = [f'{DECLIVE_BINS[i]}–{DECLIVE_BINS[i+1]}°' for i in range(len(DECLIVE_BINS)-1)]

if 'declive_graus' in df.columns and 'declive_bin' not in df.columns:
    df['declive_bin'] = pd.cut(
        df['declive_graus'],
        bins=DECLIVE_BINS, labels=DECLIVE_LABELS, include_lowest=True, right=False
    )

if 'declive_graus' in df.columns and 'slope_plano' not in df.columns:
    df['slope_plano'] = (df['declive_graus'] < 2).astype('int8')

# Normaliza intensidade_chuva (string) se existir
if 'intensidade_chuva' in df.columns:
    df['intensidade_chuva'] = df['intensidade_chuva'].astype(str).str.strip().str.lower()

# =======================
# 3) Define X / y
# =======================
drop_cols = [
    "historico_alagamento", "data_hora", "latitude",
    "longitude", "bairro", "precipitacao_diaria", "velocidade_vento"
]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
y = df["historico_alagamento"].astype(int)

# Lista de features (coerente com o XGB)
numerical_features = [c for c in [
    'temperatura', 'umidade', 'pressao',
    'precipitacao_chuva', 'ponto_orvalho',
    'precipitacao_acumulada', 'solo_elevacao',
    'tempo_chuva',            # tratada como numérica aqui
    'declive_graus',
    'slope_plano'
] if c in X.columns]

categorical_features = [c for c in ['intensidade_chuva', 'declive_bin'] if c in X.columns]

# =======================
# 4) Pré-processador + Modelo
# =======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Classe desbalanceada → usar class_weight="balanced"
logreg = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    solver='lbfgs'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', logreg)
])

# =======================
# 5) Split holdout
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# =======================
# 6) Treina
# =======================
pipeline.fit(X_train, y_train)

# =======================
# 7) Avaliação holdout
# =======================
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\nRelatório de Classificação (Holdout - thr=0.50):")
relatorio = classification_report(y_test, y_pred, digits=3)
print(relatorio)

with open(REPORTS_DIR / 'logreg_classification_report.txt', 'w', encoding='utf-8') as f:
    f.write(relatorio)

cm = confusion_matrix(y_test, y_pred)
labels = ['Não Alagou', 'Alagou']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Previsto'); plt.ylabel('Real')
plt.title('Matriz de Confusão - Regressão Logística (Holdout, thr=0.50)')
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'logreg_matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC - Regressão Logística (Holdout)')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(); plt.grid(True)
plt.savefig(REPORTS_DIR / 'logreg_curva_roc.png', dpi=300, bbox_inches='tight')
plt.close()

# Relatório tabular
rep_dict = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(rep_dict).transpose().round(3).to_csv(REPORTS_DIR / 'logreg_classification_report_table.csv')

# =======================
# 8) Cross-Validation (StratifiedKFold k=10) — sem CV por mês
# =======================
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'f2': make_scorer(fbeta_score, beta=2),
    'roc_auc': 'roc_auc'
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_results = cross_validate(
    pipeline, X, y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

summary_rows = []
for m in ['accuracy','precision','recall','f1','f2','roc_auc']:
    vals = cv_results[f'test_{m}']
    summary_rows.append({'metric': m, 'mean': np.mean(vals), 'std': np.std(vals)})
cv_df = pd.DataFrame(summary_rows).round(4)
print("\nCV - Regressão Logística (StratifiedKFold k=10):")
print(cv_df)
cv_df.to_csv(REPORTS_DIR / 'logreg_cv_metrics.csv', index=False)

# =======================
# 9) OOF (com o mesmo StratifiedKFold)
# =======================
try:
    oof_pred  = cross_val_predict(pipeline, X, y, cv=cv, method='predict', n_jobs=-1)
    oof_proba = cross_val_predict(pipeline, X, y, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]

    rep_oof = classification_report(y, oof_pred, digits=3)
    with open(REPORTS_DIR / 'logreg_classification_report_oof.txt', 'w', encoding='utf-8') as f:
        f.write(rep_oof)

    cm_oof = confusion_matrix(y, oof_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_oof, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Previsto'); plt.ylabel('Real')
    plt.title('Matriz de Confusão (OOF - Regressão Logística)')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'logreg_matriz_confusao_oof.png', dpi=300, bbox_inches='tight')
    plt.close()

    fpr_oof, tpr_oof, _ = roc_curve(y, oof_proba)
    roc_auc_oof = auc(fpr_oof, tpr_oof)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_oof, tpr_oof, label=f'AUC (OOF) = {roc_auc_oof:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Curva ROC (OOF - Regressão Logística)')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(True)
    plt.savefig(REPORTS_DIR / 'logreg_curva_roc_oof.png', dpi=300, bbox_inches='tight')
    plt.close()
except Exception as e:
    print("Aviso: OOF não gerado:", e)

# =======================
# 10) Exporta modelo
# =======================
joblib.dump(pipeline, MODELS_DIR / 'modelo_regressao_logistica.pkl')

# =======================
# 11) Versões
# =======================
import sklearn
print("\nVersões:")
print("sklearn", sklearn.__version__)
print("pandas", pd.__version__)
print("numpy", np.__version__)
print("OK - Treino RL finalizado e artefatos em ../reports e ../models")

# train_xgb_alagouai_databricks.py
# -*- coding: utf-8 -*-
"""
Treino XGBoost (Databricks/Local) SEM solo_elevacao e SEM declive_graus como feature.
- Árvores não precisam de scaler
- SimpleImputer (mediana) p/ numéricas
- OneHotEncoder(handle_unknown='ignore') p/ categóricas
- Normaliza/consistência de intensidade_chuva (ex.: 0 mm e 0 min => "sem chuva")
- Clipping leve nas caudas de precipitação (robustez)
- Salva wrapper PreprocessXGB com feature_order (compatível com sua API)
"""

import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    average_precision_score
)

from xgboost import XGBClassifier

# =========================
# Paths e config
# =========================
DATA_PATH = '../data/processed/dados_processados_01-11-2025.csv'
MODEL_OUT = '../models/modelo_xgboost_novo.pkl'
REPORTS_DIR = Path('../reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'historico_alagamento'
DROP_COLS = ["data_hora", "latitude", "longitude", "bairro", "precipitacao_diaria", "chuva_media_h",
             # Removemos também quaisquer colunas de apoio que você não queira
             "solo_elevacao", "declive_graus"  # <- NÃO entram no treinamento
]

# Se True, quando precipitação==0 e tempo_chuva==0 força intensidade="sem chuva"
ENFORCE_INTENSITY_CONSISTENCY = True

# =========================
# Wrapper p/ API
# =========================
class PreprocessXGB:
    def __init__(self, preprocessor, xgb, feature_order):
        self.preprocessor = preprocessor
        self.xgb = xgb
        self.feature_order = feature_order  # colunas esperadas ANTES do transform

    def _to_df(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_order]
        return pd.DataFrame(X, columns=self.feature_order)

    def predict_proba(self, X):
        Xdf = self._to_df(X)
        Xt = self.preprocessor.transform(Xdf)
        return self.xgb.predict_proba(Xt)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# 1) Carregar dados
# =========================
df = pd.read_csv(DATA_PATH)

# 2) Selecionar features e alvo
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET])
y = df[TARGET].astype(int)

# 3) Tratar categórica intensidade_chuva
if "intensidade_chuva" in X.columns:
    X["intensidade_chuva"] = X["intensidade_chuva"].astype(str).str.strip().str.lower()
    mapa_int = {
        "sem chuva": "sem chuva",
        "chuva fraca": "chuva fraca",
        "chuva moderada": "chuva moderada",
        "chuva forte": "chuva forte",
        "céu limpo": "sem chuva",
        "ceu limpo": "sem chuva",
        "nublado": "sem chuva",
        "trovoada": "trovoada",
        "trovoada com chuva": "trovoada com chuva",
        "trovoada com chuva forte": "trovoada com chuva forte",
    }

# 3.1) Consistência com métricas de chuva (recomendado)
if ENFORCE_INTENSITY_CONSISTENCY and {"precipitacao_chuva", "tempo_chuva"}.issubset(X.columns):
    mask_sem = (X["precipitacao_chuva"].fillna(0) == 0) & (X["tempo_chuva"].fillna(0) == 0)
    X.loc[mask_sem, "intensidade_chuva"] = "sem chuva"

# 4) Garantir tipos numéricos
for c in ["precipitacao_chuva","precipitacao_acumulada","temperatura","umidade",
          "pressao","ponto_orvalho","tempo_chuva"]:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

# 5) Clipping leve de caudas (robustez)
def clip_col(df_, col, q=0.99, nonneg=True):
    if col in df_.columns:
        up = df_[col].quantile(q)
        df_[col] = df_[col].clip(lower=0 if nonneg else None, upper=up)

for col in ["precipitacao_chuva","precipitacao_acumulada"]:
    clip_col(X, col, q=0.99, nonneg=True)

# 6) Listas de features (SEM declive/solo)
numeric = [c for c in [
    'temperatura','umidade','pressao','precipitacao_chuva','ponto_orvalho',
    'tempo_chuva','precipitacao_acumulada'
] if c in X.columns]
categorical = [c for c in ['intensidade_chuva'] if c in X.columns]

feature_order = numeric + categorical  # usado pelo wrapper

# 7) Split estratificado
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 8) Preprocessador (compat: não usar sparse_output para manter retrocompat)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ],
    remainder='drop'
)

# Fit + transform
preprocessor.fit(X_train)
Xtr = preprocessor.transform(X_train)
Xva = preprocessor.transform(X_val)

# 9) Modelo (sem constraints; declive/solo não entram)
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = neg / max(pos, 1)

xgb = XGBClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',
    scale_pos_weight=scale_pos_weight,
    max_delta_step=1
)

# 10) Treinar — tentar early stopping; se versão não suportar, segue sem
supports_es = 'early_stopping_rounds' in inspect.signature(xgb.fit).parameters
try:
    if supports_es:
        xgb.fit(Xtr, y_train, eval_set=[(Xva, y_val)], early_stopping_rounds=50, verbose=False)
    else:
        xgb.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
except TypeError:
    xgb.fit(Xtr, y_train)

# 11) Métricas rápidas
proba_val = xgb.predict_proba(Xva)[:, 1]
pred_val = (proba_val >= 0.5).astype(int)

cm = confusion_matrix(y_val, pred_val)
print("Matriz de Confusão:\n", cm)

print("\nRelatório de Classificação (limiar 0.5):")
print(classification_report(y_val, pred_val, digits=3))

fpr, tpr, _ = roc_curve(y_val, proba_val)
roc_auc = auc(fpr, tpr)
ap = average_precision_score(y_val, proba_val)
print(f"AUC-ROC: {roc_auc:.3f}")
print(f"PR-AUC : {ap:.3f}")

# 12) Salvar modelo (wrapper compatível com a API)
model_wrapper = PreprocessXGB(preprocessor=preprocessor, xgb=xgb, feature_order=feature_order)
joblib.dump(model_wrapper, MODEL_OUT)
print(f"\n[OK] Modelo salvo em: {MODEL_OUT}")

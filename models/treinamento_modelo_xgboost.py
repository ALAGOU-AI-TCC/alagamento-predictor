# -*- coding: utf-8 -*-
# Treino XGBoost (com solo_elevacao + declive_graus)
# - n_jobs=-1
# - numéricos sem StandardScaler
# - clipping de chuvas (>=0) e declive (>=0)
# - features extras: chuva_media_h, declive_bin, slope_plano
# - avaliação de thresholds (F2)
# - cross-validation em blocos temporais (por mês)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, fbeta_score
)
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

# =======================
# 0) Setup opcional MLflow
# =======================
try:
    import mlflow
    mlflow.autolog(disable=True)
except Exception:
    pass

# =======================
# 1) Carrega dados
# =======================
df = pd.read_csv('../data/processed/dados_processados.csv')

# Garantias básicas
df['tempo_chuva'] = df['tempo_chuva'].clip(0, 4).astype('int8')
df['precipitacao_chuva'] = df['precipitacao_chuva'].clip(lower=0)
df['precipitacao_acumulada'] = df['precipitacao_acumulada'].clip(lower=0)
if 'declive_graus' in df.columns:
    df['declive_graus'] = pd.to_numeric(df['declive_graus'], errors='coerce').clip(lower=0)

# Intensidade média (concentração da chuva)
df['chuva_media_h'] = df['precipitacao_acumulada'] / np.maximum(df['tempo_chuva'], 1)

# Bins de declive (categoria) e flag de plano
declive_bins = [0,2,4,6,8,10,15,60]  # colapsa bins altos (evita n pequeno)
declive_labels = [f'{declive_bins[i]}–{declive_bins[i+1]}°' for i in range(len(declive_bins)-1)]
df['declive_bin'] = pd.cut(
    df['declive_graus'],
    bins=declive_bins, labels=declive_labels, include_lowest=True, right=False
)
df['slope_plano'] = (df['declive_graus'] < 2).astype('int8')

# =======================
# 2) Define X / y
# =======================
X = df.drop(columns=[
    "historico_alagamento", "data_hora", "latitude",
    "longitude", "bairro", "precipitacao_diaria", "velocidade_vento"
], errors='ignore')
y = df["historico_alagamento"].astype(int)

numerical_features = [
    'temperatura', 'umidade', 'pressao',
    'precipitacao_chuva', 'ponto_orvalho',
    'precipitacao_acumulada', 'solo_elevacao',
    'tempo_chuva', 'chuva_media_h',
    'declive_graus',          # << NOVO
    'slope_plano'             # << NOVO (0/1)
]
# mantenho 'intensidade_chuva' e somo o bin categórico do declive
categorical_features = [c for c in ['intensidade_chuva', 'declive_bin'] if c in X.columns]

# =======================
# 3) Pré-processador
# =======================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', [c for c in numerical_features if c in X.columns]),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# =======================
# 4) Split holdout
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# =======================
# 5) Lida com desbalanceamento
# =======================
pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
scale_pos_weight = neg / max(pos, 1)

# =======================
# 6) Modelo XGBoost
# =======================
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    gamma=0.0,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist',
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb)
])

# =======================
# 7) Treino (holdout)
# =======================
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# =======================
# 8) Avaliação holdout
# =======================
print("\nRelatório de Classificação (Holdout - 0.50):")
relatorio = classification_report(y_test, y_pred, digits=3)
print(relatorio)

Path('../reports').mkdir(parents=True, exist_ok=True)
Path('../models').mkdir(parents=True, exist_ok=True)

with open('../reports/xgb_classification_report.txt', 'w') as f:
    f.write(relatorio)

cm = confusion_matrix(y_test, y_pred)
labels = ['Não Alagou', 'Alagou']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Previsto'); plt.ylabel('Real')
plt.title('Matriz de Confusão - XGBoost (Holdout, thr=0.50)')
plt.tight_layout()
plt.savefig('../reports/xgb_matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC - XGBoost (Holdout)')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(); plt.grid(True)
plt.savefig('../reports/xgb_curva_roc.png', dpi=300, bbox_inches='tight')
plt.close()

# =======================
# 9A) Threshold "trifásico" por declive (plano < médio < íngreme)
#     - plano:   declive < 2°      -> threshold tp (mais baixo)
#     - médio:   2°–angulo_corte   -> threshold tm (intermediário)
#     - íngreme: >= angulo_corte   -> threshold ti (mais alto)
#     Restrição: tp <= tm <= ti
# =======================
import numpy as np, json
from sklearn.metrics import fbeta_score, precision_score, recall_score, f1_score, accuracy_score

# Declive alinhado ao holdout
if 'declive_graus' in X_test.columns:
    declive_holdout = X_test['declive_graus'].to_numpy()
else:
    declive_holdout = df.loc[X_test.index, 'declive_graus'].to_numpy()

def melhor_threshold_trifasico(y_true, y_proba, declive,
                               angulos=[4,6,8,10],
                               grid=np.arange(0.20,0.81,0.02)):
    best = {'F2':0.0, 'tp':0.30, 'tm':0.30, 'ti':0.30, 'angulo_corte':6}
    plano_mask = (declive < 2)
    for ac in angulos:  # ângulo que separa médio de íngreme
        medio_mask   = (declive >= 2) & (declive < ac)
        ingreme_mask = (declive >= ac)
        for tp in grid:
            for tm in grid:
                if tm < tp:  # garante tp <= tm
                    continue
                for ti in grid:
                    if ti < tm:  # garante tm <= ti
                        continue
                    pred = np.empty_like(y_true)
                    pred[plano_mask]   = (y_proba[plano_mask]   >= tp).astype(int)
                    pred[medio_mask]   = (y_proba[medio_mask]   >= tm).astype(int)
                    pred[ingreme_mask] = (y_proba[ingreme_mask] >= ti).astype(int)
                    f2 = fbeta_score(y_true, pred, beta=2)
                    if f2 > best['F2']:
                        best = {'F2':float(f2), 'tp':float(tp), 'tm':float(tm), 'ti':float(ti), 'angulo_corte':int(ac)}
    return best

best = melhor_threshold_trifasico(y_test.to_numpy(), y_proba, declive_holdout)
tp, tm, ti, ac = best['tp'], best['tm'], best['ti'], best['angulo_corte']
print(f"\nThreshold trifásico ótimo (F2={best['F2']:.4f}): "
      f"tp={tp:.2f} (plano<2°) | tm={tm:.2f} (2–{ac}°) | ti={ti:.2f} (≥{ac}°)")

# Aplica os cortes trifásicos
plano_mask   = (declive_holdout < 2)
medio_mask   = (declive_holdout >= 2) & (declive_holdout < ac)
ingreme_mask = (declive_holdout >= ac)

y_pred_tri = np.empty_like(y_test)
y_pred_tri[plano_mask]   = (y_proba[plano_mask]   >= tp).astype(int)
y_pred_tri[medio_mask]   = (y_proba[medio_mask]   >= tm).astype(int)
y_pred_tri[ingreme_mask] = (y_proba[ingreme_mask] >= ti).astype(int)

met_tri = {
    'accuracy':  accuracy_score(y_test, y_pred_tri),
    'precision': precision_score(y_test, y_pred_tri),
    'recall':    recall_score(y_test, y_pred_tri),
    'f1':        f1_score(y_test, y_pred_tri),
    'F2':        fbeta_score(y_test, y_pred_tri, beta=2),
    'tp_plano':  tp, 'tm_medio': tm, 'ti_ingreme': ti, 'angulo_corte': ac,
    'pct_plano(%)':   float(100*plano_mask.mean()),
    'pct_medio(%)':   float(100*medio_mask.mean()),
    'pct_ingreme(%)': float(100*ingreme_mask.mean())
}
print("\nMétricas (threshold trifásico no holdout):")
for k,v in met_tri.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Salva métricas e config p/ API
pd.DataFrame([met_tri]).to_csv('../reports/threshold_trifasico_metrics_holdout.csv', index=False)
with open('../models/threshold_trifasico.json', 'w') as f:
    json.dump({
        'tp_plano': tp, 'tm_medio': tm, 'ti_ingreme': ti, 'angulo_corte': ac,
        'regras': ['declive<2°→tp_plano', f'2°–{ac}°→tm_medio', f'≥{ac}°→ti_ingreme'],
        'obs': 'tp<=tm<=ti (menos positivos em ruas íngremes)'
    }, f, indent=2)

# Matriz de confusão do trifásico
cm_tri = confusion_matrix(y_test, y_pred_tri)
plt.figure(figsize=(6,4))
sns.heatmap(cm_tri, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Não Alagou','Alagou'], yticklabels=['Não Alagou','Alagou'])
plt.xlabel('Previsto'); plt.ylabel('Real')
plt.title(f'Matriz de Confusão • Threshold trifásico (tp={tp:.2f}, tm={tm:.2f}, ti={ti:.2f}, corte={ac}°)')
plt.tight_layout()
plt.savefig('../reports/xgb_matriz_confusao_threshold_trifasico.png', dpi=300, bbox_inches='tight')
plt.close()

# >>> cole logo após montar y_pred_tri (no fim do bloco 9A) <<<

from sklearn.metrics import classification_report

print("\nRelatório de Classificação (Holdout • threshold trifásico):")
relatorio_tri = classification_report(y_test, y_pred_tri, digits=3)
print(relatorio_tri)

with open('../reports/xgb_classification_report_holdout_trifasico.txt', 'w') as f:
    f.write(relatorio_tri)


# =======================
# 9) Threshold sweep (F2) no holdout
# =======================
ths = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
thr_rows = []
for thr in ths:
    y_thr = (y_proba >= thr).astype(int)
    f2 = fbeta_score(y_test, y_thr, beta=2)
    thr_rows.append({'threshold': thr, 'F2': f2})
thr_df = pd.DataFrame(thr_rows)
print("\nF2 por threshold (holdout):")
print(thr_df)
thr_df.to_csv('../reports/xgb_thresholds_f2_holdout.csv', index=False)

# =======================
# 10) Cross-Validation por mês (GroupKFold)
# =======================
if 'data_hora' in df.columns:
    datas = pd.to_datetime(df['data_hora'], errors='coerce')
    groups = datas.dt.to_period('M').astype(str).fillna('NA')
else:
    groups = pd.Series(['G'] * len(df))

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'f2': 'f1',  # placeholder
    'roc_auc': 'roc_auc'
}

gkf = GroupKFold(n_splits=5)
cv_results = cross_validate(
    pipeline, X, y,
    cv=gkf.split(X, y, groups=groups),
    scoring=scoring,
    n_jobs=-1, return_train_score=False
)

rows = []
for m in ['accuracy','precision','recall','f1','roc_auc']:
    vals = cv_results[f'test_{m}']
    rows.append({'metric': m, 'mean': np.mean(vals), 'std': np.std(vals)})
cv_xgb_df = pd.DataFrame(rows).round(4)
print("\nCV (GroupKFold por mês):")
print(cv_xgb_df)
cv_xgb_df.to_csv('../reports/xgb_cv_metrics_groupkfold.csv', index=False)

# OOF
y_pred_oof  = cross_val_predict(
    pipeline, X, y, cv=gkf.split(X, y, groups=groups), method='predict', n_jobs=-1
)
y_proba_oof = cross_val_predict(
    pipeline, X, y, cv=gkf.split(X, y, groups=groups), method='predict_proba', n_jobs=-1
)[:, 1]

relatorio_oof = classification_report(y, y_pred_oof, digits=3)
print("\nRelatório de Classificação OOF (GroupKFold):")
print(relatorio_oof)
with open('../reports/xgb_classification_report_oof_groupkfold.txt', 'w') as f:
    f.write(relatorio_oof)

cm_oof = confusion_matrix(y, y_pred_oof)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_oof, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Previsto'); plt.ylabel('Real')
plt.title('Matriz de Confusão (OOF - GroupKFold por mês)')
plt.tight_layout()
plt.savefig('../reports/xgb_matriz_confusao_oof_groupkfold.png', dpi=300, bbox_inches='tight')
plt.close()

fpr_oof, tpr_oof, _ = roc_curve(y, y_proba_oof)
roc_auc_oof = auc(fpr_oof, tpr_oof)
plt.figure(figsize=(6, 6))
plt.plot(fpr_oof, tpr_oof, label=f'AUC (OOF) = {roc_auc_oof:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC (OOF - GroupKFold por mês)')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.legend(); plt.grid(True)
plt.savefig('../reports/xgb_curva_roc_oof_groupkfold.png', dpi=300, bbox_inches='tight')
plt.close()

rows = []
for thr in ths:
    y_thr = (y_proba_oof >= thr).astype(int)
    f2 = fbeta_score(y, y_thr, beta=2)
    rows.append({'threshold': thr, 'F2_oof': f2})
thr_oof_df = pd.DataFrame(rows)
print("\nF2 por threshold (OOF - GroupKFold por mês):")
print(thr_oof_df)
thr_oof_df.to_csv('../reports/xgb_thresholds_f2_oof_groupkfold.csv', index=False)

# =======================
# 11) Importâncias
# =======================
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
importancias = pipeline.named_steps['classifier'].feature_importances_

fi = (pd.DataFrame({'feature': feature_names, 'importance': importancias})
      .sort_values('importance', ascending=False))
fi.to_csv('../reports/xgb_feature_importances.csv', index=False)

plt.figure(figsize=(8, 6))
topn = min(25, len(fi))
sns.barplot(data=fi.head(topn), x='importance', y='feature')
plt.title('Top Importâncias de Features - XGBoost')
plt.tight_layout()
plt.savefig('../reports/xgb_feature_importances.png', dpi=300, bbox_inches='tight')
plt.close()

# =======================
# 12) Exporta modelo
# =======================
joblib.dump(pipeline, '../models/modelo_xgboost.pkl')

# =======================
# 13) Versões
# =======================
import sklearn, xgboost, pandas
print("\nVersões:")
print("sklearn", sklearn.__version__)
print("xgboost", xgboost.__version__)
print("pandas", pandas.__version__)
print("numpy", np.__version__)
print("OK - Treino finalizado e artefatos em ../reports e ../models")

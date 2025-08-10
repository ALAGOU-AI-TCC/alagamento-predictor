!pip install xgboost
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
import numpy as np

df = pd.read_csv('../data/processed/dados_processados.csv')

X = df.drop(columns=[
    "historico_alagamento", "data_hora", "latitude",
    "longitude", "bairro", "solo", "precipitacao_diaria", "velocidade_vento"
])
y = df["historico_alagamento"]

numerical_features = [
    'temperatura', 'umidade', 'pressao',
    'precipitacao_chuva', 'ponto_orvalho',
    'tempo_chuva', 'precipitacao_acumulada'
]
categorical_features = ['intensidade_chuva']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)


pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / max(pos, 1)

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
    scale_pos_weight=scale_pos_weight
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb)
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1] 


def classificar_risco(p):
    if p <= 0.30:
        return 'baixo'
    elif p <= 0.70:
        return 'medio'
    else:
        return 'alto'

risco = [classificar_risco(p) for p in y_proba]

df_resultados = pd.DataFrame({
    'prob_alagar': y_proba,
    'classe_binaria_prevista': y_pred,
    'risco': risco
})
df_resultados.to_csv('../reports/resultados_xgb_risco.csv', index=False)


cm = confusion_matrix(y_test, y_pred)
labels = ['Não Alagou', 'Alagou']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão - XGBoost')
plt.tight_layout()
plt.savefig('../reports/xgb_matriz_confusao.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n Relatório de Classificação (XGBoost):")
relatorio = classification_report(y_test, y_pred, digits=3)
print(relatorio)

with open('../reports/xgb_classification_report.txt', 'w') as f:
    f.write(relatorio)


fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC - XGBoost')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend()
plt.grid(True)
plt.savefig('../reports/xgb_curva_roc.png', dpi=300, bbox_inches='tight')
plt.show()


feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
importancias = pipeline.named_steps['classifier'].feature_importances_

fi = (pd.DataFrame({
    'feature': feature_names,
    'importance': importancias
})
      .sort_values('importance', ascending=False))

fi.to_csv('../reports/xgb_feature_importances.csv', index=False)

plt.figure(figsize=(8, 6))
topn = 25 
sns.barplot(data=fi.head(topn), x='importance', y='feature')
plt.title('Top Importâncias de Features - XGBoost')
plt.tight_layout()
plt.savefig('../reports/xgb_feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

joblib.dump(pipeline, '../models/modelo_xgboost.pkl')

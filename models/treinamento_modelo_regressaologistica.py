import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

df = pd.read_csv('../data/processed/dados_processados.csv')

X = df.drop(columns=["historico_alagamento", "data_hora", "latitude",
                     "longitude", "bairro", "solo", "precipitacao_diaria", "velocidade_vento"])
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

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced')) 
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1] 



cm = confusion_matrix(y_test, y_pred)
labels = ['Não Alagou', 'Alagou']

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.savefig('../reports/matriz_confusao.png')
plt.show()

print("\n Relatório de Classificação:")
relatorio = classification_report(y_test, y_pred)
print(relatorio)



report_dict = classification_report(y_test, y_pred, output_dict=True)

df_report = pd.DataFrame(report_dict).transpose().round(2)

df_report.to_csv('../reports/classification_report.csv')


joblib.dump(pipeline, '../models/modelo_regressao_logistica.pkl')

with open('../reports/classification_report.txt', 'w') as f:
    f.write(relatorio)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Curva ROC')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend()
plt.grid(True)
plt.savefig('../reports/curva_roc.png')
plt.show()

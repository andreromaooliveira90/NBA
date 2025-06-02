import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CÓDIGO PARA TREINAR E AVALIAR MODELOS (PREVISÃO DE PLAYOFFS) ---

# Instrução: Execute este script no seu notebook APÓS a célula onde
# os dataframes X_train, y_train, X_test, y_test foram criados
# usando o script 'prepare_data_playoffs.py'.

# Verificar se os dataframes de treino/teste existem
required_vars = ["X_train", "y_train", "X_test", "y_test"]
if not all(var in locals() for var in required_vars):
    print("Erro: Dataframes X_train, y_train, X_test, y_test não encontrados.")
    print("Certifique-se de ter executado o script 'prepare_data_playoffs.py' primeiro.")
    # Adicione lógica para carregar os CSVs se você os salvou
    # Exemplo:
    # try:
    #     X_train = pd.read_csv("X_train_playoffs.csv")
    #     y_train = pd.read_csv("y_train_playoffs.csv").squeeze() # .squeeze() para converter em Series
    #     X_test = pd.read_csv("X_test_playoffs.csv")
    #     y_test = pd.read_csv("y_test_playoffs.csv").squeeze()
    #     print("Dados carregados dos arquivos CSV.")
    # except FileNotFoundError:
    #     print("Arquivos CSV não encontrados. Execute o script de preparação.")
    #     exit()
    exit()

print("Dados de treino e teste encontrados. Iniciando treinamento e avaliação...")

# --- Modelo 1: RandomForestClassifier ---
print("\n--- Treinando RandomForestClassifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=\'balanced\') # Usar class_weight para dados desbalanceados
rf_model.fit(X_train, y_train)

print("\nAvaliando RandomForestClassifier no conjunto de teste...")
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1] # Probabilidades para classe 1 (Playoffs)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
report_rf = classification_report(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print(f"Acurácia (RandomForest): {accuracy_rf:.4f}")
print(f"ROC AUC (RandomForest): {roc_auc_rf:.4f}")
print("Relatório de Classificação (RandomForest):\n", report_rf)

# Plotar Matriz de Confusão para RandomForest
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt=\'d\', cmap=\'Blues
', xticklabels=["Não-Playoff", "Playoff"], yticklabels=["Não-Playoff", "Playoff"])
plt.title("Matriz de Confusão - RandomForest")
plt.ylabel("Real")
plt.xlabel("Previsto")
plt.show()

# --- Modelo 2: LGBMClassifier ---
print("\n--- Treinando LGBMClassifier ---")
lgbm_model = lgb.LGBMClassifier(random_state=42, class_weight=\'balanced\') # Usar class_weight
lgbm_model.fit(X_train, y_train)

print("\nAvaliando LGBMClassifier no conjunto de teste...")
y_pred_lgbm = lgbm_model.predict(X_test)
y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
roc_auc_lgbm = roc_auc_score(y_test, y_prob_lgbm)
report_lgbm = classification_report(y_test, y_pred_lgbm)
cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)

print(f"Acurácia (LGBM): {accuracy_lgbm:.4f}")
print(f"ROC AUC (LGBM): {roc_auc_lgbm:.4f}")
print("Relatório de Classificação (LGBM):\n", report_lgbm)

# Plotar Matriz de Confusão para LGBM
plt.figure(figsize=(6, 4))
sns.heatmap(cm_lgbm, annot=True, fmt=\'d\', cmap=\'Greens\', xticklabels=["Não-Playoff", "Playoff"], yticklabels=["Não-Playoff", "Playoff"])
plt.title("Matriz de Confusão - LGBM")
plt.ylabel("Real")
plt.xlabel("Previsto")
plt.show()

# --- Feature Importance (Exemplo com RandomForest) ---
print("\n--- Importância das Features (RandomForest) ---")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    \'Feature
': X_train.columns,
    \'Importance
': importances
}).sort_values(by=\'Importance\', ascending=False)

print(feature_importance_df.head(10)) # Mostrar as 10 mais importantes

plt.figure(figsize=(10, 6))
sns.barplot(x=\'Importance\', y=\'Feature\', data=feature_importance_df.head(10))
plt.title("Top 10 Features Mais Importantes - RandomForest")
plt.tight_layout()
plt.show()

# --- Próximos Passos ---
# 1. Analisar os resultados: Qual modelo performou melhor? Quais métricas são mais importantes?
# 2. Otimização de Hiperparâmetros: Usar GridSearchCV ou RandomizedSearchCV para encontrar os melhores parâmetros para os modelos.
# 3. Engenharia de Features: Criar novas features pode melhorar a performance.
# 4. Avançar para o Modelo 2 (Previsão do Campeão): Usar a mesma lógica, mas com o dataframe `df_team_in_playoff_grouped` e o alvo sendo a coluna `Champion Team`.

print("\nTreinamento e avaliação dos modelos concluídos.")


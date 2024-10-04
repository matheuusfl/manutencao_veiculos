import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Aqui estão os dados dos veiculos
df_veiculos = pd.read_csv('dados_veiculos.csv')

# Definindo as variáveis independentes e dependentes
X = df_veiculos[['Quilometragem', 'Idade', 'Tempo_Ultima_Manutencao', 'Temperatura_Motor']]
y = df_veiculos['Falha']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# Fazendo previsões
predictions = model_rf.predict(X_test)

# Calculando a acurácia
accuracy = accuracy_score(y_test, predictions)

# Exibindo a acurácia
print(f'Acurácia do modelo Random Forest: {accuracy:.2f}')
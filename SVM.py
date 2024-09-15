import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Função para carregar arquivos .dat
def load_dat_file(file_path, target_size=None):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32)
    if target_size:
        if len(data) > target_size:
            data = data[:target_size]
        else:
            data = np.pad(data, (0, target_size - len(data)), 'constant')
    return data


# Função para limpar dados
def clean_data(X):
    # Substituir valores infinitos e NaNs por zero
    X[np.isinf(X)] = 0
    X[np.isnan(X)] = 0
    return X


# Caminho para a pasta principal onde os arquivos .dat estão localizados
base_dir = "20181109"

# Lista de usuários
usuarios = ['user1', 'user2', 'user3']  # Ajuste conforme os usuários reais

# Quantidade de arquivos por usuário
num_files_per_user = 10

# Tamanho alvo para todos os arrays
target_size = 5000

# Dataset final
X = []
Y = []

# Carregar arquivos .dat de cada usuário
for user_idx, user in enumerate(usuarios):
    user_dir = os.path.join(base_dir, user)
    dat_files = [f for f in os.listdir(user_dir) if f.endswith('.dat')][:num_files_per_user]

    for dat_file in dat_files:
        file_path = os.path.join(user_dir, dat_file)
        data = load_dat_file(file_path, target_size=target_size)
        X.append(data)
        Y.append(1 if user_idx == 0 else 0)  # Ajuste a etiqueta conforme necessário

# Converter para arrays numpy
X = np.array(X)
Y = np.array(Y)

# Limpar os dados
X = clean_data(X)

# Normalizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Instanciar e treinar o modelo SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Fazer previsões e avaliar o modelo
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")


print("Máximo valor em X:", np.max(X))
print("Mínimo valor em X:", np.min(X))
print("Número de NaNs em X:", np.isnan(X).sum())
print("Número de infinitos em X:", np.isinf(X).sum())

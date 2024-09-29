import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    X[np.isinf(X)] = 0  # Substitui valores infinitos por zero
    X[np.isnan(X)] = 0  # Substitui valores NaN por zero
    X = np.clip(X, -1e6, 1e6)  # Limita valores extremos entre -1e6 e 1e6
    return X

# Caminho para a pasta principal onde os arquivos .dat estão localizados
base_dir = "20181109"

# Lista de usuários
usuarios = ['user1', 'user2', 'user3']  # Três usuários

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
        Y.append(user_idx)  # Ajuste para atribuir um rótulo único a cada usuário (0, 1, 2)

# Converter para arrays numpy
X = np.array(X)
Y = np.array(Y)

# Exibe informações dos dados antes da limpeza
print("Máximo valor antes da limpeza:", np.max(X))
print("Mínimo valor antes da limpeza:", np.min(X))
print("Número de NaNs antes da limpeza:", np.isnan(X).sum())
print("Número de infinitos antes da limpeza:", np.isinf(X).sum())

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

# PCA para reduzir a dimensão para 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotar os resultados
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[Y == 0, 0], X_pca[Y == 0, 1], color='red', label='Usuário 0')
plt.scatter(X_pca[Y == 1, 0], X_pca[Y == 1, 1], color='blue', label='Usuário 1')
plt.scatter(X_pca[Y == 2, 0], X_pca[Y == 2, 1], color='orange', label='Usuário 2')

# Adicionar previsões do SVM
X_test_pca = pca.transform(X_test)
plt.scatter(X_test_pca[y_pred == 0, 0], X_test_pca[y_pred == 0, 1], color='green', marker='x', label='Predições Usuário 0', alpha=0.5)
plt.scatter(X_test_pca[y_pred == 1, 0], X_test_pca[y_pred == 1, 1], color='purple', marker='x', label='Predições Usuário 1', alpha=0.5)
plt.scatter(X_test_pca[y_pred == 2, 0], X_test_pca[y_pred == 2, 1], color='yellow', marker='x', label='Predições Usuário 2', alpha=0.5)

plt.title('Visualização do SVM com PCA (2D)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(True)
plt.show()

# Verificar novamente os dados após normalização e limpeza
print("Máximo valor em X:", np.max(X))
print("Mínimo valor em X:", np.min(X))
print("Número de NaNs em X:", np.isnan(X).sum())
print("Número de infinitos em X:", np.isinf(X).sum())

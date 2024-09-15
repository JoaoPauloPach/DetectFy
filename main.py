import os

# Diretório onde os arquivos descompactados estão localizados
extract_path = "20181109"

# Listar os arquivos e diretórios
for root, dirs, files in os.walk(extract_path):
    print(f"Diretório: {root}")
    for file in files:
        print(f"  Arquivo: {file}")


#segunda parte

import numpy as np

# Definir o caminho para o arquivo .dat
data_path = "20181109/user1/user1-1-1-1-5-r4.dat"

# Ler o arquivo binário
with open(data_path, 'rb') as file:
    binary_data = file.read()

# Agora você pode processar 'binary_data' conforme a estrutura do arquivo
# Exemplo simples usando numpy se souber a estrutura dos dados:
data_array = np.frombuffer(binary_data, dtype=np.float32)  # Ajuste o tipo conforme necessário
print(data_array[:10])  # Exibir os primeiros 10 valores
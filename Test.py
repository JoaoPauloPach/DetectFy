import os

# Define o caminho completo do arquivo
file_path = '20181109/user1/user1-1-1-1-5-r4.dat'

# Verifica se o arquivo existe
if os.path.exists(file_path):
    # Carrega o arquivo em 'dataset'
    with open(file_path, 'rb') as file:
        dataset = file.read()
        #print(dataset);## aparentemente está em binário
    print('Arquivo carregado com sucesso.')
else:
    print('Arquivo não encontrado.')


import numpy as np

# Carrega o arquivo como um array NumPy
dataset = np.fromfile(file_path, dtype=np.complex64)  # ou outro dtype apropriado

# Exibe a forma do array e os primeiros elementos
print(f'Shape: {dataset.shape}')
print(f'First elements: {dataset[:10]}')


import numpy as np
import matplotlib.pyplot as plt

# Carrega o arquivo binário como um array NumPy
dataset = np.fromfile(file_path, dtype=np.float32)  # Certifique-se de que o dtype está correto

# Verifica se o arquivo tem pelo menos 500 amostras
if len(dataset) >= 500:
    # Pega as 500 amostras iniciais
    samples = dataset[:10000]

    # Faz o plot das amostras
    plt.figure(figsize=(10, 6))
    plt.plot(samples)
    plt.title('Plot das 10000 amostras iniciais')
    plt.xlabel('Amostra')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.show()
else:
    print('O arquivo não contém 500 amostras.')
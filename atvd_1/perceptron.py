import numpy as np

# Conjunto de treinamento
# x1, x2, x3, d
train_data = np.array([
    [-0.6508, 0.1097, 4.0009, -1.0000],
    [-1.4492, 0.8896, 4.4005, -1.0000],
    [2.0850, 0.6876, 12.0710, -1.0000],
    [0.2626, 1.1476, 7.7985, 1.0000],
    [0.6418, 1.0234, 7.0427, 1.0000],
    [0.2569, 0.6730, 8.3265, -1.0000],
    [1.1155, 0.6043, 7.4446, 1.0000],
    [0.0914, 0.3399, 7.0677, -1.0000],
    [0.0121, 0.5256, 4.6316, 1.0000],
    [-0.0429, 0.4660, 5.4323, 1.0000],
    [0.4340, 0.6870, 8.2287, -1.0000],
    [0.2735, 1.0287, 7.1934, 1.0000],
    [0.4839, 0.4851, 7.4850, -1.0000],
    [0.4089, -0.1267, 5.5019, -1.0000],
    [1.4391, 0.1614, 8.5843, -1.0000],
    [-0.9115, -0.1973, 2.1962, -1.0000],
    [0.3654, 1.0475, 7.4858, 1.0000],
    [0.2144, 0.7515, 7.1699, 1.0000],
    [0.2013, 1.0014, 6.5489, 1.0000],
    [0.6483, 0.2183, 5.8991, 1.0000],
    [-0.1147, 0.2242, 7.2435, -1.0000],
    [-0.7970, 0.8795, 3.8762, 1.0000],
    [-1.0625, 0.6366, 2.4707, 1.0000],
    [0.5307, 0.1285, 5.6883, 1.0000],
    [-1.2200, 0.7777, 1.7252, 1.0000],
    [0.3957, 0.1076, 5.6623, -1.0000],
    [-0.1013, 0.5989, 7.1812, -1.0000],
    [2.4482, 0.9455, 11.2095, 1.0000],
    [2.0149, 0.6192, 10.9263, -1.0000],
    [0.2012, 0.2611, 5.4631, 1.0000]
])

# Amostras para teste
test_data = np.array([
    [-0.3565, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548]
])

X_train = train_data[:, :3]
d_train = train_data[:, 3]

# Adicionando x0 = -1 aos dados de treinamento e teste
# O vetor de entrada passa a ser [x0, x1, x2, x3] e os pesos [w0, w1, w2, w3]
x0_train = np.full((X_train.shape[0], 1), -1.0)
X_train_ext = np.hstack((x0_train, X_train))

x0_test = np.full((test_data.shape[0], 1), -1.0)
X_test_ext = np.hstack((x0_test, test_data))

eta = 0.01
max_epochs = 1000

def g(v):
    return 1.0 if v >= 0 else -1.0

results_train = []
results_test = []

for i in range(5):
    np.random.seed(i * 42 + 10) # Para garantir valores diferentes a cada execução do laço
    W_initial = np.random.rand(4)
    W = W_initial.copy()
    
    epochs = 0
    while epochs < max_epochs:
        error_count = 0
        for j in range(len(X_train_ext)):
            x = X_train_ext[j]
            d = d_train[j]
            v = np.dot(W, x)
            y = g(v)
            if y != d:
                # Regra de Hebb Supervisionada / Algoritmo Perceptron
                # Como a saída é -1 ou 1, muitos usam d diretamente, ou (d - y).
                # (d-y) tem magnitude 2, então estamos efetivamente multiplicando o erro por 2. 
                # Vamos usar W = W + eta * (d - y) * x (padrão)
                # Ou a versão estrita que muitos livros no BR usam: W = W + eta * d * x se y != d
                W = W + eta * d * x
                error_count += 1
        epochs += 1
        if error_count == 0:
            break
            
    results_train.append({
        'Treinamento': f'T{i+1}',
        'W_inicial': W_initial.copy(),
        'W_final': W.copy(),
        'epocas': epochs
    })
    
    y_test = []
    for x in X_test_ext:
        v = np.dot(W, x)
        y_test.append(g(v))
    results_test.append(y_test)

print("--- RESULTADOS DOS TREINAMENTOS ---")
for res in results_train:
    w_ini = [f"{w:.4f}" for w in res['W_inicial']]
    w_fin = [f"{w:.4f}" for w in res['W_final']]
    print(f"{res['Treinamento']}: Epocas={res['epocas']}")
    print(f"  Inicial: {w_ini}")
    print(f"  Final  : {w_fin}")

print("\n--- RESULTADOS DOS TESTES ---")
for j in range(len(test_data)):
    res_linha = [f"{results_test[i][j]:>4.1f}" for i in range(5)]
    print(f"Amostra {j+1:>2}: {', '.join(res_linha)}")

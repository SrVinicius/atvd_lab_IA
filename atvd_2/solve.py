import numpy as np
import matplotlib.pyplot as plt

train_data = [
[0.4329, -1.3719, 0.7022, -0.8535, 1.0],
[0.3024, 0.2286, 0.8630, 2.7909, -1.0],
[0.1349, -0.6445, 1.0530, 0.5687, -1.0],
[0.3374, -1.7163, 0.3670, -0.6283, -1.0],
[1.1434, -0.0485, 0.6637, 1.2606, 1.0],
[1.3749, -0.5071, 0.4464, 1.3009, 1.0],
[0.7221, -0.7587, 0.7681, -0.5592, 1.0],
[0.4403, -0.8072, 0.5154, -0.3129, 1.0],
[-0.5231, 0.3548, 0.2538, 1.5776, -1.0],
[0.3255, -2.0000, 0.7112, -1.1209, 1.0],
[0.5824, 1.3915, -0.2291, 4.1735, -1.0],
[0.1340, 0.6081, 0.4450, 3.2230, -1.0],
[0.1480, -0.2988, 0.4778, 0.8649, 1.0],
[0.7359, 0.1869, -0.0872, 2.3584, 1.0],
[0.7115, -1.1469, 0.3394, 0.9573, -1.0],
[0.8251, -1.2840, 0.8452, 1.2382, -1.0],
[0.1569, 0.3712, 0.8825, 1.7633, 1.0],
[0.0033, 0.6835, 0.5389, 2.8249, -1.0],
[0.4243, 0.8313, 0.2634, 3.5855, -1.0],
[1.0490, 0.1326, 0.9138, 1.9792, 1.0],
[1.4276, 0.5331, -0.0145, 3.7286, 1.0],
[0.5971, 1.4865, 0.2904, 4.6069, -1.0],
[0.8475, 2.1479, 0.3179, 5.8235, -1.0],
[1.3967, -0.4171, 0.6443, 1.3927, 1.0],
[0.0044, 1.5378, 0.6099, 4.7755, -1.0],
[0.2201, -0.5668, 0.0515, 0.7829, 1.0],
[0.6300, -1.2480, 0.8591, 0.8093, -1.0],
[-0.2479, 0.8960, 0.0547, 1.7381, 1.0],
[-0.3088, -0.0929, 0.8659, 1.5483, -1.0],
[-0.5180, 1.4974, 0.5453, 2.3993, 1.0],
[0.6833, 0.8266, 0.0829, 2.8864, 1.0],
[0.4353, -1.4066, 0.4207, -0.4879, 1.0],
[-0.1069, -3.2329, 0.1856, -2.4572, -1.0],
[0.4662, 0.6261, 0.7304, 3.4370, -1.0],
[0.8298, -1.4089, 0.3119, 1.3235, -1.0],
]

test_data = [
[0.9694, 0.6909, 0.4334, 3.4965],
[0.5427, 1.3832, 0.6390, 4.0352],
[0.6081, -0.9196, 0.5925, 0.1016],
[-0.1618, 0.4694, 0.2030, 3.0117],
[0.1870, -0.2578, 0.6124, 1.7749],
[0.4891, -0.5276, 0.4378, 0.6439],
[0.3777, 2.0149, 0.7423, 3.3932],
[1.1498, -0.4067, 0.2469, 1.5866],
[0.9325, 1.0950, 1.0359, 3.3591],
[0.5060, 1.3317, 0.9222, 3.7174],
[0.0497, -2.0656, 0.6124, -0.6585],
[0.4004, 3.5369, 0.9766, 5.3532],
[-0.1874, 1.3343, 0.5374, 3.2189],
[0.5060, 1.3317, 0.9222, 3.7174],
[1.6375, -0.7911, 0.7537, 0.5515],
]

eta = 0.0025
epsilon = 1e-6
np.random.seed(42)

def calc_eqm(X, D, W):
    errors = []
    for x, d in zip(X, D):
        u = np.dot(W, x)
        errors.append((d - u)**2)
    return np.mean(errors)

X_train = []
D_train = []
for row in train_data:
    X_train.append([-1, row[0], row[1], row[2], row[3]])
    D_train.append(row[4])
X_train = np.array(X_train)
D_train = np.array(D_train)

X_test = []
for row in test_data:
    X_test.append([-1, row[0], row[1], row[2], row[3]])
X_test = np.array(X_test)

results = []
eqms_all = []

for run in range(5):
    initial_w = np.random.rand(5)
    W = initial_w.copy()
    
    eqms = []
    old_eqm = calc_eqm(X_train, D_train, W)
    eqms.append(old_eqm)
    
    epochs = 0
    while True:
        epochs += 1
        for i in range(len(X_train)):
            u = np.dot(W, X_train[i])
            error = D_train[i] - u
            W = W + eta * error * X_train[i]
        
        new_eqm = calc_eqm(X_train, D_train, W)
        eqms.append(new_eqm)
        
        if abs(new_eqm - old_eqm) < epsilon:
            break
        old_eqm = new_eqm
        
    results.append({
        "initial_w": initial_w,
        "final_w": W,
        "epochs": epochs,
    })
    eqms_all.append(eqms)

plt.figure(figsize=(10,6))
plt.plot(eqms_all[0], label="Treino 1")
plt.plot(eqms_all[1], label="Treino 2")
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.title("Curva de Aprendizagem (EQM x Épocas) - T1 e T2")
plt.legend()
plt.grid()
plt.savefig("graficos_T1_T2.png")

with open("README.md", "w") as f:
    f.md = """# Atividade 2 - Adaline\n\n"""
    f.write("# Atividade 2 - Adaline\n\n")
    f.write("## 1. Treinamentos\n\n")
    f.write("| Treinamento | Wi (w0, w1, w2, w3, w4) | Wf (w0, w1, w2, w3, w4) | Épocas |\n")
    f.write("|-------------|---------------------------|---------------------------|--------|\n")
    for i, r in enumerate(results):
        wi = ", ".join([f"{x:.4f}" for x in r['initial_w']])
        wf = ", ".join([f"{x:.4f}" for x in r['final_w']])
        f.write(f"| {i+1}o (T{i+1}) | {wi} | {wf} | {r['epochs']} |\n")
        
    f.write("\n## 2. Gráfico do EQM x Épocas\n\n")
    f.write("![Grafico de EQM](graficos_T1_T2.png)\n")
    
    f.write("\n## 3. Classificação das Amostras (Teste)\n\n")
    f.write("| Amostra | x1 | x2 | x3 | x4 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |\n")
    f.write("|---------|----|----|----|----|--------|--------|--------|--------|--------|\n")
    
    for i, row in enumerate(test_data):
        y_outputs = []
        for run in range(5):
            u = np.dot(results[run]['final_w'], X_test[i])
            y = 1 if u >= 0 else -1
            y_outputs.append(str(y))
        
        x_str = " | ".join([f"{v:.4f}" for v in row])
        y_str = " | ".join(y_outputs)
        f.write(f"| {i+1} | {x_str} | {y_str} |\n")
        
    f.write("\n## 4. Explicação sobre os pesos\n\n")
    f.write("Embora os pesos iniciais sejam diferentes e, portanto, o algoritmo inicie com taxas de erro distintas (o que acarreta diferentes números de épocas para atingir o critério de parada), o vetor de pesos final converge para o mínimo do Erro Quadrático Médio (que é um paraboloide com apenas um mínimo global para o ADALINE). Portanto, o ponto de convergência otimizado dos pesos sempre será o mesmo ou quase inalterado, pois todos estão aproximando o mesmo limite na superfície de erro (o hiperplano de separação linear ótimo das classes).\n")

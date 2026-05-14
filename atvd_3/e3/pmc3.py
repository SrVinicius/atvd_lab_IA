"""
PMC3 - TDNN (Time Delay Neural Network) para Previsão de Séries Temporais
Problema: Previsão do preço de mercadoria no mercado financeiro
Arquiteturas candidatas:
  Rede 1: p=5  entradas, N1=10 neurônios ocultos, 1 saída
  Rede 2: p=10 entradas, N1=15 neurônios ocultos, 1 saída
  Rede 3: p=15 entradas, N1=25 neurônios ocultos, 1 saída
Algoritmo: Backpropagation com Momentum (α=0.8)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────── Série temporal ────────────────────
# f(t) para t = 1..100 (treinamento) E t = 101..120 (teste)
F_TRAIN = np.array([
    0.1701,0.1023,0.4405,0.3609,0.7192,0.2258,0.3175,0.0127,0.4290,0.0544,
    0.8000,0.0450,0.4268,0.0112,0.3218,0.2185,0.7240,0.3516,0.4420,0.0984,
    0.1747,0.3964,0.5114,0.6183,0.3330,0.2398,0.0508,0.4497,0.2178,0.7762,
    0.1078,0.3773,0.0001,0.3877,0.0821,0.7836,0.1887,0.4483,0.0424,0.2539,
    0.3164,0.6386,0.4862,0.4068,0.1611,0.1101,0.4372,0.3795,0.7092,0.2400,
    0.3087,0.0159,0.4330,0.0733,0.7995,0.0262,0.4223,0.0085,0.3303,0.2037,
    0.7332,0.3328,0.4445,0.0909,0.1838,0.3888,0.5277,0.6042,0.3435,0.2304,
    0.0568,0.4500,0.2371,0.7705,0.1246,0.3701,0.0006,0.3943,0.0646,0.7878,
    0.1694,0.4468,0.0372,0.2632,0.3048,0.6516,0.4690,0.4132,0.1523,0.1182,
    0.4334,0.3978,0.6987,0.2538,0.2998,0.0195,0.4366,0.0924,0.7984,0.0077,
])

F_TEST = np.array([
    0.4173,0.0062,0.3387,0.1886,0.7418,0.3138,0.4466,0.0835,0.1930,0.3807,
    0.5438,0.5897,0.3536,0.2210,0.0631,0.4499,0.2564,0.7642,0.1411,0.3626,
])

T_TEST = np.arange(101, 121)

# ─────────────────────── Parâmetros gerais ─────────────────────────
ETA    = 0.1
ALPHA  = 0.8
EPS    = 0.5e-6
MAX_EP = 500_000
SEEDS  = [0, 42, 123]

REDES = [
    {"nome": "Rede 1", "p": 5,  "n1": 10},
    {"nome": "Rede 2", "p": 10, "n1": 15},
    {"nome": "Rede 3", "p": 15, "n1": 25},
]

# ─────────────────────── Funções auxiliares ────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def forward(X, W1, W2):
    N   = X.shape[0]
    Xb  = np.hstack([X,  np.ones((N, 1))])
    a1  = sigmoid(Xb @ W1.T)
    a1b = np.hstack([a1, np.ones((N, 1))])
    a2  = sigmoid(a1b @ W2.T)
    return Xb, a1, a1b, a2

def eqm(d, y):
    e = d.reshape(-1, 1) - y
    return float(np.sum(e**2) / (2 * len(d)))

def build_dataset(serie, p):
    """Constrói pares (entrada, saída) para TDNN a partir da série temporal."""
    X, d = [], []
    for t in range(p, len(serie)):
        X.append(serie[t-p:t])
        d.append(serie[t])
    return np.array(X), np.array(d)

def treinar(X, d, n_in, n_h1, seed):
    np.random.seed(seed)
    W1 = np.random.uniform(0, 1, (n_h1, n_in + 1))
    W2 = np.random.uniform(0, 1, (1,    n_h1 + 1))
    dW1_ant = np.zeros_like(W1)
    dW2_ant = np.zeros_like(W2)
    N = X.shape[0]
    historico = []
    eqm_ant   = float('inf')

    for ep in range(1, MAX_EP + 1):
        Xb, a1, a1b, a2 = forward(X, W1, W2)
        e      = d.reshape(-1, 1) - a2
        delta2 = e * a2 * (1 - a2)
        delta1 = (delta2 @ W2[:, :-1]) * a1 * (1 - a1)
        dW2 = (ETA / N) * (delta2.T @ a1b) + ALPHA * dW2_ant
        dW1 = (ETA / N) * (delta1.T @ Xb)  + ALPHA * dW1_ant
        dW2_ant, dW1_ant = dW2.copy(), dW1.copy()
        W2 += dW2
        W1 += dW1
        val = eqm(d, forward(X, W1, W2)[3])
        historico.append(val)
        if abs(eqm_ant - val) < EPS:
            break
        eqm_ant = val

    return W1, W2, historico[-1], len(historico), historico

def prever_recursivo(W1, W2, f_hist, n_passos, p):
    """Previsão autoregressiva: usa saídas anteriores como entradas futuras."""
    hist = list(f_hist[-p:])
    preds = []
    for _ in range(n_passos):
        x = np.array(hist[-p:]).reshape(1, -1)
        y = forward(x, W1, W2)[3][0, 0]
        preds.append(y)
        hist.append(y)
    return np.array(preds)

# ─────────────────────── Treinamento e validação ───────────────────
todos_resultados = {}
melhor_por_rede  = {}

print("=" * 70)
print(f"{'Rede':<10} {'T':<4} {'EQM Final':<18} {'Épocas'}")
print("=" * 70)

for cfg in REDES:
    nome, p, n1 = cfg["nome"], cfg["p"], cfg["n1"]
    X, d = build_dataset(F_TRAIN, p)
    resultados_rede = []

    for ti, seed in enumerate(SEEDS):
        W1, W2, val, ep, hist = treinar(X, d, p, n1, seed)
        resultados_rede.append({"W1": W1, "W2": W2, "eqm": val, "epocas": ep, "hist": hist})
        print(f"  {nome:<8} T{ti+1:<3} {val:<18.8f} {ep}")

    todos_resultados[nome] = resultados_rede
    melhor_idx = int(np.argmin([r["eqm"] for r in resultados_rede]))
    melhor_por_rede[nome] = (melhor_idx, resultados_rede[melhor_idx])

print("=" * 70)

# ─────────────────────── Validação com conjunto de teste ───────────
print("\nValidação (t=101..120) — Erro Relativo Médio e Variância:")
print("=" * 75)
print(f"{'Rede':<10} {'T':<4} {'ERM (%)':<14} {'Variância (%²)'}")
print("=" * 75)

erros_teste = {}
for cfg in REDES:
    nome, p = cfg["nome"], cfg["p"]
    erros_rede = []
    for ti, r in enumerate(todos_resultados[nome]):
        preds = prever_recursivo(r["W1"], r["W2"], F_TRAIN, 20, p)
        rel   = np.abs((F_TEST - preds) / (F_TEST + 1e-10)) * 100
        erm   = float(np.mean(rel))
        var   = float(np.var(rel))
        erros_rede.append((erm, var, preds))
        print(f"  {nome:<8} T{ti+1:<3} {erm:<14.4f} {var:.4f}")
    erros_teste[nome] = erros_rede
print("=" * 75)

# ─────────────────── Gráficos 1: EQM do melhor treinamento ─────────
fig, axes = plt.subplots(3, 1, figsize=(11, 12))
fig.suptitle("EQM × Época — Melhor Treinamento de Cada Topologia", fontsize=13)
cores = ['steelblue', 'tomato', 'seagreen']

for ax, cfg, cor in zip(axes, REDES, cores):
    nome = cfg["nome"]
    idx_mel, r_mel = melhor_por_rede[nome]
    hist = r_mel["hist"]
    ax.semilogy(range(1, len(hist)+1), hist, linewidth=0.8, color=cor)
    ax.set_title(f"{nome} — T{idx_mel+1}  (Épocas={r_mel['epocas']}, EQM={r_mel['eqm']:.2e})")
    ax.set_xlabel("Época"); ax.set_ylabel("EQM (log)"); ax.grid(True, which="both", alpha=0.4)

plt.tight_layout()
plt.savefig("e3/graficos_eqm.png", dpi=150)
plt.close()
print("\nGráfico EQM salvo em e3/graficos_eqm.png")

# ─────────── Gráficos 2: valores desejados vs estimados ────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 12))
fig.suptitle("Valores Desejados vs Estimados (t=101..120)", fontsize=13)

for ax, cfg, cor in zip(axes, REDES, cores):
    nome, p = cfg["nome"], cfg["p"]
    idx_mel, _ = melhor_por_rede[nome]
    _, _, preds = erros_teste[nome][idx_mel]
    ax.plot(T_TEST, F_TEST, 'ko-', markersize=5, linewidth=1.2, label="Desejado")
    ax.plot(T_TEST, preds,  color=cor, marker='s', markersize=5, linewidth=1.2,
            linestyle='--', label=f"Estimado T{idx_mel+1}")
    ax.set_title(f"{nome} — T{idx_mel+1}  (ERM={erros_teste[nome][idx_mel][0]:.2f}%)")
    ax.set_xlabel("t"); ax.set_ylabel("f(t)"); ax.legend(); ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("e3/graficos_predicao.png", dpi=150)
plt.close()
print("Gráfico de previsão salvo em e3/graficos_predicao.png")

# ─────────────────── Melhor rede global ────────────────────────────
print("\nResumo — melhor treinamento por topologia:")
melhor_global_erm = float('inf')
melhor_global = None
for cfg in REDES:
    nome = cfg["nome"]
    idx_mel, _ = melhor_por_rede[nome]
    erm, var, _ = erros_teste[nome][idx_mel]
    ep  = todos_resultados[nome][idx_mel]["epocas"]
    eqm_val = todos_resultados[nome][idx_mel]["eqm"]
    print(f"  {nome}: T{idx_mel+1}  EQM={eqm_val:.2e}  Épocas={ep}  ERM={erm:.4f}%  Var={var:.4f}%²")
    if erm < melhor_global_erm:
        melhor_global_erm = erm
        melhor_global = (nome, idx_mel+1)

print(f"\nMelhor configuração: {melhor_global[0]}, T{melhor_global[1]}  (ERM={melhor_global_erm:.4f}%)")

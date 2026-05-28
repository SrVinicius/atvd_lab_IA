# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Dados do Apêndice
# ---------------------------------------------------------------------------

TRAINING_DATA = np.array([
    # Classe A – amostras 1-20
    [0.2417, 0.2857, 0.2397],
    [0.2268, 0.2874, 0.2153],
    [0.1975, 0.3315, 0.1965],
    [0.3414, 0.3166, 0.1074],
    [0.2587, 0.1918, 0.2634],
    [0.2455, 0.2075, 0.1344],
    [0.3163, 0.1679, 0.1725],
    [0.2704, 0.2605, 0.1411],
    [0.1871, 0.2965, 0.1231],
    [0.3474, 0.2715, 0.1958],
    [0.2059, 0.2928, 0.2839],
    [0.2442, 0.2272, 0.2384],
    [0.2126, 0.3437, 0.1128],
    [0.2562, 0.2542, 0.1599],
    [0.1640, 0.2289, 0.2627],
    [0.2795, 0.1880, 0.1627],
    [0.3463, 0.1513, 0.2281],
    [0.3430, 0.1508, 0.1881],
    [0.1981, 0.2821, 0.1294],
    [0.2322, 0.3025, 0.2191],
    # Classe B – amostras 21-60
    [0.7352, 0.2722, 0.6962],
    [0.7191, 0.1825, 0.7470],
    [0.6921, 0.1537, 0.8172],
    [0.6833, 0.2048, 0.8490],
    [0.8012, 0.2684, 0.7673],
    [0.7860, 0.1734, 0.7198],
    [0.7205, 0.1542, 0.7295],
    [0.6549, 0.3288, 0.8153],
    [0.6968, 0.3173, 0.7389],
    [0.7448, 0.2095, 0.6847],
    [0.6746, 0.3277, 0.6725],
    [0.7897, 0.2801, 0.7679],
    [0.8399, 0.3067, 0.7003],
    [0.8065, 0.3206, 0.7205],
    [0.8357, 0.3220, 0.7879],
    [0.7438, 0.3230, 0.8384],
    [0.8172, 0.3319, 0.7628],
    [0.8248, 0.2614, 0.8405],
    [0.6979, 0.2142, 0.7309],
    [0.6804, 0.3181, 0.7017],
    [0.6973, 0.3194, 0.7522],
    [0.7910, 0.2239, 0.7018],
    [0.7052, 0.2148, 0.6866],
    [0.8088, 0.1908, 0.7563],
    [0.7640, 0.1676, 0.6994],
    [0.7616, 0.2881, 0.8087],
    [0.8188, 0.2461, 0.7273],
    [0.7920, 0.3178, 0.7497],
    [0.7802, 0.1871, 0.8102],
    [0.7332, 0.2543, 0.8194],
    [0.6921, 0.1529, 0.7759],
    [0.6833, 0.2197, 0.6943],
    [0.7860, 0.1745, 0.7639],
    [0.8009, 0.3082, 0.8491],
    [0.7793, 0.1935, 0.6738],
    [0.7373, 0.2698, 0.7864],
    [0.7048, 0.2380, 0.7825],
    [0.8393, 0.2857, 0.7733],
    [0.6878, 0.2126, 0.6961],
    [0.6651, 0.3492, 0.6737],
    # Classe C – amostras 61-120
    [0.4856, 0.6600, 0.4798],
    [0.4114, 0.7220, 0.5106],
    [0.5671, 0.7935, 0.5929],
    [0.4875, 0.7928, 0.5532],
    [0.5172, 0.7147, 0.5774],
    [0.5483, 0.6773, 0.4842],
    [0.5740, 0.6682, 0.5335],
    [0.4587, 0.6981, 0.5900],
    [0.5794, 0.7410, 0.4759],
    [0.4712, 0.6734, 0.5677],
    [0.5126, 0.8141, 0.5224],
    [0.5557, 0.7749, 0.4342],
    [0.4916, 0.8267, 0.4586],
    [0.4629, 0.8129, 0.4950],
    [0.5850, 0.7358, 0.5107],
    [0.4435, 0.7030, 0.4594],
    [0.4155, 0.7516, 0.5524],
    [0.4887, 0.7027, 0.5886],
    [0.5462, 0.7378, 0.5107],
    [0.5251, 0.8124, 0.5686],
    [0.4635, 0.7339, 0.5638],
    [0.5907, 0.7144, 0.4718],
    [0.4982, 0.8335, 0.4597],
    [0.5242, 0.7325, 0.4079],
    [0.4075, 0.8372, 0.4271],
    [0.5934, 0.8284, 0.5107],
    [0.5463, 0.6766, 0.5639],
    [0.4403, 0.8495, 0.4806],
    [0.4531, 0.7760, 0.5276],
    [0.5109, 0.7387, 0.5373],
    [0.5383, 0.7780, 0.4955],
    [0.5679, 0.7156, 0.5022],
    [0.5762, 0.7781, 0.5908],
    [0.5997, 0.7504, 0.5678],
    [0.4138, 0.6975, 0.5148],
    [0.5490, 0.6674, 0.4472],
    [0.4719, 0.7527, 0.4401],
    [0.4458, 0.8063, 0.4253],
    [0.4983, 0.8131, 0.5625],
    [0.5742, 0.6789, 0.5997],
    [0.5289, 0.7354, 0.4718],
    [0.5927, 0.7738, 0.5390],
    [0.5199, 0.7131, 0.4028],
    [0.5716, 0.6558, 0.4451],
    [0.5075, 0.7045, 0.4233],
    [0.4886, 0.7004, 0.4608],
    [0.5527, 0.8243, 0.5772],
    [0.4816, 0.6969, 0.4678],
    [0.5809, 0.6557, 0.4266],
    [0.5881, 0.7565, 0.4003],
    [0.5334, 0.8446, 0.4934],
    [0.4603, 0.7992, 0.4816],
    [0.5491, 0.6504, 0.4063],
    [0.4288, 0.8455, 0.5047],
    [0.5636, 0.7884, 0.5417],
    [0.5349, 0.6736, 0.4541],
    [0.5569, 0.8393, 0.5652],
    [0.4729, 0.7702, 0.5325],
    [0.5472, 0.8454, 0.5449],
    [0.5805, 0.7349, 0.4464],
])

TEST_SAMPLES = np.array([
    [0.2471, 0.1778, 0.2905],
    [0.8240, 0.2223, 0.7041],
    [0.4960, 0.7231, 0.5866],
    [0.2923, 0.2041, 0.2234],
    [0.8118, 0.2668, 0.7484],
    [0.4837, 0.8200, 0.4792],
    [0.3248, 0.2629, 0.2375],
    [0.7209, 0.2116, 0.7821],
    [0.5259, 0.6522, 0.5957],
    [0.2075, 0.1669, 0.1745],
    [0.7830, 0.3171, 0.7888],
    [0.5393, 0.7510, 0.5682],
])

# ---------------------------------------------------------------------------
# Parâmetros da rede
# ---------------------------------------------------------------------------

GRID_ROWS = 4
GRID_COLS = 4
N1 = GRID_ROWS * GRID_COLS   # 16 neurônios
ETA = 0.001
RADIUS = 1
EPOCHS = 1000
N_FEATURES = 3


# ---------------------------------------------------------------------------
# Funções auxiliares do grid
# ---------------------------------------------------------------------------

def idx_to_pos(idx):
    return (idx // GRID_COLS, idx % GRID_COLS)


def chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


def neighborhood(winner_idx, neuron_idx, radius):
    return chebyshev(idx_to_pos(winner_idx), idx_to_pos(neuron_idx)) <= radius


def find_winner(x, weights):
    dists = np.linalg.norm(weights - x, axis=1)
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def train(data, epochs=EPOCHS, eta=ETA, radius=RADIUS, seed=42):
    rng = np.random.default_rng(seed)
    weights = rng.uniform(data.min(), data.max(), (N1, N_FEATURES))

    for _ in range(epochs):
        for idx in rng.permutation(len(data)):
            x = data[idx]
            w = find_winner(x, weights)
            for j in range(N1):
                if neighborhood(w, j, radius):
                    weights[j] += eta * (x - weights[j])

    return weights


# ---------------------------------------------------------------------------
# Classificação e análise
# ---------------------------------------------------------------------------

def assign_class(neuron_idx, class_a, class_b, class_c):
    if neuron_idx in class_a:
        return 'A'
    if neuron_idx in class_b:
        return 'B'
    if neuron_idx in class_c:
        return 'C'
    return '?'


def build_class_sets(winners):
    return (
        set(winners[:20]),
        set(winners[20:60]),
        set(winners[60:120]),
    )


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

_COLORS = {'A': '#AED6F1', 'B': '#F1948A', 'C': '#A9DFBF', '?': '#EBEBEB'}


def plot_grid(class_map, winner_counts):
    fig, ax = plt.subplots(figsize=(6, 6))

    for idx in range(N1):
        r, c = idx_to_pos(idx)
        display_row = GRID_ROWS - 1 - r
        cls = class_map.get(idx, '?')
        rect = mpatches.FancyBboxPatch(
            (c + 0.05, display_row + 0.05), 0.9, 0.9,
            boxstyle='round,pad=0.05',
            facecolor=_COLORS[cls], edgecolor='#555555', linewidth=1.2,
        )
        ax.add_patch(rect)
        count = winner_counts.get(idx, 0)
        ax.text(c + 0.5, display_row + 0.5,
                f'N{idx + 1}\n({cls})\nn={count}',
                ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, GRID_COLS)
    ax.set_ylim(0, GRID_ROWS)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Grid Kohonen 4×4 – Regiões por Classe', fontsize=11)

    legend = [
        mpatches.Patch(facecolor=_COLORS['A'], edgecolor='#555555', label='Classe A (amostras 1–20)'),
        mpatches.Patch(facecolor=_COLORS['B'], edgecolor='#555555', label='Classe B (amostras 21–60)'),
        mpatches.Patch(facecolor=_COLORS['C'], edgecolor='#555555', label='Classe C (amostras 61–120)'),
    ]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=8)

    plt.tight_layout()
    out = 'atvd_5/kohonen_grid.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Figura salva em: {out}')


def plot_weights_3d(weights, class_map):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for idx in range(N1):
        cls = class_map.get(idx, '?')
        w = weights[idx]
        ax.scatter(*w, c=_COLORS[cls], s=120, edgecolors='#333333', linewidth=0.8, zorder=5)
        ax.text(w[0], w[1], w[2], f'N{idx+1}', fontsize=6)

    ax.scatter(TRAINING_DATA[:20, 0], TRAINING_DATA[:20, 1], TRAINING_DATA[:20, 2],
               c=_COLORS['A'], marker='^', s=20, alpha=0.4, label='Classe A')
    ax.scatter(TRAINING_DATA[20:60, 0], TRAINING_DATA[20:60, 1], TRAINING_DATA[20:60, 2],
               c=_COLORS['B'], marker='^', s=20, alpha=0.4, label='Classe B')
    ax.scatter(TRAINING_DATA[60:, 0], TRAINING_DATA[60:, 1], TRAINING_DATA[60:, 2],
               c=_COLORS['C'], marker='^', s=20, alpha=0.4, label='Classe C')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('Pesos dos neurônios no espaço de entrada')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = 'atvd_5/kohonen_pesos_3d.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'Figura salva em: {out}')


# ---------------------------------------------------------------------------
# Derivação matemática (Questão 3)
# ---------------------------------------------------------------------------

def print_derivation():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Questão 3 – Derivação da regra "Norma Euclidiana" por minimização       ║
║              da função de erro quadrático                                ║
╚══════════════════════════════════════════════════════════════════════════╝

Dado um padrão de entrada x ∈ ℝⁿ e os pesos do neurônio vencedor w_j* ∈ ℝⁿ,
define-se a função de erro quadrático:

    E(w_j*) = ‖x − w_j*‖² = Σ_k (x_k − w_{j*,k})²

Objetivo: minimizar E em relação a w_{j*} via descida do gradiente.

Passo 1 – Gradiente de E em relação a cada componente w_{j*,k}:

    ∂E / ∂w_{j*,k}  =  ∂/∂w_{j*,k} [ (x_k − w_{j*,k})² ]
                     =  −2 (x_k − w_{j*,k})

Passo 2 – Regra de atualização por descida do gradiente:

    w_{j*,k}(t+1) = w_{j*,k}(t) − η · ∂E/∂w_{j*,k}
                  = w_{j*,k}(t) + 2η (x_k − w_{j*,k}(t))

Passo 3 – Absorvendo o fator 2 na taxa de aprendizado η̃ = 2η, obtemos
a forma vetorial:

    Δw_{j*} = η̃ (x − w_{j*})    ← regra de Kohonen

Para os neurônios vizinhos j ∈ N(j*) a mesma dedução se aplica com
a função de vizinhança h(j*, j):

    Δw_j = η̃ · h(j*, j) · (x − w_j)

onde h(j*, j) = 1 se dist(j*, j) ≤ raio, 0 caso contrário.

Conclusão: a regra de alteração de pesos pela Norma Euclidiana é
exatamente o passo de descida do gradiente da função E = ‖x − w_j*‖².
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Treinando rede de Kohonen...')
    print(f'  Grid: {GRID_ROWS}×{GRID_COLS} ({N1} neurônios)  |  '
          f'η={ETA}  |  raio={RADIUS}  |  épocas={EPOCHS}')
    print(f'  Amostras de treinamento: {len(TRAINING_DATA)}\n')

    weights = train(TRAINING_DATA)

    winners = [find_winner(x, weights) for x in TRAINING_DATA]
    class_a, class_b, class_c = build_class_sets(winners)

    print('=== Questão 1 – Neurônios por Classe ===')
    print(f'  Classe A (amostras  1– 20): neurônios {sorted(n+1 for n in class_a)}')
    print(f'  Classe B (amostras 21– 60): neurônios {sorted(n+1 for n in class_b)}')
    print(f'  Classe C (amostras 61–120): neurônios {sorted(n+1 for n in class_c)}')

    class_map = {}
    for n in class_a:
        class_map[n] = 'A'
    for n in class_b:
        class_map[n] = 'B'
    for n in class_c:
        class_map[n] = 'C'

    winner_counts = {}
    for w in winners:
        winner_counts[w] = winner_counts.get(w, 0) + 1

    plot_grid(class_map, winner_counts)
    plot_weights_3d(weights, class_map)

    print('\n=== Questão 2 – Classificação das Amostras de Teste ===')
    header = f'{"Amostra":>8}  {"x1":>7}  {"x2":>7}  {"x3":>7}  {"Neurônio":>9}  {"Classe":>7}'
    print(header)
    print('-' * 58)

    for i, x in enumerate(TEST_SAMPLES, start=1):
        w = find_winner(x, weights)
        cls = assign_class(w, class_a, class_b, class_c)
        print(f'{i:>8}  {x[0]:>7.4f}  {x[1]:>7.4f}  {x[2]:>7.4f}  {w+1:>9}  {cls:>7}')

    print_derivation()


if __name__ == '__main__':
    main()

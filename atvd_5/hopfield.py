# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import matplotlib.pyplot as plt

ROWS, COLS = 9, 5
N = ROWS * COLS  # 45 neurônios

# Padrões 9x5 = 45 bits com baixa correlacao cruzada entre si.
# Calculado: todos os pares tem |p_a . p_b| <= 5, proximo de ortogonal.
# Substitua pelos padroes reais do enunciado se necessario.
_r = np.arange(ROWS)[:, None]   # (9,1) - indices de linha
_c = np.arange(COLS)[None, :]   # (1,5) - indices de coluna
_ones = np.ones((ROWS, COLS))   # (9,5) - para forcar broadcast a 2D

_PATTERNS_BINARY = {
    # Metade superior solida (25+ / 20-)
    'P1': ((_r < 5) * _ones).astype(int),
    # Lado direito solido (cols 3-4 = +1, cols 0-2 = -1) (18+ / 27-)
    'P2': ((_c >= 3) * _ones).astype(int),
    # Tabuleiro xadrez: (linha+coluna) par = +1 (23+ / 22-)
    'P3': (((_r + _c) % 2 == 0) * 1).astype(int),
    # Moldura: pixels de borda = +1, interior = -1 (24+ / 21-)
    'P4': (((_r == 0) | (_r == ROWS-1) | (_c == 0) | (_c == COLS-1)) * 1).astype(int),
}

PATTERNS = {
    name: (2.0 * arr.astype(float) - 1.0).ravel()
    for name, arr in _PATTERNS_BINARY.items()
}


def build_weight_matrix():
    W = np.zeros((N, N))
    for p in PATTERNS.values():
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    return W / N


def add_noise(pattern, rate=0.2, rng=None):
    rng = rng or np.random.default_rng()
    noisy = pattern.copy()
    flip_idx = rng.choice(N, size=int(rate * N), replace=False)
    noisy[flip_idx] *= -1
    return noisy


def recover(W, state, beta=100, max_iter=100):
    s = state.copy()
    for _ in range(max_iter):
        net = W @ s
        s_new = np.where(np.tanh(beta * net) >= 0, 1.0, -1.0)
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


def hamming(a, b):
    return int(np.sum(a != b))


def plot_state(ax, state, title):
    img = (state.reshape(ROWS, COLS) + 1.0) / 2.0
    ax.imshow(img, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(title, fontsize=7)
    ax.axis('off')


def run_simulations():
    rng = np.random.default_rng(42)
    W = build_weight_matrix()

    fig, axes = plt.subplots(12, 3, figsize=(5, 32))
    col_labels = ['Imagem Transmitida\n(livre de ruído)',
                  'Imagem Distorcida\n(com ruído ~20%)',
                  'Imagem Limpa\n(recuperada)']
    for ax, lbl in zip(axes[0], col_labels):
        ax.set_title(lbl, fontsize=7, fontweight='bold')

    row = 0
    print(f'{"Pat":>4} {"Tentativa":>9} {"Hamming in":>11} {"Hamming out":>12} {"OK?":>5}')
    print('-' * 47)

    for name, pat in PATTERNS.items():
        for trial in range(1, 4):
            noisy = add_noise(pat, rate=0.2, rng=rng)
            rec = recover(W, noisy)
            h_in = hamming(noisy, pat)
            h_out = hamming(rec, pat)
            ok = h_out == 0
            print(f'{name:>4} {trial:>9}         {h_in:>5} px     {h_out:>5} px  {"OK" if ok else "ERRO"}')

            plot_state(axes[row, 0], pat, f'Original ({name})')
            plot_state(axes[row, 1], noisy, f'T{trial} – {h_in} px corrompidos')
            plot_state(axes[row, 2], rec, f'{"Correto" if ok else "ERRO"} (D={h_out})')
            row += 1

    plt.suptitle('Rede de Hopfield - 45 neuronios\n(4 padroes, ruido ~20%)',
                 y=1.002, fontsize=10)
    plt.tight_layout()
    out = 'atvd_5/hopfield_resultados.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'\nFigura salva em: {out}')


def analyze_noise_effect():
    W = build_weight_matrix()
    pat0 = list(PATTERNS.values())[0]
    name0 = list(PATTERNS.keys())[0]

    print(f'\n=== Efeito do Ruído Excessivo (padrão "{name0}") ===')
    print(f'{"Ruído":>7} {"Px corrompidos":>15} {"Hamming rec.":>13} {"Resultado":>12}')
    print('-' * 52)

    for rate in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        rng = np.random.default_rng(99)
        noisy = add_noise(pat0, rate=rate, rng=rng)
        rec = recover(W, noisy)
        h_in = hamming(noisy, pat0)
        h_out = hamming(rec, pat0)
        print(f'{int(rate*100):>6}%  {h_in:>10} px     {h_out:>6} px   '
              f'{"correto" if h_out == 0 else "INCORRETO"}')

    print(
        "\nExplicacao:\n"
        "  Com ruido baixo (~20%): o padrao distorcido permanece dentro da bacia de\n"
        "  atracao do padrao original; a rede converge para ele corretamente.\n\n"
        "  Com ruido crescente (>30-40%): o padrao distorcido se aproxima da borda\n"
        "  da bacia de atracao. A rede pode convergir para outro padrao armazenado\n"
        "  ou para um estado espurio (mistura de padroes, resultado do produto\n"
        "  externo de multiplos vetores).\n\n"
        "  Com ruido >50%: o padrao corrompido esta mais proximo de outro atrator\n"
        "  do que do original. A recuperacao falha sistematicamente.\n\n"
        "  Limite teorico de Hopfield: capacidade ~0.14*N ~6 padroes para N=45.\n"
        "  Mesmo dentro desse limite, ruido excessivo destroi a recuperacao porque\n"
        "  a bacia de atracao tem tamanho finito (~12-15% dos bits para 4 padroes).\n"
    )


if __name__ == '__main__':
    run_simulations()
    analyze_noise_effect()

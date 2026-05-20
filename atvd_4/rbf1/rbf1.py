"""
RBF – Classificação de Radiação Nuclear
Rede RBF (2 entradas, 2 neurônios na camada escondida, 1 saída).
Camada escondida: K-Means (apenas padrões com d=1).
Camada de saída: Regra Delta Generalizada (η=0.01, ε=1e-7).
"""

import numpy as np

# ─── Dados de Treinamento (Apêndice) ──────────────────────────────────────────
TRAIN = np.array([
    [0.2563, 0.9503, -1],
    [0.2405, 0.9018, -1],
    [0.1157, 0.3676,  1],
    [0.5147, 0.0167,  1],
    [0.4127, 0.3275,  1],
    [0.2809, 0.5830,  1],
    [0.8263, 0.9301, -1],
    [0.9359, 0.8724, -1],
    [0.1096, 0.9165, -1],
    [0.5158, 0.8545, -1],
    [0.1334, 0.1362,  1],
    [0.6371, 0.1439,  1],
    [0.7052, 0.6277, -1],
    [0.8703, 0.8666, -1],
    [0.2612, 0.6109,  1],
    [0.0244, 0.5279,  1],
    [0.9588, 0.3672, -1],
    [0.9332, 0.5499, -1],
    [0.9623, 0.2961, -1],
    [0.7297, 0.5776, -1],
    [0.4560, 0.1871,  1],
    [0.1715, 0.7713,  1],
    [0.5571, 0.5485, -1],
    [0.3344, 0.0259,  1],
    [0.4803, 0.7635, -1],
    [0.9721, 0.4850, -1],
    [0.8318, 0.7844, -1],
    [0.1373, 0.0292,  1],
    [0.3660, 0.8581, -1],
    [0.3626, 0.7302, -1],
    [0.6474, 0.3324,  1],
    [0.3461, 0.2398,  1],
    [0.1353, 0.8120,  1],
    [0.3463, 0.1017,  1],
    [0.9086, 0.1947, -1],
    [0.5227, 0.2321,  1],
    [0.5153, 0.2041,  1],
    [0.1832, 0.0661,  1],
    [0.5015, 0.9812, -1],
    [0.5024, 0.5274, -1],
])

# ─── Dados de Teste ────────────────────────────────────────────────────────────
TEST = np.array([
    [0.8705, 0.9329, -1],
    [0.0388, 0.2703,  1],
    [0.8236, 0.4458, -1],
    [0.7075, 0.1502,  1],
    [0.9587, 0.8663, -1],
    [0.6115, 0.9365, -1],
    [0.3534, 0.3646,  1],
    [0.3268, 0.2766,  1],
    [0.6129, 0.4518, -1],
    [0.9948, 0.4962, -1],
])


# ─── K-Means ───────────────────────────────────────────────────────────────────
def kmeans(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 1000) -> tuple:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=k, replace=False)
    centers = X[indices].astype(float).copy()
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else centers[j]
            for j in range(k)
        ])
        if np.allclose(centers, new_centers) and np.array_equal(labels, new_labels):
            break
        centers, labels = new_centers, new_labels

    return centers, labels


def compute_variances(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    k = len(centers)
    variances = np.zeros(k)
    for j in range(k):
        cluster = X[labels == j]
        if len(cluster) == 0:
            variances[j] = 1e-6
        else:
            v = np.mean(np.sum((cluster - centers[j]) ** 2, axis=1))
            variances[j] = v if v > 1e-10 else 1e-6
    return variances


# ─── Ativações RBF (Gaussiana) ─────────────────────────────────────────────────
def rbf_activations(X: np.ndarray, centers: np.ndarray, variances: np.ndarray) -> np.ndarray:
    phi = np.zeros((len(X), len(centers)))
    for j, (c, v) in enumerate(zip(centers, variances)):
        diff = X - c
        phi[:, j] = np.exp(-np.sum(diff ** 2, axis=1) / (2.0 * v))
    return phi


# ─── Treinamento da Camada de Saída (Regra Delta Generalizada) ─────────────────
def train_output_layer(
    phi: np.ndarray,
    d: np.ndarray,
    eta: float = 0.01,
    eps: float = 1e-7,
    max_epochs: int = 500_000,
    seed: int = 0,
) -> tuple:
    """
    Retorna (pesos, MSE_final, épocas).
    PHI inclui coluna de bias (coluna de 1s).
    """
    N = len(d)
    PHI = np.hstack([np.ones((N, 1)), phi])
    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.5, 0.5, PHI.shape[1])

    prev_mse = float("inf")
    final_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        sq_err = 0.0
        for i in range(N):
            y = w @ PHI[i]
            e = d[i] - y
            w += eta * e * PHI[i]
            sq_err += e ** 2
        mse = sq_err / N

        if abs(prev_mse - mse) < eps:
            final_epoch = epoch
            break
        prev_mse = mse
        final_epoch = epoch

    return w, mse, final_epoch


# ─── Pós-processamento (função sinal) ─────────────────────────────────────────
def sign_fn(y: float) -> int:
    return 1 if y >= 0 else -1


# ─── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    X_train = TRAIN[:, :2]
    d_train = TRAIN[:, 2]
    X_test = TEST[:, :2]
    d_test = TEST[:, 2]

    # Etapa 1 – K-Means apenas nos padrões com presença de radiação (d = 1)
    X_pos = X_train[d_train == 1]
    centers, labels = kmeans(X_pos, k=2, seed=0)
    variances = compute_variances(X_pos, centers, labels)

    print("=" * 60)
    print("ETAPA 1 – Camada Escondida: K-Means (d = 1)")
    print("=" * 60)
    print(f"\nAmostras com presença de radiação: {len(X_pos)}\n")
    print(f"{'Cluster':<10} {'Centro (x1, x2)':<32} {'Variância':>12}")
    print("-" * 56)
    for j in range(2):
        print(
            f"  {j+1:<8} ({centers[j, 0]:.6f}, {centers[j, 1]:.6f}){'':>4}"
            f" {variances[j]:>12.6f}"
        )

    # Etapa 2 – Treinamento da camada de saída
    phi_train = rbf_activations(X_train, centers, variances)
    w, final_mse, n_epochs = train_output_layer(
        phi_train, d_train, eta=0.01, eps=1e-7, seed=0
    )

    print("\n" + "=" * 60)
    print("ETAPA 2 – Camada de Saída: Regra Delta Generalizada")
    print(f"  η = 0.01  |  ε = 1×10⁻⁷")
    print("=" * 60)
    print(f"\nConvergência: época {n_epochs} | MSE = {final_mse:.4e}\n")
    print(f"{'Peso':<12} {'Valor':>14}")
    print("-" * 27)
    print(f"{'W21,0':<12} {w[0]:>14.6f}")
    print(f"{'W21,1':<12} {w[1]:>14.6f}")
    print(f"{'W21,2':<12} {w[2]:>14.6f}")

    # Etapa 3 – Validação no conjunto de teste
    phi_test = rbf_activations(X_test, centers, variances)
    PHI_test = np.hstack([np.ones((len(X_test), 1)), phi_test])

    print("\n" + "=" * 60)
    print("ETAPA 3 – Validação no Conjunto de Teste")
    print("=" * 60)
    print(
        f"\n{'#':<4} {'x1':>8} {'x2':>8} {'d':>5} {'y':>12} {'ypós':>6} {'✓':>4}"
    )
    print("-" * 52)

    correct = 0
    results = []
    for i in range(len(X_test)):
        y_real = w @ PHI_test[i]
        y_pos = sign_fn(y_real)
        ok = y_pos == int(d_test[i])
        correct += ok
        results.append((X_test[i, 0], X_test[i, 1], int(d_test[i]), y_real, y_pos, ok))
        print(
            f"{i+1:<4} {X_test[i,0]:>8.4f} {X_test[i,1]:>8.4f}"
            f" {int(d_test[i]):>5} {y_real:>12.6f} {y_pos:>6} {'OK' if ok else 'ERRO':>4}"
        )

    accuracy = correct / len(d_test) * 100
    print(f"\nTaxa de Acerto: {accuracy:.1f}%  ({correct}/{len(d_test)} corretos)")

    # Exporta resultados para uso no README
    return centers, variances, w, results, accuracy


if __name__ == "__main__":
    main()

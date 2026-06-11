"""
LVQ-1 (Learning Vector Quantization 1)
========================================
Disciplina: Lab. Inteligência Artificial
Professor:  Lázaro Eduardo da Silva
Atividade:  Classificação de perfis de potência elétrica (LVQ1.docx)

Dados de treinamento: 16 amostras com 6 atributos (potência às 7h, 8h, 9h, 10h, 11h, 12h)
                      agrupadas em 4 classes (perfis de demanda).
Taxa de aprendizagem: alpha = 0.05
"""

import numpy as np
import random


# ---------------------------------------------------------------------------
# Dados de Treinamento
# ---------------------------------------------------------------------------

training_data = np.array([
    # amostra, [7h, 8h, 9h, 10h, 11h, 12h], classe
    [2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649],  # classe 1
    [2.3936, 1.4804, 1.9907, 1.2732, 2.2719, 1.8110],  # classe 1
    [2.2880, 1.4585, 1.9867, 1.2451, 2.3389, 1.8099],  # classe 1
    [2.2904, 1.4766, 1.8876, 1.2706, 2.2966, 1.7744],  # classe 1

    [1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276],  # classe 2
    [0.9913, 0.1524, 1.2700, 5.3808, 3.0714, 2.3331],  # classe 2
    [1.0915, 0.1881, 1.1387, 5.3701, 3.2561, 2.3383],  # classe 2
    [1.0535, 0.1229, 1.2743, 5.3226, 3.0950, 2.3193],  # classe 2

    [1.4871, 2.3448, 0.9918, 2.3160, 1.6783, 5.0850],  # classe 3
    [1.3312, 2.2553, 0.9618, 2.4702, 1.7272, 5.0645],  # classe 3
    [1.3646, 2.2945, 1.0562, 2.4763, 1.8051, 5.1470],  # classe 3
    [1.4392, 2.2296, 1.1278, 2.4230, 1.7259, 5.0876],  # classe 3

    [2.9364, 1.5233, 4.6109, 1.3160, 4.2700, 6.8749],  # classe 4
    [2.9034, 1.4640, 4.6061, 1.4598, 4.2912, 6.9142],  # classe 4
    [3.0181, 1.4918, 4.7051, 1.3521, 4.2623, 6.7966],  # classe 4
    [2.9374, 1.4896, 4.7219, 1.3977, 4.1863, 6.8336],  # classe 4
], dtype=float)

training_labels = np.array([1, 1, 1, 1,
                             2, 2, 2, 2,
                             3, 3, 3, 3,
                             4, 4, 4, 4])

# ---------------------------------------------------------------------------
# Dados de Teste (dias a classificar)
# ---------------------------------------------------------------------------

test_data = np.array([
    [2.9817, 1.5656, 4.8391, 1.4311, 4.1916, 6.9718],  # dia 1
    [1.5537, 2.2615, 1.3169, 2.5873, 1.7570, 5.0958],  # dia 2
    [1.2240, 0.2445, 1.3595, 5.4192, 3.2027, 2.5675],  # dia 3
    [2.5828, 1.5146, 2.1119, 1.2859, 2.3414, 1.8695],  # dia 4
    [2.4168, 1.4857, 1.8959, 1.3013, 2.4500, 1.7868],  # dia 5
    [1.0604, 0.2276, 1.2806, 5.4732, 3.2133, 2.4839],  # dia 6
    [1.5246, 2.4254, 1.1353, 2.5325, 1.7569, 5.2640],  # dia 7
    [3.0565, 1.6259, 4.7743, 1.3654, 4.2904, 6.9808],  # dia 8
], dtype=float)


# ---------------------------------------------------------------------------
# Classe LVQ1
# ---------------------------------------------------------------------------

class LVQ1:
    """
    Implementação da rede LVQ-1 (Learning Vector Quantization 1).

    Parâmetros
    ----------
    n_prototypes_per_class : int
        Número de protótipos (neurônios vencedores) por classe.
    alpha : float
        Taxa de aprendizagem.
    max_epochs : int
        Número máximo de épocas de treinamento.
    random_state : int | None
        Semente para reproducibilidade.
    """

    def __init__(self,
                 n_prototypes_per_class: int = 1,
                 alpha: float = 0.05,
                 max_epochs: int = 100,
                 random_state: int | None = 42):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.alpha = alpha
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.prototypes_: np.ndarray | None = None
        self.prototype_labels_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distância Euclidiana entre dois vetores."""
        return float(np.sqrt(np.sum((a - b) ** 2)))

    def _find_winner(self, x: np.ndarray) -> int:
        """Retorna o índice do protótipo mais próximo de x."""
        distances = [self._euclidean_distance(x, w) for w in self.prototypes_]
        return int(np.argmin(distances))

    def _initialize_prototypes(self,
                               X: np.ndarray,
                               y: np.ndarray) -> None:
        """
        Inicializa os protótipos como a média de cada classe
        (estratégia mais estável que inicialização aleatória pura).
        """
        classes = np.unique(y)
        prototypes = []
        proto_labels = []

        for cls in classes:
            class_samples = X[y == cls]
            for _ in range(self.n_prototypes_per_class):
                # Média da classe como protótipo inicial
                proto = np.mean(class_samples, axis=0).copy()
                # Adiciona pequena perturbação para diferenciar protótipos
                # de mesma classe (só relevante quando n_prototypes_per_class > 1)
                if self.random_state is not None:
                    rng = np.random.default_rng(self.random_state + int(cls))
                    proto += rng.normal(0, 0.01, size=proto.shape)
                prototypes.append(proto)
                proto_labels.append(cls)

        self.prototypes_ = np.array(prototypes, dtype=float)
        self.prototype_labels_ = np.array(proto_labels)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LVQ1":
        """
        Treina a rede LVQ-1.

        Parâmetros
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        self._initialize_prototypes(X, y)

        indices = list(range(len(X)))

        print("=" * 60)
        print("        TREINAMENTO LVQ-1")
        print("=" * 60)
        print(f"  Amostras de treino : {len(X)}")
        print(f"  Classes            : {np.unique(y).tolist()}")
        print(f"  Protótipos/classe  : {self.n_prototypes_per_class}")
        print(f"  Taxa aprendizagem  : {self.alpha}")
        print(f"  Épocas máximas     : {self.max_epochs}")
        print("=" * 60)

        for epoch in range(1, self.max_epochs + 1):
            if self.random_state is not None:
                random.seed(self.random_state + epoch)
            random.shuffle(indices)

            for idx in indices:
                x = X[idx]
                label = y[idx]

                winner_idx = self._find_winner(x)
                winner_label = self.prototype_labels_[winner_idx]
                w = self.prototypes_[winner_idx]

                if winner_label == label:
                    # Classe correta → aproxima protótipo da amostra
                    self.prototypes_[winner_idx] += self.alpha * (x - w)
                else:
                    # Classe errada → afasta protótipo da amostra
                    self.prototypes_[winner_idx] -= self.alpha * (x - w)

        print(f"\n  Protótipos finais após {self.max_epochs} épocas:\n")
        for i, (proto, lbl) in enumerate(
                zip(self.prototypes_, self.prototype_labels_)):
            formatted = ", ".join(f"{v:.4f}" for v in proto)
            print(f"    W{i+1} (classe {lbl}): [{formatted}]")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classifica amostras usando os protótipos treinados.

        Parâmetros
        ----------
        X : array-like, shape (n_samples, n_features)

        Retorna
        -------
        np.ndarray com a classe prevista para cada amostra.
        """
        X = np.array(X, dtype=float)
        predictions = []
        for x in X:
            winner_idx = self._find_winner(x)
            predictions.append(self.prototype_labels_[winner_idx])
        return np.array(predictions)


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def avaliar_treinamento(model: LVQ1,
                        X: np.ndarray,
                        y: np.ndarray) -> None:
    """Avalia o modelo sobre os dados de treinamento e exibe a acurácia."""
    preds = model.predict(X)
    acertos = np.sum(preds == y)
    total = len(y)
    print("\n" + "=" * 60)
    print("     AVALIAÇÃO NOS DADOS DE TREINAMENTO")
    print("=" * 60)
    print(f"  {'Amostra':<10} {'Classe Real':<15} {'Classe Prevista':<15} {'Resultado'}")
    print("  " + "-" * 52)
    for i, (real, prev) in enumerate(zip(y, preds)):
        resultado = "✓ Correto" if real == prev else "✗ Errado"
        print(f"  {i+1:<10} {real:<15} {prev:<15} {resultado}")
    print("  " + "-" * 52)
    print(f"  Acurácia: {acertos}/{total} = {acertos/total*100:.1f}%")


def classificar_dias_teste(model: LVQ1,
                           X_test: np.ndarray) -> None:
    """Classifica os dias de teste e exibe os resultados."""
    preds = model.predict(X_test)
    print("\n" + "=" * 60)
    print("       CLASSIFICAÇÃO DOS DIAS DE TESTE")
    print("=" * 60)
    print(f"  {'Dia':<8} {'Classe Prevista'}")
    print("  " + "-" * 25)
    for dia, cls in enumerate(preds, start=1):
        print(f"  {dia:<8} {cls}")
    print("=" * 60)


if __name__ == "__main__":
    # Treinamento
    modelo = LVQ1(n_prototypes_per_class=1,
                  alpha=0.05,
                  max_epochs=100,
                  random_state=42)
    modelo.fit(training_data, training_labels)

    # Avaliação no conjunto de treinamento
    avaliar_treinamento(modelo, training_data, training_labels)

    # Classificação dos dias de teste
    classificar_dias_teste(modelo, test_data)

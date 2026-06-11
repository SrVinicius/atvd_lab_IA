"""
ART-1 (Adaptive Resonance Theory 1)
======================================
Disciplina: Lab. Inteligência Artificial
Professor:  Lázaro Eduardo da Silva
Atividade:  Diagnóstico de situações de processo industrial (ART1.docx)

Dados de entrada: 10 situações com 16 variáveis binárias de status (0 ou 1).
Graus de vigilância a testar: rho = 0.5, 0.8, 0.9, 0.99
"""

import numpy as np


# ---------------------------------------------------------------------------
# Dados de Entrada (10 situações × 16 variáveis binárias)
# ---------------------------------------------------------------------------

situations = np.array([
    # x1  x2  x3  x4  x5  x6  x7  x8  x9 x10 x11 x12 x13 x14 x15 x16
    [  0,  1,  0,  1,  1,  0,  1,  0,  1,  1,  0,  1,  1,  1,  1,  1],  # Situação 1
    [  1,  0,  1,  0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  0,  0,  0],  # Situação 2
    [  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],  # Situação 3
    [  1,  1,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  1,  0,  0],  # Situação 4
    [  0,  0,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1],  # Situação 5
    [  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,  0,  1,  1,  1,  1,  1],  # Situação 6
    [  1,  0,  1,  0,  1,  1,  0,  1,  1,  1,  1,  0,  1,  1,  1,  0],  # Situação 7
    [  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  0,  1,  1,  0,  1,  1],  # Situação 8
    [  0,  1,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1],  # Situação 9
    [  0,  0,  1,  1,  1,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1],  # Situação 10
], dtype=float)

situation_names = [f"Situação {i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Classe ART1
# ---------------------------------------------------------------------------

class ART1:
    """
    Implementação da rede ART-1 (Adaptive Resonance Theory 1).

    A rede ART-1 trabalha exclusivamente com entradas binárias (0 ou 1).

    Parâmetros
    ----------
    n_features : int
        Número de atributos (dimensão do vetor de entrada).
    rho : float
        Grau de vigilância (entre 0 e 1). Controla a granularidade
        dos agrupamentos: valores maiores → classes mais específicas.
    beta : float
        Parâmetro de aprendizagem (0 < beta <= 1).
        Recomendado: beta = 1 (aprendizagem rápida / "fast learning").
    """

    def __init__(self,
                 n_features: int,
                 rho: float = 0.5,
                 beta: float = 1.0):
        self.n_features = n_features
        self.rho = rho
        self.beta = beta

        # Pesos bottom-up (F1 → F2): inicializados com valor
        # que garante que todas as entradas são igualmente candidatas
        # no início.  Valor padrão: 1 / (1 + n_features)
        self._b_init = 1.0 / (1.0 + n_features)

        # Lista de protótipos top-down (F2 → F1): um vetor por classe.
        # Cada protótipo começa como vetor unitário (todos 1s).
        self.top_down_weights_: list[np.ndarray] = []

        # Lista de pesos bottom-up: um vetor por classe.
        self.bottom_up_weights_: list[np.ndarray] = []

        # Mapeamento: índice de classe → lista de índices de amostras.
        self.clusters_: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Operações internas
    # ------------------------------------------------------------------

    @staticmethod
    def _fuzzy_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """AND binário (mínimo elemento a elemento)."""
        return np.minimum(a, b)

    @staticmethod
    def _norm(v: np.ndarray) -> float:
        """Norma L1 (soma dos elementos para vetores binários)."""
        return float(np.sum(v))

    def _bottom_up_activation(self, x: np.ndarray) -> list[float]:
        """Calcula a ativação bottom-up para cada neurônio de F2."""
        activations = []
        for bw in self.bottom_up_weights_:
            # T_j = ||x AND w_j|| / (beta + ||w_j||)
            and_result = self._fuzzy_and(x, bw)
            activation = self._norm(and_result) / (self.beta + self._norm(bw))
            activations.append(activation)
        return activations

    def _vigilance_test(self,
                        x: np.ndarray,
                        class_idx: int) -> bool:
        """
        Testa se a entrada x passa no critério de vigilância
        em relação ao protótipo da classe class_idx.

        Critério: ||x AND t_j|| / ||x|| >= rho
        """
        td = self.top_down_weights_[class_idx]
        and_result = self._fuzzy_and(x, td)
        score = self._norm(and_result) / self._norm(x)
        return score >= self.rho

    def _create_new_class(self, x: np.ndarray, sample_idx: int) -> int:
        """Cria uma nova classe e inicializa seus pesos."""
        class_idx = len(self.top_down_weights_)

        # Protótipo top-down: cópia da entrada atual
        self.top_down_weights_.append(x.copy())

        # Peso bottom-up: inicializado conforme ativação
        bw = x.copy() / (self.beta + self._norm(x))
        self.bottom_up_weights_.append(bw)

        self.clusters_[class_idx] = [sample_idx]
        return class_idx

    def _update_weights(self,
                        x: np.ndarray,
                        class_idx: int) -> None:
        """Atualiza os pesos top-down e bottom-up da classe vencedora."""
        td = self.top_down_weights_[class_idx]
        and_result = self._fuzzy_and(x, td)

        # Novo top-down: x AND t_j (aprendizagem rápida: beta = 1)
        self.top_down_weights_[class_idx] = and_result.copy()

        # Novo bottom-up: and_result / (beta + ||and_result||)
        self.bottom_up_weights_[class_idx] = (
            and_result / (self.beta + self._norm(and_result))
        )

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray,
            names: list[str] | None = None) -> "ART1":
        """
        Processa as amostras e forma os agrupamentos.

        Parâmetros
        ----------
        X     : array-like, shape (n_samples, n_features) – entradas binárias
        names : lista com os nomes das amostras (para relatório)
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]
        if names is None:
            names = [f"Amostra {i+1}" for i in range(n_samples)]

        # Reinicia os pesos e clusters
        self.top_down_weights_ = []
        self.bottom_up_weights_ = []
        self.clusters_ = {}

        for idx, x in enumerate(X):
            x_norm = self._norm(x)

            if x_norm == 0:
                # Vetor nulo: não pode ser classificado
                print(f"  AVISO: {names[idx]} é vetor nulo, ignorado.")
                continue

            assigned = False

            if len(self.top_down_weights_) == 0:
                # Primeira amostra → cria a primeira classe
                self._create_new_class(x, idx)
                assigned = True
            else:
                # Ordena neurônios por ativação (maior primeiro)
                activations = self._bottom_up_activation(x)
                ordered_classes = np.argsort(activations)[::-1]

                inhibited = set()
                for class_idx in ordered_classes:
                    if class_idx in inhibited:
                        continue

                    if self._vigilance_test(x, class_idx):
                        # Passou no teste de vigilância → atribui à classe
                        self._update_weights(x, class_idx)
                        self.clusters_[class_idx].append(idx)
                        assigned = True
                        break
                    else:
                        # Não passou → inibe este neurônio e tenta o próximo
                        inhibited.add(class_idx)

            if not assigned:
                # Nenhuma classe existente aceita → cria nova classe
                self._create_new_class(x, idx)

        return self

    def get_cluster_report(self,
                           names: list[str] | None = None,
                           n_samples: int | None = None) -> str:
        """
        Retorna um relatório textual dos agrupamentos formados.
        """
        if n_samples is None:
            n_samples = sum(len(v) for v in self.clusters_.values())
        if names is None:
            names = [f"Amostra {i+1}" for i in range(n_samples)]

        lines = []
        lines.append(f"  Classes ativas: {len(self.clusters_)}")
        lines.append("")
        for cls_idx, sample_indices in self.clusters_.items():
            member_names = [names[i] for i in sample_indices]
            lines.append(f"  Classe {cls_idx + 1}: {', '.join(member_names)}")
            lines.append(f"    Protótipo top-down: {self.top_down_weights_[cls_idx].astype(int).tolist()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execução principal
# ---------------------------------------------------------------------------

def run_simulation(X: np.ndarray,
                   names: list[str],
                   rho: float) -> None:
    """
    Executa uma simulação ART-1 completa para um dado grau de vigilância.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ART-1  |  Grau de vigilância (ρ) = {rho}")
    print(sep)

    model = ART1(n_features=X.shape[1], rho=rho, beta=1.0)
    model.fit(X, names)

    print(model.get_cluster_report(names=names, n_samples=len(names)))
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("         REDE ART-1 – DIAGNÓSTICO DE PROCESSO INDUSTRIAL")
    print("=" * 60)
    print(f"  Amostras : {len(situations)}")
    print(f"  Atributos: {situations.shape[1]}")
    print(f"  Graus de vigilância a testar: 0.5 | 0.8 | 0.9 | 0.99")

    vigilance_levels = [0.5, 0.8, 0.9, 0.99]

    for rho in vigilance_levels:
        run_simulation(situations, situation_names, rho)
